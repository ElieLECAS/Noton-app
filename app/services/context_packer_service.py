"""CAG post-retriever — packe des DOCUMENTS entiers dans le contexte de génération.

Au lieu d'injecter ~20 passages/pages tronqués, on exploite la fenêtre 256k de Mistral
Large : les passages retrouvés servent à SÉLECTIONNER des documents ; on charge ensuite
chaque document (entier s'il tient, sinon fenêtré autour des pages matchées) sous un budget
de tokens, avec un en-tête explicite (source, gamme, matériau) pour éviter la confusion de
gammes. Le rappel passe du niveau passage (fragile) au niveau document (stable).

Fonctions principales :
  * ``build_cag_context`` — remplace ``build_space_context_from_passages`` côté chat quand
    ``CAG_ENABLED`` est actif. Budget/max_documents adaptatifs par intent
    (``CAG_BUDGET_BY_INTENT``). Signature de sortie compatible (dict system).
  * ``select_cag_images`` — PNG UNIQUEMENT pour des pages réellement packées dans le
    contexte (alignement texte/visuel), avec légendes pour relier image ↔ document.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

# Plafond imposé par l'API Mistral (erreur 400, code 3051). Ce n'est PAS un réglage :
# dépasser ce nombre fait rejeter la requête entière, et le repli de secours répond
# alors sans document — donc de mémoire, avec assurance. Constat du 2026-08-26.
MISTRAL_MAX_IMAGES_PER_REQUEST = 8

# Enregistrement d'un chunk feuille aplati : (page, chunk_index, texte).
LeafRecord = Tuple[int, int, str]

# Cache TTL du texte des feuilles par document (évite ~60 lignes SQL + concat par requête).
# Valeurs PLATES (pas d'objets ORM : ils seraient détachés de leur session d'origine).
_leaf_cache: Dict[int, Tuple[float, List[LeafRecord]]] = {}


def _resolve_chunk_page(chunk: DocumentChunk) -> int:
    """Page d'un chunk depuis metadata_json/metadata_ (0 si inconnue)."""
    for meta in (getattr(chunk, "metadata_json", None), getattr(chunk, "metadata_", None)):
        if isinstance(meta, dict):
            for key in ("page_no", "page_start", "page_label", "page_idx"):
                try:
                    val = int(meta.get(key))
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    return val
    return 0


def _chunk_text(chunk: DocumentChunk) -> str:
    return (chunk.text or chunk.content or "").strip()


def estimate_tokens(text: str) -> int:
    """Estimation grossière FR (chars / CAG_CHARS_PER_TOKEN)."""
    return int(len(text) / max(0.5, settings.CAG_CHARS_PER_TOKEN))


def budget_for_intent(intent: Optional[str]) -> Tuple[int, int]:
    """(token_budget, max_documents) selon l'intent de la requête.

    Une question de spécification ponctuelle ne paie pas le prefill (coût + latence
    1er token) d'un diagnostic SAV. Table ``CAG_BUDGET_BY_INTENT`` ; la clé "default"
    couvre les intents absents/inconnus.

    CAG_TOKEN_BUDGET et CAG_MAX_DOCUMENTS sont des PLAFONDS DURS : la table par intent
    module en dessous, jamais au-dessus. Sans ce clamp, la table par défaut du code
    (product_selection=100000/8) rendait les variables d'env inopérantes — une prod
    configurée à 50000/3 packait quand même 100000/8 (constaté logs prod 2026-07-22).
    """
    table = settings.cag_budget_by_intent or {}
    key = (intent or "").strip().lower()
    entry = table.get(key) or table.get("default") or {}
    token_budget = int(entry.get("budget") or settings.CAG_TOKEN_BUDGET)
    max_documents = int(entry.get("max_documents") or settings.CAG_MAX_DOCUMENTS)
    token_budget = min(token_budget, int(settings.CAG_TOKEN_BUDGET))
    max_documents = min(max_documents, int(settings.CAG_MAX_DOCUMENTS))
    return max(1000, token_budget), max(1, max_documents)


def invalidate_document_fulltext_cache(document_id: Optional[int] = None) -> None:
    """Invalide le cache des feuilles (un document, ou tout si None) — à appeler après réindexation."""
    if document_id is None:
        _leaf_cache.clear()
    else:
        _leaf_cache.pop(int(document_id), None)


def _load_leaf_records(session: Session, document_id: int) -> List[LeafRecord]:
    """Chunks feuilles (L1) d'un document, aplatis en (page, index, texte), ordre de lecture.

    Mise en cache TTL (``CAG_FULLTEXT_CACHE_TTL``, 0 = off) : un réindex peut mettre
    jusqu'à TTL secondes à se refléter ici — acceptable, et ``invalidate_document_fulltext_cache``
    permet l'invalidation immédiate côté indexation.
    """
    ttl = settings.CAG_FULLTEXT_CACHE_TTL
    now = time.monotonic()
    if ttl > 0:
        cached = _leaf_cache.get(document_id)
        if cached and cached[0] > now:
            return cached[1]

    rows = list(
        session.exec(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.is_leaf == True,  # noqa: E712
            )
        ).all()
    )
    rows.sort(key=lambda c: (_resolve_chunk_page(c), c.chunk_index or 0, c.id or 0))

    records: List[LeafRecord] = []
    for chunk in rows:
        chunk_text = _chunk_text(chunk)
        if not chunk_text:
            continue
        # Les synthèses L2 sont du texte GÉNÉRÉ par un LLM. Sans marqueur, elles étaient
        # présentées au modèle de génération mélangées au texte source, indiscernables —
        # il pouvait donc citer une reformulation comme s'il s'agissait du document.
        meta = getattr(chunk, "metadata_json", None) or getattr(chunk, "metadata_", None) or {}
        if isinstance(meta, dict) and meta.get("content_type") == "contextual_enrichment":
            chunk_text = f"[synthèse générée par l'IA — non verbatim]\n{chunk_text}"
        records.append((_resolve_chunk_page(chunk), chunk.chunk_index or 0, chunk_text))
    if ttl > 0:
        _leaf_cache[document_id] = (now + ttl, records)
    return records


# Familles de canaux de retrieval : texte (BM25) et visuel (ColPali). Le bonus de
# familles récompense un document confirmé par les DEUX modalités, sans jamais renverser
# un meilleur passage net (cf. _election_score).
#
# La famille « graphe » (KAG) a été retirée le 2026-07-28 avec le canal, et la voie dense
# texte (pgvector) le 2026-08-25 : elle lisait la MÊME évidence que BM25 (le texte de la
# page) et votait deux fois au RRF sans rien ajouter. Le bonus de familles est borné à
# +0,10 (2 familles), ce qui laisse le plus de poids au meilleur passage — l'intention
# même du mode best_passage.
_CHANNEL_FAMILIES = {
    "bm25": "texte",
    "colpali": "visuel",
}

# Les bornes d'un passage (page_start/page_end) viennent de l'expansion de voisinage, pas
# d'un match propre : elles restent des pages matchées, mais légèrement en retrait pour
# qu'une page ANCRE gagne toujours la course aux seeds à score égal.
_SPAN_PAGE_WEIGHT = 0.95


def _passage_families(passage: Dict[str, Any]) -> Set[str]:
    """Familles de canaux ayant retrouvé ce passage (vide si l'info n'est pas portée)."""
    return {
        _CHANNEL_FAMILIES[src]
        for src in (passage.get("retrieval_sources") or [])
        if src in _CHANNEL_FAMILIES
    }


def _election_score(entry: Dict[str, Any]) -> float:
    """Score d'élection d'un document : dominé par sa MEILLEURE page.

    ``score_max × (1 + bonus_familles + bonus_pages)``, bonus multiplicatifs et BORNÉS :
    ils départagent deux documents dont les meilleures pages sont proches, sans jamais
    renverser un meilleur passage net. L'ancienne formule additive
    ``score_max + 0.2·(somme des autres pages)`` faisait l'inverse : un catalogue plaçant
    six pages moyennes battait arithmétiquement la notice qui contenait LA bonne page.
    """
    base = float(entry.get("score_max") or 0.0)
    if base <= 0:
        # Scores nuls/négatifs : le bonus multiplicatif n'a pas de sens (il aggraverait un
        # score négatif). On laisse le score brut départager.
        return base
    n_families = max(1, len(entry.get("families") or ()))
    n_pages = max(1, len(entry.get("matched_pages") or ()))
    bonus = settings.CAG_ELECTION_FAMILY_BONUS * (n_families - 1) + (
        settings.CAG_ELECTION_PAGE_BONUS
        * min(n_pages - 1, max(0, settings.CAG_ELECTION_PAGE_CAP))
    )
    return base * (1.0 + bonus)


def _legacy_election_score(entry: Dict[str, Any]) -> float:
    """Ancienne formule (volume) — conservée derrière ``CAG_ELECTION_MODE=legacy``."""
    return float(entry["score_max"]) + 0.2 * (
        float(entry["score_sum"]) - float(entry["score_max"])
    )


def _new_document_entry(title: Optional[str] = None) -> Dict[str, Any]:
    """Accumulateur par document (``matched_pages`` = page → meilleur score de passage)."""
    return {
        "score_sum": 0.0,
        "score_max": 0.0,
        "matched_pages": {},
        "families": set(),
        "title": title,
        "election_score": 0.0,
    }


def aggregate_documents(
    passages: List[Dict[str, Any]], *, max_documents: int
) -> List[Tuple[int, Dict[str, Any]]]:
    """Regroupe les passages par document et élit les meilleurs.

    Le document est élu par sa MEILLEURE page (cf. ``_election_score``), pas par le volume
    de pages moyennes qu'il place dans le top-K. ``matched_pages`` conserve le score de
    chaque page (page → meilleur score) : cette hiérarchie sert ensuite à choisir les seeds
    et à remplir la fenêtre par valeur au lieu d'un rayon aveugle.

    Retourne [(document_id, {score_max, score_sum, matched_pages, families, title,
    election_score}), …] trié décroissant.
    """
    agg: Dict[int, Dict[str, Any]] = {}
    for p in passages:
        did = p.get("document_id")
        if did is None:
            continue
        did = int(did)
        entry = agg.setdefault(did, _new_document_entry(p.get("document_title")))
        score = float(p.get("score") or 0.0)
        entry["score_sum"] += score
        entry["score_max"] = max(entry["score_max"], score)
        entry["families"].update(_passage_families(p))

        # Page ANCRE (celle que le retriever a réellement matchée) au score plein ; les
        # bornes du span au poids réduit.
        anchor_page = None
        for key in ("page_no", "page_start", "page_end"):
            val = p.get(key)
            if isinstance(val, int) and val > 0:
                anchor_page = val
                break
        for key in ("page_no", "page_start", "page_end"):
            val = p.get(key)
            if not isinstance(val, int) or val <= 0:
                continue
            page_score = score if val == anchor_page else score * _SPAN_PAGE_WEIGHT
            if page_score > entry["matched_pages"].get(val, 0.0):
                entry["matched_pages"][val] = page_score

    legacy = (settings.CAG_ELECTION_MODE or "").strip().lower() == "legacy"
    score_fn = _legacy_election_score if legacy else _election_score
    for entry in agg.values():
        entry["election_score"] = score_fn(entry)

    ranked = sorted(agg.items(), key=lambda kv: kv[1]["election_score"], reverse=True)
    return ranked[:max_documents]


def _apply_document_election(
    ranked_docs: List[Tuple[int, Dict[str, Any]]],
    elected_document_ids: Optional[List[int]],
    pinned_pages: Optional[Dict[int, List[int]]],
    *,
    max_documents: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Impose un classement de documents décidé EN AMONT du packer.

    Les documents élus passent EN TÊTE, dans l'ordre fourni : leur classement prime sur
    le score d'élection interne du packer. Les pages épinglées sont injectées dans
    ``matched_pages`` avec une valeur supérieure au meilleur score existant : elles
    gagnent la course aux seeds de la fenêtre gloutonne et ne peuvent pas être rognées
    tant que le document a du budget.

    NOTE : le juge de suffisance, qui alimentait ces deux entrées, a été supprimé
    (verdict inexploitable pour 2,7 s par requête). Le mécanisme est conservé tel quel
    car l'élection ColPali doit le reprendre — sans producteur, il est inerte.

    Fonction pure (aucun accès DB) — testable unitairement."""
    if elected_document_ids:
        by_id = {did: meta for did, meta in ranked_docs}
        head: List[Tuple[int, Dict[str, Any]]] = []
        for did in dict.fromkeys(int(d) for d in elected_document_ids):
            head.append((did, by_id.pop(did, _new_document_entry())))
        others = [(did, meta) for did, meta in ranked_docs if did in by_id]
        ranked_docs = head + others[: max(0, max_documents - len(head))]

    pinned = {
        int(did): [int(p) for p in (pages or []) if int(p) > 0]
        for did, pages in (pinned_pages or {}).items()
    }
    if pinned:
        for did, meta in ranked_docs:
            pages = pinned.get(int(did))
            if not pages:
                continue
            matched: Dict[int, float] = meta.setdefault("matched_pages", {})
            top = max(matched.values(), default=0.0)
            pin_value = (top if top > 0 else 1.0) * 1.05
            for page in pages:
                matched[page] = max(float(matched.get(page, 0.0)), pin_value)
    return ranked_docs


def _seed_pages(matched_pages: Dict[int, float], max_seeds: int) -> List[Tuple[int, float]]:
    """Pages matchées promues en SEEDS, les mieux scorées d'abord.

    Plafonner le nombre de seeds évite la « fenêtre pieuvre » : un document matché sur dix
    pages diluerait sinon son budget en dix fenêtres, au lieu de traiter à fond les
    meilleures. À score égal, la page la plus petite gagne (ordre déterministe).
    """
    if not matched_pages:
        return []
    ordered = sorted(matched_pages.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    return ordered[: max(1, max_seeds)]


def _budget_shares(n_docs: int) -> List[float]:
    """Parts du budget par rang d'élection, renormalisées sur les documents réellement élus.

    Retourne [] si le partage est désactivé (le premier document est alors servi jusqu'au
    budget global, comportement historique). Avec des parts 0,5/0,3/0,2 et deux documents
    élus, on obtient 62,5 %/37,5 % : le budget reste intégralement distribué.
    """
    if n_docs <= 0:
        return []
    declared = settings.cag_doc_budget_shares
    if not declared:
        return []
    shares = list(declared[:n_docs])
    if len(shares) < n_docs:
        # Plus de documents que de parts déclarées : les rangs suivants héritent de la dernière.
        shares.extend([shares[-1]] * (n_docs - len(shares)))
    total = sum(shares)
    if total <= 0:
        return []
    return [s / total for s in shares]


def _select_records_by_value(
    records: List[LeafRecord],
    matched_pages: Dict[int, float],
    budget: int,
    *,
    radius: int,
    max_seeds: int,
    decay: float,
) -> Tuple[List[LeafRecord], List[int]]:
    """Remplit le budget d'un document par VALEUR de page décroissante.

    Les seeds (pages réellement matchées) passent en premier : la page qui a gagné le vote
    du retriever ne peut pas être rognée tant que le document a du budget. Ses voisines
    héritent d'une valeur décroissante (``decay^distance``), si bien que sous budget serré
    ce sont les pages les plus FAIBLES qui sautent — et non les plus éloignées
    géographiquement, critère aveugle à la pertinence du rognage historique.

    Retourne (records retenus en ordre de lecture, pages seeds effectivement retenues).
    """
    by_page: Dict[int, List[LeafRecord]] = {}
    for record in records:
        by_page.setdefault(record[0], []).append(record)
    if not by_page:
        return [], []

    values: Dict[int, float] = {}

    # 1. TOUTE page matchée porte sa propre valeur. Une page que le retriever a trouvée ne
    #    doit jamais céder la place à la simple voisine d'une autre page : plafonner les
    #    seeds ferait perdre une bonne page isolée dans un gros document (cas réel : p.89
    #    d'un catalogue, trouvée par 3 canaux et reclassée 0,901, écartée parce que les
    #    3 meilleures pages matchées étaient groupées 40 pages plus tôt).
    for page, score in matched_pages.items():
        if page in by_page and page > 0:
            values[page] = max(values.get(page, 0.0), float(score))

    # 2. Le HALO, lui, reste plafonné aux meilleures seeds : c'est lui qui multiplie les
    #    pages, et c'est donc lui — pas l'évidence — qu'il faut brider.
    seeds = _seed_pages(matched_pages, max_seeds)
    for seed_page, seed_score in seeds:
        # Un score nul (ancre non retrouvée ce tour) garde quand même la page comme seed.
        base = float(seed_score) if seed_score and seed_score > 0 else 1.0
        for delta in range(-radius, radius + 1):
            page = seed_page + delta
            if page <= 0 or page not in by_page:
                continue
            value = base * (decay ** abs(delta))
            if value > values.get(page, 0.0):
                values[page] = value

    if values:
        ordered = [page for page, _ in sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))]
    else:
        # Aucun match exploitable : ordre de lecture (cas d'un document ancré non retrouvé).
        ordered = sorted(by_page)

    kept: set = set()
    used = 0
    for page in ordered:
        cost = _records_tokens(by_page[page])
        if kept and used + cost > budget:
            # Page trop volumineuse pour le reste du budget : on tente les suivantes,
            # moins chères, plutôt que d'arrêter net le remplissage.
            continue
        kept.add(page)
        used += cost

    selected = [record for record in records if record[0] in kept]
    # Toutes les pages matchées retenues sont signalées au modèle, pas seulement les seeds
    # du halo : ce sont toutes des « pages retrouvées par la recherche ».
    return selected, sorted(page for page in matched_pages if page in kept)


def _pages_span_summary(pages: List[int]) -> str:
    """Résumé de pages par plages consécutives (« 2-8, 37-43, 85-91 »).

    Une fenêtre gloutonne est souvent DISJOINTE : annoncer « 2-91 » laisserait croire au
    modèle qu'il dispose de tout l'intervalle alors qu'il n'en a que trois morceaux.
    """
    ordered = sorted(set(pages))
    if not ordered:
        return ""
    runs: List[Tuple[int, int]] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        runs.append((start, previous))
        start = previous = page
    runs.append((start, previous))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _pages_in_window(matched_pages: set, radius: int) -> Optional[set]:
    """Ensemble de pages à conserver autour des pages matchées (None = tout le document).

    Mode historique ``CAG_WINDOW_MODE=radius`` — conservé comme repli ; le mode par défaut
    remplit désormais par valeur (``_select_records_by_value``).
    """
    if not matched_pages:
        return None
    keep: set = set()
    for pg in matched_pages:
        for delta in range(-radius, radius + 1):
            if pg + delta > 0:
                keep.add(pg + delta)
    return keep


def _records_tokens(records: List[LeafRecord]) -> int:
    return estimate_tokens("\n".join(text for _, _, text in records))


# --- Profilage de document (phase B du retriever) -------------------------------
#
# Un document ne se lit pas de la même façon selon ce qu'il CONTIENT. Trois archétypes
# observés sur le corpus réel, qui appellent trois traitements différents :
#
#   * ``full_text``   — le texte tient entièrement dans le budget : inutile de chercher
#                       DEDANS, on le donne en entier. Supprime par construction le
#                       défaut « la règle et sa condition d'application séparées par le
#                       chunking » (cas du capot complet / « rénovation uniquement »).
#   * ``windowed``    — trop gros pour tenir : on relance les retrievers BORNÉS à ce
#                       document. Le palier BM25 strict (AND), inatteignable sur tout le
#                       corpus, redevient franchissable sur un seul document.
#   * ``image_first`` — planche CAO quasi muette : le texte extrait ne dit presque rien,
#                       seule l'image porte l'information (cotes). Les PNG font foi.
#
# Le mode n'est PAS un réglage : il se déduit du document. C'est ce qui remplace le

MODE_FULL_TEXT = "full_text"
MODE_WINDOWED = "windowed"
MODE_IMAGE_FIRST = "image_first"

# En dessous, le texte extrait est trop maigre pour porter une réponse (planche CAO
# transcrite en simple liste d'étiquettes, sans les cotes).
_IMAGE_FIRST_MAX_TOKENS = 800


@dataclass
class DocumentProfile:
    """Ce que contient réellement un document, et comment il doit donc être lu."""

    document_id: int
    page_count: int
    pages_with_text: int
    text_tokens: int
    mode: str

    def to_trace(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pages": self.page_count,
            "pages_with_text": self.pages_with_text,
            "text_tokens": self.text_tokens,
            "mode": self.mode,
        }


def _count_document_pages(session: Session, document_id: int) -> int:
    """Nombre de pages du document (0 si indéterminable).

    Les chunks ``page_anchor`` valent 1 par page PDF (posés à l'ingestion) ; on retombe
    sur le nombre de pages distinctes portant du texte si l'ancrage manque.
    """
    try:
        row = session.execute(
            text(
                """
                SELECT count(DISTINCT COALESCE(
                           metadata_json->>'page_no', metadata_json->>'page_start'))
                FROM documentchunk
                WHERE document_id = :doc_id
                  AND COALESCE(metadata_json->>'page_no',
                               metadata_json->>'page_start') IS NOT NULL
                """
            ),
            {"doc_id": int(document_id)},
        ).scalar()
        return int(row or 0)
    except Exception as exc:  # pragma: no cover - lecture best-effort
        logger.warning("[profil] comptage des pages impossible (doc %s) : %s", document_id, exc)
        return 0


def profile_document(
    session: Session,
    document_id: int,
    *,
    full_doc_max_tokens: Optional[int] = None,
) -> DocumentProfile:
    """Profile un document pour décider comment l'explorer (cf. archétypes ci-dessus).

    Réutilise le cache de chunks feuilles de ``_load_leaf_records`` : appelée une fois
    par document élu et par requête, elle ne coûte donc qu'une requête de comptage.
    """
    threshold = (
        full_doc_max_tokens
        if full_doc_max_tokens is not None
        else settings.CAG_FULL_DOC_MAX_TOKENS
    )
    records = _load_leaf_records(session, document_id)
    text_tokens = _records_tokens(records)
    pages_with_text = len({page for page, _, _ in records if page})
    page_count = _count_document_pages(session, document_id) or pages_with_text

    # Un document dont presque aucune page ne « parle » ne peut pas être répondu par le
    # texte, quel que soit le budget : ce test passe donc AVANT celui du texte intégral.
    mute_page_budget = max(2, page_count // 10)
    if text_tokens < _IMAGE_FIRST_MAX_TOKENS or pages_with_text <= mute_page_budget:
        mode = MODE_IMAGE_FIRST
    elif text_tokens <= threshold:
        mode = MODE_FULL_TEXT
    else:
        mode = MODE_WINDOWED

    return DocumentProfile(
        document_id=int(document_id),
        page_count=page_count,
        pages_with_text=pages_with_text,
        text_tokens=text_tokens,
        mode=mode,
    )


def _trim_records_to_budget(
    records: List[LeafRecord], matched_pages: set, remaining: int
) -> List[LeafRecord]:
    """Rogne un extrait qui dépasse le budget en retirant d'abord les pages les plus
    ÉLOIGNÉES des pages matchées (et non les dernières du document : la réponse est
    souvent juste après le match, pas avant)."""

    def dist(page: int) -> int:
        if not matched_pages:
            return 0
        return min(abs(page - m) for m in matched_pages)

    trimmed = list(records)
    while trimmed and _records_tokens(trimmed) > remaining:
        pages = {page for page, _, _ in trimmed}
        worst = max(pages, key=lambda p: (dist(p), p))
        trimmed = [r for r in trimmed if r[0] != worst]
    return trimmed


def _render_document_block(
    doc: Document,
    records: List[LeafRecord],
    *,
    index: int,
    full: bool,
    seed_pages: Optional[Set[int]] = None,
) -> Tuple[str, List[int]]:
    """Rend un document (ou extrait) avec en-tête métadonnées + marqueurs de page.

    Les pages retrouvées par la recherche sont explicitement signalées : sans ce marquage,
    tout le classement du retriever s'évapore au moment du packing et le modèle reçoit des
    dizaines de pages indifférenciées, sans savoir laquelle a motivé la sélection.
    """
    seed_pages = seed_pages or set()
    mark_seeds = settings.CAG_MARK_MATCHED_PAGES
    title = doc.title or "Document sans titre"
    header_bits: List[str] = []
    if doc.source:
        header_bits.append(f"Source : {doc.source}")
    if getattr(doc, "proferm_gammes", None):
        header_bits.append(f"Gamme : {', '.join(doc.proferm_gammes)}")
    if getattr(doc, "materials", None):
        header_bits.append(f"Matériau : {', '.join(doc.materials)}")
    if getattr(doc, "product_types", None):
        header_bits.append(f"Type : {', '.join(doc.product_types)}")

    lines: List[str] = [f"=== DOCUMENT {index} (id {doc.id}) : « {title} » ==="]
    if header_bits:
        lines.append(" | ".join(header_bits))

    pages_included: List[int] = []
    current_page = None
    for page, _, text in records:
        if page and page != current_page:
            current_page = page
            pages_included.append(page)
            if mark_seeds and page in seed_pages:
                lines.append(f"\n[page {page} — ★ page retrouvée par la recherche]")
            else:
                lines.append(f"\n[page {page}]")
        lines.append(text)

    scope = "document complet" if full else "extrait"
    if pages_included:
        span = _pages_span_summary(pages_included)
        lines.insert(1 if not header_bits else 2, f"Pages incluses : {span} ({scope})")

    return "\n".join(lines), pages_included


def build_cag_context(
    session: Session,
    passages: List[Dict[str, Any]],
    *,
    system_prompt: str,
    token_budget: Optional[int] = None,
    max_documents: Optional[int] = None,
    full_doc_max_tokens: Optional[int] = None,
    page_radius: Optional[int] = None,
    anchor_document_ids: Optional[List[int]] = None,
    intent: Optional[str] = None,
    emit_sources_tag: bool = True,
    elected_document_ids: Optional[List[int]] = None,
    pinned_pages: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, Any]:
    """Construit le message système CAG : documents entiers/étendus sous budget de tokens.

    Budget et nombre de documents résolus par priorité : paramètre explicite >
    table par intent (``CAG_BUDGET_BY_INTENT``) > plafonds globaux.

    Retourne un dict ``{"role": "system", "content": ..., "cag_documents": [...]}``
    (compatible avec l'ancien build_space_context_from_passages ; la clé cag_documents
    liste les documents réellement inclus, pour les sources côté UI et les images).
    """
    intent_budget, intent_max_docs = budget_for_intent(intent)
    token_budget = token_budget if token_budget is not None else intent_budget
    max_documents = max_documents if max_documents is not None else intent_max_docs
    full_doc_max_tokens = (
        full_doc_max_tokens if full_doc_max_tokens is not None else settings.CAG_FULL_DOC_MAX_TOKENS
    )
    page_radius = page_radius if page_radius is not None else settings.CAG_PAGE_RADIUS

    system_message: Dict[str, Any] = {"role": "system", "content": system_prompt}

    if not passages:
        system_message["content"] += "\n\nAucun passage trouvé dans cet espace pour cette requête."
        system_message["cag_documents"] = []
        return system_message

    ranked_docs = aggregate_documents(passages, max_documents=max_documents)

    # Élection du juge (B6) : quand le juge de suffisance a statué, son classement prime —
    # documents élus en tête, pages citées promues seeds. L'ancre conversationnelle est
    # alors ignorée : le juge a vu les dossiers candidats (ancre comprise) et a tranché.
    ranked_docs = _apply_document_election(
        ranked_docs, elected_document_ids, pinned_pages, max_documents=max_documents
    )

    # GARANTIE d'ancrage : les documents du sujet courant de la conversation sont TOUJOURS
    # packés, en tête, même si le retrieval de ce tour ne les a pas fait remonter (ex. suivi
    # « tu as ses dimensions ? » où le mot "dimensions" tire vers un autre manuel). Un boost
    # de ranking ne peut pas repêcher un document absent du pool — l'inclusion ici, si.
    # Plafonné par CAG_ANCHOR_SLOTS : avec 3 ancres et max_documents=3, les documents
    # (souvent faux) d'un tour raté consommaient TOUS les slots, et le bon document trouvé
    # au tour suivant — périmètre confirmé compris — n'avait plus de place. Les ancres
    # au-delà du plafond restent packées si le retrieval de ce tour les a fait remonter.
    anchor_slots = max(0, settings.CAG_ANCHOR_SLOTS)
    if anchor_document_ids and anchor_slots > 0 and not elected_document_ids:
        forced_ids: List[int] = []
        for aid in anchor_document_ids:
            aid = int(aid)
            if aid not in forced_ids:
                forced_ids.append(aid)
            if len(forced_ids) >= anchor_slots:
                break
        by_id = {did: meta for did, meta in ranked_docs}
        anchored: List[Tuple[int, Dict[str, Any]]] = []
        for aid in forced_ids:
            anchored.append((aid, by_id.pop(aid, _new_document_entry())))
        others = [(did, meta) for did, meta in ranked_docs if did in by_id]
        # Jamais tronquer les ancres ; le reste complète jusqu'au plafond documents.
        ranked_docs = anchored + others[: max(0, max_documents - len(anchored))]
        if settings.CAG_ANCHOR_RANK_BY_SCORE:
            # Garantir la PRÉSENCE n'est pas garantir la PRIORITÉ : l'ancre reste packée
            # mais reprend son rang réel, donc elle ne capte plus d'office la part de
            # budget du rang 1 (cf. CAG_DOC_BUDGET_SHARES). Une ancre non retrouvée ce
            # tour a un score nul et passe donc en dernier — présente, mais servie après.
            ranked_docs.sort(key=lambda kv: float(kv[1].get("election_score") or 0.0), reverse=True)

    doc_ids = [did for did, _ in ranked_docs]
    docs_by_id = {
        d.id: d
        for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
    }

    blocks: List[str] = []
    cag_documents: List[Dict[str, Any]] = []
    spent_tokens = 0
    position = 0
    # Partage du budget par rang d'élection, avec report de la part non consommée.
    shares = _budget_shares(len(ranked_docs))
    carry = 0
    max_seeds = settings.CAG_MAX_SEEDS_PER_DOC
    greedy_window = (settings.CAG_WINDOW_MODE or "").strip().lower() != "radius"

    for slot, (did, meta) in enumerate(ranked_docs):
        remaining = token_budget - spent_tokens
        if remaining <= 0:
            break
        # Part de ce rang (+ report) : sans partage, le document n°1 pouvait avaler tout
        # le budget et ne laisser que des miettes aux suivants.
        doc_budget = remaining
        if shares:
            doc_budget = min(remaining, int(token_budget * shares[slot]) + carry)

        doc = docs_by_id.get(did)
        leaf_records = _load_leaf_records(session, did) if doc is not None else []
        if doc is None or not leaf_records:
            carry = doc_budget if shares else 0
            continue

        matched_pages: Dict[int, float] = meta.get("matched_pages") or {}
        full_tokens = _records_tokens(leaf_records)

        # Décision : document entier vs extrait ciblé sur les pages matchées.
        if full_tokens <= full_doc_max_tokens and full_tokens <= doc_budget:
            selected = leaf_records
            seed_pages = sorted(matched_pages)
            full = True
        elif greedy_window:
            selected, seed_pages = _select_records_by_value(
                leaf_records,
                matched_pages,
                doc_budget,
                radius=page_radius,
                max_seeds=max_seeds,
                decay=settings.CAG_NEIGHBOR_DECAY,
            )
            full = False
        else:
            window = _pages_in_window(set(matched_pages), page_radius)
            selected = [r for r in leaf_records if window is None or r[0] in window]
            selected = _trim_records_to_budget(selected, set(matched_pages), doc_budget)
            seed_pages = sorted(matched_pages)
            full = False

        if not selected:
            carry = doc_budget if shares else 0
            continue

        position += 1
        block, pages_included = _render_document_block(
            doc, selected, index=position, full=full, seed_pages=set(seed_pages)
        )
        block_tokens = estimate_tokens(block)
        if block_tokens > remaining and position > 1:
            # Ne pas dépasser le budget (on garde toujours au moins le 1er document).
            position -= 1
            break

        blocks.append(block)
        spent_tokens += block_tokens
        carry = max(0, doc_budget - block_tokens) if shares else 0
        included_pages = set(pages_included)
        cag_documents.append(
            {
                "index": position,
                "document_id": did,
                "document_title": doc.title,
                "pages": sorted(included_pages),
                "full_document": full,
                "score": round(float(meta["score_max"]), 4),
                "election_score": round(float(meta.get("election_score") or 0.0), 4),
                "matched_pages": sorted(matched_pages),
                "seed_pages": sorted(p for p in seed_pages if p in included_pages),
                "has_source_file": bool(getattr(doc, "source_file_path", None)),
            }
        )

    cag_preamble = (
        "\n\nDOCUMENTS (contexte complet) — chaque document ci-dessous est fourni ENTIER ou en "
        "extrait étendu, avec un en-tête (source, gamme, matériau, identifiant « id N ») et ses "
        "numéros de page.\n"
        "IMPÉRATIF : avant d'attribuer une valeur, une cote ou une consigne à une gamme/produit, "
        "vérifie l'en-tête du document concerné. Ne transfère JAMAIS une information d'un document "
        "vers une autre gamme (ex. Perform 70 ≠ Perform 76). Les documents sont classés par "
        "pertinence décroissante.\n\n"
    )
    system_message["content"] += cag_preamble + "\n\n".join(blocks)
    system_message["content"] += f"\n\n({len(blocks)} document(s), ~{spent_tokens} tokens de contexte.)"
    if emit_sources_tag:
        system_message["content"] += (
            "\n\nFIN DE RÉPONSE OBLIGATOIRE : termine ta réponse par une ligne EXACTEMENT au format "
            '<sources>{"used":[{"doc":1,"pages":[3,4]}]}</sources> listant les index de DOCUMENT '
            "et les pages que tu as réellement utilisés pour répondre (liste vide si aucun). "
            "Cette ligne est masquée à l'utilisateur — n'en parle jamais dans le corps de la réponse."
        )
    system_message["cag_documents"] = cag_documents
    # Blocs documents SEULS (sans prompt système ni préambule) : c'est cette matière — et
    # elle seule — qui constitue la preuve. Le juge de vérification la consomme telle quelle
    # au lieu de tronquer un texte qui commençait par 5 000 caractères de consignes.
    system_message["cag_document_blocks"] = blocks

    logger.info(
        "[CAG] %d document(s) packé(s), ~%d tokens (budget %d, intent=%s, élection=%s) — %s",
        len(blocks),
        spent_tokens,
        token_budget,
        intent or "n/a",
        settings.CAG_ELECTION_MODE,
        ", ".join(
            "doc={id}{full} élu={elec} pages★={seeds}".format(
                id=d["document_id"],
                full="(complet)" if d["full_document"] else "",
                elec=d.get("election_score"),
                seeds=d.get("seed_pages") or "—",
            )
            for d in cag_documents
        ),
    )
    return system_message


def _pages_span_label(pages: List[int]) -> str:
    """Libellé humain d'un ensemble de pages ("page 5" / "pages 1-12")."""
    if not pages:
        return "pages n/a"
    lo, hi = min(pages), max(pages)
    return f"page {lo}" if lo == hi else f"pages {lo}-{hi}"


def build_document_sources(
    cag_documents: List[Dict[str, Any]], used_pages_by_index: Optional[Dict[Any, Any]] = None
) -> List[Dict[str, Any]]:
    """Sources UI par DOCUMENT packé — reflète le contexte que le modèle a réellement lu.

    ``used_pages_by_index`` vient du bloc final <sources> émis par le modèle : quand il est
    exploitable, seuls les documents réellement UTILISÉS sont affichés (avec leurs pages) ;
    sinon (None / vide / inexploitable) on retombe sur tous les documents packés. La forme
    des entrées reste compatible avec les badges front (index, document_title, page_no,
    excerpt, score, has_source_file, cag_document)."""
    used_pages_by_index = used_pages_by_index or {}
    entries: List[Dict[str, Any]] = []
    for d in cag_documents:
        idx = d.get("index")
        if used_pages_by_index and idx not in used_pages_by_index:
            continue
        used_pages = [p for p in (used_pages_by_index.get(idx) or []) if isinstance(p, int)]
        pages = [p for p in (d.get("pages") or []) if isinstance(p, int)]
        matched = [p for p in (d.get("matched_pages") or []) if isinstance(p, int)]
        landing_candidates = used_pages or matched or pages
        landing = landing_candidates[0] if landing_candidates else None

        scope = ("Document complet" if d.get("full_document") else "Extrait") + f" ({_pages_span_label(pages)})"
        if used_pages:
            scope += " — pages utilisées : " + ", ".join(str(p) for p in used_pages)

        entries.append(
            {
                "index": idx,
                "document_id": d.get("document_id"),
                "document_title": d.get("document_title") or "Document sans titre",
                "excerpt": scope,
                "passage_full": scope,
                "score": float(d.get("score") or 0.0),
                "page_no": landing,
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "section": None,
                "has_source_file": bool(d.get("has_source_file")),
                "cag_document": True,
                "pages": pages,
                "used_pages": used_pages,
                "full_document": bool(d.get("full_document")),
            }
        )

    if not entries and cag_documents:
        # Le bloc <sources> du modèle a tout écarté (ou est inexploitable) → afficher
        # tous les documents packés plutôt que rien.
        return build_document_sources(cag_documents, {})
    return entries


def select_cag_images(
    session: Session,
    cag_documents: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    *,
    max_images: Optional[int] = None,
    dpi: Optional[int] = None,
    visual_only: bool = False,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Sélectionne et rend les PNG de pages pour la génération vision, ALIGNÉS sur le
    contexte packé : uniquement des pages incluses dans un document CAG.

    Le texte des pages est TOUJOURS dans le contexte ; l'image le complète. Priorité :
      0. pages d'un document « image_first » (texte quasi absent : l'image EST le contenu),
         celles à besoin visuel d'abord ;
      1. pages à besoin visuel (``needs_page_image`` : planche vue par ColPali, texte
         placeholder) ;
      2. autres pages matchées — au score décroissant dans chaque classe.
    ``visual_only`` ne retient que les pages muettes (classes 0 et 1) : c'est le pack initial
    du lecteur agentique, qui demandera lui-même les autres images (lire_pages, zoomer).

    Retourne (images_b64, captions) — captions = [{image_index, document_index,
    document_title, page_no}, …] pour légender les images dans le message user.
    """
    import base64
    import os

    from app.services.multimodal_page_service import render_page_png_cached

    max_images = max_images if max_images is not None else settings.CAG_MAX_IMAGES
    # Limite DURE de l'API Mistral : au-delà, la requête entière est rejetée en 400
    # ("Total number of images exceeds the maximum allowed of 8", code 3051).
    if max_images > MISTRAL_MAX_IMAGES_PER_REQUEST:
        logger.warning(
            "[CAG] %d images demandées mais l'API Mistral en accepte %d au maximum — plafonné.",
            max_images,
            MISTRAL_MAX_IMAGES_PER_REQUEST,
        )
        max_images = MISTRAL_MAX_IMAGES_PER_REQUEST
    dpi = dpi if dpi is not None else settings.CAG_IMAGE_DPI
    if max_images <= 0 or not cag_documents:
        return [], []

    included: Dict[int, Dict[str, Any]] = {
        int(d["document_id"]): d for d in cag_documents if d.get("document_id") is not None
    }

    # Profil par document (cache des feuilles → une requête de comptage) : un document
    # muet en texte fait passer TOUTES ses pages matchées en tête.
    image_first_docs: set = set()
    for did in included:
        try:
            if profile_document(session, did).mode == MODE_IMAGE_FIRST:
                image_first_docs.add(did)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[CAG] profil indisponible doc=%s : %s", did, exc)

    # Candidats (doc_id, page) depuis les passages, page ancre uniquement (pas les voisins :
    # le texte des voisins est déjà dans le contexte, le PNG n'apporte que pour le match).
    candidates: List[Tuple[int, int, int]] = []  # (rang, doc_id, page)
    seen: set = set()
    for p in sorted(passages, key=lambda x: float(x.get("score") or 0.0), reverse=True):
        did = p.get("document_id")
        if did is None or int(did) not in included:
            continue
        did = int(did)
        page = p.get("page_no") or p.get("page_start")
        if not isinstance(page, int) or page <= 0:
            continue
        if page not in (included[did].get("pages") or []):
            continue
        if (did, page) in seen:
            continue
        seen.add((did, page))
        needs = bool(p.get("needs_page_image"))
        rank = (0 if did in image_first_docs else 2) + (0 if needs else 1)
        if visual_only and rank >= 3:
            continue
        candidates.append((rank, did, page))

    # Tri stable : classe d'abord, ordre score conservé au sein de chaque classe.
    candidates.sort(key=lambda t: t[0])

    images_b64: List[str] = []
    captions: List[Dict[str, Any]] = []
    doc_cache: Dict[int, Optional[Document]] = {}
    for _, did, page in candidates:
        if len(images_b64) >= max_images:
            break
        if did not in doc_cache:
            doc_cache[did] = session.get(Document, did)
        doc = doc_cache[did]
        if not doc or not doc.source_file_path or not os.path.exists(doc.source_file_path):
            continue
        try:
            png_bytes = render_page_png_cached(doc.source_file_path, page, dpi=dpi)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CAG] rendu PNG échoué (doc=%s page=%s): %s", did, page, exc)
            continue
        images_b64.append(base64.b64encode(png_bytes).decode("utf-8"))
        captions.append(
            {
                "image_index": len(images_b64),
                "document_index": included[did].get("index"),
                "document_title": included[did].get("document_title"),
                "page_no": page,
            }
        )

    logger.info(
        "[CAG] %d image(s) alignée(s) sur le contexte packé (max %d, visual_only=%s) — %s",
        len(images_b64),
        max_images,
        visual_only,
        ", ".join(f"doc{c['document_index']}:p{c['page_no']}" for c in captions),
    )
    return images_b64, captions
