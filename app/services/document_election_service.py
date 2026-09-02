"""Élection du ou des documents porteurs de la réponse (phase A du retriever).

Le retriever historique cherchait directement les bonnes PAGES parmi tout le corpus,
puis coupait à ``top_k`` avec un quota par document. Deux conséquences mesurées :

1. La preuve « ce document place 15 pages pertinentes » était DÉTRUITE avant d'être
   comptée : le quota (8 pages/doc) et la coupe (top-20) la ramenaient à 8, et
   l'élection du packer ne voyait plus que ce résidu.
2. Trois documents entraient dans le contexte dès que trois documents scoraient
   (``CAG_MAX_DOCUMENTS``), et la génération recollait leurs références entre elles —
   d'où des réponses attribuant une cote d'un produit à un autre.

Ce module répond à la question amont « DANS QUEL document est la réponse ? » sur le
pool fusionné COMPLET (avant toute coupe), et n'élit plus qu'un document par défaut.
Un second (voire un troisième) n'est admis que sur PREUVE : question comparative,
complémentarité d'archétypes (une notice texte + une planche muette lisible seulement
en image), ou détention du meilleur passage du pool. Un doute ne doit pas produire un
empilement : il produit un score serré, visible dans la trace, que le juge de suffisance
peut ensuite contredire.

Depuis le 01/09, l'élection suit d'abord les PASSAGES : les documents qui détiennent
l'un des ``TOP_PASSAGE_HOLDERS`` meilleurs passages du pool sont élus d'office (le
premier est le dominant), et l'agrégation par document ne sert plus qu'à compléter les
places restantes. Raison : le score par document récompense l'affinité thématique
(« beaucoup de pages parlent du sujet ») plutôt que la preuve (« une page porte la
réponse ») — mesuré dans les deux sens sur le corpus réel. Voir ``elect_documents``.

Aucun appel LLM ici : uniquement de l'agrégation et une lecture de métadonnées.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document

logger = logging.getLogger(__name__)

# Familles de preuve : deux canaux lisant la MÊME évidence ne comptent qu'une fois.
# (Aligné sur context_packer_service._CHANNEL_FAMILIES.)
_CHANNEL_FAMILIES = {
    "bm25": "texte",
    "colpali": "visuel",
}

# Un candidat n'est admis à côté du dominant que si son score reste dans cette
# proportion du sien. En dessous, l'écart est jugé net : le dominant part seul.
DOMINANCE_RATIO = 0.70

# Plafonds structurels.
MAX_ELECTED = 3
MAX_CANDIDATES = 5

# Les documents qui détiennent l'un des K meilleurs PASSAGES du pool fusionné sont élus
# d'office, dans l'ordre de leurs passages. Mesuré le 01/09 sur 6 requêtes réelles
# (espace 28) : la fusion place le passage du bon document en tête dans 6 cas sur 6,
# alors que l'agrégation par document ne le classait premier que 4 fois sur 5 et
# n'élisait PAS le document qui a répondu sur le cas TGY3704 (son passage était 3e).
# C'est l'observation d'Elie — « le bon document n'est jamais top 1 à 100 %, mais dans le
# top 3 à plus de 85 % » — traduite en règle : on fait confiance au retriever pour les
# passages, et à l'élection seulement pour ce qu'elle sait faire (ordonner, compléter).
TOP_PASSAGE_HOLDERS = 3

# Un passage ne compte comme « l'un des meilleurs » que s'il n'est pas DÉCROCHÉ du premier :
# même seuil que la dominance. Sur un vrai pool RRF les trois premiers passages tiennent
# dans ~10 % (compression des rangs), donc ce plancher ne mord jamais en pratique ; il
# empêche seulement qu'un pool étriqué (deux pages, 0,90 contre 0,10) élise le second
# document au seul motif qu'il n'y en avait pas de troisième.
TOP_PASSAGE_MIN_RATIO = DOMINANCE_RATIO

# Un complément « texte » (dominant visuel, candidat trouvé par BM25 seulement) doit
# apporter un VRAI match lexical, pas une miette du repli en OU. Mesuré : le repli OR
# produit des ts_rank_cd quantifiés ~0,1 par terme (0,2…0,9 sur 6 requêtes, 30 à 75 % des
# pages sous 0,4 × max) ; une page qui matche la question entière monte à 4,2. Sous cette
# barre, une notice générique de 124 pages entrait en complément sur « rallonge TGY3704 »
# par sa seule différence de canal, alors que BM25 n'y trouvait que des mots communs.
BM25_COMPLEMENT_MIN_RANK = 1.0

# Plafond du bonus de distribution. Plus généreux que le plafond du packer (2) parce
# qu'ici on compte les pages du pool COMPLET : c'est justement le signal qu'on veut
# restaurer. Borné à 4 pour que le volume ne renverse jamais une meilleure page nette
# (leçon du cas « catalogue à six pages moyennes contre notice à une bonne page »).
ELECTION_PAGE_CAP = 4

# Nombre de pages remontées par candidat dans le pack du juge.
CANDIDATE_PAGES_PER_DOC = 6

# Marqueurs d'une vraie question comparative. Volontairement lexicaux et restreints :
# une question d'ÉNUMÉRATION (« pour chaque seuil, les embouts ») cite beaucoup de
# références mais veut UN seul document — la compter comme comparative rouvrirait la
# porte au recollage inter-documents qu'on cherche à fermer.
_COMPARATIVE_PATTERNS = (
    r"\bdifferences?\b",
    r"\bcomparer\b",
    r"\bcomparaison\b",
    r"\bcomparatif\b",
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bpar rapport (?:a|au|aux)\b",
    r"\bplutot que\b",
    r"\bequivalent (?:de|du|chez)\b",
    r"\blequel (?:des|est)\b",
    r"\bquelle difference\b",
)

# Jetons de VERSION à retirer du titre pour reconnaître deux révisions du même
# document. Uniquement des marqueurs sans ambiguïté : la sur-fusion ferait disparaître
# un document légitimement distinct (perte de rappel silencieuse), alors que la
# sous-fusion est rattrapée en aval par la règle de dominance.
#
# Appliqués APRÈS normalisation des séparateurs en espaces : dans « DTA_..._V5 », le
# souligné est un caractère de mot, donc « \bv\d+\b » n'y trouve aucune frontière.
#
# Les dates exigent des valeurs PLAUSIBLES (jour 1-31, mois 1-12, année 19xx/20xx) :
# un motif large « \d{1,2} \d{1,2} \d{2,4} » avalait le numéro d'avis « 6/16-2335 » et
# réduisait tous les DTA à la même famille — exactement la sur-fusion à éviter.
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_MONTH = r"(?:0?[1-9]|1[0-2])"
_YEAR = r"(?:19|20)\d{2}"

_VERSION_TOKEN_PATTERNS = (
    r"\bv\d+(?:\s\d+)*\b",                                  # V5, v1 2 (ex-v1.2)
    r"\b(?:cc|ind|indice|rev|revision|ed|edition)\s*\d+\b",  # CC01, rev 2, indice 3
    rf"\b{_DAY}\s{_MONTH}\s{_YEAR}\b",                      # 24 08 2026
    rf"\b{_YEAR}\s{_MONTH}\s{_DAY}\b",                      # 2026 08 26
    rf"\b{_YEAR}\b",                                        # millésime nu
)

# « Notice pose (1).pdf » : marqueur de copie de téléchargement. Retiré AVANT la
# normalisation des séparateurs, tant que les parenthèses existent encore — un motif
# « chiffre isolé » appliqué après aurait mangé le « 6 » de l'avis « 6/16-2335 ».
_COPY_MARKER_RE = re.compile(r"\(\s*\d+\s*\)")


@dataclass
class ElectedDocument:
    """Un document candidat/élu, avec la preuve qui l'a porté."""

    document_id: int
    title: str
    election_score: float
    score_max: float
    matched_pages: Dict[int, float] = field(default_factory=dict)
    families: Set[str] = field(default_factory=set)
    colpali_best: float = 0.0
    bm25_best: float = 0.0
    role: str = "candidate"           # dominant | complement | candidate
    reason: str = ""

    @property
    def page_count(self) -> int:
        return len(self.matched_pages)

    def top_pages(self, limit: int) -> List[int]:
        """Pages les mieux notées, décroissant (départage stable par n° de page)."""
        ordered = sorted(
            self.matched_pages.items(), key=lambda kv: (-kv[1], kv[0])
        )
        return [page for page, _ in ordered[: max(0, limit)]]

    def to_trace(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_title": self.title,
            "election_score": round(self.election_score, 6),
            "score_max": round(self.score_max, 6),
            "pages": self.page_count,
            "top_pages": self.top_pages(5),
            "families": sorted(self.families),
            "role": self.role,
            "reason": self.reason,
        }


@dataclass
class ElectionResult:
    """Sortie de la phase A : qui est élu, qui reste candidat, et pourquoi."""

    elected: List[ElectedDocument] = field(default_factory=list)
    candidates: List[ElectedDocument] = field(default_factory=list)
    rejected_versions: List[Dict[str, Any]] = field(default_factory=list)
    decision: str = "empty"
    margin: float = 0.0               # score_2 / score_1 (0 si un seul candidat)

    @property
    def elected_ids(self) -> List[int]:
        return [d.document_id for d in self.elected]

    @property
    def dominant(self) -> Optional[ElectedDocument]:
        return self.elected[0] if self.elected else None

    def to_trace(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "margin": round(self.margin, 4),
            "elected": [d.to_trace() for d in self.elected],
            "candidates": [
                {
                    "document_id": d.document_id,
                    "document_title": d.title,
                    "election_score": round(d.election_score, 6),
                    "pages": d.page_count,
                }
                for d in self.candidates
            ],
            "rejected_versions": self.rejected_versions,
        }


def _strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", value)
        if unicodedata.category(ch) != "Mn"
    )


def normalize_title_family(title: Optional[str]) -> str:
    """Clé de « famille » d'un titre : deux révisions du même document la partagent.

    Retire l'extension, les accents, les jetons de version et les dates, puis compacte
    la ponctuation ET les espaces (« Perform 76 » et « Perform76 » doivent converger).
    Conservateur par choix : mieux vaut manquer une fusion (les deux versions restent
    candidates, la dominance en élira une) que fusionner deux documents distincts.
    """
    if not title:
        return ""
    text = _strip_accents(str(title)).lower()
    text = re.sub(r"\.(pdf|docx?|xlsx?|pptx?)$", " ", text)
    text = _COPY_MARKER_RE.sub(" ", text)
    # Séparateurs → espaces AVANT les motifs de version : sans ça « _v5 » n'offre
    # aucune frontière de mot (le souligné est un caractère de mot).
    text = re.sub(r"[^a-z0-9]+", " ", text)
    for pattern in _VERSION_TOKEN_PATTERNS:
        text = re.sub(pattern, " ", text)
    # Suppression totale des espaces : la clé devient insensible au découpage des
    # mots (« perform 76 » ≡ « perform76 »).
    return re.sub(r"\s+", "", text).strip()


def is_comparative_query(
    query_text: Optional[str], signals: Optional[Any] = None
) -> Tuple[bool, str]:
    """La question demande-t-elle explicitement une mise en regard de deux produits ?

    Renvoie ``(True, marqueur)`` uniquement sur un marqueur lexical franc. Le nombre de
    références citées n'est PAS un signal : une énumération en cite beaucoup et veut un
    seul document.
    """
    if not query_text:
        return False, ""
    text = _strip_accents(str(query_text)).lower()
    for pattern in _COMPARATIVE_PATTERNS:
        if re.search(pattern, text):
            return True, pattern.strip("\\b")
    return False, ""


def _hit_families(sources: Iterable[str]) -> Set[str]:
    return {_CHANNEL_FAMILIES[s] for s in (sources or ()) if s in _CHANNEL_FAMILIES}


def score_documents(fused_hits: Sequence[Any]) -> List[ElectedDocument]:
    """Agrège les hits fusionnés PAR DOCUMENT et classe les documents.

    ``score(doc) = score_max × (1 + bonus_familles + bonus_distribution)``

    Même philosophie que l'élection du packer (le meilleur passage domine), mais nourrie
    par le pool COMPLET : le nombre de pages retrouvées redevient une preuve, avec un
    bonus borné pour ne jamais renverser une meilleure page nette.

    Accepte tout objet portant ``document_id``, ``page_no``, ``rrf_score``,
    ``retrieval_sources`` (+ éventuellement ``document_title``, ``colpali_score``,
    ``bm25_score``) — les tests peuvent donc passer des stubs légers.
    """
    agg: Dict[int, ElectedDocument] = {}
    for hit in fused_hits or ():
        did = getattr(hit, "document_id", None)
        if did is None:
            continue
        did = int(did)
        score = float(getattr(hit, "rrf_score", 0.0) or 0.0)
        entry = agg.get(did)
        if entry is None:
            entry = ElectedDocument(
                document_id=did,
                title=str(getattr(hit, "document_title", None) or "Document sans titre"),
                election_score=0.0,
                score_max=0.0,
            )
            agg[did] = entry

        entry.score_max = max(entry.score_max, score)
        entry.families.update(_hit_families(getattr(hit, "retrieval_sources", ()) or ()))
        entry.colpali_best = max(
            entry.colpali_best, float(getattr(hit, "colpali_score", 0.0) or 0.0)
        )
        entry.bm25_best = max(
            entry.bm25_best, float(getattr(hit, "bm25_score", 0.0) or 0.0)
        )

        page = getattr(hit, "page_no", None)
        if isinstance(page, int) and page > 0:
            if score > entry.matched_pages.get(page, 0.0):
                entry.matched_pages[page] = score

    for entry in agg.values():
        entry.election_score = _compute_election_score(entry)

    return sorted(
        agg.values(),
        key=lambda d: (-d.election_score, -d.score_max, d.document_id),
    )


def _compute_election_score(entry: ElectedDocument) -> float:
    base = float(entry.score_max or 0.0)
    if base <= 0:
        # Bonus multiplicatif sur un score nul/négatif n'a pas de sens : score brut.
        return base
    n_families = max(1, len(entry.families))
    n_pages = max(1, len(entry.matched_pages))
    bonus = settings.CAG_ELECTION_FAMILY_BONUS * (n_families - 1) + (
        settings.CAG_ELECTION_PAGE_BONUS * min(n_pages - 1, ELECTION_PAGE_CAP)
    )
    return base * (1.0 + bonus)


def dedupe_versions(
    session: Optional[Session], candidates: List[ElectedDocument]
) -> Tuple[List[ElectedDocument], List[Dict[str, Any]]]:
    """Ne garde qu'une révision par famille de titre (la plus récemment mise à jour).

    Deux versions du même cahier technique (CC01 du 24/08, CC02 du 26/08) scorent
    presque identiquement et portent des cotes CONTRADICTOIRES : les packer ensemble
    gaspille le budget et laisse le modèle arbitrer sans critère. Le modèle ``Document``
    n'a pas de champ version → on arbitre sur ``updated_at`` (à défaut ``created_at``,
    puis l'id).
    """
    if not candidates:
        return [], []

    meta: Dict[int, Tuple[Optional[Any], Optional[Any]]] = {}
    if session is not None:
        try:
            rows = session.exec(
                select(Document.id, Document.updated_at, Document.created_at).where(
                    Document.id.in_([c.document_id for c in candidates])
                )
            ).all()
            meta = {int(r[0]): (r[1], r[2]) for r in rows}
        except Exception as exc:  # pragma: no cover - lecture best-effort
            logger.warning("[élection] lecture updated_at impossible : %s", exc)

    def recency_key(doc: ElectedDocument):
        updated, created = meta.get(doc.document_id, (None, None))
        stamp = updated or created
        # Un document sans horodatage ne doit pas gagner par défaut contre un daté.
        return (stamp is not None, stamp, doc.document_id)

    best_by_family: Dict[str, ElectedDocument] = {}
    order: List[str] = []
    rejected: List[Dict[str, Any]] = []

    for doc in candidates:                  # candidats déjà triés par score
        family = normalize_title_family(doc.title)
        if not family:
            family = f"__doc_{doc.document_id}"
        current = best_by_family.get(family)
        if current is None:
            best_by_family[family] = doc
            order.append(family)
            continue
        keep, drop = (
            (current, doc)
            if recency_key(current) >= recency_key(doc)
            else (doc, current)
        )
        best_by_family[family] = keep
        rejected.append(
            {
                "document_id": drop.document_id,
                "document_title": drop.title,
                "kept_document_id": keep.document_id,
                "family": family,
                "reason": "version_superseded",
            }
        )

    kept = [best_by_family[f] for f in order]
    kept.sort(key=lambda d: (-d.election_score, -d.score_max, d.document_id))
    return kept, rejected


def _is_channel_complement(dominant: ElectedDocument, other: ElectedDocument) -> bool:
    """Le candidat apporte-t-il une preuve d'une NATURE que le dominant n'a pas ?

    Cas visé : une notice retrouvée par son texte, complétée par une planche muette que
    seul ColPali voit (et dont le texte extrait ne dirait rien). L'inverse compte aussi.
    """
    if not dominant.families or not other.families:
        return False
    if dominant.families & other.families:
        return False
    if "visuel" in other.families:
        return other.colpali_best >= settings.COLPALI_DOMINANCE_MIN_SCORE
    # Un match BM25 issu du repli en OU (une miette de la question) ne prouve rien.
    return other.bm25_best >= BM25_COMPLEMENT_MIN_RANK


def elect_documents(
    session: Optional[Session],
    fused_hits: Sequence[Any],
    *,
    query_text: Optional[str] = None,
    signals: Optional[Any] = None,
    max_docs: int = MAX_ELECTED,
) -> ElectionResult:
    """Phase A : élit 1 à ``max_docs`` documents sur le pool fusionné complet.

    Deux étages, dans cet ordre :

    1. **Détenteurs des meilleurs passages** (``TOP_PASSAGE_HOLDERS``). Les documents qui
       portent l'un des K premiers passages du pool fusionné sont élus d'office, dans
       l'ordre de leurs passages — le premier est le dominant. C'est la règle qui suit le
       retriever : quand ColPali et la fusion placent une page en tête, son document part
       au contexte, quoi qu'en dise l'agrégation par document.
    2. **Compléments** sur les places restantes, parmi les candidats classés par score
       d'élection : le meilleur document au score d'élection (le « volume », s'il n'est pas
       déjà élu), puis question comparative ou complémentarité de canaux — ces deux
       dernières soumises à ``DOMINANCE_RATIO``.

    POURQUOI cet ordre. Le score d'élection agrège PAR DOCUMENT et récompense donc
    « beaucoup de pages parlent du sujet » plutôt que « une page porte la réponse ». Trois
    pannes mesurées le 01/09 ont cette forme : le catalogue général évince Lumine55 (2 pages,
    dont LA page des 40 dB, meilleur passage du pool) ; un dossier technique Perform76 dont
    aucune page ne parle de couleurs évince le dépliant qui les porte (meilleur passage) ;
    une notice Roto de 124 pages entre en « complément de canal » sur des mots communs et
    prend la place du catalogue SOLEAL que ColPali classe 2e à 0,740 — le document qui a
    répondu. Dans les trois cas la bonne page était en tête du pool fusionné. Aucun réglage
    des bonus ne sépare l'affinité thématique de la preuve ; suivre les passages, si.
    """
    ranked = score_documents(fused_hits)
    if not ranked:
        return ElectionResult(decision="empty")

    ranked, rejected = dedupe_versions(session, ranked)
    by_id = {d.document_id: d for d in ranked}
    candidates = ranked[:MAX_CANDIDATES]
    cap = max(1, min(int(max_docs or 1), MAX_ELECTED))
    comparative, marker = is_comparative_query(query_text, signals)

    # ——— Étage 1 : les détenteurs des K meilleurs passages ———
    # Le pool est trié par la fusion ; on le retrie défensivement, et à égalité de RRF
    # (fréquente : rang 1 dans un canal = rang 1 dans l'autre) le passage vu par ColPali
    # passe devant — c'est le canal fiable sur les planches muettes.
    ordered_hits = sorted(
        (h for h in fused_hits if getattr(h, "document_id", None) is not None),
        key=lambda h: (
            -float(getattr(h, "rrf_score", 0.0) or 0.0),
            -float(getattr(h, "colpali_score", 0.0) or 0.0),
            int(h.document_id),
        ),
    )
    elected: List[ElectedDocument] = []
    passage_rank = 0
    best_rrf = float(getattr(ordered_hits[0], "rrf_score", 0.0) or 0.0) if ordered_hits else 0.0
    for hit in ordered_hits:
        if passage_rank >= TOP_PASSAGE_HOLDERS or len(elected) >= cap:
            break
        if float(getattr(hit, "rrf_score", 0.0) or 0.0) < TOP_PASSAGE_MIN_RATIO * best_rrf:
            break                                   # décroché du meilleur : plus un « top »
        passage_rank += 1
        holder = by_id.get(int(hit.document_id))
        if holder is None:                      # révision écartée par dedupe_versions
            continue
        if any(d.document_id == holder.document_id for d in elected):
            continue
        holder.role = "dominant" if not elected else "complement"
        holder.reason = f"top_passage:{passage_rank}"
        elected.append(holder)

    if not elected:                             # pool sans page exploitable : repli
        dominant = candidates[0]
        dominant.role, dominant.reason = "dominant", "best_election_score"
        elected.append(dominant)
    dominant = elected[0]

    # Marge : rapport des deux meilleurs PASSAGES (pas des scores d'élection, qui ne sont
    # plus monotones dans l'ordre d'élection). 1,0 = passages à égalité parfaite.
    margin = (
        float(ordered_hits[1].rrf_score or 0.0) / float(ordered_hits[0].rrf_score or 0.0)
        if len(ordered_hits) > 1 and float(ordered_hits[0].rrf_score or 0.0) > 0
        else 0.0
    )

    # ——— Étage 2 : compléments sur les places restantes ———
    elected_ids = {d.document_id for d in elected}
    volume_best = candidates[0]
    if len(elected) < cap and volume_best.document_id not in elected_ids:
        # Le gagnant de la formule par document garde une place quand il en reste une :
        # « ce document place beaucoup de pages » reste une preuve, juste plus la première.
        volume_best.role, volume_best.reason = "complement", "best_election_score"
        elected.append(volume_best)
        elected_ids.add(volume_best.document_id)

    for other in candidates:
        if len(elected) >= cap:
            break
        if other.document_id in elected_ids:
            continue
        if dominant.election_score <= 0:
            break
        if other.election_score < DOMINANCE_RATIO * dominant.election_score:
            continue                    # écart net avec le dominant : pas de complément
        if comparative:
            other.role, other.reason = "complement", f"comparative_query:{marker}"
        elif _is_channel_complement(dominant, other):
            other.role, other.reason = "complement", "channel_complement"
        else:
            continue                    # score proche mais AUCUNE preuve → non élu
        elected.append(other)
        elected_ids.add(other.document_id)

    # Un élu peut manquer au top-N par score d'élection (c'est justement sa faiblesse) :
    # il rejoint les candidats pour rester visible dans la trace et le pack du juge.
    known = {c.document_id for c in candidates}
    candidates = candidates + [d for d in elected if d.document_id not in known]

    if len(elected) == 1:
        decision = "mono_document"
    elif comparative and any(d.reason.startswith("comparative_query") for d in elected):
        decision = f"comparative:{len(elected)}"
    elif all(d.reason.startswith("top_passage") for d in elected):
        decision = f"top_passages:{len(elected)}"
    else:
        decision = f"complement:{len(elected)}"

    return ElectionResult(
        elected=elected,
        candidates=candidates,
        rejected_versions=rejected,
        decision=decision,
        margin=margin,
    )


def election_candidate_passages(result: ElectionResult) -> List[Dict[str, Any]]:
    """Passages LÉGERS (sans texte) décrivant les candidats, pour le pack du juge.

    Le juge doit garder une vue LARGE (plusieurs documents peu profonds) alors que la
    génération reçoit une vue ÉTROITE et profonde ; sans cela il ne peut jamais
    contredire l'élection. ``build_cag_context`` rechargeant lui-même le texte depuis la
    base, ces dicts n'ont pas besoin de le porter.
    """
    passages: List[Dict[str, Any]] = []
    for doc in result.candidates:
        for page in doc.top_pages(CANDIDATE_PAGES_PER_DOC):
            passages.append(
                {
                    "document_id": doc.document_id,
                    "document_title": doc.title,
                    "page_no": page,
                    "score": doc.matched_pages.get(page, 0.0),
                    "retrieval_sources": sorted(doc.families),
                }
            )
    return passages


def format_election_log(result: ElectionResult) -> str:
    """Ligne de log lisible (une par requête) — sert au diagnostic en prod."""
    if not result.elected:
        return "[élection] aucun document (pool vide)"
    parts = [
        f"{d.title[:40]}#{d.document_id} score={d.election_score:.4f} pic={d.score_max:.4f} "
        f"pages={d.page_count} fam={'+'.join(sorted(d.families)) or '-'} ({d.role}:{d.reason})"
        for d in result.elected
    ]
    tail = ""
    if result.rejected_versions:
        tail = f" | versions écartées: {len(result.rejected_versions)}"
    others = [c for c in result.candidates if c.document_id not in set(result.elected_ids)]
    if others:
        # Le pic est affiché AUSSI pour les non élus : sans lui, la co-élection d'un
        # détenteur au score d'élection plus faible que le dominant paraît arbitraire.
        tail += " | non élus: " + ", ".join(
            f"{c.title[:30]}#{c.document_id}={c.election_score:.4f}(pic {c.score_max:.4f})"
            for c in others[:3]
        )
    return (
        f"[élection] {result.decision} marge={result.margin:.2f} → "
        + " ; ".join(parts)
        + tail
    )
