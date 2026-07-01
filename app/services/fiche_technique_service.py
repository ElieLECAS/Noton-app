"""Fiche technique — lookup par référence nue → sortie structurée et sourcée.

Problème résolu : l'équipe Qualité tape une référence nue ("Profil 76180",
"notice de montage seuil 76180"). Le pipeline RAG conversationnel demande alors au
LLM de *rédiger* à partir de chunks épars → il brode et finit sur une note
d'incertitude. Une référence n'a AUCUN sens sémantique (son embedding est du bruit)
mais une valeur lexicale exacte, et la sortie attendue n'est pas une dissertation :
c'est un GABARIT à remplir.

Ce service implémente 4 couches :
  0. DÉTECTION  — `detect_reference_query` : regex, zéro LLM, décide du fast-path.
  1. RÉSOLUTION — `_resolve_passages` : récupère les pages qui citent la référence.
  2. EXTRACTION — `extract_fiche` : le LLM REMPLIT un schéma (grounding strict),
                  chaque champ porte sa source ; champ non trouvé ⇒ null, jamais inventé.
  3. RENDU      — `render_fiche_markdown` : rendu déterministe, pas de prose libre.

Point d'entrée orchestrateur : `build_fiche_technique`.
Gardé par settings.FICHE_TECHNIQUE_ENABLED côté routeur : zéro impact tant que False.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticUndefined
from sqlalchemy import func
from sqlmodel import Session, select

from app.config import settings
from app.database import engine
from app.models.document import Document
from app.models.knowledge_entity import (
    ChunkEntityRelation,
    EntityAlias,
    EntityEntityRelation,
    KnowledgeEntity,
)
from app.services.kag_extraction_service import normalize_entity_name
from app.services.kag_graph_service import _entity_type_label, _relation_short_label
from app.services.mistral_service import chat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Couche 0 — Détection de « référence nue »
# ---------------------------------------------------------------------------

# Marqueurs de question en langage naturel : si présents, ce n'est PAS une
# référence nue → on laisse le pipeline RAG / guidé traiter (comment poser, etc.).
_NL_QUESTION_MARKERS = (
    "comment", "pourquoi", "quelle", "quelles", "quel ", "quels ", "peux-tu",
    "peux tu", "explique", "expliquer", "différence", "difference", "conseil",
    "recommande", "que faire", "est-ce", "est ce", "à quoi", "a quoi",
)

# Marqueurs documentaires : leur présence autorise une requête un peu plus longue
# à rester en mode fiche ("notice de montage seuil 76180").
_DOC_LOOKUP_MARKERS = (
    "fiche", "notice", "réf", "ref ", "reference", "référence", "profil", "profilé",
    "profile", "seuil", "vis", "gabarit", "joint", "traverse", "dormant", "ouvrant",
    "montage", "cote", "cotes", "spéc", "spec", "quincaillerie",
)


class ReferenceQuery(BaseModel):
    """Résultat de la détection : la ou les références et le contexte utile."""

    references: List[str] = Field(default_factory=list)
    primary: str = ""
    raw_message: str = ""


def _looks_like_year(token: str) -> bool:
    """Un jeton à 4 chiffres dans une plage d'années (millésimes de documents).

    Le corpus est saturé de millésimes (2023-06_DEPLIANT, catalogue 2024…) : sans ce
    filtre, "référentiel gammes 2024" déclencherait une fiche à tort. Les vraies
    références produit sont soit à 5 chiffres (76180), soit alphanumériques (A076),
    soit hors de cette plage.
    """
    if len(token) != 4 or not token.isdigit():
        return False
    return 1990 <= int(token) <= 2035


def _extract_references(message: str) -> List[str]:
    """Extrait les références techniques (numériques + alphanumériques), dédupliquées."""
    refs: List[str] = []
    seen: set = set()
    for pattern in (settings.FICHE_REFERENCE_ALNUM_PATTERN, settings.FICHE_REFERENCE_PATTERN):
        try:
            for match in re.findall(pattern, message):
                token = str(match).strip()
                key = token.lower()
                if token and key not in seen and not _looks_like_year(token):
                    seen.add(key)
                    refs.append(token)
        except re.error as exc:  # motif env mal formé → on ignore ce motif
            logger.warning("[fiche] motif de référence invalide (%s): %s", pattern, exc)
    return refs


def detect_reference_query(message: str) -> Optional[ReferenceQuery]:
    """Décide si `message` est une référence nue justifiant le fast-path fiche.

    Conservateur par construction : en cas de doute on renvoie None et la requête
    part dans le pipeline RAG normal. Ne fait AUCUN appel LLM.
    """
    text = (message or "").strip()
    if not text:
        return None

    lowered = text.lower()

    # 1. Une vraie question en langage naturel → ce n'est pas un lookup de fiche.
    if any(marker in lowered for marker in _NL_QUESTION_MARKERS) or "?" in text:
        return None

    # 2. Il faut au moins une référence exploitable.
    references = _extract_references(text)
    if not references:
        return None

    # 3. Longueur : soit la requête est courte (référence nue), soit elle porte un
    #    marqueur documentaire explicite ("notice de montage seuil 76180").
    word_count = len(text.split())
    has_doc_marker = any(marker in lowered for marker in _DOC_LOOKUP_MARKERS)
    if word_count > settings.FICHE_MAX_WORDS and not has_doc_marker:
        return None

    return ReferenceQuery(
        references=references,
        primary=references[0],
        raw_message=text,
    )


# ---------------------------------------------------------------------------
# Couche 2 — Schéma de la fiche (le LLM remplit ce gabarit)
# ---------------------------------------------------------------------------


class _NoneTolerant(BaseModel):
    """Base tolérante aux `null` renvoyés par le LLM.

    En Pydantic v2, un défaut ne s'applique QUE si la clé est absente : un `null`
    explicite sur un champ `str` lève une ValidationError. Le LLM émet fréquemment
    `"designation": null`. Ici on retire les `null` explicites lorsqu'un défaut
    (valeur ou factory) existe → le défaut reprend la main. Les champs réellement
    Optionnels (défaut None) gardent leur None.
    """

    @model_validator(mode="before")
    @classmethod
    def _drop_none_with_default(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for name, field in cls.model_fields.items():
                if name in data and data[name] is None:
                    has_default = (
                        field.default is not None and field.default is not PydanticUndefined
                    )
                    has_factory = field.default_factory is not None
                    if has_default or has_factory:
                        data.pop(name)
        return data


class Source(_NoneTolerant):
    document_title: str = ""
    page_no: Optional[int] = None


class Characteristic(_NoneTolerant):
    """Une caractéristique libellé/valeur extraite des passages (schéma non figé).

    Volontairement générique : le LLM choisit les libellés pertinents pour CETTE
    référence (largeur, matière, inertie du renfort, classe AEV, pas de perçage…),
    plutôt qu'un gabarit codé en dur qui laisserait des sections vides.
    """

    libelle: str = ""
    valeur: str = ""
    source: Optional[Source] = None


class FicheCore(_NoneTolerant):
    """Partie « caractéristiques » de la fiche, remplie par le LLM depuis les passages."""

    reference: str = ""
    designation: Optional[str] = None
    type_element: Optional[str] = None  # profilé | seuil | quincaillerie | ...
    resume: Optional[str] = None        # 1-2 phrases factuelles, sourcées
    caracteristiques: List[Characteristic] = Field(default_factory=list)
    avertissements: List[str] = Field(default_factory=list)


class RelatedEntity(BaseModel):
    """Entité qui gravite autour de la référence (issue du graphe KAG, pas du LLM)."""

    name: str = ""
    entity_type: str = "other"
    type_label: str = "Autre"
    description: Optional[str] = None
    relation: str = "Co-occurrence"  # label de relation (graphe) ou co-occurrence
    weight: float = 0.0
    mention_count: int = 0


FICHE_EXTRACTION_SYSTEM_PROMPT = """Tu es l'extracteur technique de PROFERM (menuiserie PVC/aluminium).
On te donne une RÉFÉRENCE (profilé, seuil, visserie, gabarit…) et des PASSAGES issus
des documents techniques. Ta tâche : extraire les CARACTÉRISTIQUES factuelles de cette
référence sous forme de couples libellé/valeur.

TU NE RÉDIGES PAS DE DISSERTATION. Tu extrais des faits.

CHOISIS librement les libellés pertinents pour CETTE référence, tels qu'ils
apparaissent dans les passages (ex : "Largeur", "Profondeur", "Matière", "Inertie du
renfort", "Classe AEV", "Pas de perçage", "Diamètre de vis", "Étape de montage 1"…).
N'impose aucune rubrique : ne remplis que ce que les passages disent réellement.

GROUNDING STRICT (impératif) :
- Fonde-toi EXCLUSIVEMENT sur les PASSAGES. N'invente AUCUNE valeur.
- Restitue les valeurs (cotes, diamètres, références) MOT POUR MOT.
- Pour CHAQUE caractéristique, renseigne "source" = {"document_title": ..., "page_no": N}
  d'après les passages fournis (titre + page entre crochets en tête de chaque passage).
- Si une information n'apparaît pas dans les passages, ne l'invente pas : ne crée pas
  la caractéristique. Une fiche courte mais juste vaut mieux qu'une fiche brodée.

Renseigne "avertissements" (phrases courtes) pour ce qui manque ou reste incertain.

Retourne UNIQUEMENT un objet JSON :
{
  "reference": "string",
  "designation": "string|null",
  "type_element": "profilé|seuil|quincaillerie|outillage|joint|null",
  "resume": "1-2 phrases factuelles|null",
  "caracteristiques": [
    {"libelle": "...", "valeur": "...", "source": {"document_title": "...", "page_no": N}}
  ],
  "avertissements": ["..."]
}
"""


# ---------------------------------------------------------------------------
# Couche 1 — Résolution : passages qui citent la référence
# ---------------------------------------------------------------------------


async def _resolve_passages(
    *,
    session: Session,
    space_id: int,
    user_id: int,
    ref_query: ReferenceQuery,
) -> List[Dict[str, Any]]:
    """Récupère les passages mentionnant la référence.

    Réutilise le retrieval hybride existant (accès, filtre technique, pages) mais en
    posant la référence comme requête : le canal lexical/BM25 fait remonter les pages
    qui contiennent le code exact. On injecte les références dans les signaux pour
    renforcer le boost lexical déjà câblé côté retrieval.
    """
    from app.services.query_signals_schemas import LightweightQuerySignals
    from app.services.space_search_service import search_technical_passages

    query_text = ref_query.raw_message or ref_query.primary
    signals = LightweightQuerySignals(
        intent="documentation",
        detected_references=ref_query.references,
        entity_texts=ref_query.references,
    )

    try:
        retrieval = await search_technical_passages(
            session=session,
            space_id=space_id,
            query_text=query_text,
            user_id=user_id,
            k=settings.FICHE_TECHNIQUE_K,
            signals=signals,
        )
    except Exception as exc:
        logger.exception("[fiche] retrieval échoué: %s", exc)
        return []

    passages = retrieval.get("passages") or []
    logger.info(
        "[fiche] résolution — refs=%s → %d passage(s)",
        ref_query.references,
        len(passages),
    )
    return passages


def _format_passages_for_prompt(passages: List[Dict[str, Any]], *, max_passage_chars: int = 1400) -> str:
    """Sérialise les passages pour le prompt d'extraction (titre + page en tête)."""
    if not passages:
        return "(aucun passage pertinent trouvé)"
    blocks: List[str] = []
    for i, p in enumerate(passages, 1):
        text = str(p.get("passage_raw") or p.get("passage") or "").strip()
        if not text:
            continue
        if len(text) > max_passage_chars:
            text = text[: max_passage_chars - 1] + "…"
        title = p.get("document_title") or "Document sans titre"
        page_no = p.get("page_no") or p.get("page_start")
        page_info = f", page {page_no}" if page_no else ""
        blocks.append(f"[{title}{page_info}]\n{text}")
    return "\n---\n".join(blocks) if blocks else "(aucun passage pertinent trouvé)"


def _allowed_page_keys(passages: List[Dict[str, Any]]) -> set:
    """Couples (titre_normalisé, page) réellement présents dans les passages."""
    keys: set = set()
    for p in passages:
        title = (p.get("document_title") or "").strip().lower()
        page_no = p.get("page_no") or p.get("page_start")
        keys.add((title, page_no))
        keys.add((title, None))  # tolérer une citation document sans page
    return keys


def _sanitize_source(source: Optional[Source], allowed: set) -> Optional[Source]:
    """Écarte une citation qui ne correspond à aucun passage fourni (anti-hallucination)."""
    if source is None:
        return None
    title = (source.document_title or "").strip().lower()
    if not title:
        return None
    if (title, source.page_no) in allowed or (title, None) in allowed:
        return source
    return None


# ---------------------------------------------------------------------------
# Couche 2 — Extraction structurée
# ---------------------------------------------------------------------------


def _loads_json_lenient(content: str) -> Dict[str, Any]:
    """Parse JSON en tolérant les fences Markdown et une sortie légèrement tronquée.

    Les modèles renvoient parfois ```json … ``` ou une réponse coupée par la limite de
    tokens. On tente d'abord un parse strict, puis un parse sur l'objet {…} le plus
    large, en fermant les accolades/crochets manquants en dernier recours.
    """
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        raise ValueError("aucun objet JSON dans la réponse")
    candidate = text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Sortie probablement tronquée : on rééquilibre accolades et crochets.
        repaired = candidate.rstrip().rstrip(",")
        opens = repaired.count("{") - repaired.count("}")
        brackets = repaired.count("[") - repaired.count("]")
        repaired += "]" * max(0, brackets) + "}" * max(0, opens)
        return json.loads(repaired)


async def extract_core(
    *,
    reference: str,
    passages: List[Dict[str, Any]],
) -> FicheCore:
    """Extraction LLM des caractéristiques (grounding strict), puis validation des sources."""
    if not passages:
        return FicheCore(reference=reference)

    passages_block = _format_passages_for_prompt(passages)
    user_prompt = (
        f"RÉFÉRENCE demandée : {reference}\n\n"
        f"PASSAGES :\n{passages_block}\n\n"
        "Extrais les caractéristiques en JSON en respectant le grounding strict."
    )

    try:
        response = await chat(
            "",
            model=settings.MODEL_FAST,
            context=[
                {"role": "system", "content": FICHE_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=settings.FICHE_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = _loads_json_lenient(content)
        core = FicheCore(**data)
    except Exception as exc:
        logger.exception("[fiche] extraction échouée: %s", exc)
        core = FicheCore(reference=reference)

    if not core.reference:
        core.reference = reference

    # Validation des citations : on écarte toute source hallucinée.
    allowed = _allowed_page_keys(passages)
    for item in core.caracteristiques:
        item.source = _sanitize_source(item.source, allowed)

    return core


# ---------------------------------------------------------------------------
# Couche 2bis — Entités qui gravitent autour de la référence (graphe KAG)
# ---------------------------------------------------------------------------

# Ordre d'affichage des groupes d'entités (les types absents sont ignorés).
_ENTITY_TYPE_ORDER = (
    "reference", "product", "dimension", "material", "tool",
    "process", "norm", "organization", "location", "other",
)


def _resolve_seed_entities(
    session: Session,
    space_id: int,
    references: List[str],
) -> Dict[int, KnowledgeEntity]:
    """Entités « graine » : toutes les entités KAG portant la référence (nom ou alias).

    Une même référence (« 6100 ») donne souvent plusieurs entités quasi-doublons
    ("6100", "Profil 6100", "Dormant 6100") : on les prend TOUTES comme graines pour
    agréger leur voisinage.
    """
    seeds: Dict[int, KnowledgeEntity] = {}
    for ref in references:
        norm = normalize_entity_name(ref)
        if not norm:
            continue
        # Garde-fou : le code doit apparaître comme jeton entier (6100 ≠ 61001).
        token_pat = re.compile(rf"(?<!\d){re.escape(norm)}(?!\d)")

        rows = session.exec(
            select(KnowledgeEntity).where(
                KnowledgeEntity.space_id == space_id,
                KnowledgeEntity.name_normalized.like(f"%{norm}%"),
            )
        ).all()
        for e in rows:
            if token_pat.search(e.name_normalized or ""):
                seeds[e.id] = e

        alias_ids = session.exec(
            select(EntityAlias.entity_id).where(
                EntityAlias.space_id == space_id,
                EntityAlias.alias_normalized.like(f"%{norm}%"),
            )
        ).all()
        if alias_ids:
            for e in session.exec(
                select(KnowledgeEntity).where(KnowledgeEntity.id.in_(list(alias_ids)))
            ).all():
                if token_pat.search(e.name_normalized or "") or token_pat.search(e.name or ""):
                    seeds[e.id] = e
    return seeds


def gather_related_entities(
    session: Session,
    space_id: int,
    ref_query: ReferenceQuery,
    *,
    max_total: int = 40,
    max_per_type: int = 8,
) -> List[RelatedEntity]:
    """Constellation d'entités autour de la référence : arêtes du graphe + co-occurrence.

    Le graphe entité↔entité étant clairsemé, la co-occurrence (entités partageant des
    chunks avec les graines) est le signal principal ; les arêtes KAG, quand elles
    existent, fournissent en plus un libellé de relation (« Composé de », « Compatible
    avec »…).
    """
    seeds = _resolve_seed_entities(session, space_id, ref_query.references)
    if not seeds:
        return []
    seed_ids = set(seeds.keys())

    scores: Dict[int, float] = {}
    relations: Dict[int, str] = {}

    # 1. Arêtes explicites du graphe (pondérées plus fort + libellé de relation).
    edges = session.exec(
        select(EntityEntityRelation).where(
            EntityEntityRelation.space_id == space_id,
            (EntityEntityRelation.entity_a_id.in_(seed_ids))
            | (EntityEntityRelation.entity_b_id.in_(seed_ids)),
        )
    ).all()
    for r in edges:
        other = r.entity_b_id if r.entity_a_id in seed_ids else r.entity_a_id
        if other in seed_ids:
            continue
        w = float(r.weight or 1.0) * 3.0
        if w > scores.get(other, 0.0):
            scores[other] = w
        relations[other] = _relation_short_label(r.relation_type)

    # 2. Co-occurrence : entités présentes dans les mêmes chunks que les graines.
    chunk_ids = list({
        c for c in session.exec(
            select(ChunkEntityRelation.chunk_id).where(
                ChunkEntityRelation.space_id == space_id,
                ChunkEntityRelation.entity_id.in_(seed_ids),
            )
        ).all()
    })
    if chunk_ids:
        rows = session.exec(
            select(ChunkEntityRelation.entity_id, func.count())
            .where(
                ChunkEntityRelation.space_id == space_id,
                ChunkEntityRelation.chunk_id.in_(chunk_ids),
            )
            .group_by(ChunkEntityRelation.entity_id)
        ).all()
        for eid, cnt in rows:
            if eid in seed_ids:
                continue
            w = float(cnt or 0)
            if w > scores.get(eid, 0.0):
                scores[eid] = w
            relations.setdefault(eid, "Co-occurrence")

    if not scores:
        return []

    ents = {
        e.id: e
        for e in session.exec(
            select(KnowledgeEntity).where(KnowledgeEntity.id.in_(list(scores.keys())))
        ).all()
    }

    items: List[RelatedEntity] = []
    for eid, w in scores.items():
        e = ents.get(eid)
        if not e:
            continue
        items.append(
            RelatedEntity(
                name=e.name,
                entity_type=e.entity_type or "other",
                type_label=_entity_type_label(e.entity_type),
                description=e.description,
                relation=relations.get(eid, "Co-occurrence"),
                weight=w,
                mention_count=e.mention_count or 0,
            )
        )

    items.sort(key=lambda x: (-x.weight, -x.mention_count, x.name.lower()))

    # Plafond par type pour éviter qu'un type sature la fiche.
    per_type: Dict[str, int] = {}
    capped: List[RelatedEntity] = []
    for it in items:
        c = per_type.get(it.entity_type, 0)
        if c >= max_per_type:
            continue
        per_type[it.entity_type] = c + 1
        capped.append(it)
        if len(capped) >= max_total:
            break

    logger.info(
        "[fiche] entités liées — seeds=%d → %d entité(s) sur %d type(s)",
        len(seed_ids),
        len(capped),
        len({it.entity_type for it in capped}),
    )
    return capped


# ---------------------------------------------------------------------------
# Couche 3 — Rendu déterministe (aucune génération libre ici)
# ---------------------------------------------------------------------------

_NON_PRECISE = "_Non précisé dans les documents indexés._"


def _fmt_source(source: Optional[Source]) -> str:
    if source is None or not source.document_title:
        return ""
    if source.page_no:
        return f" [{source.document_title}, page {source.page_no}]"
    return f" [{source.document_title}]"


def _group_related_by_type(related: List[RelatedEntity]) -> List[tuple]:
    """Regroupe les entités liées par type, dans l'ordre d'affichage préféré.

    Renvoie une liste de (type_label, [entités]) — uniquement les types présents.
    """
    groups: Dict[str, List[RelatedEntity]] = {}
    for it in related:
        groups.setdefault(it.entity_type, []).append(it)

    ordered_types = [t for t in _ENTITY_TYPE_ORDER if t in groups]
    ordered_types += [t for t in groups if t not in _ENTITY_TYPE_ORDER]

    out: List[tuple] = []
    for t in ordered_types:
        items = groups[t]
        out.append((items[0].type_label, items))
    return out


def render_fiche_markdown(core: FicheCore, related: List[RelatedEntity]) -> str:
    """Rendu Markdown déterministe : caractéristiques (LLM) + entités liées (KAG)."""
    lines: List[str] = []

    titre = f"## 🗂️ Fiche technique — {core.reference}"
    if core.designation:
        titre += f" · {core.designation}"
    lines.append(titre)

    if core.type_element:
        lines.append(f"**Type** : {core.type_element}")
    if core.resume:
        lines.append("")
        lines.append(core.resume)
    lines.append("")

    # Caractéristiques (dynamiques, extraites des passages)
    lines.append("### Caractéristiques")
    if core.caracteristiques:
        for c in core.caracteristiques:
            if not (c.libelle or c.valeur):
                continue
            lib = f"**{c.libelle}** : " if c.libelle else ""
            lines.append(f"- {lib}{c.valeur}{_fmt_source(c.source)}")
    else:
        lines.append(_NON_PRECISE)
    lines.append("")

    # Entités liées (constellation KAG, sections dynamiques par type)
    if related:
        lines.append("### Entités liées")
        for type_label, items in _group_related_by_type(related):
            lines.append(f"**{type_label}**")
            for it in items:
                rel = "" if it.relation == "Co-occurrence" else f" — _{it.relation}_"
                mentions = f" ({it.mention_count} mentions)" if it.mention_count > 1 else ""
                lines.append(f"- {it.name}{rel}{mentions}")
            lines.append("")

    # Avertissements
    if core.avertissements:
        lines.append("### ⚠️ À vérifier")
        for w in core.avertissements:
            lines.append(f"- {w}")
        lines.append("")

    # Pied de couverture (honnêteté structurée)
    nb_carac = len([c for c in core.caracteristiques if c.libelle or c.valeur])
    if nb_carac >= 4 and related:
        fiabilite = "élevée"
    elif nb_carac >= 1 or related:
        fiabilite = "moyenne"
    else:
        fiabilite = "faible"
    lines.append(
        f"*Fiabilité : {fiabilite} — {nb_carac} caractéristique(s), "
        f"{len(related)} entité(s) liée(s). "
        "Ce qui ne figure pas ici n'a pas été trouvé dans les documents indexés.*"
    )

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Sources (format identique au pipeline RAG → liens PDF côté frontend)
# ---------------------------------------------------------------------------


def _build_sources_data(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Construit la liste `sources` attendue par le frontend à partir des passages."""
    if not passages:
        return []

    doc_ids = list({p.get("document_id") for p in passages if p.get("document_id")})
    has_file_by_doc: Dict[int, bool] = {}
    if doc_ids:
        with Session(engine) as src_session:
            docs = src_session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
            has_file_by_doc = {d.id: bool(d.source_file_path) for d in docs}

    sources: List[Dict[str, Any]] = []
    for i, p in enumerate(passages):
        raw = p.get("passage_raw") or p.get("passage") or ""
        page_no = p.get("page_no") or p.get("page_start")
        did = p.get("document_id")
        sources.append(
            {
                "index": i + 1,
                "document_id": did,
                "document_title": p.get("document_title") or "Document",
                "chunk_id": p.get("chunk_id"),
                "source_leaf_chunk_id": p.get("source_leaf_chunk_id"),
                "chunk_index": p.get("chunk_index"),
                "excerpt": (raw[:200] + "...") if len(raw or "") > 200 else raw,
                "passage_full": raw,
                "score": round(float(p.get("score", 0.0) or 0.0), 2),
                "page_no": page_no,
                "page_start": p.get("page_start"),
                "page_end": p.get("page_end"),
                "section": p.get("section"),
                "has_source_file": has_file_by_doc.get(did, False),
            }
        )
    return sources


class FicheResult(BaseModel):
    """Sortie prête à streamer par le routeur."""

    markdown: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    core: FicheCore
    related: List[RelatedEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------


async def build_fiche_technique(
    *,
    session: Session,
    space_id: int,
    user_id: int,
    ref_query: ReferenceQuery,
) -> Optional[FicheResult]:
    """Pipeline complet : caractéristiques (passages) + entités liées (graphe KAG).

    Renvoie None seulement si RIEN n'est trouvé (ni passage ni entité graine) : le
    routeur retombe alors sur le pipeline RAG normal (message « rien trouvé »).
    """
    passages = await _resolve_passages(
        session=session,
        space_id=space_id,
        user_id=user_id,
        ref_query=ref_query,
    )

    related = gather_related_entities(session, space_id, ref_query)

    # Abstention uniquement si aucune source ET aucune entité graine.
    if not passages and not related:
        return None

    core = await extract_core(reference=ref_query.primary, passages=passages)
    markdown = render_fiche_markdown(core, related)
    sources = _build_sources_data(passages)

    return FicheResult(markdown=markdown, sources=sources, core=core, related=related)
