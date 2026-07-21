"""
Extraction d'entités, relations et catégories KAG via le modèle vision.

Seconde passe par batch de pages (fenêtre glissante) après persistance des chunks L1 :
  1. PNG des pages du batch + texte L1 concaténé
  2. Appel Ministral vision → JSON { pages: [{ entities, relations, categories }] }
  3. Normalisation + upsert entités au niveau espace
  4. Persistance chunkentityrelation + entityentityrelation + entityalias + chunkcategoryrelation
"""

from __future__ import annotations

import base64
import json
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.chunk_category_relation import ChunkCategoryRelation
from app.models.knowledge_entity import (
    ChunkEntityRelation,
    EntityAlias,
    EntityEntityRelation,
    KnowledgeEntity,
)

logger = logging.getLogger(__name__)

KAG_EXTRACTION_VERSION = "kag_vision_v4"
TAXONOMY_VERSION = "facets_v1"

_VALID_ENTITY_TYPES = frozenset(
    {
        "product",
        "material",
        "tool",
        "norm",
        "dimension",
        "process",
        "organization",
        "location",
        "reference",
        "symptom",
        "other",
    }
)

_VALID_RELATION_ROLES = frozenset({"mention", "subject", "object"})


# ---------------------------------------------------------------------------
# Schémas Pydantic réponse LLM
# ---------------------------------------------------------------------------


class KagExtractedEntity(BaseModel):
    name: str
    type: str = Field(default="other")
    aliases: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    # R1 — identité par code : le LLM voit la page et sait si « 7016 » est un RAL ou une réf.
    code: Optional[str] = None
    code_kind: Optional[str] = None  # ref_produit | couleur_ral | norme | aucun
    # R3 — attribution par chunk : indices des chunks (chunk_index de la page) qui PARLENT
    # de cette entité (elle en est le sujet), pas ceux qui la mentionnent en passant.
    chunk_indexes: List[int] = Field(default_factory=list)

    @field_validator("chunk_indexes", mode="before")
    @classmethod
    def _coerce_chunk_indexes(cls, value):
        """Tolère null / int isolé / liste ; ignore les valeurs non entières."""
        if value is None:
            return []
        if isinstance(value, int):
            return [value]
        coerced: List[int] = []
        for item in value if isinstance(value, list) else []:
            try:
                coerced.append(int(item))
            except (TypeError, ValueError):
                continue
        return coerced


class KagExtractedRelation(BaseModel):
    entity_a: str
    relation: str
    entity_b: str
    relation_label: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ScoredCategory(BaseModel):
    """Catégorie de chunk notée par le LLM (axe task/symptom)."""

    slug: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    primary: bool = False


class ChunkCategoryItem(BaseModel):
    chunk_index: int = Field(ge=0)
    categories: List[ScoredCategory] = Field(default_factory=list)

    @field_validator("categories", mode="before")
    @classmethod
    def _coerce_categories(cls, value):
        """Tolère l'ancien format (liste de slugs nus) et le nouveau (objets notés)."""
        coerced = []
        for item in value or []:
            if isinstance(item, str):
                coerced.append({"slug": item})
            else:
                coerced.append(item)
        return coerced


class KagPageResponse(BaseModel):
    page_no: int
    entities: List[KagExtractedEntity] = Field(default_factory=list)
    relations: List[KagExtractedRelation] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    chunk_categories: List[ChunkCategoryItem] = Field(default_factory=list)
    # Facettes page-level (homogènes au document) : appliquées à tous les chunks de la page
    doc_types: List[str] = Field(default_factory=list)
    lifecycle_phases: List[str] = Field(default_factory=list)
    # Symptômes proposés HORS liste connue (vocabulaire extensible) → CategoryCandidate
    symptom_candidates: List[str] = Field(default_factory=list)


class BatchKagResponse(BaseModel):
    pages: List[KagPageResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt vision KAG
# ---------------------------------------------------------------------------

_KAG_SYSTEM_PROMPT = """Tu es un expert en extraction d'entités nommées (NER) et de relations (RE) pour documents techniques.
On te donne l'image d'une page ET le texte déjà extrait de cette page (chunks sémantiques).
Ton travail : identifier les entités normalisées et les relations sémantiques pour alimenter un graphe de connaissances.

Règles impératives :
1. Renvoie UNIQUEMENT un objet JSON valide (aucun texte hors JSON).
2. Extrais les entités concrètes : produits, références, matériaux, outils, normes, dimensions, processus, organisations, lieux.
3. Normalise les noms (casse cohérente, sans bruit markdown).
4. Pour chaque entité, fournis un type parmi : product | material | tool | norm | dimension | process | organization | location | reference | symptom | other. Un symptôme ou problème SAV (ex. « infiltration d'eau », « ouvrant qui force », « condensation ») est une entité de type symptom.
5. Les aliases sont les variantes, abréviations ou codes produit (ex. "ref ABC-123").
6. Les relations décrivent un lien sémantique explicite entre deux entités de la page.
7. Types de relation suggérés : compatible_avec | est_compose_de | remplace | utilise | conforme_a | installe_sur | fabrique_par | mesure | reference | co_occurs
8. Ne pas inventer d'entités absentes du texte ou de l'image.
9. Maximum {max_entities} entités et {max_relations} relations par page.

Format de réponse OBLIGATOIRE :
{{
  "page_no": <numéro>,
  "entities": [
    {{
      "name": "<nom canonique>",
      "type": "<type>",
      "aliases": ["<alias1>"],
      "description": "<contexte court>",
      "confidence": 0.9
    }}
  ],
  "relations": [
    {{
      "entity_a": "<nom entité A>",
      "relation": "<type_relation>",
      "entity_b": "<nom entité B>",
      "relation_label": "<phrase naturelle optionnelle>",
      "confidence": 0.85
    }}
  ]
}}"""

_KAG_USER_PROMPT_TEMPLATE = (
    "Document : {title}\nPage : {page_no}\n\n"
    "Texte extrait de la page (chunks sémantiques) :\n{chunk_text}\n\n"
    "Extrait les entités et relations de cette page selon les règles du système."
)

_KAG_COMPACT_RETRY_SUFFIX = (
    "\n\nIMPORTANT — JSON compact obligatoire : maximum {max_entities} entités et "
    "{max_relations} relations par page. Omet description et relation_label. "
    "aliases : 0 ou 1 par entité. Noms courts. JSON complet et valide."
)

_KAG_BATCH_SYSTEM_PROMPT = """Tu es un expert en extraction d'entités nommées (NER), de relations (RE) et de catégorisation multi-axes pour documents techniques menuiserie (pose, SAV, notices).
On te donne les images de plusieurs pages consécutives ET le texte déjà extrait (chunks transcrits par page, identifiés par chunk_index).
Ton travail : pour CHAQUE page, identifier les entités, les relations, et classer le contenu selon une taxonomie À FACETTES (plusieurs axes orthogonaux).

Règles impératives — entités & relations :
1. Renvoie UNIQUEMENT un objet JSON valide (aucun texte hors JSON).
2. Extrais les entités concrètes : produits, références, matériaux, outils, normes, dimensions, processus, organisations, lieux.
3. Normalise les noms (casse cohérente, sans bruit markdown).
4. Pour chaque entité, fournis un type parmi : product | material | tool | norm | dimension | process | organization | location | reference | symptom | other. Un symptôme ou problème SAV (ex. « infiltration d'eau », « ouvrant qui force », « condensation ») est une entité de type symptom.
5. Les aliases sont les variantes, abréviations ou codes produit (ex. "ref ABC-123").
6. CODE D'IDENTITÉ (crucial) : si l'entité porte une référence codée (profil « 6111 », « SL1600 »…), une couleur RAL (« 7016 ») ou un code de norme, renseigne "code" (le code nu, ex. "6111") ET "code_kind" :
   - "ref_produit" : référence de profil, quincaillerie, pièce (« 6111 », « SL1600 »).
   - "couleur_ral" : teinte RAL. ATTENTION « 7016 », « 9016 », « 9005 » sont des RAL (couleurs), PAS des profils. Regarde le contexte (nuancier, finition, teinte → RAL).
   - "norme" : code de norme/DTU/PV (« 6/16-2335 »).
   - "aucun" : l'entité n'a pas de code d'identité. Mets "code": null.
   Le MÊME produit sous plusieurs formes (« Profil 6111 », « Dormant 6111 », « 6111 ») doit porter le MÊME "code":"6111".
7. ATTRIBUTION PAR CHUNK : "chunk_indexes" = la liste des chunk_index (de la page) dont le contenu PARLE de cette entité (la décrit, donne ses cotes, ses caractéristiques). PAS les chunks qui la citent juste en passant. Laisse [] si aucun chunk n'en est le sujet.
8. Les relations décrivent un lien sémantique EXPLICITE dans le texte/l'image entre deux entités d'une même page. Ne crée PAS de relation de simple co-occurrence : s'il n'y a pas de lien explicite, n'émets rien.
9. Types de relation autorisés (n'invente rien hors liste) :
   - produit : compatible_avec | est_compose_de | remplace | utilise | conforme_a | installe_sur | fabrique_par | mesure | reference
   - SAV / procédure : symptome_cause | cause_resolution | etape_precede | necessite_outil | requiert_piece
   Privilégie les relations SAV/procédure pour un dépannage ou une séquence de pose. Si la page est un TABLEAU DE COMPATIBILITÉ (matrice de codes), extrais un maximum de paires "compatible_avec".
10. Ne pas inventer d'entités ou de relations absentes du texte ou de l'image.
11. Maximum {max_entities} entités et {max_relations} relations par page.

Règles impératives — catégorisation à facettes :
12. Axe "task" et axe "symptom" → PAR CHUNK (champ chunk_categories). Pour chaque chunk_index, renvoie une liste d'objets {{slug, confidence, primary}} en choisissant UNIQUEMENT parmi les slugs des axes `task` et `symptom` fournis.
    - confidence ∈ [0,1] : à quel point le chunk traite EXPLICITEMENT ce thème (0.9 = sujet central, 0.6 = thème secondaire net, < 0.5 = ne pas inclure).
    - primary : true pour LE thème dominant du chunk. Exactement UN primary=true par chunk (ou zéro si le chunk n'a aucune catégorie).
    - N'inclus un slug que si confidence ≥ 0.5. Mieux vaut 1 catégorie juste que 3 douteuses. Un chunk peut avoir 0 catégorie.
13. N'associe un slug qu'aux chunks dont le contenu traite EXPLICITEMENT du thème. Ne propage pas un slug task/symptom à tous les chunks.
14. Axe "doc_type" et axe "lifecycle_phase" → AU NIVEAU PAGE (champs doc_types, lifecycle_phases). Ces facettes sont homogènes : décris la NATURE du document et la PHASE du cycle de vie (en général 1 valeur chacun). Choisis uniquement parmi les slugs fournis.
15. N'invente JAMAIS de slug hors des listes pour task / doc_type / lifecycle_phase / symptom. Si un symptôme n'existe pas dans la liste `symptom`, ne le propose PAS : le vocabulaire des catégories est fermé.

Taxonomie autorisée, groupée par axe (axe : [{{slug, description}}]) :
{category_list}

Format de réponse OBLIGATOIRE :
{{
  "pages": [
    {{
      "page_no": <numéro>,
      "entities": [
        {{ "name": "<nom canonique>", "type": "<type>", "code": "<code ou null>", "code_kind": "ref_produit|couleur_ral|norme|aucun", "chunk_indexes": [0, 2], "aliases": ["<alias1>"], "description": "<contexte court>", "confidence": 0.9 }}
      ],
      "relations": [
        {{ "entity_a": "<A>", "relation": "<type_relation>", "entity_b": "<B>", "relation_label": "<optionnel>", "confidence": 0.85 }}
      ],
      "doc_types": ["<slug doc_type>"],
      "lifecycle_phases": ["<slug lifecycle_phase>"],
      "chunk_categories": [
        {{ "chunk_index": 0, "categories": [
            {{ "slug": "mounting", "confidence": 0.9, "primary": true }},
            {{ "slug": "hardware_adjustment", "confidence": 0.6, "primary": false }}
        ] }},
        {{ "chunk_index": 2, "categories": [ {{ "slug": "infiltration_eau", "confidence": 0.85, "primary": true }} ] }}
      ]
    }}
  ]
}}"""

_KAG_BATCH_USER_PROMPT_TEMPLATE = (
    "Document : {title}\nPages du batch : {page_range}\n\n"
    "Texte extrait par page (chunks numérotés chunk_index=0, 1, 2…) :\n{chunk_text}\n\n"
    "Extrait entités, relations et chunk_categories pour chaque page selon les règles du système."
)


# ---------------------------------------------------------------------------
# Utilitaires normalisation
# ---------------------------------------------------------------------------


ENTITY_NORMALIZATION_RULES: Dict[str, str] = {
    "alu": "aluminium",
    "pvc": "PVC",
    "bois": "bois",
    "perform": "Gamme Perform",
    "lumine": "Gamme Lumine",
    "hybride": "Gamme Hybride",
    "textural": "Gamme Textural",
    "technal": "Technal",
    "profine": "Profine",
    "kommerling": "Kömmerling",
    "proferm": "Proferm",
}


def normalize_and_expand_entity(name: str) -> Tuple[str, List[str]]:
    """
    Normalise le nom d'entité et génère les alias automatiques.

    Returns:
        (canonical_name, aliases)
    """
    stripped = (name or "").strip()
    if not stripped:
        return "", []

    lower_name = stripped.lower()
    canonical = ENTITY_NORMALIZATION_RULES.get(lower_name, stripped)

    aliases: List[str] = []
    if canonical.lower() != lower_name:
        aliases.append(stripped)
        aliases.append(lower_name)
    if canonical != stripped:
        aliases.append(canonical)
        if canonical != canonical.title():
            aliases.append(canonical.title())

    return canonical, list(dict.fromkeys(a for a in aliases if a and a != canonical))


def normalize_entity_name(name: str) -> str:
    """Normalise un nom d'entité pour déduplication (lowercase, NFKC, espaces)."""
    if not name:
        return ""
    canonical, _ = normalize_and_expand_entity(name)
    target = canonical or name
    normalized = unicodedata.normalize("NFKC", target.strip())
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_entity_type(raw: str) -> str:
    value = (raw or "other").strip().lower().replace(" ", "_")
    if value in _VALID_ENTITY_TYPES:
        return value
    return "other"


# ---------------------------------------------------------------------------
# R1 — Identité par code (référence produit, RAL, norme)
# ---------------------------------------------------------------------------

# Motifs de code produit : alphanum (SL1600, BC01), numérique pur (6111, 155),
# alterné (6A20). Sert de FALLBACK quand le LLM n'a pas déclaré de code.
REF_CODE_RE = re.compile(r"[A-Za-z]{1,4}\d{2,6}[A-Za-z]?|\d[A-Z]\d{2,4}|\d{3,6}[A-Za-z]?")

# Codes RAL usuels (menuiserie) — aide la détection quand le LLM ne qualifie pas code_kind.
_COMMON_RAL = frozenset({
    "1013", "1015", "3004", "5011", "6005", "6009", "7016", "7015", "7021", "7022",
    "7024", "7035", "7038", "7039", "7040", "8014", "8017", "8019", "8022", "9001",
    "9005", "9006", "9007", "9010", "9016",
})


def _normalize_code_token(raw: str) -> Optional[str]:
    """Nettoie un code brut → forme canonique MAJ, ou None si ce n'est pas un code."""
    if not raw:
        return None
    token = re.sub(r"\s+", "", str(raw).strip()).upper()
    # écarte les millésimes nus (1990–2035) et les nombres trop courts
    if token.isdigit():
        if len(token) < 3:
            return None
        if len(token) == 4 and 1990 <= int(token) <= 2035:
            return None
    if not re.search(r"\d", token):  # un code contient au moins un chiffre
        return None
    if len(token) > 32:
        return None
    return token


def extract_ref_code(name: str) -> Optional[str]:
    """Extrait le code de référence dominant (le plus long) d'un nom d'entité, en MAJ.

    Fallback regex quand le LLM n'a pas déclaré de code explicite."""
    best: Optional[str] = None
    for m in REF_CODE_RE.finditer(name or ""):
        tok = m.group(0)
        if tok.isdigit() and len(tok) == 4 and 1990 <= int(tok) <= 2035:
            continue
        if best is None or len(tok) > len(best):
            best = tok
    return best.upper() if best else None


def _canonical_code_name(ref_code: str) -> str:
    """Nom d'affichage canonique d'une entité à code (« RAL:7016 » → « RAL 7016 »)."""
    if ref_code.startswith("RAL:"):
        return f"RAL {ref_code[4:]}"
    return ref_code


def _resolve_ref_code(extracted: "KagExtractedEntity", raw_name: str, entity_type: str) -> Optional[str]:
    """Détermine le code d'identité d'une entité : LLM déclaré d'abord, regex en secours.

    Retourne « 6111 », « RAL:7016 », ou None (entité sans code d'identité)."""
    declared = _normalize_code_token(extracted.code or "")
    kind = (extracted.code_kind or "").strip().lower()

    if declared:
        if kind == "aucun":
            return None
        if kind == "couleur_ral":
            return f"RAL:{declared}"
        # ref_produit | norme | (non qualifié) → code nu
        return declared

    # Fallback regex : uniquement pour les types susceptibles de porter une référence.
    if entity_type in ("product", "reference", "norm", "material"):
        fallback = extract_ref_code(raw_name)
        if fallback:
            # RAL probable non déclaré → identité couleur disjointe des réfs produit.
            if fallback in _COMMON_RAL and entity_type != "reference":
                return f"RAL:{fallback}"
            return fallback
    return None


def _normalize_relation_type(raw: str) -> str:
    value = (raw or "co_occurs").strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value or "co_occurs"


def _slugify_candidate(text: str) -> str:
    """Slugifie un libellé libre (symptôme candidat) : minuscules, sans accent, _."""
    value = unicodedata.normalize("NFKD", (text or "").strip().lower())
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:64]


def _kag_extraction_model() -> str:
    return settings.KAG_EXTRACTION_MODEL or settings.PAGE_EXTRACTION_MODEL


def _is_compact_extraction_model(model: Optional[str] = None) -> bool:
    """Détecte les petits modèles (ex. ministral-3b) qui tronquent souvent le JSON."""
    name = (model or _kag_extraction_model() or "").lower()
    return any(marker in name for marker in ("3b", "ministral-3", "ministral_3"))


def _effective_kag_limits() -> Tuple[int, int]:
    max_entities = settings.KAG_MAX_ENTITIES_PER_PAGE
    max_relations = settings.KAG_MAX_RELATIONS_PER_PAGE
    if _is_compact_extraction_model():
        max_entities = min(max_entities, settings.KAG_SMALL_MODEL_MAX_ENTITIES)
        max_relations = min(max_relations, settings.KAG_SMALL_MODEL_MAX_RELATIONS)
    return max_entities, max_relations


def _build_kag_system_prompt(*, max_entities: Optional[int] = None, max_relations: Optional[int] = None) -> str:
    ent, rel = _effective_kag_limits()
    return _KAG_SYSTEM_PROMPT.format(
        max_entities=max_entities if max_entities is not None else ent,
        max_relations=max_relations if max_relations is not None else rel,
    )


def _build_kag_batch_system_prompt(category_list: str) -> str:
    ent, rel = _effective_kag_limits()
    return _KAG_BATCH_SYSTEM_PROMPT.format(
        max_entities=ent,
        max_relations=rel,
        category_list=category_list,
    )


def build_kag_batches(
    page_numbers: List[int],
    *,
    batch_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[List[int]]:
    """Construit des batches de pages avec fenêtre glissante."""
    if not page_numbers:
        return []

    size = batch_size if batch_size is not None else settings.KAG_BATCH_SIZE
    overlap_val = overlap if overlap is not None else settings.KAG_BATCH_OVERLAP
    size = max(1, size)
    overlap_val = max(0, min(overlap_val, size - 1))
    stride = max(1, size - overlap_val)

    sorted_pages = sorted(set(page_numbers))
    batches: List[List[int]] = []
    i = 0
    while i < len(sorted_pages):
        batch = sorted_pages[i : i + size]
        if batch:
            batches.append(batch)
        if i + size >= len(sorted_pages):
            break
        i += stride
    return batches


def _normalize_category_slugs(
    raw_categories: List[str],
    valid_slugs: frozenset[str],
) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for raw in raw_categories or []:
        slug = (raw or "").strip().lower().replace(" ", "_")
        if slug and slug in valid_slugs and slug not in seen:
            seen.add(slug)
            result.append(slug)
    return result


def _format_chunks_for_kag_prompt(chunks_by_page: Dict[int, List[DocumentChunk]]) -> str:
    """Formate les chunks L1 avec chunk_index local par page pour le prompt KAG."""
    parts: List[str] = []
    for pno in sorted(chunks_by_page.keys()):
        page_chunks = chunks_by_page[pno]
        parts.append(f"--- PAGE {pno} ---")
        for idx, chunk in enumerate(page_chunks):
            meta = chunk.metadata_json or {}
            heading = meta.get("heading") or meta.get("parent_heading") or "null"
            section_type = meta.get("section_type") or "section"
            content = (chunk.content or chunk.text or "").strip()
            if not content:
                continue
            parts.append(
                f"[chunk_index={idx}] heading={heading} section_type={section_type}\n{content}"
            )
    return "\n\n".join(parts)


def _dedup_scored_categories(
    items: List[ScoredCategory],
    valid_slugs: frozenset[str],
) -> List[ScoredCategory]:
    """Valide les slugs et déduplique par slug (confiance max, primary OR)."""
    by_slug: Dict[str, ScoredCategory] = {}
    for sc in items or []:
        slug = (getattr(sc, "slug", "") or "").strip().lower().replace(" ", "_")
        if not slug or slug not in valid_slugs:
            continue
        conf = max(0.0, min(1.0, float(getattr(sc, "confidence", 0.7) or 0.0)))
        primary = bool(getattr(sc, "primary", False))
        prev = by_slug.get(slug)
        if prev is None:
            by_slug[slug] = ScoredCategory(slug=slug, confidence=conf, primary=primary)
        else:
            prev.confidence = max(prev.confidence, conf)
            prev.primary = prev.primary or primary
    return list(by_slug.values())


def _apply_threshold_topk(
    scored: List[ScoredCategory],
    min_conf: float,
    max_per_chunk: int,
) -> List[ScoredCategory]:
    """Filtre par seuil de confiance, plafonne au top-K, garantit un primary unique.

    Pas de catégorie forcée : un chunk sans slug au-dessus du seuil reste sans tag.
    """
    above = sorted(
        (s for s in scored if s.confidence >= min_conf),
        key=lambda s: s.confidence,
        reverse=True,
    )
    if max_per_chunk > 0:
        above = above[:max_per_chunk]
    if above:
        marked = [s for s in above if s.primary]
        chosen = max(marked, key=lambda s: s.confidence) if marked else above[0]
        for s in above:
            s.primary = s is chosen
    return above


def _normalize_chunk_categories(
    chunk_categories: List[ChunkCategoryItem],
    valid_slugs: frozenset[str],
) -> List[ChunkCategoryItem]:
    normalized: List[ChunkCategoryItem] = []
    for item in chunk_categories or []:
        scored = _dedup_scored_categories(item.categories, valid_slugs)
        if scored:
            normalized.append(ChunkCategoryItem(chunk_index=item.chunk_index, categories=scored))
    return normalized


# ---------------------------------------------------------------------------
# Appel API vision KAG
# ---------------------------------------------------------------------------


def _call_kag_vision_api(
    image_b64: str,
    page_no: int,
    document_title: str,
    chunk_text: str,
    *,
    compact_retry: bool = False,
    max_entities: Optional[int] = None,
    max_relations: Optional[int] = None,
) -> dict:
    from app.services.multimodal_page_service import (
        _mistral_chat_completion,
        _parse_json_with_repair,
    )

    ent_limit, rel_limit = _effective_kag_limits()
    if max_entities is not None:
        ent_limit = max_entities
    if max_relations is not None:
        rel_limit = max_relations

    user_text = _KAG_USER_PROMPT_TEMPLATE.format(
        title=document_title or "Document",
        page_no=page_no,
        chunk_text=chunk_text[:12000],
    )
    if compact_retry:
        user_text += _KAG_COMPACT_RETRY_SUFFIX.format(
            max_entities=ent_limit,
            max_relations=rel_limit,
        )
    elif _is_compact_extraction_model():
        user_text += (
            f"\n\nJSON compact : max {ent_limit} entités, max {rel_limit} relations. "
            "Omet description si inutile."
        )

    messages = [
        {"role": "system", "content": _build_kag_system_prompt(max_entities=ent_limit, max_relations=rel_limit)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        },
    ]

    raw = _mistral_chat_completion(
        messages,
        page_no=page_no,
        max_tokens=settings.KAG_EXTRACTION_MAX_TOKENS,
        temperature=0.0,
        response_format_json=True,
        timeout_seconds=settings.KAG_EXTRACTION_TIMEOUT,
        model=_kag_extraction_model(),
    )
    return _parse_json_with_repair(raw)


def _call_kag_batch_vision_api(
    images_b64: List[str],
    batch_pages: List[int],
    document_title: str,
    chunk_text: str,
    category_list: str,
    *,
    compact_retry: bool = False,
) -> dict:
    from app.services.multimodal_page_service import (
        _mistral_chat_completion,
        _parse_json_with_repair,
    )

    ent_limit, rel_limit = _effective_kag_limits()
    page_range = f"{batch_pages[0]}-{batch_pages[-1]}" if len(batch_pages) > 1 else str(batch_pages[0])
    user_text = _KAG_BATCH_USER_PROMPT_TEMPLATE.format(
        title=document_title or "Document",
        page_range=page_range,
        chunk_text=chunk_text[:18000],
    )
    if compact_retry:
        user_text += _KAG_COMPACT_RETRY_SUFFIX.format(
            max_entities=ent_limit,
            max_relations=rel_limit,
        )
    elif _is_compact_extraction_model():
        user_text += (
            f"\n\nJSON compact : max {ent_limit} entités et max {rel_limit} relations par page. "
            "Omet description si inutile."
        )

    content: List[dict] = [{"type": "text", "text": user_text}]
    for image_b64 in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

    messages = [
        {"role": "system", "content": _build_kag_batch_system_prompt(category_list)},
        {"role": "user", "content": content},
    ]

    raw = _mistral_chat_completion(
        messages,
        page_no=batch_pages[0],
        max_tokens=settings.KAG_EXTRACTION_MAX_TOKENS,
        temperature=0.0,
        response_format_json=True,
        timeout_seconds=settings.KAG_EXTRACTION_TIMEOUT,
        model=_kag_extraction_model(),
    )
    return _parse_json_with_repair(raw)


def _clean_freeform_list(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in values or []:
        v = (raw or "").strip()
        key = v.lower()
        if v and key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _coerce_kag_page_response(
    raw: dict,
    page_no: int,
    *,
    valid_category_slugs: Optional[frozenset[str]] = None,
    allowed_by_axis: Optional[Dict[str, frozenset]] = None,
) -> KagPageResponse:
    """Valide et normalise une réponse KAG multi-axes (tolère page_no absent)."""
    payload = dict(raw or {})
    payload.setdefault("page_no", page_no)
    for list_field in ("entities", "relations", "categories", "chunk_categories",
                       "doc_types", "lifecycle_phases", "symptom_candidates"):
        if not isinstance(payload.get(list_field), list):
            payload[list_field] = []

    response = KagPageResponse.model_validate(payload)
    max_ent, max_rel = _effective_kag_limits()
    response.entities = response.entities[:max_ent]
    response.relations = response.relations[:max_rel]

    if valid_category_slugs is not None:
        # chunk_categories + legacy categories = axes task ∪ symptom (par chunk)
        response.categories = _normalize_category_slugs(response.categories, valid_category_slugs)
        response.chunk_categories = _normalize_chunk_categories(
            response.chunk_categories,
            valid_category_slugs,
        )
    if allowed_by_axis is not None:
        response.doc_types = _normalize_category_slugs(
            response.doc_types, allowed_by_axis.get("doc_type", frozenset())
        )
        response.lifecycle_phases = _normalize_category_slugs(
            response.lifecycle_phases, allowed_by_axis.get("lifecycle_phase", frozenset())
        )
        # Candidats : on retire ceux qui correspondent déjà à un symptôme officiel.
        known = allowed_by_axis.get("symptom", frozenset())
        response.symptom_candidates = [
            c for c in _clean_freeform_list(response.symptom_candidates)
            if _slugify_candidate(c) not in known
        ]

    has_content = any(
        (
            response.entities,
            response.relations,
            response.categories,
            response.chunk_categories,
            response.doc_types,
            response.lifecycle_phases,
            response.symptom_candidates,
        )
    )
    if not has_content:
        raise ValueError(f"Aucune entité, relation ni catégorie extraite pour la page {page_no}")
    return response


def _coerce_batch_kag_response(
    raw: dict,
    batch_pages: List[int],
    *,
    valid_category_slugs: frozenset[str],
    allowed_by_axis: Optional[Dict[str, frozenset]] = None,
) -> BatchKagResponse:
    payload = dict(raw or {})
    pages_in = payload.get("pages")
    if not isinstance(pages_in, list):
        raise ValueError("Réponse batch KAG invalide : champ 'pages' absent")

    pages: List[KagPageResponse] = []
    for item in pages_in:
        if not isinstance(item, dict):
            continue
        page_no = int(item.get("page_no") or 0)
        if page_no not in batch_pages:
            continue
        pages.append(
            _coerce_kag_page_response(
                item,
                page_no,
                valid_category_slugs=valid_category_slugs,
                allowed_by_axis=allowed_by_axis,
            )
        )

    if not pages:
        raise ValueError(f"Aucune page valide dans le batch {batch_pages}")
    return BatchKagResponse(pages=pages)


def extract_page_kag_response(
    pdf_path: str,
    page_no: int,
    document_title: str,
    chunk_texts: List[str],
) -> Optional[KagPageResponse]:
    """Extrait entités/relations d'une page via vision + contexte texte."""
    from app.services.multimodal_page_service import render_page_png_cached

    if not chunk_texts:
        return None

    chunk_text = "\n\n---\n\n".join(t.strip() for t in chunk_texts if t and t.strip())
    if not chunk_text:
        return None

    try:
        png = render_page_png_cached(pdf_path, page_no, dpi=settings.PAGE_EXTRACTION_DPI)
        image_b64 = base64.b64encode(png).decode("ascii")
    except Exception as exc:
        logger.warning("[KAG] Rendu PNG page %s échoué : %s", page_no, exc)
        return None

    try:
        last_exc: Optional[Exception] = None
        for attempt, compact in enumerate((False, True)):
            try:
                raw = _call_kag_vision_api(
                    image_b64,
                    page_no,
                    document_title,
                    chunk_text,
                    compact_retry=compact,
                )
                return _coerce_kag_page_response(raw, page_no)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning(
                        "[KAG] Validation page %s échouée (tentative 1) : %s — retry compact",
                        page_no,
                        exc,
                    )
                    continue
                logger.warning("[KAG] Validation page %s échouée : %s", page_no, exc)
                return None
        if last_exc is not None:
            logger.warning("[KAG] Validation page %s échouée : %s", page_no, last_exc)
        return None
    except Exception as exc:
        logger.warning("[KAG] Extraction page %s échouée : %s", page_no, exc)
        return None


def extract_batch_kag_response(
    pdf_path: str,
    batch_pages: List[int],
    document_title: str,
    chunks_by_page: Dict[int, List[DocumentChunk]],
    category_list: str,
    valid_category_slugs: frozenset[str],
    allowed_by_axis: Optional[Dict[str, frozenset]] = None,
) -> Optional[BatchKagResponse]:
    """Extrait entités, relations et catégories pour un batch de pages via vision."""
    from app.services.multimodal_page_service import render_page_png_cached

    if not batch_pages:
        return None

    batch_chunks = {pno: chunks_by_page.get(pno) or [] for pno in batch_pages}
    combined_text = _format_chunks_for_kag_prompt(batch_chunks)
    if not combined_text.strip():
        return None

    images_b64: List[str] = []
    for pno in batch_pages:
        try:
            png = render_page_png_cached(pdf_path, pno, dpi=settings.PAGE_EXTRACTION_DPI)
            images_b64.append(base64.b64encode(png).decode("ascii"))
        except Exception as exc:
            logger.warning("[KAG] Rendu PNG page %s échoué : %s", pno, exc)
            return None

    if not images_b64:
        return None
    try:
        last_exc: Optional[Exception] = None
        for attempt, compact in enumerate((False, True)):
            try:
                raw = _call_kag_batch_vision_api(
                    images_b64,
                    batch_pages,
                    document_title,
                    combined_text,
                    category_list,
                    compact_retry=compact,
                )
                return _coerce_batch_kag_response(
                    raw,
                    batch_pages,
                    valid_category_slugs=valid_category_slugs,
                    allowed_by_axis=allowed_by_axis,
                )
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt == 0:
                    logger.warning(
                        "[KAG] Validation batch %s échouée (tentative 1) : %s — retry compact",
                        batch_pages,
                        exc,
                    )
                    continue
                logger.warning("[KAG] Validation batch %s échouée : %s", batch_pages, exc)
                return None
        if last_exc is not None:
            logger.warning("[KAG] Validation batch %s échouée : %s", batch_pages, last_exc)
        return None
    except Exception as exc:
        logger.warning("[KAG] Extraction batch %s échouée : %s", batch_pages, exc)
        return None


# ---------------------------------------------------------------------------
# Persistance graphe
# ---------------------------------------------------------------------------


def _get_document_space_ids(session: Session, document_id: int) -> List[int]:
    stmt = select(DocumentSpace.space_id).where(DocumentSpace.document_id == document_id)
    return list(session.exec(stmt).all())


def _load_l1_chunks_by_page(session: Session, document_id: int) -> Dict[int, List[DocumentChunk]]:
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == True,  # noqa: E712
    )
    chunks = list(session.exec(stmt).all())
    by_page: Dict[int, List[DocumentChunk]] = {}
    for chunk in chunks:
        meta = chunk.metadata_json or {}
        # Accepte les chunks-feuilles L1 : vision (« semantic_leaf ») ET texte enrichi
        # page-level (« page_raw_enriched »). Les ~17 docs à 0 couverture KAG passaient par
        # le pipeline page_raw_enriched et étaient ignorés ici — c'est LA cause racine.
        # On exclut « contextual_enrichment » (L2 dérivé) pour ne pas dupliquer les pages.
        if meta.get("content_type") not in (None, "semantic_leaf", "page_raw_enriched"):
            continue
        page_no = meta.get("page_no") or meta.get("page_start")
        if page_no is None:
            continue
        by_page.setdefault(int(page_no), []).append(chunk)
    for pno in by_page:
        by_page[pno].sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    return by_page


def _first_entity(
    session: Session,
    space_id: int,
    *,
    ref_code: Optional[str] = None,
    name_normalized: Optional[str] = None,
) -> Optional[KnowledgeEntity]:
    """Lookup d'entité (par code OU par nom normalisé), le plus mentionné d'abord.

    Le tri par mention_count rend la résolution déterministe malgré d'éventuels
    doublons transitoires (état pré-retraitement)."""
    conditions = [KnowledgeEntity.space_id == space_id]
    if ref_code is not None:
        conditions.append(KnowledgeEntity.ref_code == ref_code)
    if name_normalized is not None:
        conditions.append(KnowledgeEntity.name_normalized == name_normalized)
    stmt = (
        select(KnowledgeEntity)
        .where(*conditions)
        .order_by(KnowledgeEntity.mention_count.desc(), KnowledgeEntity.id.asc())  # type: ignore[attr-defined]
    )
    return session.exec(stmt).first()


def _upsert_entity(
    session: Session,
    space_id: int,
    extracted: KagExtractedEntity,
) -> KnowledgeEntity:
    raw_name = (extracted.name or "").strip()
    canonical_name, auto_aliases = normalize_and_expand_entity(raw_name)
    name = canonical_name or raw_name
    entity_type = _normalize_entity_type(extracted.type)

    # R1 — identité par code : un produit = un nœud, indépendant du type et de la forme.
    ref_code = _resolve_ref_code(extracted, raw_name, entity_type)

    entity: Optional[KnowledgeEntity] = None
    rename_safe = False  # renommer vers le code n'est sûr que si le nom-code est libre

    if ref_code:
        display = _canonical_code_name(ref_code)
        display_norm = normalize_entity_name(display)
        raw_norm = normalize_entity_name(raw_name)
        # 1) nœud canonique déjà tamponné avec ce code
        entity = _first_entity(session, space_id, ref_code=ref_code)
        # 2) nœud déjà nommé par le code nu (à tamponner)
        if entity is None and display_norm:
            entity = _first_entity(session, space_id, name_normalized=display_norm)
        # 3) forme descriptive existante (« Profil 6111 ») → adopter puis renommer.
        #    Sûr car (2) a confirmé qu'aucun nœud « 6111 » n'existe encore.
        if entity is None and raw_norm and raw_norm != display_norm:
            entity = _first_entity(session, space_id, name_normalized=raw_norm)
            rename_safe = entity is not None
        # la forme descriptive rejoint les alias
        if raw_name and raw_norm != display_norm:
            auto_aliases = list(dict.fromkeys([raw_name, *auto_aliases]))
        name, name_normalized = display, display_norm
    else:
        name_normalized = normalize_entity_name(name)
        entity = _first_entity(session, space_id, name_normalized=name_normalized)

    if entity is None:
        entity = KnowledgeEntity(
            space_id=space_id,
            name=name[:500],
            name_normalized=name_normalized[:500],
            entity_type=entity_type,
            ref_code=ref_code,
            description=(extracted.description or "")[:2000] or None,
            mention_count=1,
            confidence_score=extracted.confidence,
        )
        session.add(entity)
        session.flush()
    else:
        entity.mention_count += 1
        entity.updated_at = datetime.utcnow()
        # adoption : tamponne le code si absent
        if ref_code and not entity.ref_code:
            entity.ref_code = ref_code
        # promotion du nom vers le code canonique (ancien nom → alias), sans collision
        if ref_code and rename_safe and entity.name_normalized != name_normalized:
            if entity.name and normalize_entity_name(entity.name) != name_normalized:
                auto_aliases = list(dict.fromkeys([entity.name, *auto_aliases]))
            entity.name = name[:500]
            entity.name_normalized = name_normalized[:500]
        if extracted.description and not entity.description:
            entity.description = extracted.description[:2000]
        if extracted.confidence and (
            entity.confidence_score is None or extracted.confidence > entity.confidence_score
        ):
            entity.confidence_score = extracted.confidence
        session.add(entity)

    merged_aliases = list(dict.fromkeys(list(extracted.aliases) + auto_aliases))
    for alias in merged_aliases:
        alias_norm = normalize_entity_name(alias)
        if not alias_norm or alias_norm == name_normalized:
            continue
        alias_stmt = select(EntityAlias).where(
            EntityAlias.space_id == space_id,
            EntityAlias.alias_normalized == alias_norm,
        )
        existing_alias = session.exec(alias_stmt).first()
        if existing_alias is None:
            session.add(
                EntityAlias(
                    space_id=space_id,
                    entity_id=entity.id,
                    alias_normalized=alias_norm[:500],
                )
            )
        elif existing_alias.entity_id != entity.id:
            logger.debug(
                "[KAG] Alias %r déjà lié à entity_id=%s, ignoré pour entity_id=%s",
                alias_norm,
                existing_alias.entity_id,
                entity.id,
            )

    return entity


def _link_entity_to_chunks(
    session: Session,
    space_id: int,
    entity: KnowledgeEntity,
    chunks: List[DocumentChunk],
    *,
    relation_role: str = "mention",
    relevance_score: float = 1.0,
    context_snippet: Optional[str] = None,
) -> int:
    linked = 0
    role = relation_role if relation_role in _VALID_RELATION_ROLES else "mention"
    for chunk in chunks:
        stmt = select(ChunkEntityRelation).where(
            ChunkEntityRelation.chunk_id == chunk.id,
            ChunkEntityRelation.entity_id == entity.id,
            ChunkEntityRelation.relation_role == role,
        )
        if session.exec(stmt).first():
            continue
        session.add(
            ChunkEntityRelation(
                chunk_id=chunk.id,
                entity_id=entity.id,
                space_id=space_id,
                relation_role=role,
                relevance_score=max(0.0, min(1.0, relevance_score)),
                context_snippet=(context_snippet or "")[:1000] or None,
            )
        )
        linked += 1
    return linked


def _upsert_entity_relation(
    session: Session,
    space_id: int,
    entity_a: KnowledgeEntity,
    entity_b: KnowledgeEntity,
    relation_type: str,
    *,
    relation_label: Optional[str] = None,
    source_chunk_id: Optional[int] = None,
    confidence: Optional[float] = None,
) -> None:
    if entity_a.id == entity_b.id:
        return

    rel_type = _normalize_relation_type(relation_type)
    a_id, b_id = sorted((entity_a.id, entity_b.id))

    stmt = select(EntityEntityRelation).where(
        EntityEntityRelation.space_id == space_id,
        EntityEntityRelation.entity_a_id == a_id,
        EntityEntityRelation.entity_b_id == b_id,
        EntityEntityRelation.relation_type == rel_type,
    )
    existing = session.exec(stmt).first()
    if existing is None:
        session.add(
            EntityEntityRelation(
                space_id=space_id,
                entity_a_id=a_id,
                entity_b_id=b_id,
                relation_type=rel_type,
                relation_label=(relation_label or "")[:500] or None,
                weight=1.0,
                source_chunk_id=source_chunk_id,
                confidence=confidence,
            )
        )
    else:
        existing.weight += 1.0
        if confidence and (existing.confidence is None or confidence > existing.confidence):
            existing.confidence = confidence
        if relation_label and not existing.relation_label:
            existing.relation_label = relation_label[:500]
        if source_chunk_id and existing.source_chunk_id is None:
            existing.source_chunk_id = source_chunk_id
        session.add(existing)


def _compile_code_regex(codes: List[str]) -> Optional["re.Pattern"]:
    """Regex d'alternation avec frontières alphanumériques (« 6111 » sans matcher « 61110 »)."""
    parts = [re.escape(c) for c in codes if c]
    if not parts:
        return None
    return re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _lexical_link_codes(session: Session, space_id: int) -> int:
    """R2 — relie chaque entité à code à TOUS les chunks du space contenant ce code.

    Word-boundary, dans les deux sens (le doc courant vers les codes connus, et les
    codes du doc courant vers les anciens chunks). Idempotent. C'est le cœur de la
    couverture KAG : répare les chunks orphelins que le LLM n'a pas explicitement reliés
    (ex. le chunk propre « 6111 : longueur 155 mm »)."""
    code_entities = session.execute(
        text(
            "SELECT id, ref_code FROM knowledgeentity "
            "WHERE space_id = :sid AND ref_code IS NOT NULL"
        ),
        {"sid": space_id},
    ).all()
    if not code_entities:
        return 0

    plain_map: Dict[str, int] = {}  # CODE -> entity_id (réfs, normes)
    ral_map: Dict[str, int] = {}    # chiffres RAL -> entity_id
    for eid, code in code_entities:
        if not code:
            continue
        code = code.strip()
        if code.upper().startswith("RAL:"):
            ral_map[code[4:].strip().upper()] = int(eid)
            continue
        upper = code.upper()
        if upper.isdigit():
            # évite les collisions : un code numérique pur doit faire ≥ 4 chiffres
            # (« 155 » = cote, pas une réf → ne pas relier partout).
            if len(upper) < 4:
                continue
            # un code numérique nu qui est un RAL courant (9016, 7016…) n'est fiable QUE
            # dans un contexte « RAL 9016 » — sinon on relie la couleur partout (bruit).
            # Rattrape aussi le backfill regex qui a typé ces RAL comme des réfs.
            if upper in _COMMON_RAL:
                ral_map[upper] = int(eid)
                continue
        plain_map[upper] = int(eid)

    plain_re = _compile_code_regex(sorted(plain_map, key=len, reverse=True))
    ral_re = None
    if ral_map:
        ral_digits = "|".join(re.escape(d) for d in sorted(ral_map, key=len, reverse=True))
        ral_re = re.compile(r"RAL\s*[:\-]?\s*(" + ral_digits + r")", re.IGNORECASE)

    if plain_re is None and ral_re is None:
        return 0

    chunks = session.execute(
        text(
            """
            SELECT dc.id, dc.content
            FROM documentchunk dc
            JOIN document_space ds ON ds.document_id = dc.document_id
            WHERE ds.space_id = :sid
              AND dc.is_leaf = true
              AND dc.content IS NOT NULL
            """
        ),
        {"sid": space_id},
    ).all()
    if not chunks:
        return 0

    # Liens « mention » déjà présents → évite un SELECT par insertion.
    existing: Set[Tuple[int, int]] = {
        (int(c), int(e))
        for c, e in session.execute(
            text(
                "SELECT chunk_id, entity_id FROM chunkentityrelation "
                "WHERE space_id = :sid AND relation_role = 'mention'"
            ),
            {"sid": space_id},
        ).all()
    }

    created = 0
    for chunk_id, content in chunks:
        if not content:
            continue
        eids: Set[int] = set()
        if plain_re:
            for m in plain_re.finditer(content):
                eid = plain_map.get(m.group(0).upper())
                if eid:
                    eids.add(eid)
        if ral_re:
            for m in ral_re.finditer(content):
                eid = ral_map.get(m.group(1).upper())
                if eid:
                    eids.add(eid)
        for eid in eids:
            key = (int(chunk_id), eid)
            if key in existing:
                continue
            session.add(
                ChunkEntityRelation(
                    chunk_id=int(chunk_id),
                    entity_id=eid,
                    space_id=space_id,
                    relation_role="mention",
                    relevance_score=0.9,
                )
            )
            existing.add(key)
            created += 1
    return created


def _entity_target_chunks(
    extracted: KagExtractedEntity,
    page_chunks: List[DocumentChunk],
) -> Tuple[List[DocumentChunk], str]:
    """R3 — chunks cibles d'une entité.

    « subject » sur les chunks que le LLM a désignés comme parlant de l'entité
    (chunk_indexes = positions dans page_chunks) ; à défaut « mention » page-globale
    en secours. La couverture large est de toute façon assurée par R2 (lexical)."""
    if extracted.chunk_indexes:
        picked = [
            page_chunks[i]
            for i in dict.fromkeys(extracted.chunk_indexes)
            if 0 <= i < len(page_chunks)
        ]
        if picked:
            return picked, "subject"
    return page_chunks, "mention"


def _persist_page_kag(
    session: Session,
    space_ids: List[int],
    page_no: int,
    page_chunks: List[DocumentChunk],
    kag_response: KagPageResponse,
) -> Tuple[int, int]:
    """Persiste entités/relations d'une page pour chaque espace associé au document."""
    entities_count = 0
    relations_count = 0
    source_chunk_id = page_chunks[0].id if page_chunks else None
    # Ancre de page : un seul chunk pour rendre les endpoints d'une relation locatables,
    # sans « sprayer » toutes les entités sur tous les chunks de la page (bruit).
    anchor_chunks = page_chunks[:1]

    for space_id in space_ids:
        entity_by_norm: Dict[str, KnowledgeEntity] = {}

        for extracted in kag_response.entities:
            if not extracted.name or not extracted.name.strip():
                continue
            entity = _upsert_entity(session, space_id, extracted)
            entity_by_norm[normalize_entity_name(extracted.name)] = entity
            for alias in extracted.aliases:
                entity_by_norm[normalize_entity_name(alias)] = entity

            target_chunks, role = _entity_target_chunks(extracted, page_chunks)
            linked = _link_entity_to_chunks(
                session,
                space_id,
                entity,
                target_chunks,
                relation_role=role,
                context_snippet=extracted.description,
                relevance_score=extracted.confidence,
            )
            if linked:
                entities_count += 1

        for rel in kag_response.relations:
            # R4 — le pipeline n'écrit plus co_occurs (bruit) : R2 capture mieux la
            # co-présence, et le graphe ne garde que des liens sémantiques explicites.
            if _normalize_relation_type(rel.relation) == "co_occurs":
                continue

            norm_a = normalize_entity_name(rel.entity_a)
            norm_b = normalize_entity_name(rel.entity_b)
            entity_a = entity_by_norm.get(norm_a)
            entity_b = entity_by_norm.get(norm_b)

            if entity_a is None:
                entity_a = _upsert_entity(
                    session,
                    space_id,
                    KagExtractedEntity(name=rel.entity_a, type="other", confidence=rel.confidence),
                )
                entity_by_norm[norm_a] = entity_a
            if entity_b is None:
                entity_b = _upsert_entity(
                    session,
                    space_id,
                    KagExtractedEntity(name=rel.entity_b, type="other", confidence=rel.confidence),
                )
                entity_by_norm[norm_b] = entity_b

            # Endpoints reliés à l'ANCRE de page uniquement (localisation fine via R3/R2).
            _link_entity_to_chunks(
                session,
                space_id,
                entity_a,
                anchor_chunks,
                relation_role="subject",
                relevance_score=rel.confidence,
            )
            _link_entity_to_chunks(
                session,
                space_id,
                entity_b,
                anchor_chunks,
                relation_role="object",
                relevance_score=rel.confidence,
            )
            _upsert_entity_relation(
                session,
                space_id,
                entity_a,
                entity_b,
                rel.relation,
                relation_label=rel.relation_label,
                source_chunk_id=source_chunk_id,
                confidence=rel.confidence,
            )
            relations_count += 1

    return entities_count, relations_count


def _persist_page_categories(
    session: Session,
    space_ids: List[int],
    page_no: int,
    page_chunks: List[DocumentChunk],
    category_slugs: List[str],
    chunk_categories: List[ChunkCategoryItem],
    category_id_by_slug: Dict[str, int],
    document_id: int,
    valid_category_slugs: frozenset[str],
    page_wide_slugs: Optional[List[str]] = None,
) -> int:
    """Persiste les catégories multi-axes des chunks de la page.

    - chunk_categories (axes task/symptom) → chunks ciblés par chunk_index.
    - page_wide_slugs (axes doc_type/lifecycle_phase) → TOUS les chunks de la page.
    - category_slugs (legacy union page-level task) → fallback tous chunks si aucun chunk_categories.
    """
    if not page_chunks:
        return 0

    min_conf = settings.CATEGORY_MIN_CONFIDENCE
    max_per_chunk = settings.CATEGORY_MAX_PER_CHUNK

    # chunk_index → catégories task/symptom notées, filtrées (seuil + top-K).
    index_to_scored: Dict[int, List[ScoredCategory]] = {}
    for item in chunk_categories or []:
        scored = _apply_threshold_topk(
            _dedup_scored_categories(item.categories, valid_category_slugs),
            min_conf,
            max_per_chunk,
        )
        if scored:
            index_to_scored[item.chunk_index] = scored

    page_wide = list(dict.fromkeys(page_wide_slugs or []))

    # Position du chunk → {slug: (confidence, primary)}.
    position_scored: Dict[int, Dict[str, Tuple[float, bool]]] = {
        idx: {} for idx in range(len(page_chunks))
    }

    def _add(idx: int, slug: str, conf: float, primary: bool) -> None:
        bucket = position_scored.setdefault(idx, {})
        prev = bucket.get(slug)
        if prev is None:
            bucket[slug] = (conf, primary)
        else:
            bucket[slug] = (max(prev[0], conf), prev[1] or primary)

    # 1. chunk-level task/symptom (notés)
    for chunk_idx, scored in index_to_scored.items():
        if 0 <= chunk_idx < len(page_chunks):
            for sc in scored:
                _add(chunk_idx, sc.slug, sc.confidence, sc.primary)
    # 2. legacy page-level task fallback (uniquement si aucun chunk_categories noté)
    if not index_to_scored and category_slugs:
        for slug in _normalize_category_slugs(category_slugs, valid_category_slugs):
            for idx in range(len(page_chunks)):
                _add(idx, slug, 0.6, False)
    # 3. page-wide doc_type/lifecycle → tous les chunks (facette homogène, confiance pleine)
    if page_wide:
        for slug in page_wide:
            for idx in range(len(page_chunks)):
                _add(idx, slug, 1.0, False)

    linked = 0
    for idx, chunk in enumerate(page_chunks):
        bucket = position_scored.get(idx) or {}

        meta = dict(chunk.metadata_json or {})
        if bucket:
            existing = meta.get("categories") or []
            if not isinstance(existing, list):
                existing = []
            meta["categories"] = list(dict.fromkeys([*existing, *bucket.keys()]))
        meta["kag_extraction_version"] = KAG_EXTRACTION_VERSION
        meta["taxonomy_version"] = TAXONOMY_VERSION
        chunk.metadata_json = meta
        chunk.metadata_ = meta
        session.add(chunk)

        for slug, (conf, primary) in bucket.items():
            category_id = category_id_by_slug.get(slug)
            if category_id is None:
                continue
            # Lien au niveau document : une seule ligne par (chunk, catégorie), sans espace.
            exists = session.exec(
                select(ChunkCategoryRelation).where(
                    ChunkCategoryRelation.chunk_id == chunk.id,
                    ChunkCategoryRelation.category_id == category_id,
                )
            ).first()
            if exists is not None:
                # Ré-extraction : rafraîchir confiance / primaire si changé.
                changed = False
                if abs((exists.confidence or 0.0) - conf) > 1e-6:
                    exists.confidence = conf
                    changed = True
                if bool(getattr(exists, "is_primary", False)) != primary:
                    exists.is_primary = primary
                    changed = True
                if changed:
                    session.add(exists)
                continue
            session.add(
                ChunkCategoryRelation(
                    chunk_id=chunk.id,
                    category_id=category_id,
                    document_id=document_id,
                    page_no=page_no,
                    confidence=conf,
                    is_primary=primary,
                )
            )
            linked += 1
    return linked


def _persist_symptom_candidates(
    session: Session,
    candidates: List[str],
    *,
    document_id: int,
    known_symptom_slugs: frozenset,
) -> int:
    """Upsert des symptômes candidats (hors-liste) en attente de validation humaine.

    Jamais lié à un chunk ni exploité par le retrieval tant qu'un expert ne l'a pas promu
    en DocumentCategory (axis=symptom). Incrémente le compteur d'occurrences si déjà vu.
    """
    from app.models.category_candidate import CategoryCandidate

    touched = 0
    for raw in candidates or []:
        label = (raw or "").strip()
        slug = _slugify_candidate(label)
        if not slug or slug in known_symptom_slugs:
            continue
        existing = session.exec(
            select(CategoryCandidate).where(CategoryCandidate.slug == slug)
        ).first()
        if existing is not None:
            existing.occurrence_count = (existing.occurrence_count or 0) + 1
            existing.updated_at = datetime.utcnow()
            session.add(existing)
        else:
            session.add(
                CategoryCandidate(
                    slug=slug,
                    label=label[:200],
                    axis="symptom",
                    proposed_description=label[:500],
                    occurrence_count=1,
                    status="pending",
                    first_seen_document_id=document_id,
                )
            )
        touched += 1
    return touched


def _annotate_chunks_with_entities(
    session: Session,
    page_chunks: List[DocumentChunk],
    entities: List[KagExtractedEntity],
    *,
    max_entities: int = 12,
) -> None:
    """
    Écrit les noms canoniques d'entités de la page dans les métadonnées de chaque chunk L1.

    Sert au texte d'embedding (`_build_embed_text`) : le vecteur dense intègre ainsi les
    références/produits de la page. Granularité page (cohérente avec le linking entité→chunk).
    """
    if not page_chunks or not entities:
        return

    names: List[str] = []
    seen: set[str] = set()
    for extracted in entities:
        raw = (extracted.name or "").strip()
        if not raw:
            continue
        canonical, _ = normalize_and_expand_entity(raw)
        name = canonical or raw
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) >= max_entities:
            break

    if not names:
        return

    for chunk in page_chunks:
        meta = dict(chunk.metadata_json or {})
        existing = meta.get("entities") or []
        if not isinstance(existing, list):
            existing = []
        merged = list(dict.fromkeys([*existing, *names]))[:max_entities]
        meta["entities"] = merged
        chunk.metadata_json = meta
        chunk.metadata_ = meta
        session.add(chunk)


def _merge_page_kag_responses(
    target: Dict[int, KagPageResponse],
    batch_response: BatchKagResponse,
) -> None:
    """Fusionne les réponses batch (union catégories sur pages overlap)."""
    for page_resp in batch_response.pages:
        pno = page_resp.page_no
        existing = target.get(pno)
        if existing is None:
            target[pno] = page_resp
            continue

        merged_categories = list(
            dict.fromkeys([*(existing.categories or []), *(page_resp.categories or [])])
        )
        merged_chunk_map: Dict[int, Dict[str, ScoredCategory]] = {}

        def _merge_scored_into(idx: int, scored_list: List[ScoredCategory]) -> None:
            bucket = merged_chunk_map.setdefault(idx, {})
            for sc in scored_list or []:
                prev = bucket.get(sc.slug)
                if prev is None:
                    bucket[sc.slug] = ScoredCategory(
                        slug=sc.slug, confidence=sc.confidence, primary=sc.primary
                    )
                else:
                    prev.confidence = max(prev.confidence, sc.confidence)
                    prev.primary = prev.primary or sc.primary

        for item in existing.chunk_categories:
            _merge_scored_into(item.chunk_index, item.categories)
        for item in page_resp.chunk_categories:
            _merge_scored_into(item.chunk_index, item.categories)
        merged_chunk_categories = [
            ChunkCategoryItem(chunk_index=idx, categories=list(bucket.values()))
            for idx, bucket in sorted(merged_chunk_map.items())
        ]
        merged_entities = {normalize_entity_name(e.name): e for e in existing.entities}
        for ent in page_resp.entities:
            merged_entities[normalize_entity_name(ent.name)] = ent
        merged_relations = list(existing.relations)
        seen_rels = {
            (normalize_entity_name(r.entity_a), r.relation, normalize_entity_name(r.entity_b))
            for r in merged_relations
        }
        for rel in page_resp.relations:
            key = (normalize_entity_name(rel.entity_a), rel.relation, normalize_entity_name(rel.entity_b))
            if key not in seen_rels:
                seen_rels.add(key)
                merged_relations.append(rel)

        merged_doc_types = list(
            dict.fromkeys([*(existing.doc_types or []), *(page_resp.doc_types or [])])
        )
        merged_lifecycle = list(
            dict.fromkeys([*(existing.lifecycle_phases or []), *(page_resp.lifecycle_phases or [])])
        )
        merged_symptom_candidates = _clean_freeform_list(
            [*(existing.symptom_candidates or []), *(page_resp.symptom_candidates or [])]
        )

        target[pno] = KagPageResponse(
            page_no=pno,
            entities=list(merged_entities.values()),
            relations=merged_relations,
            categories=merged_categories,
            chunk_categories=merged_chunk_categories,
            doc_types=merged_doc_types,
            lifecycle_phases=merged_lifecycle,
            symptom_candidates=merged_symptom_candidates,
        )


# ---------------------------------------------------------------------------
# Nettoyage KAG document
# ---------------------------------------------------------------------------


def delete_chunk_kag_relations(
    session: Session,
    chunk_ids: Sequence[int],
) -> List[Tuple[int, int]]:
    """
    Supprime les relations KAG qui bloquent la suppression de chunks (FK).
    Retourne les paires (entity_id, mention_count_delta) pour ajustement ultérieur.
    """
    if not chunk_ids:
        return []

    chunk_ids_tuple = tuple(chunk_ids)

    affected = session.execute(
        text(
            """
            SELECT entity_id, COUNT(*) AS cnt
            FROM chunkentityrelation
            WHERE chunk_id IN :chunk_ids
            GROUP BY entity_id
            """
        ),
        {"chunk_ids": chunk_ids_tuple},
    ).all()

    session.execute(
        text("DELETE FROM chunkentityrelation WHERE chunk_id IN :chunk_ids"),
        {"chunk_ids": chunk_ids_tuple},
    )

    try:
        with session.begin_nested():
            session.execute(
                text("DELETE FROM chunkcategoryrelation WHERE chunk_id IN :chunk_ids"),
                {"chunk_ids": chunk_ids_tuple},
            )
    except Exception as exc:
        logger.warning(
            "[KAG] Suppression chunkcategoryrelation ignorée (%s chunk(s)) : %s",
            len(chunk_ids_tuple),
            exc,
        )

    session.execute(
        text("DELETE FROM entityentityrelation WHERE source_chunk_id IN :chunk_ids"),
        {"chunk_ids": chunk_ids_tuple},
    )

    return [(int(entity_id), int(cnt)) for entity_id, cnt in affected]


def prune_kag_entities_after_chunk_removal(
    session: Session,
    affected: Sequence[Tuple[int, int]],
) -> None:
    """Supprime les entités devenues ORPHELINES (plus aucun lien chunk) et rafraîchit
    mention_count des survivantes.

    On ne se fie PLUS au décompte mention_count - liens_supprimés (fragile : le linking
    lexical R2 ajoute des liens sans toucher mention_count, ce qui sur-décrémentait et
    supprimait à tort des entités encore présentes dans d'autres documents). La vraie
    condition d'orphelin est « plus aucun chunkentityrelation »."""
    if not affected:
        return

    entity_ids = [int(eid) for eid, _ in affected]

    # 1) Orphelines du lot : plus aucun lien chunk restant → suppression (+ alias + relations).
    orphan_ids = [
        int(r[0])
        for r in session.execute(
            text(
                """
                SELECT ke.id
                FROM knowledgeentity ke
                WHERE ke.id = ANY(:eids)
                  AND NOT EXISTS (
                      SELECT 1 FROM chunkentityrelation cer WHERE cer.entity_id = ke.id
                  )
                """
            ),
            {"eids": entity_ids},
        ).all()
    ]
    if orphan_ids:
        session.execute(
            text(
                "DELETE FROM entityentityrelation "
                "WHERE entity_a_id = ANY(:ids) OR entity_b_id = ANY(:ids)"
            ),
            {"ids": orphan_ids},
        )
        session.execute(
            text("DELETE FROM entityalias WHERE entity_id = ANY(:ids)"),
            {"ids": orphan_ids},
        )
        session.execute(
            text("DELETE FROM knowledgeentity WHERE id = ANY(:ids)"),
            {"ids": orphan_ids},
        )

    # 2) Survivantes : mention_count = nombre de chunks distincts encore liés (signal
    #    d'affichage/ranking cohérent avec la couverture réelle).
    orphan_set = set(orphan_ids)
    survivor_ids = [e for e in entity_ids if e not in orphan_set]
    if survivor_ids:
        session.execute(
            text(
                """
                UPDATE knowledgeentity ke
                SET mention_count = GREATEST(1, (
                        SELECT COUNT(DISTINCT cer.chunk_id)
                        FROM chunkentityrelation cer WHERE cer.entity_id = ke.id
                    )),
                    updated_at = NOW()
                WHERE ke.id = ANY(:ids)
                """
            ),
            {"ids": survivor_ids},
        )


def cleanup_kag_for_document(session: Session, document_id: int) -> None:
    """Supprime les relations KAG liées aux chunks d'un document avant retraitement."""
    chunk_ids = [
        row[0]
        for row in session.execute(
            text("SELECT id FROM documentchunk WHERE document_id = :doc_id"),
            {"doc_id": document_id},
        ).all()
    ]
    if not chunk_ids:
        return

    affected = delete_chunk_kag_relations(session, chunk_ids)
    prune_kag_entities_after_chunk_removal(session, affected)
    logger.info(
        "[KAG] Nettoyage document_id=%s — %s chunks, %s entités affectées",
        document_id,
        len(chunk_ids),
        len(affected),
    )


# ---------------------------------------------------------------------------
# Point d'entrée indexation
# ---------------------------------------------------------------------------


def extract_kag_for_document(document_id: int, pdf_path: str) -> dict:
    """
    Extrait et persiste entités/relations pour un document indexé.
    Non bloquant : retourne des compteurs même si certaines pages échouent.
    """
    if not settings.KAG_ENABLED:
        return {"entities": 0, "relations": 0, "pages": 0, "status": "disabled"}

    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document introuvable : {document_id}")

        space_ids = _get_document_space_ids(session, document_id)
        if not space_ids:
            logger.warning("[KAG] document_id=%s sans espace associé — extraction ignorée", document_id)
            return {"entities": 0, "relations": 0, "pages": 0, "status": "no_space"}

        chunks_by_page = _load_l1_chunks_by_page(session, document_id)
        if not chunks_by_page:
            logger.warning("[KAG] document_id=%s sans chunks L1 — extraction ignorée", document_id)
            return {"entities": 0, "relations": 0, "pages": 0, "status": "no_chunks"}

        doc_title = document.title or ""
        page_numbers = sorted(chunks_by_page.keys())
        concurrency = settings.KAG_EXTRACTION_CONCURRENCY

        from app.services.category_catalog import (
            AXIS_SYMPTOM,
            AXIS_TASK,
            get_allowed_slugs_by_axis,
            get_categories_for_prompt_grouped_by_axis,
            get_category_id_by_slug,
        )

        category_list = get_categories_for_prompt_grouped_by_axis(session)
        category_id_by_slug = get_category_id_by_slug(session)
        allowed_by_axis = get_allowed_slugs_by_axis(session)
        # chunk_categories = axes task ∪ symptom (par chunk) ; doc_type/lifecycle = page-level
        chunk_axis_slugs = frozenset(
            allowed_by_axis.get(AXIS_TASK, frozenset()) | allowed_by_axis.get(AXIS_SYMPTOM, frozenset())
        )
        known_symptom_slugs = allowed_by_axis.get(AXIS_SYMPTOM, frozenset())

        batches = build_kag_batches(page_numbers)
        page_responses: Dict[int, KagPageResponse] = {}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    extract_batch_kag_response,
                    pdf_path,
                    batch,
                    doc_title,
                    chunks_by_page,
                    category_list,
                    chunk_axis_slugs,
                    allowed_by_axis,
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    batch_response = future.result()
                    if batch_response:
                        _merge_page_kag_responses(page_responses, batch_response)
                except Exception as exc:
                    logger.error("[KAG] Extraction batch %s échouée : %s", batch, exc)

        total_entities = 0
        total_relations = 0
        total_categories = 0
        pages_ok = 0
        symptom_candidates_all: List[str] = []

        for pno in page_numbers:
            kag_response = page_responses.get(pno)
            if not kag_response:
                continue
            ents, rels = _persist_page_kag(
                session,
                space_ids,
                pno,
                chunks_by_page[pno],
                kag_response,
            )
            page_wide_slugs = [
                *(kag_response.doc_types or []),
                *(kag_response.lifecycle_phases or []),
            ]
            cats = _persist_page_categories(
                session,
                space_ids,
                pno,
                chunks_by_page[pno],
                kag_response.categories,
                kag_response.chunk_categories,
                category_id_by_slug,
                document_id,
                chunk_axis_slugs,
                page_wide_slugs=page_wide_slugs,
            )
            symptom_candidates_all.extend(kag_response.symptom_candidates or [])
            # Métadonnées entités sur les chunks L1 → enrichit le texte d'embedding
            _annotate_chunks_with_entities(session, chunks_by_page[pno], kag_response.entities)
            total_entities += ents
            total_relations += rels
            total_categories += cats
            pages_ok += 1

        # Catégories à vocabulaire fermé : on ne génère plus de candidats dynamiques
        # (les concepts émergents relèvent des entités KAG, pas des catégories).
        candidates_touched = 0

        # R2 — linking lexical : relie les codes à TOUS les chunks du space qui les
        # contiennent (répare les orphelins, dans les deux sens). Cœur de la couverture.
        lexical_links = 0
        for sid in space_ids:
            try:
                lexical_links += _lexical_link_codes(session, sid)
            except Exception as exc:
                logger.warning("[KAG] Linking lexical échoué space=%s : %s", sid, exc)

        session.commit()

        logger.info(
            "[KAG] Extraction terminée document_id=%s pages=%s/%s batches=%s "
            "entities=%s relations=%s category_links=%s lexical_links=%s symptom_candidates=%s model=%s",
            document_id,
            pages_ok,
            len(page_numbers),
            len(batches),
            total_entities,
            total_relations,
            total_categories,
            lexical_links,
            candidates_touched,
            _kag_extraction_model(),
        )
        return {
            "entities": total_entities,
            "relations": total_relations,
            "categories": total_categories,
            "lexical_links": lexical_links,
            "symptom_candidates": candidates_touched,
            "pages": pages_ok,
            "status": "completed",
        }


def embed_kag_entities_for_document(document_id: int) -> int:
    """Embed les entités KAG nouvellement liées au document (sans embedding)."""
    if not settings.KAG_ENABLED:
        return 0

    from app.services.embedding_service import generate_embeddings_batch

    with Session(engine) as session:
        entity_ids = {
            row[0]
            for row in session.execute(
                text(
                    """
                    SELECT DISTINCT cer.entity_id
                    FROM chunkentityrelation cer
                    INNER JOIN documentchunk dc ON dc.id = cer.chunk_id
                    WHERE dc.document_id = :doc_id
                    """
                ),
                {"doc_id": document_id},
            ).all()
        }
        if not entity_ids:
            return 0

        stmt = select(KnowledgeEntity).where(KnowledgeEntity.id.in_(entity_ids))  # type: ignore[attr-defined]
        entities = [e for e in session.exec(stmt).all() if e.embedding is None]
        if not entities:
            return 0

        texts = [
            f"{e.entity_type}: {e.name}. {e.description or ''}".strip()
            for e in entities
        ]
        embeddings = generate_embeddings_batch(texts, batch_size=settings.EMBEDDING_BATCH_SIZE)

        embedded = 0
        for entity, vector in zip(entities, embeddings):
            if vector:
                entity.embedding = vector
                entity.updated_at = datetime.utcnow()
                session.add(entity)
                embedded += 1

        session.commit()
        logger.info("[KAG] %s entités embeddées pour document_id=%s", embedded, document_id)
        return embedded
