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
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError
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

KAG_EXTRACTION_VERSION = "kag_vision_v3"

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


class KagExtractedRelation(BaseModel):
    entity_a: str
    relation: str
    entity_b: str
    relation_label: Optional[str] = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ChunkCategoryItem(BaseModel):
    chunk_index: int = Field(ge=0)
    categories: List[str] = Field(default_factory=list)


class KagPageResponse(BaseModel):
    page_no: int
    entities: List[KagExtractedEntity] = Field(default_factory=list)
    relations: List[KagExtractedRelation] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    chunk_categories: List[ChunkCategoryItem] = Field(default_factory=list)


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
4. Pour chaque entité, fournis un type parmi : product | material | tool | norm | dimension | process | organization | location | reference | other
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

_KAG_BATCH_SYSTEM_PROMPT = """Tu es un expert en extraction d'entités nommées (NER), de relations (RE) et de catégorisation de contenu pour documents techniques menuiserie.
On te donne les images de plusieurs pages consécutives ET le texte déjà extrait (chunks transcrits par page, identifiés par chunk_index).
Ton travail : pour CHAQUE page, identifier les entités, relations et catégories de contenu PAR CHUNK.

Règles impératives :
1. Renvoie UNIQUEMENT un objet JSON valide (aucun texte hors JSON).
2. Extrais les entités concrètes : produits, références, matériaux, outils, normes, dimensions, processus, organisations, lieux.
3. Normalise les noms (casse cohérente, sans bruit markdown).
4. Pour chaque entité, fournis un type parmi : product | material | tool | norm | dimension | process | organization | location | reference | other
5. Les aliases sont les variantes, abréviations ou codes produit (ex. "ref ABC-123").
6. Les relations décrivent un lien sémantique explicite entre deux entités d'une même page.
7. Types de relation suggérés : compatible_avec | est_compose_de | remplace | utilise | conforme_a | installe_sur | fabrique_par | mesure | reference | co_occurs
8. Ne pas inventer d'entités absentes du texte ou de l'image.
9. Maximum {max_entities} entités et {max_relations} relations par page.
10. Catégories par chunk : pour chaque chunk_index, choisis UNIQUEMENT parmi la liste fournie (slug exact). Un chunk peut avoir 0 à plusieurs catégories selon son contenu réel.
11. N'associe une catégorie qu'aux chunks dont le contenu traite explicitement du thème. Ne propage pas une catégorie à tous les chunks de la page.
12. N'invente pas de catégories hors liste.
13. Le champ "categories" au niveau page est optionnel (union des catégories présentes sur la page). Privilégie "chunk_categories".

Catégories autorisées (slug : description) :
{category_list}

Format de réponse OBLIGATOIRE :
{{
  "pages": [
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
      ],
      "categories": ["<slug1>"],
      "chunk_categories": [
        {{ "chunk_index": 0, "categories": ["mounting", "hardware_adjustment"] }},
        {{ "chunk_index": 2, "categories": ["dimensions_tolerances"] }}
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


def _normalize_relation_type(raw: str) -> str:
    value = (raw or "co_occurs").strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value or "co_occurs"


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


def _normalize_chunk_categories(
    chunk_categories: List[ChunkCategoryItem],
    valid_slugs: frozenset[str],
) -> List[ChunkCategoryItem]:
    normalized: List[ChunkCategoryItem] = []
    for item in chunk_categories or []:
        slugs = _normalize_category_slugs(item.categories, valid_slugs)
        if slugs:
            normalized.append(ChunkCategoryItem(chunk_index=item.chunk_index, categories=slugs))
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


def _coerce_kag_page_response(
    raw: dict,
    page_no: int,
    *,
    valid_category_slugs: Optional[frozenset[str]] = None,
) -> KagPageResponse:
    """Valide et normalise une réponse KAG (tolère page_no absent)."""
    payload = dict(raw or {})
    payload.setdefault("page_no", page_no)
    if not isinstance(payload.get("entities"), list):
        payload["entities"] = []
    if not isinstance(payload.get("relations"), list):
        payload["relations"] = []
    if not isinstance(payload.get("categories"), list):
        payload["categories"] = []
    if not isinstance(payload.get("chunk_categories"), list):
        payload["chunk_categories"] = []

    response = KagPageResponse.model_validate(payload)
    max_ent, max_rel = _effective_kag_limits()
    response.entities = response.entities[:max_ent]
    response.relations = response.relations[:max_rel]
    if valid_category_slugs is not None:
        response.categories = _normalize_category_slugs(response.categories, valid_category_slugs)
        response.chunk_categories = _normalize_chunk_categories(
            response.chunk_categories,
            valid_category_slugs,
        )
    has_chunk_categories = bool(response.chunk_categories)
    if (
        not response.entities
        and not response.relations
        and not response.categories
        and not has_chunk_categories
    ):
        raise ValueError(f"Aucune entité, relation ni catégorie extraite pour la page {page_no}")
    return response


def _coerce_batch_kag_response(
    raw: dict,
    batch_pages: List[int],
    *,
    valid_category_slugs: frozenset[str],
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
            _coerce_kag_page_response(item, page_no, valid_category_slugs=valid_category_slugs)
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
        if meta.get("content_type") not in (None, "semantic_leaf"):
            continue
        page_no = meta.get("page_no") or meta.get("page_start")
        if page_no is None:
            continue
        by_page.setdefault(int(page_no), []).append(chunk)
    for pno in by_page:
        by_page[pno].sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    return by_page


def _upsert_entity(
    session: Session,
    space_id: int,
    extracted: KagExtractedEntity,
) -> KnowledgeEntity:
    canonical_name, auto_aliases = normalize_and_expand_entity(extracted.name.strip())
    name = canonical_name or extracted.name.strip()
    name_normalized = normalize_entity_name(name)
    entity_type = _normalize_entity_type(extracted.type)

    stmt = select(KnowledgeEntity).where(
        KnowledgeEntity.space_id == space_id,
        KnowledgeEntity.name_normalized == name_normalized,
        KnowledgeEntity.entity_type == entity_type,
    )
    entity = session.exec(stmt).first()

    if entity is None:
        entity = KnowledgeEntity(
            space_id=space_id,
            name=name[:500],
            name_normalized=name_normalized[:500],
            entity_type=entity_type,
            description=(extracted.description or "")[:2000] or None,
            mention_count=1,
            confidence_score=extracted.confidence,
        )
        session.add(entity)
        session.flush()
    else:
        entity.mention_count += 1
        entity.updated_at = datetime.utcnow()
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

    for space_id in space_ids:
        entity_by_norm: Dict[str, KnowledgeEntity] = {}

        for extracted in kag_response.entities:
            if not extracted.name or not extracted.name.strip():
                continue
            entity = _upsert_entity(session, space_id, extracted)
            entity_by_norm[normalize_entity_name(extracted.name)] = entity
            for alias in extracted.aliases:
                entity_by_norm[normalize_entity_name(alias)] = entity
            linked = _link_entity_to_chunks(
                session,
                space_id,
                entity,
                page_chunks,
                context_snippet=extracted.description,
                relevance_score=extracted.confidence,
            )
            if linked:
                entities_count += 1

        for rel in kag_response.relations:
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

            _link_entity_to_chunks(
                session,
                space_id,
                entity_a,
                page_chunks,
                relation_role="subject",
                relevance_score=rel.confidence,
            )
            _link_entity_to_chunks(
                session,
                space_id,
                entity_b,
                page_chunks,
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
) -> int:
    """Persiste les catégories de contenu pour les chunks concernés (chunk-level ou fallback page)."""
    if not page_chunks:
        return 0

    index_to_slugs: Dict[int, List[str]] = {}
    for item in chunk_categories or []:
        slugs = _normalize_category_slugs(item.categories, valid_category_slugs)
        if slugs:
            index_to_slugs[item.chunk_index] = slugs

    targets: List[tuple[DocumentChunk, List[str]]] = []
    if index_to_slugs:
        for chunk_idx, slugs in index_to_slugs.items():
            if chunk_idx < 0 or chunk_idx >= len(page_chunks):
                continue
            targets.append((page_chunks[chunk_idx], slugs))
    elif category_slugs:
        for chunk in page_chunks:
            targets.append((chunk, category_slugs))

    if not targets:
        return 0

    linked = 0
    for chunk, slugs in targets:
        meta = dict(chunk.metadata_json or {})
        existing = meta.get("categories") or []
        if not isinstance(existing, list):
            existing = []
        merged_slugs = list(dict.fromkeys([*existing, *slugs]))
        meta["categories"] = merged_slugs
        meta["kag_extraction_version"] = KAG_EXTRACTION_VERSION
        chunk.metadata_json = meta
        chunk.metadata_ = meta
        session.add(chunk)

        for slug in slugs:
            category_id = category_id_by_slug.get(slug)
            if category_id is None:
                continue
            for space_id in space_ids:
                stmt = select(ChunkCategoryRelation).where(
                    ChunkCategoryRelation.chunk_id == chunk.id,
                    ChunkCategoryRelation.category_id == category_id,
                )
                if session.exec(stmt).first():
                    continue
                session.add(
                    ChunkCategoryRelation(
                        chunk_id=chunk.id,
                        category_id=category_id,
                        space_id=space_id,
                        document_id=document_id,
                        page_no=page_no,
                        confidence=1.0,
                    )
                )
                linked += 1
    return linked


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
        merged_chunk_map: Dict[int, List[str]] = {}
        for item in existing.chunk_categories:
            merged_chunk_map[item.chunk_index] = list(item.categories)
        for item in page_resp.chunk_categories:
            prev = merged_chunk_map.get(item.chunk_index, [])
            merged_chunk_map[item.chunk_index] = list(
                dict.fromkeys([*prev, *item.categories])
            )
        merged_chunk_categories = [
            ChunkCategoryItem(chunk_index=idx, categories=slugs)
            for idx, slugs in sorted(merged_chunk_map.items())
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

        target[pno] = KagPageResponse(
            page_no=pno,
            entities=list(merged_entities.values()),
            relations=merged_relations,
            categories=merged_categories,
            chunk_categories=merged_chunk_categories,
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
    """Décrémente mention_count et supprime les entités KAG devenues orphelines."""
    for entity_id, cnt in affected:
        session.execute(
            text(
                """
                UPDATE knowledgeentity
                SET mention_count = GREATEST(0, mention_count - :cnt),
                    updated_at = NOW()
                WHERE id = :entity_id
                """
            ),
            {"entity_id": entity_id, "cnt": int(cnt)},
        )

    session.execute(
        text(
            """
            DELETE FROM entityentityrelation
            WHERE entity_a_id IN (SELECT id FROM knowledgeentity WHERE mention_count <= 0)
               OR entity_b_id IN (SELECT id FROM knowledgeentity WHERE mention_count <= 0)
            """
        )
    )
    session.execute(
        text(
            """
            DELETE FROM entityalias
            WHERE entity_id IN (SELECT id FROM knowledgeentity WHERE mention_count <= 0)
            """
        )
    )
    session.execute(text("DELETE FROM knowledgeentity WHERE mention_count <= 0"))


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

        from app.services.category_catalog import get_active_categories_for_prompt, get_category_id_by_slug

        category_list = get_active_categories_for_prompt(session)
        category_id_by_slug = get_category_id_by_slug(session)
        valid_category_slugs = frozenset(category_id_by_slug.keys())

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
                    valid_category_slugs,
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
            cats = _persist_page_categories(
                session,
                space_ids,
                pno,
                chunks_by_page[pno],
                kag_response.categories,
                kag_response.chunk_categories,
                category_id_by_slug,
                document_id,
                valid_category_slugs,
            )
            # Métadonnées entités sur les chunks L1 → enrichit le texte d'embedding
            _annotate_chunks_with_entities(session, chunks_by_page[pno], kag_response.entities)
            total_entities += ents
            total_relations += rels
            total_categories += cats
            pages_ok += 1

        session.commit()

        logger.info(
            "[KAG] Extraction terminée document_id=%s pages=%s/%s batches=%s "
            "entities=%s relations=%s category_links=%s model=%s",
            document_id,
            pages_ok,
            len(page_numbers),
            len(batches),
            total_entities,
            total_relations,
            total_categories,
            _kag_extraction_model(),
        )
        return {
            "entities": total_entities,
            "relations": total_relations,
            "categories": total_categories,
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
