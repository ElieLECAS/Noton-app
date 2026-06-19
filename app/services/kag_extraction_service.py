"""
Extraction d'entités et relations KAG via le même modèle vision que l'indexation texte.

Seconde passe par page après persistance des chunks L1 :
  1. PNG page (cache) + texte des chunks L1 de la page
  2. Appel Ministral vision → JSON { entities, relations }
  3. Normalisation + upsert entités au niveau espace
  4. Persistance chunkentityrelation + entityentityrelation + entityalias
"""

from __future__ import annotations

import base64
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.knowledge_entity import (
    ChunkEntityRelation,
    EntityAlias,
    EntityEntityRelation,
    KnowledgeEntity,
)

logger = logging.getLogger(__name__)

KAG_EXTRACTION_VERSION = "kag_vision_v1"

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


class KagPageResponse(BaseModel):
    page_no: int
    entities: List[KagExtractedEntity] = Field(default_factory=list)
    relations: List[KagExtractedRelation] = Field(default_factory=list)


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


# ---------------------------------------------------------------------------
# Utilitaires normalisation
# ---------------------------------------------------------------------------


def normalize_entity_name(name: str) -> str:
    """Normalise un nom d'entité pour déduplication (lowercase, NFKC, espaces)."""
    if not name:
        return ""
    normalized = unicodedata.normalize("NFKC", name.strip())
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


def _build_kag_system_prompt() -> str:
    return _KAG_SYSTEM_PROMPT.format(
        max_entities=settings.KAG_MAX_ENTITIES_PER_PAGE,
        max_relations=settings.KAG_MAX_RELATIONS_PER_PAGE,
    )


# ---------------------------------------------------------------------------
# Appel API vision KAG
# ---------------------------------------------------------------------------


def _call_kag_vision_api(
    image_b64: str,
    page_no: int,
    document_title: str,
    chunk_text: str,
) -> dict:
    from app.services.multimodal_page_service import (
        _mistral_chat_completion,
        _parse_json_with_repair,
    )

    user_text = _KAG_USER_PROMPT_TEMPLATE.format(
        title=document_title or "Document",
        page_no=page_no,
        chunk_text=chunk_text[:12000],
    )

    messages = [
        {"role": "system", "content": _build_kag_system_prompt()},
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
        max_tokens=2048,
        temperature=0.0,
        response_format_json=True,
        timeout_seconds=settings.KAG_EXTRACTION_TIMEOUT,
        model=_kag_extraction_model(),
    )
    return _parse_json_with_repair(raw)


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
        raw = _call_kag_vision_api(image_b64, page_no, document_title, chunk_text)
        response = KagPageResponse.model_validate(raw)
        response.entities = response.entities[: settings.KAG_MAX_ENTITIES_PER_PAGE]
        response.relations = response.relations[: settings.KAG_MAX_RELATIONS_PER_PAGE]
        return response
    except (ValidationError, ValueError) as exc:
        logger.warning("[KAG] Validation page %s échouée : %s", page_no, exc)
        return None
    except Exception as exc:
        logger.warning("[KAG] Extraction page %s échouée : %s", page_no, exc)
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
    return by_page


def _upsert_entity(
    session: Session,
    space_id: int,
    extracted: KagExtractedEntity,
) -> KnowledgeEntity:
    name = extracted.name.strip()
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

    for alias in extracted.aliases:
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


# ---------------------------------------------------------------------------
# Nettoyage KAG document
# ---------------------------------------------------------------------------


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
    session.execute(
        text("DELETE FROM entityentityrelation WHERE source_chunk_id IN :chunk_ids"),
        {"chunk_ids": chunk_ids_tuple},
    )

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

    session.execute(text("DELETE FROM knowledgeentity WHERE mention_count <= 0"))
    session.commit()
    logger.info("[KAG] Nettoyage document_id=%s — %s chunks, %s entités affectées", document_id, len(chunk_ids), len(affected))


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

        page_responses: Dict[int, Optional[KagPageResponse]] = {}

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    extract_page_kag_response,
                    pdf_path,
                    pno,
                    doc_title,
                    [c.content for c in chunks_by_page[pno]],
                ): pno
                for pno in page_numbers
            }
            for future in as_completed(futures):
                pno = futures[future]
                try:
                    page_responses[pno] = future.result()
                except Exception as exc:
                    logger.error("[KAG] Extraction page %s échouée : %s", pno, exc)
                    page_responses[pno] = None

        total_entities = 0
        total_relations = 0
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
            total_entities += ents
            total_relations += rels
            pages_ok += 1

        session.commit()

        logger.info(
            "[KAG] Extraction terminée document_id=%s pages=%s/%s entities=%s relations=%s model=%s",
            document_id,
            pages_ok,
            len(page_numbers),
            total_entities,
            total_relations,
            _kag_extraction_model(),
        )
        return {
            "entities": total_entities,
            "relations": total_relations,
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
