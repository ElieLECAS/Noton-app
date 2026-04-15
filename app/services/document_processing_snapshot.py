"""Agrégats lecture seule : chunks, embeddings, KAG — pour UI et réponses stop/skip."""

from __future__ import annotations

from typing import Any, Literal, Optional

from sqlalchemy import func, select
from sqlmodel import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.chunk_entity_relation import ChunkEntityRelation
from app.models.entity_entity_relation import EntityEntityRelation

ReadinessLabel = Literal["none", "embeddings_only", "kag_partial", "kag_ready"]


def _to_int_scalar(value: Any) -> int:
    """Normalise les retours SQLAlchemy/SQLModel (int, Row, tuple) en entier."""
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    # SQLAlchemy Row expose souvent _mapping pour les agrégats.
    mapping = getattr(value, "_mapping", None)
    if mapping:
        first = next(iter(mapping.values()), 0)
        return int(first or 0)
    if isinstance(value, (tuple, list)):
        return int((value[0] if value else 0) or 0)
    return int(value or 0)


def _readiness_label(
    chunk_count: int,
    leaf_chunks_with_embedding: int,
    distinct_entity_count: int,
    relation_count: int,
) -> ReadinessLabel:
    if chunk_count == 0:
        return "none"
    if leaf_chunks_with_embedding == 0:
        return "none"
    if distinct_entity_count == 0:
        return "embeddings_only"
    if relation_count == 0 or distinct_entity_count < 2:
        return "kag_partial"
    return "kag_ready"


def build_document_processing_snapshot(
    session: Session, document_id: int
) -> dict[str, Any]:
    """Compteurs et libellé de maturité pour un document (pas de logique métier lourde)."""
    doc = session.get(Document, document_id)
    has_doc_embedding = bool(
        doc is not None and doc.embedding is not None and len(doc.embedding or []) > 0
    )

    chunk_count = session.exec(
        select(func.count()).select_from(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        )
    ).one()
    chunk_count = _to_int_scalar(chunk_count)

    leaves_with_emb = session.exec(
        select(func.count())
        .select_from(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.embedding.isnot(None),
        )
    ).one()
    leaves_with_emb = _to_int_scalar(leaves_with_emb)

    subq_chunk_ids = select(DocumentChunk.id).where(
        DocumentChunk.document_id == document_id
    )
    if chunk_count == 0:
        distinct_entities = 0
        relation_count = 0
        chunks_with_entities = 0
    else:
        distinct_entities = session.exec(
            select(func.count(func.distinct(ChunkEntityRelation.entity_id)))
            .select_from(ChunkEntityRelation)
            .where(ChunkEntityRelation.chunk_id.in_(subq_chunk_ids))
        ).one()
        distinct_entities = _to_int_scalar(distinct_entities)

        relation_count = session.exec(
            select(func.count())
            .select_from(EntityEntityRelation)
            .where(EntityEntityRelation.source_chunk_id.in_(subq_chunk_ids))
        ).one()
        relation_count = _to_int_scalar(relation_count)

        chunks_with_entities = session.exec(
            select(func.count(func.distinct(ChunkEntityRelation.chunk_id)))
            .select_from(ChunkEntityRelation)
            .where(ChunkEntityRelation.chunk_id.in_(subq_chunk_ids))
        ).one()
        chunks_with_entities = _to_int_scalar(chunks_with_entities)

    readiness = _readiness_label(
        chunk_count,
        leaves_with_emb,
        distinct_entities,
        relation_count,
    )

    coverage = (
        float(chunks_with_entities) / float(chunk_count) if chunk_count else 0.0
    )

    return {
        "document_id": document_id,
        "has_chunks": chunk_count > 0,
        "chunk_count": chunk_count,
        "has_document_embedding": has_doc_embedding,
        "chunks_with_embedding_count": leaves_with_emb,
        "knowledge_entity_count": distinct_entities,
        "entity_relation_count": relation_count,
        "readiness_label": readiness,
        "chunks_with_entities_count": chunks_with_entities,
        "entity_chunk_coverage": round(coverage, 4),
    }


def build_document_diagnostic(session: Session, document_id: int) -> dict[str, Any]:
    """Checks pour le bouton Diagnostiquer (pipeline + KAG)."""
    snap = build_document_processing_snapshot(session, document_id)
    issues: list[str] = []
    if snap["chunk_count"] == 0:
        issues.append("Aucun chunk indexé.")
    elif snap["chunks_with_embedding_count"] == 0:
        issues.append("Chunks présents mais aucun embedding vectoriel sur les feuilles.")
    if snap["knowledge_entity_count"] == 0 and snap["chunk_count"] > 0:
        issues.append("Aucune entité KAG liée aux chunks de ce document.")
    if snap["entity_relation_count"] == 0 and snap["knowledge_entity_count"] > 0:
        issues.append("Entités présentes mais aucune relation au graphe (arêtes).")
    return {
        **snap,
        "checks_ok": len(issues) == 0,
        "issues": issues,
    }
