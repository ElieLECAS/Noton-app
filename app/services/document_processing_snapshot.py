"""Agrégats lecture seule : chunks et embeddings — pour UI et réponses stop/skip."""

from __future__ import annotations

from typing import Any, Literal, Optional

from sqlalchemy import func, select
from sqlmodel import Session

from app.models.document import Document
from app.models.document_chunk import DocumentChunk

ReadinessLabel = Literal["none", "embeddings_only", "ready"]


def _to_int_scalar(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    mapping = getattr(value, "_mapping", None)
    if mapping:
        first = next(iter(mapping.values()), 0)
        return int(first or 0)
    if isinstance(value, (tuple, list)):
        return int((value[0] if value else 0) or 0)
    return int(value or 0)


def _readiness_label(chunk_count: int, leaf_chunks_with_embedding: int) -> ReadinessLabel:
    if chunk_count == 0:
        return "none"
    if leaf_chunks_with_embedding == 0:
        return "none"
    return "ready"


def build_document_processing_snapshot(
    session: Session, document_id: int
) -> dict[str, Any]:
    """Compteurs et libellé de maturité pour un document."""
    doc = session.get(Document, document_id)
    has_doc_embedding = bool(
        doc is not None and doc.embedding is not None and len(doc.embedding or []) > 0
    )

    chunk_count = _to_int_scalar(
        session.exec(
            select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            )
        ).one()
    )

    leaves_with_emb = _to_int_scalar(
        session.exec(
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding.isnot(None),
            )
        ).one()
    )

    readiness = _readiness_label(chunk_count, leaves_with_emb)

    return {
        "document_id": document_id,
        "has_chunks": chunk_count > 0,
        "chunk_count": chunk_count,
        "has_document_embedding": has_doc_embedding,
        "chunks_with_embedding_count": leaves_with_emb,
        "readiness_label": readiness,
    }


def build_document_diagnostic(session: Session, document_id: int) -> dict[str, Any]:
    """Checks pour le bouton Diagnostiquer (pipeline indexation)."""
    snap = build_document_processing_snapshot(session, document_id)
    issues: list[str] = []
    if snap["chunk_count"] == 0:
        issues.append("Aucun chunk indexé.")
    elif snap["chunks_with_embedding_count"] == 0:
        issues.append("Chunks présents mais aucun embedding vectoriel sur les feuilles.")
    return {
        **snap,
        "checks_ok": len(issues) == 0,
        "issues": issues,
    }
