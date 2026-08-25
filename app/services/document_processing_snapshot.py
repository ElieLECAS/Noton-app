"""Agrégats lecture seule : chunks et KAG — pour UI et réponses stop/skip."""

from __future__ import annotations

from typing import Any, Literal, Optional

from sqlalchemy import func, select
from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

ReadinessLabel = Literal["none", "ready"]


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


def _readiness_label(chunk_count: int, leaf_chunk_count: int) -> ReadinessLabel:
    if chunk_count == 0:
        return "none"
    if leaf_chunk_count == 0:
        return "none"
    return "ready"


def build_document_processing_snapshot(
    session: Session, document_id: int
) -> dict[str, Any]:
    """Compteurs et libellé de maturité pour un document."""
    doc = session.get(Document, document_id)

    chunk_count = _to_int_scalar(
        session.exec(
            select(func.count()).select_from(DocumentChunk).where(
                DocumentChunk.document_id == document_id
            )
        ).one()
    )

    leaf_chunk_count = _to_int_scalar(
        session.exec(
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.is_leaf == True,  # noqa: E712
            )
        ).one()
    )

    readiness = _readiness_label(chunk_count, leaf_chunk_count)

    result: dict[str, Any] = {
        "document_id": document_id,
        "has_chunks": chunk_count > 0,
        "chunk_count": chunk_count,
        "leaf_chunk_count": leaf_chunk_count,
        "readiness_label": readiness,
        "current_page": doc.phase_status_json.get("current_page") if doc and doc.phase_status_json else None,
        "total_pages": doc.phase_status_json.get("total_pages") if doc and doc.phase_status_json else None,
    }

    return result


def build_document_diagnostic(session: Session, document_id: int) -> dict[str, Any]:
    """Checks pour le bouton Diagnostiquer (pipeline indexation)."""
    snap = build_document_processing_snapshot(session, document_id)
    issues: list[str] = []
    if snap["chunk_count"] == 0:
        issues.append("Aucun chunk indexé.")
    elif snap["leaf_chunk_count"] == 0:
        issues.append("Chunks présents mais aucune feuille sémantique (semantic_leaf).")
    if settings.KAG_ENABLED and snap.get("knowledge_entity_count", 0) == 0 and snap["chunk_count"] > 0:
        issues.append("Aucune entité KAG extraite pour ce document.")
    return {
        **snap,
        "checks_ok": len(issues) == 0,
        "issues": issues,
    }
