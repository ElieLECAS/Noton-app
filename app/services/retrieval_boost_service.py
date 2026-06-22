"""Boosts souples post-retrieval basés sur les signaux extraits (catégories, source, matériau, entités)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.query_signals_schemas import LightweightQuerySignals

logger = logging.getLogger(__name__)


def _get_chunk_categories(session: Session, chunk_id: Optional[int]) -> List[str]:
    if not chunk_id:
        return []
    chunk = session.get(DocumentChunk, chunk_id)
    if not chunk or not chunk.metadata_json:
        return []
    categories = chunk.metadata_json.get("categories", [])
    return categories if isinstance(categories, list) else []


def _get_document_source(session: Session, document_id: int) -> Optional[str]:
    doc = session.get(Document, document_id)
    return doc.source if doc else None


def _get_document_materials(session: Session, document_id: int) -> List[str]:
    doc = session.get(Document, document_id)
    return list(doc.materials or []) if doc else []


def _chunk_id_from_passage(passage: Dict[str, Any]) -> Optional[int]:
    chunk_id = passage.get("chunk_id")
    if chunk_id is not None:
        try:
            return int(chunk_id)
        except (TypeError, ValueError):
            pass
    return None


def _document_id_from_passage(passage: Dict[str, Any]) -> Optional[int]:
    doc_id = passage.get("document_id")
    if doc_id is not None:
        try:
            return int(doc_id)
        except (TypeError, ValueError):
            pass
    return None


def apply_soft_boosts_to_passages(
    session: Session,
    passages: List[Dict[str, Any]],
    signals: LightweightQuerySignals,
) -> List[Dict[str, Any]]:
    """
    Applique des boosts souples sur les scores des passages retournés par le retriever.
    """
    if not passages or not signals:
        return passages

    category_boost = settings.RETRIEVAL_CATEGORY_BOOST
    source_boost_max = settings.RETRIEVAL_SOURCE_BOOST_MAX
    material_boost = settings.RETRIEVAL_MATERIAL_BOOST
    entity_boost = settings.RETRIEVAL_ENTITY_BOOST

    refined: List[Dict[str, Any]] = []
    for passage in passages:
        p_copy = dict(passage)
        score = float(p_copy.get("score", 0.0))
        boost = 0.0

        chunk_id = _chunk_id_from_passage(p_copy)
        doc_id = _document_id_from_passage(p_copy)

        if doc_id:
            doc_source = _get_document_source(session, doc_id)
            if doc_source:
                p_copy["source"] = doc_source

        if signals.inferred_categories and chunk_id:
            chunk_categories = _get_chunk_categories(session, chunk_id)
            matched = set(signals.inferred_categories) & {c.lower() for c in chunk_categories}
            boost += len(matched) * category_boost

        if signals.primary_source and doc_id:
            doc_source = _get_document_source(session, doc_id)
            if doc_source and doc_source.lower() == signals.primary_source.lower():
                boost += source_boost_max * signals.confidence

        if signals.material_hint and doc_id:
            doc_materials = _get_document_materials(session, doc_id)
            if signals.material_hint.lower() in [m.lower() for m in doc_materials]:
                boost += material_boost

        retrieval_sources = p_copy.get("retrieval_sources") or []
        if "kag" in retrieval_sources and signals.entity_texts:
            boost += entity_boost * min(len(signals.entity_texts), 3)

        if boost > 0:
            p_copy["score"] = score + boost
            p_copy["retrieval_boost"] = round(boost, 4)

        refined.append(p_copy)

    refined.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    if any(p.get("retrieval_boost") for p in refined):
        logger.info(
            "[retrieval_boost] %d passage(s) boosté(s), top score=%.4f",
            sum(1 for p in refined if p.get("retrieval_boost")),
            float(refined[0].get("score", 0)) if refined else 0,
        )
    return refined
