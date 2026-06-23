"""Catégories KAG agrégées par document (pages, détail page)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.services.page_retrieval_service import (
    CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
    build_consolidated_page_text,
    load_l1_chunks_for_page,
)


def _list_category_pages(
    session: Session,
    document_id: int,
    category_id: int,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT
                ccr.page_no,
                COUNT(DISTINCT ccr.chunk_id) AS chunk_count
            FROM chunkcategoryrelation ccr
            WHERE ccr.document_id = :document_id
              AND ccr.category_id = :category_id
            GROUP BY ccr.page_no
            ORDER BY ccr.page_no
            """
        ),
        {"document_id": document_id, "category_id": category_id},
    ).all()

    return [
        {
            "page_no": int(page_no or 0),
            "chunk_count": int(chunk_count or 0),
        }
        for page_no, chunk_count in rows
    ]


def get_document_category_pages(
    session: Session,
    document_id: int,
    category_id: int,
) -> Optional[Dict[str, Any]]:
    """Pages d'un document contenant une catégorie donnée."""
    if not settings.KAG_ENABLED:
        return None

    category = session.execute(
        text(
            """
            SELECT id, slug, label
            FROM documentcategory
            WHERE id = :category_id
            """
        ),
        {"category_id": category_id},
    ).first()
    if not category:
        return None

    pages = _list_category_pages(session, document_id, category_id)
    cat_id, slug, label = category
    return {
        "document_id": document_id,
        "category": {
            "category_id": int(cat_id),
            "slug": slug,
            "label": label,
        },
        "page_count": len(pages),
        "pages": pages,
    }


def _chunk_category_map_for_page(
    session: Session,
    document_id: int,
    category_id: int,
    page_no: int,
) -> Dict[int, float]:
    rows = session.execute(
        text(
            """
            SELECT ccr.chunk_id, ccr.confidence
            FROM chunkcategoryrelation ccr
            WHERE ccr.document_id = :document_id
              AND ccr.category_id = :category_id
              AND ccr.page_no = :page_no
            """
        ),
        {
            "document_id": document_id,
            "category_id": category_id,
            "page_no": page_no,
        },
    ).all()
    return {int(chunk_id): float(confidence or 0.0) for chunk_id, confidence in rows}


def _load_enrichment_chunks_for_category_page(
    session: Session,
    document_id: int,
    category_id: int,
    page_no: int,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT dc.id, dc.content, dc.chunk_index, dc.metadata_json, dc.metadata_, ccr.confidence
            FROM chunkcategoryrelation ccr
            INNER JOIN documentchunk dc ON dc.id = ccr.chunk_id
            WHERE ccr.document_id = :document_id
              AND ccr.category_id = :category_id
              AND ccr.page_no = :page_no
              AND COALESCE(
                  dc.metadata_json->>'content_type',
                  dc.metadata_->>'content_type',
                  ''
              ) = :content_type
            ORDER BY dc.chunk_index
            """
        ),
        {
            "document_id": document_id,
            "category_id": category_id,
            "page_no": page_no,
            "content_type": CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
        },
    ).all()

    items: List[Dict[str, Any]] = []
    for chunk_id, content, chunk_index, metadata_json, metadata_, confidence in rows:
        meta = dict(metadata_json or metadata_ or {})
        text_content = (content or "").strip()
        if not text_content:
            continue
        items.append(
            {
                "chunk_id": int(chunk_id),
                "chunk_index": chunk_index,
                "theme": meta.get("theme"),
                "category_slug": meta.get("category_slug"),
                "source_page": meta.get("source_page") or meta.get("page_no"),
                "source_pages": meta.get("source_pages") or [],
                "content": text_content,
                "is_enrichment": True,
                "confidence": float(confidence or 0.0),
            }
        )
    return items


def _build_navigation(
    pages: List[Dict[str, Any]],
    page_no: int,
) -> Dict[str, Any]:
    current_index = None
    for idx, page in enumerate(pages):
        if page["page_no"] == page_no:
            current_index = idx
            break

    prev_page = pages[current_index - 1] if current_index is not None and current_index > 0 else None
    next_page = (
        pages[current_index + 1]
        if current_index is not None and current_index < len(pages) - 1
        else None
    )

    return {
        "current_index": current_index,
        "total": len(pages),
        "prev": prev_page,
        "next": next_page,
    }


def get_document_category_page_detail(
    session: Session,
    document_id: int,
    category_id: int,
    page_no: int,
) -> Optional[Dict[str, Any]]:
    """Détail d'une page document : chunks texte + enrichissement contextuel + navigation."""
    if not settings.KAG_ENABLED:
        return None

    category_row = session.execute(
        text(
            """
            SELECT id, slug, label
            FROM documentcategory
            WHERE id = :category_id
            """
        ),
        {"category_id": category_id},
    ).first()
    if not category_row:
        return None

    pages = _list_category_pages(session, document_id, category_id)
    if not any(p["page_no"] == page_no for p in pages):
        return None

    document = session.get(Document, document_id)
    if not document:
        return None

    chunk_confidence = _chunk_category_map_for_page(
        session, document_id, category_id, page_no
    )
    l1_chunks = load_l1_chunks_for_page(session, document_id, page_no)

    chunk_items: List[Dict[str, Any]] = []
    source_chunks_for_consolidated: List = []
    for chunk in l1_chunks:
        chunk_id = int(chunk.id) if chunk.id is not None else None
        if chunk_id is None or chunk_id not in chunk_confidence:
            continue

        meta = dict(chunk.metadata_json or chunk.metadata_ or {})
        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue
        source_chunks_for_consolidated.append(chunk)
        chunk_items.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk.chunk_index,
                "heading": meta.get("heading") or meta.get("parent_heading"),
                "step_number": meta.get("step_number"),
                "section_type": meta.get("section_type") or meta.get("content_type"),
                "content": content,
                "in_category": True,
                "is_enrichment": False,
                "confidence": chunk_confidence.get(chunk_id),
            }
        )

    enrichment_items = _load_enrichment_chunks_for_category_page(
        session, document_id, category_id, page_no
    )

    has_source_file = bool(
        document.source_file_path and os.path.exists(document.source_file_path)
    )
    cat_id, slug, label = category_row

    return {
        "document_id": document_id,
        "category": {
            "category_id": int(cat_id),
            "slug": slug,
            "label": label,
        },
        "document": {
            "document_id": document.id,
            "title": document.title,
            "has_source_file": has_source_file,
        },
        "page_no": page_no,
        "chunks": chunk_items,
        "enrichment_chunks": enrichment_items,
        "consolidated_markdown": build_consolidated_page_text(source_chunks_for_consolidated),
        "navigation": _build_navigation(pages, page_no),
    }
