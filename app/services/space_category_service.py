"""Catégories KAG agrégées par espace (liste, pages, détail page)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_space import DocumentSpace
from app.services.page_retrieval_service import (
    CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
    build_consolidated_page_text,
    load_l1_chunks_for_page,
)


def _empty_categories_payload(space_id: int) -> Dict[str, Any]:
    return {
        "space_id": space_id,
        "category_count": 0,
        "categories": [],
        "status": "disabled" if not settings.KAG_ENABLED else "empty",
    }


def get_space_categories(session: Session, space_id: int) -> Dict[str, Any]:
    """Liste des catégories présentes dans un espace avec compteurs de pages."""
    if not settings.KAG_ENABLED:
        return _empty_categories_payload(space_id)

    rows = session.execute(
        text(
            """
            SELECT
                dc.id AS category_id,
                dc.slug,
                dc.label,
                COUNT(DISTINCT ccr.chunk_id) AS chunk_count,
                COUNT(DISTINCT (ccr.document_id, ccr.page_no)) AS page_count,
                COUNT(DISTINCT ccr.document_id) AS document_count,
                COALESCE(MAX(ccr.confidence), 0.0) AS max_confidence
            FROM chunkcategoryrelation ccr
            INNER JOIN documentcategory dc ON dc.id = ccr.category_id
            WHERE ccr.space_id = :space_id
            GROUP BY dc.id, dc.slug, dc.label
            ORDER BY dc.label
            """
        ),
        {"space_id": space_id},
    ).all()

    categories = [
        {
            "category_id": int(category_id),
            "slug": slug,
            "label": label,
            "chunk_count": int(chunk_count or 0),
            "page_count": int(page_count or 0),
            "document_count": int(document_count or 0),
            "max_confidence": float(max_confidence or 0.0),
        }
        for category_id, slug, label, chunk_count, page_count, document_count, max_confidence in rows
    ]

    return {
        "space_id": space_id,
        "category_count": len(categories),
        "categories": categories,
        "status": "ok" if categories else "empty",
    }


def _list_category_pages(
    session: Session,
    space_id: int,
    category_id: int,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT
                ccr.document_id,
                d.title AS document_title,
                ccr.page_no,
                COUNT(DISTINCT ccr.chunk_id) AS chunk_count,
                BOOL_OR(
                    d.source_file_path IS NOT NULL AND d.source_file_path <> ''
                ) AS has_source_file
            FROM chunkcategoryrelation ccr
            INNER JOIN document d ON d.id = ccr.document_id
            WHERE ccr.space_id = :space_id
              AND ccr.category_id = :category_id
            GROUP BY ccr.document_id, d.title, ccr.page_no
            ORDER BY d.title, ccr.page_no
            """
        ),
        {"space_id": space_id, "category_id": category_id},
    ).all()

    pages: List[Dict[str, Any]] = []
    for document_id, document_title, page_no, chunk_count, has_source_file in rows:
        pages.append(
            {
                "document_id": int(document_id),
                "document_title": document_title or "",
                "page_no": int(page_no or 0),
                "chunk_count": int(chunk_count or 0),
                "has_source_file": bool(has_source_file),
            }
        )
    return pages


def get_space_category_pages(
    session: Session,
    space_id: int,
    category_id: int,
) -> Optional[Dict[str, Any]]:
    """Pages d'un espace contenant une catégorie donnée."""
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

    pages = _list_category_pages(session, space_id, category_id)
    cat_id, slug, label = category
    return {
        "space_id": space_id,
        "category": {
            "category_id": int(cat_id),
            "slug": slug,
            "label": label,
        },
        "page_count": len(pages),
        "pages": pages,
    }


def _document_in_space(session: Session, space_id: int, document_id: int) -> bool:
    assoc = session.exec(
        select(DocumentSpace).where(
            DocumentSpace.space_id == space_id,
            DocumentSpace.document_id == document_id,
        )
    ).first()
    return assoc is not None


def _chunk_category_map_for_page(
    session: Session,
    space_id: int,
    category_id: int,
    document_id: int,
    page_no: int,
) -> Dict[int, float]:
    rows = session.execute(
        text(
            """
            SELECT ccr.chunk_id, ccr.confidence
            FROM chunkcategoryrelation ccr
            WHERE ccr.space_id = :space_id
              AND ccr.category_id = :category_id
              AND ccr.document_id = :document_id
              AND ccr.page_no = :page_no
            """
        ),
        {
            "space_id": space_id,
            "category_id": category_id,
            "document_id": document_id,
            "page_no": page_no,
        },
    ).all()
    return {int(chunk_id): float(confidence or 0.0) for chunk_id, confidence in rows}


def _load_enrichment_chunks_for_category_page(
    session: Session,
    space_id: int,
    category_id: int,
    document_id: int,
    page_no: int,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        text(
            """
            SELECT dc.id, dc.content, dc.chunk_index, dc.metadata_json, dc.metadata_, ccr.confidence
            FROM chunkcategoryrelation ccr
            INNER JOIN documentchunk dc ON dc.id = ccr.chunk_id
            WHERE ccr.space_id = :space_id
              AND ccr.category_id = :category_id
              AND ccr.document_id = :document_id
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
            "space_id": space_id,
            "category_id": category_id,
            "document_id": document_id,
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
    document_id: int,
    page_no: int,
) -> Dict[str, Any]:
    current_index = None
    for idx, page in enumerate(pages):
        if page["document_id"] == document_id and page["page_no"] == page_no:
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


def get_space_category_page_detail(
    session: Session,
    space_id: int,
    category_id: int,
    document_id: int,
    page_no: int,
) -> Optional[Dict[str, Any]]:
    """Détail d'une page : chunks texte + surlignage catégorie + navigation."""
    if not settings.KAG_ENABLED:
        return None

    if not _document_in_space(session, space_id, document_id):
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

    pages = _list_category_pages(session, space_id, category_id)
    if not any(p["document_id"] == document_id and p["page_no"] == page_no for p in pages):
        return None

    document = session.get(Document, document_id)
    if not document:
        return None

    chunk_confidence = _chunk_category_map_for_page(
        session, space_id, category_id, document_id, page_no
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
        session, space_id, category_id, document_id, page_no
    )

    has_source_file = bool(
        document.source_file_path and os.path.exists(document.source_file_path)
    )
    cat_id, slug, label = category_row

    return {
        "space_id": space_id,
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
        "navigation": _build_navigation(pages, document_id, page_no),
    }
