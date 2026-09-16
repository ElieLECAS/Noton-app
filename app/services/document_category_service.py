"""Catégories KAG agrégées par document (pages, détail page)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.services.page_retrieval_service import (
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



def _page_display_text(document_id, page_no, chunks) -> tuple:
    """(texte de la page, vient-il du markdown augmenté ?).

    Quand le document a été retranscrit, c'est cette page-là qu'on montre à côté du PDF :
    une page entière et fidèle, au lieu des fragments de l'extraction automatique.
    """
    from app.services.page_markdown_service import page_section
    from app.services.page_retrieval_service import build_consolidated_page_text

    try:
        md = page_section(int(document_id), int(page_no))
    except Exception:  # noqa: BLE001 - l'affichage ne doit jamais tomber là-dessus
        md = None
    if md:
        return md, True
    return build_consolidated_page_text(chunks), False


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
                "confidence": chunk_confidence.get(chunk_id),
            }
        )


    has_source_file = bool(
        document.source_file_path and os.path.exists(document.source_file_path)
    )
    cat_id, slug, label = category_row

    _texte_page, _md_augmente = _page_display_text(document_id, page_no, source_chunks_for_consolidated)
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
        "consolidated_markdown": _texte_page,
        "markdown_augmente": _md_augmente,
        "navigation": _build_navigation(pages, page_no),
    }
