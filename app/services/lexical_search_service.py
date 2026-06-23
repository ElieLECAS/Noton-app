"""Recherche lexicale « contient » (ILIKE) par mot-clé.

Renvoie des pages/chunks au même schéma que les services catégories
(`document_category_service` / `space_category_service`) afin de réutiliser
telle quelle la modale de détail (PDF + texte + navigation) côté frontend.

Périmètre de recherche : texte source du PDF (`semantic_leaf`) + synthèses IA
(`contextual_enrichment`). Insensible à la casse, sous-chaîne.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlmodel import Session

from app.models.document import Document
from app.services.page_retrieval_service import (
    _page_no_sql_expr,
    build_consolidated_page_text,
    get_space_document_ids,
    load_enrichment_chunks_for_pages,
    load_l1_chunks_for_page,
)

MIN_QUERY_LEN = 2

_CONTENT_TYPE_FILTER = (
    "COALESCE(dc.metadata_json->>'content_type', dc.metadata_->>'content_type', '') "
    "IN ('semantic_leaf', 'contextual_enrichment')"
)


def _like_pattern(query: str) -> str:
    """Échappe les méta-caractères ILIKE et entoure de `%` pour un « contient »."""
    esc = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


def _normalize_query(query: Optional[str]) -> str:
    return (query or "").strip()


def _contains(content: Optional[str], needle_cf: str) -> bool:
    return needle_cf in (content or "").casefold()


def _category_ref(query: str) -> Dict[str, Any]:
    """Catégorie synthétique pour réutiliser le rendu existant (titre = mot-clé)."""
    return {
        "category_id": 0,
        "slug": query,
        "label": f"Mot-clé : « {query} »",
    }


def _enrichment_item(chunk, query_cf: str) -> Optional[Dict[str, Any]]:
    content = (chunk.content or chunk.text or "").strip()
    if not content or not _contains(content, query_cf):
        return None
    meta = dict(chunk.metadata_json or chunk.metadata_ or {})
    return {
        "chunk_id": int(chunk.id) if chunk.id is not None else None,
        "chunk_index": chunk.chunk_index,
        "theme": meta.get("theme"),
        "category_slug": meta.get("category_slug"),
        "source_page": meta.get("source_page") or meta.get("page_no"),
        "source_pages": meta.get("source_pages") or [],
        "content": content,
        "is_enrichment": True,
        "confidence": None,
    }


def _source_chunk_item(chunk, query_cf: str) -> Optional[Dict[str, Any]]:
    content = (chunk.content or chunk.text or "").strip()
    if not content or not _contains(content, query_cf):
        return None
    meta = dict(chunk.metadata_json or chunk.metadata_ or {})
    return {
        "chunk_id": int(chunk.id) if chunk.id is not None else None,
        "chunk_index": chunk.chunk_index,
        "heading": meta.get("heading") or meta.get("parent_heading"),
        "step_number": meta.get("step_number"),
        "section_type": meta.get("section_type") or meta.get("content_type"),
        "content": content,
        "in_category": True,
        "is_enrichment": False,
        "confidence": None,
    }


# ---------------------------------------------------------------------------
# Document-scoped (library)
# ---------------------------------------------------------------------------


def _list_document_pages(
    session: Session,
    document_id: int,
    query: str,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        text(
            f"""
            SELECT page_no, COUNT(*) AS chunk_count
            FROM (
                SELECT {_page_no_sql_expr("dc")} AS page_no
                FROM documentchunk dc
                WHERE dc.document_id = :document_id
                  AND dc.is_leaf = true
                  AND dc.content ILIKE :pat ESCAPE '\\'
                  AND {_CONTENT_TYPE_FILTER}
            ) sub
            WHERE page_no IS NOT NULL
            GROUP BY page_no
            ORDER BY page_no
            """
        ),
        {"document_id": document_id, "pat": _like_pattern(query)},
    ).all()
    return [
        {"page_no": int(page_no), "chunk_count": int(chunk_count or 0)}
        for page_no, chunk_count in rows
    ]


def _build_document_navigation(
    pages: List[Dict[str, Any]],
    page_no: int,
) -> Dict[str, Any]:
    current_index = next(
        (idx for idx, p in enumerate(pages) if p["page_no"] == page_no), None
    )
    prev_page = pages[current_index - 1] if current_index not in (None, 0) else None
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


def search_document_pages(
    session: Session,
    document_id: int,
    query: str,
) -> Dict[str, Any]:
    """Pages d'un document contenant le mot-clé."""
    q = _normalize_query(query)
    if len(q) < MIN_QUERY_LEN:
        return {"document_id": document_id, "query": q, "page_count": 0, "pages": []}

    pages = _list_document_pages(session, document_id, q)
    return {
        "document_id": document_id,
        "query": q,
        "page_count": len(pages),
        "pages": pages,
    }


def get_document_search_page_detail(
    session: Session,
    document_id: int,
    query: str,
    page_no: int,
) -> Optional[Dict[str, Any]]:
    """Détail d'une page : chunks (source + IA) contenant le mot-clé + navigation."""
    q = _normalize_query(query)
    if len(q) < MIN_QUERY_LEN:
        return None

    pages = _list_document_pages(session, document_id, q)
    if not any(p["page_no"] == page_no for p in pages):
        return None

    document = session.get(Document, document_id)
    if not document:
        return None

    q_cf = q.casefold()
    source_chunks = load_l1_chunks_for_page(session, document_id, page_no)
    chunk_items: List[Dict[str, Any]] = []
    matched_for_consolidated: List = []
    for chunk in source_chunks:
        item = _source_chunk_item(chunk, q_cf)
        if item:
            chunk_items.append(item)
            matched_for_consolidated.append(chunk)

    enrichment_items: List[Dict[str, Any]] = []
    for chunk in load_enrichment_chunks_for_pages(session, document_id, [page_no]):
        item = _enrichment_item(chunk, q_cf)
        if item:
            enrichment_items.append(item)

    has_source_file = bool(
        document.source_file_path and os.path.exists(document.source_file_path)
    )

    return {
        "document_id": document_id,
        "query": q,
        "category": _category_ref(q),
        "document": {
            "document_id": document.id,
            "title": document.title,
            "has_source_file": has_source_file,
        },
        "page_no": page_no,
        "chunks": chunk_items,
        "enrichment_chunks": enrichment_items,
        "consolidated_markdown": build_consolidated_page_text(matched_for_consolidated),
        "navigation": _build_document_navigation(pages, page_no),
    }


# ---------------------------------------------------------------------------
# Space-scoped
# ---------------------------------------------------------------------------


def _list_space_pages(
    session: Session,
    space_id: int,
    query: str,
) -> List[Dict[str, Any]]:
    rows = session.execute(
        text(
            f"""
            SELECT document_id, document_title, page_no,
                   COUNT(*) AS chunk_count,
                   bool_or(has_src) AS has_source_file
            FROM (
                SELECT dc.document_id,
                       d.title AS document_title,
                       {_page_no_sql_expr("dc")} AS page_no,
                       (d.source_file_path IS NOT NULL AND d.source_file_path <> '') AS has_src
                FROM documentchunk dc
                INNER JOIN document d ON d.id = dc.document_id
                INNER JOIN document_space ds ON ds.document_id = dc.document_id
                WHERE ds.space_id = :space_id
                  AND dc.is_leaf = true
                  AND dc.content ILIKE :pat ESCAPE '\\'
                  AND {_CONTENT_TYPE_FILTER}
            ) sub
            WHERE page_no IS NOT NULL
            GROUP BY document_id, document_title, page_no
            ORDER BY document_title, page_no
            """
        ),
        {"space_id": space_id, "pat": _like_pattern(query)},
    ).all()
    return [
        {
            "document_id": int(document_id),
            "document_title": document_title or "",
            "page_no": int(page_no),
            "chunk_count": int(chunk_count or 0),
            "has_source_file": bool(has_source_file),
        }
        for document_id, document_title, page_no, chunk_count, has_source_file in rows
    ]


def _build_space_navigation(
    pages: List[Dict[str, Any]],
    document_id: int,
    page_no: int,
) -> Dict[str, Any]:
    current_index = next(
        (
            idx
            for idx, p in enumerate(pages)
            if p["document_id"] == document_id and p["page_no"] == page_no
        ),
        None,
    )
    prev_page = pages[current_index - 1] if current_index not in (None, 0) else None
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


def search_space_pages(
    session: Session,
    space_id: int,
    query: str,
) -> Dict[str, Any]:
    """Pages (tous documents de l'espace) contenant le mot-clé."""
    q = _normalize_query(query)
    if len(q) < MIN_QUERY_LEN:
        return {"space_id": space_id, "query": q, "page_count": 0, "pages": []}

    pages = _list_space_pages(session, space_id, q)
    return {
        "space_id": space_id,
        "query": q,
        "page_count": len(pages),
        "pages": pages,
    }


def get_space_search_page_detail(
    session: Session,
    space_id: int,
    query: str,
    document_id: int,
    page_no: int,
) -> Optional[Dict[str, Any]]:
    """Détail d'une page d'espace : chunks (source + IA) contenant le mot-clé."""
    q = _normalize_query(query)
    if len(q) < MIN_QUERY_LEN:
        return None

    if document_id not in set(get_space_document_ids(session, space_id)):
        return None

    pages = _list_space_pages(session, space_id, q)
    if not any(
        p["document_id"] == document_id and p["page_no"] == page_no for p in pages
    ):
        return None

    document = session.get(Document, document_id)
    if not document:
        return None

    q_cf = q.casefold()
    source_chunks = load_l1_chunks_for_page(session, document_id, page_no)
    chunk_items: List[Dict[str, Any]] = []
    matched_for_consolidated: List = []
    for chunk in source_chunks:
        item = _source_chunk_item(chunk, q_cf)
        if item:
            chunk_items.append(item)
            matched_for_consolidated.append(chunk)

    enrichment_items: List[Dict[str, Any]] = []
    for chunk in load_enrichment_chunks_for_pages(session, document_id, [page_no]):
        item = _enrichment_item(chunk, q_cf)
        if item:
            enrichment_items.append(item)

    has_source_file = bool(
        document.source_file_path and os.path.exists(document.source_file_path)
    )

    return {
        "space_id": space_id,
        "query": q,
        "category": _category_ref(q),
        "document": {
            "document_id": document.id,
            "title": document.title,
            "has_source_file": has_source_file,
        },
        "page_no": page_no,
        "chunks": chunk_items,
        "enrichment_chunks": enrichment_items,
        "consolidated_markdown": build_consolidated_page_text(matched_for_consolidated),
        "navigation": _build_space_navigation(pages, document_id, page_no),
    }
