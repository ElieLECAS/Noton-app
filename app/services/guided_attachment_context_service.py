"""Contexte documentaire scopé aux pièces jointes d'un nœud d'arbre SAV (niveau 1).

Deux modes, choisis par la taille — jamais de recherche corpus entier :
- CAG : plage de pages attachée → texte intégral des pages (chunks feuilles ordonnés).
  Zéro recherche, déterministe : l'auteur a déjà curé la sélection.
- RAG scopé : document entier / trop gros → BM25 (tsvector) avec FILTRE DUR document_id.

⚠️ On ne filtre PAS par content_type : les chunks `page_raw_enriched` doivent être
inclus (piège historique du loader KAG — cf. mémoire kag-refonte-pipeline-2026-07).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text as sa_text
from sqlmodel import Session

logger = logging.getLogger(__name__)

# Budget CAG en caractères (~8-10k tokens) — en dur, pas de flag.
CAG_MAX_CHARS = 24000
SCOPED_RAG_K = 6


def fetch_attachment_text(
    session: Session,
    document_id: int,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    max_chars: int = CAG_MAX_CHARS,
) -> str:
    """Texte intégral des pages attachées (chunks feuilles, ordre page puis position)."""
    params: Dict[str, Any] = {"doc_id": document_id}
    page_filter = ""
    if page_start is not None:
        page_filter = (
            "AND (metadata_json->>'page_no') IS NOT NULL "
            "AND (metadata_json->>'page_no')::int BETWEEN :p_start AND :p_end "
        )
        params["p_start"] = int(page_start)
        params["p_end"] = int(page_end if page_end is not None else page_start)

    rows = session.execute(
        sa_text(
            f"""
            SELECT content, metadata_json->>'page_no' AS page_no
            FROM documentchunk
            WHERE document_id = :doc_id
              AND is_leaf = true
              {page_filter}
            ORDER BY NULLIF(metadata_json->>'page_no', '')::int NULLS LAST, chunk_index
            """
        ),
        params,
    ).all()

    parts: List[str] = []
    total = 0
    last_page: Optional[str] = None
    for content, page_no in rows:
        block = (content or "").strip()
        if not block:
            continue
        if page_no and page_no != last_page:
            block = f"[Page {page_no}]\n{block}"
            last_page = page_no
        if total + len(block) > max_chars:
            parts.append(block[: max(0, max_chars - total)])
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts).strip()


def _run_scoped_bm25_query(
    session: Session,
    document_ids: List[int],
    tsquery_sql: str,
    param_key: str,
    param_value: str,
    k: int,
) -> List[Any]:
    return session.execute(
        sa_text(
            f"""
            SELECT dc.id, dc.document_id, dc.content, dc.metadata_json,
                   ts_rank_cd(dc.tsv_content, {tsquery_sql}) AS rank
            FROM documentchunk dc
            WHERE dc.document_id = ANY(:doc_ids)
              AND dc.is_leaf = true
              AND dc.tsv_content IS NOT NULL
              AND dc.tsv_content @@ {tsquery_sql}
            ORDER BY rank DESC
            LIMIT :k
            """
        ),
        {"doc_ids": list(document_ids), "k": int(k), param_key: param_value},
    ).mappings().all()


def scoped_lexical_passages(
    session: Session,
    question: str,
    document_ids: List[int],
    k: int = SCOPED_RAG_K,
) -> List[Dict[str, Any]]:
    """RAG scopé : BM25 (tsvector) restreint aux documents attachés (filtre dur)."""
    if not document_ids or not question.strip():
        return []
    from app.services.page_retrieval_service import (
        _bm25_tsquery_fn,
        _build_bm25_or_tsquery,
        _extract_bm25_fallback_query,
        _normalize_bm25_query,
    )

    tsquery_fn = _bm25_tsquery_fn()
    normalized = _normalize_bm25_query(question.strip())
    rows = _run_scoped_bm25_query(
        session, document_ids, f"{tsquery_fn}('french', :q)", "q", normalized, k
    )

    # Repli OR : une question conversationnelle ne matche souvent rien en AND.
    if not rows:
        or_query = _extract_bm25_fallback_query(question)
        if or_query:
            terms = [t.strip() for t in or_query.replace(" OR ", "|").split("|") if t.strip()]
            or_tsquery = _build_bm25_or_tsquery(terms)
            if or_tsquery:
                rows = _run_scoped_bm25_query(
                    session, document_ids, "to_tsquery('french', :tsq)", "tsq", or_tsquery, k
                )

    passages: List[Dict[str, Any]] = []
    for row in rows:
        meta = row["metadata_json"] or {}
        passages.append(
            {
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "content": row["content"] or "",
                "page_no": meta.get("page_no"),
                "score": round(float(row["rank"] or 0.0), 4),
            }
        )
    return passages


def build_node_context(
    session: Session,
    attachments: List[Dict[str, Any]],
    question: str = "",
    max_chars: int = CAG_MAX_CHARS,
) -> Dict[str, Any]:
    """Contexte de génération depuis les pièces d'un nœud : CAG si ça tient dans le
    budget, sinon RAG scopé sur les documents attachés. Retourne {mode, text, sources}."""
    doc_attachments = [
        a for a in (attachments or []) if a.get("kind") in (None, "", "notice", "schema")
    ]
    if not doc_attachments:
        return {"mode": "none", "text": "", "sources": []}

    # CAG d'abord : concaténer les plages de pages attachées, budget partagé.
    share = max(2000, max_chars // max(1, len(doc_attachments)))
    blocks: List[str] = []
    sources: List[Dict[str, Any]] = []
    for att in doc_attachments:
        txt = fetch_attachment_text(
            session,
            int(att["document_id"]),
            att.get("page_start"),
            att.get("page_end"),
            max_chars=share,
        )
        if txt:
            title = att.get("document_title") or f"Document {att['document_id']}"
            blocks.append(f"=== {title} ===\n{txt}")
            sources.append(att)

    combined = "\n\n".join(blocks).strip()
    if combined and len(combined) <= max_chars:
        return {"mode": "cag", "text": combined, "sources": sources}

    # Trop gros (ou plages absentes → docs entiers) : RAG scopé sur ces documents.
    doc_ids = list({int(a["document_id"]) for a in doc_attachments})
    passages = scoped_lexical_passages(session, question or combined[:500], doc_ids)
    text = "\n\n".join(p["content"] for p in passages).strip()
    return {"mode": "scoped_rag", "text": text[:max_chars], "sources": passages}
