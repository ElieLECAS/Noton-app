"""
Recherche ColPali-only dans les espaces.

Pipeline :
1. Embed requête ColPali
2. MaxSim sur LanceDB (patches par page)
3. Retour métadonnées page (document_id, page_no, score) — le LLM reçoit les images PNG
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import text
from sqlmodel import Session

from app.services.space_service import get_space_by_id
from app.tracing import trace_run

logger = logging.getLogger(__name__)


def _merged_chunk_metadata(primary: Optional[dict], legacy: Optional[dict]) -> Dict:
    merged: Dict = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(primary, dict):
        merged.update(primary)
    return merged


def _parse_chunk_id_from_node(node: TextNode) -> Optional[int]:
    nid = getattr(node, "id_", None) or ""
    if isinstance(nid, str) and nid.startswith("chunk-"):
        try:
            return int(nid.split("-", 1)[1])
        except ValueError:
            return None
    return None


def _retrieve_colpali_pages(
    session: Session,
    space_id: int,
    query_text: str,
    candidate_k: int,
    document_filter: str = "all",
) -> List[NodeWithScore]:
    """Recherche ColPali MaxSim sur les pages indexées de l'espace."""
    from app.services.document_service_new import feedback_corrective_sql_filter
    from app.services.colpali_service import embed_query_colpali
    from app.services.lancedb_service import search_colpali_lancedb

    filter_clause = feedback_corrective_sql_filter(document_filter, "d")

    sql_docs = text(f"""
        SELECT DISTINCT d.id
        FROM document d
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          {filter_clause}
    """)
    doc_ids = [row[0] for row in session.execute(sql_docs, {"space_id": space_id})]
    if not doc_ids:
        logger.info("ColPali (space): aucun document dans l'espace %s", space_id)
        return []

    query_token_embeddings = embed_query_colpali(query_text)
    search_results = search_colpali_lancedb(query_token_embeddings, doc_ids, limit=candidate_k)
    if not search_results:
        return []

    chunk_ids = [row["id"] for row in search_results]
    sql_chunks = text("""
        SELECT
            dc.id,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title,
            d.source AS document_source
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.id IN :chunk_ids
    """)
    chunk_rows = session.execute(sql_chunks, {"chunk_ids": tuple(chunk_ids)}).all()
    rows_map = {row.id: row for row in chunk_rows}

    nodes: List[NodeWithScore] = []
    for row_lancedb in search_results:
        chunk_id = row_lancedb["id"]
        row = rows_map.get(chunk_id)
        if not row:
            continue

        distance = float(row_lancedb.get("_distance", 1.0))
        similarity_score = 1.0 - distance

        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        metadata.setdefault("document_id", row.document_id)
        metadata.setdefault("document_title", row.document_title or "Document sans titre")
        metadata.setdefault("chunk_index", row.chunk_index)
        if getattr(row, "document_source", None):
            metadata.setdefault("source", row.document_source)

        node = TextNode(
            id_=f"chunk-{row.id}",
            text="",
            metadata=metadata,
        )
        nodes.append(NodeWithScore(node=node, score=similarity_score))

    logger.info("ColPali (space): %d pages (limit=%d)", len(nodes), candidate_k)
    return nodes


def _format_colpali_pages(nodes: List[NodeWithScore], k: int) -> List[Dict[str, Any]]:
    """
    Formate les résultats ColPali en passages légers (métadonnées page uniquement).
    Pas de texte chunk, pas d'augmentation parent/feuille.
    """
    passages: List[Dict[str, Any]] = []
    seen_pages: set = set()

    for nws in nodes:
        meta = dict(nws.node.metadata or {})
        doc_id = meta.get("document_id")
        page_no = meta.get("page_no") or meta.get("page_start")
        if doc_id is None or page_no is None:
            continue

        page_key = (doc_id, int(page_no))
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)

        chunk_id = _parse_chunk_id_from_node(nws.node)
        passages.append(
            {
                "document_id": doc_id,
                "document_title": meta.get("document_title") or "Document sans titre",
                "page_no": int(page_no),
                "page_start": int(page_no),
                "page_end": int(page_no),
                "chunk_id": chunk_id,
                "chunk_index": int(meta.get("chunk_index") or 0),
                "score": float(nws.score),
                "source": meta.get("source"),
                "content_type": "colpali_page",
            }
        )
        if len(passages) >= k:
            break

    return passages


async def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    document_filter: str = "all",
) -> Dict:
    """
    RAG espace ColPali-only : LanceDB MaxSim → métadonnées page pour rendu PNG.
    """
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d inaccessible (user %d)", space_id, user_id)
        return {"passages": [], "status": "disabled", "reason": "space_not_found"}
    if not query_text or not query_text.strip():
        return {"passages": [], "status": "disabled", "reason": "empty_query"}

    try:
        candidate_k = max(k * 3, 30)

        with trace_run(
            "colpali_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "candidate_k": candidate_k},
            tags=["retrieval", "colpali", "space"],
        ) as vr:
            nodes = _retrieve_colpali_pages(
                session, space_id, query_text, candidate_k, document_filter=document_filter
            )
            vr.end(outputs={"nb": len(nodes)})

        if not nodes:
            return {"passages": [], "status": "ok", "reason": "no_results"}

        passages = _format_colpali_pages(nodes, k)
        return {"passages": passages, "status": "ok", "reason": None}

    except Exception as e:
        logger.error("search_relevant_passages (space): %s", e, exc_info=True)
        return {"passages": [], "status": "disabled", "reason": f"error: {str(e)}"}


async def search_technical_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
) -> Dict:
    """Recherche ColPali limitée aux documents techniques (hors FAQ correctives)."""
    return await search_relevant_passages(
        session=session,
        space_id=space_id,
        query_text=query_text,
        user_id=user_id,
        k=k,
        document_filter="technical",
    )


async def search_corrective_faq_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    draft_response: str = "",
    k: Optional[int] = None,
) -> Dict:
    """Désactivé — pipeline FAQ texte supprimé."""
    return {"passages": [], "status": "disabled", "reason": "faq_disabled"}
