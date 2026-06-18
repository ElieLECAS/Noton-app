"""
Recherche dans les espaces : retrieval hybride ColPali + pgvector L1 + BM25,
fusion RRF page-centric, expansion L1 + PNG multimodal pour le LLM.

Pipeline RAG (USE_MULTIMODAL_RETRIEVAL=true, défaut) :
1. Retrieval ColPali + pgvector + BM25 au niveau page
2. Fusion RRF sans pré-filtrage ColPali
3. Expansion L1 + voisinage conditionnel
4. Génération : texte consolidé + PNG pour toutes les pages top K
"""

from __future__ import annotations

import logging
import math
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from llama_index.core.schema import NodeWithScore, TextNode
from sqlalchemy import or_, text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.embedding_service import generate_embedding
from app.services.space_service import get_space_by_id
from app.tracing import trace_run
from app.services import reranker_service

logger = logging.getLogger(__name__)

MIN_VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("MIN_VECTOR_SIMILARITY", "0.25"))
TITLE_QUERY_BOOST_PER_MATCH = float(os.getenv("TITLE_QUERY_BOOST_PER_MATCH", "0.5"))
TITLE_QUERY_BOOST_CAP = float(os.getenv("TITLE_QUERY_BOOST_CAP", "2.0"))

_FALLBACK_STOPWORDS = {
    # English
    "the", "and", "for", "with", "this", "that", "what", "how",
    "from", "you", "your", "not", "are", "was", "were", "have", "has",
    "will", "can", "could", "would", "should", "been", "being", "about",
    # French
    "dans", "avec", "pour", "une", "des", "les", "est", "sur", "pas",
    "plus", "que", "qui", "quoi", "comment", "quel", "quelle", "quels",
    "quelles", "par", "sans", "mais", "donc", "car", "son", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs", "tout", "tous",
    "toute", "toutes", "autre", "autres", "même", "aussi", "très",
    "bien", "encore", "ici", "entre", "après", "avant", "sous",
    "chez", "vers", "depuis", "pendant", "comme",
}

def _merged_chunk_metadata(primary: Optional[dict], legacy: Optional[dict]) -> Dict:
    merged: Dict = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(primary, dict):
        merged.update(primary)
    return merged


def _retrieve_leaves_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
    query_embedding: Optional[List[float]] = None,
    document_filter: str = "all",
) -> List[NodeWithScore]:
    """
    Recherche vectorielle LanceDB/ColPali sur les feuilles.
    
    Args:
        query_embedding: Ignoré (conservé pour compatibilité de signature)
        document_filter: "all" (tous), "technical" (exclut FAQ), "faq_corrective" (FAQ uniquement)
    """
    logger.info(
        "[_retrieve_leaves_sql] Starting retrieval for space_id=%s, user_id=%s, query_text='%s', candidate_k=%d, document_filter='%s'",
        space_id,
        user_id,
        query_text,
        candidate_k,
        document_filter,
    )
    from app.services.document_service_new import feedback_corrective_sql_filter

    filter_clause = feedback_corrective_sql_filter(document_filter, "d")

    # 1. Récupérer la liste des document_ids appartenant à cette space_id et satisfaisant le filter_clause
    sql_docs = text(f"""
        SELECT DISTINCT d.id
        FROM document d
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          {filter_clause}
    """)
    doc_ids = [row[0] for row in session.execute(sql_docs, {"space_id": space_id})]
    logger.info(
        "[_retrieve_leaves_sql] Resolved document IDs in space %s: %s",
        space_id,
        doc_ids,
    )
    if not doc_ids:
        logger.info("LanceDB (space): aucun document correspondant au filtre dans l'espace %s", space_id)
        return []

    # 2. Rechercher dans LanceDB avec ColPali
    from app.services.colpali_service import embed_query_colpali
    from app.services.lancedb_service import search_colpali_lancedb
    
    logger.info("[_retrieve_leaves_sql] Generating ColPali query token embeddings...")
    query_token_embeddings = embed_query_colpali(query_text)
    logger.info(
        "[_retrieve_leaves_sql] Generated %d query token embeddings for ColPali. Querying LanceDB...",
        len(query_token_embeddings) if query_token_embeddings else 0,
    )
    search_results = search_colpali_lancedb(query_token_embeddings, doc_ids, limit=candidate_k)
    logger.info(
        "[_retrieve_leaves_sql] LanceDB search returned %d raw colpali patch matches.",
        len(search_results),
    )
        
    if not search_results:
        logger.info("[_retrieve_leaves_sql] No search results returned from LanceDB.")
        return []

    # 3. Récupérer les données textuelles complètes et métadonnées depuis PostgreSQL pour les chunks trouvés
    chunk_ids = [row["id"] for row in search_results]
    logger.info(
        "[_retrieve_leaves_sql] Fetching details from PostgreSQL for chunk IDs: %s",
        chunk_ids,
    )
    sql_chunks = text("""
        SELECT
            dc.id,
            dc.content,
            dc.text,
            dc.chunk_index,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            dc.source AS chunk_source,
            d.title AS document_title,
            d.source AS document_source,
            d.id AS document_id
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.id IN :chunk_ids
    """)
    chunk_rows = session.execute(sql_chunks, {"chunk_ids": tuple(chunk_ids)}).all()
    logger.info(
        "[_retrieve_leaves_sql] PostgreSQL returned %d rows for chunk details.",
        len(chunk_rows),
    )

    # Conserver l'ordre trié retourné par LanceDB
    rows_map = {row.id: row for row in chunk_rows}
    nodes: List[NodeWithScore] = []
    
    for row_lancedb in search_results:
        chunk_id = row_lancedb["id"]
        row = rows_map.get(chunk_id)
        if not row:
            logger.warning(
                "[_retrieve_leaves_sql] Chunk ID %d found in LanceDB but not found in PostgreSQL!",
                chunk_id,
            )
            continue
        
        # LanceDB retourne '_distance' en tant que distance cosinus (1 - cosine_similarity)
        # Donc similarity_score = 1.0 - _distance
        distance = float(row_lancedb.get("_distance", 1.0))
        similarity_score = 1.0 - distance
        
        metadata = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        metadata["document_id"] = row.document_id
        metadata["document_title"] = row.document_title or "Document sans titre"
        metadata["chunk_index"] = row.chunk_index
        if getattr(row, "document_source", None):
            metadata["source"] = row.document_source
        if getattr(row, "chunk_source", None):
            metadata["source"] = row.chunk_source
            
        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes.append(NodeWithScore(node=node, score=similarity_score))
        
    logger.info(
        "[_retrieve_leaves_sql] Vector LanceDB/ColPali (space): %d feuilles (limit=%d)",
        len(nodes),
        candidate_k,
    )
    return nodes


def _parse_chunk_id_from_node(node: TextNode) -> Optional[int]:
    nid = getattr(node, "id_", None) or ""
    if isinstance(nid, str) and nid.startswith("chunk-"):
        try:
            return int(nid.split("-", 1)[1])
        except ValueError:
            return None
    return None


def _augment_and_format_passages(
    session: Session,
    nodes: List[NodeWithScore],
    k: int,
) -> List[Dict]:
    """
    Augmente et formate les passages après reranking / sélection finale.
    Regroupe et déduplique par window_id (pour la v4) ou par (document_id, chunk_index) pour le reste.
    Charge tout le contexte de la page/fenêtre ou les chunks adjacents pour ne pas tronquer l'information.
    """
    passages: List[Dict] = []
    seen_windows = set()
    seen_chunk_ids = set()

    for nws in nodes:
        node = nws.node
        score = nws.score
        meta = dict(node.metadata or {})
        
        doc_id = meta.get("document_id")
        wid = meta.get("window_id")
        chunk_id = _parse_chunk_id_from_node(node)
        
        # 1. Cas Multimodal v4 (avec window_id)
        if wid and doc_id is not None:
            if wid in seen_windows:
                continue
            seen_windows.add(wid)
            
            # Récupérer tous les chunks de la même fenêtre
            stmt = select(DocumentChunk).where(
                DocumentChunk.document_id == doc_id,
                text("(coalesce(metadata_json->>'window_id', metadata_->>'window_id')) = :window_id")
            )
            window_chunks = session.execute(stmt, {"window_id": wid}).scalars().all()
            
            # Séparer reports et raw text
            report_contents = []
            raw_contents = []
            
            # Trier pour conserver l'ordre de lecture
            sorted_chunks = sorted(window_chunks, key=lambda c: (c.chunk_index or 0, c.id or 0))
            
            doc_title = meta.get("document_title") or "Document sans titre"
            page_start = meta.get("page_start") or meta.get("page_no") or 0
            page_end = meta.get("page_end") or page_start
            
            for chunk in sorted_chunks:
                chunk_meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
                ctype = chunk_meta.get("content_type")
                content_text = (chunk.content or chunk.text or "").strip()
                if not content_text:
                    continue
                
                if ctype == "page_window_report":
                    report_contents.append(content_text)
                else:
                    raw_contents.append(content_text)
            
            # Si pas de raw trouvé, utiliser le contenu du nœud actuel comme fallback
            if not raw_contents:
                raw_contents.append((node.get_content() if hasattr(node, "get_content") else str(node)).strip())
                
            joined_reports = "\n\n".join(report_contents).strip()
            joined_raw = "\n\n".join(raw_contents).strip()
            
            # Formatage propre de liaison
            parts = []
            if joined_reports:
                parts.append("--- RAPPORT DE SYNTHÈSE DE LA FENÊTRE ---")
                parts.append(joined_reports)
            parts.append("--- TEXTE BRUT DU DOCUMENT ---")
            parts.append(joined_raw)
            
            augmented_content = "\n\n".join(parts)
            passage_text = f"**{doc_title}**\n{augmented_content}"
            
            out = {
                "passage": passage_text,
                "passage_raw": joined_raw,
                "document_title": doc_title,
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": int(meta.get("chunk_index", 0)),
                "score": float(score),
                "page_no": page_start,
                "page_start": page_start,
                "page_end": page_end,
                "section": meta.get("parent_heading") or meta.get("heading"),
                "source": meta.get("source"),
                "content_type": "augmented_multimodal_window",
            }
            if meta.get("row_index") is not None:
                out["row_index"] = meta.get("row_index")
            if meta.get("table_id"):
                out["table_id"] = meta.get("table_id")
            raw_rrf = meta.get("raw_rrf_score")
            if raw_rrf is not None:
                out["raw_rrf_score"] = float(raw_rrf)
            if meta.get("rerank_score") is not None:
                out["rerank_score"] = float(meta.get("rerank_score"))
                
            passages.append(out)
            
        # 2. Cas classique / Legacy / FAQ (sans window_id)
        else:
            if chunk_id is not None:
                if chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)
            
            # Essayer d'augmenter avec les chunks adjacents (+/- 1) du même document
            chunk_index = meta.get("chunk_index")
            doc_title = meta.get("document_title") or "Document sans titre"
            node_text = (node.get_content() if hasattr(node, "get_content") else str(node)).strip()
            
            if doc_id is not None and chunk_index is not None:
                # Récupérer chunk_index - 1, chunk_index, chunk_index + 1
                stmt = select(DocumentChunk).where(
                    DocumentChunk.document_id == doc_id,
                    DocumentChunk.chunk_index.in_([chunk_index - 1, chunk_index, chunk_index + 1])
                )
                adj_chunks = session.execute(stmt).scalars().all()
                sorted_adj = sorted(adj_chunks, key=lambda c: c.chunk_index)
                
                joined_text = "\n\n".join([(c.content or c.text or "").strip() for c in sorted_adj if (c.content or c.text or "").strip()])
                if not joined_text:
                    joined_text = node_text
            else:
                joined_text = node_text
                
            passage_text = f"**{doc_title}**\n{joined_text}"
            
            out = {
                "passage": passage_text,
                "passage_raw": node_text,
                "document_title": doc_title,
                "document_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": int(chunk_index or 0),
                "score": float(score),
                "page_no": meta.get("page_no") or meta.get("page_start"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "section": meta.get("parent_heading") or meta.get("heading"),
                "source": meta.get("source"),
                "content_type": meta.get("content_type", "augmented_legacy_sliding_window"),
            }
            if meta.get("row_index") is not None:
                out["row_index"] = meta.get("row_index")
            if meta.get("table_id"):
                out["table_id"] = meta.get("table_id")
            raw_rrf = meta.get("raw_rrf_score")
            if raw_rrf is not None:
                out["raw_rrf_score"] = float(raw_rrf)
            if meta.get("rerank_score") is not None:
                out["rerank_score"] = float(meta.get("rerank_score"))
                
            passages.append(out)
            
        # Limiter à k passages finaux
        if len(passages) >= k:
            break
            
    return passages





def _enrich_content_with_heading_and_figure(content: str, metadata: dict) -> str:
    section_label = (
        metadata.get("heading_path")
        or metadata.get("section_parent_heading")
        or metadata.get("scope_label")
        or metadata.get("parent_heading")
        or metadata.get("heading")
    )
    figure_title = metadata.get("figure_title") or metadata.get("image_anchor")
    parts = []
    
    # Injection du résumé de page en contexte additionnel (Pass 2)
    page_summary = metadata.get("page_summary")
    if page_summary and str(page_summary).strip():
        parts.append(f"[Contexte de la page: {page_summary.strip()}]")
        
    if section_label and str(section_label).strip():
        parts.append(f"[Section: {section_label.strip()}]")
    if figure_title and str(figure_title).strip():
        parts.append(str(figure_title).strip())
    if not parts:
        return content
        
    if page_summary:
        prefix = parts[0] + "\n" + " ".join(parts[1:])
    else:
        prefix = " ".join(parts)
        
    return prefix.strip() + "\n\n" + content if content else prefix.strip()


def _merge_leaf_page_into_node_metadata(leaf_node, target_node) -> None:
    leaf_meta = dict(getattr(leaf_node, "metadata", {}) or {})
    m = dict(getattr(target_node, "metadata", {}) or {})
    pn = leaf_meta.get("page_no")
    if pn is not None:
        try:
            m["page_no"] = int(pn)
        except (TypeError, ValueError):
            pass
    elif m.get("page_start") is not None:
        try:
            m["page_no"] = int(m["page_start"])
        except (TypeError, ValueError):
            pass
    ps = leaf_meta.get("page_start")
    pe = leaf_meta.get("page_end")
    if ps is not None:
        try:
            m.setdefault("page_start", int(ps))
        except (TypeError, ValueError):
            pass
    if pe is not None:
        try:
            m.setdefault("page_end", int(pe))
        except (TypeError, ValueError):
            pass
    leaf_chunk_id = _parse_chunk_id_from_node(leaf_node)
    if leaf_chunk_id is not None:
        m["source_leaf_chunk_id"] = leaf_chunk_id
    # Transférer le score RRF brut de la feuille vers le parent
    raw_rrf = leaf_meta.get("raw_rrf_score")
    if raw_rrf is not None:
        m["raw_rrf_score"] = raw_rrf
    setattr(target_node, "metadata", m)


def _chunk_row_to_text_node(row: DocumentChunk, score: float) -> NodeWithScore:
    meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
    meta["document_id"] = row.document_id
    meta["chunk_index"] = row.chunk_index
    node = TextNode(
        text=row.content or "",
        id_=str(row.id),
        metadata=meta,
    )
    return NodeWithScore(node=node, score=score)


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    document_title = metadata.get("document_title", "Document sans titre")
    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index", 0)
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    raw_page = metadata.get("page_no")
    resolved_page = None
    if raw_page is not None:
        try:
            resolved_page = int(raw_page)
        except (TypeError, ValueError):
            pass
    if resolved_page is None:
        for key in ["page_start", "page_label", "page_idx"]:
            val = metadata.get(key)
            if val is not None:
                try:
                    resolved_page = int(val)
                    break
                except (TypeError, ValueError):
                    continue
    page_no = resolved_page
    parent_heading = metadata.get("parent_heading")
    content = node.get_content() if hasattr(node, "get_content") else str(node)
    content_enriched = _enrich_content_with_heading_and_figure(content, metadata)
    passage_text = f"**{document_title}**\n{content_enriched}"
    chunk_id = _parse_chunk_id_from_node(node)
    out = {
        "passage": passage_text,
        "passage_raw": content,
        "document_title": document_title,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "source_leaf_chunk_id": metadata.get("source_leaf_chunk_id"),
        "chunk_index": int(chunk_index) if isinstance(chunk_index, (int, str)) else 0,
        "score": float(fallback_score or 0.0),
        "page_no": page_no,
        "section": parent_heading,
        "source": metadata.get("source"),
    }
    if page_start is not None:
        try:
            out["page_start"] = int(page_start)
        except (TypeError, ValueError):
            pass
    if page_end is not None:
        try:
            out["page_end"] = int(page_end)
        except (TypeError, ValueError):
            pass
    content_type = metadata.get("content_type")
    if content_type:
        out["content_type"] = content_type
    if metadata.get("row_index") is not None:
        out["row_index"] = metadata.get("row_index")
    if metadata.get("table_id"):
        out["table_id"] = metadata.get("table_id")
    raw_rrf = metadata.get("raw_rrf_score")
    if raw_rrf is not None:
        out["raw_rrf_score"] = float(raw_rrf)
    return out


async def search_multimodal_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    *,
    k: Optional[int] = None,
    document_filter: str = "all",
    include_retrieval_stages: bool = False,
) -> Dict:
    """
    Pipeline retrieval multimodal page-centric unifié.
    ColPali + pgvector + BM25 → RRF → expansion L1 → texte + PNG.
    """
    from app.services.page_retrieval_service import (
        expand_page_context,
        filter_colpali_pages_dynamic,
        format_multimodal_passages,
        fuse_multimodal_hits,
        get_space_document_ids,
        log_multimodal_retrieval_summary,
        retrieve_bm25_pages,
        retrieve_colpali_pages,
        retrieve_pgvector_pages,
        unified_hits_to_eval_passages,
    )

    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d inaccessible (user %d)", space_id, user_id)
        return {"passages": [], "images": [], "status": "disabled", "reason": "space_not_found"}
    if not query_text or not query_text.strip():
        return {"passages": [], "images": [], "status": "disabled", "reason": "empty_query"}

    top_k = k if k is not None else settings.RAG_TOP_K
    pool_size = max(settings.RERANK_POOL, settings.RAG_POOL_SIZE, top_k)

    logger.info(
        "[RAG multimodal] Démarrage — space_id=%s top_k=%s pool=%s rerank=%s filter=%s query=%r",
        space_id,
        top_k,
        pool_size,
        settings.RERANKER_ENABLED,
        document_filter,
        query_text[:120],
    )

    try:
        doc_ids = get_space_document_ids(session, space_id, document_filter)
        if not doc_ids:
            result: Dict = {"passages": [], "images": [], "status": "ok", "reason": "no_results"}
            if include_retrieval_stages:
                result["retrieval_stages"] = {"colpali": [], "post_rerank": [], "reason": "no_results"}
            return result

        query_embedding: Optional[List[float]] = None
        try:
            query_embedding = generate_embedding(query_text)
        except Exception as exc:
            logger.warning("[RAG multimodal] Embedding requête indisponible : %s", exc)

        with trace_run(
            "multimodal_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "pool_size": pool_size},
            tags=["retrieval", "multimodal", "space"],
        ) as hr:
            colpali_hits = retrieve_colpali_pages(session, doc_ids, query_text, pool_size)
            colpali_hits = filter_colpali_pages_dynamic(colpali_hits)
            pgvector_hits = retrieve_pgvector_pages(session, doc_ids, query_embedding or [], pool_size)
            bm25_hits = retrieve_bm25_pages(session, doc_ids, query_text, pool_size)
            hr.end(
                outputs={
                    "colpali": len(colpali_hits),
                    "pgvector": len(pgvector_hits),
                    "bm25": len(bm25_hits),
                }
            )

        logger.info(
            "[RAG multimodal] Retrievers — colpali=%d | pgvector=%d | bm25=%d",
            len(colpali_hits),
            len(pgvector_hits),
            len(bm25_hits),
        )

        fused_hits = fuse_multimodal_hits(
            colpali_hits,
            pgvector_hits,
            bm25_hits,
            rrf_k=settings.RRF_K,
            top_k=pool_size,
        )

        if not fused_hits:
            result = {"passages": [], "images": [], "status": "ok", "reason": "no_results", "dynamic_k": 0}
            if include_retrieval_stages:
                result["retrieval_stages"] = {
                    "colpali_only": unified_hits_to_eval_passages(colpali_hits, top_k),
                    "colpali": unified_hits_to_eval_passages(colpali_hits, top_k),
                    "post_rrf": [],
                    "post_rerank": [],
                    "minilm_rerank_enabled": settings.RERANKER_ENABLED,
                    "vision_rerank_enabled": False,
                    "reason": "no_results",
                }
            return result

        rerank_status = "disabled"
        dynamic_k = len(fused_hits)
        protected_hits: List[Any] = []
        final_hits = fused_hits

        if settings.RERANKER_ENABLED:
            from app.services.page_reranker_service import rerank_unified_page_hits

            with trace_run(
                "minilm_rerank",
                run_type="reranker",
                inputs={"query": query_text, "pool_size": len(fused_hits), "max_k": top_k},
                tags=["rerank", "minilm", "page"],
            ) as rr:
                final_hits, rerank_result, protected_hits = await rerank_unified_page_hits(
                    session,
                    query_text,
                    fused_hits,
                    max_k=top_k,
                )
                rerank_status = rerank_result.status
                dynamic_k = len(final_hits)
                rr.end(
                    outputs={
                        "status": rerank_status,
                        "dynamic_k": dynamic_k,
                        "protected": len(protected_hits),
                    }
                )
        else:
            final_hits = fused_hits[:top_k]
            for rank, hit in enumerate(final_hits, start=1):
                hit.final_rank = rank

        if rerank_status == "low_confidence_clarification" and not protected_hits:
            passages: List[Dict[str, Any]] = []
            images: List[str] = []
            log_multimodal_retrieval_summary(
                query_text=query_text,
                doc_ids=doc_ids,
                colpali_hits=colpali_hits,
                pgvector_hits=pgvector_hits,
                bm25_hits=bm25_hits,
                fused_hits=fused_hits,
                final_hits=final_hits,
                passages=passages,
                images=images,
                top_k=top_k,
                pool_size=pool_size,
                rerank_enabled=settings.RERANKER_ENABLED,
                rerank_status=rerank_status,
                dynamic_k=0,
                protected_hits=protected_hits,
            )
            low_conf_result = {
                "passages": [],
                "images": [],
                "status": "low_confidence_clarification",
                "reason": rerank_status,
                "dynamic_k": len(final_hits),
                "rerank_status": rerank_status,
            }
            if include_retrieval_stages:
                low_conf_result["retrieval_stages"] = {
                    "colpali_only": unified_hits_to_eval_passages(colpali_hits, top_k),
                    "colpali": unified_hits_to_eval_passages(colpali_hits, top_k),
                    "post_rrf": unified_hits_to_eval_passages(fused_hits, top_k),
                    "post_rerank": [],
                    "minilm_rerank_enabled": settings.RERANKER_ENABLED,
                    "vision_rerank_enabled": False,
                    "reason": rerank_status,
                }
            return low_conf_result

        if not final_hits:
            log_multimodal_retrieval_summary(
                query_text=query_text,
                doc_ids=doc_ids,
                colpali_hits=colpali_hits,
                pgvector_hits=pgvector_hits,
                bm25_hits=bm25_hits,
                fused_hits=fused_hits,
                final_hits=[],
                passages=[],
                images=[],
                top_k=top_k,
                pool_size=pool_size,
                rerank_enabled=settings.RERANKER_ENABLED,
                rerank_status=rerank_status,
                dynamic_k=0,
                protected_hits=protected_hits,
            )
            return {
                "passages": [],
                "images": [],
                "status": "ok",
                "reason": "no_results_after_rerank",
                "dynamic_k": 0,
                "rerank_status": rerank_status,
                **(
                    {
                        "retrieval_stages": {
                            "colpali_only": unified_hits_to_eval_passages(colpali_hits, top_k),
                            "colpali": unified_hits_to_eval_passages(colpali_hits, top_k),
                            "post_rrf": unified_hits_to_eval_passages(fused_hits, top_k),
                            "post_rerank": [],
                            "minilm_rerank_enabled": settings.RERANKER_ENABLED,
                            "vision_rerank_enabled": False,
                            "reason": "no_results_after_rerank",
                        }
                    }
                    if include_retrieval_stages
                    else {}
                ),
            }

        expanded_hits = expand_page_context(
            session,
            final_hits,
            neighbor_strategy=settings.RAG_NEIGHBOR_STRATEGY,
        )
        passages, images = format_multimodal_passages(
            session,
            expanded_hits,
            render_all_images=settings.RAG_RENDER_ALL_IMAGES,
        )

        max_images = settings.RAG_MAX_IMAGES
        if len(images) > max_images:
            logger.warning(
                "[RAG multimodal] Troncature images: %d → %d",
                len(images),
                max_images,
            )
            images = images[:max_images]

        log_multimodal_retrieval_summary(
            query_text=query_text,
            doc_ids=doc_ids,
            colpali_hits=colpali_hits,
            pgvector_hits=pgvector_hits,
            bm25_hits=bm25_hits,
            fused_hits=fused_hits,
            final_hits=final_hits,
            passages=passages,
            images=images,
            top_k=top_k,
            pool_size=pool_size,
            rerank_enabled=settings.RERANKER_ENABLED,
            rerank_status=rerank_status,
            dynamic_k=dynamic_k,
            protected_hits=protected_hits,
        )

        reason = "multimodal_rrf_minilm" if settings.RERANKER_ENABLED else "multimodal_rrf"
        result = {
            "passages": passages,
            "images": images,
            "status": "ok",
            "reason": reason,
            "total_hits": len(final_hits),
            "dynamic_k": dynamic_k,
            "rerank_status": rerank_status,
        }
        if include_retrieval_stages:
            result["retrieval_stages"] = {
                "colpali_only": unified_hits_to_eval_passages(colpali_hits, top_k),
                "colpali": unified_hits_to_eval_passages(colpali_hits, top_k),
                "post_rrf": unified_hits_to_eval_passages(fused_hits, top_k),
                "post_rerank": passages,
                "minilm_rerank_enabled": settings.RERANKER_ENABLED,
                "vision_rerank_enabled": False,
                "reason": reason,
            }
        return result

    except Exception as exc:
        logger.error("search_multimodal_passages (space): %s", exc, exc_info=True)
        return {
            "passages": [],
            "images": [],
            "status": "disabled",
            "reason": f"error: {str(exc)}",
        }


async def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 15,
    document_filter: str = "all",
    include_retrieval_stages: bool = False,
) -> Dict:
    """
    RAG espace : retrieval hybride ColPali + pgvector L1 + BM25, fusion RRF,
    expansion page entière + voisinage conditionnel, PNG conditionnel si ColPali seul.

    Si USE_MULTIMODAL_RETRIEVAL=true (défaut), délègue au pipeline page-centric unifié.
    """
    if settings.USE_MULTIMODAL_RETRIEVAL:
        return await search_multimodal_passages(
            session=session,
            space_id=space_id,
            query_text=query_text,
            user_id=user_id,
            k=k,
            document_filter=document_filter,
            include_retrieval_stages=include_retrieval_stages,
        )

    from app.services.page_retrieval_service import (
        build_weak_hit_pool,
        filter_colpali_page_hits_dynamic,
        format_hybrid_passages,
        fuse_page_hits_rrf,
        get_space_document_ids,
        page_hits_to_eval_passages,
        retrieve_bm25_page_hits,
        retrieve_colpali_page_hits,
        retrieve_pgvector_page_hits,
        top_hit_scores,
    )

    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d inaccessible (user %d)", space_id, user_id)
        return {"passages": [], "status": "disabled", "reason": "space_not_found"}
    if not query_text or not query_text.strip():
        return {"passages": [], "status": "disabled", "reason": "empty_query"}

    logger.info(
        "[RAG] Démarrage retrieval — space_id=%s k=%s filter=%s query=%r",
        space_id,
        k,
        document_filter,
        query_text[:120],
    )

    try:
        pool_size = max(k, settings.RETRIEVAL_EXPAND_POOL)
        doc_ids = get_space_document_ids(session, space_id, document_filter)
        if not doc_ids:
            result: Dict = {"passages": [], "status": "ok", "reason": "no_results"}
            if include_retrieval_stages:
                result["retrieval_stages"] = {"colpali": [], "post_rerank": [], "reason": "no_results"}
            return result

        query_embedding: Optional[List[float]] = None
        try:
            logger.info("[RAG] Génération embedding requête (pgvector)...")
            query_embedding = generate_embedding(query_text)
        except Exception as exc:
            logger.warning("[RAG] Embedding requête indisponible : %s", exc)

        logger.info("[RAG] Lancement des 3 retrievers (pool=%d, docs=%s)...", pool_size, doc_ids)
        with trace_run(
            "hybrid_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id, "pool_size": pool_size},
            tags=["retrieval", "hybrid", "space"],
        ) as hr:
            colpali_hits = retrieve_colpali_page_hits(session, doc_ids, query_text, pool_size)
            pgvector_hits = retrieve_pgvector_page_hits(session, doc_ids, query_embedding or [], pool_size)
            bm25_hits = retrieve_bm25_page_hits(session, doc_ids, query_text, pool_size)
            hr.end(
                outputs={
                    "colpali": len(colpali_hits),
                    "pgvector": len(pgvector_hits),
                    "bm25": len(bm25_hits),
                }
            )

        logger.info(
            "[RAG] Résultats retrievers — colpali=%d pages %s | pgvector=%d pages %s | bm25=%d pages %s",
            len(colpali_hits),
            top_hit_scores(colpali_hits),
            len(pgvector_hits),
            top_hit_scores(pgvector_hits),
            len(bm25_hits),
            top_hit_scores(bm25_hits),
        )

        # Seuil dynamique ColPali (absolu + marge relative)
        colpali_hits = filter_colpali_page_hits_dynamic(colpali_hits)

        weak_pool = build_weak_hit_pool(colpali_hits, pgvector_hits, bm25_hits)
        fused_hits = fuse_page_hits_rrf(colpali_hits, pgvector_hits, bm25_hits, top_n=k)

        logger.info(
            "[RAG] Après filtre ColPali + fusion RRF — colpali_retenu=%d, fusionnés=%d",
            len(colpali_hits),
            len(fused_hits),
        )

        if not fused_hits:
            result = {"passages": [], "status": "ok", "reason": "no_results"}
            if include_retrieval_stages:
                result["retrieval_stages"] = {
                    "colpali": [],
                    "post_rerank": [],
                    "vision_rerank_enabled": settings.VISION_RERANK_ENABLED,
                    "reason": "no_results",
                }
            return result

        reason = "hybrid_rrf"
        final_hits = list(fused_hits)

        # Vision rerank optionnel sur les pages ColPali
        if settings.VISION_RERANK_ENABLED:
            from llama_index.core.schema import TextNode
            from app.services.vision_reranker_service import rerank_pages_vision

            colpali_nodes = []
            hit_by_key = {h.page_key: h for h in final_hits}
            for hit in final_hits:
                if "colpali" not in hit.retrieval_sources:
                    continue
                meta = {
                    "document_id": hit.document_id,
                    "page_no": hit.page_no,
                    "page_start": hit.page_no,
                    "document_title": hit.document_title,
                }
                node = TextNode(
                    id_=f"page-{hit.document_id}-{hit.page_no}",
                    text=f"Page {hit.page_no}",
                    metadata=meta,
                )
                colpali_nodes.append(NodeWithScore(node=node, score=hit.colpali_score or hit.score))

            if colpali_nodes:
                with trace_run(
                    "vision_rerank",
                    run_type="reranker",
                    inputs={"query": query_text, "nb_candidates": len(colpali_nodes)},
                    tags=["rerank", "vision", "hybrid"],
                ) as rr:
                    reranked = await rerank_pages_vision(session, query_text, colpali_nodes)
                    rr.end(outputs={"nb": len(reranked)})
                if reranked:
                    reranked_keys = []
                    for nws in reranked:
                        meta = dict(nws.node.metadata or {})
                        doc_id = meta.get("document_id")
                        page_no = meta.get("page_no")
                        if doc_id is not None and page_no is not None:
                            reranked_keys.append(f"{doc_id}:{page_no}")
                    if reranked_keys:
                        ordered = []
                        seen = set()
                        for key in reranked_keys:
                            if key in hit_by_key and key not in seen:
                                ordered.append(hit_by_key[key])
                                seen.add(key)
                        for hit in final_hits:
                            if hit.page_key not in seen:
                                ordered.append(hit)
                        final_hits = ordered[:k]
                        reason = "hybrid_rrf_vision_rerank"

        passages = format_hybrid_passages(session, final_hits, weak_pool, k)

        result = {"passages": passages, "status": "ok", "reason": reason}
        if include_retrieval_stages:
            colpali_only_passages = page_hits_to_eval_passages(colpali_hits, k)
            post_rrf_passages = format_hybrid_passages(session, fused_hits, weak_pool, k)
            result["retrieval_stages"] = {
                "colpali_only": colpali_only_passages,
                "colpali": colpali_only_passages,
                "post_rrf": post_rrf_passages,
                "post_rerank": passages,
                "minilm_rerank_enabled": False,
                "vision_rerank_enabled": settings.VISION_RERANK_ENABLED,
                "reason": reason,
            }
        return result

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
    """
    Recherche RAG limitée aux documents techniques (exclut les FAQ correctives).
    
    Wrapper autour de search_relevant_passages avec document_filter="technical".
    
    Returns:
        Dict avec clés : passages (List[Dict]), status (str), reason (Optional[str])
    """
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
    """
    Recherche post-brouillon dédiée aux FAQ correctives issues des feedbacks négatifs.
    Désactivée en dur.
    """
    return {"passages": [], "status": "disabled", "reason": "faq_post_draft_disabled"}


def refine_with_source_authority(
    passages: List[Dict],
    query: str,
    reasoning_result: Any,
) -> List[Dict]:
    """
    Optimise l'autorité des sources par rapport à l'intention détectée.
    Si primary_source correspond à la source du passage, on applique un boost au score.
    """
    if not passages or not reasoning_result:
        return passages
    
    primary = getattr(reasoning_result, "primary_source", None)
    confidence = getattr(reasoning_result, "confidence", 0.0)
    
    boost_val = 0.8 * confidence if primary else 0.0
    
    refined = []
    for p in passages:
        p_copy = dict(p)
        score = p_copy.get("score", 0.0)
        source = p_copy.get("source")
        
        if primary and source and source.lower() == primary.lower():
            p_copy["score"] = score + boost_val
            
        refined.append(p_copy)
        
    refined.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return refined


def reciprocal_rank_fusion(
    vector_results: List[NodeWithScore],
    lexical_results: List[NodeWithScore],
    alphanumeric_results: Optional[List[NodeWithScore]] = None,
    k: int = 60,
    top_n: int = 15,
    normalize: bool = False,
) -> List[NodeWithScore]:
    """
    RRF (Reciprocal Rank Fusion) unifié pour fusionner les canaux vectoriel, lexical et alphanumérique.
    """
    id_to_node = {}
    ranks: Dict[str, Dict[str, int]] = {}

    def _add_results(results: List[NodeWithScore], channel_name: str):
        if not results:
            return
        for rank_idx, nws in enumerate(results, start=1):
            node_id = nws.node.id_
            id_to_node[node_id] = nws.node
            ranks.setdefault(node_id, {})[channel_name] = rank_idx

    _add_results(vector_results, "vector")
    _add_results(lexical_results, "lexical")
    if alphanumeric_results:
        _add_results(alphanumeric_results, "alphanumeric")

    if not id_to_node:
        return []

    fused_results = []
    for node_id, node in id_to_node.items():
        rrf_score = 0.0
        for channel_name, rank_idx in ranks[node_id].items():
            rrf_score += 1.0 / (k + rank_idx)

        meta = dict(node.metadata or {})
        meta["raw_rrf_score"] = rrf_score
        node.metadata = meta

        fused_results.append(NodeWithScore(node=node, score=rrf_score))

    fused_results.sort(key=lambda x: x.score, reverse=True)
    fused_results = fused_results[:top_n]

    if normalize and fused_results:
        scores = [x.score for x in fused_results]
        min_score = min(scores)
        max_score = max(scores)

        for nws in fused_results:
            if max_score > min_score:
                norm = 0.1 + 0.8 * ((nws.score - min_score) / (max_score - min_score))
            else:
                norm = 0.9
            nws.score = norm

    return fused_results


def _extract_alphanumeric_codes(query: str) -> List[str]:
    """Extrait les codes alphanumériques d'une requête."""
    words = re.findall(r'[a-zA-Z0-9.\-]+', query.lower())
    codes = []
    for w in words:
        w_clean = w.strip(".,;:!?()")
        if not w_clean or w_clean in _FALLBACK_STOPWORDS:
            continue
        if any(c.isdigit() for c in w_clean) or len(w_clean) >= 3:
            if w_clean not in codes:
                codes.append(w_clean)
    return codes


def _retrieve_leaves_bm25_sql(*args, **kwargs):
    """BM25 tsvector — implémenté dans page_retrieval_service."""
    from app.services.page_retrieval_service import _retrieve_leaves_bm25_sql as _bm25_impl
    return _bm25_impl(*args, **kwargs)


def _retrieve_leaves_alphanumeric_sql(*args, **kwargs):
    """Stub pour compatibilité de tests."""
    return []
