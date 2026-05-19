"""
Service de recherche sémantique pour les espaces (architecture Document/Space).

Pipeline lean :
  1. analyze_query (regex + pivots KAG)
  2. Vector pgvector + match exact refs + boost KAG
  3. Score composite + adaptive_gate
  4. 1-hop conditionnel (comparatif + 2+ pivots)
  5. Rerank cross-encoder
  6. adaptive_top_n (K dynamique)
  7. smart_parent_or_leaf + source authority
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from typing import Dict, List, Optional, Set

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import or_, text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.chunk_metadata_utils import (
    apply_row_metadata_defaults,
    enrich_docling_page_metadata,
    enrich_passage_content_for_llm,
    merged_chunk_metadata,
    resolve_page_range_from_metadata,
    table_citation_hint,
)
from app.services.query_reasoning_service import QueryIntent, reason_query_intent
from app.services.retrieval_pipeline import (
    VECTOR_RETRIEVE_MAX,
    RetrievalStats,
    adaptive_gate,
    adaptive_top_n,
    analyze_query,
    annotate_kag_matches_space,
    expand_one_hop_space,
    log_retrieval_stats,
    merge_vector_and_exact,
    parse_chunk_id,
    retrieve_exact_refs_space,
    smart_parent_or_leaf,
)
from app.services.space_service import get_space_by_id
from app.tracing import trace_run
import logging

logger = logging.getLogger(__name__)

try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

    FLAG_RERANKER_AVAILABLE = True
except ImportError:
    FLAG_RERANKER_AVAILABLE = False

try:
    from llama_index.core.postprocessor import SentenceTransformersRerank

    ST_RERANKER_AVAILABLE = True
except ImportError:
    ST_RERANKER_AVAILABLE = False

RERANKER_AVAILABLE = FLAG_RERANKER_AVAILABLE or ST_RERANKER_AVAILABLE
RERANKER_MODEL = settings.RERANKER_MODEL
RERANKER_ENABLED = settings.RERANKER_ENABLED
RERANK_MIN_SCORE = 0.30
_FLAG_RERANK_TOP_N = int(os.getenv("RERANKER_TOP_N", "4096"))
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"
TRACE_TEXT_MAX_CHARS = int(os.getenv("TRACE_TEXT_MAX_CHARS", "12000"))

TITLE_QUERY_BOOST_PER_MATCH = float(os.getenv("TITLE_QUERY_BOOST_PER_MATCH", "0.5"))
TITLE_QUERY_BOOST_CAP = float(os.getenv("TITLE_QUERY_BOOST_CAP", "2.0"))

_FALLBACK_STOPWORDS = {
    "the", "and", "for", "with", "dans", "avec", "pour", "une", "des", "les",
    "est", "sur", "pas", "plus", "que", "qui", "this", "that", "what", "how",
    "quoi", "comment", "quel", "quelle", "quels", "quelles", "from", "par",
}

_reranker_instance = None
_reranker_lock = threading.Lock() if RERANKER_AVAILABLE else None
_embed_model_instance = None


def classify_query_type(query_text: str) -> str:
    return analyze_query(query_text).intent


def is_factual_query(query_text: str) -> bool:
    return classify_query_type(query_text) == "factual"


def _text_for_trace(node: TextNode) -> str:
    raw = (
        node.get_content()
        if hasattr(node, "get_content")
        else getattr(node, "text", "") or ""
    )
    if not isinstance(raw, str):
        raw = str(raw)
    if TRACE_TEXT_MAX_CHARS > 0 and len(raw) > TRACE_TEXT_MAX_CHARS:
        return raw[:TRACE_TEXT_MAX_CHARS]
    return raw


def _nodes_for_trace(candidates: List[NodeWithScore], limit: int = 80) -> List[Dict]:
    rows: List[Dict] = []
    for nws in candidates[:limit]:
        meta = dict(getattr(nws.node, "metadata", {}) or {})
        rows.append(
            {
                "score": round(float(nws.score or 0.0), 4),
                "document_title": meta.get("document_title"),
                "section": meta.get("parent_heading") or meta.get("heading"),
                "kag_entity": meta.get("kag_matched_entity"),
                "text": _text_for_trace(nws.node),
            }
        )
    return rows


def _get_reranker():
    global _reranker_instance
    if not RERANKER_AVAILABLE:
        return None
    with _reranker_lock:
        if _reranker_instance is not None:
            return _reranker_instance
        use_fp16 = os.getenv("RERANKER_USE_FP16", "false").lower() == "true"
        is_bge = "bge-reranker" in RERANKER_MODEL.lower()
        logger.info("Initialisation reranker %s...", RERANKER_MODEL)
        try:
            if is_bge and FLAG_RERANKER_AVAILABLE:
                _reranker_instance = FlagEmbeddingReranker(
                    model=RERANKER_MODEL,
                    top_n=_FLAG_RERANK_TOP_N,
                    use_fp16=use_fp16,
                )
            elif ST_RERANKER_AVAILABLE:
                _reranker_instance = SentenceTransformersRerank(
                    model=RERANKER_MODEL,
                    top_n=_FLAG_RERANK_TOP_N,
                    device=os.getenv("EMBEDDING_DEVICE", "cpu"),
                )
            elif FLAG_RERANKER_AVAILABLE:
                _reranker_instance = FlagEmbeddingReranker(
                    model=RERANKER_MODEL,
                    top_n=_FLAG_RERANK_TOP_N,
                    use_fp16=use_fp16,
                )
            else:
                return None
        except Exception as e:
            logger.error("Reranker %s : %s", RERANKER_MODEL, e)
            return None
        logger.info("Reranker initialisé")
    return _reranker_instance


def _get_embed_model() -> HuggingFaceEmbedding:
    global _embed_model_instance
    if _embed_model_instance is None:
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")
        _embed_model_instance = HuggingFaceEmbedding(
            model_name=model_name,
            device=device,
            embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
    return _embed_model_instance


def _set_node_text_content(node, text: str) -> None:
    if hasattr(node, "set_content"):
        node.set_content(text)
    else:
        setattr(node, "text", text)


def _retrieve_leaves_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
) -> List[NodeWithScore]:
    embed_model = _get_embed_model()
    query_embedding = embed_model.get_query_embedding(query_text)
    query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

    sql_query = text(f"""
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
            d.id AS document_id,
            1 - (dc.embedding <=> '{query_embedding_str}'::vector) AS similarity_score
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          AND dc.embedding IS NOT NULL
          AND dc.is_leaf = true
        ORDER BY dc.embedding <=> '{query_embedding_str}'::vector
        LIMIT :limit_k
    """)

    result = session.execute(sql_query, {"space_id": space_id, "limit_k": candidate_k})
    nodes_with_scores: List[NodeWithScore] = []
    for row in result:
        metadata = merged_chunk_metadata(row.metadata_json, row.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=row.document_id,
            document_title=row.document_title or "Document sans titre",
            chunk_index=row.chunk_index,
            source=getattr(row, "chunk_source", None),
        )
        node = TextNode(
            id_=f"chunk-{row.id}",
            text=row.content or row.text or "",
            metadata=metadata,
        )
        nodes_with_scores.append(
            NodeWithScore(node=node, score=float(row.similarity_score))
        )

    logger.info(
        "Vector pgvector (space): %d feuilles (k=%d)",
        len(nodes_with_scores),
        candidate_k,
    )
    return nodes_with_scores


def _single_stage_rerank_leaves(
    filtered_candidates: List[NodeWithScore],
    query_text: str,
    k: int,
    char_cap: int = 2000,
) -> List[NodeWithScore]:
    reranker = _get_reranker()
    if not reranker:
        return filtered_candidates[:k]

    pool = filtered_candidates
    backup: Dict[str, str] = {}
    for nws in pool:
        node = nws.node
        nid = str(getattr(node, "id_", None) or "")
        raw = (
            node.get_content()
            if hasattr(node, "get_content")
            else getattr(node, "text", "") or ""
        )
        backup[nid] = raw
        meta = dict(getattr(node, "metadata", {}) or {})
        enriched = enrich_passage_content_for_llm(raw, meta)
        short = enriched[:char_cap] if len(enriched) > char_cap else enriched
        _set_node_text_content(node, short)

    try:
        r = reranker.postprocess_nodes(
            pool,
            query_bundle=QueryBundle(query_str=query_text),
        )
        for nws in pool:
            nid = str(getattr(nws.node, "id_", None) or "")
            if nid in backup:
                _set_node_text_content(nws.node, backup[nid])
        for nws in r:
            nid = str(getattr(nws.node, "id_", None) or "")
            raw = backup.get(nid, "")
            meta = dict(getattr(nws.node, "metadata", {}) or {})
            enriched = enrich_passage_content_for_llm(raw, meta)
            _set_node_text_content(nws.node, enriched)
        logger.info("Rerank (space): pool=%d → %d", len(pool), len(r))
        return r[:k]
    except Exception as e:
        logger.warning("Rerank (space) échoué: %s", e)
        for nws in pool:
            nid = str(getattr(nws.node, "id_", None) or "")
            if nid in backup:
                _set_node_text_content(nws.node, backup[nid])
        return filtered_candidates[:k]


def _apply_rerank_min_score(top_leaves: List[NodeWithScore], k: int) -> List[NodeWithScore]:
    if not top_leaves:
        return []
    ranked = sorted(top_leaves, key=lambda n: float(n.score or 0), reverse=True)
    max_score = float(ranked[0].score or 0)
    if max_score < 0:
        return ranked[:k]
    filtered = [nws for nws in ranked if float(nws.score or 0) >= RERANK_MIN_SCORE]
    return (filtered or ranked)[:k]


def _build_parent_node_dict(
    session: Session, space_id: int, user_id: int
) -> Dict[str, TextNode]:
    statement = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
            DocumentChunk.is_leaf.is_(False),
        )
    )
    rows = session.exec(statement).all()
    node_dict: Dict[str, TextNode] = {}
    for chunk, document_title in rows:
        metadata = merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=chunk.document_id,
            document_title=document_title or "Document sans titre",
            chunk_index=chunk.chunk_index,
            node_id=chunk.node_id,
            source=chunk.source,
        )
        llama_id = f"chunk-{chunk.id}"
        node = TextNode(
            id_=llama_id,
            text=chunk.content or chunk.text or "",
            metadata=metadata,
        )
        if chunk.node_id:
            node_dict[str(chunk.node_id)] = node
        node_dict[llama_id] = node
    logger.info("Parents chargés (space): %d", len(rows))
    return node_dict


_PARENT_MULTIHOP_MAX = 4


def _resolve_space_parent_with_multihop(
    session: Session,
    space_id: int,
    user_id: int,
    document_id: Optional[int],
    parent_node_id: Optional[str],
    parent_node_dict: Dict[str, TextNode],
) -> Optional[TextNode]:
    if not parent_node_id or document_id is None:
        return None
    if parent_node_id in parent_node_dict:
        return None

    intermediates: List[str] = []
    current_pid: Optional[str] = parent_node_id
    hops = 0

    while current_pid and hops < _PARENT_MULTIHOP_MAX:
        hops += 1
        stmt = (
            select(DocumentChunk, Document.title)
            .join(Document, Document.id == DocumentChunk.document_id)
            .join(DocumentSpace, DocumentSpace.document_id == Document.id)
            .where(
                DocumentSpace.space_id == space_id,
                DocumentChunk.document_id == document_id,
                Document.user_id == user_id,
                DocumentChunk.node_id == current_pid,
            )
        )
        row = session.exec(stmt).first()
        if not row:
            break
        chunk, document_title = row
        metadata = merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            metadata,
            document_id=chunk.document_id,
            document_title=document_title or "Document sans titre",
            chunk_index=chunk.chunk_index,
            node_id=chunk.node_id,
            source=chunk.source,
        )
        llama_id = f"chunk-{chunk.id}"
        text_body = (chunk.content or chunk.text or "").strip()

        if not chunk.is_leaf:
            if intermediates:
                prefix = "\n\n---\n\n".join(reversed(intermediates))
                text_body = f"{prefix}\n\n---\n\n{text_body}"
            return TextNode(id_=llama_id, text=text_body, metadata=metadata)

        if text_body:
            intermediates.append(text_body)
        current_pid = chunk.parent_node_id

    return None


def _extract_query_terms(query_text: str) -> List[str]:
    terms = [t.lower() for t in re.findall(r"[A-Za-zÀ-ÿ0-9]+", query_text or "")]
    return [t for t in terms if len(t) >= 3 and t not in _FALLBACK_STOPWORDS][:8]


def _get_meaningful_words(text: str) -> Set[str]:
    if not text:
        return set()
    n = unicodedata.normalize("NFD", text.lower())
    n = "".join(c for c in n if unicodedata.category(c) != "Mn")
    return {
        w
        for w in re.findall(r"[a-z0-9]+", n)
        if len(w) > 3 and w not in _FALLBACK_STOPWORDS
    }


def refine_with_source_authority(
    passages: List[Dict],
    query_text: str,
    reasoning_result: Optional[QueryIntent] = None,
) -> List[Dict]:
    if not passages:
        return passages

    if reasoning_result and reasoning_result.primary_source:
        source_to_boost = reasoning_result.primary_source.lower()
        for p in passages:
            doc_source = (p.get("source") or "").lower()
            if doc_source == source_to_boost:
                p["score"] = float(p.get("score") or 0.0) + 0.8

    if query_text and query_text.strip():
        query_words = _get_meaningful_words(query_text)
        if query_words:
            for p in passages:
                document_title = (p.get("document_title") or "").strip()
                if not document_title:
                    continue
                title_words = _get_meaningful_words(document_title)
                common = query_words & title_words
                if common:
                    boost = min(
                        TITLE_QUERY_BOOST_PER_MATCH * len(common),
                        TITLE_QUERY_BOOST_CAP,
                    )
                    p["score"] = float(p.get("score") or 0.0) + boost

    passages.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return passages


async def _keyword_fallback_passages(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    k: int,
) -> List[Dict]:
    terms = _extract_query_terms(query_text)
    base_stmt = (
        select(DocumentChunk, Document.title)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(DocumentSpace.space_id == space_id)
        .order_by(
            DocumentChunk.is_leaf.desc(),
            Document.updated_at.desc(),
            DocumentChunk.chunk_index,
        )
    )
    rows = []
    if terms:
        stmt = base_stmt.where(
            or_(*[DocumentChunk.content.ilike(f"%{term}%") for term in terms])
        ).limit(max(k * 4, 12))
        rows = session.exec(stmt).all()
    if not rows:
        rows = session.exec(base_stmt.limit(max(k * 2, 8))).all()

    passages: List[Dict] = []
    seen_chunk_ids: set = set()
    for chunk, document_title in rows:
        if chunk.id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.id)
        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue
        lowered = content.lower()
        match_count = sum(1 for term in terms if term in lowered) if terms else 0
        score = (match_count / max(len(terms), 1)) if terms else 0.05
        node_metadata = merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        apply_row_metadata_defaults(
            node_metadata,
            document_id=chunk.document_id,
            document_title=document_title or "Document sans titre",
            chunk_index=chunk.chunk_index,
            source=chunk.source,
        )
        node = TextNode(id_=f"chunk-{chunk.id}", text=content, metadata=node_metadata)
        passages.append(_node_to_passage(node, fallback_score=score))
        if len(passages) >= k:
            break
    return passages


def _merge_leaf_page_into_node_metadata(leaf_node, target_node) -> None:
    leaf_meta = enrich_docling_page_metadata(
        dict(getattr(leaf_node, "metadata", {}) or {})
    )
    m = dict(getattr(target_node, "metadata", {}) or {})
    pn, ps, pe = resolve_page_range_from_metadata(leaf_meta)
    if pn is not None:
        m["page_no"] = pn
    elif m.get("page_start") is not None:
        try:
            m["page_no"] = int(m["page_start"])
        except (TypeError, ValueError):
            pass
    if ps is not None:
        m.setdefault("page_start", ps)
    if pe is not None:
        m.setdefault("page_end", pe)
    leaf_chunk_id = parse_chunk_id(leaf_node)
    if leaf_chunk_id is not None:
        m["source_leaf_chunk_id"] = leaf_chunk_id
    setattr(target_node, "metadata", m)


def _node_to_passage(node, fallback_score: float = 0.0) -> Dict:
    metadata = dict(getattr(node, "metadata", {}) or {})
    document_title = metadata.get("document_title", "Document sans titre")
    document_id = metadata.get("document_id")
    chunk_index = metadata.get("chunk_index", 0)
    page_no, page_start, page_end = resolve_page_range_from_metadata(metadata)
    parent_heading = metadata.get("parent_heading") or metadata.get("heading")
    content = node.get_content() if hasattr(node, "get_content") else str(node)
    content_enriched = enrich_passage_content_for_llm(content, metadata)
    passage_text = f"**{document_title}**\n{content_enriched}"
    chunk_id = parse_chunk_id(node)
    table_hint = table_citation_hint(metadata)
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
        "content_type": metadata.get("content_type"),
        "is_image_chunk": bool(metadata.get("is_image_chunk")),
        "image_path": metadata.get("image_path"),
        "image_filename": metadata.get("image_filename"),
        "caption": metadata.get("caption") or metadata.get("figure_title"),
    }
    if table_hint:
        out["table_citation"] = table_hint
    if settings.RAG_DEBUG_METADATA:
        out["kag_matched_entity"] = metadata.get("kag_matched_entity")
        out["vector_similarity"] = metadata.get("vector_similarity")
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
    return out


async def search_relevant_passages(
    session: Session,
    space_id: int,
    query_text: str,
    user_id: int,
    k: int = 10,
) -> List[Dict]:
    """Recherche lean : vector + refs exactes + KAG boost + rerank + K dynamique."""
    space = get_space_by_id(session, space_id, user_id)
    if not space:
        logger.warning("Espace %d inaccessible (user %d)", space_id, user_id)
        return []
    if not query_text or not query_text.strip():
        return []

    reasoning_result = await reason_query_intent(query_text)
    stats = RetrievalStats()

    try:
        qa = analyze_query(query_text)
        logger.info(
            "Query analysis (space): intent=%s refs=%s pivots=%s one_hop=%s",
            qa.intent,
            qa.product_refs[:4],
            qa.pivot_entities[:4],
            qa.needs_one_hop,
        )

        with trace_run(
            "vector_retrieval",
            run_type="retriever",
            inputs={"query": query_text, "space_id": space_id},
            tags=["retrieval", "vector", "space"],
        ) as vr_run:
            vector_nodes = _retrieve_leaves_sql(
                session, space_id, user_id, query_text, VECTOR_RETRIEVE_MAX
            )
            vr_run.end(
                outputs={
                    "nb": len(vector_nodes),
                    "top3": [round(float(c.score or 0), 4) for c in vector_nodes[:3]],
                }
            )

        exact_nodes: List[NodeWithScore] = []
        if qa.product_refs:
            exact_nodes = retrieve_exact_refs_space(
                session, space_id, qa.product_refs
            )
            stats.ref_match_used = bool(exact_nodes)

        candidates = merge_vector_and_exact(vector_nodes, exact_nodes, qa)
        stats.pool_after_merge = len(candidates)
        stats.top1_vector_similarity = max(
            (c.vector_similarity for c in candidates), default=0.0
        )

        if settings.KAG_ENABLED and qa.pivot_entities:
            stats.kag_boost_used = annotate_kag_matches_space(
                session, space_id, candidates, qa.pivot_entities
            )

        gated = adaptive_gate(candidates)
        stats.pool_after_gate = len(gated)

        if qa.needs_one_hop:
            gated = expand_one_hop_space(session, space_id, gated, qa)
            stats.one_hop_used = True

        if not gated:
            return await _keyword_fallback_passages(
                session, space_id, user_id, query_text, k
            )

        rerank_input = [c.to_node_with_score() for c in gated]

        if RERANKER_AVAILABLE and RERANKER_ENABLED and _get_reranker():
            with trace_run(
                "reranking",
                run_type="chain",
                inputs={"nb": len(rerank_input), "k_max": k},
                tags=["reranking", "space"],
            ) as rr:
                reranked = _single_stage_rerank_leaves(
                    rerank_input, query_text, k=len(rerank_input), char_cap=2000
                )
                reranked = _apply_rerank_min_score(reranked, k=len(rerank_input))
                rr.end(outputs={"nb": len(reranked)})
        else:
            reranked = rerank_input

        top_n = adaptive_top_n(reranked, k_max=k)
        stats.final_k = len(top_n)

        parent_node_dict = _build_parent_node_dict(session, space_id, user_id)

        def _resolve_parent(parent_node_id: str, leaf_node: TextNode) -> Optional[TextNode]:
            leaf_meta = dict(getattr(leaf_node, "metadata", {}) or {})
            doc_id = leaf_meta.get("document_id")
            try:
                doc_id_int = int(doc_id) if doc_id is not None else None
            except (TypeError, ValueError):
                doc_id_int = None
            return _resolve_space_parent_with_multihop(
                session,
                space_id,
                user_id,
                doc_id_int,
                parent_node_id,
                parent_node_dict,
            )

        final_nodes: List[NodeWithScore] = []
        seen_node_ids: set = set()
        for nws in top_n:
            leaf = nws.node
            target = smart_parent_or_leaf(
                leaf,
                parent_node_dict,
                resolve_parent_fn=_resolve_parent,
            )
            if target is not leaf:
                _merge_leaf_page_into_node_metadata(leaf, target)
            node_id = getattr(target, "id_", None)
            if node_id and node_id in seen_node_ids:
                continue
            if node_id:
                seen_node_ids.add(node_id)
            final_nodes.append(
                NodeWithScore(node=target, score=float(nws.score or 0.0))
            )

        passages = [
            _node_to_passage(nws.node, fallback_score=float(nws.score or 0.0))
            for nws in final_nodes
        ]
        passages = refine_with_source_authority(
            passages, query_text, reasoning_result=reasoning_result
        )
        log_retrieval_stats(stats, "space")

        if not passages:
            return await _keyword_fallback_passages(
                session, space_id, user_id, query_text, k
            )
        return passages

    except Exception as e:
        logger.error("Erreur retrieval (space): %s", e, exc_info=True)
        return []
