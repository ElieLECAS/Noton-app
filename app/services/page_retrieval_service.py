"""
Retrieval hybride au niveau page : ColPali + pgvector L1 + BM25, fusion RRF,
expansion small-to-big (L1 consolidés) et voisinage conditionnel N±1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.document_service_new import feedback_corrective_sql_filter

logger = logging.getLogger(__name__)

CONTENT_TYPE_SEMANTIC_LEAF = "semantic_leaf"
CONTENT_TYPE_PAGE_ANCHOR = "page_anchor"


@dataclass
class PageRetrievalHit:
    document_id: int
    page_no: int
    score: float = 0.0
    rrf_score: float = 0.0
    retrieval_sources: List[str] = field(default_factory=list)
    colpali_score: Optional[float] = None
    pgvector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    document_title: str = "Document sans titre"
    chunk_id: Optional[int] = None

    @property
    def page_key(self) -> str:
        return f"{self.document_id}:{self.page_no}"


def _merged_chunk_metadata(primary: Optional[dict], legacy: Optional[dict]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if isinstance(legacy, dict):
        merged.update(legacy)
    if isinstance(primary, dict):
        merged.update(primary)
    return merged


def _page_no_from_metadata(meta: dict) -> Optional[int]:
    for key in ("page_no", "page_start", "page"):
        val = meta.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def get_space_document_ids(
    session: Session,
    space_id: int,
    document_filter: str = "all",
) -> List[int]:
    filter_clause = feedback_corrective_sql_filter(document_filter, "d")
    sql_docs = text(f"""
        SELECT DISTINCT d.id
        FROM document d
        INNER JOIN document_space ds ON ds.document_id = d.id
        WHERE ds.space_id = :space_id
          {filter_clause}
    """)
    return [row[0] for row in session.execute(sql_docs, {"space_id": space_id})]


def _aggregate_hits_by_page(hits: List[PageRetrievalHit]) -> List[PageRetrievalHit]:
    """Garde le meilleur score par (document_id, page_no)."""
    best: Dict[str, PageRetrievalHit] = {}
    for hit in hits:
        existing = best.get(hit.page_key)
        if existing is None or hit.score > existing.score:
            best[hit.page_key] = hit
        else:
            for src in hit.retrieval_sources:
                if src not in existing.retrieval_sources:
                    existing.retrieval_sources.append(src)
    return list(best.values())


def top_hit_scores(hits: List[PageRetrievalHit], n: int = 5) -> List[float]:
    return [round(h.score, 4) for h in sorted(hits, key=lambda h: h.score, reverse=True)[:n]]


def retrieve_colpali_page_hits(
    session: Session,
    doc_ids: List[int],
    query_text: str,
    limit: int,
) -> List[PageRetrievalHit]:
    if not doc_ids or not settings.COLPALI_ENABLED:
        logger.info("[retrieve_colpali] ignoré — doc_ids=%d colpali_enabled=%s", len(doc_ids), settings.COLPALI_ENABLED)
        return []

    logger.info("[retrieve_colpali] démarrage — %d docs, limit=%d, query=%r", len(doc_ids), limit, query_text[:80])

    from app.services.colpali_service import embed_query_colpali
    from app.services.lancedb_service import search_colpali_lancedb

    query_token_embeddings = embed_query_colpali(query_text)
    if not query_token_embeddings:
        logger.info("[retrieve_colpali] aucun embedding requête")
        return []

    search_results = search_colpali_lancedb(query_token_embeddings, doc_ids, limit=limit)
    if not search_results:
        logger.info("[retrieve_colpali] aucun résultat LanceDB")
        return []

    logger.info("[retrieve_colpali] LanceDB — %d patches candidats", len(search_results))

    chunk_ids = [row["id"] for row in search_results]
    sql_chunks = text("""
        SELECT
            dc.id,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.id IN :chunk_ids
    """)
    chunk_rows = session.execute(sql_chunks, {"chunk_ids": tuple(chunk_ids)}).all()
    rows_map = {row.id: row for row in chunk_rows}

    hits: List[PageRetrievalHit] = []
    for row_lancedb in search_results:
        chunk_id = row_lancedb["id"]
        row = rows_map.get(chunk_id)
        if not row:
            continue
        distance = float(row_lancedb.get("_distance", 1.0))
        similarity = 1.0 - distance
        meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        page_no = _page_no_from_metadata(meta)
        if page_no is None:
            continue
        hits.append(
            PageRetrievalHit(
                document_id=int(row.document_id),
                page_no=page_no,
                score=similarity,
                colpali_score=similarity,
                retrieval_sources=["colpali"],
                document_title=row.document_title or "Document sans titre",
                chunk_id=int(chunk_id),
            )
        )
    hits = _aggregate_hits_by_page(hits)
    if search_results and not hits:
        sample_ids = [row.get("id") for row in search_results[:5]]
        logger.warning(
            "[retrieve_colpali] %d résultats LanceDB mais 0 page mappée — chunk_ids échantillon=%s",
            len(search_results),
            sample_ids,
        )
    logger.info(
        "[retrieve_colpali] %d pages — top scores: %s",
        len(hits),
        top_hit_scores(hits),
    )
    return hits


def retrieve_pgvector_page_hits(
    session: Session,
    doc_ids: List[int],
    query_embedding: List[float],
    limit: int,
) -> List[PageRetrievalHit]:
    if not doc_ids or not query_embedding:
        logger.info(
            "[retrieve_pgvector] ignoré — doc_ids=%d embedding=%s",
            len(doc_ids),
            "ok" if query_embedding else "absent",
        )
        return []

    logger.info("[retrieve_pgvector] démarrage — %d docs, limit=%d", len(doc_ids), limit)

    embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    sql = text("""
        SELECT
            dc.id,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title,
            1 - (dc.embedding <=> CAST(:query_vec AS vector)) AS similarity
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.document_id IN :doc_ids
          AND dc.is_leaf = true
          AND dc.embedding IS NOT NULL
          AND COALESCE(dc.metadata_json->>'content_type', dc.metadata_->>'content_type', '') = 'semantic_leaf'
        ORDER BY dc.embedding <=> CAST(:query_vec AS vector)
        LIMIT :limit
    """)
    rows = session.execute(
        sql,
        {"doc_ids": tuple(doc_ids), "query_vec": embedding_str, "limit": limit},
    ).all()

    hits: List[PageRetrievalHit] = []
    for row in rows:
        meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        page_no = _page_no_from_metadata(meta)
        if page_no is None:
            continue
        sim = float(row.similarity or 0.0)
        hits.append(
            PageRetrievalHit(
                document_id=int(row.document_id),
                page_no=page_no,
                score=sim,
                pgvector_score=sim,
                retrieval_sources=["pgvector"],
                document_title=row.document_title or "Document sans titre",
                chunk_id=int(row.id),
            )
        )
    hits = _aggregate_hits_by_page(hits)
    logger.info(
        "[retrieve_pgvector] %d pages — top scores: %s",
        len(hits),
        top_hit_scores(hits),
    )
    return hits


def retrieve_bm25_page_hits(
    session: Session,
    doc_ids: List[int],
    query_text: str,
    limit: int,
) -> List[PageRetrievalHit]:
    if not doc_ids or not query_text.strip():
        logger.info("[retrieve_bm25] ignoré — doc_ids=%d query vide=%s", len(doc_ids), not query_text.strip())
        return []

    logger.info("[retrieve_bm25] démarrage — %d docs, limit=%d, query=%r", len(doc_ids), limit, query_text[:80])

    sql = text("""
        SELECT
            dc.id,
            dc.document_id,
            dc.metadata_json,
            dc.metadata_,
            d.title AS document_title,
            ts_rank_cd(dc.tsv_content, plainto_tsquery('french', :query)) AS rank
        FROM documentchunk dc
        INNER JOIN document d ON dc.document_id = d.id
        WHERE dc.document_id IN :doc_ids
          AND dc.is_leaf = true
          AND dc.tsv_content @@ plainto_tsquery('french', :query)
        ORDER BY rank DESC
        LIMIT :limit
    """)
    try:
        rows = session.execute(
            sql,
            {"doc_ids": tuple(doc_ids), "query": query_text.strip(), "limit": limit},
        ).all()
    except Exception as exc:
        logger.warning("[BM25] recherche échouée : %s", exc)
        return []

    hits: List[PageRetrievalHit] = []
    for row in rows:
        meta = _merged_chunk_metadata(row.metadata_json, row.metadata_)
        page_no = _page_no_from_metadata(meta)
        if page_no is None:
            continue
        rank = float(row.rank or 0.0)
        hits.append(
            PageRetrievalHit(
                document_id=int(row.document_id),
                page_no=page_no,
                score=rank,
                bm25_score=rank,
                retrieval_sources=["bm25"],
                document_title=row.document_title or "Document sans titre",
                chunk_id=int(row.id),
            )
        )
    hits = _aggregate_hits_by_page(hits)
    logger.info(
        "[retrieve_bm25] %d pages — top scores: %s",
        len(hits),
        top_hit_scores(hits),
    )
    return hits


def fuse_page_hits_rrf(
    colpali_hits: List[PageRetrievalHit],
    pgvector_hits: List[PageRetrievalHit],
    bm25_hits: List[PageRetrievalHit],
    *,
    rrf_k: int = 60,
    top_n: int = 15,
) -> List[PageRetrievalHit]:
    merged: Dict[str, PageRetrievalHit] = {}

    def _add_channel(hits: List[PageRetrievalHit], channel: str) -> None:
        sorted_hits = sorted(hits, key=lambda h: h.score, reverse=True)
        for rank_idx, hit in enumerate(sorted_hits, start=1):
            key = hit.page_key
            if key not in merged:
                merged[key] = PageRetrievalHit(
                    document_id=hit.document_id,
                    page_no=hit.page_no,
                    document_title=hit.document_title,
                    chunk_id=hit.chunk_id,
                    retrieval_sources=[],
                )
            target = merged[key]
            target.rrf_score += 1.0 / (rrf_k + rank_idx)
            if channel not in target.retrieval_sources:
                target.retrieval_sources.append(channel)
            if channel == "colpali":
                target.colpali_score = hit.colpali_score
                target.score = max(target.score, hit.score)
            elif channel == "pgvector":
                target.pgvector_score = hit.pgvector_score
                target.score = max(target.score, hit.score)
            elif channel == "bm25":
                target.bm25_score = hit.bm25_score
                target.score = max(target.score, hit.score)
            if hit.document_title:
                target.document_title = hit.document_title
            if hit.chunk_id is not None:
                target.chunk_id = hit.chunk_id

    _add_channel(colpali_hits, "colpali")
    _add_channel(pgvector_hits, "pgvector")
    _add_channel(bm25_hits, "bm25")

    if not merged:
        return []

    result = sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)
    return result[:top_n]


def build_weak_hit_pool(
    colpali_hits: List[PageRetrievalHit],
    pgvector_hits: List[PageRetrievalHit],
    bm25_hits: List[PageRetrievalHit],
) -> Dict[str, Dict[str, Any]]:
    """Pool élargi pour détecter les weak hits voisins."""
    pool: Dict[str, Dict[str, Any]] = {}
    for hit in colpali_hits + pgvector_hits + bm25_hits:
        key = hit.page_key
        entry = pool.setdefault(
            key,
            {
                "document_id": hit.document_id,
                "page_no": hit.page_no,
                "max_score": 0.0,
                "sources": set(),
            },
        )
        entry["max_score"] = max(entry["max_score"], hit.score)
        entry["sources"].update(hit.retrieval_sources)
    return pool


def load_l1_chunks_for_page(
    session: Session,
    document_id: int,
    page_no: int,
) -> List[DocumentChunk]:
    """Charge tous les semantic_leaf couvrant une page donnée."""
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == True,  # noqa: E712
    )
    all_leaves = list(session.exec(stmt).all())
    result: List[DocumentChunk] = []
    for chunk in all_leaves:
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        if meta.get("content_type") == CONTENT_TYPE_PAGE_ANCHOR:
            continue
        p_start = meta.get("page_start") or meta.get("page_no")
        p_end = meta.get("page_end") or p_start
        try:
            p_start_i = int(p_start) if p_start is not None else None
            p_end_i = int(p_end) if p_end is not None else p_start_i
        except (TypeError, ValueError):
            continue
        if p_start_i is not None and p_end_i is not None and p_start_i <= page_no <= p_end_i:
            result.append(chunk)
    result.sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    return result


def build_consolidated_page_text(chunks: List[DocumentChunk]) -> str:
    """Concatène les L1 d'une ou plusieurs pages en texte structuré."""
    parts: List[str] = []
    for chunk in chunks:
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        content = (chunk.content or chunk.text or "").strip()
        if not content:
            continue
        heading = meta.get("heading") or meta.get("parent_heading")
        step_no = meta.get("step_number")
        if heading:
            parts.append(f"### {heading}")
        elif step_no is not None:
            parts.append(f"### Étape {step_no}")
        parts.append(content)
    return "\n\n".join(parts).strip()


def _chunk_covers_page(meta: dict, page_no: int) -> bool:
    p_start = meta.get("page_start") or meta.get("page_no")
    p_end = meta.get("page_end") or p_start
    try:
        p_start_i = int(p_start) if p_start is not None else None
        p_end_i = int(p_end) if p_end is not None else p_start_i
    except (TypeError, ValueError):
        return False
    return p_start_i is not None and p_end_i is not None and p_start_i <= page_no <= p_end_i


def _has_cross_page_coverage(session: Session, document_id: int, page_no: int) -> bool:
    """True si un chunk L1 couvre déjà page_no..page_no+1 (merge indexation)."""
    for chunk in load_l1_chunks_for_page(session, document_id, page_no):
        meta = _merged_chunk_metadata(chunk.metadata_json, chunk.metadata_)
        if meta.get("cross_page_merge"):
            p_end = meta.get("page_end") or meta.get("page_no")
            try:
                if int(p_end) > page_no:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def expand_neighbor_pages(
    session: Session,
    hit: PageRetrievalHit,
    weak_pool: Dict[str, Dict[str, Any]],
) -> Tuple[List[int], List[int], Optional[str]]:
    """
    Détermine quelles pages inclure (page principale + voisines).
    Retourne (pages_incluses, expanded_neighbor_pages, expansion_reason).
    """
    doc_id = hit.document_id
    page_no = hit.page_no
    pages: Set[int] = {page_no}
    expanded_neighbors: Set[int] = set()
    reason: Optional[str] = None

    if _has_cross_page_coverage(session, doc_id, page_no):
        return sorted(pages), [], None

    l1_chunks = load_l1_chunks_for_page(session, doc_id, page_no)

    # Signal 1 : continues_on_next_page sur le dernier chunk
    if l1_chunks:
        last_meta = _merged_chunk_metadata(l1_chunks[-1].metadata_json, l1_chunks[-1].metadata_)
        if last_meta.get("continues_on_next_page"):
            pages.add(page_no + 1)
            expanded_neighbors.add(page_no + 1)
            reason = reason or "continues_on_next_page"

    # Signal 2 : continues_from_previous_page sur N+1
    next_chunks = load_l1_chunks_for_page(session, doc_id, page_no + 1)
    if next_chunks:
        first_meta = _merged_chunk_metadata(next_chunks[0].metadata_json, next_chunks[0].metadata_)
        if first_meta.get("continues_from_previous_page"):
            pages.add(page_no + 1)
            expanded_neighbors.add(page_no + 1)
            reason = reason or "continues_from_previous_page"

    # Signal 3 : weak retrieval hit dans le pool
    for neighbor in (page_no - 1, page_no + 1):
        if neighbor < 1:
            continue
        nkey = f"{doc_id}:{neighbor}"
        if nkey in weak_pool and neighbor not in pages:
            pages.add(neighbor)
            expanded_neighbors.add(neighbor)
            reason = reason or "weak_hit"

    # Signal 4 : radius config (filet de sécurité)
    if settings.RETRIEVAL_EXPAND_ENABLED and settings.RETRIEVAL_PAGE_RADIUS >= 1:
        ratio = settings.RETRIEVAL_NEIGHBOR_MIN_SCORE_RATIO
        for neighbor in (page_no - 1, page_no + 1):
            if neighbor < 1:
                continue
            nkey = f"{doc_id}:{neighbor}"
            pool_entry = weak_pool.get(nkey)
            if pool_entry and neighbor not in pages:
                if pool_entry["max_score"] >= hit.score * ratio:
                    pages.add(neighbor)
                    expanded_neighbors.add(neighbor)
                    reason = reason or "radius"

    expanded_neighbors.discard(page_no)
    return sorted(pages), sorted(expanded_neighbors), reason


def compute_needs_page_image(
    retrieval_sources: List[str],
    expanded_pages_sources: Optional[Dict[int, Set[str]]] = None,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    PNG requis si ColPali seul (sans pgvector ni bm25) sur une page.
    Retourne (needs_any, list of (doc_id, page_no) needing PNG) — doc_id filled by caller.
    """
    sources_set = set(retrieval_sources or [])
    text_sources = sources_set & {"pgvector", "bm25"}
    if "colpali" in sources_set and not text_sources:
        return True, []
    return False, []


def compute_image_pages_for_passage(
    document_id: int,
    primary_sources: List[str],
    pages_included: List[int],
    weak_pool: Dict[str, Dict[str, Any]],
) -> List[Tuple[int, int]]:
    """Liste les (doc_id, page_no) nécessitant un PNG pour la génération."""
    image_pages: List[Tuple[int, int]] = []
    for pno in pages_included:
        pkey = f"{document_id}:{pno}"
        pool_entry = weak_pool.get(pkey, {})
        sources = set(pool_entry.get("sources") or [])
        if pno == pages_included[0]:
            sources.update(primary_sources)
        text_hit = sources & {"pgvector", "bm25"}
        if "colpali" in sources and not text_hit:
            image_pages.append((document_id, pno))
    return image_pages


def format_hybrid_passages(
    session: Session,
    fused_hits: List[PageRetrievalHit],
    weak_pool: Dict[str, Dict[str, Any]],
    k: int,
) -> List[Dict[str, Any]]:
    """Construit les passages finaux avec L1 consolidés, expansion voisine et flags image."""
    passages: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for hit in fused_hits:
        if len(passages) >= k:
            break

        pages_included, expanded_neighbors, expansion_reason = expand_neighbor_pages(
            session, hit, weak_pool
        )
        passage_key = f"{hit.document_id}:{pages_included[0]}-{pages_included[-1]}"
        if passage_key in seen_keys:
            continue
        seen_keys.add(passage_key)

        all_chunks: List[DocumentChunk] = []
        for pno in pages_included:
            all_chunks.extend(load_l1_chunks_for_page(session, hit.document_id, pno))
        # Dédupliquer par chunk id
        seen_chunk_ids: Set[int] = set()
        unique_chunks: List[DocumentChunk] = []
        for c in all_chunks:
            if c.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(c.id)
            unique_chunks.append(c)
        unique_chunks.sort(key=lambda c: (c.chunk_index or 0, c.id or 0))

        consolidated = build_consolidated_page_text(unique_chunks)
        if not consolidated:
            consolidated = f"Page {hit.page_no} — contenu visuel uniquement"

        page_start = min(pages_included)
        page_end = max(pages_included)
        doc_title = hit.document_title or "Document sans titre"
        passage_text = f"**{doc_title}**\n{consolidated}"

        image_pages = compute_image_pages_for_passage(
            hit.document_id,
            hit.retrieval_sources,
            pages_included,
            weak_pool,
        )
        needs_page_image = len(image_pages) > 0

        out: Dict[str, Any] = {
            "passage": passage_text,
            "passage_raw": consolidated,
            "document_title": doc_title,
            "document_id": hit.document_id,
            "chunk_id": hit.chunk_id,
            "score": float(hit.rrf_score or hit.score),
            "page_no": page_start,
            "page_start": page_start,
            "page_end": page_end,
            "retrieval_sources": list(hit.retrieval_sources),
            "colpali_score": hit.colpali_score,
            "pgvector_score": hit.pgvector_score,
            "bm25_score": hit.bm25_score,
            "raw_rrf_score": hit.rrf_score,
            "expanded_neighbor_pages": expanded_neighbors,
            "expansion_reason": expansion_reason,
            "needs_page_image": needs_page_image,
            "image_pages": image_pages,
            "content_type": "hybrid_page_passage",
        }
        passages.append(out)

    return passages


def _retrieve_leaves_bm25_sql(
    session: Session,
    space_id: int,
    user_id: int,
    query_text: str,
    candidate_k: int,
    query_embedding: Optional[List[float]] = None,
    document_filter: str = "all",
) -> List[Any]:
    """Stub de compatibilité tests — délègue à retrieve_bm25_page_hits."""
    doc_ids = get_space_document_ids(session, space_id, document_filter)
    hits = retrieve_bm25_page_hits(session, doc_ids, query_text, candidate_k)
    from llama_index.core.schema import NodeWithScore, TextNode

    nodes: List[NodeWithScore] = []
    for hit in hits:
        meta = {
            "document_id": hit.document_id,
            "page_no": hit.page_no,
            "document_title": hit.document_title,
            "content_type": CONTENT_TYPE_SEMANTIC_LEAF,
            "retrieval_sources": hit.retrieval_sources,
        }
        node = TextNode(
            id_=f"chunk-{hit.chunk_id or hit.page_key}",
            text="",
            metadata=meta,
        )
        nodes.append(NodeWithScore(node=node, score=hit.score))
    return nodes
