"""
Reranker MiniLM au niveau page : cross-encoder sur texte L1 consolidé,
K dynamique 0–12, protection des pages visuelles ColPali-dominantes.
Canaux texte : BM25 seul (ColPali reste le canal visuel).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

from llama_index.core.schema import NodeWithScore, TextNode
from sqlmodel import Session

from app.config import settings
from app.models.document import Document
from app.services.page_retrieval_service import (
    UnifiedPageHit,
    build_consolidated_page_text,
    load_l1_chunks_for_page,
)
from app.services.rag_generation_service import extract_page_text_from_pdf
from app.services.reranker_service import RerankResult, apply_dynamic_filtering, rerank_nodes

logger = logging.getLogger(__name__)

VISUAL_PLACEHOLDER_MARKER = "contenu visuel uniquement"


def is_colpali_visual_priority(
    hit: UnifiedPageHit,
    *,
    min_colpali_score: Optional[float] = None,
    dominance_min_score: Optional[float] = None,
    bm25_weak_threshold: Optional[float] = None,
) -> bool:
    """
    Page visuelle prioritaire ColPali :
    - ColPali-only (sans bm25), ou
    - ColPali fort ET score bm25 faible (même si bm25 a matché).
    """
    min_colpali_score = (
        min_colpali_score if min_colpali_score is not None else settings.COLPALI_POST_FUSION_MIN_SCORE
    )
    dominance_min_score = (
        dominance_min_score if dominance_min_score is not None else settings.COLPALI_DOMINANCE_MIN_SCORE
    )
    bm25_weak_threshold = (
        bm25_weak_threshold if bm25_weak_threshold is not None else settings.COLPALI_BM25_WEAK_THRESHOLD
    )

    if "colpali" not in (hit.retrieval_sources or []):
        return False

    colpali = hit.colpali_score or 0.0
    if colpali < min_colpali_score:
        return False

    text_sources = set(hit.retrieval_sources or []) & {"bm25"}
    if not text_sources:
        return True

    if colpali < dominance_min_score:
        return False

    bm25 = hit.bm25_score or 0.0

    if bm25 >= bm25_weak_threshold:
        return False

    return True


def _is_visual_placeholder_text(text: str) -> bool:
    normalized = (text or "").strip().lower()
    if not normalized:
        return True
    if VISUAL_PLACEHOLDER_MARKER in normalized:
        return True
    if normalized.startswith("page ") and len(normalized) < 80:
        return True
    return False


def compute_page_image_policy(hit: UnifiedPageHit, consolidated_text: str) -> bool:
    """
    PNG requis si la page est priorité visuelle ColPali (seul ou dominant)
    ou si le texte L1 est un placeholder visuel malgré un match texte faible.
    """
    if not is_colpali_visual_priority(hit):
        return False

    text_sources = set(hit.retrieval_sources or []) & {"bm25"}
    if not text_sources:
        return True

    return _is_visual_placeholder_text(consolidated_text)


def _prepare_rerank_text(session: Session, hit: UnifiedPageHit) -> str:
    """Charge L1 et enrichit pymupdf si page ColPali-dominante sans texte exploitable."""
    if not hit.text_chunks:
        hit.text_chunks = load_l1_chunks_for_page(session, hit.document_id, hit.page_no)

    consolidated = build_consolidated_page_text(hit.text_chunks)
    if consolidated and not _is_visual_placeholder_text(consolidated):
        return consolidated

    if is_colpali_visual_priority(hit):
        doc = session.get(Document, hit.document_id)
        if doc and doc.source_file_path and os.path.exists(doc.source_file_path):
            try:
                pymupdf_text = extract_page_text_from_pdf(doc.source_file_path, hit.page_no)
                if pymupdf_text.strip():
                    title = hit.document_title or "Document sans titre"
                    return f"{title}\n[Page {hit.page_no}]\n{pymupdf_text.strip()}"
            except Exception as exc:
                logger.debug(
                    "Enrichissement pymupdf rerank échoué doc=%s p=%s: %s",
                    hit.document_id,
                    hit.page_no,
                    exc,
                )

    if consolidated:
        return consolidated
    return f"Page {hit.page_no} — {VISUAL_PLACEHOLDER_MARKER}"


def _hit_from_node(node: NodeWithScore, pool_by_key: Dict[str, UnifiedPageHit]) -> Optional[UnifiedPageHit]:
    meta = dict(node.node.metadata or {})
    page_key = meta.get("page_key")
    if not page_key or page_key not in pool_by_key:
        return None

    source = pool_by_key[page_key]
    hit = UnifiedPageHit(
        document_id=source.document_id,
        page_no=source.page_no,
        colpali_score=source.colpali_score,
        bm25_score=source.bm25_score,
        rrf_score=source.rrf_score,
        document_title=source.document_title,
        retrieval_sources=list(source.retrieval_sources),
        chunk_id=source.chunk_id,
        text_chunks=list(source.text_chunks),
        neighbor_pages=list(source.neighbor_pages),
        expansion_reason=source.expansion_reason,
    )
    hit.rerank_score = float(node.score or 0.0)
    meta_raw = meta.get("rerank_raw_score")
    if meta_raw is not None:
        hit.rerank_raw_score = float(meta_raw)
    return hit


def protect_colpali_visual_hits(
    reranked: List[UnifiedPageHit],
    rrf_pool: List[UnifiedPageHit],
    *,
    max_slots: Optional[int] = None,
) -> Tuple[List[UnifiedPageHit], List[UnifiedPageHit]]:
    """
    Réinjecte des pages ColPali-dominantes absentes du top rerank (slots réservés).
    Retourne (liste finale, hits protégés injectés).
    """
    max_slots = max_slots if max_slots is not None else settings.COLPALI_PROTECTED_SLOTS
    if max_slots <= 0:
        return reranked, []

    selected_keys = {h.page_key for h in reranked}
    candidates = [
        h for h in rrf_pool
        if is_colpali_visual_priority(h) and h.page_key not in selected_keys
    ]
    candidates.sort(key=lambda h: h.colpali_score or 0.0, reverse=True)

    protected: List[UnifiedPageHit] = []
    for hit in candidates[:max_slots]:
        protected.append(hit)
        selected_keys.add(hit.page_key)
        reason = "colpali-only" if set(hit.retrieval_sources or []) == {"colpali"} else "colpali-dominant"
        logger.info(
            "[ColPali protect] slot réservé (%s) doc=%s p.%s colpali=%.3f bm25=%s",
            reason,
            hit.document_id,
            hit.page_no,
            hit.colpali_score or 0,
            f"{hit.bm25_score:.3f}" if hit.bm25_score is not None else "—",
        )

    if not protected:
        return reranked, []

    final = list(reranked) + protected
    for rank, hit in enumerate(final, start=1):
        hit.final_rank = rank
    return final, protected


def _pool_in_rerank_order(
    scored: List[Tuple[NodeWithScore, float]],
    pool_by_key: Dict[str, UnifiedPageHit],
) -> List[UnifiedPageHit]:
    """Pool COMPLET remis dans l'ordre des scores de rerank décroissants (B2).

    Le quota par document a besoin de voir AU-DELÀ de la coupe dynamique pour repêcher
    les pages d'autres documents ; la coupe seule ne connaît que le volume."""
    ordered: List[UnifiedPageHit] = []
    seen: set = set()
    for nws, raw_score in sorted(scored, key=lambda t: float(t[1]), reverse=True):
        key = (nws.node.metadata or {}).get("page_key")
        hit = pool_by_key.get(key)
        if hit is None or key in seen:
            continue
        seen.add(key)
        ordered.append(hit)
    return ordered


async def rerank_unified_page_hits(
    session: Session,
    query_text: str,
    hits: List[UnifiedPageHit],
    *,
    max_k: Optional[int] = None,
) -> Tuple[List[UnifiedPageHit], RerankResult, List[UnifiedPageHit]]:
    """
    Rerank MiniLM sur le pool RRF, K dynamique, puis protection ColPali-dominant.
    """
    max_k = max_k if max_k is not None else settings.MAX_DYNAMIC_K
    pool_by_key = {h.page_key: h for h in hits}

    if not hits:
        empty = RerankResult(
            nodes=[],
            status="ok",
            reason="no_candidates",
            raw_scores=[],
            softmax_scores=[],
            gap_top1_top2=None,
            zscore_flatness=None,
        )
        return [], empty, []

    nodes: List[NodeWithScore] = []
    for hit in hits:
        text = _prepare_rerank_text(session, hit)
        meta = {
            "page_key": hit.page_key,
            "document_id": hit.document_id,
            "page_no": hit.page_no,
            "document_title": hit.document_title,
        }
        node = TextNode(id_=f"page-{hit.page_key}", text=text, metadata=meta)
        nodes.append(NodeWithScore(node=node, score=float(hit.rrf_score or 0.0)))

    scored = await rerank_nodes(
        query_text,
        nodes,
        char_cap=settings.RERANK_CHAR_CAP,
        batch_size=settings.RERANK_BATCH_SIZE,
    )

    for nws, raw_score in scored:
        nws.node.metadata["rerank_raw_score"] = raw_score

    rerank_result = apply_dynamic_filtering(
        scored,
        min_k=settings.MIN_DYNAMIC_K,
        max_k=max_k,
        softmax_cum_threshold=settings.SOFTMAX_CUM_THRESHOLD,
        stutter_gap=settings.STUTTER_GAP,
        zscore_flat_threshold=settings.ZSCORE_FLAT_THRESHOLD,
        high_confidence_floor=settings.STUTTER_HIGH_CONFIDENCE_FLOOR,
    )

    reranked: List[UnifiedPageHit] = []
    for nws in rerank_result.nodes:
        hit = _hit_from_node(nws, pool_by_key)
        if hit:
            reranked.append(hit)

    # Quota par document AUSSI sur le chemin reranké (B2, plan 2026-07-29). La coupe
    # dynamique décide COMBIEN de pages passent ; le quota décide LESQUELLES : sans lui,
    # 17 passages sur 19 pouvaient venir du même document (cas mesuré 27/07) et l'élection
    # CAG comme le juge de suffisance n'avaient rien à comparer. On rejoue la sélection sur
    # le pool COMPLET ordonné par score de rerank, à taille inchangée (le k dynamique reste
    # souverain sur le volume), slots ColPali désactivés ici — ils sont réservés juste après.
    if settings.RERANK_PER_DOC_QUOTA_ENABLED and reranked:
        ordered_pool = _pool_in_rerank_order(scored, pool_by_key)
        if ordered_pool:
            from app.services.page_retrieval_service import select_final_hits

            reranked, _ = select_final_hits(
                ordered_pool, len(reranked), colpali_slots=0
            )

    final_hits, protected = protect_colpali_visual_hits(reranked, hits)

    logger.info(
        "[page_reranker] MiniLM — pool=%d → rerank=%d → final=%d (protected=%d, status=%s)",
        len(hits),
        len(reranked),
        len(final_hits),
        len(protected),
        rerank_result.status,
    )

    return final_hits, rerank_result, protected
