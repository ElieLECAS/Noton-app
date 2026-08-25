"""Tests reranker page-level : ColPali-dominant, protection slots, PNG policy, K dynamique."""
from __future__ import annotations

from unittest import mock

import pytest

from app.services.page_retrieval_service import UnifiedPageHit
from app.services.page_reranker_service import (
    compute_page_image_policy,
    is_colpali_visual_priority,
    protect_colpali_visual_hits,
    rerank_unified_page_hits,
)


def _hit(
    doc_id: int,
    page: int,
    *,
    sources: list[str],
    colpali: float | None = None,
    bm25: float | None = None,
    rrf: float = 0.01,
) -> UnifiedPageHit:
    return UnifiedPageHit(
        document_id=doc_id,
        page_no=page,
        document_title="Doc",
        retrieval_sources=sources,
        colpali_score=colpali,
        bm25_score=bm25,
        rrf_score=rrf,
    )


def test_is_colpali_visual_priority_colpali_only():
    hit = _hit(1, 5, sources=["colpali"], colpali=0.6)
    assert is_colpali_visual_priority(hit) is True


def test_is_colpali_visual_priority_dominant_weak_text():
    """ColPali fort + bm25 faible → priorité visuelle même si bm25 a matché."""
    hit = _hit(
        1,
        12,
        sources=["colpali", "bm25"],
        colpali=0.72,
        bm25=0.05,
    )
    assert is_colpali_visual_priority(hit) is True


def test_is_colpali_visual_priority_not_dominant_when_text_strong():
    hit = _hit(
        1,
        12,
        sources=["colpali", "bm25"],
        colpali=0.72,
        bm25=0.40,
    )
    assert is_colpali_visual_priority(hit) is False


def test_is_colpali_visual_priority_colpali_too_weak_for_dominance():
    hit = _hit(
        1,
        12,
        sources=["colpali", "bm25"],
        colpali=0.50,
        bm25=0.10,
    )
    assert is_colpali_visual_priority(hit) is False


def test_compute_page_image_policy_colpali_only():
    hit = _hit(1, 3, sources=["colpali"], colpali=0.8)
    assert compute_page_image_policy(hit, "Page 3 — contenu visuel uniquement") is True


def test_compute_page_image_policy_dominant_placeholder():
    hit = _hit(
        1,
        7,
        sources=["colpali", "bm25"],
        colpali=0.65,
        bm25=0.20,
    )
    assert compute_page_image_policy(hit, "Page 7 — contenu visuel uniquement") is True


def test_compute_page_image_policy_dominant_rich_text_no_png():
    hit = _hit(
        1,
        7,
        sources=["colpali", "bm25"],
        colpali=0.65,
        bm25=0.20,
    )
    rich = "Section 4.2 — Réglage tension ressort report de charge ROTO NX NT Designo II"
    assert compute_page_image_policy(hit, rich) is False


def test_protect_colpali_visual_hits_injects_dominant_slot():
    reranked = [_hit(1, 1, sources=["bm25"], bm25=0.9, rrf=0.05)]
    pool = [
        reranked[0],
        _hit(
            1,
            112,
            sources=["colpali", "bm25"],
            colpali=0.78,
            bm25=0.08,
            rrf=0.02,
        ),
    ]
    final, protected = protect_colpali_visual_hits(reranked, pool, max_slots=2)
    assert len(protected) == 1
    assert protected[0].page_no == 112
    assert any(h.page_no == 112 for h in final)


@pytest.mark.asyncio
async def test_rerank_unified_page_hits_dynamic_k_zero():
    from app.services import reranker_service
    from llama_index.core.schema import NodeWithScore, TextNode

    hit = _hit(1, 1, sources=["bm25"], bm25=0.5, rrf=0.01)
    session = mock.MagicMock()

    low_conf = reranker_service.RerankResult(
        nodes=[],
        status="low_confidence_clarification",
        reason="stutter",
        raw_scores=[0.1, 0.09],
        softmax_scores=[0.5, 0.5],
        gap_top1_top2=0.01,
        zscore_flatness=0.02,
    )

    with mock.patch(
        "app.services.page_reranker_service._prepare_rerank_text",
        return_value="texte page",
    ), mock.patch(
        "app.services.page_reranker_service.rerank_nodes",
        return_value=[
            (NodeWithScore(node=TextNode(id_="p", text="t"), score=0.1), 0.1),
        ],
    ), mock.patch(
        "app.services.page_reranker_service.apply_dynamic_filtering",
        return_value=low_conf,
    ):
        final, result, protected = await rerank_unified_page_hits(
            session,
            "query test",
            [hit],
            max_k=12,
        )

    assert result.status == "low_confidence_clarification"
    assert final == []
    assert protected == []
