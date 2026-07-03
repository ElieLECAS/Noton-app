"""Tests retrieval multimodal page-centric."""
from __future__ import annotations

from unittest import mock

import pytest

from app.services.page_retrieval_service import (
    UnifiedPageHit,
    expand_page_context,
    fuse_multimodal_hits,
)


def _unified_hit(
    doc_id: int,
    page: int,
    score: float,
    source: str,
) -> UnifiedPageHit:
    hit = UnifiedPageHit(
        document_id=doc_id,
        page_no=page,
        document_title="Doc",
        retrieval_sources=[source],
    )
    if source == "colpali":
        hit.colpali_score = score
    elif source == "pgvector":
        hit.pgvector_score = score
    elif source == "bm25":
        hit.bm25_score = score
    return hit


def test_fuse_multimodal_hits_merges_sources():
    colpali = [_unified_hit(1, 2, 0.9, "colpali")]
    pgvector = [_unified_hit(1, 2, 0.8, "pgvector")]
    bm25 = [_unified_hit(1, 3, 0.5, "bm25")]

    fused = fuse_multimodal_hits(colpali, pgvector, bm25, top_k=5)
    by_key = {h.page_key: h for h in fused}

    assert "1:2" in by_key
    assert set(by_key["1:2"].retrieval_sources) == {"colpali", "pgvector"}
    assert "1:3" in by_key
    assert by_key["1:2"].rrf_score > by_key["1:3"].rrf_score


def test_fuse_multimodal_hits_colpali_only_filtering():
    colpali = [_unified_hit(1, 5, 0.2, "colpali")]
    fused = fuse_multimodal_hits(colpali, [], [], top_k=5, min_colpali_score=0.25)
    assert fused == []


def test_fuse_multimodal_hits_colpali_weak_with_pgvector_kept():
    colpali = [_unified_hit(1, 5, 0.15, "colpali")]
    pgvector = [_unified_hit(1, 5, 0.9, "pgvector")]

    fused = fuse_multimodal_hits(colpali, pgvector, [], top_k=5, min_colpali_score=0.25)

    assert len(fused) == 1
    assert fused[0].page_no == 5
    assert len(fused[0].retrieval_sources) == 2


def test_expand_page_context_conditional_neighbor():
    session = mock.MagicMock()
    chunk = mock.MagicMock()
    chunk.id = 1
    chunk.chunk_index = 1
    chunk.content = "Suite"
    chunk.text = None
    chunk.metadata_json = {
        "page_no": 1,
        "page_start": 1,
        "page_end": 1,
        "continues_on_next_page": True,
        "content_type": "semantic_leaf",
    }
    chunk.metadata_ = None
    next_chunk = mock.MagicMock()
    next_chunk.id = 2
    next_chunk.chunk_index = 2
    next_chunk.content = "Page 2"
    next_chunk.text = None
    next_chunk.metadata_json = {"page_no": 2, "content_type": "semantic_leaf"}
    next_chunk.metadata_ = None

    hit = _unified_hit(1, 1, 0.9, "pgvector")

    with mock.patch(
        "app.services.page_retrieval_service.load_l1_chunks_for_page",
        side_effect=lambda _s, _d, pno: [chunk] if pno == 1 else [next_chunk],
    ), mock.patch(
        "app.services.page_retrieval_service._has_cross_page_coverage",
        return_value=False,
    ):
        expanded = expand_page_context(session, [hit], neighbor_strategy="conditional")

    assert len(expanded) == 1
    assert expanded[0].page_no + 1 in expanded[0].neighbor_pages
    assert expanded[0].expansion_reason == "continues_on_next_page"


def test_fuse_multimodal_hits_propagates_enrichment_source_pages():
    pgvector = _unified_hit(1, 3, 0.8, "pgvector")
    pgvector.enrichment_source_pages = [3, 4, 5]
    bm25 = _unified_hit(1, 3, 0.5, "bm25")

    fused = fuse_multimodal_hits([], [pgvector], [bm25], top_k=5)
    by_key = {h.page_key: h for h in fused}

    assert by_key["1:3"].enrichment_source_pages == [3, 4, 5]


def test_expand_page_context_enrichment_span_unfolds_all_pages():
    """Un chunk contextuel retrouvé déplie tout son batch, même sans voisinage."""
    session = mock.MagicMock()

    def _make_chunk(cid: int, page: int):
        c = mock.MagicMock()
        c.id = cid
        c.chunk_index = cid
        c.content = f"Page {page}"
        c.text = None
        c.metadata_json = {"page_no": page, "content_type": "semantic_leaf"}
        c.metadata_ = None
        return c

    chunks_by_page = {
        3: [_make_chunk(30, 3)],
        4: [_make_chunk(40, 4)],
        5: [_make_chunk(50, 5)],
    }

    hit = _unified_hit(1, 3, 0.9, "pgvector")
    hit.enrichment_source_pages = [3, 4, 5]

    with mock.patch(
        "app.services.page_retrieval_service.load_l1_chunks_for_page",
        side_effect=lambda _s, _d, pno: chunks_by_page.get(pno, []),
    ):
        expanded = expand_page_context(session, [hit], neighbor_strategy="none")

    assert expanded[0].neighbor_pages == [4, 5]
    assert expanded[0].expansion_reason == "enrichment_span"
    loaded_pages = {
        (c.metadata_json or {}).get("page_no") for c in expanded[0].text_chunks
    }
    assert loaded_pages == {3, 4, 5}


@pytest.mark.asyncio
async def test_search_multimodal_passages_pipeline():
    from app.services import space_search_service

    session = mock.MagicMock()
    fused_hit = UnifiedPageHit(
        document_id=123,
        page_no=1,
        rrf_score=0.05,
        final_rank=1,
        retrieval_sources=["pgvector"],
        pgvector_score=0.9,
        document_title="Doc1",
    )

    with mock.patch(
        "app.services.space_search_service.get_space_by_id",
        return_value=mock.MagicMock(),
    ), mock.patch(
        "app.services.space_search_service.generate_embedding",
        return_value=[0.1] * 1024,
    ), mock.patch(
        "app.services.page_retrieval_service.get_space_document_ids",
        return_value=[123],
    ), mock.patch(
        "app.services.page_retrieval_service.retrieve_colpali_pages",
        return_value=[],
    ), mock.patch(
        "app.services.page_retrieval_service.retrieve_pgvector_pages",
        return_value=[fused_hit],
    ), mock.patch(
        "app.services.page_retrieval_service.retrieve_bm25_pages",
        return_value=[],
    ), mock.patch(
        "app.services.page_retrieval_service.expand_page_context",
        side_effect=lambda _s, hits, **kwargs: hits,
    ), mock.patch(
        "app.services.page_retrieval_service.format_multimodal_passages",
        return_value=(
            [
                {
                    "passage": "**Doc1**\nContenu page 1.",
                    "passage_raw": "Contenu page 1.",
                    "document_title": "Doc1",
                    "document_id": 123,
                    "score": 0.05,
                    "page_no": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "retrieval_sources": ["pgvector"],
                    "needs_page_image": False,
                    "image_pages": [],
                    "content_type": "multimodal_page_passage",
                }
            ],
            [],
        ),
    ), mock.patch(
        "app.services.space_search_service.settings"
    ) as mock_settings:
        mock_settings.RAG_TOP_K = 10
        mock_settings.RAG_POOL_SIZE = 20
        mock_settings.RRF_K = 60
        mock_settings.RAG_NEIGHBOR_STRATEGY = "conditional"
        mock_settings.RAG_RENDER_ALL_IMAGES = False
        mock_settings.RAG_MAX_IMAGES = 12
        mock_settings.RERANKER_ENABLED = False
        mock_settings.RERANK_POOL = 40
        # Gating ColPali désactivé pour ce test : on conserve l'intention d'origine
        # (ColPali s'exécute — ici mocké à []) sans passer par le chemin de fallback.
        mock_settings.COLPALI_ENABLED = True
        mock_settings.COLPALI_GATING_ENABLED = False

        result = await space_search_service.search_multimodal_passages(
            session=session,
            space_id=1,
            query_text="test multimodal",
            user_id=1,
            k=10,
        )

    assert result["status"] == "ok"
    assert len(result["passages"]) == 1
    assert result["images"] == []
    assert result["total_hits"] == 1
    assert result["dynamic_k"] == 1
    assert result["rerank_status"] == "disabled"


@pytest.mark.asyncio
async def test_search_relevant_passages_uses_multimodal_when_enabled():
    from app.services import space_search_service

    session = mock.MagicMock()
    expected = {"passages": [], "images": [], "status": "ok", "reason": "multimodal_rrf"}

    with mock.patch(
        "app.services.space_search_service.settings"
    ) as mock_settings, mock.patch(
        "app.services.space_search_service.search_multimodal_passages",
        return_value=expected,
    ) as mock_multimodal:
        mock_settings.USE_MULTIMODAL_RETRIEVAL = True

        result = await space_search_service.search_relevant_passages(
            session=session,
            space_id=1,
            query_text="query",
            user_id=1,
            k=10,
        )

    mock_multimodal.assert_called_once()
    assert result == expected


def test_extract_bm25_fallback_query_keeps_brand_terms():
    from app.services.page_retrieval_service import (
        _build_bm25_or_tsquery,
        _extract_bm25_fallback_query,
    )

    query = "Pour régler la tension du ressort d'un report de charge ROTO NX NT Designo II"
    fallback = _extract_bm25_fallback_query(query)
    assert "ROTO" in fallback
    assert " OR " in fallback
    or_tsq = _build_bm25_or_tsquery(fallback.split(" OR "))
    assert "roto" in or_tsq.lower() or "ROTO" in or_tsq


def test_log_multimodal_retrieval_summary():
    # NB : caplog ne capture pas les logs de ce projet (la config logging de
    # l'app remplace le handler racine de pytest). On capture donc directement
    # l'appel logger.info du module.
    from app.services import page_retrieval_service as prs
    from app.services.page_retrieval_service import log_multimodal_retrieval_summary

    hit = UnifiedPageHit(
        document_id=383,
        page_no=12,
        pgvector_score=0.92,
        colpali_score=0.67,
        rrf_score=0.032,
        final_rank=1,
        retrieval_sources=["pgvector", "colpali"],
        document_title="Notice ROTO",
    )
    kag_hit = UnifiedPageHit(
        document_id=425,
        page_no=2,
        kag_score=0.81,
        retrieval_sources=["kag"],
        document_title="DEPLIANT",
    )
    with mock.patch.object(prs.logger, "info") as mock_info:
        log_multimodal_retrieval_summary(
            query_text="test query",
            doc_ids=[383, 384],
            colpali_hits=[hit],
            pgvector_hits=[hit],
            bm25_hits=[],
            kag_hits=[kag_hit],
            pre_kag_fused_hits=[hit],
            fused_hits=[hit, kag_hit],
            final_hits=[hit],
            passages=[{"page_start": 12, "page_end": 12, "page_no": 12}],
            images=["img1"],
            top_k=10,
            pool_size=20,
        )

    logged = "\n".join(str(c.args[0]) for c in mock_info.call_args_list if c.args)
    assert "RAG MULTIMODAL — RÉSUMÉ" in logged
    assert "Triple retriever + graphe" in logged
    assert "doc=383 p.12" in logged
    assert "KAG" in logged
