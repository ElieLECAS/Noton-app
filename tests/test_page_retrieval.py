"""Tests unitaires du retrieval hybride page-level."""
from __future__ import annotations

from unittest import mock

import pytest

from app.services.page_retrieval_service import (
    PageRetrievalHit,
    UnifiedPageHit,
    build_consolidated_page_text,
    compute_image_pages_for_passage,
    expand_neighbor_pages,
    format_multimodal_passages,
    fuse_page_hits_rrf,
)


def _hit(doc_id: int, page: int, score: float, source: str) -> PageRetrievalHit:
    h = PageRetrievalHit(
        document_id=doc_id,
        page_no=page,
        score=score,
        document_title="Doc",
        retrieval_sources=[source],
    )
    if source == "colpali":
        h.colpali_score = score
    elif source == "pgvector":
        h.pgvector_score = score
    elif source == "bm25":
        h.bm25_score = score
    return h


def test_fuse_page_hits_rrf_merges_sources():
    colpali = [_hit(1, 2, 0.9, "colpali")]
    pgvector = [_hit(1, 2, 0.8, "pgvector")]
    bm25 = [_hit(1, 3, 0.5, "bm25")]

    fused = fuse_page_hits_rrf(colpali, pgvector, bm25, top_n=5)
    by_key = {h.page_key: h for h in fused}

    assert "1:2" in by_key
    assert set(by_key["1:2"].retrieval_sources) == {"colpali", "pgvector"}
    assert "1:3" in by_key


def test_compute_image_pages_colpali_only():
    weak_pool = {
        "10:5": {"max_score": 0.8, "sources": {"colpali"}},
        "10:6": {"max_score": 0.4, "sources": {"colpali", "pgvector"}},
    }
    images = compute_image_pages_for_passage(
        document_id=10,
        primary_sources=["colpali"],
        pages_included=[5, 6],
        weak_pool=weak_pool,
    )
    assert images == [(10, 5)]


def test_expand_neighbor_pages_continues_on_next_page():
    session = mock.MagicMock()
    chunk = mock.MagicMock()
    chunk.id = 1
    chunk.chunk_index = 1
    chunk.content = "Suite de l'étape"
    chunk.text = None
    chunk.metadata_json = {
        "page_no": 1,
        "page_start": 1,
        "page_end": 1,
        "continues_on_next_page": True,
        "content_type": "semantic_leaf",
    }
    chunk.metadata_ = None

    with mock.patch(
        "app.services.page_retrieval_service.load_l1_chunks_for_page",
        side_effect=lambda _s, doc_id, pno: [chunk] if pno == 1 else [],
    ), mock.patch(
        "app.services.page_retrieval_service._has_cross_page_coverage",
        return_value=False,
    ):
        hit = _hit(1, 1, 0.9, "pgvector")
        pages, expanded, reason = expand_neighbor_pages(session, hit, {})
        assert 2 in pages
        assert 2 in expanded
        assert reason == "continues_on_next_page"


def test_expand_neighbor_pages_enrichment_span():
    """Un chunk contextuel retrouvé déplie ses source_pages (chemin hybrid legacy)."""
    session = mock.MagicMock()

    with mock.patch(
        "app.services.page_retrieval_service.load_l1_chunks_for_page",
        side_effect=lambda _s, _d, _pno: [],
    ), mock.patch(
        "app.services.page_retrieval_service._has_cross_page_coverage",
        return_value=False,
    ):
        hit = _hit(1, 3, 0.9, "pgvector")
        hit.enrichment_source_pages = [3, 4, 5]
        pages, expanded, reason = expand_neighbor_pages(session, hit, {})

    assert pages == [3, 4, 5]
    assert set(expanded) == {4, 5}
    assert reason == "enrichment_span"


def test_format_multimodal_passage_keeps_matched_page_no():
    """Garde-fou : page_no = page ancre matchée, pas min(span) après dépliage enrichissement.

    Un hit page 3 dont un chunk d'enrichissement déplie le batch [1,2,3] doit produire
    un passage avec page_no=3 (page réelle), page_start=1, page_end=3 (span). Régression
    historique : page_no était écrasé par page_start=1, ce qui faussait éval et citations.
    """
    session = mock.MagicMock()
    leaf = mock.MagicMock()
    leaf.id = 42
    leaf.chunk_index = 0
    leaf.content = "Contenu de la page pertinente."
    leaf.text = None
    leaf.metadata_json = {"page_no": 3, "content_type": "semantic_leaf"}
    leaf.metadata_ = None

    hit = UnifiedPageHit(
        document_id=7,
        page_no=3,
        pgvector_score=0.8,
        rrf_score=0.05,
        document_title="Doc",
        retrieval_sources=["pgvector"],
        text_chunks=[leaf],
        neighbor_pages=[1, 2],  # dépliage enrichissement du batch [1,2,3]
    )

    with mock.patch(
        "app.services.page_retrieval_service.load_l1_chunks_for_page",
        return_value=[leaf],
    ), mock.patch(
        "app.services.page_retrieval_service.load_enrichment_chunks_for_pages",
        return_value=[],
    ), mock.patch(
        "app.services.page_reranker_service.compute_page_image_policy",
        return_value=False,
    ):
        passages, images = format_multimodal_passages(session, [hit])

    assert len(passages) == 1
    passage = passages[0]
    assert passage["page_no"] == 3
    assert passage["page_start"] == 1
    assert passage["page_end"] == 3


def test_build_consolidated_page_text_with_headings():
    chunk = mock.MagicMock()
    chunk.content = "Contenu étape."
    chunk.text = None
    chunk.metadata_json = {"heading": "Étape 1", "step_number": 1}
    chunk.metadata_ = None
    text = build_consolidated_page_text([chunk])
    assert "### Étape 1" in text
    assert "Contenu étape." in text
