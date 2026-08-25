"""Tests unitaires du retrieval hybride page-level."""
from __future__ import annotations

from unittest import mock

from app.services.page_retrieval_service import (
    UnifiedPageHit,
    build_consolidated_page_text,
    format_multimodal_passages,
)


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
        bm25_score=0.8,
        rrf_score=0.05,
        document_title="Doc",
        retrieval_sources=["bm25"],
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
