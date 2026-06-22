"""Tests retrieval boost service."""
from unittest.mock import MagicMock, patch

from app.services.page_retrieval_service import UnifiedPageHit
from app.services.query_signals_schemas import LightweightQuerySignals
from app.services.retrieval_boost_service import apply_soft_boosts_to_passages


def test_soft_boosts_categories_on_passages():
    signals = LightweightQuerySignals(
        inferred_categories=["mounting"],
        confidence=0.9,
    )
    passages = [
        {"document_id": 1, "chunk_id": 123, "score": 1.0, "retrieval_sources": ["pgvector"]},
        {"document_id": 2, "chunk_id": 456, "score": 0.95, "retrieval_sources": ["pgvector"]},
    ]

    with patch("app.services.retrieval_boost_service._get_chunk_categories") as mock_cats:
        mock_cats.side_effect = lambda _s, cid: ["mounting"] if cid == 123 else []
        with patch("app.services.retrieval_boost_service._get_document_source", return_value=None):
            with patch("app.services.retrieval_boost_service._get_document_materials", return_value=[]):
                boosted = apply_soft_boosts_to_passages(MagicMock(), passages, signals)

    assert boosted[0]["score"] > 1.0
    assert boosted[1]["score"] == 0.95


def test_soft_boosts_kag_entity_channel():
    signals = LightweightQuerySignals(
        entity_texts=["coulisses", "MONOBLOC"],
        confidence=0.8,
    )
    passages = [
        {"document_id": 1, "score": 0.5, "retrieval_sources": ["kag"]},
    ]

    with patch("app.services.retrieval_boost_service._get_document_source", return_value=None):
        with patch("app.services.retrieval_boost_service._get_document_materials", return_value=[]):
            boosted = apply_soft_boosts_to_passages(MagicMock(), passages, signals)

    assert boosted[0]["score"] > 0.5


def test_apply_soft_boosts_unified_hits_via_passages():
    """Compatibilité : boosts sur dict passages avec document_id."""
    signals = LightweightQuerySignals(
        primary_source="Profine",
        confidence=1.0,
    )
    passages = [{"document_id": 10, "score": 0.4, "retrieval_sources": ["bm25"]}]

    with patch("app.services.retrieval_boost_service._get_document_source", return_value="Profine"):
        with patch("app.services.retrieval_boost_service._get_document_materials", return_value=[]):
            boosted = apply_soft_boosts_to_passages(MagicMock(), passages, signals)

    assert boosted[0]["score"] > 0.4
    assert boosted[0].get("source") == "Profine"
