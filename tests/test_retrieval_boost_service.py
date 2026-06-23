"""Tests retrieval boost service — boosts souples source / matériau / entités KAG.

Les boosts sont exprimés en fraction de l'étendue des scores du pool (scale-adaptatif),
puis ajoutés ; ils n'incluent PAS les catégories (boostées en amont sur rrf_score).
"""
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.services.query_signals_schemas import LightweightQuerySignals
from app.services.retrieval_boost_service import (
    _passage_score_span,
    apply_soft_boosts_to_passages,
)

_SRC = "app.services.retrieval_boost_service._get_document_source"
_MAT = "app.services.retrieval_boost_service._get_document_materials"


# ---------------------------------------------------------------------------
# _passage_score_span
# ---------------------------------------------------------------------------


def test_score_span_uses_range_when_spread():
    passages = [{"score": 0.5}, {"score": 0.1}]
    assert _passage_score_span(passages) == pytest.approx(0.4)


def test_score_span_falls_back_to_magnitude_single_passage():
    assert _passage_score_span([{"score": 0.5}]) == pytest.approx(0.5)


def test_score_span_handles_negative_rerank_logits():
    # logits cross-encoder : -2 .. +3 → étendue 5
    passages = [{"score": 3.0}, {"score": -2.0}]
    assert _passage_score_span(passages) == pytest.approx(5.0)


def test_score_span_defaults_to_one_when_all_zero():
    assert _passage_score_span([{"score": 0.0}, {"score": 0.0}]) == 1.0


# ---------------------------------------------------------------------------
# apply_soft_boosts_to_passages
# ---------------------------------------------------------------------------


def test_soft_boosts_noop_without_signals():
    passages = [{"document_id": 1, "score": 0.5}]
    assert apply_soft_boosts_to_passages(MagicMock(), passages, None) == passages


def test_soft_boosts_kag_entity_channel():
    signals = LightweightQuerySignals(entity_texts=["coulisses", "MONOBLOC"], confidence=0.8)
    passages = [
        {"document_id": 1, "score": 0.5, "retrieval_sources": ["kag"]},
        {"document_id": 2, "score": 0.1, "retrieval_sources": ["pgvector"]},
    ]

    with patch(_SRC, return_value=None), patch(_MAT, return_value=[]):
        boosted = apply_soft_boosts_to_passages(MagicMock(), passages, signals)

    kag_passage = next(p for p in boosted if p["document_id"] == 1)
    assert kag_passage["score"] > 0.5
    assert kag_passage.get("retrieval_boost") is not None


def test_soft_boosts_source_match_increases_and_sets_source():
    signals = LightweightQuerySignals(primary_source="Profine", confidence=1.0)
    passages = [
        {"document_id": 10, "score": 0.5, "retrieval_sources": ["bm25"]},
        {"document_id": 11, "score": 0.1, "retrieval_sources": ["bm25"]},
    ]

    with patch(_SRC, return_value="Profine"), patch(_MAT, return_value=[]):
        boosted = apply_soft_boosts_to_passages(MagicMock(), passages, signals)

    top = boosted[0]
    assert top["document_id"] == 10
    assert top["source"] == "Profine"
    # delta = source_boost_max(0.8) * conf(1.0) * span(0.4)
    expected = 0.5 + settings.RETRIEVAL_SOURCE_BOOST_MAX * 1.0 * 0.4
    assert top["score"] == pytest.approx(expected)


def test_soft_boosts_material_match():
    signals = LightweightQuerySignals(material_hint="pvc", confidence=0.9)
    passages = [
        {"document_id": 1, "score": 0.8, "retrieval_sources": ["pgvector"]},
        {"document_id": 2, "score": 0.0, "retrieval_sources": ["pgvector"]},
    ]

    with patch(_SRC, return_value=None), patch(_MAT, return_value=["PVC"]):
        boosted = apply_soft_boosts_to_passages(MagicMock(), passages, signals)

    p1 = next(p for p in boosted if p["document_id"] == 1)
    # delta = material_boost * span(0.8)
    assert p1["score"] == pytest.approx(0.8 + settings.RETRIEVAL_MATERIAL_BOOST * 0.8)


def test_soft_boosts_noop_when_nothing_matches():
    signals = LightweightQuerySignals(primary_source="Technal", confidence=1.0)
    passages = [
        {"document_id": 1, "score": 0.5, "retrieval_sources": ["pgvector"]},
        {"document_id": 2, "score": 0.1, "retrieval_sources": ["pgvector"]},
    ]

    with patch(_SRC, return_value="Profine"), patch(_MAT, return_value=[]):
        boosted = apply_soft_boosts_to_passages(MagicMock(), passages, signals)

    assert {p["score"] for p in boosted} == {0.5, 0.1}
    assert all("retrieval_boost" not in p for p in boosted)


def test_soft_boosts_scale_with_span():
    """Le même boost source donne un delta proportionnel à l'étendue des scores."""
    signals = LightweightQuerySignals(primary_source="Profine", confidence=1.0)

    small = [
        {"document_id": 10, "score": 0.05, "retrieval_sources": ["bm25"]},
        {"document_id": 11, "score": 0.00, "retrieval_sources": ["bm25"]},
    ]
    large = [
        {"document_id": 10, "score": 5.0, "retrieval_sources": ["bm25"]},
        {"document_id": 11, "score": 0.0, "retrieval_sources": ["bm25"]},
    ]

    with patch(_SRC, return_value="Profine"), patch(_MAT, return_value=[]):
        b_small = apply_soft_boosts_to_passages(MagicMock(), small, signals)
        b_large = apply_soft_boosts_to_passages(MagicMock(), large, signals)

    delta_small = b_small[0]["retrieval_boost"]
    delta_large = b_large[0]["retrieval_boost"]
    assert delta_large > delta_small * 50  # span 5.0 vs 0.05 → ×100
