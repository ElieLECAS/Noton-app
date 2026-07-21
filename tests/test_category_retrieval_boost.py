"""Tests pytest — boost catégorie post-fusion (multiplicatif sur rrf_score)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.models.document_chunk import DocumentChunk
from app.services.page_retrieval_service import (
    PageRetrievalHit,
    UnifiedPageHit,
    fuse_multimodal_hits,
)
from app.services.query_signals_schemas import LightweightQuerySignals
from app.services.retrieval_boost_service import (
    _bulk_get_page_categories,
    apply_category_boost_to_fused_hits,
    apply_soft_boosts_to_passages,
)

_BOOST_PATH = "app.services.retrieval_boost_service._bulk_get_page_categories"


# NB (2026-07-21) : les tests des helpers chunk-level (_categories_from_metadata,
# _category_boost_for_chunk, _bulk_get_chunk_categories) ont été retirés — ces
# fonctions n'existent plus depuis le passage du boost catégorie au niveau PAGE
# (chunkcategoryrelation), le module d'import était cassé.


# ---------------------------------------------------------------------------
# _bulk_get_page_categories — catégories par page via chunkcategoryrelation
# ---------------------------------------------------------------------------


def test_bulk_get_page_categories_groups_by_page():
    session = MagicMock()
    # rows: (document_id, page_no, slug, axis, confidence)
    session.execute.return_value.all.return_value = [
        (1, 2, "mounting", "task", 0.9),
        (1, 2, "warranty", "task", 0.7),
        (3, 5, "regulatory", "task", 1.0),
    ]

    result = _bulk_get_page_categories(session, [(1, 2), (3, 5)])

    assert result == {
        (1, 2): {"mounting": ("task", 0.9), "warranty": ("task", 0.7)},
        (3, 5): {"regulatory": ("task", 1.0)},
    }


def test_bulk_get_page_categories_filters_unwanted_pairs():
    """Le filtre IN doc/page peut sur-récupérer (1,9) — il doit être écarté."""
    session = MagicMock()
    session.execute.return_value.all.return_value = [
        (1, 2, "mounting", "task", 0.8),
        (1, 9, "warranty", "task", 0.8),  # même doc, page non demandée
    ]

    result = _bulk_get_page_categories(session, [(1, 2), (3, 2)])

    assert result == {(1, 2): {"mounting": ("task", 0.8)}}


def test_bulk_get_page_categories_empty():
    session = MagicMock()
    assert _bulk_get_page_categories(session, []) == {}
    session.execute.assert_not_called()


# ---------------------------------------------------------------------------
# apply_category_boost_to_fused_hits — multiplicatif sur rrf_score
# ---------------------------------------------------------------------------


def _unified(doc_id: int, page: int, rrf: float, chunk_id: int = 0) -> UnifiedPageHit:
    return UnifiedPageHit(document_id=doc_id, page_no=page, rrf_score=rrf, chunk_id=chunk_id)


@pytest.fixture
def mounting_signals() -> LightweightQuerySignals:
    return LightweightQuerySignals(inferred_categories=["mounting"], confidence=0.9)


def test_fused_boost_noop_without_signals():
    hits = [_unified(1, 1, 0.05)]
    apply_category_boost_to_fused_hits(MagicMock(), hits, None)
    assert hits[0].rrf_score == 0.05


def test_fused_boost_noop_without_inferred_categories():
    hits = [_unified(1, 1, 0.05)]
    signals = LightweightQuerySignals(inferred_categories=[], confidence=0.9)
    apply_category_boost_to_fused_hits(MagicMock(), hits, signals)
    assert hits[0].rrf_score == 0.05


def test_fused_boost_noop_empty_hits(mounting_signals):
    assert apply_category_boost_to_fused_hits(MagicMock(), [], mounting_signals) == []


def test_fused_boost_multiplicative_single_match(mounting_signals):
    hit = _unified(1, 2, 0.04)
    with patch(_BOOST_PATH, return_value={(1, 2): {"mounting": ("task", 1.0)}}):
        apply_category_boost_to_fused_hits(MagicMock(), [hit], mounting_signals)
    assert hit.rrf_score == pytest.approx(0.04 * (1 + settings.RETRIEVAL_CATEGORY_BOOST))


def test_fused_boost_multiple_matches_cumulative_factor():
    signals = LightweightQuerySignals(
        inferred_categories=["mounting", "warranty"], confidence=0.9
    )
    hit = _unified(1, 2, 0.04)
    with patch(
        _BOOST_PATH,
        return_value={(1, 2): {"mounting": ("task", 1.0), "warranty": ("task", 1.0)}},
    ):
        apply_category_boost_to_fused_hits(MagicMock(), [hit], signals)
    assert hit.rrf_score == pytest.approx(0.04 * (1 + 2 * settings.RETRIEVAL_CATEGORY_BOOST))


def test_fused_boost_weighted_by_confidence():
    """Un tag faiblement noté booste moins qu'un tag à pleine confiance."""
    signals = LightweightQuerySignals(inferred_categories=["mounting"], confidence=0.9)
    strong = _unified(1, 2, 0.04)
    weak = _unified(2, 2, 0.04)
    with patch(_BOOST_PATH, return_value={(1, 2): {"mounting": ("task", 1.0)}}):
        apply_category_boost_to_fused_hits(MagicMock(), [strong], signals)
    with patch(_BOOST_PATH, return_value={(2, 2): {"mounting": ("task", 0.5)}}):
        apply_category_boost_to_fused_hits(MagicMock(), [weak], signals)
    assert strong.rrf_score == pytest.approx(0.04 * (1 + settings.RETRIEVAL_CATEGORY_BOOST * 1.0))
    assert weak.rrf_score == pytest.approx(0.04 * (1 + settings.RETRIEVAL_CATEGORY_BOOST * 0.5))
    assert weak.rrf_score < strong.rrf_score


def test_fused_boost_case_insensitive(mounting_signals):
    hit = _unified(1, 2, 0.04)
    with patch(_BOOST_PATH, return_value={(1, 2): {"Mounting": ("task", 1.0)}}):
        apply_category_boost_to_fused_hits(MagicMock(), [hit], mounting_signals)
    assert hit.rrf_score > 0.04


def test_fused_boost_no_match_unchanged(mounting_signals):
    hit = _unified(1, 2, 0.04)
    with patch(_BOOST_PATH, return_value={(1, 2): {"warranty": ("task", 1.0)}}):
        apply_category_boost_to_fused_hits(MagicMock(), [hit], mounting_signals)
    assert hit.rrf_score == 0.04


def test_fused_boost_resorts_and_promotes_categorized_page(mounting_signals):
    """Une page légèrement en dessous remonte au-dessus après boost catégorie."""
    high = _unified(1, 1, 0.030)   # pas de catégorie
    low = _unified(1, 2, 0.028)    # catégorisée mounting
    hits = [high, low]

    with patch(_BOOST_PATH, return_value={(1, 1): {}, (1, 2): {"mounting": ("task", 1.0)}}):
        apply_category_boost_to_fused_hits(MagicMock(), hits, mounting_signals)

    # 0.028 * 1.15 = 0.0322 > 0.030
    assert hits[0].page_no == 2
    assert hits[0].final_rank == 1
    assert hits[1].final_rank == 2


def test_fused_boost_works_on_page_retrieval_hit_without_final_rank(mounting_signals):
    """PageRetrievalHit n'a pas de final_rank : pas d'erreur, rrf_score boosté."""
    hit = PageRetrievalHit(document_id=1, page_no=2, rrf_score=0.04, chunk_id=10)
    assert not hasattr(hit, "final_rank")

    with patch(_BOOST_PATH, return_value={(1, 2): {"mounting": ("task", 1.0)}}):
        apply_category_boost_to_fused_hits(MagicMock(), [hit], mounting_signals)

    assert hit.rrf_score == pytest.approx(0.04 * (1 + settings.RETRIEVAL_CATEGORY_BOOST))


def test_fused_boost_custom_magnitude_from_settings(mounting_signals):
    hit = _unified(1, 2, 0.04)
    with patch.object(settings, "RETRIEVAL_CATEGORY_BOOST", 0.5):
        with patch(_BOOST_PATH, return_value={(1, 2): {"mounting": ("task", 1.0)}}):
            apply_category_boost_to_fused_hits(MagicMock(), [hit], mounting_signals)
    assert hit.rrf_score == pytest.approx(0.04 * 1.5)


def test_fused_boost_colpali_page_via_page_level_lookup(mounting_signals):
    """Un hit ColPali porte le chunk_id de l'ancre L0 (sans catégories) : le lookup
    par page le booste quand même."""
    colpali_hit = _unified(7, 3, 0.05, chunk_id=999)  # 999 = ancre L0
    with patch(_BOOST_PATH, return_value={(7, 3): {"mounting": ("task", 1.0)}}):
        apply_category_boost_to_fused_hits(MagicMock(), [colpali_hit], mounting_signals)
    assert colpali_hit.rrf_score == pytest.approx(0.05 * (1 + settings.RETRIEVAL_CATEGORY_BOOST))


# ---------------------------------------------------------------------------
# Intégration : fusion RRF puis boost
# ---------------------------------------------------------------------------


def test_boost_after_fusion_changes_top_rank(mounting_signals):
    colpali = [
        _unified_for_fusion(1, 1, 0.55),
        _unified_for_fusion(1, 2, 0.50),
    ]
    fused = fuse_multimodal_hits(colpali, [], [], top_k=2)
    assert fused[0].page_no == 1  # meilleur score ColPali avant boost

    with patch(_BOOST_PATH, return_value={(1, 1): {}, (1, 2): {"mounting": ("task", 1.0)}}):
        apply_category_boost_to_fused_hits(MagicMock(), fused, mounting_signals)

    # page 2 (catégorisée) doit pouvoir repasser devant si l'écart rrf est faible
    assert fused[0].page_no == 2


def _unified_for_fusion(doc_id: int, page: int, colpali_score: float) -> UnifiedPageHit:
    return UnifiedPageHit(
        document_id=doc_id,
        page_no=page,
        colpali_score=colpali_score,
        chunk_id=page,
        retrieval_sources=["colpali"],
    )


# ---------------------------------------------------------------------------
# Post-retrieval : les catégories ne doivent PAS être boostées ici
# ---------------------------------------------------------------------------


def test_soft_boosts_does_not_apply_category_boost():
    signals = LightweightQuerySignals(inferred_categories=["mounting"], confidence=0.9)
    passages = [
        {"document_id": 1, "chunk_id": 123, "score": 1.0, "retrieval_sources": ["pgvector"]},
        {"document_id": 2, "chunk_id": 456, "score": 0.95, "retrieval_sources": ["pgvector"]},
    ]

    with patch("app.services.retrieval_boost_service._get_document_source", return_value=None):
        with patch("app.services.retrieval_boost_service._get_document_materials", return_value=[]):
            boosted = apply_soft_boosts_to_passages(MagicMock(), passages, signals)

    assert {p["score"] for p in boosted} == {1.0, 0.95}
    assert all("retrieval_boost" not in p for p in boosted)
