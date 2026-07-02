"""Axe 2 — parallélisation des retrievers + plafond du boost catégorie."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from app.services import retrieval_boost_service as rbs
from app.services import space_search_service as sss
from app.services.query_signals_schemas import LightweightQuerySignals


# ---------------------------------------------------------------------------
# Plafond du boost catégorie (RETRIEVAL_CATEGORY_BOOST_MAX)
# ---------------------------------------------------------------------------


def test_category_boost_factor_is_capped():
    hit = SimpleNamespace(document_id=1, page_no=2, rrf_score=1.0, final_rank=1)
    signals = LightweightQuerySignals(
        inferred_categories=["infiltration_eau", "sealing", "casse_quincaillerie"]
    )
    # 3 matches d'axe symptôme (poids 2.0) × confiance 1.0 → facteur brut 1.9.
    cats = {
        "infiltration_eau": ("symptom", 1.0),
        "sealing": ("symptom", 1.0),
        "casse_quincaillerie": ("symptom", 1.0),
    }
    with mock.patch("app.config.settings.RETRIEVAL_CATEGORY_BOOST", 0.15), mock.patch(
        "app.config.settings.RETRIEVAL_CATEGORY_BOOST_MAX", 1.5
    ), mock.patch(
        "app.config.settings.RETRIEVAL_AXIS_BOOST_WEIGHTS", {"symptom": 2.0}
    ), mock.patch.object(
        rbs, "_bulk_get_page_categories", return_value={(1, 2): cats}
    ):
        rbs.apply_category_boost_to_fused_hits(mock.MagicMock(), [hit], signals)

    # Facteur brut 1.9 → plafonné à 1.5.
    assert hit.rrf_score == 1.5


def test_category_boost_below_cap_unaffected():
    hit = SimpleNamespace(document_id=1, page_no=2, rrf_score=1.0, final_rank=1)
    signals = LightweightQuerySignals(inferred_categories=["mounting"])
    cats = {"mounting": ("task", 1.0)}
    with mock.patch("app.config.settings.RETRIEVAL_CATEGORY_BOOST", 0.15), mock.patch(
        "app.config.settings.RETRIEVAL_CATEGORY_BOOST_MAX", 1.5
    ), mock.patch(
        "app.config.settings.RETRIEVAL_AXIS_BOOST_WEIGHTS", {"task": 1.0}
    ), mock.patch.object(
        rbs, "_bulk_get_page_categories", return_value={(1, 2): cats}
    ):
        rbs.apply_category_boost_to_fused_hits(mock.MagicMock(), [hit], signals)

    # Facteur 1 + 0.15 × 1.0 = 1.15, sous le plafond.
    assert abs(hit.rrf_score - 1.15) < 1e-9


# ---------------------------------------------------------------------------
# _run_retrievers — parallèle vs séquentiel équivalents
# ---------------------------------------------------------------------------


def _patch_retrievers():
    import app.services.kag_retrieval_service as krs
    import app.services.page_retrieval_service as prs

    return (
        mock.patch.object(prs, "retrieve_colpali_pages", lambda s, d, q, p: ["colpali"]),
        mock.patch.object(prs, "filter_colpali_pages_dynamic", lambda hits: hits),
        mock.patch.object(prs, "retrieve_pgvector_pages", lambda s, d, e, p: ["pgvector"]),
        mock.patch.object(prs, "retrieve_bm25_pages", lambda s, d, q, p: ["bm25"]),
        mock.patch.object(krs, "retrieve_kag_pages", lambda s, sp, d, q, e, p: ["kag"]),
    )


@pytest.mark.asyncio
async def test_run_retrievers_parallel_matches_sequential(db_session):
    patches = _patch_retrievers()
    with patches[0], patches[1], patches[2], patches[3], patches[4], mock.patch(
        "app.config.settings.KAG_ENABLED", True
    ):
        with mock.patch("app.config.settings.RETRIEVAL_PARALLEL_ENABLED", True):
            par = await sss._run_retrievers(
                db_session, 1, [1, 2], "cq", "sq", "lq", [0.1], 10
            )
        with mock.patch("app.config.settings.RETRIEVAL_PARALLEL_ENABLED", False):
            seq = await sss._run_retrievers(
                db_session, 1, [1, 2], "cq", "sq", "lq", [0.1], 10
            )

    assert par == (["colpali"], ["pgvector"], ["bm25"], ["kag"])
    assert seq == par


@pytest.mark.asyncio
async def test_run_retrievers_skips_kag_when_disabled(db_session):
    patches = _patch_retrievers()
    with patches[0], patches[1], patches[2], patches[3], patches[4], mock.patch(
        "app.config.settings.KAG_ENABLED", False
    ), mock.patch("app.config.settings.RETRIEVAL_PARALLEL_ENABLED", True):
        result = await sss._run_retrievers(
            db_session, 1, [1], "cq", "sq", "lq", [0.1], 10
        )

    assert result == (["colpali"], ["pgvector"], ["bm25"], [])
