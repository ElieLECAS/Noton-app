"""Gating ColPali — ColPali (retriever visuel coûteux sur CPU) ne tourne que si utile.

Couvre :
  * should_use_colpali : marqueurs visuels, intents forçants, requêtes texte écartées,
    flags (gating off, ColPali désactivé).
  * _run_retrievers : use_colpali=False n'appelle PAS retrieve_colpali_pages.
"""
from __future__ import annotations

from unittest import mock

import pytest

from app.services import space_search_service as sss
from app.services.query_signals_schemas import LightweightQuerySignals


# ---------------------------------------------------------------------------
# should_use_colpali
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,intent,expected,reason_contains",
    [
        ("Quelles sont les dimensions du dormant 6101 ?", "specification", False, "text_query_skipped"),
        ("différence entre Perform 70 et 76", "product_selection", False, "text_query_skipped"),
        ("parle moi de la gamme Perform", None, False, "text_query_skipped"),
        ("montre-moi le schéma de montage", "documentation", True, "visual_marker"),
        ("où se trouve la vis de réglage", "troubleshooting", True, "visual_marker"),
        ("voir la coupe verticale du seuil", "specification", True, "visual_marker"),
        ("comment poser le seuil PMR", "installation", True, "intent:installation"),
    ],
)
def test_should_use_colpali_decisions(query, intent, expected, reason_contains):
    signals = LightweightQuerySignals(intent=intent) if intent else None
    with mock.patch.object(sss.settings, "COLPALI_ENABLED", True), mock.patch.object(
        sss.settings, "COLPALI_GATING_ENABLED", True
    ), mock.patch.object(sss.settings, "COLPALI_GATING_INTENTS", "installation"):
        use, reason = sss.should_use_colpali(query, signals)
    assert use is expected
    assert reason_contains in reason


def test_gating_off_always_runs_colpali():
    with mock.patch.object(sss.settings, "COLPALI_ENABLED", True), mock.patch.object(
        sss.settings, "COLPALI_GATING_ENABLED", False
    ):
        use, reason = sss.should_use_colpali("dimensions du 6101", None)
    assert use is True
    assert reason == "gating_off"


def test_colpali_disabled_never_runs():
    with mock.patch.object(sss.settings, "COLPALI_ENABLED", False), mock.patch.object(
        sss.settings, "COLPALI_GATING_ENABLED", True
    ):
        use, reason = sss.should_use_colpali("montre-moi le schéma", None)
    assert use is False
    assert reason == "colpali_disabled"


def test_configurable_intents_widen_scope():
    """Ajouter 'specification' aux intents forçants réactive ColPali sur ces requêtes."""
    signals = LightweightQuerySignals(intent="specification")
    with mock.patch.object(sss.settings, "COLPALI_ENABLED", True), mock.patch.object(
        sss.settings, "COLPALI_GATING_ENABLED", True
    ), mock.patch.object(sss.settings, "COLPALI_GATING_INTENTS", "installation,specification"):
        use, reason = sss.should_use_colpali("dimensions maximales d'un vantail", signals)
    assert use is True
    assert reason == "intent:specification"


# ---------------------------------------------------------------------------
# _run_retrievers — use_colpali=False n'exécute pas ColPali
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_retrievers_skips_colpali_when_gated(db_session):
    """use_colpali=False → retrieve_colpali_pages n'est jamais appelé (0 encode/MaxSim)."""
    import app.services.page_retrieval_service as prs

    with mock.patch.object(sss.settings, "RETRIEVAL_PARALLEL_ENABLED", False), mock.patch.object(
        sss.settings, "KAG_ENABLED", False
    ), mock.patch.object(
        prs, "retrieve_colpali_pages", side_effect=AssertionError("ColPali ne doit PAS tourner")
    ), mock.patch.object(
        prs, "retrieve_pgvector_pages", return_value=[]
    ), mock.patch.object(
        prs, "retrieve_bm25_pages", return_value=[]
    ):
        colpali_hits, pgvector_hits, bm25_hits, kag_hits = await sss._run_retrievers(
            db_session,
            space_id=1,
            doc_ids=[1, 2],
            colpali_q="q",
            semantic_q="q",
            lexical_q="q",
            query_embedding=[0.0] * 1024,
            pool_size=20,
            use_colpali=False,
        )

    assert colpali_hits == []


@pytest.mark.asyncio
async def test_run_retrievers_runs_colpali_when_enabled(db_session):
    import app.services.page_retrieval_service as prs

    sentinel = object()
    with mock.patch.object(sss.settings, "RETRIEVAL_PARALLEL_ENABLED", False), mock.patch.object(
        sss.settings, "KAG_ENABLED", False
    ), mock.patch.object(
        prs, "retrieve_colpali_pages", return_value=[sentinel]
    ), mock.patch.object(
        prs, "filter_colpali_pages_dynamic", side_effect=lambda x: x
    ), mock.patch.object(
        prs, "retrieve_pgvector_pages", return_value=[]
    ), mock.patch.object(
        prs, "retrieve_bm25_pages", return_value=[]
    ):
        colpali_hits, *_ = await sss._run_retrievers(
            db_session,
            space_id=1,
            doc_ids=[1, 2],
            colpali_q="q",
            semantic_q="q",
            lexical_q="q",
            query_embedding=[0.0] * 1024,
            pool_size=20,
            use_colpali=True,
        )

    assert colpali_hits == [sentinel]
