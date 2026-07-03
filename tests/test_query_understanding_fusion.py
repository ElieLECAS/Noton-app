"""P0.4 — Fusion de la décision guidée + requêtes déterministes dans la compréhension.

Objectif : UN SEUL appel LLM avant le retrieval (fused), la décision guidée portée par
ce même appel, et les requêtes retriever construites sans LLM.
"""
import json
from unittest import mock

import pytest

from app.services.lightweight_query_understanding import (
    GuidedDecision,
    _node_build_queries_fast,
    _parse_guided_fields,
    run_lightweight_understanding,
)
from app.services.query_reasoning_service import GuidedModeDecision, resolve_guided_mode


def _fused_response(**overrides):
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "Comment poser le seuil PMR 76100 ?",
        "too_vague": False,
        "clarification_question": None,
        "current_topic": "pose seuil PMR 76100",
        "intent": "installation",
        "entities": [{"text": "seuil PMR 76100"}],
        "inferred_categories": [],
    }
    payload.update(overrides)
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


# ---------------------------------------------------------------------------
# 1 seul appel LLM avant le retrieval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@mock.patch("app.config.settings.QUERY_FUSED_UNDERSTANDING_ENABLED", True)
@mock.patch("app.config.settings.QUERY_GENERATE_QUERIES_LLM", False)
@mock.patch("app.config.settings.GUIDED_DECISION_IN_FUSED", False)
async def test_single_llm_call_before_retrieval(db_session):
    fake_chat = mock.AsyncMock(return_value=_fused_response())
    with mock.patch("app.services.lightweight_query_understanding.chat", fake_chat):
        result = await run_lightweight_understanding(
            user_message="comment poser le seuil PMR 76100 ?",
            history=[],
            session=db_session,
        )
    # UN SEUL appel LLM : le fused. Les requêtes sont déterministes (0 appel).
    assert fake_chat.call_count == 1
    assert result.ready_for_retrieval is True
    assert result.retrieval_queries is not None
    assert result.retrieval_queries.semantic  # requête construite sans LLM


@pytest.mark.asyncio
@mock.patch("app.config.settings.QUERY_FUSED_UNDERSTANDING_ENABLED", True)
@mock.patch("app.config.settings.QUERY_GENERATE_QUERIES_LLM", False)
@mock.patch("app.config.settings.GUIDED_DECISION_IN_FUSED", True)
@mock.patch("app.config.settings.GUIDED_FLOW_ENABLED", True)
async def test_guided_decision_in_same_call(db_session):
    fake_chat = mock.AsyncMock(
        return_value=_fused_response(
            is_guided=True,
            flow_kind="howto",
            product_named=False,
            needs_intent_clarification=False,
        )
    )
    with mock.patch("app.services.lightweight_query_understanding.chat", fake_chat):
        result = await run_lightweight_understanding(
            user_message="comment poser le seuil ?",
            history=[],
            session=db_session,
        )
    # Toujours un seul appel LLM, mais la décision guidée en est extraite.
    assert fake_chat.call_count == 1
    assert result.guided.present is True
    assert result.guided.is_guided is True
    assert result.guided.product_named is False


# ---------------------------------------------------------------------------
# _parse_guided_fields (normalisation)
# ---------------------------------------------------------------------------


def test_parse_guided_fields_normalizes():
    parsed = _parse_guided_fields(
        {"is_guided": True, "flow_kind": "DIAGNOSTIC", "detected_symptom": "inconnu_xyz",
         "product_named": False, "needs_intent_clarification": True}
    )
    assert parsed["present"] is True
    assert parsed["flow_kind"] == "diagnostic"
    assert parsed["detected_symptom"] == ""  # slug inconnu → écarté
    assert parsed["product_named"] is False


def test_parse_guided_fields_defaults_product_named_true_when_absent():
    parsed = _parse_guided_fields({"is_guided": False})
    assert parsed["product_named"] is True  # absent → True (comportement historique)


# ---------------------------------------------------------------------------
# requêtes déterministes (0 LLM)
# ---------------------------------------------------------------------------


def test_deterministic_queries_enrich_lexical():
    state = {
        "standalone_question": "dimensions du dormant 6101",
        "signals": {
            "entity_texts": ["dormant 6101", "PVC"],
            "detected_references": ["6101"],
        },
    }
    out = _node_build_queries_fast(state)
    rq = out["retrieval_queries"]
    assert rq["semantic"] == "dimensions du dormant 6101"
    assert rq["colpali"] == "dimensions du dormant 6101"
    # Le canal lexical est enrichi des entités + références.
    assert "dormant 6101" in rq["lexical"] and "PVC" in rq["lexical"]
    assert out["query_strategy"] == "single"


# ---------------------------------------------------------------------------
# resolve_guided_mode (dérive du fused vs fallback)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_guided_mode_uses_fused_without_llm():
    fused = GuidedDecision(
        present=True, is_guided=True, flow_kind="diagnostic",
        detected_symptom="", product_named=False, needs_intent_clarification=True,
    )
    with mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(side_effect=AssertionError("ne doit PAS être appelé")),
    ):
        decision = await resolve_guided_mode(
            "comment régler la hauteur ?", [], fused_guided=fused, topic="réglage hauteur"
        )
    assert decision.is_guided is True
    assert decision.flow_kind == "diagnostic"
    assert decision.product_named is False
    assert decision.needs_intent_clarification is True
    assert decision.topic == "réglage hauteur"


@pytest.mark.asyncio
async def test_resolve_guided_mode_falls_back_when_not_present():
    fused = GuidedDecision(present=False)
    fallback = GuidedModeDecision(is_guided=True, flow_kind="howto", topic="pose")
    with mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(return_value=fallback),
    ) as m:
        decision = await resolve_guided_mode("comment poser ?", [], fused_guided=fused)
    m.assert_awaited_once()
    assert decision.is_guided is True


@pytest.mark.asyncio
async def test_resolve_guided_mode_falls_back_when_none():
    fallback = GuidedModeDecision(is_guided=False)
    with mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(return_value=fallback),
    ) as m:
        decision = await resolve_guided_mode("bonjour", [], fused_guided=None)
    m.assert_awaited_once()
    assert decision.is_guided is False
