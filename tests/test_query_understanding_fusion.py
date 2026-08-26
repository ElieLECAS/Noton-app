"""Compréhension fusionnée : UN SEUL appel LLM avant le retrieval, requêtes déterministes.

Refonte Arbre SAV 2026-07-30 : la décision guidée (is_guided/product_named/…) a été
RETIRÉE du prompt fusionné — le RAG répond toujours, l'entrée en diagnostic est
déterministe (guided_entry_index_service).

Nettoyage 2026-08-26 : c'est désormais le SEUL chemin de compréhension (les graphes
legacy et multi-groupe ont été supprimés), et les requêtes retriever se réduisent à
deux canaux (colpali / lexical) depuis le retrait de la voie dense texte.
"""
import json
from unittest import mock

import pytest

from app.services.lightweight_query_understanding import (
    _node_build_queries_fast,
    run_lightweight_understanding,
)


def _fused_response(**overrides):
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "Comment poser le seuil PMR 76100 ?",
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
    assert result.retrieval_queries.colpali  # requête construite sans LLM
    assert result.retrieval_queries.lexical


@pytest.mark.asyncio
async def test_fused_prompt_has_no_guided_fields(db_session):
    """Le bloc F-J (is_guided/product_named/…) ne doit plus apparaître dans le prompt :
    le guidé ne démarre plus depuis une classification du message libre."""
    fake_chat = mock.AsyncMock(return_value=_fused_response())
    with mock.patch("app.services.lightweight_query_understanding.chat", fake_chat):
        result = await run_lightweight_understanding(
            user_message="mon volet roulant est bloqué",
            history=[],
            session=db_session,
        )
    assert fake_chat.call_count == 1
    _, kwargs = fake_chat.call_args
    joined_prompt = " ".join(m.get("content", "") for m in (kwargs.get("context") or []))
    assert "is_guided" not in joined_prompt
    assert "product_named" not in joined_prompt
    assert "needs_intent_clarification" not in joined_prompt
    # Le résultat ne porte plus de décision guidée.
    assert not hasattr(result, "guided")


@pytest.mark.asyncio
async def test_fused_prompt_no_longer_asks_for_vagueness(db_session):
    """Le contrôle de vagueness pré-retrieval a été supprimé (2026-08-26) : il bloquait
    des questions techniques légitimes avant toute recherche."""
    fake_chat = mock.AsyncMock(return_value=_fused_response())
    with mock.patch("app.services.lightweight_query_understanding.chat", fake_chat):
        await run_lightweight_understanding(
            user_message="comment régler la compression ?",
            history=[],
            session=db_session,
        )
    _, kwargs = fake_chat.call_args
    joined_prompt = " ".join(m.get("content", "") for m in (kwargs.get("context") or []))
    assert "too_vague" not in joined_prompt
    assert "clarification_question" not in joined_prompt


@pytest.mark.asyncio
async def test_direct_route_skips_retrieval(db_session):
    fake_chat = mock.AsyncMock(return_value=_fused_response(route="direct"))
    with mock.patch("app.services.lightweight_query_understanding.chat", fake_chat):
        result = await run_lightweight_understanding(
            user_message="bonjour",
            history=[],
            session=db_session,
        )
    assert result.route == "direct"
    assert result.ready_for_retrieval is False


@pytest.mark.asyncio
async def test_invalid_json_falls_back_to_rag(db_session):
    """JSON invalide → repli sûr : on cherche quand même (chercher ne peut pas inventer)."""
    bad = {"choices": [{"message": {"content": "pas du json"}}]}
    fake_chat = mock.AsyncMock(return_value=bad)
    with mock.patch("app.services.lightweight_query_understanding.chat", fake_chat):
        result = await run_lightweight_understanding(
            user_message="dimensions du dormant 6101",
            history=[],
            session=db_session,
        )
    # 1 retry avant repli (P0.3).
    assert fake_chat.call_count == 2
    assert result.route == "rag"
    assert result.ready_for_retrieval is True
    assert result.retrieval_queries.colpali == "dimensions du dormant 6101"


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
    assert rq["colpali"] == "dimensions du dormant 6101"
    # Le canal lexical est enrichi des entités + références.
    assert "dormant 6101" in rq["lexical"] and "PVC" in rq["lexical"]
    # La voie dense texte a disparu : plus de requête « semantic ».
    assert "semantic" not in rq
