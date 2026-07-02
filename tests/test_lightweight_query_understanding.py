"""Tests lightweight query understanding."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.lightweight_query_understanding import run_lightweight_understanding


@pytest.mark.asyncio
@patch("app.config.settings.QUERY_FUSED_UNDERSTANDING_ENABLED", False)
@patch("app.services.lightweight_query_understanding.decide_retrieval_route")
@patch("app.services.lightweight_query_understanding.chat")
async def test_lightweight_ready_generates_queries(mock_chat, mock_route):
    from app.services.query_reasoning_service import RetrievalDecision

    mock_route.return_value = RetrievalDecision(decision="rag", reasoning="technique")

    async def chat_side_effect(*args, **kwargs):
        context = kwargs.get("context") or []
        system = context[0].get("content", "") if context else ""
        if "Extrais les signaux" in system or "ENTITIES" in system:
            payload = {
                "intent": "troubleshooting",
                "entities": [{"text": "coulisses"}, {"text": "MONOBLOC"}],
                "inferred_categories": ["mounting"],
                "confidence": 0.9,
            }
        elif "too_vague" in system.lower() or "évalues si une demande" in system.lower():
            payload = {"too_vague": False, "reasoning": "précis", "clarification_question": ""}
        else:
            payload = {
                "colpali": "notice coulisses dormant monobloc schéma montage",
                "semantic": "Problème mise en place coulisses dormant MONOBLOC",
                "lexical": "coulisses MONOBLOC montage",
                "reasoning": "test",
            }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    mock_chat.side_effect = chat_side_effect
    session = MagicMock()

    result = await run_lightweight_understanding(
        user_message="J'ai un souci mise en place des coulisses sur un dormant MONOBLOC",
        history=[],
        session=session,
    )

    assert result.ready_for_retrieval is True
    assert result.retrieval_queries is not None
    assert result.retrieval_queries.colpali
    assert any("coulisse" in t.lower() for t in result.signals.entity_texts)


@pytest.mark.asyncio
@patch("app.config.settings.QUERY_FUSED_UNDERSTANDING_ENABLED", False)
@patch("app.services.lightweight_query_understanding.decide_retrieval_route")
@patch("app.services.lightweight_query_understanding.chat")
async def test_lightweight_vague_requests_clarification(mock_chat, mock_route):
    from app.services.query_reasoning_service import RetrievalDecision

    mock_route.return_value = RetrievalDecision(decision="rag", reasoning="technique")

    async def chat_side_effect(*args, **kwargs):
        context = kwargs.get("context") or []
        system = context[0].get("content", "") if context else ""
        if "ENTITIES" in system or "Extrais les signaux" in system:
            payload = {"entities": [], "inferred_categories": [], "confidence": 0.3}
        elif "too_vague" in system.lower() or "évalues si une demande" in system.lower():
            payload = {
                "too_vague": True,
                "reasoning": "trop vague",
                "clarification_question": "Pouvez-vous préciser le symptôme rencontré ?",
            }
        else:
            payload = {}
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    mock_chat.side_effect = chat_side_effect

    result = await run_lightweight_understanding(
        user_message="J'ai un problème",
        history=[],
        session=MagicMock(),
    )

    assert result.ready_for_retrieval is False
    assert result.clarification is not None
    assert "symptôme" in result.clarification.question.lower()


@pytest.mark.asyncio
@patch("app.config.settings.QUERY_FUSED_UNDERSTANDING_ENABLED", False)
@patch("app.services.lightweight_query_understanding.decide_retrieval_route")
async def test_lightweight_direct_route(mock_route):
    from app.services.query_reasoning_service import RetrievalDecision

    mock_route.return_value = RetrievalDecision(decision="direct", reasoning="salutation")

    result = await run_lightweight_understanding(
        user_message="Bonjour",
        history=[],
        session=MagicMock(),
    )

    assert result.route == "direct"
    assert result.ready_for_retrieval is False
