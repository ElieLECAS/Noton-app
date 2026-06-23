"""Tests query understanding — slot filling, validation, requêtes retriever."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.services.slot_catalog import (
    build_slot_prompt,
    empty_slots,
    next_missing_optional,
    next_missing_required,
)
from app.services.query_understanding_graph import (
    _node_merge_slots,
    _node_validate_and_next,
    run_query_understanding,
    SlotAction,
)


def test_empty_slots_all_required_missing():
    slots = empty_slots()
    assert next_missing_required(slots) == "intent"
    assert next_missing_optional(slots, []) == "product_range"


def test_required_fields_order():
    slots = empty_slots()
    slots["intent"] = "specification"
    assert next_missing_required(slots) == "product_family"
    slots["product_family"] = "fenetres"
    assert next_missing_required(slots) == "material"
    slots["material"] = "pvc"
    assert next_missing_required(slots) is None


def test_optional_fields_respect_skip():
    slots = empty_slots()
    slots.update({"intent": "installation", "product_family": "fenetres", "material": "pvc"})
    assert next_missing_optional(slots, []) == "product_range"
    assert next_missing_optional(slots, ["product_range"]) == "supplier"
    assert next_missing_optional(slots, ["product_range", "supplier"]) == "content_categories"
    assert next_missing_optional(slots, ["product_range", "supplier", "content_categories"]) is None


def test_build_slot_prompt_required_has_no_skip():
    prompt = build_slot_prompt("product_family", phase="required", allow_skip=False)
    assert prompt["field"] == "product_family"
    assert prompt["allow_skip"] is False
    assert len(prompt["choices"]) == 3


def test_build_slot_prompt_optional_has_skip():
    prompt = build_slot_prompt("supplier", phase="optional", allow_skip=True)
    assert prompt["allow_skip"] is True
    assert prompt["skip_label"] == "Passer"
    assert prompt["unknown_label"] == "Je ne sais pas"


def test_merge_slots_unknown_optional_text():
    state = {
        "persisted_context": {
            "slots": {
                **empty_slots(),
                "intent": "specification",
                "product_family": "fenetres",
                "material": "pvc",
            },
            "skipped_optional": [],
            "pending_field": "supplier",
            "phase": "collecting_optional",
        },
        "user_message": "Je ne sais pas",
    }
    result = _node_merge_slots(state)
    assert "supplier" in result["skipped_optional"]


def test_merge_slots_applies_fill_action():
    state = {
        "persisted_context": None,
        "slot_action": {"field": "product_family", "value": "fenetres", "action": "fill"},
        "user_message": "Fenêtres",
    }
    result = _node_merge_slots(state)
    assert result["slots"]["product_family"] == "fenetres"


def test_merge_slots_applies_skip_action():
    state = {
        "persisted_context": {
            "slots": {
                **empty_slots(),
                "intent": "specification",
                "product_family": "fenetres",
                "material": "pvc",
            },
            "skipped_optional": [],
        },
        "slot_action": {"field": "product_range", "value": "", "action": "skip"},
        "user_message": "Passer",
    }
    result = _node_merge_slots(state)
    assert "product_range" in result["skipped_optional"]


def test_validate_not_ready_when_intent_missing():
    state = {
        "slots": empty_slots(),
        "skipped_optional": [],
    }
    result = _node_validate_and_next(state)
    assert result["ready_for_retrieval"] is False
    assert result["pending_field"] == "intent"


def test_validate_ready_when_all_slots_filled_or_skipped():
    state = {
        "slots": {
            **empty_slots(),
            "intent": "installation",
            "product_family": "fenetres",
            "material": "pvc",
            "product_range": "perform",
            "supplier": "profine",
        },
        "skipped_optional": ["content_categories"],
    }
    result = _node_validate_and_next(state)
    assert result["ready_for_retrieval"] is True


@pytest.mark.asyncio
@patch("app.services.query_understanding_graph.decide_retrieval_route")
@patch("app.services.query_understanding_graph.chat")
async def test_vague_question_asks_intent_no_retrieval(mock_chat, mock_route):
    from app.services.query_reasoning_service import RetrievalDecision

    mock_route.return_value = RetrievalDecision(decision="rag", reasoning="technique")
    mock_chat.return_value = {
        "choices": [{"message": {"content": json.dumps({"slots": {}, "new_question": False, "reasoning": "vague"})}}]
    }

    result = await run_query_understanding(user_message="quelle est la cote ?")
    assert result.route == "rag"
    assert result.ready_for_retrieval is False
    assert result.clarification is not None
    assert result.clarification.pending_field == "intent"


@pytest.mark.asyncio
@patch("app.services.query_understanding_graph.decide_retrieval_route")
@patch("app.services.query_understanding_graph.chat")
async def test_ready_generates_three_queries(mock_chat, mock_route):
    from app.services.query_reasoning_service import RetrievalDecision

    mock_route.return_value = RetrievalDecision(decision="rag", reasoning="technique")

    async def chat_side_effect(*args, **kwargs):
        context = kwargs.get("context") or []
        system = context[0].get("content", "") if context else ""
        if "évalues si une demande" in system.lower() or "too_vague" in system.lower():
            payload = {"too_vague": False, "reasoning": "précis", "clarification_question": ""}
        else:
            payload = {
                "colpali": "notice fenêtre PVC Perform Profine schéma dimensions",
                "semantic": "Quelles dimensions pour fenêtre PVC Perform Profine ?",
                "lexical": "fenêtre PVC Perform Profine dimension",
                "reasoning": "test",
                "slots_used": {"intent": "specification"},
            }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    mock_chat.side_effect = chat_side_effect

    persisted = {
        "slots": {
            **empty_slots(),
            "intent": "specification",
            "product_family": "fenetres",
            "material": "pvc",
            "product_range": "perform",
            "supplier": "profine",
        },
        "skipped_optional": ["product_range", "supplier", "content_categories"],
        "original_user_message": "dimensions dormant Perform",
    }

    result = await run_query_understanding(
        user_message="Profine",
        persisted_context=persisted,
        slot_action=SlotAction(field="supplier", value="profine", action="fill"),
    )
    assert result.ready_for_retrieval is True
    assert result.retrieval_queries is not None
    assert result.retrieval_queries.colpali
    assert mock_chat.call_count == 2


@pytest.mark.asyncio
@patch("app.services.query_understanding_graph.decide_retrieval_route")
@patch("app.services.query_understanding_graph.chat")
async def test_vague_after_slots_asks_text_clarification(mock_chat, mock_route):
    from app.services.query_reasoning_service import RetrievalDecision

    mock_route.return_value = RetrievalDecision(decision="rag", reasoning="technique")

    async def chat_side_effect(*args, **kwargs):
        context = kwargs.get("context") or []
        system = context[0].get("content", "") if context else ""
        if "évalues si une demande" in system.lower():
            return {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "too_vague": True,
                            "reasoning": "symptôme absent",
                            "clarification_question": (
                                "Pouvez-vous préciser le symptôme sur les coulisses "
                                "(jeu, blocage au clipage, mauvaise cote) ?"
                            ),
                        }),
                    },
                }],
            }
        return {"choices": [{"message": {"content": "{}"}}]}

    mock_chat.side_effect = chat_side_effect

    persisted = {
        "slots": {
            **empty_slots(),
            "intent": "troubleshooting",
            "product_family": "fenetres",
            "material": "pvc",
            "product_range": "perform",
            "supplier": "profine",
        },
        "skipped_optional": ["product_range", "supplier", "content_categories"],
        "original_user_message": "J'ai un souci mise en place des coulisses sur un dormant MONOBLOC",
        "phase": "ready",
    }

    result = await run_query_understanding(
        user_message="Profine",
        persisted_context=persisted,
        slot_action=SlotAction(field="supplier", value="profine", action="fill"),
    )
    assert result.ready_for_retrieval is False
    assert result.clarification is not None
    assert result.clarification.phase == "awaiting_vague_clarification"
    assert result.clarification.slot_prompt is None
    assert "symptôme" in result.clarification.question.lower()


@pytest.mark.asyncio
@patch("app.services.query_understanding_graph.decide_retrieval_route")
async def test_direct_route_skips_graph_retrieval(mock_route):
    from app.services.query_reasoning_service import RetrievalDecision

    mock_route.return_value = RetrievalDecision(decision="direct", reasoning="salutation")
    result = await run_query_understanding(user_message="bonjour")
    assert result.route == "direct"
    assert result.ready_for_retrieval is False
