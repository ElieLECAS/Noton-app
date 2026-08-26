"""Axe 1 — compréhension fusionnée + fil conversationnel.

Couvre :
  * _node_fused_understand : parsing route / topic_shift / standalone_question /
    signaux en un seul appel, court-circuit direct, repli sur erreur LLM.
  * run_lightweight_understanding : topic_shift et current_topic bout-en-bout.

Nettoyage 2026-08-26 : _node_merge_context et le contrôle de vagueness ont été
supprimés (le premier ne servait qu'à la reprise de clarification du second). Il n'y a
plus qu'un seul appel LLM dans tout le pipeline, donc plus de side_effect à aiguiller.
"""
from __future__ import annotations

import json
from unittest import mock

import pytest

from app.services import lightweight_query_understanding as lqu


def _fake_chat(content: dict):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


# ---------------------------------------------------------------------------
# _node_fused_understand — 1 appel LLM combiné (session réelle, LLM mocké)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fused_understand_parses_route_topic_shift_and_signals(db_session):
    state = {
        "user_message": "oublie le profil 76, parle-moi du DTU 36.5",
        "session": db_session,
        "history": [
            {"role": "user", "content": "parle-moi du profil 76"},
            {"role": "assistant", "content": "Le profil 76 ..."},
        ],
    }
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "À quoi sert le DTU 36.5 ?",
        "intent": "regulatory",
        "entities": [{"text": "DTU 36.5"}],
        "detected_references": ["DTU 36.5"],
        "confidence": 0.9,
    }
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))) as mock_chat:
        out = await lqu._node_fused_understand(state)

    mock_chat.assert_awaited_once()
    assert out["route"] == "rag"
    assert out["topic_shift"] is True
    assert out["ready_for_retrieval"] is True
    assert out["standalone_question"] == "À quoi sert le DTU 36.5 ?"
    assert out["signals"]["intent"] == "regulatory"
    assert "DTU 36.5" in out["signals"]["entity_texts"]


@pytest.mark.asyncio
async def test_fused_understand_direct_route_short_circuits(db_session):
    state = {"user_message": "bonjour", "session": db_session, "history": []}
    payload = {"route": "direct", "topic_shift": True, "entities": [], "confidence": 0.2}
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))):
        out = await lqu._node_fused_understand(state)

    assert out["route"] == "direct"
    # Court-circuit : pas de passage au retrieval.
    assert out.get("ready_for_retrieval") is not True


@pytest.mark.asyncio
async def test_fused_understand_never_blocks_on_vague_request(db_session):
    """Le contrôle de vagueness pré-retrieval a été supprimé : une question courte part
    quand même au retrieval (il bloquait des questions techniques légitimes à sigles)."""
    state = {"user_message": "j'ai un problème", "session": db_session, "history": []}
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "j'ai un problème",
        # Même si le LLM renvoie ces champs par habitude, ils sont ignorés.
        "too_vague": True,
        "clarification_question": "Quel produit ?",
        "entities": [],
        "confidence": 0.3,
    }
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))):
        out = await lqu._node_fused_understand(state)

    assert out["ready_for_retrieval"] is True
    assert "clarification" not in out


@pytest.mark.asyncio
async def test_fused_understand_falls_back_on_llm_error(db_session):
    state = {"user_message": "profil 76 dimensions", "session": db_session, "history": []}
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(side_effect=RuntimeError("boom"))):
        out = await lqu._node_fused_understand(state)

    # Repli sûr : route rag, prêt pour retrieval, message inchangé, pas de crash.
    assert out["route"] == "rag"
    assert out["ready_for_retrieval"] is True
    assert out["standalone_question"] == "profil 76 dimensions"


# ---------------------------------------------------------------------------
# run_lightweight_understanding — topic_shift et current_topic bout-en-bout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fused_surfaces_topic_shift(db_session):
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "dimensions du seuil PMR 76100",
        "entities": [{"text": "seuil PMR"}, {"text": "76100"}],
        "detected_references": ["76100"],
        "confidence": 0.9,
    }
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))):
        result = await lqu.run_lightweight_understanding(
            user_message="et le seuil PMR 76100 ?",
            history=[
                {"role": "user", "content": "parle-moi des poignées"},
                {"role": "assistant", "content": "Les poignées ..."},
            ],
            session=db_session,
        )

    assert result.ready_for_retrieval is True
    assert result.topic_shift is True
    assert result.query_context["topic_shift"] is True
    assert result.retrieval_queries is not None


@pytest.mark.asyncio
async def test_current_topic_reset_on_topic_shift(db_session):
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "dimensions du seuil PMR 76100",
        "entities": [{"text": "76100"}],
        "confidence": 0.9,
    }
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))):
        result = await lqu.run_lightweight_understanding(
            user_message="et le seuil PMR 76100 ?",
            history=[
                {"role": "user", "content": "parle des poignées"},
                {"role": "assistant", "content": "Les poignées ..."},
            ],
            session=db_session,
            persisted_context={"current_topic": "poignées Soleal"},
        )

    # Nouveau sujet → current_topic remplacé par la question courante.
    assert result.query_context["current_topic"] == "dimensions du seuil PMR 76100"


@pytest.mark.asyncio
async def test_current_topic_carried_forward_without_shift(db_session):
    payload = {
        "route": "rag",
        "topic_shift": False,
        "standalone_question": "et ses dimensions ?",
        "entities": [],
        "confidence": 0.8,
    }
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))):
        result = await lqu.run_lightweight_understanding(
            user_message="et ses dimensions ?",
            history=[
                {"role": "user", "content": "parle du seuil PMR 76100"},
                {"role": "assistant", "content": "Le seuil PMR ..."},
            ],
            session=db_session,
            persisted_context={"current_topic": "seuil PMR 76100"},
        )

    # Même sujet → current_topic conservé d'un tour à l'autre.
    assert result.topic_shift is False
    assert result.query_context["current_topic"] == "seuil PMR 76100"
