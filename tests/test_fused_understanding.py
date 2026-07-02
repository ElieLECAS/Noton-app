"""Axe 1 — compréhension fusionnée + correction du fil conversationnel.

Couvre :
  * _node_merge_context : un tour normal repart du message COURANT (ne « colle » plus
    au message du tour précédent) ; une reprise de clarification combine bien les deux.
  * _node_fused_understand : parsing route / topic_shift / standalone_question /
    vagueness / signaux en un seul appel, court-circuit direct, repli sur erreur LLM.
  * run_lightweight_understanding (chemin fusionné) : topic_shift remonté au résultat.
"""
from __future__ import annotations

import json
from unittest import mock

import pytest

from app.services import lightweight_query_understanding as lqu


def _fake_chat(content: dict):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


# ---------------------------------------------------------------------------
# _node_merge_context — correction du bug « conversation collée au 1er message »
# ---------------------------------------------------------------------------


def test_merge_context_normal_turn_uses_current_message():
    """Hors clarification, on NE réutilise PAS l'original_user_message persisté."""
    state = {
        "user_message": "et le DTU 36.5 ?",
        "persisted_context": {
            "original_user_message": "parle-moi du profil 76",
            "enriched_user_message": "",
            "awaiting_vague_clarification": False,
            "phase": "ready",
        },
    }
    out = lqu._node_merge_context(state)
    assert out["original_user_message"] == "et le DTU 36.5 ?"
    assert out["enriched_user_message"] == ""
    assert out["awaiting_vague_clarification"] is False


def test_merge_context_vague_continuation_combines():
    """Reprise après question de clarification : on recombine demande initiale + précision."""
    state = {
        "user_message": "en aluminium",
        "persisted_context": {
            "original_user_message": "quel seuil choisir ?",
            "enriched_user_message": "quel seuil choisir ?",
            "awaiting_vague_clarification": True,
            "phase": "awaiting_vague_clarification",
        },
    }
    out = lqu._node_merge_context(state)
    assert out["awaiting_vague_clarification"] is True
    assert "quel seuil choisir" in out["enriched_user_message"]
    assert "Précision utilisateur : en aluminium" in out["enriched_user_message"]


# ---------------------------------------------------------------------------
# _node_fused_understand — 1 appel LLM combiné (session réelle, LLM mocké)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fused_understand_parses_route_topic_shift_and_signals(db_session):
    state = {
        "user_message": "oublie le profil 76, parle-moi du DTU 36.5",
        "original_user_message": "oublie le profil 76, parle-moi du DTU 36.5",
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
        "too_vague": False,
        "clarification_question": None,
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
    state = {
        "user_message": "bonjour",
        "original_user_message": "bonjour",
        "session": db_session,
        "history": [],
    }
    payload = {"route": "direct", "topic_shift": True, "entities": [], "confidence": 0.2}
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))):
        out = await lqu._node_fused_understand(state)

    assert out["route"] == "direct"
    # Court-circuit : pas de passage au retrieval.
    assert out.get("ready_for_retrieval") is not True


@pytest.mark.asyncio
async def test_fused_understand_vague_requests_clarification(db_session):
    state = {
        "user_message": "j'ai un problème",
        "original_user_message": "j'ai un problème",
        "session": db_session,
        "history": [],
    }
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "j'ai un problème",
        "too_vague": True,
        "clarification_question": "Quel produit et quel symptôme précisément ?",
        "entities": [],
        "confidence": 0.3,
    }
    with mock.patch("app.config.settings.QUERY_VAGUENESS_CHECK_ENABLED", True), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))
    ):
        out = await lqu._node_fused_understand(state)

    assert out["ready_for_retrieval"] is False
    assert out["awaiting_vague_clarification"] is True
    assert out["clarification"]["question"].startswith("Quel produit")


@pytest.mark.asyncio
async def test_fused_understand_vague_ignored_when_check_disabled(db_session):
    state = {
        "user_message": "j'ai un problème",
        "original_user_message": "j'ai un problème",
        "session": db_session,
        "history": [],
    }
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "j'ai un problème",
        "too_vague": True,
        "clarification_question": "Précisez ?",
        "entities": [],
        "confidence": 0.3,
    }
    with mock.patch("app.config.settings.QUERY_VAGUENESS_CHECK_ENABLED", False), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))
    ):
        out = await lqu._node_fused_understand(state)

    # Vagueness désactivée → on file au retrieval malgré too_vague=True.
    assert out["ready_for_retrieval"] is True


@pytest.mark.asyncio
async def test_fused_understand_falls_back_on_llm_error(db_session):
    state = {
        "user_message": "profil 76 dimensions",
        "original_user_message": "profil 76 dimensions",
        "session": db_session,
        "history": [],
    }
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(side_effect=RuntimeError("boom"))):
        out = await lqu._node_fused_understand(state)

    # Repli sûr : route rag, prêt pour retrieval, message inchangé, pas de crash.
    assert out["route"] == "rag"
    assert out["ready_for_retrieval"] is True
    assert out["standalone_question"] == "profil 76 dimensions"


# ---------------------------------------------------------------------------
# run_lightweight_understanding (chemin fusionné) — topic_shift bout-en-bout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_fused_surfaces_topic_shift(db_session):
    understand_payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "dimensions du seuil PMR 76100",
        "too_vague": False,
        "clarification_question": None,
        "entities": [{"text": "seuil PMR"}, {"text": "76100"}],
        "detected_references": ["76100"],
        "confidence": 0.9,
    }
    queries_payload = {
        "colpali": "seuil PMR 76100 schéma dimensions",
        "semantic": "dimensions seuil PMR 76100",
        "lexical": "seuil PMR 76100",
        "reasoning": "test",
    }

    async def chat_side_effect(*args, **kwargs):
        context = kwargs.get("context") or []
        system = context[0].get("content", "") if context else ""
        # Le prompt fusionné contient "topic_shift" ; celui de génération de requêtes non.
        if "topic_shift" in system:
            return _fake_chat(understand_payload)
        return _fake_chat(queries_payload)

    with mock.patch("app.config.settings.QUERY_FUSED_UNDERSTANDING_ENABLED", True), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock(side_effect=chat_side_effect)
    ):
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


# ---------------------------------------------------------------------------
# Axe 4 — current_topic : réinitialisé sur topic_shift, conservé sinon
# ---------------------------------------------------------------------------


def _understand_and_queries_side_effect(understand_payload: dict):
    queries_payload = {
        "colpali": "x",
        "semantic": "y",
        "lexical": "z",
        "reasoning": "t",
    }

    async def side_effect(*args, **kwargs):
        context = kwargs.get("context") or []
        system = context[0].get("content", "") if context else ""
        if "topic_shift" in system:
            return _fake_chat(understand_payload)
        return _fake_chat(queries_payload)

    return side_effect


@pytest.mark.asyncio
async def test_current_topic_reset_on_topic_shift(db_session):
    payload = {
        "route": "rag",
        "topic_shift": True,
        "standalone_question": "dimensions du seuil PMR 76100",
        "too_vague": False,
        "entities": [{"text": "76100"}],
        "confidence": 0.9,
    }
    with mock.patch("app.config.settings.QUERY_FUSED_UNDERSTANDING_ENABLED", True), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock(side_effect=_understand_and_queries_side_effect(payload))
    ):
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
        "too_vague": False,
        "entities": [],
        "confidence": 0.8,
    }
    with mock.patch("app.config.settings.QUERY_FUSED_UNDERSTANDING_ENABLED", True), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock(side_effect=_understand_and_queries_side_effect(payload))
    ):
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
