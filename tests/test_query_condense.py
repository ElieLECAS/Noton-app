"""Tests du nœud de condensation history-aware (reformulation en question autonome).

Vérifie que les messages de suivi ("et le tgy3834 ?") sont reformulés en une
question autonome avant le retrieval, sans dépendre de l'historique conversationnel.
"""
import json
from unittest import mock

import pytest

from app.services import lightweight_query_understanding as lqu


def _fake_chat_response(content: dict):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


@pytest.mark.asyncio
async def test_condense_reformulates_followup_with_history():
    state = {
        "user_message": "et le tgy3834 ?",
        "original_user_message": "et le tgy3834 ?",
        "enriched_user_message": "et le tgy3834 ?",
        "history": [
            {"role": "user", "content": "référence des embouts de montant pour coulissant galandage SOLEAL GY 55"},
            {"role": "assistant", "content": "Bouchon percussion central : TGY3811..."},
        ],
    }

    standalone = "À quoi sert la référence TGY3834 pour coulissant galandage SOLEAL GY 55 ?"

    with mock.patch("app.config.settings.QUERY_CONDENSE_ENABLED", True), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat_response({"standalone_question": standalone}))
    ) as mock_chat:
        out = await lqu._node_condense_question(state)

    assert out["standalone_question"] == standalone
    mock_chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_condense_noop_without_history():
    state = {
        "user_message": "parle moi du tgy3834",
        "original_user_message": "parle moi du tgy3834",
        "enriched_user_message": "parle moi du tgy3834",
        "history": [],
    }

    with mock.patch("app.config.settings.QUERY_CONDENSE_ENABLED", True), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock()
    ) as mock_chat:
        out = await lqu._node_condense_question(state)

    assert out["standalone_question"] == "parle moi du tgy3834"
    mock_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_condense_disabled_flag_skips_llm():
    state = {
        "user_message": "et le tgy3834 ?",
        "original_user_message": "et le tgy3834 ?",
        "enriched_user_message": "et le tgy3834 ?",
        "history": [{"role": "user", "content": "sujet précédent"}],
    }

    with mock.patch("app.config.settings.QUERY_CONDENSE_ENABLED", False), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock()
    ) as mock_chat:
        out = await lqu._node_condense_question(state)

    assert out["standalone_question"] == "et le tgy3834 ?"
    mock_chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_condense_falls_back_on_llm_error():
    state = {
        "user_message": "et le tgy3834 ?",
        "original_user_message": "et le tgy3834 ?",
        "enriched_user_message": "et le tgy3834 ?",
        "history": [{"role": "user", "content": "sujet précédent"}],
    }

    with mock.patch("app.config.settings.QUERY_CONDENSE_ENABLED", True), mock.patch.object(
        lqu, "chat", new=mock.AsyncMock(side_effect=RuntimeError("boom"))
    ):
        out = await lqu._node_condense_question(state)

    # En cas d'échec LLM, on retombe sur la demande complète (pas de crash).
    assert out["standalone_question"] == "et le tgy3834 ?"


def test_search_text_prefers_standalone_question():
    state = {
        "original_user_message": "et le tgy3834 ?",
        "enriched_user_message": "et le tgy3834 ?",
        "standalone_question": "À quoi sert la référence TGY3834 ?",
    }
    assert lqu._search_text(state) == "À quoi sert la référence TGY3834 ?"


def test_search_text_fallbacks_without_standalone():
    state = {
        "original_user_message": "parle moi du tgy3834",
        "enriched_user_message": "",
    }
    assert lqu._search_text(state) == "parle moi du tgy3834"
