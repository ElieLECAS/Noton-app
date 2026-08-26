"""Reformulation history-aware (condense) : les suivis elliptiques deviennent autonomes.

Vérifie qu'un message de suivi ("et le tgy3834 ?") est reformulé en question autonome
AVANT le retrieval, sans dépendre de l'historique conversationnel.

Nettoyage 2026-08-26 : le condense n'est plus un nœud séparé — il est produit par le
champ "standalone_question" de l'appel fusionné (nœud unique). Ces tests portent donc
sur le comportement observable de bout en bout plutôt que sur un nœud dédié.
"""
import json
from unittest import mock

import pytest

from app.services import lightweight_query_understanding as lqu


def _fused_response(**overrides):
    payload = {
        "route": "rag",
        "topic_shift": False,
        "standalone_question": "",
        "current_topic": "",
        "intent": "documentation",
        "entities": [],
        "inferred_categories": [],
    }
    payload.update(overrides)
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


@pytest.mark.asyncio
async def test_followup_is_reformulated_before_retrieval(db_session):
    """La requête envoyée au retrieval est la question autonome, pas le message brut."""
    standalone = "À quoi sert la référence TGY3834 pour coulissant galandage SOLEAL GY 55 ?"
    fake_chat = mock.AsyncMock(return_value=_fused_response(standalone_question=standalone))

    with mock.patch.object(lqu, "chat", fake_chat):
        result = await lqu.run_lightweight_understanding(
            user_message="et le tgy3834 ?",
            history=[
                {
                    "role": "user",
                    "content": "référence des embouts de montant pour coulissant galandage SOLEAL GY 55",
                },
                {"role": "assistant", "content": "Bouchon percussion central : TGY3811..."},
            ],
            session=db_session,
        )

    assert result.query_context["standalone_question"] == standalone
    # C'est bien la question autonome qui part au retrieval, pas « et le tgy3834 ? ».
    assert result.retrieval_queries.colpali == standalone


@pytest.mark.asyncio
async def test_history_is_passed_to_the_understanding_prompt(db_session):
    fake_chat = mock.AsyncMock(return_value=_fused_response(standalone_question="question autonome"))

    with mock.patch.object(lqu, "chat", fake_chat):
        await lqu.run_lightweight_understanding(
            user_message="et le tgy3834 ?",
            history=[{"role": "user", "content": "sujet précédent SOLEAL"}],
            session=db_session,
        )

    _, kwargs = fake_chat.call_args
    joined = " ".join(m.get("content", "") for m in (kwargs.get("context") or []))
    assert "sujet précédent SOLEAL" in joined
    assert "standalone_question" in joined


@pytest.mark.asyncio
async def test_falls_back_to_raw_message_on_llm_error(db_session):
    """Échec LLM → on cherche avec le message brut plutôt que de bloquer le tour."""
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(side_effect=RuntimeError("boom"))):
        result = await lqu.run_lightweight_understanding(
            user_message="et le tgy3834 ?",
            history=[{"role": "user", "content": "sujet précédent"}],
            session=db_session,
        )

    assert result.ready_for_retrieval is True
    assert result.retrieval_queries.colpali == "et le tgy3834 ?"


def test_search_text_prefers_standalone_question():
    state = {
        "user_message": "et le tgy3834 ?",
        "standalone_question": "À quoi sert la référence TGY3834 ?",
    }
    assert lqu._search_text(state) == "À quoi sert la référence TGY3834 ?"


def test_search_text_fallbacks_without_standalone():
    state = {"user_message": "parle moi du tgy3834"}
    assert lqu._search_text(state) == "parle moi du tgy3834"
