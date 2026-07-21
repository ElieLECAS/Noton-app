import json
import types
from unittest import mock
import pytest

async def _fake_mistral_stream(*args, **kwargs):
    yield json.dumps({"message": {"content": "Direct response chunk"}})


def _fake_lw_result(route: str):
    """Résultat de compréhension minimal pour piloter la route (refonte C1)."""
    return types.SimpleNamespace(
        route=route,
        ready_for_retrieval=True,
        topic_shift=False,
        signals=None,
        query_context={},
        guided=None,
        clarification=None,
        retrieval_queries=None,
        query_groups=[],
        query_strategy=None,
    )


def test_space_chat_routing_direct(client, responsable_headers):
    """Refonte C1 : la route « direct » est décidée par la COMPRÉHENSION, plus par un
    routeur LLM autonome."""
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace routing test direct"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]

    try:
        with mock.patch("app.config.settings.QUERY_UNDERSTANDING_ENABLED", True), mock.patch(
            "app.services.lightweight_query_understanding.run_lightweight_understanding",
            new=mock.AsyncMock(return_value=_fake_lw_result("direct")),
        ), mock.patch(
            "app.routers.chat.mistral_chat_stream",
            _fake_mistral_stream
        ), mock.patch(
            "app.services.space_search_service.search_technical_passages"
        ) as mock_search:

            r = client.post(
                f"/api/spaces/{space_id}/chat/stream",
                headers=responsable_headers,
                json={
                    "message": "bonjour",
                    "model": "mistral-small-latest",
                    "provider": "mistral",
                    "conversation_id": None,
                },
            )

            assert r.status_code == 200
            assert "Direct response chunk" in r.text
            assert "done" in r.text.lower()
            # La voie directe ne déclenche PAS de recherche documentaire.
            mock_search.assert_not_called()

    finally:
        client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)

def test_space_chat_routing_rag(client, responsable_headers):
    # 1. Create space
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace routing test rag"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]
    
    try:
        # Refonte C1 : compréhension désactivée → route=search DIRECTE, plus aucun
        # appel au routeur LLM autonome (decide_retrieval_route).
        with mock.patch("app.config.settings.QUERY_UNDERSTANDING_ENABLED", False), mock.patch(
            "app.services.query_reasoning_service.decide_retrieval_route",
            new=mock.AsyncMock()
        ) as mock_route_decision, mock.patch(
            "app.routers.chat.mistral_chat_stream",
            _fake_mistral_stream
        ), mock.patch(
            "app.services.space_search_service.search_technical_passages",
            new=mock.AsyncMock(
                return_value={"passages": [], "status": "ok", "reason": "no_results"}
            )
        ) as mock_search:
            
            r = client.post(
                f"/api/spaces/{space_id}/chat/stream",
                headers=responsable_headers,
                json={
                    "message": "quel est le dormant Profine ?",
                    "model": "mistral-small-latest",
                    "provider": "mistral",
                    "conversation_id": None,
                },
            )
            
            assert r.status_code == 200
            # Since no passages were found, it should display the RAG threshold warning
            assert "75%" in r.text
            
            # Refonte C1 : plus AUCUN appel au routeur LLM autonome quand la
            # compréhension est off — la recherche est le défaut.
            mock_route_decision.assert_not_called()
            # La recherche documentaire a bien eu lieu.
            mock_search.assert_called_once()
            
    finally:
        client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)
