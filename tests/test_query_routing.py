import json
from unittest import mock
import pytest
from app.services.query_reasoning_service import RetrievalDecision
from tests.conftest import extract_sse_message_text

async def _fake_mistral_stream(*args, **kwargs):
    yield json.dumps({"message": {"content": "Direct response chunk"}})

def test_space_chat_routing_direct(client, responsable_headers):
    # 1. Create space
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace routing test direct"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]
    
    try:
        # Mock decide_retrieval_route to return direct
        mock_decision = RetrievalDecision(decision="direct", reasoning="Politesse ou salutation")
        
        with mock.patch(
            "app.services.query_reasoning_service.decide_retrieval_route",
            new=mock.AsyncMock(return_value=mock_decision)
        ) as mock_route_decision, mock.patch(
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
            
            # Verify decide_retrieval_route was called
            mock_route_decision.assert_called_once_with("bonjour")
            # Verify RAG search was bypassed (not called)
            mock_search.assert_not_called()
            
    finally:
        client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)

def test_space_chat_routing_rag(client, responsable_headers, ready_slot_validation):
    # 1. Create space
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace routing test rag"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]
    
    try:
        # Mock decide_retrieval_route to return RAG
        mock_decision = RetrievalDecision(decision="rag", reasoning="Question technique")
        
        with mock.patch(
            "app.services.query_reasoning_service.decide_retrieval_route",
            new=mock.AsyncMock(return_value=mock_decision)
        ) as mock_route_decision, mock.patch(
            "app.services.slot_filling_service.process_slot_filling",
            new=mock.AsyncMock(return_value=ready_slot_validation),
        ), mock.patch(
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
            text_content = extract_sse_message_text(r.text)
            assert "seuil minimum de 75%" in text_content
            
            # Verify decide_retrieval_route was called
            mock_route_decision.assert_called_once_with("quel est le dormant Profine ?")
            # Verify RAG search was NOT bypassed (it was called)
            mock_search.assert_called_once()
            
    finally:
        client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)
