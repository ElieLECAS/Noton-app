import pytest
from unittest.mock import AsyncMock, patch
from app.services.document_service_new import infer_document_source
from app.services.query_reasoning_service import reason_query_intent, QueryIntent

# --- Tests Inférence de Source ---

@pytest.mark.parametrize("path, content, expected", [
    ("C:/Documents/PROFERM/catalogue.pdf", "", "Proferm"),
    ("admin/technal/notice.pdf", "", "Technal"),
    ("", "Voici le nouveau dormant Profine pour vos fenêtres.", "Profine"),
    ("", "Chez Proferm, nous aimons la qualité.", "Proferm"),
    ("generic/manual.pdf", "Something completely unrelated.", "Inconnu"),
])
def test_infer_document_source(path, content, expected):
    result = infer_document_source(file_path=path, content=content)
    assert result == expected


# --- Tests Query Reasoning (Mocked Mistral) ---

@pytest.mark.asyncio
@patch("app.services.query_reasoning_service.chat")
async def test_reason_query_intent_proferm(mock_chat):
    # Mock de la réponse Mistral
    mock_chat.return_value = {
        "choices": [{
            "message": {
                "content": '{"intent": "company_info", "primary_source": "Proferm", "reasoning": "L\'utilisateur utilise le possessif vos", "confidence": 0.9}'
            }
        }]
    }
    
    result = await reason_query_intent("Parle moi de vos gammes")
    assert result.intent == "company_info"
    assert result.primary_source == "Proferm"
    assert result.confidence == 0.9

@pytest.mark.asyncio
@patch("app.services.query_reasoning_service.chat")
async def test_reason_query_intent_technal(mock_chat):
    mock_chat.return_value = {
        "choices": [{
            "message": {
                "content": '{"intent": "supplier_info", "primary_source": "Technal", "reasoning": "Mention explicite de Technal", "confidence": 0.95}'
            }
        }]
    }
    
    result = await reason_query_intent("Quelles sont les couleurs chez Technal ?")
    assert result.primary_source == "Technal"
