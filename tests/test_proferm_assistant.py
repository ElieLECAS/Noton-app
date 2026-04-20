import pytest
from unittest.mock import AsyncMock, patch
from app.services.document_service_new import infer_document_source
from app.services.query_reasoning_service import reason_query_intent, QueryIntent
from app.services.space_search_service import refine_with_source_authority

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


# --- Tests Boosting Retrieval ---

def test_refine_with_source_authority_boost():
    passages = [
        {"note_title": "Doc A", "source": "Technal", "score": 0.5},
        {"note_title": "Doc B", "source": "Proferm", "score": 0.4},
    ]
    query = "nos produits"
    reasoning = QueryIntent(
        intent="company_info", 
        primary_source="Proferm", 
        reasoning="Test", 
        confidence=1.0
    )
    
    # Sans boost, Doc A est premier
    # Avec boost, Doc B devrait passer devant (0.4 + 0.8 = 1.2)
    result = refine_with_source_authority(passages, query, reasoning_result=reasoning)
    
    assert result[0]["source"] == "Proferm"
    assert result[0]["score"] > 1.0
    assert result[1]["source"] == "Technal"

def test_refine_with_source_authority_no_boost():
    passages = [
        {"note_title": "Doc A", "source": "Technal", "score": 0.5},
        {"note_title": "Doc B", "source": "Proferm", "score": 0.4},
    ]
    query = "fenêtres"
    # Raisonnement générique
    reasoning = QueryIntent(intent="generic", reasoning="Test", confidence=0.5)
    
    result = refine_with_source_authority(passages, query, reasoning_result=reasoning)
    
    # L'ordre ne change pas (sauf si le titre matchait)
    assert result[0]["source"] == "Technal"
    assert result[0]["score"] == 0.5
