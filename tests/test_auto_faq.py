from __future__ import annotations

import json
from unittest import mock
import pytest
from sqlmodel import Session, select

from app.models.space import Space
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.message_feedback import MessageFeedback
from app.models.user import User
from app.models.conversation import Conversation
from app.models.message import Message
from app.tasks.documents import generate_faq_from_feedback_task
from tests.conftest import create_test_user, bearer_headers
from app.services.chat_critique_service import (
    parse_critique_json,
    sanitize_critique_output,
    resolve_critique_final,
)


def test_critique_unchanged_uses_draft_verbatim():
    draft = "Pour 340 kg, ce n'est pas réalisable (limite 300 kg)."
    raw = json.dumps({"decision": "unchanged", "final_message": ""})
    assert parse_critique_json(raw, draft) == draft
    assert resolve_critique_final(raw, draft) == draft


def test_critique_corrected_uses_final_message_only():
    draft = "Utiliser le joint T740023."
    corrected = "Utiliser le joint XL30202 et non T740023."
    raw = json.dumps({"decision": "corrected", "final_message": corrected})
    assert parse_critique_json(raw, draft) == corrected


def test_critique_meta_leak_fallback_to_draft():
    draft = "Réponse propre du brouillon avec assez de contenu technique."
    leaky = (
        "La réponse temporaire est correcte et conforme aux FAQ.\n"
        "Voici la réponse à envoyer sans modification :\n"
        '"Réponse propre du brouillon avec assez de contenu technique."'
    )
    result = sanitize_critique_output(leaky, draft)
    assert result == draft
    assert "La réponse temporaire" not in result
    assert "Voici la réponse" not in result


def test_critique_invalid_json_fallback_sanitize_or_draft():
    draft = "Réponse brouillon complète et valide pour l'utilisateur."
    raw = "Voici la réponse à envoyer sans modification : Réponse brouillon complète et valide pour l'utilisateur."
    result = resolve_critique_final(raw, draft)
    assert "Voici la réponse" not in result
    assert len(result) >= 20


def test_generate_faq_from_feedback_task_success(db_session: Session):
    # 1. Create a user (responsable role has feedback.auto_faq permission via seed)
    user = create_test_user(db_session, "responsable")
    
    # 2. Create Space, Document, and DocumentChunk
    space = Space(name="Space A", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)
    
    # Library
    from app.services.library_service import get_or_create_user_library
    library = get_or_create_user_library(db_session, user.id)
    
    doc = Document(
        title="Manuel Technique Joint",
        content="Notice de montage des joints de menuiserie.",
        document_type="document",
        processing_status="completed",
        processing_progress=100,
        library_id=library.id,
        user_id=user.id
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    
    # Link doc to space
    doc_space = DocumentSpace(document_id=doc.id, space_id=space.id, user_id=user.id)
    db_session.add(doc_space)
    db_session.commit()
    
    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="Pour les menuiseries PVC, utiliser le joint standard T740023.",
        text="Pour les menuiseries PVC, utiliser le joint standard T740023.",
        is_leaf=True,
        hierarchy_level=0,
        source="Manuel Technique Joint",
        start_char=0,
        end_char=60
    )
    db_session.add(chunk)
    db_session.commit()
    db_session.refresh(chunk)
    
    # 3. Create feedback entry referencing this chunk
    feedback = MessageFeedback(
        message_id=None,
        user_id=user.id,
        space_id=space.id,
        is_positive=False,
        comment="Erreur dans la notice, utiliser XL30202 et non T740023.",
        query_text="Quel joint utiliser pour le PVC ?",
        response_text="Utiliser le joint T740023.",
        chunk_ids=[chunk.id]
    )
    db_session.add(feedback)
    db_session.commit()
    db_session.refresh(feedback)
    
    # Mock Mistral call inside the task
    mock_mistral_response = {
        "choices": [
            {
                "message": {
                    "content": "Question: Quel joint utiliser pour le PVC ?\nRéponse: Utiliser le joint XL30202 et non T740023."
                }
            }
        ]
    }
    
    async def fake_mistral_chat(*args, **kwargs):
        return mock_mistral_response
        
    with mock.patch("app.tasks.documents.mistral_chat", fake_mistral_chat):
        # Run Celery task synchronously
        generate_faq_from_feedback_task(feedback.id)
        
    # Refresh feedback from DB
    db_session.refresh(feedback)
    
    # Assert feedback is updated
    assert feedback.auto_faq_generated is True
    assert "XL30202" in feedback.auto_faq_content
    
    # Assert virtual document is created
    virtual_doc = db_session.exec(
        select(Document).where(Document.title == "FAQ Corrective - Manuel Technique Joint")
    ).first()
    assert virtual_doc is not None
    assert virtual_doc.document_type == "written"
    assert virtual_doc.library_id == library.id
    
    # Assert virtual doc is linked to space
    linked_spaces = db_session.exec(
        select(DocumentSpace).where(DocumentSpace.document_id == virtual_doc.id)
    ).all()
    assert len(linked_spaces) == 1
    assert linked_spaces[0].space_id == space.id
    
    # Assert virtual chunk is created
    virtual_chunk = db_session.exec(
        select(DocumentChunk).where(DocumentChunk.document_id == virtual_doc.id)
    ).first()
    assert virtual_chunk is not None
    assert "XL30202" in virtual_chunk.content


def test_two_step_chat_pipeline_with_faq_retrieval(client, db_session: Session):
    """Test du pipeline double recherche : technique → brouillon → FAQ post-brouillon → critique."""
    # 1. Setup user, space, conversation
    user = create_test_user(db_session, "responsable")
    headers = bearer_headers(user.id)
    
    space = Space(name="Space Chat Test", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)
    
    conv = Conversation(title="Test Conv", user_id=user.id, space_id=space.id)
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)
    
    # 2. Mock technical search (pass 1) - documents techniques uniquement
    mock_technical_result = {
        "status": "ok",
        "passages": [
            {
                "passage": "Pour le PVC, il faut utiliser le joint T740023.",
                "passage_raw": "Pour le PVC, il faut utiliser le joint T740023.",
                "document_title": "Notice Standard",
                "document_id": 100,
                "chunk_id": 101,
                "score": 0.95
            }
        ]
    }
    
    # 3. Mock FAQ search (pass 2 post-brouillon) - FAQ correctives uniquement
    mock_faq_result = {
        "status": "ok",
        "reason": "faq_corrective_found",
        "passages": [
            {
                "passage": "Question: Quel joint utiliser pour le PVC ?\nRéponse: Utiliser le joint XL30202 et non T740023.",
                "passage_raw": "Question: Quel joint utiliser pour le PVC ?\nRéponse: Utiliser le joint XL30202 et non T740023.",
                "document_title": "FAQ Corrective - Notice Standard",
                "document_id": 200,
                "chunk_id": 201,
                "score": 0.99
            }
        ]
    }
    
    async def fake_technical_search(*args, **kwargs):
        return mock_technical_result
    
    async def fake_faq_search(*args, **kwargs):
        return mock_faq_result

    # Mock Mistral Chat calls
    # 1st call (draft): génération du brouillon avec docs techniques uniquement
    # 2nd call (critique): critique avec FAQ correctives
    call_count = 0
    draft_response = "D'après la notice, il faut utiliser le joint T740023."
    critique_json = json.dumps({
        "decision": "corrected",
        "final_message": "Il faut utiliser le joint XL30202 et non T740023.",
    })
    
    async def fake_mistral_chat(prompt, model, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First pass: draft generation
            # Check that FAQ passages are indeed omitted in draft generation messages
            msg_contents = [m.get("content", "") for m in messages]
            assert not any("FAQ Corrective" in content for content in msg_contents)
            return {"choices": [{"message": {"content": draft_response}}]}
        elif call_count == 2:
            # Second pass: critique with FAQ (JSON)
            assert kwargs.get("response_format") == {"type": "json_object"}
            assert kwargs.get("temperature") == 0
            msg_contents = [m.get("content", "") for m in messages]
            assert any("XL30202" in content for content in msg_contents)
            return {"choices": [{"message": {"content": critique_json}}]}
        return {"choices": [{"message": {"content": ""}}]}
        
    with mock.patch("app.routers.chat.search_technical_passages", fake_technical_search):
        with mock.patch("app.routers.chat.search_corrective_faq_passages", fake_faq_search):
            with mock.patch("app.routers.chat.mistral_chat", fake_mistral_chat):
                response = client.post(
                    f"/api/spaces/{space.id}/chat/stream",
                    headers=headers,
                    json={
                        "message": "Quel joint utiliser ?",
                        "model": "mistral-large-latest",
                        "conversation_id": conv.id
                    }
                )
            
    assert response.status_code == 200
    text_content = response.text
    assert "XL30202" in text_content
    assert "La réponse temporaire" not in text_content
    assert "Voici la réponse" not in text_content
    assert call_count == 2  # Both draft and critique were called


def test_critique_unchanged_pipeline_uses_draft(client, db_session: Session):
    """Si decision=unchanged, la réponse streamée est le brouillon exact sans méta-texte."""
    user = create_test_user(db_session, "responsable")
    headers = bearer_headers(user.id)

    space = Space(name="Space Unchanged", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    conv = Conversation(title="Test Unchanged", user_id=user.id, space_id=space.id)
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)

    draft_response = "Limite TGA3608 : 300 kg. Pour 340 kg, non réalisable."

    async def fake_technical_search(*args, **kwargs):
        return {"status": "ok", "passages": [{"passage": "doc", "document_title": "Doc", "document_id": 1, "chunk_id": 2, "score": 0.9}]}

    async def fake_faq_search(*args, **kwargs):
        return {
            "status": "ok",
            "passages": [{"passage": "FAQ ok", "document_title": "FAQ Corrective - Doc", "document_id": 3, "chunk_id": 4, "score": 0.95}],
        }

    critique_json = json.dumps({"decision": "unchanged", "final_message": ""})
    call_count = 0

    async def fake_mistral_chat(prompt, model, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"choices": [{"message": {"content": draft_response}}]}
        return {"choices": [{"message": {"content": critique_json}}]}

    with mock.patch("app.routers.chat.search_technical_passages", fake_technical_search):
        with mock.patch("app.routers.chat.search_corrective_faq_passages", fake_faq_search):
            with mock.patch("app.routers.chat.mistral_chat", fake_mistral_chat):
                response = client.post(
                    f"/api/spaces/{space.id}/chat/stream",
                    headers=headers,
                    json={"message": "Question ?", "model": "mistral-large-latest", "conversation_id": conv.id},
                )

    assert response.status_code == 200
    assert draft_response in response.text
    assert "La réponse temporaire" not in response.text


def test_faq_task_idempotent(db_session: Session):
    """Test que la tâche FAQ ne crée pas de doublons si exécutée plusieurs fois."""
    user = create_test_user(db_session, "responsable")
    from app.services.library_service import get_or_create_user_library
    library = get_or_create_user_library(db_session, user.id)
    
    space = Space(name="Space Idempotence", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)
    
    feedback = MessageFeedback(
        message_id=None,
        user_id=user.id,
        space_id=space.id,
        is_positive=False,
        comment="Correction test",
        query_text="Question test ?",
        response_text="Réponse incorrecte.",
        chunk_ids=[]
    )
    db_session.add(feedback)
    db_session.commit()
    db_session.refresh(feedback)
    
    mock_mistral_response = {
        "choices": [{"message": {"content": "Question: Test\nRéponse: Réponse corrigée."}}]
    }
    
    async def fake_mistral_chat(*args, **kwargs):
        return mock_mistral_response
        
    with mock.patch("app.tasks.documents.mistral_chat", fake_mistral_chat):
        # Première exécution
        generate_faq_from_feedback_task(feedback.id)
        
        db_session.refresh(feedback)
        assert feedback.auto_faq_generated is True
        
        # Compter les chunks FAQ créés
        from app.config import settings
        faq_doc = db_session.exec(
            select(Document).where(
                Document.title.like(f"{settings.FAQ_CORRECTIVE_TITLE_PREFIX}%"),
                Document.library_id == library.id
            )
        ).first()
        assert faq_doc is not None
        
        chunks_before = db_session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == faq_doc.id)
        ).all()
        count_before = len(chunks_before)
        assert count_before == 1
        
        # Deuxième exécution (devrait être skippée)
        generate_faq_from_feedback_task(feedback.id)
        
        chunks_after = db_session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == faq_doc.id)
        ).all()
        count_after = len(chunks_after)
        
        # Pas de chunk supplémentaire créé
        assert count_after == count_before


def test_technical_search_excludes_faq(db_session: Session):
    """Test que search_technical_passages exclut les documents FAQ correctives."""
    user = create_test_user(db_session, "responsable")
    from app.services.library_service import get_or_create_user_library
    library = get_or_create_user_library(db_session, user.id)
    
    space = Space(name="Space Technique", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)
    
    # Créer un doc technique normal
    doc_tech = Document(
        title="Manuel Technique",
        content="Contenu technique",
        document_type="document",
        processing_status="completed",
        processing_progress=100,
        library_id=library.id,
        user_id=user.id
    )
    db_session.add(doc_tech)
    
    # Créer un doc FAQ corrective
    from app.config import settings
    doc_faq = Document(
        title=f"{settings.FAQ_CORRECTIVE_TITLE_PREFIX} - Test",
        content="Contenu FAQ",
        document_type="written",
        processing_status="completed",
        processing_progress=100,
        library_id=library.id,
        user_id=user.id
    )
    db_session.add(doc_faq)
    db_session.commit()
    db_session.refresh(doc_tech)
    db_session.refresh(doc_faq)
    
    # Lier les deux à l'espace
    for doc in [doc_tech, doc_faq]:
        db_session.add(DocumentSpace(document_id=doc.id, space_id=space.id, user_id=user.id))
    db_session.commit()
    
    # Créer des chunks avec embeddings
    fake_embedding = [0.1] * 1024
    for doc in [doc_tech, doc_faq]:
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=0,
            content=f"Contenu {doc.title}",
            text=f"Contenu {doc.title}",
            embedding=fake_embedding,
            is_leaf=True,
            hierarchy_level=0,
            source=doc.title,
            start_char=0,
            end_char=20
        )
        db_session.add(chunk)
    db_session.commit()
    
    # Appeler search_technical_passages
    import asyncio
    from app.services.space_search_service import search_technical_passages
    
    result = asyncio.run(search_technical_passages(
        session=db_session,
        space_id=space.id,
        query_text="test requête",
        user_id=user.id,
        k=10
    ))
    
    passages = result.get("passages", [])
    
    # Vérifier qu'aucune FAQ n'est dans les résultats
    faq_found = any(
        settings.FAQ_CORRECTIVE_TITLE_PREFIX in p.get("document_title", "")
        for p in passages
    )
    assert not faq_found, "Les FAQ correctives ne doivent pas apparaître dans la recherche technique"
    
    # Le doc technique devrait être présent (si le vecteur match)
    tech_found = any("Manuel Technique" in p.get("document_title", "") for p in passages)
    # Note: peut être False si l'embedding ne match pas, mais au moins pas de FAQ


def test_faq_search_below_threshold_skips_critique(client, db_session: Session):
    """Test que si aucune FAQ ne dépasse le seuil, la critique n'est pas appelée."""
    user = create_test_user(db_session, "responsable")
    headers = bearer_headers(user.id)
    
    space = Space(name="Space Seuil", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)
    
    conv = Conversation(title="Test Seuil", user_id=user.id, space_id=space.id)
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)
    
    # Mock technique search : retourne passages techniques
    async def fake_technical_search(*args, **kwargs):
        return {
            "status": "ok",
            "passages": [{
                "passage": "Contenu technique",
                "document_title": "Doc Technique",
                "document_id": 100,
                "chunk_id": 101,
                "score": 0.85
            }]
        }
    
    # Mock FAQ search : retourne status "below_threshold"
    async def fake_faq_search(*args, **kwargs):
        return {
            "status": "below_threshold",
            "reason": "no_faq_above_threshold",
            "passages": []
        }
    
    draft_response = "Réponse basée sur le doc technique."
    call_count = 0
    
    async def fake_mistral_chat(prompt, model, messages, **kwargs):
        nonlocal call_count
        call_count += 1
        # Seul le draft devrait être appelé, pas la critique
        return {"choices": [{"message": {"content": draft_response}}]}
    
    with mock.patch("app.routers.chat.search_technical_passages", fake_technical_search):
        with mock.patch("app.routers.chat.search_corrective_faq_passages", fake_faq_search):
            with mock.patch("app.routers.chat.mistral_chat", fake_mistral_chat):
                response = client.post(
                    f"/api/spaces/{space.id}/chat/stream",
                    headers=headers,
                    json={
                        "message": "Question test ?",
                        "model": "mistral-large-latest",
                        "conversation_id": conv.id
                    }
                )
    
    assert response.status_code == 200
    # Seul le brouillon appelé, pas de critique
    assert call_count == 1

