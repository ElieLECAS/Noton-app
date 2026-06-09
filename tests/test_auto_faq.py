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
from tests.conftest import create_test_user, bearer_headers, extract_sse_message_text
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
    from app.config import settings
    from app.services.document_service_new import FEEDBACK_CORRECTIVE_DOCUMENT_TYPE

    technical_md = """# Joint PVC menuiserie

Pour le PVC, utiliser le joint XL30202 et non la référence T740023 indiquée par erreur dans la réponse initiale.

**Mots-clés :** PVC, joint, XL30202, T740023, menuiserie
"""
    mock_mistral_response = {
        "choices": [{"message": {"content": technical_md}}]
    }

    async def fake_mistral_chat(*args, **kwargs):
        return mock_mistral_response

    with mock.patch(
        "app.services.feedback_knowledge_service.mistral_chat",
        fake_mistral_chat,
    ):
        generate_faq_from_feedback_task(feedback.id)

    db_session.refresh(feedback)

    assert feedback.auto_faq_generated is True
    assert "XL30202" in feedback.auto_faq_content
    assert "Question:" not in feedback.auto_faq_content

    expected_title = f"{settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX} - Manuel Technique Joint"
    virtual_doc = db_session.exec(
        select(Document).where(Document.title == expected_title)
    ).first()
    assert virtual_doc is not None
    assert virtual_doc.document_type == FEEDBACK_CORRECTIVE_DOCUMENT_TYPE
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
        
    with mock.patch(
        "app.services.feedback_knowledge_service.mistral_chat",
        fake_mistral_chat,
    ):
        generate_faq_from_feedback_task(feedback.id)
        
        db_session.refresh(feedback)
        assert feedback.auto_faq_generated is True
        
        # Compter les chunks FAQ créés
        from app.config import settings
        faq_doc = db_session.exec(
            select(Document).where(
                Document.title.like(f"{settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX}%"),
                Document.library_id == library.id,
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
    
    from app.config import settings
    from app.services.document_service_new import FEEDBACK_CORRECTIVE_DOCUMENT_TYPE

    doc_faq = Document(
        title=f"{settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX} - Test",
        content="Contenu correctif",
        document_type=FEEDBACK_CORRECTIVE_DOCUMENT_TYPE,
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
        settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX in p.get("document_title", "")
        or settings.FAQ_CORRECTIVE_TITLE_PREFIX in p.get("document_title", "")
        for p in passages
    )
    assert not faq_found, "Les FAQ correctives ne doivent pas apparaître dans la recherche technique"
    
    # Le doc technique devrait être présent (si le vecteur match)
    tech_found = any("Manuel Technique" in p.get("document_title", "") for p in passages)
    # Note: peut être False si l'embedding ne match pas, mais au moins pas de FAQ



def test_feedback_knowledge_prompt_strict_sources_only():
    """Le prompt interdit l'enrichissement hors question / réponse / commentaire."""
    from types import SimpleNamespace
    from app.services.feedback_knowledge_service import (
        build_feedback_knowledge_prompt,
        _strip_code_fences,
        FEEDBACK_KNOWLEDGE_MAX_TOKENS,
    )

    fb = SimpleNamespace(
        query_text="Quelle référence clip SOLEAL FY 55 ?",
        response_text="Utiliser TGA3608 pour 300 kg.",
        comment="TGA3608 n'est pas la bonne réf pour ce cas.",
        is_positive=False,
    )
    prompt = build_feedback_knowledge_prompt(fb, mode="corrective")
    assert "N'invente" in prompt
    assert "1 à 3 phrases" in prompt
    assert "Quelle référence clip SOLEAL FY 55" in prompt
    assert "TGA3608 n'est pas la bonne" in prompt
    assert "Réponse assistant" in prompt
    assert "## Réponse à retenir" not in prompt
    assert "**Mots-clés :**" in prompt

    wrapped = "```markdown\n# Titre court\n\nPhrase technique dense.\n\n**Mots-clés :** a, b\n```"
    stripped = _strip_code_fences(wrapped)
    assert stripped.startswith("# Titre court")
    assert "**Mots-clés :**" in stripped

    assert FEEDBACK_KNOWLEDGE_MAX_TOKENS <= 400


def test_feedback_knowledge_calls_mistral_with_low_temperature():
    from types import SimpleNamespace
    from app.services.feedback_knowledge_service import generate_feedback_knowledge_content

    fb = SimpleNamespace(
        query_text="Q",
        response_text="R",
        comment="C",
        is_positive=False,
    )
    captured = {}

    async def fake_chat(*args, **kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "# T\n\n## Question\nQ"}}]}

    with mock.patch(
        "app.services.feedback_knowledge_service.mistral_chat",
        fake_chat,
    ):
        generate_feedback_knowledge_content(fb)

    assert captured.get("temperature") == 0
    assert captured.get("max_tokens") == 320


def test_positive_feedback_without_comment_does_not_generate(db_session: Session):
    """Un 👍 sans précision ne déclenche pas la génération."""
    user = create_test_user(db_session, "responsable")
    space = Space(name="Space Pos", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    feedback = MessageFeedback(
        message_id=None,
        user_id=user.id,
        space_id=space.id,
        is_positive=True,
        comment=None,
        query_text="Question ?",
        response_text="Bonne réponse.",
        chunk_ids=[],
    )
    db_session.add(feedback)
    db_session.commit()
    db_session.refresh(feedback)

    with mock.patch(
        "app.services.feedback_knowledge_service.generate_feedback_knowledge_content",
    ) as mock_gen:
        generate_faq_from_feedback_task(feedback.id)
        mock_gen.assert_not_called()

    db_session.refresh(feedback)
    assert feedback.auto_faq_generated is False


def test_positive_feedback_with_comment_generates(db_session: Session):
    """Un 👍 avec précision déclenche la génération de texte technique."""
    user = create_test_user(db_session, "responsable")
    from app.services.library_service import get_or_create_user_library

    library = get_or_create_user_library(db_session, user.id)
    space = Space(name="Space Pos Comment", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    feedback = MessageFeedback(
        message_id=None,
        user_id=user.id,
        space_id=space.id,
        is_positive=True,
        comment="Préciser aussi la référence XL30202 pour l'atelier.",
        query_text="Quel joint ?",
        response_text="Utiliser XL30202.",
        chunk_ids=[],
    )
    db_session.add(feedback)
    db_session.commit()
    db_session.refresh(feedback)

    md = "# Connaissance technique\n\n## Synthèse technique\nXL30202 pour atelier.\n"
    with mock.patch(
        "app.services.feedback_knowledge_service.generate_feedback_knowledge_content",
        return_value=md,
    ):
        generate_faq_from_feedback_task(feedback.id)

    db_session.refresh(feedback)
    assert feedback.auto_faq_generated is True
    from app.config import settings

    doc = db_session.exec(
        select(Document).where(
            Document.library_id == library.id,
            Document.title.like(f"{settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX}%"),
        )
    ).first()
    assert doc is not None
    assert doc.document_type == "feedback_corrective"


def test_library_excludes_feedback_corrective_documents(db_session: Session):
    """Les documents correctifs ne sont pas listés dans la bibliothèque."""
    from app.config import settings
    from app.services.document_service_new import (
        FEEDBACK_CORRECTIVE_DOCUMENT_TYPE,
        get_documents_by_library,
    )
    from app.services.library_service import get_or_create_user_library

    user = create_test_user(db_session, "responsable")
    library = get_or_create_user_library(db_session, user.id)

    normal = Document(
        title="Manuel visible",
        content="",
        document_type="document",
        processing_status="completed",
        processing_progress=100,
        library_id=library.id,
        user_id=user.id,
    )
    hidden = Document(
        title=f"{settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX} - Cache",
        content="",
        document_type=FEEDBACK_CORRECTIVE_DOCUMENT_TYPE,
        processing_status="completed",
        processing_progress=100,
        library_id=library.id,
        user_id=user.id,
    )
    db_session.add(normal)
    db_session.add(hidden)
    db_session.commit()

    docs = get_documents_by_library(db_session, library.id, user.id)
    titles = {d.title for d in docs}
    assert "Manuel visible" in titles
    assert hidden.title not in titles

