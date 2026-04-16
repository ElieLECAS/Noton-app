import pytest
from unittest import mock
from sqlmodel import Session
from datetime import datetime
from app.database import engine
from app.models.document import Document
from app.services.document_service_new import (
    _process_document_for_id,
    DOCUMENT_STATUS_CANCELLED_BY_USER,
    _mark_document_processing_cancelled,
    _clear_document_processing_cancelled
)

from app.models.user import User
from app.models.library import Library

def test_status_overwrite_race_condition(db_session: Session):
    """
    Vérifie qu'un worker ne peut pas écraser le statut 'cancelled_by_user' 
    avec 'processing' après une annulation.
    """
    # 0. Créer un user et une library réels
    user = User(username="test_race", email="race@noton.app", password_hash="xxx")
    db_session.add(user)
    db_session.flush()
    lib = Library(name="Test Lib", user_id=user.id)
    db_session.add(lib)
    db_session.flush()

    # 1. Créer un document en attente
    doc = Document(
        title="Test Race Condition",
        document_type="document",
        processing_status="pending",
        library_id=lib.id,
        user_id=user.id,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    doc_id = doc.id

    # 2. Simuler une annulation par l'utilisateur
    doc.processing_status = DOCUMENT_STATUS_CANCELLED_BY_USER
    db_session.add(doc)
    db_session.commit()
    _mark_document_processing_cancelled(doc_id)

    # 3. Simuler le worker qui tente de passer en 'processing'
    # On mocke les étapes lourdes pour aller vite
    with mock.patch("app.services.document_service_new.ensure_pdf_for_docling"), \
         mock.patch("app.services.document_service_new.process_document_file", return_value=(None, None, None)):
        
        # On appelle le début de la fonction de traitement
        try:
            _process_document_for_id(doc_id, "dummy_path")
        except Exception:
            pass # On s'attend à ce qu'il sorte via un abort ou une erreur gérée

    # 4. Vérifier que le statut n'a pas bougé de 'cancelled_by_user'
    db_session.refresh(doc)
    assert doc.processing_status == DOCUMENT_STATUS_CANCELLED_BY_USER, \
        f"Le statut a été écrasé par {doc.processing_status} !"
    
    _clear_document_processing_cancelled(doc_id)

def test_should_abort_processing_honors_db_status(db_session: Session):
    """Vérifie que _should_abort_processing voit bien le statut en base."""
    from app.services.document_service_new import _should_abort_processing
    
    # 0. Créer un user et une library réels
    user = User(username="test_abort", email="abort@noton.app", password_hash="xxx")
    db_session.add(user)
    db_session.flush()
    lib = Library(name="Test Lib Abort", user_id=user.id)
    db_session.add(lib)
    db_session.flush()

    doc = Document(
        title="Test Abort",
        document_type="document",
        processing_status=DOCUMENT_STATUS_CANCELLED_BY_USER,
        library_id=lib.id,
        user_id=user.id,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    
    assert _should_abort_processing(doc.id) is True
