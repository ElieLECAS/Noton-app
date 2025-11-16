from typing import List, Optional
from datetime import datetime
from sqlmodel import Session, select
from app.models.note import Note, NoteCreate, NoteUpdate
from app.services.project_service import get_project_by_id
from app.services.chunk_service import create_chunks_for_note, delete_chunks_for_note
from app.services.faiss_service import get_faiss_manager
import logging

logger = logging.getLogger(__name__)


def get_notes_by_project(session: Session, project_id: int, user_id: int) -> List[Note]:
    """Récupérer toutes les notes d'un projet (vérifie que l'utilisateur en est propriétaire)"""
    # Vérifier que le projet appartient à l'utilisateur
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        return []
    
    statement = select(Note).where(Note.project_id == project_id).order_by(Note.updated_at.desc())
    return list(session.exec(statement).all())


def get_note_by_id(session: Session, note_id: int, user_id: int) -> Optional[Note]:
    """Récupérer une note par son ID (vérifie que l'utilisateur en est propriétaire)"""
    note = session.get(Note, note_id)
    if note and note.user_id == user_id:
        return note
    return None


def create_note(session: Session, note_create: NoteCreate, project_id: int, user_id: int) -> Optional[Note]:
    """Créer une nouvelle note"""
    # Vérifier que le projet appartient à l'utilisateur
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        return None
    
    note = Note(
        **note_create.model_dump(),
        project_id=project_id,
        user_id=user_id
    )
    
    session.add(note)
    session.commit()
    session.refresh(note)
    
    # Créer les chunks en arrière-plan pour ne pas bloquer la réponse HTTP
    import threading
    def create_chunks_async():
        try:
            from app.services.chunk_service import recreate_chunks_for_note_async
            recreate_chunks_for_note_async(note.id, project_id)
        except Exception as e:
            logger.error(f"Erreur lors de la création asynchrone des chunks: {e}")
    
    thread = threading.Thread(target=create_chunks_async, daemon=True)
    thread.start()
    logger.info(f"Tâche de création des chunks lancée en arrière-plan pour la note {note.id}")
    
    return note


def update_note(session: Session, note_id: int, note_update: NoteUpdate, user_id: int) -> Optional[Note]:
    """Mettre à jour une note"""
    note = get_note_by_id(session, note_id, user_id)
    if not note:
        return None
    
    # Vérifier si le titre ou le contenu changent (nécessite re-chunking)
    update_data = note_update.model_dump(exclude_unset=True)
    needs_rechunking = 'title' in update_data or 'content' in update_data
    
    for field, value in update_data.items():
        setattr(note, field, value)
    
    note.updated_at = datetime.utcnow()
    
    session.add(note)
    session.commit()
    session.refresh(note)
    
    # Re-créer les chunks si le titre ou le contenu ont changé (en arrière-plan)
    if needs_rechunking:
        # Lancer la re-création des chunks en arrière-plan pour ne pas bloquer la réponse HTTP
        import threading
        def rechunk_note_async():
            try:
                from app.services.chunk_service import recreate_chunks_for_note_async
                recreate_chunks_for_note_async(note.id, note.project_id)
            except Exception as e:
                logger.error(f"Erreur lors de la re-création asynchrone des chunks: {e}")
        
        thread = threading.Thread(target=rechunk_note_async, daemon=True)
        thread.start()
        logger.info(f"Tâche de re-création des chunks lancée en arrière-plan pour la note {note.id}")
    
    return note


def delete_note(session: Session, note_id: int, user_id: int) -> bool:
    """Supprimer une note"""
    note = get_note_by_id(session, note_id, user_id)
    if not note:
        return False
    
    # Supprimer les chunks de FAISS avant de supprimer de la DB
    try:
        faiss_manager = get_faiss_manager()
        faiss_manager.remove_chunks_for_note(note_id)
    except Exception as e:
        logger.error(f"Erreur lors de la suppression des chunks de FAISS: {e}")
        # Continuer même si FAISS échoue
    
    # Supprimer les chunks de la DB (cascade devrait le faire automatiquement, mais on le fait explicitement)
    delete_chunks_for_note(session, note_id)
    
    session.delete(note)
    session.commit()
    return True

