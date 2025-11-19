from typing import List, Optional
from datetime import datetime
from sqlmodel import Session, select
from app.models.note import Note, NoteCreate, NoteUpdate
from app.services.project_service import get_project_by_id
from app.services.chunk_service import create_chunks_for_note, delete_chunks_for_note
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
    """Créer une nouvelle note avec chunks et embeddings générés immédiatement"""
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
    
    # Créer les chunks avec embeddings IMMÉDIATEMENT (synchrone)
    # SAUF si le statut est 'pending' (documents en cours de traitement)
    if note.processing_status != 'pending':
        try:
            chunks = create_chunks_for_note(session, note, generate_embeddings=True)
            logger.info(f"✅ {len(chunks)} chunks créés avec embeddings pour la note '{note.title}'")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création des chunks pour la note '{note.title}': {e}", exc_info=True)
            # Ne pas bloquer la création si les chunks échouent
    else:
        logger.info(f"⏳ Note '{note.title}' créée avec statut 'pending', chunks seront créés après traitement")
    
    logger.info(f"📝 Note {note.id} créée avec succès")
    return note


def update_note(session: Session, note_id: int, note_update: NoteUpdate, user_id: int) -> Optional[Note]:
    """Mettre à jour une note avec regénération des chunks et embeddings si nécessaire"""
    note = get_note_by_id(session, note_id, user_id)
    if not note:
        return None
    
    # Vérifier si le titre ou le contenu changent (nécessite re-création des chunks)
    update_data = note_update.model_dump(exclude_unset=True)
    needs_rechunking = 'title' in update_data or 'content' in update_data
    
    for field, value in update_data.items():
        setattr(note, field, value)
    
    note.updated_at = datetime.utcnow()
    
    session.add(note)
    session.commit()
    session.refresh(note)
    
    # Regénérer les chunks IMMÉDIATEMENT si le titre ou le contenu ont changé
    if needs_rechunking:
        try:
            chunks = create_chunks_for_note(session, note, generate_embeddings=True)
            logger.info(f"✅ {len(chunks)} chunks régénérés avec embeddings pour la note {note.id}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la régénération des chunks pour la note {note.id}: {e}", exc_info=True)
            # Ne pas bloquer la mise à jour si les chunks échouent
    
    logger.info(f"✏️ Note {note.id} mise à jour (chunks regénérés: {'oui' if needs_rechunking else 'non'})")
    return note


def delete_note(session: Session, note_id: int, user_id: int) -> bool:
    """Supprimer une note et ses chunks associés (transaction atomique)"""
    note = get_note_by_id(session, note_id, user_id)
    if not note:
        return False
    
    try:
        # Supprimer d'abord tous les chunks associés à la note (sans commit)
        delete_chunks_for_note(session, note_id, commit=False)
        logger.debug(f"Chunks marqués pour suppression pour la note {note_id}")
        
        # Ensuite supprimer la note elle-même
        session.delete(note)
        
        # Commit atomique : soit tout est supprimé, soit rien
        session.commit()
        
        logger.info(f"🗑️ Note {note_id} et ses chunks supprimés avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la note {note_id}: {e}", exc_info=True)
        session.rollback()
        raise

