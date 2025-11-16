from typing import List, Optional
from datetime import datetime
from sqlmodel import Session, select
from app.models.note import Note, NoteCreate, NoteUpdate
from app.services.project_service import get_project_by_id
from app.services.embedding_service import generate_note_embedding
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
    """Créer une nouvelle note avec embedding généré immédiatement"""
    # Vérifier que le projet appartient à l'utilisateur
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        return None
    
    note = Note(
        **note_create.model_dump(),
        project_id=project_id,
        user_id=user_id
    )
    
    # Générer l'embedding IMMÉDIATEMENT (synchrone)
    try:
        embedding = generate_note_embedding(note.title, note.content)
        if embedding:
            note.embedding = embedding
            logger.info(f"✅ Embedding généré pour la note '{note.title}'")
        else:
            logger.warning(f"⚠️ Impossible de générer l'embedding pour la note '{note.title}'")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération d'embedding pour la note '{note.title}': {e}")
        # Ne pas bloquer la création si l'embedding échoue
    
    session.add(note)
    session.commit()
    session.refresh(note)
    
    logger.info(f"📝 Note {note.id} créée avec succès (embedding: {'oui' if note.embedding else 'non'})")
    return note


def update_note(session: Session, note_id: int, note_update: NoteUpdate, user_id: int) -> Optional[Note]:
    """Mettre à jour une note avec regénération d'embedding si nécessaire"""
    note = get_note_by_id(session, note_id, user_id)
    if not note:
        return None
    
    # Vérifier si le titre ou le contenu changent (nécessite re-génération d'embedding)
    update_data = note_update.model_dump(exclude_unset=True)
    needs_reembedding = 'title' in update_data or 'content' in update_data
    
    for field, value in update_data.items():
        setattr(note, field, value)
    
    note.updated_at = datetime.utcnow()
    
    # Regénérer l'embedding IMMÉDIATEMENT si le titre ou le contenu ont changé
    if needs_reembedding:
        try:
            embedding = generate_note_embedding(note.title, note.content)
            if embedding:
                note.embedding = embedding
                logger.info(f"✅ Embedding régénéré pour la note {note.id}")
            else:
                logger.warning(f"⚠️ Impossible de régénérer l'embedding pour la note {note.id}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la régénération d'embedding pour la note {note.id}: {e}")
            # Ne pas bloquer la mise à jour si l'embedding échoue
    
    session.add(note)
    session.commit()
    session.refresh(note)
    
    logger.info(f"✏️ Note {note.id} mise à jour (embedding regénéré: {'oui' if needs_reembedding else 'non'})")
    return note


def delete_note(session: Session, note_id: int, user_id: int) -> bool:
    """Supprimer une note"""
    note = get_note_by_id(session, note_id, user_id)
    if not note:
        return False
    
    session.delete(note)
    session.commit()
    
    logger.info(f"🗑️ Note {note_id} supprimée")
    return True

