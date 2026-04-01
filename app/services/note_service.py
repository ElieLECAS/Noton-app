from typing import List, Optional
from datetime import datetime
from sqlmodel import Session, select
from app.models.note import Note, NoteCreate, NoteUpdate
from app.models.document_chunk import DocumentChunk
from app.services.project_service import get_project_by_id
from app.services.embedding_service import generate_note_embedding
from app.services.document_chunking_service import get_chunking_service
from app.services.file_storage_service import get_file_storage_service
import logging
import threading

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
    """
    Créer une nouvelle note avec chunks et embeddings générés via le pipeline unifié.
    Le système de chunks remplace l'embedding direct sur la note.
    """
    # Vérifier que le projet appartient à l'utilisateur
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        return None
    
    note = Note(
        **note_create.model_dump(),
        project_id=project_id,
        user_id=user_id
    )
    
    # DEPRECATED: Conservation temporaire de l'ancien système d'embedding pour compatibilité
    # Sera supprimé après migration complète vers DocumentChunk
    try:
        embedding = generate_note_embedding(note.title, note.content)
        if embedding:
            note.embedding = embedding
    except Exception as e:
        logger.warning(f"⚠️ Ancien système d'embedding échoué (non critique): {e}")
    
    # Sauvegarder la note d'abord pour obtenir un ID
    session.add(note)
    session.commit()
    session.refresh(note)
    
    # NOUVEAU SYSTÈME: Créer les chunks avec embeddings
    try:
        chunking_service = get_chunking_service()
        
        # 1. Créer les chunks (sans embeddings)
        chunks = chunking_service.chunk_note(note)
        logger.info(f"📦 {len(chunks)} chunks créés pour la note {note.id}")
        
        # 2. Générer les embeddings pour tous les chunks
        chunks = chunking_service.generate_embeddings_for_chunks(chunks)
        
        # 3. Sauvegarder les chunks en base
        for chunk in chunks:
            session.add(chunk)
        
        session.commit()
        
        # Compter les chunks avec embeddings
        chunks_with_embeddings = sum(1 for c in chunks if c.embedding is not None)
        logger.info(f"✅ Note {note.id} créée avec {chunks_with_embeddings}/{len(chunks)} chunks avec embeddings")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la création des chunks pour la note {note.id}: {e}", exc_info=True)
        # Ne pas bloquer la création de la note si le chunking échoue
    
    return note


def update_note(session: Session, note_id: int, note_update: NoteUpdate, user_id: int) -> Optional[Note]:
    """
    Mettre à jour une note avec regénération des chunks si nécessaire.
    Si le contenu change, les chunks sont recréés avec nouveaux embeddings.
    """
    note = get_note_by_id(session, note_id, user_id)
    if not note:
        return None
    
    # Vérifier si le titre ou le contenu changent (nécessite re-génération des chunks)
    update_data = note_update.model_dump(exclude_unset=True)
    needs_rechunking = 'title' in update_data or 'content' in update_data
    
    for field, value in update_data.items():
        setattr(note, field, value)
    
    note.updated_at = datetime.utcnow()
    
    # DEPRECATED: Ancien système d'embedding (compatibilité)
    if needs_rechunking:
        try:
            embedding = generate_note_embedding(note.title, note.content)
            if embedding:
                note.embedding = embedding
        except Exception as e:
            logger.warning(f"⚠️ Ancien système d'embedding échoué (non critique): {e}")
    
    session.add(note)
    session.commit()
    
    # NOUVEAU SYSTÈME: Regénérer les chunks si le contenu a changé
    if needs_rechunking:
        try:
            chunking_service = get_chunking_service()
            
            # 1. Supprimer les anciens chunks
            old_chunks = session.exec(
                select(DocumentChunk).where(DocumentChunk.note_id == note.id)
            ).all()
            for chunk in old_chunks:
                session.delete(chunk)
            session.commit()
            
            logger.info(f"🗑️ {len(old_chunks)} anciens chunks supprimés pour la note {note.id}")
            
            # 2. Créer les nouveaux chunks
            chunks = chunking_service.chunk_note(note)
            logger.info(f"📦 {len(chunks)} nouveaux chunks créés pour la note {note.id}")
            
            # 3. Générer les embeddings
            chunks = chunking_service.generate_embeddings_for_chunks(chunks)
            
            # 4. Sauvegarder les nouveaux chunks
            for chunk in chunks:
                session.add(chunk)
            
            session.commit()
            
            chunks_with_embeddings = sum(1 for c in chunks if c.embedding is not None)
            logger.info(f"✅ Note {note.id} mise à jour avec {chunks_with_embeddings}/{len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la régénération des chunks pour la note {note.id}: {e}", exc_info=True)
    
    session.refresh(note)
    logger.info(f"✏️ Note {note.id} mise à jour (chunks regénérés: {'oui' if needs_rechunking else 'non'})")
    return note


def delete_note(session: Session, note_id: int, user_id: int) -> bool:
    """
    Supprimer une note et tous ses fichiers associés.
    Les chunks sont supprimés automatiquement via la cascade.
    """
    note = get_note_by_id(session, note_id, user_id)
    if not note:
        return False
    
    # Supprimer les fichiers uploadés si c'est un document
    if note.source_file_path:
        try:
            file_storage = get_file_storage_service()
            file_storage.delete_note_files(user_id, note.project_id, note_id)
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de la suppression des fichiers: {e}")
    
    session.delete(note)
    session.commit()
    
    logger.info(f"🗑️ Note {note_id} supprimée (avec chunks et fichiers)")
    return True

