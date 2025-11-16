"""
Script de migration pour générer les embeddings des notes existantes.

Usage:
    python -m app.scripts.migrate_existing_notes
    ou depuis le répertoire racine:
    python -m app.scripts.migrate_existing_notes
"""
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlmodel import Session, select
from app.database import engine
from app.models.note import Note
from app.services.embedding_service import generate_note_embedding
from app.services.faiss_service import get_faiss_manager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_existing_notes():
    """Générer les embeddings pour toutes les notes existantes qui n'en ont pas"""
    logger.info("Démarrage de la migration des embeddings...")
    
    faiss_manager = get_faiss_manager()
    faiss_manager.initialize()
    
    with Session(engine) as session:
        # Récupérer toutes les notes sans embedding
        statement = select(Note).where(Note.embedding.is_(None))
        notes = list(session.exec(statement).all())
        
        logger.info(f"Trouvé {len(notes)} notes sans embedding")
        
        if not notes:
            logger.info("Aucune note à migrer")
            return
        
        success_count = 0
        error_count = 0
        
        for i, note in enumerate(notes, 1):
            try:
                logger.info(f"Traitement de la note {i}/{len(notes)}: {note.title} (ID: {note.id})")
                
                # Générer l'embedding
                embedding = generate_note_embedding(note.title, note.content)
                
                if embedding:
                    # Mettre à jour la note dans la DB
                    note.embedding = embedding
                    session.add(note)
                    session.commit()
                    session.refresh(note)
                    
                    # Ajouter à FAISS
                    faiss_manager.add_note(note.id, embedding, note.project_id)
                    
                    success_count += 1
                    logger.info(f"✓ Embedding généré pour la note {note.id}")
                else:
                    logger.warning(f"✗ Impossible de générer l'embedding pour la note {note.id}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"✗ Erreur lors du traitement de la note {note.id}: {e}")
                error_count += 1
                session.rollback()
                continue
        
        logger.info(f"\nMigration terminée:")
        logger.info(f"  - Succès: {success_count}")
        logger.info(f"  - Erreurs: {error_count}")
        logger.info(f"  - Total: {len(notes)}")


if __name__ == "__main__":
    migrate_existing_notes()

