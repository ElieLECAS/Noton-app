"""
Script de migration pour créer les chunks des notes existantes.

Usage:
    python -m app.scripts.migrate_notes_to_chunks
"""
import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sqlmodel import Session, select
from app.database import engine
from app.models.note import Note
from app.services.chunk_service import create_chunks_for_note, generate_embeddings_for_chunks_async
from app.services.faiss_service import get_faiss_manager
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_notes_to_chunks():
    """Créer les chunks pour toutes les notes existantes (sans embeddings, qui seront générés en arrière-plan)"""
    logger.info("Démarrage de la migration des notes vers chunks...")
    
    faiss_manager = get_faiss_manager()
    faiss_manager.initialize()
    
    with Session(engine) as session:
        # Récupérer toutes les notes
        statement = select(Note)
        notes = list(session.exec(statement).all())
        
        logger.info(f"Trouvé {len(notes)} notes à migrer")
        
        if not notes:
            logger.info("Aucune note à migrer")
            return
        
        success_count = 0
        error_count = 0
        
        for i, note in enumerate(notes, 1):
            try:
                logger.info(f"Traitement de la note {i}/{len(notes)}: {note.title} (ID: {note.id})")
                
                # Créer les chunks SANS embeddings (rapide)
                chunks = create_chunks_for_note(session, note, generate_embeddings=False)
                
                if chunks:
                    # Ajouter à la file d'attente pour génération d'embeddings en arrière-plan
                    generate_embeddings_for_chunks_async(note.id, note.project_id)
                    success_count += 1
                    logger.info(f"✓ Créé {len(chunks)} chunks pour la note {note.id} (embeddings en cours de génération en arrière-plan)")
                else:
                    logger.warning(f"✗ Aucun chunk créé pour la note {note.id} (contenu vide ?)")
                    error_count += 1
                    
                # Petit délai pour éviter de surcharger
                time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"✗ Erreur lors du traitement de la note {note.id}: {e}", exc_info=True)
                error_count += 1
                session.rollback()
                continue
        
        logger.info(f"\nMigration terminée:")
        logger.info(f"  - Succès: {success_count}")
        logger.info(f"  - Erreurs: {error_count}")
        logger.info(f"  - Total: {len(notes)}")
        logger.info(f"\nLes embeddings sont générés en arrière-plan par les workers.")
        logger.info(f"Vous pouvez vérifier la progression via l'interface web.")


if __name__ == "__main__":
    migrate_notes_to_chunks()

