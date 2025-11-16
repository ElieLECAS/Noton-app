"""
Script pour régénérer tous les embeddings des notes existantes.

Ce script :
1. Supprime tous les chunks existants (ancienne architecture)
2. Génère un embedding pour chaque note (nouvelle architecture simplifiée)

Usage:
    python -m app.scripts.regenerate_all_embeddings
"""

import sys
import logging
from sqlmodel import Session, select
from app.database import engine
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.services.embedding_service import generate_note_embedding

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_all_chunks(session: Session):
    """Supprimer tous les chunks existants (ancienne architecture)"""
    try:
        # Compter les chunks avant suppression
        statement = select(NoteChunk)
        chunks = list(session.exec(statement).all())
        count = len(chunks)
        
        if count == 0:
            logger.info("Aucun chunk à supprimer")
            return
        
        logger.info(f"Suppression de {count} chunks...")
        
        # Supprimer tous les chunks
        for chunk in chunks:
            session.delete(chunk)
        
        session.commit()
        logger.info(f"✅ {count} chunks supprimés avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la suppression des chunks: {e}")
        session.rollback()
        raise


def regenerate_all_embeddings(session: Session):
    """Régénérer les embeddings pour toutes les notes"""
    try:
        # Récupérer toutes les notes
        statement = select(Note)
        notes = list(session.exec(statement).all())
        
        if not notes:
            logger.info("Aucune note trouvée")
            return
        
        logger.info(f"Régénération des embeddings pour {len(notes)} notes...")
        
        success_count = 0
        error_count = 0
        
        for i, note in enumerate(notes, 1):
            try:
                # Générer l'embedding
                embedding = generate_note_embedding(note.title, note.content)
                
                if embedding:
                    note.embedding = embedding
                    session.add(note)
                    success_count += 1
                    logger.info(f"✅ [{i}/{len(notes)}] Embedding généré pour note {note.id}: '{note.title}'")
                else:
                    error_count += 1
                    logger.warning(f"⚠️ [{i}/{len(notes)}] Impossible de générer l'embedding pour note {note.id}: '{note.title}'")
                
                # Commit tous les 10 notes pour éviter les grosses transactions
                if i % 10 == 0:
                    session.commit()
                    logger.info(f"💾 Sauvegarde intermédiaire ({i}/{len(notes)} notes traitées)")
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ [{i}/{len(notes)}] Erreur pour note {note.id}: {e}")
                continue
        
        # Commit final
        session.commit()
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║               MIGRATION TERMINÉE                             ║
╠══════════════════════════════════════════════════════════════╣
║  Total notes traitées : {len(notes):>4}                              ║
║  Succès               : {success_count:>4}                              ║
║  Erreurs              : {error_count:>4}                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la régénération des embeddings: {e}")
        session.rollback()
        raise


def main():
    """Point d'entrée principal du script"""
    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║     RÉGÉNÉRATION DES EMBEDDINGS - NOUVELLE ARCHITECTURE      ║
╠══════════════════════════════════════════════════════════════╣
║  Ce script va :                                              ║
║  1. Supprimer tous les chunks existants (ancienne archi)     ║
║  2. Générer un embedding par note (nouvelle archi)           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        with Session(engine) as session:
            # Étape 1 : Supprimer tous les chunks
            logger.info("\n📋 ÉTAPE 1/2 : Suppression des chunks...")
            delete_all_chunks(session)
            
            # Étape 2 : Régénérer les embeddings
            logger.info("\n📋 ÉTAPE 2/2 : Régénération des embeddings...")
            regenerate_all_embeddings(session)
            
        logger.info("\n✅ Migration terminée avec succès!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Migration échouée: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

