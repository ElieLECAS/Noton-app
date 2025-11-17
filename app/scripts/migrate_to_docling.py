"""
Script de migration pour convertir les notes existantes vers le système DocumentChunk.
Ce script doit être exécuté après la migration Alembic du schéma de base de données.
"""
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sqlmodel import Session, select
from app.database import engine
from app.models.note import Note
from app.models.document_chunk import DocumentChunk
from app.services.document_chunking_service import get_chunking_service
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def migrate_notes_to_chunks():
    """
    Migrer toutes les notes existantes vers le nouveau système de chunks.
    """
    logger.info("🚀 Début de la migration vers le système DocumentChunk")
    
    with Session(engine) as session:
        # Récupérer toutes les notes
        statement = select(Note)
        notes = session.exec(statement).all()
        
        logger.info(f"📝 {len(notes)} notes trouvées à migrer")
        
        if len(notes) == 0:
            logger.info("✅ Aucune note à migrer")
            return
        
        chunking_service = get_chunking_service()
        
        total_chunks_created = 0
        notes_migrated = 0
        notes_failed = 0
        
        for i, note in enumerate(notes, 1):
            try:
                logger.info(f"[{i}/{len(notes)}] Migration de la note {note.id}: '{note.title}'")
                
                # Vérifier si la note a déjà des chunks
                existing_chunks = session.exec(
                    select(DocumentChunk).where(DocumentChunk.note_id == note.id)
                ).all()
                
                if existing_chunks:
                    logger.info(f"  ⚠️ Note {note.id} a déjà {len(existing_chunks)} chunks, skip")
                    continue
                
                # Créer les chunks
                chunks = chunking_service.chunk_note(note)
                logger.info(f"  📦 {len(chunks)} chunks créés")
                
                # Générer les embeddings
                chunks = chunking_service.generate_embeddings_for_chunks(chunks)
                
                # Compter les chunks avec embeddings
                chunks_with_embeddings = sum(1 for c in chunks if c.embedding is not None)
                logger.info(f"  🔢 {chunks_with_embeddings}/{len(chunks)} embeddings générés")
                
                # Sauvegarder les chunks
                for chunk in chunks:
                    session.add(chunk)
                
                session.commit()
                
                total_chunks_created += len(chunks)
                notes_migrated += 1
                logger.info(f"  ✅ Note {note.id} migrée avec succès")
                
            except Exception as e:
                logger.error(f"  ❌ Erreur lors de la migration de la note {note.id}: {e}", exc_info=True)
                notes_failed += 1
                session.rollback()
                continue
        
        logger.info("\n" + "="*60)
        logger.info("🎉 Migration terminée !")
        logger.info(f"  ✅ Notes migrées: {notes_migrated}/{len(notes)}")
        logger.info(f"  ❌ Notes échouées: {notes_failed}")
        logger.info(f"  📦 Total chunks créés: {total_chunks_created}")
        logger.info("="*60)


def cleanup_old_embeddings():
    """
    Nettoyer les anciens embeddings au niveau des notes (optionnel).
    Cette fonction peut être appelée après avoir vérifié que le nouveau système fonctionne.
    """
    logger.info("\n🧹 Nettoyage des anciens embeddings (champ deprecated)")
    logger.info("  ⚠️ Cette opération supprime les anciens embeddings des notes")
    logger.info("  ⚠️ Les nouveaux embeddings dans DocumentChunk seront conservés")
    
    response = input("\nContinuer ? (oui/non): ")
    if response.lower() not in ['oui', 'yes', 'y', 'o']:
        logger.info("❌ Nettoyage annulé")
        return
    
    with Session(engine) as session:
        # Compter les notes avec embeddings
        statement = select(Note).where(Note.embedding.is_not(None))
        notes_with_embeddings = session.exec(statement).all()
        
        logger.info(f"  📊 {len(notes_with_embeddings)} notes ont des anciens embeddings")
        
        if len(notes_with_embeddings) == 0:
            logger.info("  ✅ Aucun ancien embedding à nettoyer")
            return
        
        # Supprimer les anciens embeddings
        for note in notes_with_embeddings:
            note.embedding = None
            session.add(note)
        
        session.commit()
        logger.info(f"  ✅ {len(notes_with_embeddings)} anciens embeddings supprimés")


def verify_migration():
    """
    Vérifier que la migration s'est bien passée.
    """
    logger.info("\n🔍 Vérification de la migration")
    
    with Session(engine) as session:
        # Compter les notes
        notes_count = len(session.exec(select(Note)).all())
        
        # Compter les chunks
        chunks_count = len(session.exec(select(DocumentChunk)).all())
        
        # Compter les chunks avec embeddings
        chunks_with_embeddings = len(session.exec(
            select(DocumentChunk).where(DocumentChunk.embedding.is_not(None))
        ).all())
        
        # Compter les notes sans chunks
        notes_without_chunks = 0
        for note in session.exec(select(Note)).all():
            chunks = session.exec(
                select(DocumentChunk).where(DocumentChunk.note_id == note.id)
            ).all()
            if len(chunks) == 0:
                notes_without_chunks += 1
        
        logger.info(f"  📊 Statistiques :")
        logger.info(f"    - Notes totales: {notes_count}")
        logger.info(f"    - Chunks totaux: {chunks_count}")
        logger.info(f"    - Chunks avec embeddings: {chunks_with_embeddings}/{chunks_count} ({chunks_with_embeddings/chunks_count*100 if chunks_count > 0 else 0:.1f}%)")
        logger.info(f"    - Notes sans chunks: {notes_without_chunks}")
        
        if notes_without_chunks > 0:
            logger.warning(f"  ⚠️ {notes_without_chunks} notes n'ont pas de chunks !")
        else:
            logger.info("  ✅ Toutes les notes ont des chunks")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migration vers le système DocumentChunk")
    parser.add_argument('--migrate', action='store_true', help='Migrer les notes vers chunks')
    parser.add_argument('--verify', action='store_true', help='Vérifier la migration')
    parser.add_argument('--cleanup', action='store_true', help='Nettoyer les anciens embeddings')
    parser.add_argument('--all', action='store_true', help='Tout faire : migrer + vérifier')
    
    args = parser.parse_args()
    
    if args.all:
        migrate_notes_to_chunks()
        verify_migration()
    elif args.migrate:
        migrate_notes_to_chunks()
    elif args.verify:
        verify_migration()
    elif args.cleanup:
        cleanup_old_embeddings()
    else:
        parser.print_help()

