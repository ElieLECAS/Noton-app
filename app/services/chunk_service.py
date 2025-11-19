from typing import List, Optional
from sqlmodel import Session, select
from app.models.note_chunk import NoteChunk
from app.models.note import Note
from app.services.chunking_service import chunk_note
from app.services.embedding_service import generate_embedding
from app.database import engine
import logging
import threading
import time
from queue import Queue

logger = logging.getLogger(__name__)

# File d'attente pour limiter le nombre de générations simultanées
embedding_queue = Queue()
MAX_CONCURRENT_EMBEDDINGS = 1  # Limiter à 1 génération à la fois pour éviter 100% CPU
embedding_workers = []
_workers_lock = threading.Lock()  # Lock pour la synchronisation des workers


def create_chunks_for_note(session: Session, note: Note, generate_embeddings: bool = False) -> List[NoteChunk]:
    """
    Créer les chunks pour une note.
    Si generate_embeddings=False, les chunks sont créés sans embeddings (rapide).
    Les embeddings peuvent être générés en arrière-plan avec generate_embeddings_for_chunks_async.
    
    Args:
        session: Session SQLModel
        note: La note pour laquelle créer les chunks
        generate_embeddings: Si True, génère les embeddings de manière synchrone (peut être lent)
        
    Returns:
        Liste des chunks créés
    """
    # Supprimer les anciens chunks de la note
    delete_chunks_for_note(session, note.id)
    
    # Créer les nouveaux chunks
    chunks = chunk_note(note)
    
    # Générer les embeddings seulement si demandé (synchrone)
    if generate_embeddings:
        for chunk in chunks:
            embedding = generate_embedding(chunk.content)
            if embedding:
                chunk.embedding = embedding
            else:
                logger.warning(f"Impossible de générer l'embedding pour le chunk {chunk.chunk_index} de la note {note.id}")
    
    # Sauvegarder les chunks dans la DB
    for chunk in chunks:
        session.add(chunk)
    
    session.commit()
    
    logger.info(f"Créé {len(chunks)} chunks pour la note {note.id} (embeddings: {'oui' if generate_embeddings else 'non, sera fait en arrière-plan'})")
    return chunks


def _generate_embeddings_worker():
    """Worker thread qui traite les tâches de génération d'embeddings depuis la file d'attente"""
    logger.info("Worker d'embeddings démarré et en attente de tâches...")
    while True:
        task = None
        try:
            # Worker en attente (pas besoin de logger à chaque itération)
            task = embedding_queue.get()
            if task is None:  # Signal d'arrêt
                logger.info("Signal d'arrêt reçu, arrêt du worker")
                break
            
            note_id, project_id = task
            logger.info(f"Worker traite la tâche pour la note {note_id}")
            _process_embeddings_for_note(note_id, project_id)
            embedding_queue.task_done()
            logger.info(f"Worker a terminé la tâche pour la note {note_id}")
            
            # Délai entre les notes pour éviter de surcharger le CPU
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Erreur dans le worker d'embeddings: {e}", exc_info=True)
            if task:
                embedding_queue.task_done()


def _process_embeddings_for_note(note_id: int, project_id: int):
    """
    Traiter la génération des embeddings pour une note (appelé par le worker).
    
    Args:
        note_id: ID de la note
        project_id: ID du projet
    """
    try:
        logger.info(f"Démarrage de la génération des embeddings pour la note {note_id}")
        
        # Créer une nouvelle session pour ce thread
        with Session(engine) as session:
            # Récupérer la note
            note = session.get(Note, note_id)
            if not note:
                logger.error(f"Note {note_id} non trouvée pour génération d'embeddings")
                return
            
            # Récupérer les chunks sans embeddings
            statement = select(NoteChunk).where(
                NoteChunk.note_id == note_id,
                NoteChunk.embedding.is_(None)
            )
            chunks = list(session.exec(statement).all())
            
            if not chunks:
                logger.info(f"Aucun chunk sans embedding trouvé pour la note {note_id}")
                return
            
            logger.info(f"Génération des embeddings pour {len(chunks)} chunks de la note {note_id}")
            
            # Utiliser le batch processing pour générer tous les embeddings en une seule passe (beaucoup plus rapide)
            from app.services.embedding_service import generate_embeddings_batch
            
            chunk_contents = [chunk.content for chunk in chunks]
            embeddings = generate_embeddings_batch(chunk_contents, batch_size=8)
            
            # Sauvegarder les embeddings dans la base de données (pgvector)
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
                try:
                    if embedding:
                        chunk.embedding = embedding
                        session.add(chunk)
                        logger.debug(f"Embedding généré pour chunk {i}/{len(chunks)} de la note {note_id}")
                    else:
                        logger.warning(f"Impossible de générer l'embedding pour le chunk {chunk.chunk_index} de la note {note_id}")
                        
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde d'embedding pour chunk {chunk.id}: {e}")
                    continue
            
            # Commit une seule fois pour tous les chunks (plus efficace)
            try:
                session.commit()
                logger.info(f"Tous les embeddings sauvegardés pour la note {note_id}")
            except Exception as e:
                logger.error(f"Erreur lors du commit des embeddings: {e}")
                session.rollback()
            
            logger.info(f"Génération des embeddings terminée pour la note {note_id} ({len(chunks)} chunks traités)")
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération des embeddings pour la note {note_id}: {e}")


def recreate_chunks_for_note_async(note_id: int, project_id: int):
    """
    Re-créer les chunks pour une note en arrière-plan (suppression + création + embeddings).
    Cette fonction doit être appelée dans un thread séparé.
    
    Args:
        note_id: ID de la note
        project_id: ID du projet
    """
    try:
        logger.info(f"Démarrage de la re-création des chunks pour la note {note_id}")
        
        # Créer une nouvelle session pour ce thread
        with Session(engine) as session:
            # Récupérer la note
            note = session.get(Note, note_id)
            if not note:
                logger.error(f"Note {note_id} non trouvée pour re-création des chunks")
                return
            
            # Créer les nouveaux chunks SANS embeddings (rapide)
            # create_chunks_for_note supprime déjà les anciens chunks en interne
            chunks = create_chunks_for_note(session, note, generate_embeddings=False)
            
            # Ajouter à la file d'attente pour génération d'embeddings en arrière-plan
            if chunks:
                generate_embeddings_for_chunks_async(note.id, project_id)
                logger.info(f"Tâche de génération d'embeddings ajoutée à la file pour la note {note_id}")
            else:
                logger.info(f"Aucun chunk créé pour la note {note_id}")
                
    except Exception as e:
        logger.error(f"Erreur lors de la re-création des chunks pour la note {note_id}: {e}")


def generate_embeddings_for_chunks_async(note_id: int, project_id: int):
    """
    Ajouter une note à la file d'attente pour génération d'embeddings en arrière-plan.
    Cette fonction est non-bloquante et retourne immédiatement.
    
    Args:
        note_id: ID de la note
        project_id: ID du projet
    """
    # S'assurer que les workers sont démarrés
    _ensure_embedding_workers()
    
    # Ajouter la tâche à la file d'attente
    embedding_queue.put((note_id, project_id))
    queue_size = embedding_queue.qsize()
    logger.info(f"✅ Tâche de génération d'embeddings ajoutée à la file pour la note {note_id} (taille de la file: {queue_size})")


def _ensure_embedding_workers():
    """S'assurer que les workers d'embedding sont démarrés"""
    global embedding_workers
    
    with _workers_lock:
        if not embedding_workers or not any(w.is_alive() for w in embedding_workers):
            # Redémarrer les workers s'ils sont morts
            embedding_workers = []
            for i in range(MAX_CONCURRENT_EMBEDDINGS):
                worker = threading.Thread(target=_generate_embeddings_worker, daemon=True)
                worker.start()
                embedding_workers.append(worker)
                logger.info(f"Worker d'embeddings {i+1} démarré")


def delete_chunks_for_note(session: Session, note_id: int, commit: bool = True):
    """
    Supprimer tous les chunks d'une note.
    
    Args:
        session: Session SQLModel
        note_id: ID de la note
        commit: Si True, fait un commit après la suppression (par défaut: True)
    """
    statement = select(NoteChunk).where(NoteChunk.note_id == note_id)
    chunks = list(session.exec(statement).all())
    
    for chunk in chunks:
        session.delete(chunk)
    
    if commit:
        session.commit()
    logger.debug(f"Supprimé {len(chunks)} chunks pour la note {note_id}")


def get_chunks_by_note(session: Session, note_id: int) -> List[NoteChunk]:
    """
    Récupérer tous les chunks d'une note.
    
    Args:
        session: Session SQLModel
        note_id: ID de la note
        
    Returns:
        Liste des chunks triés par chunk_index
    """
    statement = select(NoteChunk).where(NoteChunk.note_id == note_id).order_by(NoteChunk.chunk_index)
    return list(session.exec(statement).all())


def get_chunks_by_ids(session: Session, chunk_ids: List[int]) -> List[NoteChunk]:
    """
    Récupérer des chunks par leurs IDs.
    
    Args:
        session: Session SQLModel
        chunk_ids: Liste des IDs de chunks
        
    Returns:
        Liste des chunks
    """
    if not chunk_ids:
        return []
    
    statement = select(NoteChunk).where(NoteChunk.id.in_(chunk_ids))
    return list(session.exec(statement).all())

