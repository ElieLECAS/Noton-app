from typing import List, Optional
from sqlmodel import Session, select
from sqlalchemy import or_
from app.models.note_chunk import NoteChunk
from app.models.note import Note
from app.services.chunking_service import chunk_note, chunk_note_from_docling_docs
from app.database import engine
from app.config import settings
import logging
import threading
import time
from queue import Queue
import io

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
    
    # Générer les embeddings seulement sur les feuilles si demandé (synchrone)
    if generate_embeddings:
        from app.services.embedding_service import generate_embeddings_batch

        leaf_chunks = [chunk for chunk in chunks if chunk.is_leaf]
        if leaf_chunks:
            contents = [chunk.content for chunk in leaf_chunks]
            embeddings = generate_embeddings_batch(contents, batch_size=settings.EMBEDDING_BATCH_SIZE)
            failed_count = 0
            for chunk, embedding in zip(leaf_chunks, embeddings):
                if embedding:
                    chunk.embedding = embedding
                else:
                    failed_count += 1
            if failed_count:
                logger.warning(
                    "Embeddings partiels pour note=%s: %s/%s en échec",
                    note.id,
                    failed_count,
                    len(leaf_chunks),
                )
    
    # Sauvegarder les chunks dans la DB
    try:
        for chunk in chunks:
            session.add(chunk)
        
        session.commit()
        
        logger.info(f"Créé {len(chunks)} chunks pour la note {note.id} (embeddings: {'oui' if generate_embeddings else 'non, sera fait en arrière-plan'})")
        return chunks
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des chunks pour la note {note.id}: {e}", exc_info=True)
        session.rollback()
        raise


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
            note.processing_status = "processing"
            note.processing_progress = max(note.processing_progress or 0, 90)
            session.add(note)
            session.commit()
            
            # Récupérer les chunks feuilles sans embeddings (les parents servent au contexte)
            statement = select(NoteChunk).where(
                NoteChunk.note_id == note_id,
                NoteChunk.embedding.is_(None),
                or_(NoteChunk.is_leaf.is_(True), NoteChunk.is_leaf.is_(None)),
            )
            chunks = list(session.exec(statement).all())
            
            if not chunks:
                logger.info(f"Aucun chunk sans embedding trouvé pour la note {note_id}")
                note.processing_status = "completed"
                note.processing_progress = 100
                session.add(note)
                session.commit()
                return
            
            logger.info(f"Génération des embeddings pour {len(chunks)} chunks feuilles de la note {note_id}")
            
            # Utiliser le batch processing pour générer tous les embeddings en une seule passe (beaucoup plus rapide)
            from app.services.embedding_service import generate_embeddings_batch
            
            chunk_contents = [chunk.content for chunk in chunks]
            embeddings = generate_embeddings_batch(chunk_contents, batch_size=settings.EMBEDDING_BATCH_SIZE)
            
            # Filtrer les chunks avec embeddings valides
            chunks_with_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                if embedding:
                    chunks_with_embeddings.append((chunk.id, embedding))
            failed_count = len(chunks) - len(chunks_with_embeddings)
            
            if not chunks_with_embeddings:
                logger.warning(f"Aucun embedding valide généré pour la note {note_id}")
                note.processing_status = "failed"
                note.processing_progress = max(note.processing_progress or 0, 90)
                session.add(note)
                session.commit()
                return
            
            # Utiliser copy_expert pour insertion batch ultra-rapide (beaucoup plus rapide que inserts un par un)
            try:
                # Obtenir la connexion raw psycopg2 depuis SQLAlchemy
                connection = session.connection()
                raw_connection = connection.connection
                
                # Créer une table temporaire pour le COPY
                cursor = raw_connection.cursor()
                try:
                    cursor.execute("""
                        CREATE TEMP TABLE temp_chunk_embeddings (
                            chunk_id INTEGER,
                            embedding_vector TEXT
                        ) ON COMMIT DROP
                    """)
                    
                    # Créer un buffer pour COPY avec les données formatées
                    buffer = io.StringIO()
                    for chunk_id, embedding in chunks_with_embeddings:
                        # Formater l'embedding comme un array PostgreSQL
                        embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                        buffer.write(f"{chunk_id}\t{embedding_str}\n")
                    
                    buffer.seek(0)
                    
                    # Utiliser COPY pour copier les données dans la table temporaire (ultra-rapide)
                    cursor.copy_expert(
                        "COPY temp_chunk_embeddings FROM STDIN WITH (FORMAT text, DELIMITER E'\\t')",
                        buffer
                    )
                    
                    # Mettre à jour les chunks avec les embeddings en une seule requête batch
                    cursor.execute("""
                        UPDATE notechunk
                        SET embedding = temp_chunk_embeddings.embedding_vector::vector
                        FROM temp_chunk_embeddings
                        WHERE notechunk.id = temp_chunk_embeddings.chunk_id
                    """)
                    
                    raw_connection.commit()
                    logger.info(f"✅ {len(chunks_with_embeddings)} embeddings sauvegardés avec copy_expert pour la note {note_id}")
                    
                    # Informer SQLAlchemy que la transaction a été commitée
                    session.commit()
                    
                except Exception as e:
                    raw_connection.rollback()
                    session.rollback()
                    raise e
                finally:
                    cursor.close()
                    buffer.close()
                    
            except Exception as e:
                logger.error(f"Erreur lors de l'insertion batch avec copy_expert: {e}", exc_info=True)
                # Fallback sur la méthode classique si copy_expert échoue
                logger.info("Fallback sur la méthode d'insertion classique")
                for chunk, embedding in zip(chunks, embeddings):
                    if embedding:
                        chunk.embedding = embedding
                        session.add(chunk)
                try:
                    session.commit()
                    logger.info(f"Tous les embeddings sauvegardés (méthode fallback) pour la note {note_id}")
                except Exception as e2:
                    logger.error(f"Erreur lors du commit des embeddings (fallback): {e2}")
                    session.rollback()
            
            logger.info(f"Génération des embeddings terminée pour la note {note_id} ({len(chunks)} chunks traités)")
            if failed_count:
                logger.warning(
                    "Embeddings partiels note=%s: %s/%s en échec",
                    note_id,
                    failed_count,
                    len(chunks),
                )
            
            # Extraction KAG si activée
            if settings.KAG_ENABLED:
                _process_kag_extraction_for_note(session, note_id, project_id)
            
            note.processing_status = "completed"
            note.processing_progress = 100
            session.add(note)
            session.commit()
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération des embeddings pour la note {note_id}: {e}")


def _process_kag_extraction_for_note(session: Session, note_id: int, project_id: int):
    """
    Extraction des entités KAG pour une note (appelé après les embeddings).
    
    Args:
        session: Session SQLModel
        note_id: ID de la note
        project_id: ID du projet
    """
    try:
        from app.services.kag_extraction_service import extract_entities_sync
        from app.services.kag_graph_service import (
            save_entities_for_chunk,
            delete_entities_for_note,
        )
        
        logger.info(f"Démarrage extraction KAG pour la note {note_id}")
        
        delete_entities_for_note(session, note_id)
        
        statement = select(NoteChunk).where(
            NoteChunk.note_id == note_id,
            NoteChunk.is_leaf == True,
        )
        chunks = list(session.exec(statement).all())
        
        if not chunks:
            logger.info(f"Aucun chunk leaf pour extraction KAG note={note_id}")
            return
        
        total_entities = 0
        total_relations = 0
        
        for chunk in chunks:
            if not chunk.content or len(chunk.content.strip()) < 20:
                continue
            
            try:
                entities = extract_entities_sync(chunk.content)
                if entities:
                    relations_count = save_entities_for_chunk(
                        session, chunk, entities, project_id
                    )
                    total_entities += len(entities)
                    total_relations += relations_count
            except Exception as e:
                logger.warning(
                    "Erreur extraction KAG chunk_id=%s: %s",
                    chunk.id,
                    e,
                )
                continue
        
        session.commit()
        logger.info(
            "✅ Extraction KAG terminée note=%s: %d entités, %d relations",
            note_id,
            total_entities,
            total_relations,
        )
        
    except Exception as e:
        logger.error(f"Erreur extraction KAG pour note {note_id}: {e}", exc_info=True)
        session.rollback()


def create_chunks_for_note_from_docling(
    session: Session,
    note: Note,
    llama_docs: list,
    generate_embeddings: bool = False,
) -> List[NoteChunk]:
    """
    Créer les chunks pour un document importé via Docling.

    Utilise DoclingNodeParser (chunking sémantique) au lieu de HierarchicalNodeParser.
    Les llama_docs doivent contenir le JSON sérialisé du DoclingDocument
    (tel que retourné par document_service.process_document).

    Args:
        session          : Session SQLModel
        note             : La note cible
        llama_docs       : Liste de LlamaIndex Document avec JSON Docling
        generate_embeddings : Si True, génère les embeddings synchronement

    Returns:
        Liste des chunks créés
    """
    delete_chunks_for_note(session, note.id)

    chunks = chunk_note_from_docling_docs(note, llama_docs)

    if generate_embeddings:
        from app.services.embedding_service import generate_embeddings_batch

        leaf_chunks = [chunk for chunk in chunks if chunk.is_leaf]
        if leaf_chunks:
            contents = [chunk.content for chunk in leaf_chunks]
            embeddings = generate_embeddings_batch(
                contents, batch_size=settings.EMBEDDING_BATCH_SIZE
            )
            failed_count = 0
            for chunk, embedding in zip(leaf_chunks, embeddings):
                if embedding:
                    chunk.embedding = embedding
                else:
                    failed_count += 1
            if failed_count:
                logger.warning(
                    "Embeddings partiels pour note=%s: %s/%s en échec",
                    note.id,
                    failed_count,
                    len(leaf_chunks),
                )

    try:
        for chunk in chunks:
            session.add(chunk)
        session.commit()
        logger.info(
            "Créé %d chunks (Docling) pour la note %d "
            "(embeddings: %s)",
            len(chunks),
            note.id,
            "oui" if generate_embeddings else "non, sera fait en arrière-plan",
        )
        return chunks
    except Exception as e:
        logger.error(
            "Erreur lors de la sauvegarde des chunks Docling pour la note %d: %s",
            note.id,
            e,
            exc_info=True,
        )
        session.rollback()
        raise


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
    Supprimer tous les chunks d'une note et les relations KAG associées.
    
    Args:
        session: Session SQLModel
        note_id: ID de la note
        commit: Si True, fait un commit après la suppression (par défaut: True)
    """
    if settings.KAG_ENABLED:
        try:
            from app.services.kag_graph_service import delete_entities_for_note
            delete_entities_for_note(session, note_id)
        except Exception as e:
            logger.warning(f"Erreur suppression relations KAG note={note_id}: {e}")
    
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


def reindex_notes(
    session: Session,
    note_ids: Optional[List[int]] = None,
    project_id: Optional[int] = None,
) -> int:
    """
    Réindexer des notes en recréant leurs nœuds hiérarchiques + embeddings.

    Args:
        session: Session SQLModel
        note_ids: Liste optionnelle d'IDs de notes à réindexer
        project_id: ID de projet optionnel (ignoré si note_ids est fourni)

    Returns:
        Nombre de notes réindexées avec succès
    """
    statement = select(Note)
    if note_ids:
        statement = statement.where(Note.id.in_(note_ids))
    elif project_id is not None:
        statement = statement.where(Note.project_id == project_id)

    notes = list(session.exec(statement).all())
    reindexed_count = 0

    for note in notes:
        try:
            create_chunks_for_note(session, note, generate_embeddings=True)
            reindexed_count += 1
        except Exception as exc:
            logger.error("Erreur de réindexation pour la note %s: %s", note.id, exc, exc_info=True)

    logger.info("Réindexation terminée: %s/%s notes", reindexed_count, len(notes))
    return reindexed_count

