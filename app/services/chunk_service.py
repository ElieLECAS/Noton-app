import os
from typing import List, Optional
from sqlmodel import Session, select
from sqlalchemy import or_, delete
from app.models.note_chunk import NoteChunk
from app.models.note import Note
from app.models.document_chunk import DocumentChunk
from app.models.document import Document
import re

from app.services.chunking_service import (
    chunk_note,
    chunk_markdown_hierarchical_with_tables,
    specs_to_note_chunks,
    specs_to_document_chunks,
    CHUNKING_VERSION_MARKDOWN_H2,
    CHUNKING_VERSION_ADAPTIVE,
    resolve_adaptive_chunk_params,
    _detect_content_type,
    _build_page_marker_index,
    _page_no_from_char_offset,
)
from app.database import engine
from app.config import settings
from app.library_document_logging import get_library_document_logger, log_chunk_inventory
import logging
import threading
import time
from datetime import datetime
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
        session.add_all(chunks)
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
            
            note.processing_status = "completed"
            note.processing_progress = 100
            session.add(note)
            session.commit()
            
    except Exception as e:
        logger.error(f"Erreur lors de la génération des embeddings pour la note {note_id}: {e}")


def create_chunks_for_note_from_markdown(
    session: Session,
    note: Note,
    markdown: str,
    generate_embeddings: bool = False,
) -> List[NoteChunk]:
    """Créer les chunks hiérarchiques (HierarchicalNodeParser) à partir du markdown OCR."""
    delete_chunks_for_note(session, note.id, commit=False)

    metadata_base = {
        "note_id": note.id,
        "project_id": note.project_id,
        "user_id": note.user_id,
        "note_title": note.title or "",
    }
    specs = chunk_markdown_hierarchical_with_tables(markdown, metadata_base)
    if not specs:
        chunks = chunk_note(note)
    else:
        chunks = specs_to_note_chunks(note, specs)

    if generate_embeddings and chunks:
        from app.services.embedding_service import generate_embeddings_batch

        leaf_chunks = [c for c in chunks if c.is_leaf]
        if leaf_chunks:
            embeddings = generate_embeddings_batch(
                [c.content for c in leaf_chunks],
                batch_size=settings.EMBEDDING_BATCH_SIZE,
            )
            for chunk, embedding in zip(leaf_chunks, embeddings):
                if embedding:
                    chunk.embedding = embedding

    session.add_all(chunks)
    session.commit()
    logger.info(
        "Créé %d chunks (markdown hiérarchique) pour la note %d",
        len(chunks),
        note.id,
    )
    return chunks


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
                logger.info(
                    "Tâche embeddings dispatchée pour la note %s", note_id
                )
            else:
                logger.info(f"Aucun chunk créé pour la note {note_id}")
                
    except Exception as e:
        logger.error(f"Erreur lors de la re-création des chunks pour la note {note_id}: {e}")


def _enqueue_embeddings_thread_queue(note_id: int, project_id: int):
    """
    Ajouter une note à la file thread pour génération d'embeddings en arrière-plan.
    """
    _ensure_embedding_workers()

    embedding_queue.put((note_id, project_id))
    queue_size = embedding_queue.qsize()
    logger.info(
        "✅ Tâche embeddings (thread) note_id=%s (file: %s)",
        note_id,
        queue_size,
    )


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
    """Supprimer tous les chunks d'une note."""
    result = session.execute(delete(NoteChunk).where(NoteChunk.note_id == note_id))
    deleted_count = result.rowcount

    if commit:
        session.commit()
    logger.debug(f"Supprimé {deleted_count} chunks pour la note {note_id}")


def delete_chunks_for_document(
    session: Session, document_id: int, commit: bool = True
):
    """Supprime tous les chunks d'un document bibliothèque."""
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
    if commit:
        session.commit()
    logger.debug("Supprimé chunks pour document_id=%s", document_id)


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


# ---------------------------------------------------------------------------
# Compatibilité architecture "Document"
# ---------------------------------------------------------------------------

def _try_markdown_h2_sections(text: str) -> Optional[List[dict]]:
    """
    Découpe intermédiaire par sections Markdown (##) si le texte en contient plusieurs.
    Retourne None si non applicable (meilleur que la seule fenêtre glissante).
    """
    if not text or "\n## " not in text:
        return None
    parts = text.split("\n## ")
    if len(parts) < 2:
        return None
    chunks: List[dict] = []
    offset = 0
    for i, part in enumerate(parts):
        block = (f"## {part}" if i else part).strip()
        if not block:
            continue
        start = offset
        end = offset + len(block)
        chunks.append({"content": block, "start_char": start, "end_char": end})
        offset = end + 2
    return chunks if len(chunks) >= 2 else None


def _split_text_for_document(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[dict]:
    """Découpe simple d'un texte en chunks avec overlap."""
    if not text:
        return []
    chunks: List[dict] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "content": chunk_text,
                    "start_char": start,
                    "end_char": end,
                }
            )
        if end >= text_len:
            break
        start = max(0, end - overlap)
    return chunks


def _split_text_adaptive_for_document(text: str) -> List[dict]:
    """
    Découpe par paragraphe avec taille/overlap selon le type (procédure, normatif, description).
    """
    if not text or not text.strip():
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return _split_text_for_document(text.strip())
    chunks: List[dict] = []
    for para in paragraphs:
        ct = _detect_content_type(para)
        chunk_size, overlap = resolve_adaptive_chunk_params(ct)
        sub = _split_text_for_document(para, chunk_size=chunk_size, overlap=overlap)
        for item in sub:
            item = dict(item)
            item["content_type"] = ct
            chunks.append(item)
    return chunks if chunks else _split_text_for_document(text.strip())


def create_chunks_for_document(
    session: Session, document: Document, generate_embeddings: bool = False
) -> List[DocumentChunk]:
    """Créer les chunks pour un document (nouvelle architecture)."""
    ld = get_library_document_logger()
    ld.info(
        "[Chunking] document_id=%s — stratégie FALLBACK (markdown H2 ou fenêtre adaptative). "
        "Raison typique : markdown vide ou échec du chunking hiérarchique. "
        "Conséquence : is_leaf=True partout, node_id/parent_node_id NULL, pas de parents RAG.",
        document.id,
    )
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    session.commit()

    source_text = f"{document.title}\n\n{document.content or ''}".strip()
    page_index = _build_page_marker_index(source_text)
    raw_chunks = _try_markdown_h2_sections(source_text)
    chunking_version = CHUNKING_VERSION_MARKDOWN_H2
    if raw_chunks is None:
        raw_chunks = _split_text_adaptive_for_document(source_text)
        chunking_version = CHUNKING_VERSION_ADAPTIVE
    chunks: List[DocumentChunk] = []
    for idx, item in enumerate(raw_chunks):
        ct = item.get("content_type") or _detect_content_type(item.get("content", ""))
        metadata = {
            "document_id": document.id,
            "library_id": document.library_id,
            "user_id": document.user_id,
            "document_title": document.title or "",
            "chunk_index": idx,
            "chunking_version": chunking_version,
            "content_type": ct,
        }
        page_no = _page_no_from_char_offset(int(item.get("start_char", 0) or 0), page_index)
        if page_no is not None:
            metadata["page_no"] = page_no
        chunks.append(
            DocumentChunk(
                document_id=document.id,
                chunk_index=idx,
                source=document.source,
                content=item["content"],
                text=item["content"],
                start_char=item["start_char"],
                end_char=item["end_char"],
                is_leaf=True,
                hierarchy_level=0,
                metadata_json=metadata,
                metadata_=metadata,
            )
        )

    if generate_embeddings and chunks:
        from app.services.embedding_service import generate_embeddings_batch
        embeddings = generate_embeddings_batch(
            [c.content for c in chunks], batch_size=settings.EMBEDDING_BATCH_SIZE
        )
        model_name = settings.EMBEDDING_MODEL
        for chunk, embedding in zip(chunks, embeddings):
            if embedding:
                chunk.embedding = embedding
                meta = dict(chunk.metadata_json or {})
                meta["embedding_model"] = model_name
                chunk.metadata_json = meta
                chunk.metadata_ = meta

    if chunks:
        session.add_all(chunks)
        session.commit()
    log_chunk_inventory(ld, document.id, chunks, "Chunking fallback terminé")
    return chunks


def create_chunks_for_document_from_markdown(
    session: Session,
    document: Document,
    markdown: str,
    generate_embeddings: bool = False,
) -> List[DocumentChunk]:
    """Chunking hiérarchique depuis markdown Mistral OCR."""
    ld = get_library_document_logger()
    ld.info(
        "[Chunking] document_id=%s — markdown hiérarchique + expansion tableaux.",
        document.id,
    )
    session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    session.commit()

    metadata_base = {
        "document_id": document.id,
        "library_id": document.library_id,
        "user_id": document.user_id,
        "document_title": document.title or "",
    }
    specs = chunk_markdown_hierarchical_with_tables(markdown, metadata_base)
    if not specs:
        return create_chunks_for_document(
            session=session, document=document, generate_embeddings=generate_embeddings
        )

    chunks = specs_to_document_chunks(document, specs)

    if generate_embeddings and chunks:
        from app.services.embedding_service import generate_embeddings_batch

        leafs = [c for c in chunks if c.is_leaf]
        if leafs:
            embeddings = generate_embeddings_batch(
                [c.content for c in leafs], batch_size=settings.EMBEDDING_BATCH_SIZE
            )
            model_name = settings.EMBEDDING_MODEL
            for chunk, embedding in zip(leafs, embeddings):
                if embedding:
                    chunk.embedding = embedding
                    meta = dict(chunk.metadata_json or {})
                    meta["embedding_model"] = model_name
                    chunk.metadata_json = meta
                    chunk.metadata_ = meta

    for c in chunks:
        c.source = document.source

    session.add_all(chunks)
    session.commit()
    log_chunk_inventory(ld, document.id, chunks, "Chunking markdown hiérarchique (persisté)")
    return chunks


def _process_embeddings_for_document(
    document_id: int, run_id: Optional[str] = None
):
    """Génère les embeddings des seuls chunks feuilles (parents exclus), puis marque le document completed."""
    from app.services.document_service_new import (
        LIBRARY_USER_STOPPED_STATUSES,
        _finalize_pipeline_abort,
        _should_abort_processing,
    )
    from app.services.document_run import is_processing_run_current

    ld = get_library_document_logger()
    if run_id is not None and not is_processing_run_current(document_id, run_id):
        ld.info(
            "[Embeddings] document_id=%s — abandon : run_id obsolète.",
            document_id,
        )
        return
    ld.info(
        "[Embeddings] document_id=%s — début : vectorisation des feuilles uniquement "
        "(is_leaf=True ; les parents ne reçoivent pas d'embedding).",
        document_id,
    )
    t_embed = time.perf_counter()
    try:
        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                ld.warning(
                    "[Embeddings] document_id=%s — document introuvable, arrêt.",
                    document_id,
                )
                return

            if _should_abort_processing(document_id):
                _finalize_pipeline_abort(document_id)
                return

            # Feuilles uniquement (parents hiérarchiques sans vecteur)
            statement = select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding.is_(None),
                or_(DocumentChunk.is_leaf == True, DocumentChunk.is_leaf.is_(None)),
            )
            chunks = list(session.exec(statement).all())
            ld.info(
                "[Embeddings] document_id=%s — %d chunk(s) feuille sans embedding à traiter.",
                document_id,
                len(chunks),
            )
            if not chunks:
                document.processing_status = "completed"
                document.processing_progress = 100
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
                ld.info(
                    "[Embeddings] document_id=%s — rien à embedder (déjà fait ou aucune feuille), statut completed.",
                    document_id,
                )
                return

            document.processing_progress = max(document.processing_progress or 0, 90)
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()

            logger.info(
                "document_id=%s génération embeddings pour %s chunks feuilles",
                document_id,
                len(chunks),
            )
            from app.services.embedding_service import generate_embeddings_batch

            model_name = settings.EMBEDDING_MODEL
            ok = 0
            batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
            for i in range(0, len(chunks), batch_size):
                if _should_abort_processing(document_id):
                    ld.warning(
                        "[Embeddings] document_id=%s — interrompu pendant embeddings (lot %s/%s).",
                        document_id,
                        i // batch_size + 1,
                        (len(chunks) + batch_size - 1) // batch_size,
                    )
                    _finalize_pipeline_abort(document_id)
                    return
                batch = chunks[i : i + batch_size]
                embeddings = generate_embeddings_batch(
                    [c.content for c in batch], batch_size=len(batch)
                )
                for chunk, embedding in zip(batch, embeddings):
                    if embedding:
                        chunk.embedding = embedding
                        meta = dict(chunk.metadata_json or {})
                        meta["embedding_model"] = model_name
                        chunk.metadata_json = meta
                        chunk.metadata_ = meta
                        session.add(chunk)
                        ok += 1
                session.commit()

            if ok == 0:
                logger.warning("Aucun embedding valide pour document_id=%s", document_id)
                ld.error(
                    "[Embeddings] document_id=%s — aucun vecteur valide écrit, statut failed.",
                    document_id,
                )
                document.processing_status = "failed"
                document.processing_progress = max(document.processing_progress or 0, 90)
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
                return

            session.refresh(document)
            logger.info(
                "document_id=%s embeddings OK en %.2fs (%s vecteurs)",
                document_id,
                time.perf_counter() - t_embed,
                ok,
            )
            ld.info(
                "[Embeddings] document_id=%s — %d vecteur(s) écrits en %.2fs.",
                document_id,
                ok,
                time.perf_counter() - t_embed,
            )

            document.processing_status = "completed"
            document.processing_progress = 100
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()
            ld.info(
                "[Pipeline] document_id=%s — traitement terminé (completed), progress=100.",
                document_id,
            )
    except Exception as e:
        logger.error("Erreur embeddings document %s: %s", document_id, e, exc_info=True)
        get_library_document_logger().error(
            "[Embeddings] document_id=%s — erreur globale : %s",
            document_id,
            e,
            exc_info=True,
        )
        try:
            with Session(engine) as session:
                document = session.get(Document, document_id)
                if document and document.processing_status not in LIBRARY_USER_STOPPED_STATUSES:
                    document.processing_status = "failed"
                    document.processing_progress = max(document.processing_progress or 0, 85)
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                    session.commit()
        except Exception as upd:
            logger.error(
                "Impossible de marquer le document %s en échec: %s", document_id, upd
            )


def complete_document_embeddings_sync(
    document_id: int, run_id: Optional[str] = None
) -> None:
    """Finalise l'indexation : embeddings des feuilles puis statut completed."""
    _process_embeddings_for_document(document_id, run_id)


# Alias rétrocompatibilité (imports existants)
complete_document_embeddings_and_kag_sync = complete_document_embeddings_sync


def generate_embeddings_for_chunks_async(note_id: int, project_id: int):
    """
    Embeddings en arrière-plan : Celery si activé, sinon thread queue ou sync document.
    """
    from app.services.task_dispatch import try_dispatch_embeddings_job

    if try_dispatch_embeddings_job(note_id, project_id):
        return

    document_id_sync: Optional[int] = None
    try:
        with Session(engine) as session:
            note = session.get(Note, note_id)
            if note:
                _enqueue_embeddings_thread_queue(note_id, project_id)
                return
            document = session.get(Document, note_id)
            if document:
                document_id_sync = document.id
    except Exception as e:
        logger.error(
            "Erreur dispatch embeddings async (id=%s): %s", note_id, e, exc_info=True
        )

    if document_id_sync is not None:
        _process_embeddings_for_document(document_id_sync)
        return

    _enqueue_embeddings_thread_queue(note_id, project_id)

