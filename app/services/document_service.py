from typing import Optional
import logging
import os
from pathlib import Path
import threading
import time
from queue import Queue
from sqlmodel import Session
from app.config import settings
from app.database import engine
from app.models.note import Note
from app.services.chunk_service import (
    create_chunks_for_note,
    create_chunks_for_note_from_markdown,
    generate_embeddings_for_chunks_async,
)
from datetime import datetime

logger = logging.getLogger(__name__)

# Files d'attente par projet pour traiter les documents séquentiellement par projet
project_queues: dict[int, Queue] = {}
project_locks: dict[int, threading.Lock] = {}
_queues_lock = threading.Lock()
document_workers = []
_document_workers_lock = threading.Lock()

def process_document(file_path: str) -> Optional[str]:
    """Extrait le markdown via Mistral OCR (notes/projet)."""
    from app.services.document_service_new import process_document_file

    return process_document_file(file_path)


def extract_text_from_epub(file_path: str) -> Optional[str]:
    """
    Extrait le texte brut d'un EPUB via stdlib (zip+xml/html), sans dépendance externe.
    """
    import re
    import zipfile
    from html import unescape

    try:
        chunks: list[str] = []
        with zipfile.ZipFile(file_path, "r") as zf:
            names = [
                n for n in zf.namelist()
                if n.lower().endswith((".xhtml", ".html", ".htm"))
            ]
            for name in sorted(names):
                try:
                    raw = zf.read(name)
                    text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    continue

                # Retirer scripts/styles puis balises HTML.
                text = re.sub(
                    r"<(script|style)\b[^>]*>.*?</\1>",
                    " ",
                    text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                text = re.sub(r"<[^>]+>", " ", text)
                text = unescape(text)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    chunks.append(text)

        if not chunks:
            return None

        return "\n\n".join(chunks)
    except Exception as e:
        logger.error("Fallback EPUB échoué pour %s: %s", file_path, e, exc_info=True)
        return None


def save_uploaded_file(
    file_content: bytes, filename: str, upload_dir: str = "media/documents"
) -> Optional[str]:
    """
    Sauvegarde un fichier uploadé sur le disque.

    Args:
        file_content: Contenu binaire du fichier
        filename: Nom du fichier original
        upload_dir: Répertoire de destination

    Returns:
        Chemin complet du fichier sauvegardé ou None en cas d'erreur
    """
    try:
        upload_path = Path(upload_dir)
        upload_path.mkdir(parents=True, exist_ok=True)

        file_extension = Path(filename).suffix
        import uuid

        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = upload_path / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info("Fichier sauvegardé: %s", file_path)
        return str(file_path)

    except Exception as e:
        logger.error(
            "Erreur lors de la sauvegarde du fichier %s: %s", filename, e, exc_info=True
        )
        return None


def _process_document_worker():
    """
    Worker thread qui traite les documents depuis les files d'attente par projet.
    """
    try:
        import os

        if hasattr(os, "nice"):
            os.nice(5)
    except Exception:
        pass

    logger.info("Worker de traitement de documents démarré et en attente de tâches...")
    while True:
        task = None
        project_id = None
        project_lock = None
        lock_acquired = False
        try:
            with _queues_lock:
                available_projects = [
                    pid
                    for pid, queue in project_queues.items()
                    if not queue.empty()
                ]

            if not available_projects:
                time.sleep(2)
                continue

            for pid in available_projects:
                with _queues_lock:
                    if pid not in project_locks:
                        project_locks[pid] = threading.Lock()
                    project_lock = project_locks[pid]

                if project_lock.acquire(blocking=False):
                    lock_acquired = True
                    try:
                        with _queues_lock:
                            if pid not in project_queues or project_queues[pid].empty():
                                project_lock.release()
                                lock_acquired = False
                                continue
                            try:
                                task = project_queues[pid].get(block=False)
                            except Exception:
                                project_lock.release()
                                lock_acquired = False
                                continue

                        if task is None:
                            project_lock.release()
                            lock_acquired = False
                            logger.info(
                                "Signal d'arrêt reçu, arrêt du worker de documents"
                            )
                            return

                        project_id = pid
                        note_id, file_path = task
                        logger.info(
                            "Worker traite le document pour la note %d (projet %d)",
                            note_id,
                            project_id,
                        )

                        _process_document_for_note(note_id, file_path)

                        with _queues_lock:
                            if project_id in project_queues:
                                project_queues[project_id].task_done()

                        logger.info(
                            "Worker a terminé le traitement du document pour la note %d (projet %d)",
                            note_id,
                            project_id,
                        )

                        time.sleep(1.0)
                        break

                    except Exception as e:
                        logger.error(
                            "Erreur dans le worker de traitement de documents: %s",
                            e,
                            exc_info=True,
                        )
                        if task and project_id:
                            try:
                                with _queues_lock:
                                    if project_id in project_queues:
                                        project_queues[project_id].task_done()
                            except Exception:
                                pass
                        raise
                    finally:
                        if lock_acquired:
                            try:
                                project_lock.release()
                            except Exception:
                                pass
                            lock_acquired = False
                else:
                    continue

        except Exception as e:
            logger.error(
                "Erreur dans le worker de traitement de documents: %s", e, exc_info=True
            )
            if lock_acquired and project_lock:
                try:
                    project_lock.release()
                except Exception:
                    pass


def _process_document_for_note(note_id: int, file_path: str):
    """
    Traiter un document pour une note (appelé par le worker).

    Mistral OCR → markdown, puis chunking hiérarchique LlamaIndex.
    """
    try:
        logger.info("Démarrage du traitement du document pour la note %d", note_id)

        with Session(engine) as session:
            note = session.get(Note, note_id)
            if not note:
                logger.error(
                    "Note %d non trouvée pour traitement de document", note_id
                )
                return

            note.processing_status = "processing"
            note.processing_progress = 10
            note.updated_at = datetime.utcnow()
            session.add(note)
            session.commit()

            original_ext = Path(file_path).suffix.lower()
            pdf_input_path = file_path
            converted_to_pdf = False
            try:
                from app.services.file_conversion import ensure_pdf_for_ocr

                pdf_input_path = ensure_pdf_for_ocr(file_path)
                converted_to_pdf = pdf_input_path != file_path
            except Exception as e:
                logger.error(
                    "Conversion vers PDF impossible pour la note %d (%s): %s",
                    note_id,
                    file_path,
                    e,
                    exc_info=True,
                )
                raise

            markdown_content = process_document(pdf_input_path)

            if not markdown_content:
                note.processing_status = "failed"
                note.processing_progress = max(note.processing_progress or 0, 10)
                note.content = (
                    "❌ Erreur lors du traitement du document. "
                    "Le fichier peut être corrompu ou dans un format non supporté."
                )
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()
                logger.error("Échec du traitement du document pour la note %d", note_id)
                return

            output_ext = Path(pdf_input_path).suffix.lower() or ".bin"
            permanent_output_path = Path(f"media/documents/{note_id}{output_ext}")
            permanent_output_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil

            # Retenter proprement sur relances/erreurs précédentes
            if permanent_output_path.exists():
                permanent_output_path.unlink()

            # Si conversion effectuée, on conserve aussi l'original dans le dossier du document.
            if converted_to_pdf:
                original_permanent_path = Path(
                    f"media/documents/{note_id}{original_ext}"
                )
                original_permanent_path.parent.mkdir(parents=True, exist_ok=True)
                if original_permanent_path.exists():
                    original_permanent_path.unlink()
                shutil.move(file_path, str(original_permanent_path))

            shutil.move(pdf_input_path, str(permanent_output_path))
            note.source_file_path = str(permanent_output_path)
            logger.info(
                "Fichier traité déplacé vers chemin permanent: %s",
                permanent_output_path,
            )
            
            # Garder le markdown dans note.content pour le RAG (chunking, fallback, recherche).
            # L’UI affiche le PDF via source_file_path, pas le contenu éditable.
            note.content = markdown_content
            note.processing_status = "processing"
            note.processing_progress = 55
            note.updated_at = datetime.utcnow()
            session.add(note)
            session.commit()

            logger.info(
                "Document traité avec succès pour la note %d (%d caractères extraits)",
                note_id,
                len(markdown_content),
            )
            
            try:
                if markdown_content:
                    chunks = create_chunks_for_note_from_markdown(
                        session,
                        note,
                        markdown_content,
                        generate_embeddings=False,
                    )
                    logger.info(
                        "Créé %d chunks (markdown hiérarchique) pour la note %d",
                        len(chunks),
                        note_id,
                    )
                else:
                    chunks = create_chunks_for_note(
                        session, note, generate_embeddings=False
                    )
                    logger.info(
                        "Créé %d chunks (HierarchicalNodeParser fallback) pour la note %d",
                        len(chunks),
                        note_id,
                    )

                note.processing_progress = 75
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()

                if chunks:
                    note.processing_progress = 85
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()
                    generate_embeddings_for_chunks_async(note.id, note.project_id)
                    logger.info(
                        "Tâche de génération d'embeddings ajoutée à la file pour la note %d",
                        note_id,
                    )
                else:
                    note.processing_status = "completed"
                    note.processing_progress = 100
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()

            except Exception as e:
                logger.error(
                    "Erreur lors de la création des chunks pour la note %d: %s",
                    note_id,
                    e,
                    exc_info=True,
                )
                note.processing_status = "failed"
                note.processing_progress = max(note.processing_progress or 0, 55)
                note.updated_at = datetime.utcnow()
                session.add(note)
                session.commit()

            # Le fichier traité est conservé de manière permanente dans media/documents/{note_id}{ext}
            # (PDF converti via LibreOffice ou format natif selon le type d'entrée).

    except Exception as e:
        logger.error(
            "Erreur lors du traitement du document pour la note %d: %s",
            note_id,
            e,
            exc_info=True,
        )
        try:
            with Session(engine) as session:
                note = session.get(Note, note_id)
                if note:
                    note.processing_status = "failed"
                    note.processing_progress = max(note.processing_progress or 0, 10)
                    note.content = (
                        f"❌ Erreur lors du traitement du document: {str(e)}"
                    )
                    note.updated_at = datetime.utcnow()
                    session.add(note)
                    session.commit()
        except Exception as update_error:
            logger.error(
                "Erreur lors de la mise à jour du statut d'erreur pour la note %d: %s",
                note_id,
                update_error,
            )


def enqueue_project_document_thread(note_id: int, file_path: str):
    """Ajoute un document à la file thread du projet (sans Celery)."""
    try:
        with Session(engine) as session:
            note = session.get(Note, note_id)
            if not note:
                logger.error(
                    "Note %d non trouvée pour ajout à la queue", note_id
                )
                return
            project_id = note.project_id
    except Exception as e:
        logger.error(
            "Erreur lors de la récupération du project_id pour la note %d: %s",
            note_id,
            e,
            exc_info=True,
        )
        return

    _ensure_document_workers()

    with _queues_lock:
        if project_id not in project_queues:
            project_queues[project_id] = Queue()
            project_locks[project_id] = threading.Lock()

        project_queues[project_id].put((note_id, file_path))
        queue_size = project_queues[project_id].qsize()

    logger.info(
        "✅ Tâche de traitement de document ajoutée à la file du projet %d "
        "pour la note %d (taille de la file: %d)",
        project_id,
        note_id,
        queue_size,
    )


def process_document_async(note_id: int, file_path: str):
    """Délègue à Celery ou file thread selon TASK_BACKEND_MODE."""
    from app.services.task_dispatch import dispatch_project_document

    dispatch_project_document(note_id, file_path)


def _ensure_document_workers():
    """S'assurer que les workers de traitement de documents sont démarrés."""
    global document_workers

    with _document_workers_lock:
        if not document_workers or not any(w.is_alive() for w in document_workers):
            document_workers = []
            # Par défaut: traitement séquentiel (1 document à la fois).
            # La valeur est bornée à >= 1 pour éviter toute config invalide.
            num_workers = max(1, settings.MAX_CONCURRENT_DOCUMENTS)
            for i in range(num_workers):
                worker = threading.Thread(
                    target=_process_document_worker, daemon=True
                )
                worker.start()
                document_workers.append(worker)
                logger.info("Worker de traitement de documents %d démarré", i + 1)
