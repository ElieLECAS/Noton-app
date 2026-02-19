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
    create_chunks_for_note_from_docling,
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

# DocumentConverter Docling partagé (singleton) — remplace DoclingReader
_docling_converter = None
_docling_converter_lock = threading.Lock()

# Configurer PyTorch pour CPU avant l'import de docling
try:
    import torch
    import warnings

    torch.backends.cudnn.enabled = False

    warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
    warnings.filterwarnings(
        "ignore", message=".*dataloader.*pin_memory.*", category=UserWarning
    )
    warnings.filterwarnings("ignore", message=".*accelerator.*", category=UserWarning)
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.data.dataloader"
    )

    if settings.TORCH_NUM_THREADS is not None:
        torch.set_num_threads(settings.TORCH_NUM_THREADS)
        logger.info(
            "PyTorch configuré avec %d threads (valeur explicite)",
            settings.TORCH_NUM_THREADS,
        )
    else:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        if settings.USE_ALL_CPU_CORES:
            default_threads = cpu_count
            logger.info(
                "PyTorch configuré avec %d threads (tous les %d cœurs disponibles)",
                default_threads,
                cpu_count,
            )
        else:
            default_threads = max(1, cpu_count // 2)
            logger.info(
                "PyTorch configuré avec %d threads (moitié des %d cœurs disponibles)",
                default_threads,
                cpu_count,
            )
        torch.set_num_threads(default_threads)

    if "OMP_NUM_THREADS" in os.environ:
        omp_value = os.environ["OMP_NUM_THREADS"].strip()
        if not omp_value or not omp_value.isdigit() or int(omp_value) <= 0:
            del os.environ["OMP_NUM_THREADS"]

    if settings.OMP_NUM_THREADS is not None and settings.OMP_NUM_THREADS > 0:
        os.environ["OMP_NUM_THREADS"] = str(settings.OMP_NUM_THREADS)
        logger.info(
            "OMP_NUM_THREADS configuré à %d (valeur explicite)", settings.OMP_NUM_THREADS
        )
    elif "OMP_NUM_THREADS" not in os.environ:
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        if settings.USE_ALL_CPU_CORES:
            default_omp_threads = cpu_count
        else:
            default_omp_threads = max(1, cpu_count // 2)
        os.environ["OMP_NUM_THREADS"] = str(default_omp_threads)

    if settings.DOCLING_USE_GPU is False or (
        settings.DOCLING_CPU_ONLY and settings.DOCLING_USE_GPU is None
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logger.info("GPU désactivé pour Docling (mode CPU uniquement)")
    elif settings.DOCLING_USE_GPU is True:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        logger.info("GPU activé pour Docling")

except ImportError:
    logger.warning(
        "PyTorch non disponible, certaines optimisations CPU ne seront pas appliquées"
    )


def get_docling_converter():
    """
    Récupère ou crée le DocumentConverter Docling partagé (singleton).

    Utiliser le DocumentConverter directement (plutôt que DoclingReader) permet
    d'obtenir en une seule passe :
    - le markdown pour note.content (export_to_markdown)
    - le JSON sérialisé pour DoclingNodeParser (model_dump_json)
    """
    global _docling_converter

    with _docling_converter_lock:
        if _docling_converter is None:
            init_start = time.time()
            logger.info(
                "Initialisation du DocumentConverter Docling (une seule fois)..."
            )
            try:
                from docling.document_converter import DocumentConverter

                _docling_converter = DocumentConverter()
                init_time = time.time() - init_start
                logger.info(
                    "✅ DocumentConverter Docling initialisé en %.2fs et prêt à être réutilisé",
                    init_time,
                )
            except Exception as e:
                logger.error(
                    "Erreur lors de l'initialisation du DocumentConverter: %s",
                    e,
                    exc_info=True,
                )
                raise

        return _docling_converter


def process_document(file_path: str) -> tuple[Optional[str], Optional[list]]:
    """
    Traite un document et retourne (markdown, llama_docs_json).

    Une seule passe Docling produit :
    - markdown : texte lisible stocké dans note.content (pour l'affichage)
    - llama_docs : liste de LlamaIndex Document en format JSON Docling,
      prête à être consommée par DoclingNodeParser pour un chunking sémantique

    Args:
        file_path: Chemin vers le fichier à traiter

    Returns:
        (markdown_content, llama_docs) ou (None, None) en cas d'erreur
    """
    import time as _time

    start_time = _time.time()

    if not os.path.exists(file_path):
        logger.error("Fichier non trouvé: %s", file_path)
        return None, None

    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(
            "Démarrage du traitement Docling: %s (%.2f MB)", file_path, file_size_mb
        )

        converter = get_docling_converter()

        # Filtrer les messages de progression Docling
        import sys

        class ProgressFilter:
            def __init__(self, original_stream):
                self.original_stream = original_stream

            def write(self, text):
                if (
                    "Progress:" not in text
                    and "Complete" not in text
                    and "|" not in text[:20]
                ):
                    self.original_stream.write(text)

            def flush(self):
                self.original_stream.flush()

        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = ProgressFilter(sys.stdout)
            sys.stderr = ProgressFilter(sys.stderr)

            conversion_start = _time.time()
            result = converter.convert(file_path)
            conversion_time = _time.time() - conversion_start
            logger.info(
                "Conversion Docling terminée en %.2fs (%.2f min)",
                conversion_time,
                conversion_time / 60,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        docling_doc = result.document

        # Markdown pour note.content (affichage)
        markdown_content = docling_doc.export_to_markdown().strip()

        if not markdown_content:
            logger.warning(
                "Le document %s a été traité mais le contenu markdown est vide",
                file_path,
            )
            return None, None

        # JSON sérialisé pour DoclingNodeParser (chunking sémantique)
        from llama_index.core import Document as LlamaDocument

        llama_docs = [
            LlamaDocument(
                text=docling_doc.model_dump_json(),
                metadata={"source": str(file_path)},
            )
        ]

        total_time = _time.time() - start_time
        logger.info(
            "✅ Document converti en %.2fs (%.2f min) — %d caractères markdown, "
            "%d document(s) JSON Docling",
            total_time,
            total_time / 60,
            len(markdown_content),
            len(llama_docs),
        )
        return markdown_content, llama_docs

    except Exception as e:
        logger.error(
            "Erreur lors du traitement du document %s: %s", file_path, e, exc_info=True
        )
        return None, None


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

    Utilise DocumentConverter Docling directement pour obtenir en une seule passe
    le markdown (note.content) et les LlamaIndex Documents JSON (pour DoclingNodeParser).
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

            # Une seule passe Docling → markdown + llama_docs JSON
            markdown_content, llama_docs = process_document(file_path)

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

            note.content = markdown_content
            note.processing_status = "processing"
            note.processing_progress = 55
            note.updated_at = datetime.utcnow()
            session.add(note)
            session.commit()

            logger.info(
                "Document traité avec succès pour la note %d (%d caractères)",
                note_id,
                len(markdown_content),
            )

            try:
                # Chunking sémantique via DoclingNodeParser si llama_docs disponibles,
                # sinon fallback sur HierarchicalNodeParser
                if llama_docs:
                    chunks = create_chunks_for_note_from_docling(
                        session, note, llama_docs, generate_embeddings=False
                    )
                    logger.info(
                        "Créé %d chunks (DoclingNodeParser) pour la note %d",
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

            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info("✅ Fichier original supprimé: %s", file_path)
                else:
                    logger.warning(
                        "Fichier déjà supprimé ou introuvable: %s", file_path
                    )
            except Exception as e:
                logger.error(
                    "Erreur lors de la suppression du fichier %s: %s",
                    file_path,
                    e,
                    exc_info=True,
                )

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


def process_document_async(note_id: int, file_path: str):
    """
    Ajouter un document à la file d'attente pour traitement en arrière-plan.
    Cette fonction est non-bloquante et retourne immédiatement.
    """
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


def _ensure_document_workers():
    """S'assurer que les workers de traitement de documents sont démarrés."""
    global document_workers

    with _document_workers_lock:
        if not document_workers or not any(w.is_alive() for w in document_workers):
            document_workers = []
            num_workers = settings.MAX_CONCURRENT_DOCUMENTS
            for i in range(num_workers):
                worker = threading.Thread(
                    target=_process_document_worker, daemon=True
                )
                worker.start()
                document_workers.append(worker)
                logger.info("Worker de traitement de documents %d démarré", i + 1)
