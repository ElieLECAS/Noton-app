"""
Dispatch des jobs background : Celery (Redis) ou threads locaux (fallback).
Modes : thread | celery | hybrid (Celery prioritaire, repli threads si échec connexion broker).
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from sqlmodel import Session

from app.config import settings
from app.database import engine
from app.models.document import Document
from app.models.note import Note

logger = logging.getLogger(__name__)

TaskBackendMode = Literal["thread", "celery", "hybrid"]


def get_task_backend_mode() -> TaskBackendMode:
    raw = (getattr(settings, "TASK_BACKEND_MODE", None) or "thread").strip().lower()
    if raw in ("celery", "hybrid", "thread"):
        return raw  # type: ignore[return-value]
    logger.warning("TASK_BACKEND_MODE invalide %r, fallback thread", raw)
    return "thread"


def _use_celery_first() -> bool:
    return get_task_backend_mode() in ("celery", "hybrid")


def _celery_only_failure_message() -> str:
    return "Celery indisponible alors que TASK_BACKEND_MODE=celery"


def _send_library_document(document_id: int, file_path: str) -> bool:
    from app.tasks.documents import process_library_document

    res = process_library_document.apply_async(
        args=[document_id, file_path],
        queue="documents",
    )
    logger.info(
        "task_dispatch library_document document_id=%s celery_task_id=%s",
        document_id,
        res.id,
    )
    return True


def _send_project_document(note_id: int, file_path: str) -> bool:
    from app.tasks.documents import process_project_document

    res = process_project_document.apply_async(
        args=[note_id, file_path],
        queue="documents",
    )
    logger.info(
        "task_dispatch project_document note_id=%s celery_task_id=%s",
        note_id,
        res.id,
    )
    return True


def _send_reindex_library(document_id: int, user_id: int) -> str:
    from app.tasks.documents import reindex_library_document_task

    async_result = reindex_library_document_task.apply_async(
        args=[document_id, user_id],
        queue="documents",
    )
    logger.info(
        "task_dispatch reindex_library document_id=%s celery_task_id=%s",
        document_id,
        async_result.id,
    )
    return async_result.id


def _send_note_embeddings(note_id: int, project_id: int) -> bool:
    from app.tasks.documents import process_note_embeddings

    res = process_note_embeddings.apply_async(
        args=[note_id, project_id],
        queue="embeddings",
    )
    logger.info(
        "task_dispatch note_embeddings note_id=%s project_id=%s celery_task_id=%s",
        note_id,
        project_id,
        res.id,
    )
    return True


def _send_document_embeddings(document_id: int) -> bool:
    from app.tasks.documents import process_document_embeddings

    res = process_document_embeddings.apply_async(
        args=[document_id],
        queue="embeddings",
    )
    logger.info(
        "task_dispatch document_embeddings document_id=%s celery_task_id=%s",
        document_id,
        res.id,
    )
    return True


def dispatch_library_document(document_id: int, file_path: str) -> None:
    """Enqueue traitement document bibliothèque (Celery ou thread)."""
    mode = get_task_backend_mode()
    if mode == "thread":
        from app.services.document_service_new import enqueue_library_document_thread

        enqueue_library_document_thread(document_id, file_path)
        return

    try:
        _send_library_document(document_id, file_path)
    except Exception as exc:
        logger.warning(
            "Celery indisponible pour library document_id=%s: %s", document_id, exc
        )
        if mode == "hybrid":
            from app.services.document_service_new import enqueue_library_document_thread

            enqueue_library_document_thread(document_id, file_path)
            return
        raise RuntimeError(_celery_only_failure_message()) from exc


def dispatch_project_document(note_id: int, file_path: str) -> None:
    """Enqueue traitement document projet / note document."""
    mode = get_task_backend_mode()
    if mode == "thread":
        from app.services.document_service import enqueue_project_document_thread

        enqueue_project_document_thread(note_id, file_path)
        return

    try:
        _send_project_document(note_id, file_path)
    except Exception as exc:
        logger.warning("Celery indisponible pour note_id=%s: %s", note_id, exc)
        if mode == "hybrid":
            from app.services.document_service import enqueue_project_document_thread

            enqueue_project_document_thread(note_id, file_path)
            return
        raise RuntimeError(_celery_only_failure_message()) from exc


def dispatch_reindex_library(document_id: int, user_id: int):
    """
    Réindexation : Celery si activé, sinon exécution synchrone.
    Retourne str (celery_task_id) si file async, ou dict résultat si synchrone.
    """
    mode = get_task_backend_mode()
    if mode == "thread":
        from app.services.document_service_new import reindex_library_document

        return reindex_library_document(document_id, user_id)

    try:
        return _send_reindex_library(document_id, user_id)
    except Exception as exc:
        logger.warning(
            "Celery indisponible pour reindex document_id=%s: %s", document_id, exc
        )
        if mode == "hybrid":
            from app.services.document_service_new import reindex_library_document

            return reindex_library_document(document_id, user_id)
        raise RuntimeError(_celery_only_failure_message()) from exc


def try_dispatch_embeddings_job(note_id: int, project_id: int) -> bool:
    """
    Si Celery (ou hybrid avec broker OK), envoie la bonne tâche embeddings.
    Retourne True si délégué à Celery ; False si l'appelant doit utiliser les threads/sync.
    """
    if not _use_celery_first():
        return False

    try:
        with Session(engine) as session:
            if session.get(Note, note_id) is not None:
                _send_note_embeddings(note_id, project_id)
                return True
            doc = session.get(Document, note_id)
            if doc is not None:
                _send_document_embeddings(doc.id)
                return True
    except Exception as exc:
        logger.warning(
            "Celery indisponible pour embeddings note_id=%s: %s", note_id, exc
        )
        if get_task_backend_mode() == "hybrid":
            return False
        raise RuntimeError(_celery_only_failure_message()) from exc

    logger.warning(
        "try_dispatch_embeddings_job: ni Note ni Document pour id=%s", note_id
    )
    return False


def should_start_thread_workers() -> bool:
    """Démarrer les workers threads embedding + documents sur le process web."""
    return get_task_backend_mode() in ("thread", "hybrid")
