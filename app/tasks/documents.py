"""
Tâches Celery : traitement documents bibliothèque/projet, embeddings, réindexation.
"""
import logging

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=60)
def process_library_document(self, document_id: int, file_path: str) -> None:
    """Pipeline Docling → chunks → embeddings/KAG pour un document bibliothèque."""
    from app.services.document_service_new import _process_document_for_id

    logger.info(
        "Celery process_library_document document_id=%s file_path=%s task_id=%s",
        document_id,
        file_path,
        self.request.id,
    )
    try:
        _process_document_for_id(document_id, file_path)
    except Exception as exc:
        logger.exception(
            "process_library_document échec document_id=%s: %s", document_id, exc
        )
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=60)
def process_project_document(self, note_id: int, file_path: str) -> None:
    """Pipeline Docling → chunks → file d'embeddings pour une note document projet."""
    from app.services.document_service import _process_document_for_note

    logger.info(
        "Celery process_project_document note_id=%s file_path=%s task_id=%s",
        note_id,
        file_path,
        self.request.id,
    )
    try:
        _process_document_for_note(note_id, file_path, sync_embeddings=True)
    except Exception as exc:
        logger.exception("process_project_document échec note_id=%s: %s", note_id, exc)
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=0)
def reindex_library_document_task(self, document_id: int, user_id: int) -> dict:
    """Réindexation complète d'un document bibliothèque."""
    from app.services.document_service_new import reindex_library_document

    logger.info(
        "Celery reindex_library_document_task document_id=%s user_id=%s task_id=%s",
        document_id,
        user_id,
        self.request.id,
    )
    try:
        return reindex_library_document(document_id, user_id)
    except Exception:
        logger.exception(
            "reindex_library_document_task échec document_id=%s", document_id
        )
        raise


@celery_app.task(bind=True, max_retries=1, default_retry_delay=60)
def process_note_embeddings(self, note_id: int, project_id: int) -> None:
    """Embeddings + KAG pour les chunks d'une note projet."""
    from app.services.chunk_service import _process_embeddings_for_note

    logger.info(
        "Celery process_note_embeddings note_id=%s project_id=%s task_id=%s",
        note_id,
        project_id,
        self.request.id,
    )
    try:
        _process_embeddings_for_note(note_id, project_id)
    except Exception as exc:
        logger.exception(
            "process_note_embeddings échec note_id=%s: %s", note_id, exc
        )
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=60)
def process_document_embeddings(self, document_id: int) -> None:
    """Embeddings + KAG pour un document bibliothèque (post-chunking)."""
    from app.services.chunk_service import _process_embeddings_for_document

    logger.info(
        "Celery process_document_embeddings document_id=%s task_id=%s",
        document_id,
        self.request.id,
    )
    try:
        _process_embeddings_for_document(document_id)
    except Exception as exc:
        logger.exception(
            "process_document_embeddings échec document_id=%s: %s", document_id, exc
        )
        raise self.retry(exc=exc)
