"""
Tâches Celery : traitement documents bibliothèque/projet, embeddings, réindexation.
"""
import logging

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=60)
def process_library_document(
    self, document_id: int, file_path: str, run_id: str | None = None
) -> None:
    """Pipeline Mistral OCR → chunks → embeddings pour un document bibliothèque."""
    from app.library_document_logging import get_library_document_logger
    from app.services.document_service_new import _process_document_for_id

    get_library_document_logger().info(
        "[Celery] Tâche process_library_document démarrée document_id=%s fichier=%s task_id=%s",
        document_id,
        file_path,
        self.request.id,
    )
    logger.info(
        "Celery process_library_document document_id=%s file_path=%s task_id=%s",
        document_id,
        file_path,
        self.request.id,
    )
    try:
        _process_document_for_id(document_id, file_path, run_id)
    except Exception as exc:
        logger.exception(
            "process_library_document échec document_id=%s: %s", document_id, exc
        )
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=0)
def reindex_library_document_task(
    self, document_id: int, user_id: int, run_id: str | None = None
) -> dict:
    """Réindexation complète d'un document bibliothèque."""
    from app.library_document_logging import get_library_document_logger
    from app.services.document_service_new import reindex_library_document

    get_library_document_logger().info(
        "[Celery] Tâche reindex_library_document_task démarrée document_id=%s user_id=%s task_id=%s",
        document_id,
        user_id,
        self.request.id,
    )
    logger.info(
        "Celery reindex_library_document_task document_id=%s user_id=%s task_id=%s",
        document_id,
        user_id,
        self.request.id,
    )
    try:
        return reindex_library_document(document_id, user_id, run_id)
    except Exception:
        logger.exception(
            "reindex_library_document_task échec document_id=%s", document_id
        )
        raise


@celery_app.task(bind=True, max_retries=0)
def reindex_all_library_documents_task(self, user_id: int) -> dict:
    """Réindexation séquentielle de tous les documents fichier de la bibliothèque."""
    from app.library_document_logging import get_library_document_logger
    from app.services.document_service_new import reindex_all_library_documents

    get_library_document_logger().info(
        "[Celery] reindex_all_library_documents_task user_id=%s task_id=%s",
        user_id,
        self.request.id,
    )
    logger.info(
        "Celery reindex_all_library_documents_task user_id=%s task_id=%s",
        user_id,
        self.request.id,
    )
    return reindex_all_library_documents(user_id)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=60)
def process_document_embeddings(
    self, document_id: int, run_id: str | None = None
) -> None:
    """Embeddings feuilles pour un document bibliothèque puis statut completed."""
    from app.library_document_logging import get_library_document_logger
    from app.services.chunk_service import _process_embeddings_for_document

    get_library_document_logger().info(
        "[Celery] Tâche process_document_embeddings démarrée document_id=%s task_id=%s",
        document_id,
        self.request.id,
    )
    logger.info(
        "Celery process_document_embeddings document_id=%s task_id=%s",
        document_id,
        self.request.id,
    )
    try:
        _process_embeddings_for_document(document_id, run_id)
    except Exception as exc:
        logger.exception(
            "process_document_embeddings échec document_id=%s: %s", document_id, exc
        )
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=0)
def update_document_spaces_task(
    self,
    document_id: int,
    add_space_ids: list[int],
    remove_space_ids: list[int],
    user_id: int,
) -> dict:
    """Ajout/retrait d'un document à des espaces via worker."""
    from app.services.document_service_new import apply_document_spaces_update

    logger.info(
        "Celery update_document_spaces_task document_id=%s user_id=%s add=%s remove=%s task_id=%s",
        document_id,
        user_id,
        add_space_ids,
        remove_space_ids,
        self.request.id,
    )
    success = apply_document_spaces_update(
        document_id=document_id,
        add_space_ids=add_space_ids,
        remove_space_ids=remove_space_ids,
        user_id=user_id,
    )
    return {
        "status": "success" if success else "failed",
        "document_id": document_id,
        "add_space_ids": add_space_ids,
        "remove_space_ids": remove_space_ids,
    }
