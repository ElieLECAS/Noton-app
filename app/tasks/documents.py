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
    self,
    document_id: int,
    user_id: int,
    run_id: str | None = None,
    mode: str = "full",
) -> dict:
    """Réindexation d'un document bibliothèque (full / text_only / colpali_only)."""
    from app.library_document_logging import get_library_document_logger
    from app.services.document_service_new import reindex_library_document

    get_library_document_logger().info(
        "[Celery] reindex_library_document_task document_id=%s user_id=%s mode=%s task_id=%s",
        document_id,
        user_id,
        mode,
        self.request.id,
    )
    logger.info(
        "Celery reindex_library_document_task document_id=%s user_id=%s mode=%s task_id=%s",
        document_id,
        user_id,
        mode,
        self.request.id,
    )
    try:
        return reindex_library_document(document_id, user_id, run_id, mode=mode)
    except Exception:
        logger.exception(
            "reindex_library_document_task échec document_id=%s", document_id
        )
        raise


@celery_app.task(bind=True, max_retries=0)
def multimodal_reindex_library_document_task(
    self, document_id: int, user_id: int, run_id: str | None = None
) -> dict:
    """Retraitement multimodal additif (pymupdf + mistral-small, 1 chunk/page)."""
    from app.library_document_logging import get_library_document_logger
    from app.services.document_service_new import multimodal_reindex_library_document

    get_library_document_logger().info(
        "[Celery] multimodal_reindex_library_document_task document_id=%s user_id=%s task_id=%s",
        document_id,
        user_id,
        self.request.id,
    )
    logger.info(
        "Celery multimodal_reindex document_id=%s user_id=%s task_id=%s",
        document_id,
        user_id,
        self.request.id,
    )
    try:
        return multimodal_reindex_library_document(document_id, user_id, run_id)
    except Exception:
        logger.exception(
            "multimodal_reindex_library_document_task échec document_id=%s",
            document_id,
        )
        raise


@celery_app.task(bind=True, max_retries=0)
def reindex_all_library_documents_task(self, user_id: int, mode: str = "full") -> dict:
    """Tâche Celery pour enfiler individuellement chaque document éligible sous forme de tâche séparée."""
    from app.library_document_logging import get_library_document_logger
    from app.services.document_service_new import (
        mark_all_eligible_documents_reindex_queued,
        get_documents_by_library,
    )
    from app.services.library_service import get_or_create_user_library
    from app.services.task_dispatch import dispatch_reindex_library
    from app.database import engine
    from sqlmodel import Session
    from pathlib import Path

    ld = get_library_document_logger()
    ld.info(
        "[Celery] reindex_all_library_documents_task user_id=%s mode=%s task_id=%s",
        user_id,
        mode,
        self.request.id,
    )
    logger.info(
        "Celery reindex_all_library_documents_task user_id=%s mode=%s task_id=%s",
        user_id,
        mode,
        self.request.id,
    )

    marked = mark_all_eligible_documents_reindex_queued(user_id)
    ld.info("[Celery] reindex_all_library_documents_task — %s document(s) marqués en attente.", marked)

    if marked == 0:
        return {"ok": 0, "failed": [], "skipped": 0, "marked_queued": 0}

    with Session(engine) as session:
        library = get_or_create_user_library(session, user_id)
        docs = get_documents_by_library(session, library.id, user_id)

    results: dict = {"ok": 0, "failed": [], "skipped": 0, "marked_queued": marked}
    for doc in docs:
        if doc.document_type != "document" or not doc.source_file_path:
            results["skipped"] += 1
            continue
        src = Path(doc.source_file_path)
        if not src.is_file():
            results["skipped"] += 1
            continue

        try:
            dispatch_reindex_library(doc.id, user_id, mode=mode)
            results["ok"] += 1
        except Exception as e:
            logger.exception("reindex_all_task: échec enfilage document_id=%s: %s", doc.id, e)
            results["failed"].append(
                {
                    "document_id": doc.id,
                    "title": doc.title,
                    "error": str(e),
                }
            )

    ld.info(
        "[Celery] reindex_all_library_documents_task terminé. ok=%s skipped=%s failed=%s",
        results["ok"],
        results["skipped"],
        len(results["failed"]),
    )
    return results


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


@celery_app.task(bind=True, max_retries=0)
def generate_faq_from_feedback_task(self, feedback_id: int) -> None:
    """Génère un texte technique RAG à partir d'un feedback (négatif ou positif avec précision)."""
    from sqlmodel import Session, select, func
    from app.database import engine
    from app.models.message_feedback import MessageFeedback
    from app.models.space import Space
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk
    from app.models.document_space import DocumentSpace
    from app.services.library_service import get_or_create_user_library
    from app.services.embedding_service import generate_embedding
    from app.services.feedback_knowledge_service import (
        FEEDBACK_KNOWLEDGE_CONTENT_TYPE,
        generate_feedback_knowledge_content,
    )
    from app.services.document_service_new import FEEDBACK_CORRECTIVE_DOCUMENT_TYPE
    from app.config import settings

    logger.info(
        "Celery generate_faq_from_feedback_task feedback_id=%s task_id=%s",
        feedback_id,
        self.request.id,
    )

    with Session(engine) as session:
        feedback = session.get(MessageFeedback, feedback_id)
        if not feedback:
            logger.error("Feedback avec id %s non trouvé", feedback_id)
            return

        if not feedback.comment or not feedback.comment.strip():
            logger.warning(
                "Le feedback %s n'a pas de commentaire/précision, aucune génération",
                feedback_id,
            )
            return

        if feedback.auto_faq_generated:
            logger.info(
                "Texte technique déjà généré pour feedback %s, skip",
                feedback_id,
            )
            return

        space = session.get(Space, feedback.space_id)
        if not space:
            logger.error(
                "Espace %s non trouvé pour le feedback %s",
                feedback.space_id,
                feedback_id,
            )
            return

        try:
            logger.info("Appel Mistral pour génération du texte technique feedback...")
            generated_content = generate_feedback_knowledge_content(feedback)
            logger.info("Génération texte technique réussie. Association aux espaces...")

            parent_doc_ids = set()
            if feedback.chunk_ids:
                statement = select(DocumentChunk.document_id).where(
                    DocumentChunk.id.in_(feedback.chunk_ids)
                )
                parent_doc_ids = set(session.exec(statement).all())

            library = get_or_create_user_library(session, feedback.user_id)
            prefix = settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX
            targets: list[tuple[str, list[int]]] = []

            if not parent_doc_ids:
                targets.append((f"{prefix} - Espace {space.name}", [space.id]))
            else:
                for doc_id in parent_doc_ids:
                    orig_doc = session.get(Document, doc_id)
                    if not orig_doc:
                        continue
                    spaces_linked = session.exec(
                        select(DocumentSpace.space_id).where(
                            DocumentSpace.document_id == orig_doc.id
                        )
                    ).all()
                    space_ids = list(spaces_linked) if spaces_linked else [feedback.space_id]
                    targets.append((f"{prefix} - {orig_doc.title}", space_ids))

            if not targets:
                targets.append((f"{prefix} - Espace {space.name}", [space.id]))

            embedding = generate_embedding(generated_content)
            if not embedding:
                error_msg = (
                    f"Impossible de générer l'embedding pour le feedback {feedback_id}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            chunk_metadata = {
                "content_type": FEEDBACK_KNOWLEDGE_CONTENT_TYPE,
                "feedback_id": feedback_id,
                "is_positive": feedback.is_positive,
            }

            for doc_title, space_ids in targets:
                truncated_title = doc_title[:200]

                document = session.exec(
                    select(Document).where(
                        Document.title == truncated_title,
                        Document.library_id == library.id,
                    )
                ).first()

                if not document:
                    document = Document(
                        title=truncated_title,
                        content="",
                        document_type=FEEDBACK_CORRECTIVE_DOCUMENT_TYPE,
                        processing_status="completed",
                        processing_progress=100,
                        library_id=library.id,
                        user_id=feedback.user_id,
                    )
                    session.add(document)
                    session.commit()
                    session.refresh(document)
                    logger.info("Nouveau document connaissance technique créé : %s", truncated_title)
                elif document.document_type != FEEDBACK_CORRECTIVE_DOCUMENT_TYPE:
                    document.document_type = FEEDBACK_CORRECTIVE_DOCUMENT_TYPE
                    session.add(document)
                    session.commit()

                for sp_id in space_ids:
                    doc_space = session.exec(
                        select(DocumentSpace).where(
                            DocumentSpace.document_id == document.id,
                            DocumentSpace.space_id == sp_id,
                        )
                    ).first()
                    if not doc_space:
                        session.add(
                            DocumentSpace(
                                document_id=document.id,
                                space_id=sp_id,
                                user_id=feedback.user_id,
                            )
                        )
                        session.commit()
                        logger.info(
                            "Document '%s' lié à l'espace %s",
                            truncated_title,
                            sp_id,
                        )

                chunk_count = (
                    session.exec(
                        select(func.count(DocumentChunk.id)).where(
                            DocumentChunk.document_id == document.id
                        )
                    ).first()
                    or 0
                )

                chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk_count,
                    content=generated_content,
                    text=generated_content,
                    embedding=embedding,
                    is_leaf=True,
                    hierarchy_level=0,
                    source=truncated_title,
                    start_char=0,
                    end_char=len(generated_content),
                    metadata_json=chunk_metadata,
                    metadata_=chunk_metadata,
                )
                session.add(chunk)
                session.commit()
                logger.info(
                    "Chunk connaissance technique inséré sous '%s' (index: %s)",
                    truncated_title,
                    chunk_count,
                )

            feedback.auto_faq_generated = True
            feedback.auto_faq_content = generated_content
            session.add(feedback)
            session.commit()
            logger.info("Feedback %s mis à jour avec le texte technique.", feedback_id)

        except Exception as exc:
            logger.exception(
                "Erreur génération texte technique feedback_id=%s: %s",
                feedback_id,
                exc,
            )
            raise
