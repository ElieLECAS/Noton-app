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


@celery_app.task(bind=True, max_retries=0)
def generate_faq_from_feedback_task(self, feedback_id: int) -> None:
    """Génère automatiquement une FAQ corrective RAG à partir d'un feedback négatif."""
    import asyncio
    from sqlmodel import Session, select, func
    from app.database import engine
    from app.models.message_feedback import MessageFeedback
    from app.models.space import Space
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk
    from app.models.document_space import DocumentSpace
    from app.services.library_service import get_or_create_user_library
    from app.services.mistral_service import chat as mistral_chat
    from app.services.embedding_service import generate_embedding
    from app.config import settings

    logger.info("Celery generate_faq_from_feedback_task feedback_id=%s task_id=%s", feedback_id, self.request.id)

    with Session(engine) as session:
        feedback = session.get(MessageFeedback, feedback_id)
        if not feedback:
            logger.error("Feedback avec id %s non trouvé", feedback_id)
            return
        
        # Idempotence : skip si FAQ déjà générée (sauf force_regenerate)
        if feedback.auto_faq_generated:
            logger.info("FAQ déjà générée pour feedback %s, skip génération", feedback_id)
            return

        if feedback.is_positive:
            logger.warning("Le feedback %s est positif, aucune FAQ à générer", feedback_id)
            return

        if not feedback.comment or not feedback.comment.strip():
            logger.warning("Le feedback %s n'a pas de commentaire/correction, aucune FAQ à générer", feedback_id)
            return

        space = session.get(Space, feedback.space_id)
        if not space:
            logger.error("Espace %s non trouvé pour le feedback %s", feedback.space_id, feedback_id)
            return

        # 1. Construction du prompt
        prompt = f"""Tu es un assistant expert chargé d'éditer et de corriger des couples de Question/Réponse pour enrichir une base de connaissances (FAQ).
Un utilisateur a signalé qu'une réponse fournie par l'assistant RAG était incorrecte et a fourni un commentaire correctif.

Voici les informations :
- Question originale de l'utilisateur : {feedback.query_text}
- Réponse incorrecte fournie par l'assistant : {feedback.response_text}
- Commentaire correctif de l'utilisateur : {feedback.comment}

Génère un couple Question/Réponse corrigé et structuré.
La Question doit être claire, directe et correspondre au besoin de l'utilisateur.
La Réponse doit être correcte, précise, concise, et intégrer les corrections fournies par l'utilisateur tout en restant professionnelle. Elle doit corriger l'erreur de la réponse précédente.

Format de sortie attendu :
Question: [Insérer la question ici]
Réponse: [Insérer la réponse corrigée ici]

Ne réponds rien d'autre que ce format. Pas d'introduction, pas de conclusion, pas de blabla."""

        try:
            logger.info("Appel Mistral pour génération de la FAQ corrective...")
            response = asyncio.run(
                mistral_chat(
                    prompt,
                    settings.MODEL_FAST,
                    [{"role": "user", "content": prompt}]
                )
            )

            generated_content = ""
            if "choices" in response and len(response["choices"]) > 0:
                generated_content = response["choices"][0]["message"].get("content", "").strip()

            if not generated_content:
                logger.error("Mistral n'a généré aucun contenu pour le feedback %s", feedback_id)
                return

            logger.info("Génération FAQ réussie. Association aux documents/espaces...")

            # 2. Identification des documents et des espaces cibles
            parent_doc_ids = set()
            if feedback.chunk_ids:
                statement = select(DocumentChunk.document_id).where(DocumentChunk.id.in_(feedback.chunk_ids))
                parent_doc_ids = set(session.exec(statement).all())

            # Bibliothèque générale commune
            library = get_or_create_user_library(session, feedback.user_id)

            # Liste de tuples (doc_title, list[space_ids])
            targets = []

            if not parent_doc_ids:
                # Fallback sur l'espace courant si pas de chunks d'origine
                targets.append((f"{settings.FAQ_CORRECTIVE_TITLE_PREFIX} - Espace {space.name}", [space.id]))
            else:
                for doc_id in parent_doc_ids:
                    orig_doc = session.get(Document, doc_id)
                    if not orig_doc:
                        continue
                    # Récupérer tous les espaces du document d'origine
                    spaces_linked = session.exec(
                        select(DocumentSpace.space_id).where(DocumentSpace.document_id == orig_doc.id)
                    ).all()
                    space_ids = list(spaces_linked) if spaces_linked else [feedback.space_id]
                    targets.append((f"{settings.FAQ_CORRECTIVE_TITLE_PREFIX} - {orig_doc.title}", space_ids))

            if not targets:
                targets.append((f"{settings.FAQ_CORRECTIVE_TITLE_PREFIX} - Espace {space.name}", [space.id]))

            # 3. Génération de l'embedding du couple Q&A (obligatoire)
            embedding = generate_embedding(generated_content)
            if not embedding:
                error_msg = f"Impossible de générer l'embedding pour la FAQ du feedback {feedback_id}"
                logger.error(error_msg)
                raise ValueError(error_msg)  # Force retry Celery

            # 4. Création des documents virtuels et des chunks FAQ
            for doc_title, space_ids in targets:
                # Limiter le titre à 200 caractères max pour respecter les contraintes SQLModel
                truncated_title = doc_title[:200]
                
                # Récupérer ou créer le document virtuel
                document = session.exec(
                    select(Document).where(
                        Document.title == truncated_title,
                        Document.library_id == library.id
                    )
                ).first()

                if not document:
                    document = Document(
                        title=truncated_title,
                        content="",
                        document_type="written",
                        processing_status="completed",
                        processing_progress=100,
                        library_id=library.id,
                        user_id=feedback.user_id
                    )
                    session.add(document)
                    session.commit()
                    session.refresh(document)
                    logger.info("Nouveau document FAQ créé : %s", truncated_title)

                # Lier le document virtuel à tous les espaces cibles
                for sp_id in space_ids:
                    doc_space = session.exec(
                        select(DocumentSpace).where(
                            DocumentSpace.document_id == document.id,
                            DocumentSpace.space_id == sp_id
                        )
                    ).first()

                    if not doc_space:
                        doc_space = DocumentSpace(
                            document_id=document.id,
                            space_id=sp_id,
                            user_id=feedback.user_id
                        )
                        session.add(doc_space)
                        session.commit()
                        logger.info("Document FAQ '%s' lié à l'espace %s", truncated_title, sp_id)

                # Calculer le chunk_index
                chunk_count = session.exec(
                    select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == document.id)
                ).first() or 0

                # Insérer le chunk FAQ
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
                    end_char=len(generated_content)
                )
                session.add(chunk)
                session.commit()
                logger.info("Chunk FAQ inséré sous '%s' (index: %s)", truncated_title, chunk_count)

            # 5. Mettre à jour le feedback original avec le statut et contenu
            feedback.auto_faq_generated = True
            feedback.auto_faq_content = generated_content
            session.add(feedback)
            session.commit()
            logger.info("Feedback %s mis à jour avec le contenu FAQ.", feedback_id)

        except Exception as exc:
            logger.exception("Erreur lors de la génération automatique de la FAQ pour feedback_id=%s: %s", feedback_id, exc)
            raise
