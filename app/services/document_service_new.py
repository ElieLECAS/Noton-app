from typing import Any, Optional, List
import logging
import os
from pathlib import Path
import threading
import time
from queue import Queue, Empty
from sqlmodel import Session, select
from app.config import settings
from app.database import engine
from app.library_document_logging import get_library_document_logger
from app.models.document import Document, DocumentCreate, DocumentUpdate
from app.models.library import Library
from app.models.document_space import DocumentSpace
from app.models.document_chunk import DocumentChunk
from app.tracing import trace_run, trace_pipeline
from datetime import datetime

logger = logging.getLogger(__name__)

FEEDBACK_CORRECTIVE_DOCUMENT_TYPE = "feedback_corrective"
LEGACY_FEEDBACK_TITLE_PREFIX = "FAQ Corrective"


def is_feedback_corrective_document(doc: Document) -> bool:
    """Document généré par feedback (masqué de la bibliothèque générale)."""
    title = doc.title or ""
    return (
        doc.document_type == FEEDBACK_CORRECTIVE_DOCUMENT_TYPE
        or title.startswith(settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX)
        or title.startswith(LEGACY_FEEDBACK_TITLE_PREFIX)
    )


def feedback_corrective_sql_filter(document_filter: str, table_alias: str = "d") -> str:
    """
    Clause SQL brute pour filtrer les documents correctifs (retrieval RAG).
    """
    prefix = settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX.replace("'", "''")
    legacy = LEGACY_FEEDBACK_TITLE_PREFIX.replace("'", "''")
    t = table_alias
    include = (
        f"({t}.document_type = '{FEEDBACK_CORRECTIVE_DOCUMENT_TYPE}' "
        f"OR {t}.title LIKE '{prefix}%' "
        f"OR {t}.title LIKE '{legacy}%')"
    )
    if document_filter == "faq_corrective":
        return f"AND {include}"
    # Par défaut, on exclut systématiquement les correctifs/feedbacks de toutes les recherches
    return f"AND NOT {include}"


def _exclude_feedback_corrective_where():
    """Conditions SQLModel pour exclure les docs correctifs des listes bibliothèque."""
    prefix = settings.FEEDBACK_KNOWLEDGE_TITLE_PREFIX
    return (
        Document.document_type != FEEDBACK_CORRECTIVE_DOCUMENT_TYPE,
        ~Document.title.like(f"{prefix}%"),
        ~Document.title.like(f"{LEGACY_FEEDBACK_TITLE_PREFIX}%"),
    )


# En file d’attente de réindexation : chunks et embeddings actuels restent servis jusqu’au début effectif du worker.
DOCUMENT_STATUS_REINDEX_QUEUED = "reindex_queued"
DOCUMENT_STATUS_MULTIMODAL_QUEUED = "multimodal_queued"
# Arrêt explicite (sans suppression) ou file ignorée.
DOCUMENT_STATUS_CANCELLED_BY_USER = "cancelled_by_user"
DOCUMENT_STATUS_SKIPPED = "skipped"
DOCUMENT_STATUS_PARTIAL_KAG_DONE = "partial_kag_done"
DOCUMENT_STATUS_FAILED_RETRY_EXHAUSTED = "failed_retry_exhausted"
# Compat lecture anciennes données / code tiers
DOCUMENT_STATUS_CANCELLED = DOCUMENT_STATUS_CANCELLED_BY_USER

LIBRARY_USER_STOPPED_STATUSES = frozenset(
    {
        DOCUMENT_STATUS_CANCELLED_BY_USER,
        "cancelled",  # legacy avant migration
        DOCUMENT_STATUS_SKIPPED,
    }
)

# Documents susceptibles d’occuper la file de traitement ou d’être en attente.
LIBRARY_QUEUE_ACTIVE_STATUSES = frozenset(
    {
        "pending",
        "processing",
        DOCUMENT_STATUS_REINDEX_QUEUED,
        DOCUMENT_STATUS_MULTIMODAL_QUEUED,
    }
)

# Une seule file globale : ordre FIFO strict, un document terminé entièrement avant le suivant.
document_task_queue: Queue = Queue()
_document_queue_lock = threading.Lock()
document_workers = []
_document_workers_lock = threading.Lock()
_cancelled_document_ids: set[int] = set()
_cancelled_documents_lock = threading.Lock()

# Liste des marques connues pour l'inférence de source
KNOWN_BRANDS = [
    "Proferm",
    "Technal",
    "Askey",
    "Profine",
    "Roto",
    "Somfy",
    "Maco",
    "Gu",
    "VBH",
    "KBE",
]


def infer_document_source(
    file_path: Optional[str] = None, content: Optional[str] = None
) -> str:
    """
    Infère la source (marque/origine) d'un document.
    1. Regarde d'abord le chemin du fichier (signal fort).
    2. Regarde ensuite le contenu sémantique pour des mots-clés.
    """
    # 1. Analyse du chemin
    if file_path:
        path_str = file_path.lower()
        for brand in KNOWN_BRANDS:
            if brand.lower() in path_str:
                return brand

    # 2. Analyse du contenu (premiers 5000 caractères)
    if content:
        content_sample = content[:5000].lower()
        # Priorité Proferm si présent dans le texte
        if "proferm" in content_sample:
            return "Proferm"
        # Autres marques
        for brand in KNOWN_BRANDS:
            if brand.lower() in content_sample:
                return brand

    return "Inconnu"


def _mark_document_processing_cancelled(document_id: int) -> None:
    """Marque un document comme annulé pour stopper son traitement asynchrone."""
    with _cancelled_documents_lock:
        _cancelled_document_ids.add(document_id)


def _is_document_processing_cancelled(document_id: int) -> bool:
    with _cancelled_documents_lock:
        return document_id in _cancelled_document_ids


def _clear_document_processing_cancelled(document_id: int) -> None:
    with _cancelled_documents_lock:
        _cancelled_document_ids.discard(document_id)


def _should_abort_processing(document_id: int) -> bool:
    """
    Indique si le traitement doit être interrompu:
    - drapeau d’annulation (stop/skip/suppression)
    - statut final cancelled/skipped en base
    - document supprimé en base
    """
    if _is_document_processing_cancelled(document_id):
        return True

    try:
        with Session(engine) as session:
            doc = session.get(Document, document_id)
            if doc is None:
                return True
            if doc.processing_status in LIBRARY_USER_STOPPED_STATUSES:
                return True
    except Exception:
        # En cas d'erreur transitoire DB, on n'interrompt pas par défaut.
        return False
    return False


def _finalize_pipeline_abort(document_id: int) -> None:
    """
    Après sortie anticipée du pipeline : évite de laisser le document en « processing »
    si l’arrêt n’a pas encore été persisté (ex. annulation par drapeau uniquement).
    """
    try:
        with Session(engine) as session:
            document = session.get(Document, document_id)
            if document is None:
                _clear_document_processing_cancelled(document_id)
                return
            if document.processing_status in LIBRARY_USER_STOPPED_STATUSES:
                _clear_document_processing_cancelled(document_id)
                return
            if _is_document_processing_cancelled(document_id):
                document.processing_status = DOCUMENT_STATUS_CANCELLED_BY_USER
                document.processing_progress = 0
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
            _clear_document_processing_cancelled(document_id)
    except Exception:
        logger.debug(
            "finalize_pipeline_abort document_id=%s (ignoré)", document_id, exc_info=True
        )


def stop_library_document_processing(
    session: Session, document_id: int, user_id: int
) -> tuple[Optional[Document], dict[str, Any]]:
    """
    Arrête le traitement pour un document (sans suppression) : cancelled_by_user + invalidation run + revoke Celery.
    """
    from app.services.document_run import refresh_document_processing_run_id

    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return None, {}
    if document.processing_status not in LIBRARY_QUEUE_ACTIVE_STATUSES:
        return None, {}
    document.processing_status = DOCUMENT_STATUS_CANCELLED_BY_USER
    document.processing_progress = 0
    refresh_document_processing_run_id(document)
    document.updated_at = datetime.utcnow()
    session.add(document)
    session.commit()
    session.refresh(document)
    _mark_document_processing_cancelled(document_id)
    revoke_info: dict[str, Any] = {"revoked_count": 0, "revoked_task_ids": []}
    try:
        from app.services.task_dispatch import revoke_library_document_tasks

        revoke_info = revoke_library_document_tasks(document_id)
    except Exception:
        logger.debug(
            "Révocation Celery ignorée pour document_id=%s", document_id, exc_info=True
        )
    logger.info(
        "Traitement bibliothèque arrêté (cancelled_by_user) pour document_id=%s",
        document_id,
    )
    return document, revoke_info


def skip_library_document_processing(
    session: Session, document_id: int, user_id: int
) -> tuple[Optional[Document], dict[str, Any]]:
    """
    Ignore un document en attente (skipped) ; si déjà en cours, équivalent à un stop (cancelled_by_user).
    """
    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return None, {}
    st = document.processing_status
    if st == "processing":
        return stop_library_document_processing(session, document_id, user_id)
    if st in ("pending", DOCUMENT_STATUS_REINDEX_QUEUED):
        from app.services.document_run import refresh_document_processing_run_id

        document.processing_status = DOCUMENT_STATUS_SKIPPED
        document.processing_progress = 0
        refresh_document_processing_run_id(document)
        document.updated_at = datetime.utcnow()
        session.add(document)
        session.commit()
        session.refresh(document)
        _mark_document_processing_cancelled(document_id)
        revoke_info: dict[str, Any] = {"revoked_count": 0, "revoked_task_ids": []}
        try:
            from app.services.task_dispatch import revoke_library_document_tasks

            revoke_info = revoke_library_document_tasks(document_id)
        except Exception:
            logger.debug(
                "Révocation Celery ignorée pour document_id=%s", document_id, exc_info=True
            )
        logger.info(
            "Document bibliothèque ignoré (skipped) pour document_id=%s", document_id
        )
        return document, revoke_info
    return None, {}


def stop_all_library_documents_processing(session: Session, user_id: int) -> dict:
    """Annule tous les documents encore en file (pending / processing / reindex_queued / multimodal_queued)."""
    from app.services.library_service import get_or_create_user_library
    from app.services.document_run import refresh_document_processing_run_id

    library = get_or_create_user_library(session, user_id)
    # On révoque sur TOUTE la bibliothèque (même si statut local non actif),
    # pour s'assurer qu'aucune tâche zombie ne reste en queue.
    all_library_docs = list(
        session.exec(
            select(Document).where(
                Document.library_id == library.id,
            )
        ).all()
    )
    docs = list(
        session.exec(
            select(Document).where(
                Document.library_id == library.id,
                Document.processing_status.in_(LIBRARY_QUEUE_ACTIVE_STATUSES),
            )
        ).all()
    )
    n = 0
    all_revoked: list[str] = []
    doc_ids_all = {int(d.id) for d in all_library_docs if d.id is not None}
    try:
        from app.services.task_dispatch import revoke_library_tasks_bulk

        bulk_info = revoke_library_tasks_bulk(doc_ids_all, user_id=user_id)
        all_revoked.extend(bulk_info.get("revoked_task_ids") or [])
    except Exception:
        logger.debug("Révocation bulk stop-all ignorée", exc_info=True)
    for doc in docs:
        doc.processing_status = DOCUMENT_STATUS_CANCELLED_BY_USER
        doc.processing_progress = 0
        refresh_document_processing_run_id(doc)
        doc.updated_at = datetime.utcnow()
        session.add(doc)
        _mark_document_processing_cancelled(doc.id)
        n += 1
    if n:
        session.commit()
    return {
        "cancelled": n,
        "revoked_count": len(all_revoked),
        "revoked_task_ids": all_revoked,
    }


def skip_all_library_documents_processing(session: Session, user_id: int) -> dict:
    """
    Ignore les documents en attente (skipped) et arrête celui en cours (cancelled_by_user).
    """
    from app.services.library_service import get_or_create_user_library
    from app.services.document_run import refresh_document_processing_run_id

    library = get_or_create_user_library(session, user_id)
    docs = list(
        session.exec(
            select(Document).where(
                Document.library_id == library.id,
                Document.processing_status.in_(LIBRARY_QUEUE_ACTIVE_STATUSES),
            )
        ).all()
    )
    skipped = 0
    cancelled = 0
    all_revoked: list[str] = []
    for doc in docs:
        if doc.processing_status == "processing":
            doc.processing_status = DOCUMENT_STATUS_CANCELLED_BY_USER
            cancelled += 1
        elif doc.processing_status in ("pending", DOCUMENT_STATUS_REINDEX_QUEUED):
            doc.processing_status = DOCUMENT_STATUS_SKIPPED
            skipped += 1
        else:
            continue
        doc.processing_progress = 0
        refresh_document_processing_run_id(doc)
        doc.updated_at = datetime.utcnow()
        session.add(doc)
        _mark_document_processing_cancelled(doc.id)
        try:
            from app.services.task_dispatch import revoke_library_document_tasks

            info = revoke_library_document_tasks(doc.id)
            all_revoked.extend(info.get("revoked_task_ids") or [])
        except Exception:
            logger.debug(
                "Révocation Celery ignorée pour document_id=%s", doc.id, exc_info=True
            )
    if skipped or cancelled:
        session.commit()
    return {
        "skipped": skipped,
        "cancelled_running": cancelled,
        "revoked_count": len(all_revoked),
        "revoked_task_ids": all_revoked,
    }

def process_document_file(file_path: str) -> Optional[str]:
    """
    DEPRECATED: Cette fonction est conservée temporairement pour compatibilité 
    mais n'est plus utilisée par le pipeline principal qui a migré vers le multimodal.
    
    Extrait le markdown via Mistral OCR ou extraction native.
    """
    from app.services.mistral_ocr_service import extract_markdown_from_file, ExtractedMarkdown

    if not os.path.exists(file_path):
        logger.error("Fichier non trouvé: %s", file_path)
        return None
    try:
        markdown = extract_markdown_from_file(file_path)
        method = getattr(markdown, "method", "ocr")
        if not markdown or not markdown.strip():
            label = "Mistral OCR" if method == "ocr" else "extraction"
            logger.warning("Markdown vide après %s: %s", label, file_path)
            return None
        return ExtractedMarkdown(markdown.strip(), method)
    except Exception as e:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".epub":
            fallback_markdown = extract_text_from_epub(file_path)
            if fallback_markdown:
                return ExtractedMarkdown(fallback_markdown.strip(), "text")
        logger.error("Erreur d'extraction pour %s: %s", file_path, e, exc_info=True)
        return None


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


def save_uploaded_file(file_content: bytes, filename: str, upload_dir: str = "media/documents") -> Optional[str]:
    """Sauvegarde un fichier uploadé sur le disque."""
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
        logger.error("Erreur lors de la sauvegarde du fichier %s: %s", filename, e, exc_info=True)
        return None


def reindex_library_document(
    document_id: int, user_id: int, run_id: Optional[str] = None
) -> dict:
    """
    Retraite un document en utilisant le pipeline multimodal unifié.
    
    Cette fonction remplace l'ancien pipeline PyMuPDF4LLM + MistralOCR par le pipeline
    multimodal de meilleure qualité. Elle réutilise le fichier déjà stocké en base.
    
    Utilise ``document.source_file_path`` (fichier déjà stocké sous media/documents).
    """
    from app.services.document_run import is_processing_run_current

    ld = get_library_document_logger()
    ld.info(
        "[Réindex] Démarrage document_id=%s user_id=%s — pipeline : multimodal page par page → chunks → embeddings.",
        document_id,
        user_id,
    )
    
    if run_id is not None and not is_processing_run_current(document_id, run_id):
        ld.info(
            "[Réindex] document_id=%s — abandon : run_id obsolète (stop utilisateur).",
            document_id,
        )
        return {
            "document_id": document_id,
            "status": "aborted",
            "reason": "stale_run",
        }

    # Récupérer le source_file_path depuis la DB
    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            raise ValueError("Document introuvable ou accès refusé")
        library = session.get(Library, document.library_id)
        if not library:
            raise ValueError("Document introuvable ou accès refusé")
        # Bibliothèque globale : tout document listé côté API est réindexable par un
        # utilisateur autorisé (library.write) ; user_id du document = auteur upload.
        if library.is_global:
            pass
        elif library.user_id == user_id or document.user_id == user_id:
            pass
        else:
            raise ValueError("Document introuvable ou accès refusé")
        if not document.source_file_path:
            raise ValueError("Aucun fichier source enregistré pour ce document")
        src = Path(document.source_file_path)
        if not src.is_file():
            raise ValueError("Fichier source introuvable sur le disque")
        file_path = str(src)
        ld.info(
            "[Réindex] document_id=%s — fichier source : %s",
            document_id,
            src,
        )
    
    # Appeler le pipeline multimodal unifié en mode "complet" (supprime tous les chunks)
    result = process_document_multimodal(
        document_id=document_id,
        file_path=file_path,
        user_id=user_id,
        run_id=run_id,
        delete_existing_chunks=True,  # Retraitement complet: on supprime tous les chunks
    )
    
    chunk_count = result.get("chunks", 0)
    logger.info(
        "reindex_library_document terminé document_id=%s chunks=%s",
        document_id,
        chunk_count,
    )
    ld.info(
        "[Réindex] document_id=%s — FIN OK chunks=%s.",
        document_id,
        chunk_count,
    )
    
    return {
        "document_id": document_id,
        "chunks": chunk_count,
        "status": result.get("status", "completed"),
    }


def mark_document_reindex_queued(session: Session, document_id: int, user_id: int) -> bool:
    """Marque un document en attente de réindexation sans toucher aux chunks."""
    from app.services.library_service import get_or_create_user_library

    library = get_or_create_user_library(session, user_id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if not document:
        return False
    document.processing_status = DOCUMENT_STATUS_REINDEX_QUEUED
    document.processing_progress = 0
    document.updated_at = datetime.utcnow()
    session.add(document)
    session.commit()
    return True


def mark_document_multimodal_queued(session: Session, document_id: int, user_id: int) -> bool:
    """
    DEPRECATED: Utilisé uniquement pour le retraitement multimodal séparé.
    Le pipeline unifié utilise maintenant mark_document_reindex_queued.
    
    Marque un document en attente de retraitement multimodal (chunks OCR inchangés).
    """
    from app.services.library_service import get_or_create_user_library

    library = get_or_create_user_library(session, user_id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if not document:
        return False
    document.processing_status = DOCUMENT_STATUS_MULTIMODAL_QUEUED
    document.processing_progress = 0
    document.updated_at = datetime.utcnow()
    session.add(document)
    session.commit()
    return True


def process_document_multimodal(
    document_id: int,
    file_path: str,
    user_id: int,
    run_id: Optional[str] = None,
    delete_existing_chunks: bool = False,
) -> dict:
    """
    Pipeline multimodal unifié pour import initial et retraitement.
    
    Args:
        document_id: ID du document à traiter
        file_path: Chemin du fichier source (peut être différent de source_file_path en DB lors de l'import initial)
        user_id: ID de l'utilisateur
        run_id: ID de run pour vérification d'annulation
        delete_existing_chunks: Si True, supprime TOUS les chunks existants (retraitement complet).
                                Si False, supprime seulement les chunks multimodaux (mode additif).
    
    Returns:
        dict avec document_id, chunks, status
    """
    from app.services.document_run import is_processing_run_current
    from app.services.file_conversion import ensure_pdf_for_ocr
    from app.services.multimodal_page_service import (
        append_multimodal_page_chunks,
        build_multimodal_pages_for_pdf,
        delete_multimodal_chunks_for_document,
        embed_new_multimodal_chunks,
    )
    from app.services.chunk_service import delete_chunks_for_document

    if not settings.MULTIMODAL_ENABLED:
        raise ValueError(
            "Le traitement multimodal est désactivé (MULTIMODAL_ENABLED=false)."
        )

    ld = get_library_document_logger()
    ld.info(
        "[Multimodal] Démarrage document_id=%s user_id=%s file=%s delete_existing=%s",
        document_id,
        user_id,
        file_path,
        delete_existing_chunks,
    )
    
    if run_id is not None and not is_processing_run_current(document_id, run_id):
        return {"document_id": document_id, "status": "aborted", "reason": "stale_run"}

    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            raise ValueError("Document introuvable ou accès refusé")
        library = session.get(Library, document.library_id)
        if not library:
            raise ValueError("Document introuvable ou accès refusé")
        if library.is_global:
            pass
        elif library.user_id == user_id or document.user_id == user_id:
            pass
        else:
            raise ValueError("Document introuvable ou accès refusé")
        
        title = document.title or ""
        
        # Nettoyage des chunks existants selon le mode
        if delete_existing_chunks:
            ld.info("[Multimodal] document_id=%s — suppression de TOUS les chunks", document_id)
            delete_chunks_for_document(session, document_id, commit=True)
        else:
            ld.info("[Multimodal] document_id=%s — suppression des chunks multimodaux uniquement", document_id)
            delete_multimodal_chunks_for_document(session, document_id, commit=True)
        
        document = session.get(Document, document_id)
        if not document:
            raise ValueError("Document introuvable après nettoyage des chunks")

        document.processing_status = "processing"
        document.processing_progress = 10
        document.updated_at = datetime.utcnow()
        session.add(document)
        session.commit()

    try:
        # Vérifier que le fichier existe
        if not Path(file_path).is_file():
            raise ValueError(f"Fichier source introuvable: {file_path}")
        
        pdf_input = ensure_pdf_for_ocr(str(file_path))
        ld.info("[Multimodal] document_id=%s PDF=%s", document_id, pdf_input)

        chunk_specs = build_multimodal_pages_for_pdf(
            pdf_input, title, document_id
        )

        if run_id is not None and not is_processing_run_current(document_id, run_id):
            return {
                "document_id": document_id,
                "status": "aborted",
                "reason": "stale_run",
            }

        if not chunk_specs:
            raise ValueError("Aucune page synthétisée")

        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                raise ValueError("Document introuvable")
            chunks = append_multimodal_page_chunks(session, document, chunk_specs)
            document.processing_progress = 90
            document.updated_at = datetime.utcnow()
            session.add(document)
            session.commit()
            chunk_count = len(chunks)

        if run_id is not None and not is_processing_run_current(document_id, run_id):
            return {
                "document_id": document_id,
                "status": "aborted",
                "reason": "stale_run",
                "chunks": chunk_count,
            }

        embed_new_multimodal_chunks(document_id)

        with Session(engine) as session:
            document = session.get(Document, document_id)
            if document:
                document.processing_status = "completed"
                document.processing_progress = 100
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()

        ld.info(
            "[Multimodal] document_id=%s — FIN OK chunks=%s",
            document_id,
            chunk_count,
        )
        return {
            "document_id": document_id,
            "chunks": chunk_count,
            "status": "completed",
        }
    except Exception as e:
        logger.error(
            "process_document_multimodal échec document_id=%s: %s",
            document_id,
            e,
            exc_info=True,
        )
        with Session(engine) as session:
            d = session.get(Document, document_id)
            if d:
                d.processing_status = "failed"
                d.updated_at = datetime.utcnow()
                session.add(d)
                session.commit()
        raise


def multimodal_reindex_library_document(
    document_id: int, user_id: int, run_id: Optional[str] = None
) -> dict:
    """
    Ajoute des chunks multimodal v2 (1–5 sections + synthèse/page) via pymupdf + mistral-small.
    Ne modifie pas les chunks existants ; remplace les anciens chunks multimodal à la relance.
    
    DEPRECATED: Cette fonction est maintenue pour compatibilité mais appelle process_document_multimodal.
    """
    ld = get_library_document_logger()
    ld.info(
        "[Multimodal Reindex] document_id=%s user_id=%s (appel via wrapper legacy)",
        document_id,
        user_id,
    )
    
    # Récupérer le source_file_path depuis la DB
    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            raise ValueError("Document introuvable ou accès refusé")
        if not document.source_file_path:
            raise ValueError("Aucun fichier source enregistré pour ce document")
        src = Path(document.source_file_path)
        if not src.is_file():
            raise ValueError("Fichier source introuvable sur le disque")
        file_path = str(src)
    
    # Appeler le pipeline unifié en mode "additif" (ne supprime que les chunks multimodaux)
    return process_document_multimodal(
        document_id=document_id,
        file_path=file_path,
        user_id=user_id,
        run_id=run_id,
        delete_existing_chunks=False,
    )


def mark_all_eligible_documents_reindex_queued(user_id: int) -> int:
    """
    Marque tous les documents fichier éligibles en reindex_queued (chunks inchangés).
    Appelé au début de reindex_all_library_documents dans le worker.
    """
    from app.services.library_service import get_or_create_user_library

    n = 0
    with Session(engine) as session:
        library = get_or_create_user_library(session, user_id)
        docs = get_documents_by_library(session, library.id, user_id)
        for doc in docs:
            if doc.document_type != "document" or not doc.source_file_path:
                continue
            if not Path(doc.source_file_path).is_file():
                continue
            doc.processing_status = DOCUMENT_STATUS_REINDEX_QUEUED
            doc.processing_progress = 0
            doc.updated_at = datetime.utcnow()
            session.add(doc)
            n += 1
        if n:
            session.commit()
    return n


def reindex_all_library_documents(user_id: int) -> dict:
    """
    Réindexe séquentiellement tous les documents fichier de la bibliothèque utilisateur.
    Réservé au worker Celery (tâche reindex_all_library_documents_task).
    Marque d'abord tous les éligibles en reindex_queued (chunks conservés), puis traite un par un.
    """
    from app.services.library_service import get_or_create_user_library

    ld = get_library_document_logger()
    ld.info(
        "[Réindex tous] user_id=%s — marquage reindex_queued puis boucle.",
        user_id,
    )
    marked = mark_all_eligible_documents_reindex_queued(user_id)
    ld.info("[Réindex tous] user_id=%s — %s document(s) marqués en attente.", user_id, marked)

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
            logger.warning(
                "reindex_all: fichier absent, doc_id=%s path=%s",
                doc.id,
                doc.source_file_path,
            )
            results["skipped"] += 1
            continue
        try:
            reindex_library_document(doc.id, user_id)
            results["ok"] += 1
        except Exception as e:
            logger.exception("reindex_all: échec document_id=%s: %s", doc.id, e)
            results["failed"].append(
                {
                    "document_id": doc.id,
                    "title": doc.title,
                    "error": str(e),
                }
            )

    ld.info(
        "[Réindex tous] user_id=%s — fin ok=%s skipped=%s failed=%s",
        user_id,
        results["ok"],
        results["skipped"],
        len(results["failed"]),
    )
    return results


def create_document(
    session: Session,
    document_create: DocumentCreate,
    library_id: int,
    user_id: int,
    space_ids: List[int]
) -> Optional[Document]:
    """Crée un nouveau document et l'associe aux espaces spécifiés."""
    from app.services.library_service import get_library_by_id
    from app.services.space_service import get_space_by_id
    from app.services.document_space_service import link_document_to_space
    
    library = get_library_by_id(session, library_id, user_id)
    if not library:
        logger.error(f"Bibliothèque {library_id} non trouvée")
        return None
    
    for space_id in space_ids:
        space = get_space_by_id(session, space_id, user_id)
        if not space:
            logger.error(f"Espace {space_id} non trouvé")
            return None
    
    document = Document(
        title=document_create.title,
        content=document_create.content,
        document_type=document_create.document_type,
        source_file_path=document_create.source_file_path,
        processing_status=document_create.processing_status,
        processing_progress=document_create.processing_progress,
        is_paid=document_create.is_paid,
        folder_id=document_create.folder_id,
        library_id=library_id,
        user_id=user_id
    )
    session.add(document)
    session.commit()
    session.refresh(document)
    
    for space_id in space_ids:
        link_document_to_space(session, document.id, space_id, user_id)
    
    logger.info(f"Document créé: {document.title} (ID: {document.id}) accessible dans {len(space_ids)} espace(s)")
    return document


def get_document_by_id(session: Session, document_id: int, user_id: int) -> Optional[Document]:
    """Récupère un document par son ID (bibliothèque globale partagée)."""
    statement = select(Document).where(
        Document.id == document_id
    )
    return session.exec(statement).first()


def get_documents_by_folder(session: Session, folder_id: Optional[int], library_id: int, user_id: int) -> List[Document]:
    """Récupère tous les documents d'un dossier (ou racine si folder_id est None), hors correctifs feedback."""
    statement = select(Document).where(
        Document.library_id == library_id,
        Document.folder_id == folder_id,
        *_exclude_feedback_corrective_where(),
    ).order_by(Document.created_at.desc())
    return list(session.exec(statement).all())


def get_documents_by_library(session: Session, library_id: int, user_id: int) -> List[Document]:
    """Récupère tous les documents visibles d'une bibliothèque (hors correctifs feedback)."""
    statement = select(Document).where(
        Document.library_id == library_id,
        *_exclude_feedback_corrective_where(),
    ).order_by(Document.created_at.desc())
    return list(session.exec(statement).all())


def get_documents_by_space(session: Session, space_id: int, user_id: int) -> List[Document]:
    """Récupère tous les documents accessibles dans un espace."""
    from app.services.document_space_service import get_documents_for_space
    return get_documents_for_space(session, space_id, user_id)


def update_document(
    session: Session,
    document_id: int,
    document_update: DocumentUpdate,
    user_id: int
) -> Optional[Document]:
    """Met à jour un document."""
    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return None
    
    update_data = document_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(document, key, value)
    
    document.updated_at = datetime.utcnow()
    session.add(document)
    session.commit()
    session.refresh(document)
    
    return document


def move_document(
    session: Session,
    document_id: int,
    new_folder_id: Optional[int],
    user_id: int
) -> Optional[Document]:
    """Déplace un document vers un nouveau dossier."""
    from app.services.folder_service import get_folder_by_id
    
    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return None
    
    if new_folder_id:
        folder = get_folder_by_id(session, new_folder_id, user_id)
        if not folder:
            return None

        # Bibliothèque générale partagée:
        # autoriser le déplacement entre bibliothèques historiques et
        # aligner le document sur la bibliothèque du dossier cible.
        if folder.library_id != document.library_id:
            logger.info(
                "Déplacement inter-bibliothèque du document %d: %s -> %s",
                document_id,
                document.library_id,
                folder.library_id,
            )
            document.library_id = folder.library_id
    
    document.folder_id = new_folder_id
    document.updated_at = datetime.utcnow()
    session.add(document)
    session.commit()
    session.refresh(document)
    
    return document


def delete_document(session: Session, document_id: int, user_id: int) -> bool:
    """Supprime un document, ses chunks, et toutes ses associations."""
    from app.services.chunk_service import delete_chunks_for_document
    from app.services.document_space_service import get_document_spaces
    # Empêche un traitement asynchrone tardif de recréer des chunks.
    _mark_document_processing_cancelled(document_id)

    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return False
    
    doc_spaces = get_document_spaces(session, document_id, user_id)
    delete_chunks_for_document(session, document_id, commit=False)

    for doc_space in doc_spaces:
        session.delete(doc_space)
    
    if document.source_file_path and os.path.exists(document.source_file_path):
        try:
            os.remove(document.source_file_path)
            logger.info(f"Fichier source supprimé: {document.source_file_path}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le fichier source: {e}")
    
    images_dir = Path(f"media/images/{document_id}")
    if images_dir.exists():
        try:
            import shutil
            shutil.rmtree(images_dir)
            logger.info(f"Images supprimées: {images_dir}")
        except Exception as e:
            logger.warning(f"Impossible de supprimer le dossier d'images: {e}")
    
    session.delete(document)
    session.commit()
    
    logger.info(f"Document supprimé: {document.title} (ID: {document_id})")
    return True


def add_document_to_spaces(
    session: Session,
    document_id: int,
    space_ids: List[int],
    user_id: int
) -> bool:
    """Ajoute un document à plusieurs espaces."""
    from app.services.document_space_service import link_document_to_space

    document = get_document_by_id(session, document_id, user_id)
    if not document:
        return False

    for space_id in space_ids:
        link = link_document_to_space(session, document_id, space_id, user_id)
        if link is None:
            logger.error(
                "Impossible de lier le document %s à l'espace %s",
                document_id,
                space_id,
            )
            return False

    return True


def remove_document_from_spaces(
    session: Session,
    document_id: int,
    space_ids: List[int],
    user_id: int
) -> bool:
    """Retire un document de plusieurs espaces."""
    from app.services.document_space_service import unlink_document_from_space
    
    for space_id in space_ids:
        unlink_document_from_space(session, document_id, space_id, user_id)
    
    return True


def apply_document_spaces_update(
    document_id: int,
    add_space_ids: List[int],
    remove_space_ids: List[int],
    user_id: int,
) -> bool:
    """
    Applique les changements d'association document/espaces dans une session dédiée.
    Utilisé par les workers (Celery/thread) pour ne pas bloquer le process web.
    """
    with Session(engine) as session:
        document = get_document_by_id(session, document_id, user_id)
        if not document:
            logger.error("Document %s non trouvé pour mise à jour des espaces", document_id)
            return False

        if add_space_ids:
            success = add_document_to_spaces(session, document_id, add_space_ids, user_id)
            if not success:
                return False

        if remove_space_ids:
            success = remove_document_from_spaces(
                session, document_id, remove_space_ids, user_id
            )
            if not success:
                return False

        return True


def _process_document_worker():
    """Worker thread : un document à la fois, de bout en bout (pas de chevauchement)."""
    try:
        import os
        if hasattr(os, "nice"):
            os.nice(5)
    except Exception:
        pass

    logger.info("Worker de traitement de documents démarré et en attente de tâches...")
    while True:
        try:
            task = document_task_queue.get(timeout=2)
        except Empty:
            continue

        if task is None:
            document_task_queue.task_done()
            logger.info("Signal d'arrêt reçu, arrêt du worker de documents")
            return

        if not isinstance(task, tuple):
            document_task_queue.task_done()
            continue
        if len(task) >= 3:
            document_id, file_path, run_id = task[0], task[1], task[2]
        else:
            document_id, file_path = task[0], task[1]
            run_id = None
        try:
            logger.info("Worker traite le document %d (file globale)", document_id)
            _process_document_for_id(document_id, file_path, run_id)
            logger.info(
                "Worker a terminé le document %d (toutes étapes incluses)", document_id
            )
        except Exception as e:
            logger.error(
                "Erreur dans le worker de traitement de documents: %s", e, exc_info=True
            )
        finally:
            document_task_queue.task_done()
            time.sleep(0.5)


def _process_document_for_id(
    document_id: int, file_path: str, run_id: Optional[str] = None
):
    """
    Traite un document pour un ID donné en utilisant le pipeline multimodal unifié.
    
    Ce pipeline remplace l'ancien flux PyMuPDF4LLM + MistralOCR par un traitement
    multimodal de meilleure qualité (pymupdf + mistral-small vision par page).
    """
    from app.services.document_run import is_processing_run_current

    ld = get_library_document_logger()
    try:
        logger.info("Démarrage du traitement multimodal du document %d", document_id)
        ld.info(
            "[Upload/Pipeline Multimodal] document_id=%s — DÉBUT traitement bibliothèque fichier=%s "
            "(worker thread ou Celery). Pipeline : PDF → multimodal page par page → chunks → embeddings.",
            document_id,
            file_path,
        )

        if run_id is not None and not is_processing_run_current(document_id, run_id):
            logger.info(
                "Traitement ignoré (run_id obsolète) pour document_id=%s", document_id
            )
            ld.info(
                "[Upload/Pipeline Multimodal] document_id=%s — abandon : run_id ne correspond plus (stop/reprise).",
                document_id,
            )
            return

        if _should_abort_processing(document_id):
            logger.info(
                "Traitement annulé avant démarrage effectif pour document %d",
                document_id,
            )
            ld.info(
                "[Upload/Pipeline Multimodal] document_id=%s — annulé avant démarrage (cancel flag).",
                document_id,
            )
            _finalize_pipeline_abort(document_id)
            return

        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                logger.error("Document %d non trouvé pour traitement", document_id)
                ld.error(
                    "[Upload/Pipeline Multimodal] document_id=%s — document introuvable en base, arrêt.",
                    document_id,
                )
                _clear_document_processing_cancelled(document_id)
                return

            if document.processing_status in LIBRARY_USER_STOPPED_STATUSES:
                ld.info(
                    "[Upload/Pipeline Multimodal] document_id=%s — statut final %s, pas de traitement.",
                    document_id,
                    document.processing_status,
                )
                _clear_document_processing_cancelled(document_id)
                return

            user_id = document.user_id

        # Avant le traitement multimodal, on doit sauvegarder le fichier de manière permanente
        # pour que process_document_multimodal puisse y accéder
        from app.services.file_conversion import ensure_pdf_for_ocr
        
        original_ext = Path(file_path).suffix.lower()
        pdf_input_path = file_path
        converted_to_pdf = False
        
        try:
            pdf_input_path = ensure_pdf_for_ocr(file_path)
            converted_to_pdf = pdf_input_path != file_path
            ld.info(
                "[Upload/Pipeline Multimodal] document_id=%s — préparation : entrée=%s converti_pdf=%s",
                document_id,
                pdf_input_path,
                converted_to_pdf,
            )
        except Exception as e:
            logger.error(
                "Conversion vers PDF impossible pour le document %d (%s): %s",
                document_id,
                file_path,
                e,
                exc_info=True,
            )
            ld.error(
                "[Upload/Pipeline Multimodal] document_id=%s — échec ensure_pdf_for_ocr : %s",
                document_id,
                e,
                exc_info=True,
            )
            with Session(engine) as session:
                doc = session.get(Document, document_id)
                if doc:
                    doc.processing_status = "failed"
                    doc.processing_progress = 10
                    doc.content = f"❌ Erreur lors de la conversion du document: {str(e)}"
                    doc.updated_at = datetime.utcnow()
                    session.add(doc)
                    session.commit()
            return

        # Sauvegarder le fichier de manière permanente
        output_ext = Path(pdf_input_path).suffix.lower() or ".bin"
        permanent_output_path = Path(f"media/documents/{document_id}{output_ext}")
        permanent_output_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil

        # Retenter proprement sur relances/erreurs précédentes
        if permanent_output_path.exists():
            permanent_output_path.unlink()

        if converted_to_pdf:
            original_permanent_path = Path(
                f"media/documents/{document_id}{original_ext}"
            )
            original_permanent_path.parent.mkdir(parents=True, exist_ok=True)
            if original_permanent_path.exists():
                original_permanent_path.unlink()
            shutil.move(file_path, str(original_permanent_path))

        shutil.move(pdf_input_path, str(permanent_output_path))
        
        with Session(engine) as session:
            document = session.get(Document, document_id)
            if document:
                document.source_file_path = str(permanent_output_path)
                document.updated_at = datetime.utcnow()
                session.add(document)
                session.commit()
        
        logger.info(
            "Fichier traité déplacé vers chemin permanent: %s",
            permanent_output_path,
        )
        ld.info(
            "[Upload/Pipeline Multimodal] document_id=%s — fichier enregistré sous %s",
            document_id,
            permanent_output_path,
        )

        if _should_abort_processing(document_id):
            logger.info(
                "Traitement annulé après sauvegarde fichier pour document %d",
                document_id,
            )
            ld.info(
                "[Upload/Pipeline Multimodal] document_id=%s — annulé après sauvegarde fichier.",
                document_id,
            )
            _finalize_pipeline_abort(document_id)
            return

        # Appeler le pipeline multimodal unifié
        with trace_pipeline(
            "multimodal_import",
            inputs={
                "document_id": document_id,
                "file_path": str(permanent_output_path),
            },
            tags=["ingestion", "multimodal"],
        ):
            result = process_document_multimodal(
                document_id=document_id,
                file_path=str(permanent_output_path),
                user_id=user_id,
                run_id=run_id,
                delete_existing_chunks=True,  # Import initial: on supprime tous les chunks
            )

        if result.get("status") == "completed":
            ld.info(
                "[Upload/Pipeline Multimodal] document_id=%s — FIN OK chunks=%s",
                document_id,
                result.get("chunks", 0),
            )
        elif result.get("status") == "aborted":
            ld.info(
                "[Upload/Pipeline Multimodal] document_id=%s — abandonné (%s)",
                document_id,
                result.get("reason", "unknown"),
            )
        else:
            ld.warning(
                "[Upload/Pipeline Multimodal] document_id=%s — statut inattendu: %s",
                document_id,
                result.get("status"),
            )

    except Exception as e:
        logger.error("Erreur lors du traitement multimodal du document %d: %s", document_id, e, exc_info=True)
        ld.error(
            "[Upload/Pipeline Multimodal] document_id=%s — ERREUR non gérée : %s",
            document_id,
            e,
            exc_info=True,
        )
        try:
            with Session(engine) as session:
                document = session.get(Document, document_id)
                if document and document.processing_status not in LIBRARY_USER_STOPPED_STATUSES:
                    document.processing_status = "failed"
                    document.processing_progress = max(document.processing_progress or 0, 10)
                    document.content = f"❌ Erreur lors du traitement du document: {str(e)}"
                    document.updated_at = datetime.utcnow()
                    session.add(document)
                    session.commit()
        except Exception as update_error:
            logger.error("Erreur lors de la mise à jour du statut d'erreur pour le document %d: %s", document_id, update_error)
    finally:
        try:
            _clear_document_processing_cancelled(document_id)
        except Exception:
            pass


def enqueue_library_document_thread(
    document_id: int, file_path: str, run_id: Optional[str] = None
):
    """Ajoute un document à la file thread globale (sans Celery)."""
    try:
        with Session(engine) as session:
            document = session.get(Document, document_id)
            if not document:
                logger.error("Document %d non trouvé pour ajout à la queue", document_id)
                return
            if document.processing_status in LIBRARY_USER_STOPPED_STATUSES:
                logger.info(
                    "Document %d non ajouté à la file (statut final %s)",
                    document_id,
                    document.processing_status,
                )
                get_library_document_logger().info(
                    "[Queue] document_id=%s — ignoré (cancelled/skipped), pas d’enqueue.",
                    document_id,
                )
                return
    except Exception as e:
        logger.error(
            "Erreur lors de la vérification du document %d: %s", document_id, e, exc_info=True
        )
        return

    _ensure_document_workers()

    with _document_queue_lock:
        document_task_queue.put((document_id, file_path, run_id))
        queue_size = document_task_queue.qsize()

    logger.info(
        "✅ Document %d ajouté à la file globale (taille: %d)", document_id, queue_size
    )
    get_library_document_logger().info(
        "[Queue] document_id=%s — entrée file thread worker (position file ~%d) fichier=%s",
        document_id,
        queue_size,
        file_path,
    )


def process_document_async(document_id: int, file_path: str):
    """Délègue à Celery ou à la file thread selon TASK_BACKEND_MODE."""
    from app.services.task_dispatch import dispatch_library_document
    from app.services.document_run import refresh_document_processing_run_id

    with Session(engine) as session:
        doc = session.get(Document, document_id)
        if not doc:
            return
        rid = refresh_document_processing_run_id(doc)
        doc.updated_at = datetime.utcnow()
        session.add(doc)
        session.commit()
    dispatch_library_document(document_id, file_path, rid)


def _ensure_document_workers():
    """S'assurer que les workers de traitement de documents sont démarrés."""
    global document_workers

    with _document_workers_lock:
        if not document_workers or not any(w.is_alive() for w in document_workers):
            document_workers = []
            # Toujours 1 worker : un document va au bout (OCR → chunks → embeddings) avant le suivant.
            num_workers = 1
            for i in range(num_workers):
                worker = threading.Thread(target=_process_document_worker, daemon=True)
                worker.start()
                document_workers.append(worker)
                logger.info("Worker de traitement de documents %d démarré", i + 1)


def enqueue_reindex_library_document_thread(
    document_id: int, user_id: int, run_id: Optional[str] = None
) -> None:
    """Exécute la réindexation d'un document dans un thread d'arrière-plan."""
    def _runner():
        try:
            reindex_library_document(document_id, user_id, run_id)
        except Exception:
            logger.exception(
                "Thread reindex_library_document échec document_id=%s", document_id
            )

    threading.Thread(
        target=_runner,
        name=f"reindex-document-{document_id}",
        daemon=True,
    ).start()


def enqueue_reindex_all_library_documents_thread(user_id: int) -> None:
    """Exécute la réindexation globale de la bibliothèque dans un thread d'arrière-plan."""
    def _runner():
        try:
            reindex_all_library_documents(user_id)
        except Exception:
            logger.exception(
                "Thread reindex_all_library_documents échec user_id=%s", user_id
            )

    threading.Thread(
        target=_runner,
        name=f"reindex-all-library-{user_id}",
        daemon=True,
    ).start()


def enqueue_multimodal_reindex_library_document_thread(
    document_id: int, user_id: int, run_id: Optional[str] = None
) -> None:
    """
    DEPRECATED: Le pipeline multimodal est maintenant unifié avec le retraitement classique.
    Cette fonction est conservée temporairement pour compatibilité.
    
    Exécute le retraitement multimodal dans un thread d'arrière-plan.
    """
    def _runner():
        try:
            multimodal_reindex_library_document(document_id, user_id, run_id)
        except Exception:
            logger.exception(
                "Thread multimodal_reindex échec document_id=%s", document_id
            )

    threading.Thread(
        target=_runner,
        name=f"multimodal-reindex-document-{document_id}",
        daemon=True,
    ).start()
