"""
Dispatch des jobs background : Celery (Redis) ou threads locaux (fallback).
Modes : thread | celery | hybrid (Celery prioritaire, repli threads si échec connexion broker).
"""
from __future__ import annotations

import logging
import threading
import ast
from datetime import datetime
from typing import Any, Literal, Optional

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


def _parse_task_args(task: dict) -> tuple[Any, ...]:
    args = task.get("args", ())
    if isinstance(args, (list, tuple)):
        return tuple(args)
    if isinstance(args, str):
        try:
            parsed = ast.literal_eval(args)
        except Exception:
            return ()
        if isinstance(parsed, (list, tuple)):
            return tuple(parsed)
    return ()


def _extract_document_id_from_celery_task(task: dict) -> Optional[int]:
    """Extrait le document_id des tâches Celery liées à un document bibliothèque."""
    if not isinstance(task, dict):
        return None

    task_name = task.get("name")
    if task_name not in {
        "app.tasks.documents.process_library_document",
        "app.tasks.documents.reindex_library_document_task",
        "app.tasks.documents.process_library_document_kag",
        "app.tasks.documents.process_document_embeddings",
        "app.tasks.documents.update_document_spaces_task",
    }:
        return None

    args = _parse_task_args(task)
    if not args:
        return None
    first_arg = args[0]
    try:
        return int(first_arg) if first_arg is not None else None
    except (TypeError, ValueError):
        return None


def revoke_library_document_tasks(document_id: int) -> dict[str, Any]:
    """
    Révoque les tâches Celery actives/réservées liées à un document (files documents, embeddings, kag).
    Retourne preuve d'opération pour l'API admin.
    """
    mode = get_task_backend_mode()
    if mode == "thread":
        return {"revoked_count": 0, "revoked_task_ids": []}

    try:
        from app.celery_app import celery_app

        inspector = celery_app.control.inspect()
        task_groups = [
            inspector.active() or {},
            inspector.reserved() or {},
            inspector.scheduled() or {},
        ]
    except Exception as exc:
        logger.warning(
            "Impossible d'inspecter Celery pour revoke document_id=%s: %s",
            document_id,
            exc,
        )
        return {"revoked_count": 0, "revoked_task_ids": []}

    revoked_count = 0
    revoked_ids: list[str] = []
    seen: set[str] = set()
    for group in task_groups:
        for tasks in group.values():
            for raw_task in tasks or []:
                task = raw_task.get("request", raw_task) if isinstance(raw_task, dict) else raw_task
                if not isinstance(task, dict):
                    continue

                current_document_id = _extract_document_id_from_celery_task(task)
                task_id = task.get("id")
                if current_document_id != document_id or not task_id:
                    continue
                if task_id in seen:
                    continue

                try:
                    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
                    seen.add(task_id)
                    revoked_ids.append(task_id)
                    revoked_count += 1
                    logger.info(
                        "Tâche Celery révoquée pour document_id=%s task_id=%s",
                        document_id,
                        task_id,
                    )
                except Exception as exc:
                    logger.warning(
                        "Échec revoke task Celery document_id=%s task_id=%s: %s",
                        document_id,
                        task_id,
                        exc,
                    )

    return {"revoked_count": revoked_count, "revoked_task_ids": revoked_ids}


def revoke_library_tasks_bulk(
    document_ids: set[int], user_id: Optional[int] = None
) -> dict[str, Any]:
    """
    Révoque en masse les tâches liées à la bibliothèque:
    - tasks document (documents/embeddings/kag) par document_id
    - task globale reindex_all_library_documents_task par user_id (si fourni)
    """
    mode = get_task_backend_mode()
    if mode == "thread":
        return {"revoked_count": 0, "revoked_task_ids": []}

    if not document_ids and user_id is None:
        return {"revoked_count": 0, "revoked_task_ids": []}

    try:
        from app.celery_app import celery_app

        inspector = celery_app.control.inspect()
        task_groups = [
            inspector.active() or {},
            inspector.reserved() or {},
            inspector.scheduled() or {},
        ]
    except Exception as exc:
        logger.warning("Impossible d'inspecter Celery pour revoke bulk: %s", exc)
        return {"revoked_count": 0, "revoked_task_ids": []}

    revoked_ids: list[str] = []
    seen: set[str] = set()
    for group in task_groups:
        for tasks in group.values():
            for raw_task in tasks or []:
                task = (
                    raw_task.get("request", raw_task)
                    if isinstance(raw_task, dict)
                    else raw_task
                )
                if not isinstance(task, dict):
                    continue

                task_id = task.get("id")
                if not task_id or task_id in seen:
                    continue

                name = str(task.get("name") or "")
                should_revoke = False

                doc_id = _extract_document_id_from_celery_task(task)
                if doc_id is not None and doc_id in document_ids:
                    should_revoke = True

                if (
                    not should_revoke
                    and user_id is not None
                    and name == "app.tasks.documents.reindex_all_library_documents_task"
                ):
                    args = _parse_task_args(task)
                    first_arg = args[0] if args else None
                    try:
                        should_revoke = int(first_arg) == int(user_id)
                    except Exception:
                        should_revoke = False

                if not should_revoke:
                    continue

                try:
                    celery_app.control.revoke(task_id, terminate=True, signal="SIGTERM")
                    seen.add(task_id)
                    revoked_ids.append(task_id)
                except Exception as exc:
                    logger.warning("Échec revoke bulk task_id=%s: %s", task_id, exc)

    return {"revoked_count": len(revoked_ids), "revoked_task_ids": revoked_ids}


def _send_library_document(
    document_id: int, file_path: str, run_id: Optional[str]
) -> bool:
    from app.library_document_logging import get_library_document_logger
    from app.tasks.documents import process_library_document

    res = process_library_document.apply_async(
        args=[document_id, file_path, run_id],
        queue="documents",
    )
    logger.info(
        "task_dispatch library_document document_id=%s celery_task_id=%s",
        document_id,
        res.id,
    )
    get_library_document_logger().info(
        "[Dispatch] document_id=%s — tâche Celery process_library_document file documents queue, "
        "task_id=%s fichier=%s",
        document_id,
        res.id,
        file_path,
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


def _send_reindex_library(
    document_id: int, user_id: int, run_id: Optional[str]
) -> str:
    from app.library_document_logging import get_library_document_logger
    from app.tasks.documents import reindex_library_document_task

    async_result = reindex_library_document_task.apply_async(
        args=[document_id, user_id, run_id],
        queue="documents",
    )
    logger.info(
        "task_dispatch reindex_library document_id=%s celery_task_id=%s",
        document_id,
        async_result.id,
    )
    get_library_document_logger().info(
        "[Dispatch] document_id=%s — reindex Celery task_id=%s user_id=%s",
        document_id,
        async_result.id,
        user_id,
    )
    return async_result.id


def _send_reindex_all_library(user_id: int) -> str:
    from app.library_document_logging import get_library_document_logger
    from app.tasks.documents import reindex_all_library_documents_task

    async_result = reindex_all_library_documents_task.apply_async(
        args=[user_id],
        queue="documents",
    )
    logger.info(
        "task_dispatch reindex_all_library user_id=%s celery_task_id=%s",
        user_id,
        async_result.id,
    )
    get_library_document_logger().info(
        "[Dispatch] reindex_all_library — task_id=%s user_id=%s",
        async_result.id,
        user_id,
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


def _send_document_embeddings(document_id: int, run_id: Optional[str]) -> bool:
    from app.tasks.documents import process_document_embeddings

    res = process_document_embeddings.apply_async(
        args=[document_id, run_id],
        queue="embeddings",
    )
    logger.info(
        "task_dispatch document_embeddings document_id=%s celery_task_id=%s",
        document_id,
        res.id,
    )
    return True


def _send_library_document_kag(document_id: int, run_id: Optional[str]) -> bool:
    from app.tasks.documents import process_library_document_kag

    res = process_library_document_kag.apply_async(
        args=[document_id, run_id],
        queue="kag",
    )
    logger.info(
        "task_dispatch library_kag document_id=%s celery_task_id=%s",
        document_id,
        res.id,
    )
    return True


def _run_kag_thread(document_id: int, run_id: Optional[str]) -> None:
    from app.services.chunk_service import run_kag_for_library_document

    def _runner():
        try:
            run_kag_for_library_document(document_id, run_id)
        except Exception:
            logger.exception("Thread KAG échoué document_id=%s", document_id)

    threading.Thread(
        target=_runner,
        name=f"kag-doc-{document_id}",
        daemon=True,
    ).start()


def dispatch_library_document_kag(document_id: int) -> None:
    """Enqueue la phase KAG bibliothèque après embeddings (Celery queue « kag » ou thread)."""
    mode = get_task_backend_mode()
    run_id: Optional[str] = None
    with Session(engine) as session:
        doc = session.get(Document, document_id)
        if doc:
            run_id = doc.processing_run_id
    if mode == "thread":
        _run_kag_thread(document_id, run_id)
        return
    try:
        _send_library_document_kag(document_id, run_id)
    except Exception as exc:
        logger.warning(
            "Celery indisponible pour KAG document_id=%s: %s", document_id, exc
        )
        if mode == "hybrid":
            _run_kag_thread(document_id, run_id)
            return
        raise RuntimeError(_celery_only_failure_message()) from exc


def _send_document_spaces_update(
    document_id: int,
    add_space_ids: list[int],
    remove_space_ids: list[int],
    user_id: int,
) -> str:
    from app.tasks.documents import update_document_spaces_task

    async_result = update_document_spaces_task.apply_async(
        args=[document_id, add_space_ids, remove_space_ids, user_id],
        queue="documents",
    )
    logger.info(
        "task_dispatch update_document_spaces document_id=%s celery_task_id=%s",
        document_id,
        async_result.id,
    )
    return async_result.id


def _run_document_spaces_update_thread(
    document_id: int,
    add_space_ids: list[int],
    remove_space_ids: list[int],
    user_id: int,
) -> None:
    from app.services.document_service_new import apply_document_spaces_update

    def _runner():
        try:
            apply_document_spaces_update(
                document_id=document_id,
                add_space_ids=add_space_ids,
                remove_space_ids=remove_space_ids,
                user_id=user_id,
            )
        except Exception:
            logger.exception(
                "Thread update_document_spaces échec document_id=%s", document_id
            )

    threading.Thread(
        target=_runner,
        name=f"document-spaces-{document_id}",
        daemon=True,
    ).start()


def dispatch_library_document(
    document_id: int, file_path: str, run_id: Optional[str] = None
) -> None:
    """Enqueue traitement document bibliothèque (Celery ou thread)."""
    from app.library_document_logging import get_library_document_logger

    mode = get_task_backend_mode()
    get_library_document_logger().info(
        "[Dispatch] document_id=%s — dispatch_library_document mode=%s fichier=%s",
        document_id,
        mode,
        file_path,
    )
    if mode == "thread":
        from app.services.document_service_new import enqueue_library_document_thread

        enqueue_library_document_thread(document_id, file_path, run_id)
        return

    try:
        _send_library_document(document_id, file_path, run_id)
    except Exception as exc:
        logger.warning(
            "Celery indisponible pour library document_id=%s: %s", document_id, exc
        )
        get_library_document_logger().warning(
            "[Dispatch] document_id=%s — Celery indisponible (%s), mode=%s",
            document_id,
            exc,
            mode,
        )
        if mode == "hybrid":
            from app.services.document_service_new import enqueue_library_document_thread

            enqueue_library_document_thread(document_id, file_path, run_id)
            get_library_document_logger().info(
                "[Dispatch] document_id=%s — repli hybrid vers file thread.",
                document_id,
            )
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


def dispatch_reindex_library(document_id: int, user_id: int) -> str:
    """
    Enfile la réindexation sur la queue Celery « documents » uniquement.
    Docling, chunks, embeddings et KAG s'exécutent dans le worker, pas dans l'API.
    Retourne l'identifiant de tâche Celery.
    """
    run_id: Optional[str] = None
    with Session(engine) as session:
        doc = session.get(Document, document_id)
        if doc:
            from app.services.document_run import refresh_document_processing_run_id

            run_id = refresh_document_processing_run_id(doc)
            doc.updated_at = datetime.utcnow()
            session.add(doc)
            session.commit()
    try:
        return _send_reindex_library(document_id, user_id, run_id)
    except Exception as exc:
        logger.warning(
            "Échec enqueue reindex document_id=%s user_id=%s: %s",
            document_id,
            user_id,
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            "Impossible d'enfiler la réindexation : le service de tâches (Celery) est indisponible."
        ) from exc


def dispatch_reindex_all_library(user_id: int) -> str:
    """
    Enfile la réindexation globale de la bibliothèque sur la queue Celery « documents ».
    """
    try:
        return _send_reindex_all_library(user_id)
    except Exception as exc:
        logger.warning(
            "Échec enqueue reindex_all_library user_id=%s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            "Impossible d'enfiler la réindexation globale : le service de tâches (Celery) est indisponible."
        ) from exc


def dispatch_document_spaces_update(
    document_id: int,
    add_space_ids: list[int],
    remove_space_ids: list[int],
    user_id: int,
) -> str:
    """
    Ajout/retrait des espaces d'un document via worker.
    Retourne un identifiant de tâche (Celery id ou id logique thread).
    """
    mode = get_task_backend_mode()
    if mode == "thread":
        _run_document_spaces_update_thread(
            document_id=document_id,
            add_space_ids=add_space_ids,
            remove_space_ids=remove_space_ids,
            user_id=user_id,
        )
        return f"thread-document-spaces-{document_id}"

    try:
        return _send_document_spaces_update(
            document_id=document_id,
            add_space_ids=add_space_ids,
            remove_space_ids=remove_space_ids,
            user_id=user_id,
        )
    except Exception as exc:
        logger.warning(
            "Celery indisponible pour update_document_spaces document_id=%s: %s",
            document_id,
            exc,
        )
        if mode == "hybrid":
            _run_document_spaces_update_thread(
                document_id=document_id,
                add_space_ids=add_space_ids,
                remove_space_ids=remove_space_ids,
                user_id=user_id,
            )
            return f"thread-document-spaces-{document_id}"
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
                _send_document_embeddings(doc.id, doc.processing_run_id)
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
