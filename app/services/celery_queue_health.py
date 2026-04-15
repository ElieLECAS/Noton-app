"""Inspection légère Celery : files, workers (admin)."""

from __future__ import annotations

import logging
import ast
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlmodel import Session, select

from app.database import engine
from app.models.document import Document

logger = logging.getLogger(__name__)

QUEUE_NAMES = ("documents", "embeddings", "kag")


def get_queue_health_payload() -> dict[str, Any]:
    """Profondeur approximative + workers actifs (best-effort si inspect indisponible)."""
    out: dict[str, Any] = {
        "queues": {},
        "workers": {},
        "inspect_error": None,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        from app.celery_app import celery_app

        inspector = celery_app.control.inspect()
        active = inspector.active() or {}
        reserved = inspector.reserved() or {}
        scheduled = inspector.scheduled() or {}
        stats = inspector.stats() or {}
        ping = inspector.ping() or {}

        for q in QUEUE_NAMES:
            depth = 0
            oldest_hint: Optional[str] = None
            for bucket in (reserved, scheduled):
                for _wname, tasks in (bucket or {}).items():
                    for t in tasks or []:
                        req = t.get("request", t) if isinstance(t, dict) else t
                        if not isinstance(req, dict):
                            continue
                        delivery = req.get("delivery_info") or {}
                        if delivery.get("routing_key") == q or q in str(
                            req.get("hostname", "")
                        ):
                            depth += 1
            out["queues"][q] = {
                "approx_depth": depth,
                "oldest_waiting_age_hint": oldest_hint,
            }

        out["workers"] = {
            "stats": stats,
            "ping": ping,
            "active_tasks_by_worker": {
                w: len(tasks or []) for w, tasks in active.items()
            },
        }
    except Exception as exc:
        logger.warning("Celery inspect indisponible: %s", exc, exc_info=True)
        out["inspect_error"] = str(exc)
    return out


def _parse_args(raw_args: Any) -> tuple[Any, ...]:
    if isinstance(raw_args, (list, tuple)):
        return tuple(raw_args)
    if isinstance(raw_args, str):
        try:
            parsed = ast.literal_eval(raw_args)
            if isinstance(parsed, (list, tuple)):
                return tuple(parsed)
        except Exception:
            return ()
    return ()


def _extract_doc_task(task: dict[str, Any], state: str, worker_name: str) -> Optional[dict[str, Any]]:
    name = str(task.get("name") or "")
    if name not in {
        "app.tasks.documents.process_library_document",
        "app.tasks.documents.process_document_embeddings",
        "app.tasks.documents.process_library_document_kag",
        "app.tasks.documents.reindex_library_document_task",
    }:
        return None

    args = _parse_args(task.get("args"))
    first = args[0] if args else None
    try:
        document_id = int(first) if first is not None else None
    except Exception:
        document_id = None

    queue = (task.get("delivery_info") or {}).get("routing_key")
    return {
        "worker": worker_name,
        "state": state,  # active / reserved / scheduled
        "task_id": task.get("id"),
        "task_name": name,
        "queue": queue,
        "document_id": document_id,
        "document_name": None,
    }


def get_workers_document_tasks_view() -> dict[str, Any]:
    """
    Vue workers demandée: uniquement documents en cours/en attente
    pour les 2 workers principaux (worker documents, worker-kag).
    """
    out: dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "workers": {
            "worker": [],
            "worker-kag": [],
        },
        "inspect_error": None,
    }
    seen_doc_ids_by_worker: dict[str, set[int]] = {
        "worker": set(),
        "worker-kag": set(),
    }
    try:
        from app.celery_app import celery_app

        inspector = celery_app.control.inspect()
        groups = {
            "active": inspector.active() or {},
            "reserved": inspector.reserved() or {},
            "scheduled": inspector.scheduled() or {},
        }

        for state, by_worker in groups.items():
            for worker_name, tasks in by_worker.items():
                key = "worker-kag" if "kag" in str(worker_name).lower() else "worker"
                for raw in tasks or []:
                    task = raw.get("request", raw) if isinstance(raw, dict) else raw
                    if not isinstance(task, dict):
                        continue
                    view = _extract_doc_task(task, state=state, worker_name=str(worker_name))
                    if view is not None:
                        out["workers"][key].append(view)
                        if view.get("document_id") is not None:
                            seen_doc_ids_by_worker[key].add(int(view["document_id"]))
    except Exception as exc:
        logger.warning("Workers view inspect indisponible: %s", exc, exc_info=True)
        out["inspect_error"] = str(exc)

    # Fallback DB: utile quand inspect ne remonte que l'active
    # et que la file d'attente n'apparaît pas (cas fréquent selon config Celery).
    try:
        with Session(engine) as session:
            all_docs = session.exec(select(Document)).all()
            doc_name_by_id = {
                int(d.id): (d.title or f"Document {d.id}")
                for d in all_docs
                if d.id is not None
            }
            waiting_docs = session.exec(
                select(Document).where(
                    Document.processing_status.in_(("pending", "reindex_queued", "processing"))
                )
            ).all()

        for d in waiting_docs:
            if d.id is None:
                continue
            doc_id = int(d.id)
            status = str(d.processing_status or "")
            progress = int(d.processing_progress or 0)

            # Heuristique de placement worker/queue selon l'état pipeline.
            if status in ("pending", "reindex_queued"):
                target_worker = "worker"
                queue_name = "documents"
                task_name = "db_waiting.process_library_document"
            elif status == "processing" and progress >= 95:
                target_worker = "worker-kag"
                queue_name = "kag"
                task_name = "db_waiting.process_library_document_kag"
            elif status == "processing" and progress >= 90:
                target_worker = "worker"
                queue_name = "embeddings"
                task_name = "db_waiting.process_document_embeddings"
            else:
                # Document probablement déjà réellement en cours (active), on n'ajoute pas.
                continue

            if doc_id in seen_doc_ids_by_worker[target_worker]:
                continue

            out["workers"][target_worker].append(
                {
                    "worker": "db-fallback",
                    "state": "reserved",
                    "task_id": None,
                    "task_name": task_name,
                    "queue": queue_name,
                    "document_id": doc_id,
                    "document_name": doc_name_by_id.get(doc_id, f"Document {doc_id}"),
                }
            )
            seen_doc_ids_by_worker[target_worker].add(doc_id)
    except Exception:
        logger.debug("Fallback DB workers documents ignoré", exc_info=True)

    # Enrichit aussi les items issus de Celery inspect avec le titre bibliothèque.
    try:
        missing_ids: set[int] = set()
        for key in ("worker", "worker-kag"):
            for item in out["workers"].get(key, []):
                if item.get("document_name"):
                    continue
                doc_id = item.get("document_id")
                if isinstance(doc_id, int):
                    missing_ids.add(doc_id)
        if missing_ids:
            with Session(engine) as session:
                docs = session.exec(
                    select(Document).where(Document.id.in_(missing_ids))
                ).all()
            by_id = {
                int(d.id): (d.title or f"Document {d.id}")
                for d in docs
                if d.id is not None
            }
            for key in ("worker", "worker-kag"):
                for item in out["workers"].get(key, []):
                    doc_id = item.get("document_id")
                    if isinstance(doc_id, int):
                        item["document_name"] = by_id.get(doc_id, f"Document {doc_id}")
    except Exception:
        logger.debug("Enrichissement document_name workers view ignoré", exc_info=True)

    return out


def list_stuck_processing_documents(minutes: int) -> list[dict[str, Any]]:
    """Documents en processing depuis plus de N minutes (heuristique ``updated_at``)."""
    if minutes < 1:
        minutes = 1
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    with Session(engine) as session:
        docs = session.exec(
            select(Document).where(
                Document.processing_status == "processing",
                Document.updated_at < cutoff,
            )
        ).all()
    return [
        {
            "id": d.id,
            "title": d.title,
            "processing_progress": d.processing_progress,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
        }
        for d in docs
    ]
