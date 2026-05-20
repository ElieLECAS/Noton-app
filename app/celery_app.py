"""
Application Celery : broker Redis, backend résultats Redis.
Les tâches sont définies dans app.tasks.
"""
import os

from celery import Celery


def _broker_url() -> str:
    try:
        from app.config import settings

        u = getattr(settings, "CELERY_BROKER_URL", None) or getattr(
            settings, "REDIS_URL", None
        )
        if u:
            return str(u).strip()
    except Exception:
        pass
    return (
        os.getenv("CELERY_BROKER_URL")
        or os.getenv("REDIS_URL")
        or "redis://localhost:6379/0"
    )


def _result_backend() -> str:
    try:
        from app.config import settings

        b = getattr(settings, "CELERY_RESULT_BACKEND", None)
        if b:
            return str(b).strip()
    except Exception:
        pass
    return os.getenv("CELERY_RESULT_BACKEND") or _broker_url()


celery_app = Celery(
    "noton",
    broker=_broker_url(),
    backend=_result_backend(),
    include=["app.tasks.documents"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_default_queue="celery",
    task_queues={
        "documents": {
            "exchange": "documents",
            "routing_key": "documents",
        },
        "embeddings": {
            "exchange": "embeddings",
            "routing_key": "embeddings",
        },
    },
    broker_connection_retry_on_startup=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# Import explicite : enregistre toutes les @celery_app.task sur l’instance (évite « unregistered task »
# si le worker n’a pas rechargé le module après ajout d’une tâche — redémarrer le worker reste obligatoire).
import app.tasks.documents  # noqa: F401
