"""Identifiant de run de traitement document : invalidation au stop, contrôle des workers."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from sqlmodel import Session

from app.database import engine
from app.models.document import Document

logger = logging.getLogger(__name__)


def new_processing_run_id() -> str:
    return str(uuid.uuid4())


def refresh_document_processing_run_id(document: Document) -> str:
    """Nouvel identifiant de run (upload, réindex, ou invalidation après stop)."""
    rid = new_processing_run_id()
    document.processing_run_id = rid
    return rid


def is_processing_run_current(document_id: int, expected_run_id: Optional[str]) -> bool:
    """
    True si le run attendu est encore celui en base (ou pas de contrainte ancienne tâche).
    expected_run_id None = tâche avant migration : on accepte si le doc n'a pas encore de run_id.
    """
    if expected_run_id is None:
        try:
            with Session(engine) as session:
                doc = session.get(Document, document_id)
                if doc is None:
                    return False
                # Pas de run_id en base : pipeline historique, on laisse passer.
                if doc.processing_run_id is None:
                    return True
                # Document déjà migré avec run_id mais tâche sans argument : considérer comme obsolète.
                return False
        except Exception:
            logger.debug(
                "is_processing_run_current document_id=%s fallback True",
                document_id,
                exc_info=True,
            )
            return True

    try:
        with Session(engine) as session:
            doc = session.get(Document, document_id)
            if doc is None:
                return False
            return doc.processing_run_id == expected_run_id
    except Exception:
        logger.warning(
            "is_processing_run_current erreur document_id=%s", document_id, exc_info=True
        )
        return False
