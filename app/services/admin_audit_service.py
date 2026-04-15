"""Enregistrement du journal d'audit admin."""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlmodel import Session

from app.models.admin_audit_log import AdminAuditLog

logger = logging.getLogger(__name__)


def log_admin_action(
    *,
    user_id: int,
    action: str,
    detail: Optional[dict[str, Any]] = None,
) -> None:
    from app.database import engine

    try:
        with Session(engine) as session:
            row = AdminAuditLog(user_id=user_id, action=action, detail_json=detail or {})
            session.add(row)
            session.commit()
    except Exception:
        logger.warning(
            "Impossible d'enregistrer l'audit action=%s user_id=%s",
            action,
            user_id,
            exc_info=True,
        )
