"""Journal des actions admin (stop-all, réindex, etc.)."""

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class AdminAuditLog(SQLModel, table=True):
    """Entrée d'audit pour les opérations sensibles (files, documents)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    action: str = Field(max_length=128, index=True)
    detail_json: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSONB, nullable=True),
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
