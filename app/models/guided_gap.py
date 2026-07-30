"""Angles morts du SAV guidé : symptômes/questions sans arbre publié.

Alimenté au fil de l'eau par le chat (symptôme détecté sans correspondance dans
l'index d'entrée). C'est le backlog qui dit QUELS arbres écrire en priorité —
trié par fréquence, avec les vraies questions posées en exemples.
"""

from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

GAP_STATUSES = ("open", "planned", "covered", "dismissed")


class GuidedGap(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(
        sa_column=Column(ForeignKey("space.id", ondelete="CASCADE"), index=True)
    )
    detected_symptom: Optional[str] = Field(default=None, max_length=120, index=True)
    query_text: str = Field(default="")
    count: int = Field(default=1)
    status: str = Field(default="open", max_length=20, index=True)
    sample_conversation_ids: Optional[List[int]] = Field(default=None, sa_column=Column(JSONB))
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
