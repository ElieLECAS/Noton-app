from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional


class EntityAlias(SQLModel, table=True):
    """Alias normalisé pointant vers une entité canonique (expansion requête KAG)."""

    __tablename__ = "entityalias"

    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    entity_id: int = Field(foreign_key="knowledgeentity.id", index=True)
    alias_normalized: str = Field(max_length=500, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
