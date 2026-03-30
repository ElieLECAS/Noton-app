from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional


class EntityEntityRelation(SQLModel, table=True):
    """Relation entre deux entités dans un espace (graphe KAG)."""

    __tablename__ = "entityentityrelation"

    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    entity_a_id: int = Field(foreign_key="knowledgeentity.id", index=True)
    entity_b_id: int = Field(foreign_key="knowledgeentity.id", index=True)
    relation_type: str = Field(default="co_occurs", max_length=64, index=True)
    weight: float = Field(default=1.0)
    source_chunk_id: Optional[int] = Field(
        default=None,
        foreign_key="documentchunk.id",
        index=True,
    )
    confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
