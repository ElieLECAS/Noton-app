from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional


class ChunkEntityRelation(SQLModel, table=True):
    """
    Relation entre un chunk et une entité de connaissance.
    
    Représente le fait qu'un chunk mentionne une entité,
    avec un score de pertinence (importance retournée par le LLM).
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(foreign_key="notechunk.id", index=True)
    entity_id: int = Field(foreign_key="knowledgeentity.id", index=True)
    relevance_score: float = Field(default=1.0)
    project_id: int = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChunkEntityRelationRead(SQLModel):
    """Modèle de lecture pour une relation chunk-entité."""
    id: int
    chunk_id: int
    entity_id: int
    relevance_score: float
    project_id: int
    created_at: datetime
