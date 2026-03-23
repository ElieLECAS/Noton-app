from sqlmodel import SQLModel, Field, Column
from datetime import datetime
from typing import Optional, List
from pgvector.sqlalchemy import Vector
from app.embedding_config import EMBEDDING_DIMENSION


class KnowledgeEntity(SQLModel, table=True):
    """
    Entité de connaissance extraite des chunks pour le système KAG.
    
    Représente un concept, équipement, procédure, paramètre, etc.
    mentionné dans les documents d'un projet.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    name_normalized: str = Field(index=True)
    entity_type: str = Field(index=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    mention_count: int = Field(default=1)
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(EMBEDDING_DIMENSION), nullable=True)
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeEntityRead(SQLModel):
    """Modèle de lecture pour une entité de connaissance."""
    id: int
    name: str
    name_normalized: str
    entity_type: str
    project_id: int
    mention_count: int
    created_at: datetime
    updated_at: datetime
