from sqlmodel import SQLModel, Field, Column
from datetime import datetime
from typing import Optional, List
from pgvector.sqlalchemy import Vector
from app.embedding_config import EMBEDDING_DIMENSION


class KnowledgeEntity(SQLModel, table=True):
    """
    Entité de connaissance extraite des chunks pour le système KAG.
    
    Représente un concept, équipement, procédure, paramètre, etc.
    mentionné dans les documents d'un espace.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    name_normalized: str = Field(index=True)
    entity_type: str = Field(index=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    mention_count: int = Field(default=1)
    # Score calibré (importance moyenne × renforcement mentions) pour filtrer retrieval
    confidence_score: Optional[float] = Field(default=None, index=True)
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
    space_id: int
    mention_count: int
    confidence_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime
