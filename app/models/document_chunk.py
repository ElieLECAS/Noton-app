from sqlmodel import SQLModel, Field, Relationship, Column
from typing import Optional, List, TYPE_CHECKING
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from app.embedding_config import EMBEDDING_DIMENSION

if TYPE_CHECKING:
    from .document import Document


class DocumentChunk(SQLModel, table=True):
    """Chunk d'un document pour la recherche sémantique RAG."""
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    chunk_index: int = Field(default=0)
    content: str
    text: Optional[str] = None
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(EMBEDDING_DIMENSION), nullable=True))
    start_char: int = Field(default=0)
    end_char: int = Field(default=0)
    node_id: Optional[str] = Field(default=None, index=True)
    parent_node_id: Optional[str] = Field(default=None, index=True)
    is_leaf: bool = Field(default=True, index=True)
    hierarchy_level: int = Field(default=0, index=True)
    metadata_json: Optional[dict] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    metadata_: Optional[dict] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    source: Optional[str] = Field(default=None, index=True)  # Brand/Origin denormalized for speed
    
    document: Optional["Document"] = Relationship(back_populates="chunks")


class DocumentChunkRead(SQLModel):
    """Modèle de lecture pour un chunk."""
    id: int
    document_id: int
    chunk_index: int
    content: str
    start_char: int
    end_char: int
    node_id: Optional[str] = None
    parent_node_id: Optional[str] = None
    is_leaf: bool = True
    hierarchy_level: int = 0
