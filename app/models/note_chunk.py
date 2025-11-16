from sqlmodel import SQLModel, Field, Relationship, Column
from typing import Optional, List, TYPE_CHECKING
from pgvector.sqlalchemy import Vector
from app.embedding_config import EMBEDDING_DIMENSION

if TYPE_CHECKING:
    from .note import Note


class NoteChunk(SQLModel, table=True):
    """Chunk d'une note pour la recherche sémantique RAG"""
    id: Optional[int] = Field(default=None, primary_key=True)
    note_id: int = Field(foreign_key="note.id", index=True)
    chunk_index: int = Field(default=0)  # Position du chunk dans la note (0, 1, 2...)
    content: str  # Le texte du chunk
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(EMBEDDING_DIMENSION), nullable=True))
    start_char: int = Field(default=0)  # Position de début dans la note originale
    end_char: int = Field(default=0)  # Position de fin dans la note originale
    
    # Relations
    note: Optional["Note"] = Relationship(back_populates="chunks")


class NoteChunkRead(SQLModel):
    """Modèle de lecture pour un chunk"""
    id: int
    note_id: int
    chunk_index: int
    content: str
    start_char: int
    end_char: int

