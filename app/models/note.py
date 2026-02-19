from sqlmodel import SQLModel, Field, Relationship, Column
from datetime import datetime
from typing import Optional, TYPE_CHECKING, List
from pgvector.sqlalchemy import Vector
from app.embedding_config import EMBEDDING_DIMENSION

if TYPE_CHECKING:
    from .project import Project
    from .note_chunk import NoteChunk


class Note(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(max_length=200)
    content: Optional[str] = None
    note_type: str = Field(default="written")  # 'written', 'voice', ou 'document'
    source_file_path: Optional[str] = None  # Chemin du fichier original si document uploadé
    processing_status: str = Field(default="completed")  # 'pending', 'processing', 'completed', 'failed'
    processing_progress: Optional[int] = Field(default=100)  # Progression de traitement document (0-100)
    project_id: int = Field(foreign_key="project.id")
    user_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(EMBEDDING_DIMENSION), nullable=True))
    
    # Relations
    project: Optional["Project"] = Relationship(back_populates="notes")
    chunks: List["NoteChunk"] = Relationship(back_populates="note")


class NoteCreate(SQLModel):
    title: str
    content: Optional[str] = None
    note_type: str = "written"
    source_file_path: Optional[str] = None
    processing_status: str = "completed"
    processing_progress: Optional[int] = 100


class NoteRead(SQLModel):
    id: int
    title: str
    content: Optional[str] = None
    note_type: str
    processing_status: str
    processing_progress: Optional[int] = None
    project_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class NoteUpdate(SQLModel):
    title: Optional[str] = None
    content: Optional[str] = None
    processing_progress: Optional[int] = None

