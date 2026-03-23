from sqlmodel import SQLModel, Field, Relationship, Column
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from pgvector.sqlalchemy import Vector
from app.embedding_config import EMBEDDING_DIMENSION

if TYPE_CHECKING:
    from .library import Library
    from .folder import Folder
    from .document_chunk import DocumentChunk
    from .document_space import DocumentSpace


class Document(SQLModel, table=True):
    """Document dans la bibliothèque générale, accessible dans plusieurs espaces."""
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(max_length=200)
    content: Optional[str] = None
    document_type: str = Field(default="written")
    source_file_path: Optional[str] = None
    processing_status: str = Field(default="completed")
    processing_progress: Optional[int] = Field(default=100)
    folder_id: Optional[int] = Field(default=None, foreign_key="folder.id", index=True)
    library_id: int = Field(foreign_key="library.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = Field(default=None, sa_column=Column(Vector(EMBEDDING_DIMENSION), nullable=True))
    
    library: Optional["Library"] = Relationship(back_populates="documents")
    folder: Optional["Folder"] = Relationship(back_populates="documents")
    chunks: List["DocumentChunk"] = Relationship(back_populates="document")
    space_associations: List["DocumentSpace"] = Relationship(back_populates="document")


class DocumentCreate(SQLModel):
    """Schéma de création d'un document."""
    title: str
    content: Optional[str] = None
    document_type: str = "written"
    source_file_path: Optional[str] = None
    processing_status: str = "completed"
    processing_progress: Optional[int] = 100
    folder_id: Optional[int] = None


class DocumentRead(SQLModel):
    """Schéma de lecture pour un document."""
    id: int
    title: str
    content: Optional[str] = None
    document_type: str
    processing_status: str
    processing_progress: Optional[int] = None
    folder_id: Optional[int] = None
    library_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class DocumentListItem(SQLModel):
    """Schéma pour la liste des documents (sans contenu)."""
    id: int
    title: str
    document_type: str
    processing_status: str
    processing_progress: Optional[int] = None
    folder_id: Optional[int] = None
    library_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class DocumentUpdate(SQLModel):
    """Schéma de mise à jour d'un document."""
    title: Optional[str] = None
    content: Optional[str] = None
    processing_progress: Optional[int] = None
    folder_id: Optional[int] = None
