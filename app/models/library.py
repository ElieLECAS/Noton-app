from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .folder import Folder
    from .document import Document


class Library(SQLModel, table=True):
    """Bibliothèque générale d'un utilisateur contenant tous ses documents organisés en dossiers."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200, default="Ma Bibliothèque")
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    is_global: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    folders: List["Folder"] = Relationship(back_populates="library")
    documents: List["Document"] = Relationship(back_populates="library")


class LibraryRead(SQLModel):
    """Schéma de lecture pour une bibliothèque."""
    id: int
    name: str
    user_id: Optional[int]
    is_global: bool
    created_at: datetime
    updated_at: datetime


class LibraryStats(SQLModel):
    """Statistiques d'une bibliothèque."""
    library_id: int
    total_documents: int
    total_folders: int
    total_size_mb: float
