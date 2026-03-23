from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .library import Library
    from .document import Document


class Folder(SQLModel, table=True):
    """Dossier pour organiser les documents dans une arborescence hiérarchique."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200)
    parent_folder_id: Optional[int] = Field(default=None, foreign_key="folder.id", index=True)
    library_id: int = Field(foreign_key="library.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    library: Optional["Library"] = Relationship(back_populates="folders")
    parent_folder: Optional["Folder"] = Relationship(
        back_populates="subfolders",
        sa_relationship_kwargs={"remote_side": "Folder.id"}
    )
    subfolders: List["Folder"] = Relationship(back_populates="parent_folder")
    documents: List["Document"] = Relationship(back_populates="folder")


class FolderCreate(SQLModel):
    """Schéma de création d'un dossier."""
    name: str
    parent_folder_id: Optional[int] = None


class FolderRead(SQLModel):
    """Schéma de lecture pour un dossier."""
    id: int
    name: str
    parent_folder_id: Optional[int]
    library_id: int
    user_id: int
    created_at: datetime
    updated_at: datetime


class FolderUpdate(SQLModel):
    """Schéma de mise à jour d'un dossier."""
    name: Optional[str] = None


class FolderWithContents(FolderRead):
    """Dossier avec ses sous-dossiers et documents."""
    subfolders: List[FolderRead] = []
    document_count: int = 0
