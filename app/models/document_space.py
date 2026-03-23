from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .document import Document
    from .space import Space


class DocumentSpace(SQLModel, table=True):
    """Association entre documents et espaces (many-to-many)."""
    __tablename__ = "document_space"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    added_at: datetime = Field(default_factory=datetime.utcnow)
    
    document: Optional["Document"] = Relationship(back_populates="space_associations")
    space: Optional["Space"] = Relationship(back_populates="document_associations")


class DocumentSpaceRead(SQLModel):
    """Schéma de lecture pour une association document-espace."""
    id: int
    document_id: int
    space_id: int
    user_id: int
    added_at: datetime
