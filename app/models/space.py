from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation import Conversation
    from .document_space import DocumentSpace


class Space(SQLModel, table=True):
    """Espace compartimenté (ex: Interne, Client A, Client B) avec accès à certains documents."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=200)
    description: Optional[str] = None
    user_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)
    is_shared: bool = Field(default=True)
    color: Optional[str] = Field(default=None, max_length=20)
    icon: Optional[str] = Field(default=None, max_length=50)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    conversations: List["Conversation"] = Relationship(back_populates="space")
    document_associations: List["DocumentSpace"] = Relationship(back_populates="space")


class SpaceCreate(SQLModel):
    """Schéma de création d'un espace."""
    name: str
    description: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None


class SpaceRead(SQLModel):
    """Schéma de lecture pour un espace."""
    id: int
    name: str
    description: Optional[str] = None
    user_id: Optional[int]
    is_shared: bool
    color: Optional[str] = None
    icon: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SpaceUpdate(SQLModel):
    """Schéma de mise à jour d'un espace."""
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None
