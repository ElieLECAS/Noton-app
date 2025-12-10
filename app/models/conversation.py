from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .message import Message
    from .project import Project
    from .user import User


class Conversation(SQLModel, table=True):
    """Modèle pour les conversations de chat"""
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(max_length=200, default="Nouvelle conversation")
    user_id: int = Field(foreign_key="user.id")
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relations
    messages: List["Message"] = Relationship(back_populates="conversation")


class ConversationCreate(SQLModel):
    """Schéma pour créer une conversation"""
    title: Optional[str] = "Nouvelle conversation"
    project_id: Optional[int] = None


class ConversationRead(SQLModel):
    """Schéma pour lire une conversation"""
    id: int
    title: str
    user_id: int
    project_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0


class ConversationUpdate(SQLModel):
    """Schéma pour mettre à jour une conversation"""
    title: Optional[str] = None

