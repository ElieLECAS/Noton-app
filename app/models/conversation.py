from sqlmodel import SQLModel, Field, Relationship, Column
from sqlalchemy import JSON
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .message import Message


class Conversation(SQLModel, table=True):
    """Une conversation avec LIA. Un seul corpus (le wiki) : plus d'espace."""
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(max_length=200, default="Nouvelle conversation")
    user_id: int = Field(foreign_key="user.id")
    query_context: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    messages: List["Message"] = Relationship(back_populates="conversation")


class ConversationCreate(SQLModel):
    title: Optional[str] = "Nouvelle conversation"


class ConversationRead(SQLModel):
    id: int
    title: str
    user_id: int
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0


class ConversationUpdate(SQLModel):
    title: Optional[str] = None
