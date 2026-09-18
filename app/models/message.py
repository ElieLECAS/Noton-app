from sqlmodel import SQLModel, Field, Relationship, Column, Text
from sqlalchemy import ForeignKey, JSON
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .conversation import Conversation


class Message(SQLModel, table=True):
    """Un message d'une conversation : la question de l'utilisateur ou la réponse de LIA.

    Les messages ne sont écrits que par le tour de chat (``/api/chat/stream``) : la réponse
    porte le modèle Mistral qui l'a produite, les pages du wiki citées (``sources``, JSON) et
    la trace du tour (``metadata_json`` : tokens, cache, latence, anomalies)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(sa_column=Column(ForeignKey("conversation.id", ondelete="CASCADE")))
    role: str = Field(max_length=50)  # "user" | "assistant"
    content: str = Field(sa_column=Column(Text))
    model: Optional[str] = Field(default=None, max_length=100)
    sources: Optional[str] = Field(default=None, sa_column=Column(Text))
    metadata_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    conversation: "Conversation" = Relationship(back_populates="messages")


class MessageRead(SQLModel):
    id: int
    conversation_id: int
    role: str
    content: str
    model: Optional[str] = None
    sources: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime
