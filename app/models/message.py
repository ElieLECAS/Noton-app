from sqlmodel import SQLModel, Field, Relationship, Column, Text
from sqlalchemy import ForeignKey, JSON
from datetime import datetime
from typing import Optional, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from .conversation import Conversation


class Message(SQLModel, table=True):
    """Modèle pour les messages dans les conversations"""
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(sa_column=Column(ForeignKey("conversation.id", ondelete="CASCADE")))
    role: str = Field(max_length=50)  # "user", "assistant", "system"
    content: str = Field(sa_column=Column(Text))  # Contenu du message (peut être long)
    model: Optional[str] = Field(default=None, max_length=100)  # Modèle utilisé (pour les réponses assistant)
    provider: Optional[str] = Field(default=None, max_length=50)  # Provider (mistral/openai)
    sources: Optional[str] = Field(default=None, sa_column=Column(Text))  # Sources stockées sous format JSON string
    metadata_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON, nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relations
    conversation: "Conversation" = Relationship(back_populates="messages")


class MessageCreate(SQLModel):
    """Schéma pour créer un message"""
    conversation_id: int
    role: str
    content: str
    model: Optional[str] = None
    provider: Optional[str] = None
    sources: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None


class MessageRead(SQLModel):
    """Schéma pour lire un message"""
    id: int
    conversation_id: int
    role: str
    content: str
    model: Optional[str] = None
    provider: Optional[str] = None
    sources: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None
    created_at: datetime

