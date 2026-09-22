from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .message import Message


MODES = ("chat", "vocal")


class Conversation(SQLModel, table=True):
    """Une conversation avec LIA. Un seul corpus (le wiki) : pas d'espace, pas de contexte
    de question mémorisé — l'historique des messages suffit.

    ``mode`` distingue les deux écrans : ``chat`` (l'accueil) et ``vocal`` (l'assistant vocal,
    dont les échanges sont transcrits et persistés comme les autres, mais listés à part)."""
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(max_length=200, default="Nouvelle conversation")
    user_id: int = Field(foreign_key="user.id")
    mode: str = Field(default="chat", max_length=20)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    messages: List["Message"] = Relationship(back_populates="conversation")


class ConversationCreate(SQLModel):
    title: Optional[str] = "Nouvelle conversation"
    mode: Optional[str] = "chat"


class ConversationRead(SQLModel):
    id: int
    title: str
    user_id: int
    mode: str = "chat"
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0


class ConversationUpdate(SQLModel):
    title: Optional[str] = None
