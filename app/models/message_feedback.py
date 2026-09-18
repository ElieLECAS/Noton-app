from sqlmodel import SQLModel, Field, Column, Text
from sqlalchemy import ForeignKey, UniqueConstraint
from datetime import datetime
from typing import Optional


class MessageFeedback(SQLModel, table=True):
    """Retour utilisateur sur une réponse de LIA (👍/👎, catégorie, correction)."""
    __tablename__ = "message_feedback"

    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: Optional[int] = Field(
        default=None,
        sa_column=Column(ForeignKey("message.id", ondelete="SET NULL"), index=True, nullable=True),
    )
    user_id: int = Field(foreign_key="user.id", index=True)
    is_positive: bool                         # True = 👍, False = 👎
    comment: Optional[str] = Field(default=None, sa_column=Column(Text))
    query_text: str                           # Question originale
    response_text: Optional[str] = Field(default=None, sa_column=Column(Text))
    category: Optional[str] = Field(default=None, max_length=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("message_id", "user_id", name="uq_feedback_message_user"),
    )


class FeedbackCreate(SQLModel):
    is_positive: bool
    comment: Optional[str] = None
    category: Optional[str] = None


class FeedbackRead(SQLModel):
    id: int
    message_id: Optional[int] = None
    user_id: int
    is_positive: bool
    comment: Optional[str] = None
    query_text: str
    response_text: Optional[str] = None
    category: Optional[str] = None
    created_at: datetime
    updated_at: datetime
