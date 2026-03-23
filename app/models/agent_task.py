from sqlmodel import SQLModel, Field, Relationship, Column, Text
from sqlalchemy import ForeignKey
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent


class AgentTask(SQLModel, table=True):
    """Modèle pour les tâches assignées à un agent"""
    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: int = Field(sa_column=Column(ForeignKey("agent.id", ondelete="CASCADE")))
    name: str = Field(max_length=200)
    instruction: str = Field(sa_column=Column(Text))  # Prompt ou objectif de la tâche
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relations
    agent: "Agent" = Relationship(back_populates="tasks")


class AgentTaskCreate(SQLModel):
    """Schéma pour créer une tâche"""
    name: str
    instruction: str


class AgentTaskRead(SQLModel):
    """Schéma pour lire une tâche"""
    id: int
    agent_id: int
    name: str
    instruction: str
    created_at: datetime


class AgentTaskUpdate(SQLModel):
    """Schéma pour mettre à jour une tâche"""
    name: Optional[str] = None
    instruction: Optional[str] = None
