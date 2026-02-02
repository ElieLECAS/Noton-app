from sqlmodel import SQLModel, Field, Relationship, Column, Text
from pydantic import ConfigDict
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_task import AgentTask


class Agent(SQLModel, table=True):
    """Modèle pour les agents IA personnalisés"""
    model_config = ConfigDict(protected_namespaces=())
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    name: str = Field(max_length=200)
    personality: str = Field(sa_column=Column(Text))  # System prompt / personnalité de l'agent
    model_preset: Optional[str] = Field(default=None, max_length=50)  # "private", "fast", "powerful"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relations
    tasks: List["AgentTask"] = Relationship(back_populates="agent")


class AgentCreate(SQLModel):
    """Schéma pour créer un agent"""
    model_config = ConfigDict(protected_namespaces=())
    name: str
    personality: str
    model_preset: Optional[str] = None


class AgentRead(SQLModel):
    """Schéma pour lire un agent"""
    model_config = ConfigDict(protected_namespaces=())
    id: int
    user_id: int
    name: str
    personality: str
    model_preset: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    task_count: Optional[int] = 0


class AgentUpdate(SQLModel):
    """Schéma pour mettre à jour un agent"""
    model_config = ConfigDict(protected_namespaces=())
    name: Optional[str] = None
    personality: Optional[str] = None
    model_preset: Optional[str] = None
