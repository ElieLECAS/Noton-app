from sqlmodel import SQLModel, Field, Column, Text
from datetime import datetime
from typing import Optional


class TaskRunLog(SQLModel, table=True):
    """Modèle pour l'historique d'exécution des tâches planifiées"""
    id: Optional[int] = Field(default=None, primary_key=True)
    scheduled_job_id: int = Field(foreign_key="scheduledjob.id")
    agent_task_id: int = Field(foreign_key="agenttask.id")
    task_name: str = Field(max_length=200)  # Dénormalisé pour affichage rapide
    output: Optional[str] = Field(default=None, sa_column=Column(Text))
    error: Optional[str] = Field(default=None, sa_column=Column(Text))
    run_at: datetime = Field(default_factory=datetime.utcnow)
    user_id: int = Field(foreign_key="user.id")


class TaskRunLogRead(SQLModel):
    """Schéma pour lire un log d'exécution"""
    id: int
    scheduled_job_id: int
    agent_task_id: int
    task_name: str
    output: Optional[str] = None
    error: Optional[str] = None
    run_at: datetime
