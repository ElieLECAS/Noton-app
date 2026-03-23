from sqlmodel import SQLModel, Field, Column, Text
from sqlalchemy.dialects.postgresql import JSON
from datetime import datetime
from typing import Optional, List


class ScheduledJob(SQLModel, table=True):
    """Modèle pour les tâches planifiées (scheduler)"""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    agent_id: int = Field(foreign_key="agent.id")
    task_ids: List[int] = Field(sa_column=Column(JSON))  # Liste des IDs de tâches à exécuter
    cron_expression: str = Field(max_length=100)  # Expression cron (ex: "0 18 * * *") - calculé depuis schedule_*
    schedule_hour: int = Field(ge=0, le=23)  # Heure (0-23)
    schedule_minute: int = Field(ge=0, le=59)  # Minute (0-59)
    schedule_days: List[int] = Field(sa_column=Column(JSON))  # Jours (0-6, 0=Lundi)
    enabled: bool = Field(default=True)
    last_run_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


def cron_from_schedule(hour: int, minute: int, days: List[int]) -> str:
    """Construire une expression cron depuis heure, minute et jours"""
    if not days:
        raise ValueError("Au moins un jour doit être sélectionné")
    days_str = ','.join(str(d) for d in sorted(days))
    return f"{minute} {hour} * * {days_str}"


class ScheduledJobCreate(SQLModel):
    """Schéma pour créer un job planifié"""
    agent_id: int
    task_ids: List[int]
    schedule_hour: int = Field(ge=0, le=23)
    schedule_minute: int = Field(ge=0, le=59)
    schedule_days: List[int]  # Liste de 0-6 (0=Lundi)
    enabled: bool = True


class ScheduledJobRead(SQLModel):
    """Schéma pour lire un job planifié"""
    id: int
    user_id: int
    agent_id: int
    task_ids: List[int]
    cron_expression: str
    schedule_hour: int
    schedule_minute: int
    schedule_days: List[int]
    enabled: bool
    last_run_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    agent_name: Optional[str] = None  # Enrichi par la requête


class ScheduledJobUpdate(SQLModel):
    """Schéma pour mettre à jour un job planifié"""
    agent_id: Optional[int] = None
    task_ids: Optional[List[int]] = None
    schedule_hour: Optional[int] = Field(default=None, ge=0, le=23)
    schedule_minute: Optional[int] = Field(default=None, ge=0, le=59)
    schedule_days: Optional[List[int]] = None
    enabled: Optional[bool] = None
