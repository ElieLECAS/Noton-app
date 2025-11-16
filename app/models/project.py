from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .note import Note


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(max_length=200)
    description: Optional[str] = None
    user_id: int = Field(foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Relations
    notes: List["Note"] = Relationship(back_populates="project")


class ProjectCreate(SQLModel):
    title: str
    description: Optional[str] = None


class ProjectRead(SQLModel):
    id: int
    title: str
    description: Optional[str] = None
    user_id: int
    created_at: datetime
    updated_at: datetime


class ProjectUpdate(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None

