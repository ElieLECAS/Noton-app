"""Modèle SQLModel pour les catégories de contenu documentaire."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class DocumentCategory(SQLModel, table=True):
    """Catégorie de contenu globale (enum admin-manageable)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(max_length=64, unique=True, index=True)
    label: str = Field(max_length=200)
    description: str = Field(default="")
    is_active: bool = Field(default=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentCategoryRead(SQLModel):
    id: int
    slug: str
    label: str
    description: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class DocumentCategoryCreate(SQLModel):
    slug: str
    label: str
    description: str = ""


class DocumentCategoryUpdate(SQLModel):
    label: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
