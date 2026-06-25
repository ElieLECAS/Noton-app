"""Modèle SQLModel pour les catégories de contenu documentaire."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class DocumentCategory(SQLModel, table=True):
    """Catégorie de contenu globale (enum admin-manageable).

    La colonne ``axis`` discrimine la facette de la taxonomie : ``task`` (les 16
    catégories historiques pilotant les flows guidés et le boost), ``doc_type``
    (nature du document), ``lifecycle_phase`` (avant-vente/chantier/SAV) et
    ``symptom`` (vocabulaire diagnostic SAV, extensible). Tout le code lisant par
    ``slug`` reste transparent à l'axe ; ``axis`` ne sert qu'au filtrage/pondération.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(max_length=64, unique=True, index=True)
    label: str = Field(max_length=200)
    description: str = Field(default="")
    axis: str = Field(default="task", max_length=32, index=True)
    parent_slug: Optional[str] = Field(default=None, max_length=64)
    is_active: bool = Field(default=True, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentCategoryRead(SQLModel):
    id: int
    slug: str
    label: str
    description: str
    axis: str
    parent_slug: Optional[str] = None
    is_active: bool
    created_at: datetime
    updated_at: datetime


class DocumentCategoryCreate(SQLModel):
    slug: str
    label: str
    description: str = ""
    axis: str = "task"
    parent_slug: Optional[str] = None


class DocumentCategoryUpdate(SQLModel):
    label: Optional[str] = None
    description: Optional[str] = None
    axis: Optional[str] = None
    parent_slug: Optional[str] = None
    is_active: Optional[bool] = None
