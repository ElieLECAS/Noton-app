"""Modèle SQLModel pour les catégories/symptômes candidats (vocabulaire contrôlé extensible).

Pendant l'ingestion, le LLM ne classe que parmi les catégories ACTIVES (formater). En
parallèle, il peut PROPOSER des symptômes hors-liste : ceux-ci atterrissent ici avec
``status='pending'`` et un compteur d'occurrences. Tant qu'un expert ne les a pas promus
en :class:`DocumentCategory` active (``axis='symptom'``), ils n'ont AUCUN effet sur le
retrieval, le boost ou le guidage. C'est le mécanisme « le LLM propose, l'humain valide ».
"""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class CategoryCandidate(SQLModel, table=True):
    """Catégorie candidate proposée par le LLM, en attente de validation humaine."""

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(max_length=64, unique=True, index=True)
    label: str = Field(default="", max_length=200)
    axis: str = Field(default="symptom", max_length=32, index=True)
    proposed_description: str = Field(default="")
    occurrence_count: int = Field(default=1)
    status: str = Field(default="pending", max_length=20, index=True)  # pending | approved | rejected
    first_seen_document_id: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CategoryCandidateRead(SQLModel):
    id: int
    slug: str
    label: str
    axis: str
    proposed_description: str
    occurrence_count: int
    status: str
    first_seen_document_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime
