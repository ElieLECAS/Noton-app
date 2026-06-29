"""Lien chunk sémantique ↔ catégorie de contenu."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Index
from sqlmodel import Field, SQLModel


class ChunkCategoryRelation(SQLModel, table=True):
    """Relation chunk ↔ catégorie de contenu, AU NIVEAU DOCUMENT (indépendante de l'espace).

    La catégorisation découle du contenu (KAG / enrichissement), identique quel que soit
    l'espace : on ne stocke donc qu'une ligne par (chunk, catégorie). Le scope par espace
    est dérivé à la requête via l'appartenance ``DocumentSpace`` (document ↔ espace), si bien
    qu'ajouter un document à un espace y fait apparaître ses catégories dynamiquement.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(foreign_key="documentchunk.id", index=True)
    category_id: int = Field(foreign_key="documentcategory.id", index=True)
    document_id: int = Field(index=True)
    page_no: int = Field(default=0)
    confidence: float = Field(default=1.0)
    # Catégorie dominante du chunk (axe task/symptom) : 1 primaire par chunk au plus.
    is_primary: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index(
            "uq_chunkcategoryrelation_chunk_category",
            "chunk_id",
            "category_id",
            unique=True,
        ),
    )
