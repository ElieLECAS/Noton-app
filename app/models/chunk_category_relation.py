"""Lien chunk sémantique ↔ catégorie de contenu."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Index
from sqlmodel import Field, SQLModel


class ChunkCategoryRelation(SQLModel, table=True):
    """Relation page/chunk ↔ catégorie de contenu."""

    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(foreign_key="documentchunk.id", index=True)
    category_id: int = Field(foreign_key="documentcategory.id", index=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    document_id: int = Field(index=True)
    page_no: int = Field(default=0)
    confidence: float = Field(default=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index(
            "uq_chunkcategoryrelation_chunk_category",
            "chunk_id",
            "category_id",
            unique=True,
        ),
    )
