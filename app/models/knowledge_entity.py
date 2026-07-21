"""Modèles SQLModel pour le graphe de connaissances KAG."""

from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Index
from sqlmodel import Column, Field, SQLModel

from app.embedding_config import EMBEDDING_DIMENSION


class KnowledgeEntity(SQLModel, table=True):
    """Entité normalisée au niveau espace."""

    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    name: str = Field(max_length=500)
    name_normalized: str = Field(max_length=500, index=True)
    entity_type: str = Field(max_length=100, index=True)
    # Code d'identité canonique (référence produit « 6111 », RAL « RAL:7016 », norme…).
    # Pivot de la résolution d'identité : un produit = un nœud, indépendamment du type
    # ou de la forme descriptive (« Profil 6111 » et « 6111 » partagent ref_code="6111").
    ref_code: Optional[str] = Field(default=None, max_length=64, index=True)
    description: Optional[str] = Field(default=None)
    mention_count: int = Field(default=1)
    embedding: Optional[List[float]] = Field(
        default=None,
        sa_column=Column(Vector(EMBEDDING_DIMENSION), nullable=True),
    )
    confidence_score: Optional[float] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index(
            "uq_knowledgeentity_space_name_type",
            "space_id",
            "name_normalized",
            "entity_type",
            unique=True,
        ),
        # Lookup d'identité par code (non-unique : l'unicité est garantie par le pipeline
        # d'upsert, pas par la base, pour survivre à l'état transitoire pré-retraitement).
        Index(
            "ix_knowledgeentity_space_ref_code",
            "space_id",
            "ref_code",
        ),
    )


class ChunkEntityRelation(SQLModel, table=True):
    """Lien chunk sémantique ↔ entité."""

    id: Optional[int] = Field(default=None, primary_key=True)
    chunk_id: int = Field(foreign_key="documentchunk.id", index=True)
    entity_id: int = Field(foreign_key="knowledgeentity.id", index=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    relation_role: str = Field(default="mention", max_length=32)
    relevance_score: float = Field(default=1.0)
    context_snippet: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index(
            "uq_chunkentityrelation_chunk_entity_role",
            "chunk_id",
            "entity_id",
            "relation_role",
            unique=True,
        ),
    )


class EntityAlias(SQLModel, table=True):
    """Alias / variante lexicale d'une entité."""

    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    entity_id: int = Field(foreign_key="knowledgeentity.id", index=True)
    alias_normalized: str = Field(max_length=500, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index(
            "uq_entityalias_space_alias",
            "space_id",
            "alias_normalized",
            unique=True,
        ),
    )


class EntityEntityRelation(SQLModel, table=True):
    """Relation sémantique entre deux entités (graphe inter-chunk / inter-document)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(foreign_key="space.id", index=True)
    entity_a_id: int = Field(foreign_key="knowledgeentity.id", index=True)
    entity_b_id: int = Field(foreign_key="knowledgeentity.id", index=True)
    relation_type: str = Field(default="co_occurs", max_length=64, index=True)
    relation_label: Optional[str] = Field(default=None)
    weight: float = Field(default=1.0)
    source_chunk_id: Optional[int] = Field(
        default=None,
        foreign_key="documentchunk.id",
        index=True,
    )
    confidence: Optional[float] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index(
            "uq_entityentityrelation_space_pair_type",
            "space_id",
            "entity_a_id",
            "entity_b_id",
            "relation_type",
            unique=True,
        ),
    )
