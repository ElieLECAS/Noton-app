"""Cache des synthèses de carte mentale (CAG) par nœud d'arbre thématique.

Une synthèse est coûteuse (gros contexte injecté) : on la mémorise par
``(space_id, node_key, axis)`` et on l'invalide via ``content_hash`` (empreinte de
l'ensemble des chunks ayant alimenté la synthèse). Si l'ensemble change, on régénère.
"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Column, Index, JSON
from sqlmodel import Field, SQLModel


class SpaceThemeSynthesis(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: int = Field(index=True)
    node_key: str = Field(max_length=128, index=True)
    axis: str = Field(default="task", max_length=32)
    content_hash: str = Field(default="", max_length=64)
    node_label: str = Field(default="", max_length=300)
    synthesis_markdown: str = Field(default="")
    sources_json: List = Field(default_factory=list, sa_column=Column(JSON))
    chunk_count: int = Field(default=0)
    truncated: bool = Field(default=False)
    model: str = Field(default="", max_length=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index(
            "uq_space_theme_synthesis_node",
            "space_id",
            "node_key",
            "axis",
            unique=True,
        ),
    )
