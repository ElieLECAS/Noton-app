"""Snapshots publiés des arbres SAV.

Le RUNTIME ne lit que ces snapshots : les sessions en cours épinglent (tree_id, version),
donc re-publier un arbre ne casse jamais un parcours actif. Le snapshot JSONB contient
la méta de l'arbre + tous ses nœuds + leurs pièces jointes (forme produite par
guided_authoring_service.build_snapshot).
"""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class GuidedTreeVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tree_id: int = Field(
        sa_column=Column(ForeignKey("guidedtree.id", ondelete="CASCADE"), index=True)
    )
    version: int = Field(default=1)
    snapshot: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSONB, nullable=False))
    note: str = Field(default="", max_length=300)
    published_by: Optional[int] = Field(default=None)
    published_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("tree_id", "version", name="uq_guidedtreeversion_tree_version"),
    )
