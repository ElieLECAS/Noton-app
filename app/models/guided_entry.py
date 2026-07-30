"""Index d'entrée sémantique des arbres SAV (niveau 0) + alias de symptômes.

GuidedEntryIndex est une table à plat reconstruite À LA PUBLICATION d'un arbre (et à
chaque modification du registre de symptômes) : une ligne par symptôme / arbre / nœud,
portant le texte embeddé (mistral-embed), et les filtres à plat (matériau / famille /
fournisseur / gamme — mêmes axes que Document). La recherche d'entrée = pré-filtre de
périmètre puis cosinus pgvector. On cherche dans l'ARBRE, jamais dans la doc.
"""

from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

from app.embedding_config import EMBEDDING_DIMENSION

ENTRY_KINDS = ("symptom", "tree", "node")


class GuidedSymptomAlias(SQLModel, table=True):
    """Vocabulaire client d'un symptôme (« ça frotte », « coince », « dur à fermer »)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    symptom_slug: str = Field(max_length=120, index=True)
    alias: str = Field(max_length=200)

    __table_args__ = (
        UniqueConstraint("symptom_slug", "alias", name="uq_guidedsymptomalias_slug_alias"),
    )


class GuidedEntryIndex(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    space_id: Optional[int] = Field(default=None, index=True)  # null = global
    tree_id: Optional[int] = Field(
        default=None,
        sa_column=Column(ForeignKey("guidedtree.id", ondelete="CASCADE"), index=True),
    )
    tree_version: Optional[int] = Field(default=None)
    entry_kind: str = Field(max_length=20)  # symptom | tree | node
    ref_key: str = Field(max_length=160)  # slug symptôme / slug arbre / node_key
    label: str = Field(default="", max_length=300)
    # Texte embeddé (label + message + alias) — conservé pour le matching lexical.
    text: str = Field(default="")
    # {materials, product_types, source, proferm_gammes} hérités de l'arbre,
    # surchargés par nœud (perimeter_condition).
    filters: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    embedding: Optional[List[float]] = Field(
        default=None, sa_column=Column(Vector(EMBEDDING_DIMENSION), nullable=True)
    )
