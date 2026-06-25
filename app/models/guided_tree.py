"""Arbres de guidage validés ("capitalisés") — Phase 2.

Les tables sont créées dès la Phase 1 (migration), mais ne sont activées
(`GuidedTree.is_active`) et exploitées qu'en Phase 2. Un arbre validé prime
sur la génération dynamique quand il correspond à la demande (priorité hybride).

Un arbre est un graphe de nœuds : chaque nœud porte une étape (instruction ou
question) et des choix `{label, value, hint, next_node_key}` qui pointent vers
le nœud suivant. Pour un how-to linéaire, un seul choix "Continuer" suffit.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, ForeignKey, Index, UniqueConstraint
from sqlmodel import Column, Field, SQLModel


class GuidedTree(SQLModel, table=True):
    """Arbre de guidage validé par les experts (diagnostic SAV ou procédure)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(max_length=120)
    title: str = Field(max_length=300)
    flow_kind: str = Field(default="diagnostic", max_length=20)  # howto | diagnostic
    space_id: Optional[int] = Field(default=None, index=True)  # null = global
    is_active: bool = Field(default=False, index=True)
    priority: int = Field(default=0)

    # Critères de correspondance avec une demande entrante
    match_keywords: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    match_categories: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    # Symptômes SAV (slugs axis=symptom) — critère fort pour les arbres diagnostic
    match_symptoms: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    root_node_key: str = Field(default="root", max_length=120)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index("uq_guidedtree_slug", "slug", unique=True),
    )


class GuidedTreeNode(SQLModel, table=True):
    """Nœud d'un arbre validé : une étape + ses choix d'aiguillage."""

    id: Optional[int] = Field(default=None, primary_key=True)
    tree_id: int = Field(
        sa_column=Column(ForeignKey("guidedtree.id", ondelete="CASCADE"), index=True)
    )
    node_key: str = Field(max_length=120)

    step_type: str = Field(default="instruction", max_length=20)
    message: str = Field(default="")
    is_terminal: bool = Field(default=False)
    termination_type: Optional[str] = Field(default=None, max_length=20)  # resolution | escalation

    # Indices de récupération documentaire pour illustrer l'étape
    retrieval_categories: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    retrieval_entities: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    step_number_hint: Optional[int] = Field(default=None)

    # Choix d'aiguillage : [{label, value, hint, next_node_key}]
    choices: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))

    __table_args__ = (
        UniqueConstraint("tree_id", "node_key", name="uq_guidedtreenode_tree_node"),
    )
