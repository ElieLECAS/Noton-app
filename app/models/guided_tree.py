"""Arbres SAV édités par l'équipe (refonte 2026-07-30).

Un arbre est un graphe de nœuds édité dans le builder admin : chaque nœud porte une
étape (question / instruction / feuille) et des choix `{label, value, hint, next_node_key}`.
Le cycle de vie est draft → published (snapshot dans GuidedTreeVersion) → archived.
Le RUNTIME ne lit que les snapshots publiés ; les lignes de cette table sont le brouillon.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, ForeignKey, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

TREE_STATUSES = ("draft", "published", "archived")
# step_type élargi : link = renvoi vers un nœud d'un autre arbre (ou du même).
NODE_STEP_TYPES = ("question", "instruction", "diagnosis", "diagnostic", "resolution", "escalation", "link")


class GuidedTree(SQLModel, table=True):
    """Arbre SAV (diagnostic) — brouillon éditable ; le runtime lit GuidedTreeVersion."""

    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(max_length=120)
    title: str = Field(max_length=300)
    flow_kind: str = Field(default="diagnostic", max_length=20)  # howto | diagnostic
    space_id: Optional[int] = Field(default=None, index=True)  # null = global
    status: str = Field(default="draft", max_length=20, index=True)
    priority: int = Field(default=0)

    # Point d'entrée : symptôme (slug axis=symptom) affiché dans le picker.
    entry_symptom: Optional[str] = Field(default=None, max_length=120)
    description: str = Field(default="")
    # Filtres à plat {materials, product_types, source, proferm_gammes} — mêmes axes
    # que Document : éligibilité + pré-filtre de l'index d'entrée sémantique.
    perimeter: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    # Positions libres des cas dans le graphe d'édition, normalisées 0..1 :
    # {node_key: {"nx": .., "ny": ..}}. Purement visuel (l'ordre logique reste porté
    # par les choices) ; absent → placement automatique par niveaux.
    layout: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    current_version: int = Field(default=0)  # 0 = jamais publié
    created_by: Optional[int] = Field(default=None)
    updated_by: Optional[int] = Field(default=None)

    # Anciens critères de matching (dépréciés — remplacés par guidedentryindex ;
    # colonnes droppées par la migration guided_cleanup différée).
    match_keywords: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    match_categories: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    match_symptoms: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    root_node_key: str = Field(default="root", max_length=120)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    __table_args__ = (
        Index("uq_guidedtree_slug", "slug", unique=True),
    )


class GuidedTreeNode(SQLModel, table=True):
    """Nœud d'un arbre SAV (brouillon) : une étape + ses choix d'aiguillage."""

    id: Optional[int] = Field(default=None, primary_key=True)
    tree_id: int = Field(
        sa_column=Column(ForeignKey("guidedtree.id", ondelete="CASCADE"), index=True)
    )
    node_key: str = Field(max_length=120)

    step_type: str = Field(default="question", max_length=20)
    # Libellé court (outline du builder + fil d'Ariane du chat).
    title: str = Field(default="", max_length=200)
    # Message complet montré au client.
    message: str = Field(default="")
    # Note interne équipe — jamais montrée au client.
    internal_note: str = Field(default="")
    is_terminal: bool = Field(default=False)
    termination_type: Optional[str] = Field(default=None, max_length=20)  # resolution | escalation

    # Comportement du nœud côté client.
    ask_photo: bool = Field(default=False)
    allow_free_text: bool = Field(default=True)
    # Visible uniquement si le périmètre de la session est compatible (mêmes axes que l'arbre).
    perimeter_condition: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSONB))
    tools_hint: str = Field(default="", max_length=200)

    # Dépréciés (plus de retrieval par étape) — droppés par guided_cleanup.
    retrieval_categories: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    retrieval_entities: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    step_number_hint: Optional[int] = Field(default=None)

    # Choix d'aiguillage : [{label, value, hint, next_node_key}]
    choices: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))

    __table_args__ = (
        UniqueConstraint("tree_id", "node_key", name="uq_guidedtreenode_tree_node"),
    )
