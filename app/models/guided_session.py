"""État runtime d'un parcours de guidage procédural ("aiguillage" SAV / chantier).

Une GuidedSession suit un cheminement multi-étapes attaché à une conversation :
à chaque tour le moteur récupère de la doc puis propose UNE étape + des choix,
sans donner la réponse finale. Le `path` accumule l'historique ordonné des étapes
et des observations recueillies, exploité pour l'escalade SAV en fin de parcours.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel

# Valeurs de référence (documentation ; pas de contrainte SQL pour rester souple)
FLOW_KINDS = ("howto", "diagnostic")
SOURCE_MODES = ("dynamic", "authored")
SESSION_STATUSES = ("active", "resolved", "escalated", "abandoned")


class GuidedSession(SQLModel, table=True):
    """Session de guidage procédural rattachée à une conversation."""

    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(
        sa_column=Column(ForeignKey("conversation.id", ondelete="CASCADE"), index=True)
    )
    space_id: int = Field(foreign_key="space.id", index=True)
    user_id: int = Field(foreign_key="user.id")

    topic: str = Field(default="", max_length=300)
    flow_kind: str = Field(default="howto", max_length=20)  # howto | diagnostic
    source_mode: str = Field(default="dynamic", max_length=20)  # dynamic | authored
    authored_tree_id: Optional[int] = Field(default=None, index=True)
    # Version d'arbre ÉPINGLÉE au démarrage : re-publier ne casse pas un parcours actif.
    tree_version: Optional[int] = Field(default=None)
    current_node_key: Optional[str] = Field(default=None, max_length=120)
    status: str = Field(default="active", max_length=20, index=True)
    # Réponse à « Le problème est-il résolu ? » sur une feuille résolution.
    resolved_feedback: Optional[bool] = Field(default=None)
    # Photos client : [{path, node_key, uploaded_at}]
    uploaded_files: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSONB)
    )

    # Historique ordonné des étapes (GuidedStepRecord sérialisés)
    path: List[Dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    # Signaux accumulés (catégories inférées, entités produit/composant, références)
    accumulated_signals: Dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
