"""Lecture et traversée des arbres SAV PUBLIÉS (snapshots).

Le runtime ne lit jamais les lignes draft : il charge le snapshot épinglé par la
session (GuidedTreeVersion) et le traverse de façon purement déterministe — zéro
appel LLM, zéro retrieval. Les pièces jointes d'auteur remplacent les passages.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from app.models.guided_tree import GuidedTree
from app.models.guided_tree_version import GuidedTreeVersion

logger = logging.getLogger(__name__)

# Valeurs réservées des actions de parcours (préfixe __ pour ne jamais collisionner
# avec les valeurs de choix d'auteur).
BACK_VALUE = "__back"
QUIT_VALUE = "__quit"
FEEDBACK_YES_VALUE = "__feedback_yes"
FEEDBACK_NO_VALUE = "__feedback_no"


def load_published_snapshot(
    session: Session, tree_id: int, version: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Charge le snapshot d'une version publiée (par défaut : la version courante)."""
    if version is None:
        tree = session.get(GuidedTree, tree_id)
        if tree is None or not tree.current_version:
            return None
        version = tree.current_version
    row = session.exec(
        select(GuidedTreeVersion).where(
            GuidedTreeVersion.tree_id == tree_id,
            GuidedTreeVersion.version == version,
        )
    ).first()
    return dict(row.snapshot) if row is not None else None


def get_snapshot_node(snapshot: Dict[str, Any], node_key: str) -> Optional[Dict[str, Any]]:
    nodes = snapshot.get("nodes") or {}
    node = nodes.get(node_key)
    return dict(node) if isinstance(node, dict) else None


def _perimeter_visible(condition: Optional[Dict[str, Any]], perimeter: Optional[Dict[str, Any]]) -> bool:
    from app.services.guided_entry_index_service import perimeter_compatible

    return perimeter_compatible(condition, perimeter)


def visible_choices(
    node: Dict[str, Any], snapshot: Dict[str, Any], perimeter: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Choix du nœud, filtrés par le périmètre de la session (un choix pointant vers un
    nœud conditionné hors périmètre est masqué)."""
    out: List[Dict[str, Any]] = []
    for c in node.get("choices") or []:
        if not isinstance(c, dict) or not c.get("label"):
            continue
        nxt = str(c.get("next_node_key") or "")
        target = get_snapshot_node(snapshot, nxt) if nxt else None
        if target is not None and not _perimeter_visible(target.get("perimeter_condition"), perimeter):
            continue
        out.append(
            {
                "label": str(c.get("label")),
                "value": str(c.get("value") or ""),
                "hint": str(c.get("hint") or ""),
                "next_node_key": nxt or None,
            }
        )
    return out


def resolve_next_key(node: Dict[str, Any], choice_value: str) -> Optional[str]:
    """node_key suivant pour la valeur cliquée ; None si valeur inconnue."""
    for c in node.get("choices") or []:
        if str(c.get("value")) == str(choice_value):
            return str(c.get("next_node_key") or "") or None
    # Nœud linéaire (un seul choix) : on suit l'unique transition quel que soit le clic.
    choices = [c for c in (node.get("choices") or []) if c.get("next_node_key")]
    if len(choices) == 1:
        return str(choices[0].get("next_node_key"))
    return None


def breadcrumb_from_path(path: List[Dict[str, Any]], limit: int = 6) -> List[str]:
    """Fil d'Ariane compact : libellés des réponses données (les plus récentes)."""
    crumbs: List[str] = []
    for rec in path:
        sel = rec.get("user_selection") or {}
        label = str(sel.get("label") or "").strip()
        if label:
            crumbs.append(label)
    return crumbs[-limit:]


def step_payload_from_node(
    node: Dict[str, Any],
    snapshot: Dict[str, Any],
    *,
    perimeter: Optional[Dict[str, Any]] = None,
    path: Optional[List[Dict[str, Any]]] = None,
    can_go_back: bool = False,
    awaiting_feedback: bool = False,
) -> Dict[str, Any]:
    """Payload d'étape (SSE + persistance message) construit depuis un nœud de snapshot.

    Une feuille « résolution » N'EST PAS terminale côté session : elle attend le feedback
    (« résolu ? ») via les choix réservés __feedback_yes / __feedback_no.
    """
    is_leaf = bool(node.get("is_terminal"))
    termination = node.get("termination_type")
    step_type = str(node.get("step_type") or "question")

    if is_leaf and awaiting_feedback:
        choices = [
            {"label": "✅ Oui, problème résolu", "value": FEEDBACK_YES_VALUE, "hint": "", "next_node_key": None},
            {"label": "❌ Non, toujours un problème", "value": FEEDBACK_NO_VALUE, "hint": "", "next_node_key": None},
        ]
    elif is_leaf:
        choices = []
    else:
        choices = visible_choices(node, snapshot, perimeter)

    # L'auteur SAV ne rédige AUCUN texte : il ne liste que des cas. C'est donc ici que
    # LIA prend la parole — une invite neutre pour un embranchement, l'intitulé du cas
    # pour une fin (que guided_flow_service enrichit ensuite depuis la notice rattachée).
    message = str(node.get("message") or "").strip()
    if not message:
        message = (
            str(node.get("title") or "").strip()
            if is_leaf
            else "Parmi ces situations, laquelle correspond à la vôtre ?"
        )

    return {
        "step_type": step_type,
        "node_key": node.get("node_key"),
        "title": node.get("title") or "",
        "message": message,
        "choices": choices,
        "attachments": list(node.get("attachments") or []),
        "ask_photo": bool(node.get("ask_photo")),
        "allow_free_text": bool(node.get("allow_free_text", True)),
        "tools_hint": node.get("tools_hint") or "",
        "is_terminal": is_leaf and not awaiting_feedback,
        "termination_type": termination,
        "breadcrumb": breadcrumb_from_path(path or []),
        "can_go_back": can_go_back,
        "tree_title": snapshot.get("title") or "",
        "flow_kind": snapshot.get("flow_kind") or "diagnostic",
    }
