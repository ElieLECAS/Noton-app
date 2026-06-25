"""Phase 2 — exploitation des arbres d'accompagnement validés (authored trees).

Quand un arbre ACTIF correspond à la demande (symptôme / catégories / mots-clés), il PRIME
sur la génération dynamique : on déroule ses nœuds pas-à-pas (pas d'appel LLM de routage).
Fallback dynamique systématique si aucun arbre ne matche ou si l'arbre est en impasse —
on ne bloque jamais l'utilisateur.

Le hook ``match_authored_tree`` était volontairement absent en Phase 1 ; il est branché ici.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from app.config import settings
from app.models.guided_tree import GuidedTree, GuidedTreeNode
from app.services.category_catalog import suggested_categories_for_intent
from app.services.procedural_router_service import (
    STEP_TYPES,
    EscalationRecap,
    RoutingChoice,
    RoutingStep,
)
from app.services.query_signals_schemas import LightweightQuerySignals

logger = logging.getLogger(__name__)

# Score minimal pour qu'un arbre prime sur le dynamique (au moins un critère significatif).
_MATCH_MIN_SCORE = 1.0
_WEIGHT_SYMPTOM = 3.0
_WEIGHT_CATEGORY = 1.0
_WEIGHT_KEYWORD = 0.5


def _tokenize(value: str) -> set:
    return {t for t in re.split(r"[^a-zA-Z0-9àâäéèêëïîôöùûüç]+", (value or "").lower()) if len(t) > 2}


def match_authored_tree(
    session: Session,
    *,
    space_id: int,
    flow_kind: str,
    symptom: Optional[str] = None,
    inferred_categories: Optional[List[str]] = None,
    topic: str = "",
    user_message: str = "",
) -> Optional[GuidedTree]:
    """Sélectionne le meilleur arbre actif correspondant à la demande, ou None.

    Score = symptôme (fort) + catégories communes + mots-clés communs. Départage par priority.
    """
    if not settings.GUIDED_AUTHORED_TREES_ENABLED:
        return None

    stmt = select(GuidedTree).where(
        GuidedTree.is_active == True,  # noqa: E712
        GuidedTree.flow_kind == flow_kind,
    )
    trees = [
        t for t in session.exec(stmt).all() if t.space_id is None or t.space_id == space_id
    ]
    if not trees:
        return None

    tokens = _tokenize(f"{topic} {user_message}")
    inferred = {c.strip().lower() for c in (inferred_categories or []) if c}
    sym = (symptom or "").strip().lower()

    best: Optional[GuidedTree] = None
    best_rank: tuple = (0.0, -1)
    for tree in trees:
        score = 0.0
        if sym and sym in {s.strip().lower() for s in (tree.match_symptoms or [])}:
            score += _WEIGHT_SYMPTOM
        score += len(inferred & {c.strip().lower() for c in (tree.match_categories or [])}) * _WEIGHT_CATEGORY
        score += len(tokens & {k.strip().lower() for k in (tree.match_keywords or [])}) * _WEIGHT_KEYWORD
        rank = (score, tree.priority)
        if score >= _MATCH_MIN_SCORE and rank > best_rank:
            best, best_rank = tree, rank

    if best is not None:
        logger.info(
            "[authored_tree] match tree=%s score=%.1f flow=%s symptom=%r",
            best.slug,
            best_rank[0],
            flow_kind,
            sym or None,
        )
    return best


def _load_node(session: Session, tree_id: int, node_key: str) -> Optional[GuidedTreeNode]:
    return session.exec(
        select(GuidedTreeNode).where(
            GuidedTreeNode.tree_id == tree_id,
            GuidedTreeNode.node_key == node_key,
        )
    ).first()


def resolve_authored_node(
    session: Session,
    gsession,
    *,
    guided_choice: Optional[Dict[str, Any]],
    resuming: bool,
) -> Optional[GuidedTreeNode]:
    """Nœud à afficher ce tour ; avance via le choix utilisateur en reprise.

    Met à jour ``gsession.current_node_key``. Retourne None en impasse → fallback dynamique.
    """
    tree_id = gsession.authored_tree_id
    if not tree_id:
        return None

    current_key = gsession.current_node_key or "root"

    if resuming:
        current_node = _load_node(session, tree_id, current_key)
        if current_node is None or current_node.is_terminal:
            return None
        chosen = str((guided_choice or {}).get("value") or "")
        next_key: Optional[str] = None
        for choice in current_node.choices or []:
            if str(choice.get("value")) == chosen:
                next_key = choice.get("next_node_key")
                break
        # Choix inconnu mais nœud linéaire (un seul choix) → on suit l'unique transition.
        if not next_key and len(current_node.choices or []) == 1:
            next_key = (current_node.choices or [{}])[0].get("next_node_key")
        if not next_key:
            logger.info("[authored_tree] impasse node=%s choix=%r → fallback dynamique", current_key, chosen)
            return None
        current_key = next_key

    gsession.current_node_key = current_key
    return _load_node(session, tree_id, current_key)


def routing_step_from_node(
    node: GuidedTreeNode,
    passages: Optional[List[Dict[str, Any]]] = None,
) -> RoutingStep:
    """Construit un RoutingStep (compatible run_guided_turn) à partir d'un nœud d'arbre."""
    choices = [
        RoutingChoice(
            label=str(c.get("label") or ""),
            value=str(c.get("value") or ""),
            hint=str(c.get("hint") or ""),
        )
        for c in (node.choices or [])
        if c.get("label") and c.get("value") is not None
    ]
    step_type = node.step_type if node.step_type in STEP_TYPES else "instruction"
    is_terminal = bool(node.is_terminal)

    cited_pages: List[Dict[str, Any]] = []
    for p in (passages or [])[:3]:
        page_no = p.get("page_no") or p.get("page_start")
        if page_no is not None:
            cited_pages.append(
                {"document_title": p.get("document_title", "Document"), "page_no": page_no}
            )

    recap = None
    if is_terminal and node.termination_type == "escalation":
        recap = EscalationRecap(summary=node.message or "")

    return RoutingStep(
        step_type=step_type,
        message=node.message or "",
        choices=[] if is_terminal else choices,
        cited_pages=cited_pages,
        is_terminal=is_terminal,
        escalation_recap=recap,
    )


def signals_for_authored_node(
    flow_kind: str,
    accumulated: Dict[str, Any],
    topic: str,
    node: GuidedTreeNode,
) -> LightweightQuerySignals:
    """Signaux de retrieval scopés par le nœud (ses retrieval_categories / retrieval_entities)."""
    intent = "installation" if flow_kind == "howto" else "troubleshooting"
    inferred = list(node.retrieval_categories or []) or suggested_categories_for_intent(intent)
    entity_texts = [str(t) for t in (accumulated.get("entity_texts") or []) if str(t).strip()]
    entity_texts += [str(e) for e in (node.retrieval_entities or []) if str(e).strip()]
    entity_texts = list(dict.fromkeys(entity_texts))
    detected_refs = list(entity_texts)
    if topic and topic not in detected_refs:
        detected_refs.insert(0, topic)
    return LightweightQuerySignals(
        intent=intent,
        inferred_categories=inferred,
        entity_texts=entity_texts,
        detected_references=detected_refs,
    )
