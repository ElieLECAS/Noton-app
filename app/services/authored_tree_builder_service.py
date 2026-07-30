"""Génération semi-automatique de brouillons d'arbres d'accompagnement (Phase 2).

À partir de la matière déjà structurée par l'ingestion :
  - chunks L2 ``diagnostic_unit`` (symptôme → cause → vérification → résolution) + relations KAG
    ``symptome_cause`` / ``cause_resolution`` pour les arbres DIAGNOSTIC (SAV) ;
  - chunks L2 ``procedural_step`` (étapes ordonnées) + relations ``etape_precede`` pour les arbres HOWTO.

Le LLM produit un brouillon d'arbre (GuidedTree + GuidedTreeNode) ; il est TOUJOURS persisté
``status="draft"`` : un expert métier doit le relire dans le builder et le PUBLIER.
Aucun arbre auto-publié.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlmodel import Session

from app.config import settings
from app.models.guided_tree import GuidedTree, GuidedTreeNode
from app.services.category_catalog import SYMPTOM_LABELS

logger = logging.getLogger(__name__)

_DIAGNOSTIC_RELATIONS = ("symptome_cause", "cause_resolution")
_HOWTO_RELATIONS = ("etape_precede", "necessite_outil", "requiert_piece")


# ---------------------------------------------------------------------------
# Schéma de brouillon
# ---------------------------------------------------------------------------


class DraftChoice(BaseModel):
    label: str
    value: str
    hint: str = ""
    next_node_key: Optional[str] = None


class DraftNode(BaseModel):
    node_key: str
    step_type: str = "question"
    message: str
    is_terminal: bool = False
    termination_type: Optional[str] = None
    retrieval_categories: List[str] = Field(default_factory=list)
    retrieval_entities: List[str] = Field(default_factory=list)
    choices: List[DraftChoice] = Field(default_factory=list)


class DraftTree(BaseModel):
    title: str
    root_node_key: str = "root"
    nodes: List[DraftNode] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Collecte de la matière
# ---------------------------------------------------------------------------


def _collect_chunks_for_category(
    session: Session, space_id: int, category_slug: str, chunk_role: str
) -> List[str]:
    """Récupère les contenus des chunks L2 d'un rôle donné, liés à une catégorie dans l'espace."""
    rows = session.execute(
        text(
            """
            SELECT DISTINCT dc.id, dc.content, dc.metadata_json
            FROM documentchunk dc
            INNER JOIN chunkcategoryrelation ccr ON ccr.chunk_id = dc.id
            INNER JOIN documentcategory cat ON cat.id = ccr.category_id
            INNER JOIN document_space ds ON ds.document_id = ccr.document_id
            WHERE ds.space_id = :space_id AND cat.slug = :slug
            """
        ),
        {"space_id": space_id, "slug": category_slug},
    ).all()

    out: List[str] = []
    for _id, content, meta in rows:
        meta = meta or {}
        if meta.get("chunk_role") != chunk_role:
            continue
        text_block = (content or "").strip()
        structured = meta.get("structured") or {}
        if structured:
            text_block += "\n[structuré] " + "; ".join(
                f"{k}={v}" for k, v in structured.items() if v
            )
        if text_block:
            out.append(text_block)
    return out


def _collect_relations(session: Session, space_id: int, rel_types: Tuple[str, ...]) -> List[str]:
    """Relations KAG d'un ensemble de types, formatées 'A —type→ B'."""
    rows = session.execute(
        text(
            """
            SELECT ea.name, eer.relation_type, eb.name
            FROM entityentityrelation eer
            INNER JOIN knowledgeentity ea ON ea.id = eer.entity_a_id
            INNER JOIN knowledgeentity eb ON eb.id = eer.entity_b_id
            WHERE eer.space_id = :space_id AND eer.relation_type = ANY(:types)
            """
        ),
        {"space_id": space_id, "types": list(rel_types)},
    ).all()
    return [f"{a} —{rel}→ {b}" for a, rel, b in rows if a and b]


# ---------------------------------------------------------------------------
# Génération LLM
# ---------------------------------------------------------------------------


_BUILDER_SYSTEM_PROMPT = """Tu es concepteur de protocoles d'accompagnement technique PROFERM (menuiserie).
On te fournit de la matière documentaire déjà structurée (unités de diagnostic ou étapes de pose)
et des relations extraites. Tu construis un ARBRE de guidage pas-à-pas.

Règles :
1. Renvoie UNIQUEMENT un objet JSON valide.
2. Un arbre = une liste de nœuds reliés par des choix. Chaque nœud a un node_key unique
   (ex: "root", "verif_joint", "resolution_ok").
3. step_type ∈ instruction | question | diagnosis | resolution | escalation.
4. Un nœud non terminal porte 2 à 4 choix mutuellement exclusifs ; chaque choix a
   un next_node_key pointant vers un node_key EXISTANT.
5. Les nœuds terminaux : is_terminal=true, choices=[], termination_type = "resolution" ou "escalation".
6. Toujours prévoir une branche d'ESCALADE (transmission SAV) si le diagnostic échoue.
7. retrieval_categories : slugs utiles pour illustrer le nœud (ex: sealing, infiltration_eau, guide_sav).
8. Reste fidèle à la matière fournie : n'invente pas de geste/cause/résolution absent.
9. root_node_key doit exister dans nodes.

Format :
{{
  "title": "...",
  "root_node_key": "root",
  "nodes": [
    {{
      "node_key": "root",
      "step_type": "question",
      "message": "...",
      "is_terminal": false,
      "termination_type": null,
      "retrieval_categories": ["..."],
      "retrieval_entities": [],
      "choices": [
        {{ "label": "...", "value": "oui", "hint": "", "next_node_key": "..." }}
      ]
    }}
  ]
}}"""


def _generate_tree_draft(
    flow_kind: str,
    subject_label: str,
    material_blocks: List[str],
    relations: List[str],
) -> Optional[DraftTree]:
    from app.services.multimodal_page_service import (
        _mistral_chat_completion,
        _parse_json_with_repair,
    )

    if not material_blocks:
        logger.info("[tree_builder] aucune matière pour %s — abandon", subject_label)
        return None

    material_text = "\n\n---\n\n".join(b[:1500] for b in material_blocks[:20])
    relations_text = "\n".join(relations[:40]) or "(aucune)"
    user_text = (
        f"Type d'arbre : {flow_kind}\nSujet : {subject_label}\n\n"
        f"Matière documentaire structurée :\n{material_text}\n\n"
        f"Relations extraites :\n{relations_text}\n\n"
        "Construis l'arbre de guidage selon les règles du système."
    )
    messages = [
        {"role": "system", "content": _BUILDER_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    try:
        raw = _mistral_chat_completion(
            messages,
            page_no=0,
            max_tokens=4096,
            temperature=0.1,
            response_format_json=True,
            timeout_seconds=120,
            model=settings.CONTEXTUAL_ENRICHMENT_MODEL or settings.MODEL_FAST,
        )
        data = _parse_json_with_repair(raw)
        draft = DraftTree.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tree_builder] génération échouée pour %s : %s", subject_label, exc)
        return None

    if not draft.nodes:
        return None
    return _sanitize_draft(draft)


def _sanitize_draft(draft: DraftTree) -> DraftTree:
    """Garantit la cohérence : root présent, next_node_key valides, terminaux sans choix."""
    valid_types = {"instruction", "question", "diagnosis", "resolution", "escalation"}
    keys = {n.node_key for n in draft.nodes}
    if draft.root_node_key not in keys:
        draft.root_node_key = draft.nodes[0].node_key

    for node in draft.nodes:
        if node.step_type not in valid_types:
            node.step_type = "question"
        if node.is_terminal:
            node.choices = []
            if node.termination_type not in ("resolution", "escalation"):
                node.termination_type = "resolution"
        else:
            node.choices = [c for c in node.choices if c.next_node_key in keys]
            # Un nœud non terminal sans choix valide devient une résolution (évite l'impasse).
            if not node.choices:
                node.is_terminal = True
                node.termination_type = "resolution"
    return draft


# ---------------------------------------------------------------------------
# Persistance (brouillon inactif)
# ---------------------------------------------------------------------------


def _slugify(value: str) -> str:
    v = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return v[:100] or "tree"


def _persist_tree_draft(
    session: Session,
    draft: DraftTree,
    *,
    flow_kind: str,
    space_id: Optional[int],
    match_symptoms: List[str],
    match_categories: List[str],
    match_keywords: List[str],
) -> GuidedTree:
    base_slug = _slugify(f"{flow_kind}_{draft.title}")
    slug = base_slug
    suffix = 1
    from sqlmodel import select

    while session.exec(select(GuidedTree).where(GuidedTree.slug == slug)).first() is not None:
        suffix += 1
        slug = f"{base_slug}_{suffix}"

    tree = GuidedTree(
        slug=slug,
        title=draft.title[:300],
        flow_kind=flow_kind,
        space_id=space_id,
        status="draft",  # JAMAIS auto-publié : relecture + publication humaine requise
        entry_symptom=(match_symptoms[0] if match_symptoms else None),
        priority=0,
        match_keywords=match_keywords,
        match_categories=match_categories,
        match_symptoms=match_symptoms,
        root_node_key=draft.root_node_key,
    )
    session.add(tree)
    session.flush()

    for node in draft.nodes:
        session.add(
            GuidedTreeNode(
                tree_id=tree.id,
                node_key=node.node_key[:120],
                step_type=node.step_type,
                message=node.message,
                is_terminal=node.is_terminal,
                termination_type=node.termination_type,
                retrieval_categories=node.retrieval_categories,
                retrieval_entities=node.retrieval_entities,
                choices=[c.model_dump() for c in node.choices],
            )
        )
    session.commit()
    session.refresh(tree)
    logger.info(
        "[tree_builder] brouillon créé tree=%s flow=%s nodes=%s (is_active=false)",
        tree.slug,
        flow_kind,
        len(draft.nodes),
    )
    return tree


# ---------------------------------------------------------------------------
# Points d'entrée
# ---------------------------------------------------------------------------


def build_diagnostic_tree_for_symptom(
    session: Session, *, space_id: int, symptom: str
) -> Optional[GuidedTree]:
    """Construit un brouillon d'arbre diagnostic SAV pour un symptôme, ou None si pas de matière."""
    label = SYMPTOM_LABELS.get(symptom, symptom)
    material = _collect_chunks_for_category(session, space_id, symptom, "diagnostic_unit")
    if not material:
        # Repli : chunks troubleshooting génériques de l'espace.
        material = _collect_chunks_for_category(session, space_id, "troubleshooting", "diagnostic_unit")
    relations = _collect_relations(session, space_id, _DIAGNOSTIC_RELATIONS)
    draft = _generate_tree_draft("diagnostic", label, material, relations)
    if draft is None:
        return None
    return _persist_tree_draft(
        session,
        draft,
        flow_kind="diagnostic",
        space_id=space_id,
        match_symptoms=[symptom],
        match_categories=["troubleshooting", symptom],
        match_keywords=[w for w in re.split(r"\W+", label.lower()) if len(w) > 2],
    )


def build_howto_tree_for_category(
    session: Session, *, space_id: int, category: str, topic: str = ""
) -> Optional[GuidedTree]:
    """Construit un brouillon d'arbre how-to (pose/montage) pour une catégorie tâche."""
    material = _collect_chunks_for_category(session, space_id, category, "procedural_step")
    relations = _collect_relations(session, space_id, _HOWTO_RELATIONS)
    draft = _generate_tree_draft("howto", topic or category, material, relations)
    if draft is None:
        return None
    return _persist_tree_draft(
        session,
        draft,
        flow_kind="howto",
        space_id=space_id,
        match_symptoms=[],
        match_categories=[category],
        match_keywords=[w for w in re.split(r"\W+", (topic or category).lower()) if len(w) > 2],
    )
