"""Autorat des arbres SAV : draft atomique, publication (snapshot + index), versions.

Le builder UI pousse TOUT l'arbre en une requête (save_tree_draft) — pas de CRUD
nœud-par-nœud, pas d'état à moitié sauvé. La publication vérifie le lint (bloquant),
fige un snapshot JSONB dans GuidedTreeVersion et reconstruit l'index d'entrée
sémantique. Le runtime ne lit QUE les snapshots.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models.document import Document
from app.models.guided_node_attachment import GuidedNodeAttachment
from app.models.guided_tree import GuidedTree, GuidedTreeNode
from app.models.guided_tree_version import GuidedTreeVersion
from app.services.guided_lint_service import LintIssue, has_blocking_issues, lint_tree

logger = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    v = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower()).strip("_")
    return v[:100] or "arbre"


def _unique_slug(session: Session, base: str, exclude_tree_id: Optional[int] = None) -> str:
    slug = base
    suffix = 1
    while True:
        stmt = select(GuidedTree).where(GuidedTree.slug == slug)
        found = session.exec(stmt).first()
        if found is None or (exclude_tree_id is not None and found.id == exclude_tree_id):
            return slug
        suffix += 1
        slug = f"{base}_{suffix}"


# ---------------------------------------------------------------------------
# Lecture / écriture du draft
# ---------------------------------------------------------------------------


def _load_nodes(session: Session, tree_id: int) -> List[GuidedTreeNode]:
    return list(
        session.exec(
            select(GuidedTreeNode).where(GuidedTreeNode.tree_id == tree_id)
        ).all()
    )


def _load_attachments(session: Session, node_ids: List[int]) -> Dict[int, List[GuidedNodeAttachment]]:
    if not node_ids:
        return {}
    rows = session.exec(
        select(GuidedNodeAttachment).where(GuidedNodeAttachment.node_id.in_(node_ids))
    ).all()
    by_node: Dict[int, List[GuidedNodeAttachment]] = {}
    for a in rows:
        by_node.setdefault(a.node_id, []).append(a)
    for lst in by_node.values():
        lst.sort(key=lambda a: (a.display_order, a.id or 0))
    return by_node


def _attachment_dict(a: GuidedNodeAttachment, doc_titles: Dict[int, str]) -> Dict[str, Any]:
    return {
        "document_id": a.document_id,
        "document_title": doc_titles.get(a.document_id, f"Document {a.document_id}"),
        "page_start": a.page_start,
        "page_end": a.page_end,
        "caption": a.caption or "",
        "kind": a.kind or "notice",
        "display_order": a.display_order,
    }


def get_tree_draft(session: Session, tree_id: int) -> Dict[str, Any]:
    """Retourne l'arbre complet (méta + nœuds + pièces jointes) en forme draft."""
    tree = session.get(GuidedTree, tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Arbre introuvable")
    nodes = _load_nodes(session, tree_id)
    atts = _load_attachments(session, [n.id for n in nodes if n.id])
    doc_ids = list({a.document_id for lst in atts.values() for a in lst})
    doc_titles: Dict[int, str] = {}
    if doc_ids:
        for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all():
            doc_titles[d.id] = d.title

    return {
        "meta": {
            "id": tree.id,
            "slug": tree.slug,
            "title": tree.title,
            "flow_kind": tree.flow_kind,
            "space_id": tree.space_id,
            "status": tree.status,
            "entry_symptom": tree.entry_symptom,
            "description": tree.description,
            "perimeter": tree.perimeter,
            "layout": tree.layout or {},
            "current_version": tree.current_version,
            "root_node_key": tree.root_node_key,
            "updated_at": tree.updated_at.isoformat() if tree.updated_at else None,
        },
        "nodes": [
            {
                "node_key": n.node_key,
                "step_type": n.step_type,
                "title": n.title,
                "message": n.message,
                "internal_note": n.internal_note,
                "is_terminal": n.is_terminal,
                "termination_type": n.termination_type,
                "ask_photo": n.ask_photo,
                "allow_free_text": n.allow_free_text,
                "perimeter_condition": n.perimeter_condition,
                "tools_hint": n.tools_hint,
                "choices": n.choices or [],
                "attachments": [
                    _attachment_dict(a, doc_titles) for a in atts.get(n.id or -1, [])
                ],
            }
            for n in nodes
        ],
    }


def save_tree_draft(
    session: Session, tree_id: int, payload: Dict[str, Any], user_id: int
) -> Dict[str, Any]:
    """Remplacement ATOMIQUE du draft : upsert des nœuds par node_key, delete des absents,
    resynchronisation complète des pièces jointes. Retourne le draft relu + lint informatif."""
    tree = session.get(GuidedTree, tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Arbre introuvable")

    meta = payload.get("meta") or {}
    if meta.get("title"):
        tree.title = str(meta["title"])[:300]
    if "entry_symptom" in meta:
        tree.entry_symptom = (str(meta["entry_symptom"]) or None) if meta["entry_symptom"] else None
    if "description" in meta:
        tree.description = str(meta.get("description") or "")
    if "perimeter" in meta:
        tree.perimeter = meta.get("perimeter") or None
    if "layout" in meta and isinstance(meta.get("layout"), dict):
        # Positions libres des cas (déplacement à la souris) — visuel uniquement.
        tree.layout = meta["layout"] or None
    if meta.get("root_node_key"):
        tree.root_node_key = str(meta["root_node_key"])[:120]
    tree.updated_by = user_id
    tree.updated_at = datetime.utcnow()
    session.add(tree)

    incoming = {str(n.get("node_key") or ""): n for n in (payload.get("nodes") or []) if n.get("node_key")}
    if not incoming:
        raise HTTPException(status_code=422, detail="Arbre sans nœuds")

    existing = {n.node_key: n for n in _load_nodes(session, tree_id)}

    # Delete des nœuds absents du payload (les attachments suivent par CASCADE).
    for key, node in existing.items():
        if key not in incoming:
            session.delete(node)

    # Upsert des nœuds.
    saved_nodes: Dict[str, GuidedTreeNode] = {}
    for key, data in incoming.items():
        node = existing.get(key) or GuidedTreeNode(tree_id=tree_id, node_key=key[:120])
        node.step_type = str(data.get("step_type") or "question")[:20]
        node.title = str(data.get("title") or "")[:200]
        node.message = str(data.get("message") or "")
        node.internal_note = str(data.get("internal_note") or "")
        node.is_terminal = bool(data.get("is_terminal"))
        term = data.get("termination_type")
        node.termination_type = str(term)[:20] if term else None
        node.ask_photo = bool(data.get("ask_photo"))
        node.allow_free_text = bool(data.get("allow_free_text", True))
        node.perimeter_condition = data.get("perimeter_condition") or None
        node.tools_hint = str(data.get("tools_hint") or "")[:200]
        choices = []
        for c in data.get("choices") or []:
            if not isinstance(c, dict) or not str(c.get("label") or "").strip():
                continue
            choices.append(
                {
                    "label": str(c.get("label") or "").strip()[:200],
                    "value": str(c.get("value") or _slugify(str(c.get("label") or "")))[:120],
                    "hint": str(c.get("hint") or "")[:300],
                    "next_node_key": str(c.get("next_node_key") or "")[:120] or None,
                }
            )
        node.choices = choices
        session.add(node)
        saved_nodes[key] = node

    session.flush()  # ids des nouveaux nœuds

    # Resync complet des pièces jointes des nœuds présents.
    node_ids = [n.id for n in saved_nodes.values() if n.id]
    if node_ids:
        for a in session.exec(
            select(GuidedNodeAttachment).where(GuidedNodeAttachment.node_id.in_(node_ids))
        ).all():
            session.delete(a)
        session.flush()
    for key, data in incoming.items():
        node = saved_nodes[key]
        for order, att in enumerate(data.get("attachments") or []):
            if not isinstance(att, dict) or not att.get("document_id"):
                continue
            session.add(
                GuidedNodeAttachment(
                    node_id=node.id,
                    document_id=int(att["document_id"]),
                    page_start=int(att["page_start"]) if att.get("page_start") else None,
                    page_end=int(att["page_end"]) if att.get("page_end") else None,
                    caption=str(att.get("caption") or "")[:300],
                    kind=str(att.get("kind") or "notice")[:20],
                    display_order=order,
                )
            )

    session.commit()

    draft = get_tree_draft(session, tree_id)
    issues = lint_tree(draft)
    draft["lint"] = [i.model_dump() for i in issues]
    return draft


# ---------------------------------------------------------------------------
# Cycle de vie
# ---------------------------------------------------------------------------


def create_tree(
    session: Session,
    *,
    title: str,
    entry_symptom: Optional[str],
    space_id: Optional[int],
    description: str = "",
    user_id: int,
) -> GuidedTree:
    slug = _unique_slug(session, _slugify(f"diagnostic_{title}"))
    tree = GuidedTree(
        slug=slug,
        title=title[:300],
        flow_kind="diagnostic",
        space_id=space_id,
        status="draft",
        entry_symptom=entry_symptom or None,
        description=description or "",
        root_node_key="root",
        created_by=user_id,
        updated_by=user_id,
    )
    session.add(tree)
    session.flush()
    root = GuidedTreeNode(
        tree_id=tree.id,
        node_key="root",
        step_type="question",
        title="Première question",
        message="",
        choices=[],
    )
    session.add(root)
    session.commit()
    session.refresh(tree)
    return tree


def build_snapshot(session: Session, tree: GuidedTree, version: int) -> Dict[str, Any]:
    """Fige le draft en snapshot canonique (nodes indexés par node_key, titres de
    documents dénormalisés) — la forme que le runtime consomme."""
    draft = get_tree_draft(session, tree.id)
    nodes_by_key = {n["node_key"]: n for n in draft["nodes"]}
    return {
        "slug": tree.slug,
        "title": tree.title,
        "flow_kind": tree.flow_kind,
        "space_id": tree.space_id,
        "entry_symptom": tree.entry_symptom,
        "description": tree.description,
        "perimeter": tree.perimeter,
        "root_node_key": tree.root_node_key,
        "version": version,
        "nodes": nodes_by_key,
    }


def publish_tree(
    session: Session, tree_id: int, *, note: str = "", user_id: int
) -> Dict[str, Any]:
    """Lint bloquant → snapshot → version++ → status published → rebuild index d'entrée."""
    tree = session.get(GuidedTree, tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Arbre introuvable")

    draft = get_tree_draft(session, tree_id)
    issues = lint_tree(draft)
    if has_blocking_issues(issues):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Lint bloquant — corriger avant publication",
                "issues": [i.model_dump() for i in issues],
            },
        )

    version = int(tree.current_version or 0) + 1
    snapshot = build_snapshot(session, tree, version)
    session.add(
        GuidedTreeVersion(
            tree_id=tree.id,
            version=version,
            snapshot=snapshot,
            note=note[:300],
            published_by=user_id,
        )
    )
    tree.current_version = version
    tree.status = "published"
    tree.updated_by = user_id
    tree.updated_at = datetime.utcnow()
    session.add(tree)
    session.commit()

    from app.services.guided_entry_index_service import rebuild_tree_entries

    rebuild_tree_entries(session, tree, snapshot)
    logger.info("[guided_authoring] publié tree=%s v%s (%d nœuds)", tree.slug, version, len(snapshot["nodes"]))
    return {"version": version, "lint": [i.model_dump() for i in issues]}


def rollback_tree(session: Session, tree_id: int, version: int, *, user_id: int) -> Dict[str, Any]:
    """Republie un snapshot antérieur comme nouvelle version (le draft n'est pas modifié)."""
    tree = session.get(GuidedTree, tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Arbre introuvable")
    old = session.exec(
        select(GuidedTreeVersion).where(
            GuidedTreeVersion.tree_id == tree_id, GuidedTreeVersion.version == version
        )
    ).first()
    if old is None:
        raise HTTPException(status_code=404, detail=f"Version {version} introuvable")

    new_version = int(tree.current_version or 0) + 1
    snapshot = dict(old.snapshot or {})
    snapshot["version"] = new_version
    session.add(
        GuidedTreeVersion(
            tree_id=tree.id,
            version=new_version,
            snapshot=snapshot,
            note=f"Rollback vers v{version}",
            published_by=user_id,
        )
    )
    tree.current_version = new_version
    tree.status = "published"
    tree.updated_at = datetime.utcnow()
    session.add(tree)
    session.commit()

    from app.services.guided_entry_index_service import rebuild_tree_entries

    rebuild_tree_entries(session, tree, snapshot)
    return {"version": new_version}


def archive_tree(session: Session, tree_id: int, *, user_id: int) -> None:
    tree = session.get(GuidedTree, tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Arbre introuvable")
    tree.status = "archived"
    tree.updated_by = user_id
    tree.updated_at = datetime.utcnow()
    session.add(tree)
    session.commit()

    from app.services.guided_entry_index_service import remove_tree_entries

    remove_tree_entries(session, tree_id)


def delete_tree(session: Session, tree_id: int) -> Dict[str, Any]:
    """Suppression DÉFINITIVE de l'arbre. Les nœuds, pièces jointes, versions publiées et
    entrées d'index partent en cascade (ON DELETE CASCADE) ; l'index d'entrée est purgé
    explicitement pour que rien ne subsiste côté recherche même si la cascade évolue.

    Les GuidedSession déjà tenues gardent leur `authored_tree_id` (pas de contrainte FK) :
    c'est de l'historique, il ne doit pas disparaître avec l'arbre — mais un parcours en
    cours perd son snapshot, d'où le décompte retourné pour prévenir l'auteur.
    """
    tree = session.get(GuidedTree, tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Arbre introuvable")

    from app.models.guided_session import GuidedSession
    from app.services.guided_entry_index_service import remove_tree_entries

    sessions = session.exec(
        select(GuidedSession).where(GuidedSession.authored_tree_id == tree_id)
    ).all()
    live = sum(1 for s in sessions if s.status == "active")

    remove_tree_entries(session, tree_id)
    title, slug = tree.title, tree.slug
    session.delete(tree)
    session.commit()
    logger.info("[guided_authoring] arbre supprimé : %s (%s)", title, slug)
    return {"deleted": True, "title": title, "sessions": len(sessions), "active_sessions": live}


def duplicate_tree(
    session: Session, tree_id: int, *, target_space_id: Optional[int] = None, user_id: int
) -> GuidedTree:
    draft = get_tree_draft(session, tree_id)
    src = session.get(GuidedTree, tree_id)
    new = create_tree(
        session,
        title=f"{src.title} (copie)",
        entry_symptom=src.entry_symptom,
        space_id=target_space_id if target_space_id is not None else src.space_id,
        description=src.description,
        user_id=user_id,
    )
    draft["meta"]["title"] = new.title
    save_tree_draft(session, new.id, draft, user_id)
    return new


def export_tree(session: Session, tree_id: int) -> Dict[str, Any]:
    draft = get_tree_draft(session, tree_id)
    draft["meta"].pop("id", None)
    draft["meta"].pop("current_version", None)
    draft["meta"].pop("status", None)
    draft.pop("lint", None)
    return draft


def import_tree(
    session: Session, payload: Dict[str, Any], *, space_id: Optional[int], user_id: int
) -> GuidedTree:
    meta = payload.get("meta") or {}
    tree = create_tree(
        session,
        title=str(meta.get("title") or "Arbre importé"),
        entry_symptom=meta.get("entry_symptom"),
        space_id=space_id,
        description=str(meta.get("description") or ""),
        user_id=user_id,
    )
    save_tree_draft(session, tree.id, payload, user_id)
    return tree
