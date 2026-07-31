"""API admin des Arbres SAV : autorat, publication, symptômes, angles morts, vignettes.

Édition : rôle admin OU permission `guided_trees:manage` (seedée par la migration
guided_runtime_v2). Publication / rollback / archivage : admin uniquement.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.database import get_session
from app.models.document import Document
from app.models.document_category import DocumentCategory
from app.models.guided_entry import GuidedSymptomAlias
from app.models.guided_session import GuidedSession
from app.models.guided_tree import GuidedTree
from app.models.guided_tree_version import GuidedTreeVersion
from app.models.user import UserRead
from app.routers.auth import get_current_user, require_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/sav", tags=["sav-trees"])

THUMBS_CACHE_DIR = "media/thumbs_cache"


def require_sav_editor(current_user: UserRead = Depends(get_current_user)) -> UserRead:
    """Édition des arbres : admin, ou porteur de la permission dédiée."""
    roles = current_user.roles or []
    permissions = getattr(current_user, "permissions", None) or []
    if "admin" in roles or "guided_trees:manage" in permissions:
        return current_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Permission requise : guided_trees:manage",
    )


# ---------------------------------------------------------------------------
# Arbres
# ---------------------------------------------------------------------------


class TreeCreateRequest(BaseModel):
    title: str
    entry_symptom: Optional[str] = None
    space_id: Optional[int] = None
    description: str = ""


class PublishRequest(BaseModel):
    note: str = ""


class DuplicateRequest(BaseModel):
    target_space_id: Optional[int] = None


class ImportRequest(BaseModel):
    space_id: Optional[int] = None
    payload: Dict[str, Any]


class ImportJsonRequest(BaseModel):
    """Texte collé tel quel (le service tolère les ``` et un préambule)."""

    space_id: Optional[int] = None
    json_text: str


@router.get("/trees")
async def list_trees(
    space_id: Optional[int] = None,
    symptom: str = "",
    tree_status: str = "",
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    stmt = select(GuidedTree)
    trees = session.exec(stmt).all()
    out: List[Dict[str, Any]] = []
    for t in trees:
        if space_id is not None and t.space_id not in (None, space_id):
            continue
        if symptom and (t.entry_symptom or "") != symptom:
            continue
        if tree_status and t.status != tree_status:
            continue
        sessions = session.exec(
            select(GuidedSession).where(GuidedSession.authored_tree_id == t.id)
        ).all()
        out.append(
            {
                "id": t.id,
                "slug": t.slug,
                "title": t.title,
                "status": t.status,
                "entry_symptom": t.entry_symptom,
                "space_id": t.space_id,
                "current_version": t.current_version,
                "description": t.description,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
                "stats": {
                    "sessions": len(sessions),
                    "resolved": sum(1 for s in sessions if s.status == "resolved"),
                    "escalated": sum(1 for s in sessions if s.status == "escalated"),
                    "abandoned": sum(1 for s in sessions if s.status == "abandoned"),
                },
            }
        )
    out.sort(key=lambda x: (x["entry_symptom"] or "zzz", x["title"]))
    return {"trees": out}


@router.post("/trees", status_code=201)
async def create_tree_endpoint(
    request: TreeCreateRequest,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import create_tree, get_tree_draft

    tree = create_tree(
        session,
        title=request.title,
        entry_symptom=request.entry_symptom,
        space_id=request.space_id,
        description=request.description,
        user_id=current_user.id,
    )
    return get_tree_draft(session, tree.id)


@router.get("/trees/{tree_id}")
async def get_tree(
    tree_id: int,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import get_tree_draft
    from app.services.guided_lint_service import lint_tree

    draft = get_tree_draft(session, tree_id)
    draft["lint"] = [i.model_dump() for i in lint_tree(draft)]
    return draft


@router.put("/trees/{tree_id}/draft")
async def save_draft(
    tree_id: int,
    payload: Dict[str, Any],
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import save_tree_draft

    return save_tree_draft(session, tree_id, payload, current_user.id)


@router.post("/trees/{tree_id}/publish")
async def publish(
    tree_id: int,
    request: PublishRequest,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import publish_tree

    return publish_tree(session, tree_id, note=request.note, user_id=current_user.id)


@router.post("/trees/{tree_id}/rollback/{version}")
async def rollback(
    tree_id: int,
    version: int,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import rollback_tree

    return rollback_tree(session, tree_id, version, user_id=current_user.id)


@router.post("/trees/{tree_id}/archive")
async def archive(
    tree_id: int,
    current_user: UserRead = Depends(require_role("admin")),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import archive_tree

    archive_tree(session, tree_id, user_id=current_user.id)
    return {"ok": True}


@router.delete("/trees/{tree_id}")
async def delete_tree_endpoint(
    tree_id: int,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    """Suppression définitive. Un brouillon jamais publié peut être supprimé par tout
    éditeur SAV ; dès qu'un arbre a été mis en service (il a un historique de versions et
    peut avoir des parcours clients), seul un administrateur peut le supprimer."""
    from app.services.guided_authoring_service import delete_tree

    tree = session.get(GuidedTree, tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Arbre introuvable")
    if int(tree.current_version or 0) > 0 and "admin" not in (current_user.roles or []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cet arbre a déjà été mis en service : seul un administrateur peut le supprimer.",
        )
    return delete_tree(session, tree_id)


@router.post("/trees/{tree_id}/duplicate", status_code=201)
async def duplicate(
    tree_id: int,
    request: DuplicateRequest,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import duplicate_tree, get_tree_draft

    new = duplicate_tree(
        session, tree_id, target_space_id=request.target_space_id, user_id=current_user.id
    )
    return get_tree_draft(session, new.id)


@router.get("/trees/{tree_id}/export")
async def export(
    tree_id: int,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import export_tree

    return export_tree(session, tree_id)


@router.post("/trees/import", status_code=201)
async def import_endpoint(
    request: ImportRequest,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import get_tree_draft, import_tree

    tree = import_tree(
        session, request.payload, space_id=request.space_id, user_id=current_user.id
    )
    return get_tree_draft(session, tree.id)


@router.get("/import-json/prompt")
async def import_json_prompt(current_user: UserRead = Depends(require_sav_editor)):
    """Le mode d'emploi à coller dans Gemini / ChatGPT avec la notice (source unique)."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "arbre_sav_json.md")
    try:
        with open(path, encoding="utf-8") as fh:
            return {"prompt": fh.read()}
    except OSError:
        raise HTTPException(status_code=500, detail="Mode d'emploi introuvable sur le serveur.")


@router.post("/trees/import-json", status_code=201)
async def import_json_endpoint(
    request: ImportJsonRequest,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    """Crée un arbre en BROUILLON depuis un JSON pivot (produit par un LLM à partir
    d'une notice) ou depuis un export interne. Rien n'est publié : l'auteur relit le
    graphe puis publie."""
    from app.services.guided_json_import_service import import_from_json_text

    return import_from_json_text(
        session, request.json_text, space_id=request.space_id, user_id=current_user.id
    )


@router.post("/trees/import-json/check")
async def check_json_endpoint(
    request: ImportJsonRequest,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    """Vérifie le JSON sans rien créer : mêmes erreurs bloquantes, même rapport."""
    from app.services.guided_json_import_service import (
        convert_pivot,
        is_native_payload,
        parse_json_text,
    )

    data = parse_json_text(request.json_text)
    if is_native_payload(data):
        return {
            "title": (data.get("meta") or {}).get("title") or "",
            "report": {"cases": len(data.get("nodes") or []), "warnings": ["Format interne."]},
        }
    converted = convert_pivot(session, data)
    return {
        "title": converted["payload"]["meta"]["title"],
        "symptom": converted["symptom"],
        "report": converted["report"],
    }


@router.get("/trees/{tree_id}/lint")
async def lint(
    tree_id: int,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_authoring_service import get_tree_draft
    from app.services.guided_lint_service import lint_tree

    draft = get_tree_draft(session, tree_id)
    return {"issues": [i.model_dump() for i in lint_tree(draft)]}


@router.get("/trees/{tree_id}/versions")
async def versions(
    tree_id: int,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    rows = session.exec(
        select(GuidedTreeVersion).where(GuidedTreeVersion.tree_id == tree_id)
    ).all()
    rows = sorted(rows, key=lambda r: r.version, reverse=True)
    return {
        "versions": [
            {
                "version": r.version,
                "note": r.note,
                "published_at": r.published_at.isoformat() if r.published_at else None,
                "published_by": r.published_by,
                "node_count": len((r.snapshot or {}).get("nodes") or {}),
            }
            for r in rows
        ]
    }


@router.post("/trees/{tree_id}/generate-draft")
async def generate_llm_draft(
    tree_id: int,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    """Brouillon LLM depuis la doc structurée (builder existant) — écrase le draft
    courant de CET arbre. Jamais publié automatiquement."""
    tree = session.get(GuidedTree, tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Arbre introuvable")
    if not tree.entry_symptom or tree.space_id is None:
        raise HTTPException(
            status_code=422,
            detail="Un symptôme d'entrée et un espace sont requis pour générer un brouillon.",
        )
    from app.services.authored_tree_builder_service import build_diagnostic_tree_for_symptom
    from app.services.guided_authoring_service import export_tree, get_tree_draft, save_tree_draft

    generated = build_diagnostic_tree_for_symptom(
        session, space_id=tree.space_id, symptom=tree.entry_symptom
    )
    if generated is None:
        raise HTTPException(
            status_code=404,
            detail="Pas assez de matière documentaire structurée pour ce symptôme.",
        )
    # Recopier les nœuds générés dans l'arbre courant puis supprimer l'arbre temporaire.
    payload = export_tree(session, generated.id)
    payload["meta"]["title"] = tree.title
    payload["meta"]["entry_symptom"] = tree.entry_symptom
    save_tree_draft(session, tree.id, payload, current_user.id)
    session.delete(generated)
    session.commit()
    return get_tree_draft(session, tree.id)


# ---------------------------------------------------------------------------
# Symptômes + alias
# ---------------------------------------------------------------------------


class SymptomCreateRequest(BaseModel):
    slug: str
    label: str
    description: str = ""


class AliasRequest(BaseModel):
    alias: str


@router.get("/symptoms")
async def list_symptoms(
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_entry_index_service import symptom_aliases, symptom_catalog

    catalog = symptom_catalog(session)
    aliases = symptom_aliases(session)
    trees = session.exec(select(GuidedTree)).all()
    by_symptom: Dict[str, int] = {}
    for t in trees:
        if t.entry_symptom:
            by_symptom[t.entry_symptom] = by_symptom.get(t.entry_symptom, 0) + 1
    return {
        "symptoms": [
            {
                "slug": slug,
                "label": info["label"],
                "description": info.get("description", ""),
                "aliases": aliases.get(slug, []),
                "tree_count": by_symptom.get(slug, 0),
            }
            for slug, info in sorted(catalog.items(), key=lambda kv: kv[1]["label"])
        ]
    }


@router.post("/symptoms", status_code=201)
async def create_symptom(
    request: SymptomCreateRequest,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    slug = request.slug.strip().lower().replace(" ", "_")[:64]
    existing = session.exec(select(DocumentCategory).where(DocumentCategory.slug == slug)).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Le slug « {slug} » existe déjà")
    cat = DocumentCategory(
        slug=slug, label=request.label[:200], description=request.description, axis="symptom"
    )
    session.add(cat)
    session.commit()

    from app.services.guided_entry_index_service import rebuild_symptom_entries

    rebuild_symptom_entries(session)
    return {"slug": slug, "label": cat.label}


@router.post("/symptoms/{slug}/aliases", status_code=201)
async def add_alias(
    slug: str,
    request: AliasRequest,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    alias = request.alias.strip()[:200]
    if not alias:
        raise HTTPException(status_code=422, detail="Alias vide")
    existing = session.exec(
        select(GuidedSymptomAlias).where(
            GuidedSymptomAlias.symptom_slug == slug, GuidedSymptomAlias.alias == alias
        )
    ).first()
    if existing is None:
        session.add(GuidedSymptomAlias(symptom_slug=slug, alias=alias))
        session.commit()

    from app.services.guided_entry_index_service import rebuild_symptom_entries

    rebuild_symptom_entries(session)
    return {"ok": True}


@router.delete("/symptoms/{slug}/aliases", status_code=204)
async def delete_alias(
    slug: str,
    alias: str,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    row = session.exec(
        select(GuidedSymptomAlias).where(
            GuidedSymptomAlias.symptom_slug == slug, GuidedSymptomAlias.alias == alias
        )
    ).first()
    if row is not None:
        session.delete(row)
        session.commit()
        from app.services.guided_entry_index_service import rebuild_symptom_entries

        rebuild_symptom_entries(session)


# ---------------------------------------------------------------------------
# Angles morts (gaps)
# ---------------------------------------------------------------------------


class GapStatusRequest(BaseModel):
    status: str


@router.get("/gaps")
async def list_gaps_endpoint(
    space_id: Optional[int] = None,
    gap_status: str = "",
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_gap_service import list_gaps

    gaps = list_gaps(session, space_id=space_id, status=gap_status)
    return {
        "gaps": [
            {
                "id": g.id,
                "space_id": g.space_id,
                "detected_symptom": g.detected_symptom,
                "query_text": g.query_text,
                "count": g.count,
                "status": g.status,
                "first_seen": g.first_seen.isoformat() if g.first_seen else None,
                "last_seen": g.last_seen.isoformat() if g.last_seen else None,
                "sample_conversation_ids": g.sample_conversation_ids or [],
            }
            for g in gaps
        ]
    }


@router.patch("/gaps/{gap_id}")
async def patch_gap(
    gap_id: int,
    request: GapStatusRequest,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    from app.services.guided_gap_service import update_gap_status

    gap = update_gap_status(session, gap_id, request.status)
    if gap is None:
        raise HTTPException(status_code=404, detail="Gap introuvable")
    return {"ok": True, "status": gap.status}


# ---------------------------------------------------------------------------
# Picker bibliothèque : recherche de documents + vignettes de pages
# ---------------------------------------------------------------------------


@router.get("/library/documents")
async def search_documents(
    q: str = "",
    limit: int = 20,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    """Recherche simple par titre pour le picker de pièces jointes."""
    stmt = select(Document)
    docs = session.exec(stmt).all()
    needle = (q or "").strip().lower()
    out = []
    for d in docs:
        if needle and needle not in (d.title or "").lower():
            continue
        out.append(
            {
                "id": d.id,
                "title": d.title,
                "source": d.source,
                "has_file": bool(d.source_file_path),
                "proferm_gammes": d.proferm_gammes or [],
                "materials": d.materials or [],
            }
        )
        if len(out) >= max(1, min(limit, 50)):
            break
    return {"documents": out}


@router.get("/library/documents/{document_id}/page-count")
async def page_count(
    document_id: int,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    doc = session.get(Document, document_id)
    if doc is None or not doc.source_file_path or not os.path.exists(doc.source_file_path):
        raise HTTPException(status_code=404, detail="Document ou fichier introuvable")
    import fitz

    with fitz.open(doc.source_file_path) as pdf:
        return {"document_id": document_id, "page_count": pdf.page_count}


@router.get("/library/documents/{document_id}/page-thumbnail/{page_no}")
async def page_thumbnail(
    document_id: int,
    page_no: int,
    current_user: UserRead = Depends(require_sav_editor),
    session: Session = Depends(get_session),
):
    """Vignette JPEG d'une page (PyMuPDF, cache disque)."""
    doc = session.get(Document, document_id)
    if doc is None or not doc.source_file_path or not os.path.exists(doc.source_file_path):
        raise HTTPException(status_code=404, detail="Document ou fichier introuvable")

    cache_dir = os.path.join(THUMBS_CACHE_DIR, str(document_id))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{page_no}.jpg")
    if not os.path.exists(cache_path):
        import fitz

        with fitz.open(doc.source_file_path) as pdf:
            if page_no < 1 or page_no > pdf.page_count:
                raise HTTPException(status_code=404, detail="Page hors limites")
            page = pdf.load_page(page_no - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(0.35, 0.35))
            pix.save(cache_path, jpg_quality=70)
    return FileResponse(cache_path, media_type="image/jpeg")
