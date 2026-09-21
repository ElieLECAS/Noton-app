"""Le wiki servi à l'interface : graphe, pages, PDF sources, statistiques."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.models.user import UserRead
from app.routers.auth import get_current_user, require_permission
from app.services.wiki_service import WikiUnavailable, get_snapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/wiki", tags=["wiki"])


def _snapshot():
    try:
        return get_snapshot()
    except WikiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/graph")
async def wiki_graph(current_user: UserRead = Depends(get_current_user)):
    """Nœuds (sans corps) et liens : le contrat de l'ancien ``graph.json``."""
    return _snapshot().graph_payload()


@router.get("/pages/{path:path}")
async def wiki_page(path: str, current_user: UserRead = Depends(get_current_user)):
    """Une page par son chemin OKF (``profiles/perform76-parcloses.md``, avec ou sans ``/``).

    Le chemin est cherché dans le dictionnaire de l'instantané, jamais résolu sur disque :
    ``..`` et consorts tombent en 404 comme n'importe quel chemin inconnu."""
    page_id = "/" + path.strip("/")
    payload = _snapshot().page_payload(page_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Page inconnue du wiki")
    return payload


@router.get("/search")
async def wiki_search(q: str = "", current_user: UserRead = Depends(get_current_user)):
    """Les pages où tous les mots de ``q`` apparaissent, corps compris.

    L'accueil du wiki s'en sert pour retrouver une référence citée dans un tableau
    (``TGY3704``) : le filtre du navigateur ne voit que titres et métadonnées."""
    return {"q": q, "pages": _snapshot().search(q)}


@router.get("/raw/{name}")
async def wiki_raw(name: str, current_user: UserRead = Depends(get_current_user)):
    """Un PDF source de ``raw/``, par son nom exact (liste blanche des fichiers présents)."""
    path = _snapshot().raw_path(name)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Document source introuvable")
    return FileResponse(
        path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{name}"'},
    )


@router.get("/stats")
async def wiki_stats(current_user: UserRead = Depends(require_permission("config.manage_users"))):
    """La carte d'administration : pages, liens, lint, budget de contexte, dernier appel."""
    return _snapshot().stats()
