"""Le wiki servi à l'interface : graphe, pages, PDF sources, statistiques, dépôt."""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.models.user import UserRead
from app.routers.auth import get_current_user, require_permission
from app.services import wiki_depot
from app.services.wiki_service import WikiUnavailable, get_snapshot, wiki_root

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


# ---------------------------------------------------------------------------
# Le dépôt : mettre à jour le wiki et joindre les PDF depuis l'administration
# ---------------------------------------------------------------------------
#
# Trois temps, parce qu'on ne fait pas confiance à un envoi de 800 Mo :
#   1. POST /depot         — le navigateur annonce ce qu'il a, le serveur répond le plan
#                            (pages ajoutées, pages qui disparaîtront, PDF manquants) ;
#   2. PUT  /depot/{id}/fichier?chemin=…  — un fichier par requête, corps brut, en flux ;
#   3. POST /depot/{id}/valider — la bascule et le rechargement de l'instantané.
#
# Un fichier par requête, et pas une archive : derrière nginx, c'est la taille du plus gros PDF
# qui fixe ``client_max_body_size``, pas celle du dossier.


class _FichierAnnonce(BaseModel):
    chemin: str
    taille: int = 0


class _Annonce(BaseModel):
    fichiers: List[_FichierAnnonce] = Field(default_factory=list)


@router.post("/depot")
async def wiki_depot_ouvrir(
    annonce: _Annonce,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
):
    """Ouvre un dépôt : renvoie ce que le serveur réclame et ce que ça changera."""
    try:
        return wiki_depot.ouvrir(
            wiki_root(), [f.model_dump() for f in annonce.fichiers]
        )
    except wiki_depot.DepotRefuse as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.put("/depot/{depot_id}/fichier")
async def wiki_depot_fichier(
    depot_id: str,
    request: Request,
    chemin: str = Query(..., description="Le chemin relatif annoncé, tel quel"),
    current_user: UserRead = Depends(require_permission("config.manage_users")),
):
    """Reçoit un fichier annoncé, en flux, corps brut (pas de multipart : rien à tamponner)."""
    try:
        return await wiki_depot.deposer(wiki_root(), depot_id, chemin, request.stream())
    except wiki_depot.DepotInconnu as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except wiki_depot.DepotRefuse as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/depot/{depot_id}/valider")
async def wiki_depot_valider(
    depot_id: str,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
):
    """Bascule le wiki préparé, recharge l'instantané, renvoie le résumé et les statistiques."""
    try:
        return wiki_depot.valider(wiki_root(), depot_id)
    except wiki_depot.DepotInconnu as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except wiki_depot.DepotIncomplet as exc:
        raise HTTPException(
            status_code=409,
            detail={"message": str(exc), "manquants": exc.manquants[:40]},
        )
    except WikiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.delete("/depot/{depot_id}", status_code=204)
async def wiki_depot_annuler(
    depot_id: str,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
):
    """Abandonne un dépôt en cours : le wiki préparé est jeté, rien n'a bougé."""
    try:
        wiki_depot.annuler(wiki_root(), depot_id)
    except wiki_depot.DepotInconnu as exc:
        raise HTTPException(status_code=404, detail=str(exc))
