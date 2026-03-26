from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.database import get_session
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services.kag_graph_service import (
    get_space_bipartite_graph,
    get_space_kag_stats,
    rebuild_kag_for_space,
)
from app.services.space_service import get_space_by_id


router = APIRouter(prefix="/api/kag", tags=["kag"])


def _ensure_space_access(session: Session, space_id: int, current_user: UserRead):
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé",
        )


@router.get("/spaces/{space_id}/stats", response_model=Dict)
async def get_space_kag_stats_route(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Statistiques KAG d'un espace (entités, relations, types)."""
    _ensure_space_access(session, space_id, current_user)
    return get_space_kag_stats(session, space_id)


@router.get("/spaces/{space_id}/graph", response_model=Dict)
async def get_space_kag_graph_route(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Graphe biparti KAG (entités <-> chunks) pour un espace."""
    _ensure_space_access(session, space_id, current_user)
    return get_space_bipartite_graph(session=session, space_id=space_id)


@router.post("/spaces/{space_id}/rebuild", response_model=Dict)
async def rebuild_space_kag_route(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Purge puis reconstruit le KAG pour tous les documents de l'espace."""
    _ensure_space_access(session, space_id, current_user)
    return rebuild_kag_for_space(session, space_id)
