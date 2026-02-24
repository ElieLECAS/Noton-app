from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session

from app.database import get_session
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services.project_service import get_project_by_id
from app.services.kag_graph_service import get_kag_stats, get_project_bipartite_graph


router = APIRouter(prefix="/api/kag", tags=["kag"])


def _ensure_project_access(
    session: Session,
    project_id: int,
    current_user: UserRead,
):
    project = get_project_by_id(session, project_id, current_user.id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé",
        )


@router.get("/projects/{project_id}/stats", response_model=Dict)
async def get_project_kag_stats(
    project_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Statistiques KAG pour un projet (nombre d'entités, de relations, etc.).
    """
    _ensure_project_access(session, project_id, current_user)
    return get_kag_stats(session, project_id)


@router.get("/projects/{project_id}/graph", response_model=Dict)
async def get_project_kag_graph(
    project_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """
    Graphe biparti (entités <-> chunks) pour visualisation front.
    """
    _ensure_project_access(session, project_id, current_user)
    return get_project_bipartite_graph(
        session=session,
        project_id=project_id,
        user_id=current_user.id,
    )

