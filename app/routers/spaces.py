from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from app.database import get_session
from app.models.space import SpaceCreate, SpaceRead, SpaceUpdate
from app.models.document import DocumentListItem
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services.space_service import (
    create_space, get_spaces_by_user, get_space_by_id,
    update_space, delete_space
)
from app.services.document_service_new import get_documents_by_space
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spaces", tags=["spaces"])


@router.get("", response_model=List[SpaceRead])
async def list_spaces(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Liste tous les espaces de l'utilisateur."""
    spaces = get_spaces_by_user(session, current_user.id)
    return [SpaceRead.model_validate(s) for s in spaces]


@router.post("", response_model=SpaceRead, status_code=status.HTTP_201_CREATED)
async def create_new_space(
    space_create: SpaceCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Crée un nouvel espace."""
    space = create_space(session, space_create, current_user.id)
    return SpaceRead.model_validate(space)


@router.get("/{space_id}", response_model=SpaceRead)
async def get_space(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère un espace par son ID."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )
    return SpaceRead.model_validate(space)


@router.put("/{space_id}", response_model=SpaceRead)
async def update_existing_space(
    space_id: int,
    space_update: SpaceUpdate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Met à jour un espace."""
    space = update_space(session, space_id, space_update, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )
    return SpaceRead.model_validate(space)


@router.delete("/{space_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_space(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Supprime un espace et ses associations."""
    success = delete_space(session, space_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )


@router.get("/{space_id}/documents", response_model=List[DocumentListItem])
async def list_space_documents(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Liste tous les documents accessibles dans un espace."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )
    
    documents = get_documents_by_space(session, space_id, current_user.id)
    return [DocumentListItem.model_validate(d) for d in documents]


@router.get("/{space_id}/kag/stats")
async def get_space_kag_stats(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère les statistiques KAG d'un espace."""
    from app.services.kag_graph_service import get_space_kag_stats as get_space_kag_stats_service
    
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )
    
    stats = get_space_kag_stats_service(session, space_id)
    return stats


@router.get("/{space_id}/kag/graph")
async def get_space_kag_graph(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère le graphe KAG d'un espace."""
    from app.services.kag_graph_service import get_space_bipartite_graph
    
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )
    
    graph = get_space_bipartite_graph(session, space_id)
    return graph


@router.post("/{space_id}/kag/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def rebuild_space_kag(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Reconstruit le KAG d'un espace."""
    from app.services.kag_graph_service import rebuild_kag_for_space
    
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )
    
    rebuild_kag_for_space(session, space_id)
    return {"message": "Reconstruction du KAG en cours"}
