from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlmodel import Session
from app.database import get_session
from app.models.space import SpaceCreate, SpaceRead, SpaceUpdate
from app.models.document import DocumentListItem
from app.models.user import UserRead
from app.routers.auth import get_current_user, require_permission
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
    current_user: UserRead = Depends(require_permission("space.create")),
    session: Session = Depends(get_session)
):
    """Crée un nouvel espace."""
    is_admin = "admin" in (current_user.roles or [])
    if space_create.is_shared and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seuls les administrateurs peuvent créer des espaces communs."
        )
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
    current_user: UserRead = Depends(require_permission("space.update")),
    session: Session = Depends(get_session)
):
    """Met à jour un espace."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )
    
    is_admin = "admin" in (current_user.roles or [])
    
    # Restriction : Seuls les admins peuvent modifier un espace commun
    if space.is_shared and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seuls les administrateurs peuvent modifier un espace commun."
        )
        
    # Restriction : Un non-admin ne peut pas modifier un espace privé qui ne lui appartient pas
    if not space.is_shared and space.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'êtes pas autorisé à modifier cet espace privé."
        )

    # Restriction : Un non-admin ne peut pas transformer son espace privé en espace commun
    if space_update.is_shared is True and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seuls les administrateurs peuvent rendre un espace commun."
        )
        
    space = update_space(session, space_id, space_update, current_user.id)
    return SpaceRead.model_validate(space)


@router.delete("/{space_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_space(
    space_id: int,
    current_user: UserRead = Depends(require_permission("space.delete")),
    session: Session = Depends(get_session)
):
    """Supprime un espace et ses associations."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé"
        )
        
    is_admin = "admin" in (current_user.roles or [])
    
    # Restriction : Seuls les admins peuvent supprimer un espace commun
    if space.is_shared and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seuls les administrateurs peuvent supprimer un espace commun."
        )
        
    # Restriction : Un non-admin ne peut pas supprimer un espace privé qui ne lui appartient pas
    if not space.is_shared and space.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Vous n'êtes pas autorisé à supprimer cet espace privé."
        )
        
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

