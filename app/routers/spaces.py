from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
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
from app.services.kag_graph_service import build_space_kag_graph
from app.services.space_category_service import (
    get_space_categories,
    get_space_category_page_detail,
    get_space_category_pages,
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spaces", tags=["spaces"])


class SpaceCategoryItem(BaseModel):
    category_id: int
    slug: str
    label: str
    chunk_count: int = 0
    page_count: int = 0
    document_count: int = 0
    max_confidence: float = 0.0


class SpaceCategoriesResponse(BaseModel):
    space_id: int
    category_count: int = 0
    categories: List[SpaceCategoryItem] = Field(default_factory=list)
    status: str = "ok"


class SpaceCategoryRef(BaseModel):
    category_id: int
    slug: str
    label: str


class SpaceCategoryPageItem(BaseModel):
    document_id: int
    document_title: str
    page_no: int
    chunk_count: int = 0
    has_source_file: bool = False


class SpaceCategoryPagesResponse(BaseModel):
    space_id: int
    category: SpaceCategoryRef
    page_count: int = 0
    pages: List[SpaceCategoryPageItem] = Field(default_factory=list)


class SpaceCategoryPageChunkItem(BaseModel):
    chunk_id: Optional[int] = None
    chunk_index: Optional[int] = None
    heading: Optional[str] = None
    step_number: Optional[int] = None
    section_type: Optional[str] = None
    content: str
    in_category: bool = False
    confidence: Optional[float] = None


class SpaceCategoryEnrichmentChunkItem(BaseModel):
    chunk_id: Optional[int] = None
    chunk_index: Optional[int] = None
    theme: Optional[str] = None
    category_slug: Optional[str] = None
    source_page: Optional[int] = None
    source_pages: List[int] = Field(default_factory=list)
    content: str
    confidence: Optional[float] = None


class SpaceCategoryPageNavRef(BaseModel):
    document_id: int
    document_title: str
    page_no: int
    chunk_count: int = 0
    has_source_file: bool = False


class SpaceCategoryPageNavigation(BaseModel):
    current_index: Optional[int] = None
    total: int = 0
    prev: Optional[SpaceCategoryPageNavRef] = None
    next: Optional[SpaceCategoryPageNavRef] = None


class SpaceCategoryPageDetailResponse(BaseModel):
    space_id: int
    category: SpaceCategoryRef
    document: dict
    page_no: int
    chunks: List[SpaceCategoryPageChunkItem] = Field(default_factory=list)
    enrichment_chunks: List[SpaceCategoryEnrichmentChunkItem] = Field(default_factory=list)
    consolidated_markdown: str = ""
    navigation: SpaceCategoryPageNavigation


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


@router.get("/{space_id}/kag/graph")
async def get_space_kag_graph(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
    max_nodes: int = Query(150, ge=10, le=500),
    max_edges: int = Query(300, ge=10, le=1000),
):
    """Graphe de connaissances KAG (entités + relations) pour visualisation UI."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé",
        )
    return build_space_kag_graph(
        session,
        space_id,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


@router.get("/{space_id}/categories", response_model=SpaceCategoriesResponse)
async def list_space_categories(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Catégories KAG présentes dans l'espace avec nombre de pages associées."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé",
        )
    payload = get_space_categories(session, space_id)
    return SpaceCategoriesResponse.model_validate(payload)


@router.get("/{space_id}/categories/{category_id}/pages", response_model=SpaceCategoryPagesResponse)
async def list_space_category_pages(
    space_id: int,
    category_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Pages de l'espace contenant une catégorie donnée."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé",
        )
    payload = get_space_category_pages(session, space_id, category_id)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Catégorie non trouvée",
        )
    return SpaceCategoryPagesResponse.model_validate(payload)


@router.get(
    "/{space_id}/categories/{category_id}/pages/{document_id}/{page_no}",
    response_model=SpaceCategoryPageDetailResponse,
)
async def get_space_category_page(
    space_id: int,
    category_id: int,
    document_id: int,
    page_no: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Détail d'une page : chunks texte avec surlignage catégorie et navigation."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Espace non trouvé",
        )
    payload = get_space_category_page_detail(
        session,
        space_id,
        category_id,
        document_id,
        page_no,
    )
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Page ou catégorie non trouvée",
        )
    return SpaceCategoryPageDetailResponse.model_validate(payload)

