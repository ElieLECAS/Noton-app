from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select
from app.database import get_session
from app.models.library import LibraryRead, LibraryStats
from app.models.folder import FolderCreate, FolderRead, FolderUpdate, FolderWithContents
from app.models.document import DocumentRead, DocumentListItem, DocumentCreate, DocumentUpdate, Document
from app.models.folder import Folder
from app.models.space import SpaceRead
from app.models.user import UserRead
from app.routers.auth import get_current_user, require_permission
from app.services.library_service import get_or_create_user_library, get_library_stats
from app.services.folder_service import (
    create_folder, get_folder_by_id, get_folders_by_parent,
    get_folder_path, get_folder_with_contents, rename_folder,
    move_folder, delete_folder
)
from app.services.document_service_new import (
    create_document, get_document_by_id, get_documents_by_folder,
    get_documents_by_library, save_uploaded_file, process_document_async,
    add_document_to_spaces, remove_document_from_spaces, move_document, delete_document,
    update_document, reindex_library_document,
)
from app.services.document_space_service import get_spaces_for_document
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/library", tags=["library"])


class DocumentSpacesManageRequest(BaseModel):
    add_space_ids: List[int] = Field(default_factory=list)
    remove_space_ids: List[int] = Field(default_factory=list)


@router.get("", response_model=LibraryRead)
async def get_library(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère la bibliothèque de l'utilisateur."""
    library = get_or_create_user_library(session, current_user.id)
    return LibraryRead.model_validate(library)


@router.get("/stats", response_model=LibraryStats)
async def get_library_statistics(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère les statistiques de la bibliothèque."""
    library = get_or_create_user_library(session, current_user.id)
    stats = get_library_stats(session, library.id, current_user.id)
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Statistiques non disponibles"
        )
    return stats


@router.get("/folders", response_model=List[FolderRead])
async def list_root_folders(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Liste tous les dossiers racine de la bibliothèque."""
    library = get_or_create_user_library(session, current_user.id)
    folders = session.exec(
        select(Folder).where(
            Folder.parent_folder_id.is_(None),
            Folder.library_id == library.id,
        ).order_by(Folder.name)
    ).all()
    return [FolderRead.model_validate(f) for f in folders]


@router.post("/folders", response_model=FolderRead, status_code=status.HTTP_201_CREATED)
async def create_new_folder(
    folder_create: FolderCreate,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Crée un nouveau dossier."""
    library = get_or_create_user_library(session, current_user.id)
    folder = create_folder(session, folder_create, library.id, current_user.id)
    return FolderRead.model_validate(folder)


@router.get("/folders/{folder_id}", response_model=FolderWithContents)
async def get_folder(
    folder_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère un dossier avec ses sous-dossiers et documents."""
    folder = get_folder_with_contents(session, folder_id, current_user.id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dossier non trouvé"
        )
    return folder


@router.get("/folders/{folder_id}/path", response_model=List[FolderRead])
async def get_folder_breadcrumb(
    folder_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère le chemin complet d'un dossier (breadcrumb)."""
    path = get_folder_path(session, folder_id, current_user.id)
    return path


@router.put("/folders/{folder_id}", response_model=FolderRead)
async def update_folder(
    folder_id: int,
    folder_update: FolderUpdate,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Renomme un dossier."""
    folder = rename_folder(session, folder_id, folder_update, current_user.id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dossier non trouvé"
        )
    return FolderRead.model_validate(folder)


@router.post("/folders/{folder_id}/move", response_model=FolderRead)
async def move_folder_to_parent(
    folder_id: int,
    new_parent_id: Optional[int] = None,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Déplace un dossier vers un nouveau parent."""
    folder = move_folder(session, folder_id, new_parent_id, current_user.id)
    if not folder:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de déplacer le dossier"
        )
    return FolderRead.model_validate(folder)


@router.delete("/folders/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder_recursive(
    folder_id: int,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Supprime un dossier et tout son contenu (récursif)."""
    success = delete_folder(session, folder_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dossier non trouvé"
        )


@router.get("/documents", response_model=List[DocumentListItem])
async def list_documents(
    folder_id: Optional[int] = None,
    include_all: bool = False,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Liste les documents d'un dossier, de la racine, ou de toute la bibliothèque."""
    library = get_or_create_user_library(session, current_user.id)
    if include_all:
        documents = get_documents_by_library(session, library.id, current_user.id)
    else:
        documents = session.exec(
            select(Document).where(
                Document.library_id == library.id,
                Document.folder_id == folder_id,
            ).order_by(Document.created_at.desc())
        ).all()
    return [DocumentListItem.model_validate(d) for d in documents]


@router.get("/documents/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère un document par son ID."""
    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    return DocumentRead.model_validate(document)


@router.put("/documents/{document_id}", response_model=DocumentRead)
async def update_library_document(
    document_id: int,
    document_update: DocumentUpdate,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Met à jour les métadonnées d'un document."""
    document = update_document(session, document_id, document_update, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    return DocumentRead.model_validate(document)


@router.post("/documents/{document_id}/reindex", status_code=status.HTTP_200_OK)
async def reindex_library_document_endpoint(
    document_id: int,
    current_user: UserRead = Depends(require_permission("library.write")),
):
    """
    Re-extrait le texte, rechunke (Docling hiérarchique ou fallback) et ré-embed
    sans réupload — le fichier source doit exister (media/documents).
    """
    try:
        return reindex_library_document(document_id, current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/documents/{document_id}/file")
async def get_document_file(
    document_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupère le fichier source d'un document."""
    library = get_or_create_user_library(session, current_user.id)
    document = session.exec(
        select(Document).where(
            Document.id == document_id,
            Document.library_id == library.id,
        )
    ).first()
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    
    if document.document_type != "document" or not document.source_file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ce document n'a pas de fichier source"
        )
    
    file_path = Path(document.source_file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Le fichier source n'existe plus"
        )
    
    media_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else "application/octet-stream"
    
    response = FileResponse(
        path=str(file_path),
        filename=f"{document.title}{file_path.suffix}",
        media_type=media_type
    )
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Range, Accept-Ranges"
    
    return response


@router.post("/upload", response_model=List[DocumentRead], status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: List[UploadFile] = File(...),
    space_ids: str = Form("[]"),
    is_paid: bool = Form(False),
    folder_id: Optional[int] = Form(None),
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """
    Upload unifié de documents avec sélection d'espaces.
    
    - files: Liste de fichiers à uploader
    - space_ids: JSON array d'IDs d'espaces (ex: "[1,2,3]")
    - folder_id: ID du dossier destination (optionnel, None = racine)
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucun fichier fourni"
        )
    
    try:
        space_ids_list = json.loads(space_ids)
        if not isinstance(space_ids_list, list):
            raise ValueError("space_ids doit être un tableau")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Format space_ids invalide: {str(e)}"
        )
    
    library = get_or_create_user_library(session, current_user.id)
    
    created_documents = []
    errors = []
    
    import os

    # Traitement strictement séquentiel, fichier par fichier.
    for file in files:
        filename = file.filename or "fichier_inconnu"
        try:
            file_content = await file.read()

            file_path = save_uploaded_file(file_content, filename)

            if not file_path:
                errors.append(f"Erreur lors de la sauvegarde du fichier '{filename}'")
                continue

            filename_without_ext = os.path.splitext(filename)[0]

            document_create = DocumentCreate(
                title=filename_without_ext,
                content="⏳ Traitement en cours...",
                document_type="document",
                source_file_path=file_path,
                processing_status="pending",
                processing_progress=0,
                is_paid=is_paid,
                folder_id=folder_id
            )
            
            document = create_document(
                session,
                document_create,
                library.id,
                current_user.id,
                space_ids_list
            )
            
            if not document:
                errors.append(f"Impossible de créer le document pour '{filename}'")
                continue
            
            process_document_async(document.id, file_path)
            
            created_documents.append(document)
            logger.info(f"Document '{filename}' uploadé, traitement en arrière-plan (document ID: {document.id})")
        except Exception as e:
            logger.error(
                f"Erreur lors du traitement du document '{filename}': {e}",
                exc_info=True,
            )
            errors.append(f"Erreur lors du traitement de '{filename}': {str(e)}")
    
    if not created_documents:
        error_message = "; ".join(errors) if errors else "Aucun fichier n'a pu être uploadé"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )
    
    if errors:
        logger.warning(f"Certains fichiers n'ont pas pu être uploadés: {errors}")
    
    return [DocumentRead.model_validate(doc) for doc in created_documents]


@router.get("/documents/{document_id}/spaces", response_model=List[SpaceRead])
async def list_document_spaces(
    document_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Liste les espaces ayant accès à un document."""
    document = get_document_by_id(session, document_id, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    
    spaces = get_spaces_for_document(session, document_id, current_user.id)
    return [SpaceRead.model_validate(s) for s in spaces]


@router.post("/documents/{document_id}/spaces", status_code=status.HTTP_200_OK)
async def manage_document_spaces(
    document_id: int,
    payload: DocumentSpacesManageRequest,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Ajoute ou retire un document de plusieurs espaces."""
    document = get_document_by_id(session, document_id, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
    
    if payload.add_space_ids:
        success = add_document_to_spaces(session, document_id, payload.add_space_ids, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Erreur lors de l'ajout aux espaces"
            )
    
    if payload.remove_space_ids:
        success = remove_document_from_spaces(session, document_id, payload.remove_space_ids, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Erreur lors du retrait des espaces"
            )
    
    return {"message": "Espaces mis à jour avec succès"}


@router.post("/documents/{document_id}/move", response_model=DocumentRead)
async def move_document_to_folder(
    document_id: int,
    new_folder_id: Optional[int] = None,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Déplace un document vers un nouveau dossier."""
    document = move_document(session, document_id, new_folder_id, current_user.id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de déplacer le document"
        )
    return DocumentRead.model_validate(document)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_library_document(
    document_id: int,
    current_user: UserRead = Depends(require_permission("library.write")),
    session: Session = Depends(get_session)
):
    """Supprime un document de la bibliothèque et ses données associées."""
    success = delete_document(session, document_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document non trouvé"
        )
