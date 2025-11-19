from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlmodel import Session
from app.database import get_session
from app.models.note import NoteCreate, NoteRead, NoteUpdate
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services.note_service import (
    get_notes_by_project,
    get_note_by_id,
    create_note,
    update_note,
    delete_note
)
from app.services.document_service import process_document, save_uploaded_file
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["notes"])


class EmbeddingStatus(BaseModel):
    """Statut des embeddings d'une note (compatibilité avec le frontend)"""
    note_id: int
    total_chunks: int = 0  # Plus utilisé (ancienne architecture)
    chunks_with_embeddings: int = 0  # Plus utilisé
    chunks_without_embeddings: int = 0  # Plus utilisé
    status: str  # 'completed' si embedding présent, 'none' sinon


@router.get("/projects/{project_id}/notes", response_model=List[NoteRead])
async def list_notes(
    project_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Liste toutes les notes d'un projet"""
    notes = get_notes_by_project(session, project_id, current_user.id)
    return notes


@router.post("/projects/{project_id}/notes", response_model=NoteRead, status_code=status.HTTP_201_CREATED)
async def create_new_note(
    project_id: int,
    note_create: NoteCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Créer une nouvelle note"""
    note = create_note(session, note_create, project_id, current_user.id)
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Projet non trouvé"
        )
    return NoteRead.model_validate(note)


@router.get("/notes/{note_id}", response_model=NoteRead)
async def get_note(
    note_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupérer une note par son ID"""
    note = get_note_by_id(session, note_id, current_user.id)
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )
    return NoteRead.model_validate(note)


@router.put("/notes/{note_id}", response_model=NoteRead)
async def update_existing_note(
    note_id: int,
    note_update: NoteUpdate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Mettre à jour une note"""
    note = update_note(session, note_id, note_update, current_user.id)
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )
    return NoteRead.model_validate(note)


@router.delete("/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_existing_note(
    note_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Supprimer une note et ses chunks associés"""
    try:
        success = delete_note(session, note_id, current_user.id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Note non trouvée"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la note {note_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la suppression de la note: {str(e)}"
        )


@router.post("/projects/{project_id}/documents", response_model=NoteRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: int,
    file: UploadFile = File(...),
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Uploader et traiter un document avec docling.
    Le document est converti en markdown et créé comme une note.
    """
    try:
        # Lire le contenu du fichier
        file_content = await file.read()
        
        # Sauvegarder le fichier temporairement
        file_path = save_uploaded_file(file_content, file.filename)
        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur lors de la sauvegarde du fichier"
            )
        
        # Traiter le document avec docling
        markdown_content = process_document(file_path)
        if not markdown_content:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur lors du traitement du document"
            )
        
        # Créer une note avec le contenu markdown
        # Utiliser le nom du fichier comme titre (sans l'extension)
        import os
        filename_without_ext = os.path.splitext(file.filename)[0]
        
        note_create = NoteCreate(
            title=filename_without_ext,
            content=markdown_content,
            note_type="document",
            source_file_path=file_path
        )
        
        note = create_note(session, note_create, project_id, current_user.id)
        if not note:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Projet non trouvé"
            )
        
        logger.info(f"Document '{file.filename}' uploadé et traité avec succès (note ID: {note.id})")
        return NoteRead.model_validate(note)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'upload du document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement du document: {str(e)}"
        )


@router.get("/notes/{note_id}/embedding-status", response_model=EmbeddingStatus)
async def get_note_embedding_status(
    note_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Vérifier l'état des embeddings d'une note (compatibilité frontend).
    
    Dans la nouvelle architecture, les embeddings sont générés par chunk.
    Cet endpoint retourne le statut des chunks.
    """
    from app.services.chunk_service import get_chunks_by_note
    
    note = get_note_by_id(session, note_id, current_user.id)
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )
    
    # Récupérer les chunks de la note
    chunks = get_chunks_by_note(session, note_id)
    total_chunks = len(chunks)
    chunks_with_embeddings = sum(1 for chunk in chunks if chunk.embedding is not None)
    chunks_without_embeddings = total_chunks - chunks_with_embeddings
    
    # Déterminer le statut global
    if total_chunks == 0:
        status_value = 'none'
    elif chunks_without_embeddings == 0:
        status_value = 'completed'
    elif chunks_with_embeddings > 0:
        status_value = 'processing'
    else:
        status_value = 'pending'
    
    return EmbeddingStatus(
        note_id=note_id,
        total_chunks=total_chunks,
        chunks_with_embeddings=chunks_with_embeddings,
        chunks_without_embeddings=chunks_without_embeddings,
        status=status_value
    )

