from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
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
from app.services.chunk_service import get_chunks_by_note
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["notes"])


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
    """Supprimer une note"""
    success = delete_note(session, note_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )


class EmbeddingStatus(BaseModel):
    """Statut des embeddings d'une note"""
    note_id: int
    total_chunks: int
    chunks_with_embeddings: int
    chunks_without_embeddings: int
    status: str  # 'completed', 'processing', 'pending', 'none'


@router.get("/notes/{note_id}/embedding-status", response_model=EmbeddingStatus)
async def get_note_embedding_status(
    note_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Vérifier l'état des embeddings d'une note"""
    note = get_note_by_id(session, note_id, current_user.id)
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )
    
    # Récupérer tous les chunks de la note
    chunks = get_chunks_by_note(session, note_id)
    
    total_chunks = len(chunks)
    chunks_with_embeddings = sum(1 for chunk in chunks if chunk.embedding is not None)
    chunks_without_embeddings = total_chunks - chunks_with_embeddings
    
    # Déterminer le statut
    if total_chunks == 0:
        status = 'none'
    elif chunks_without_embeddings == 0:
        status = 'completed'
    elif chunks_with_embeddings > 0:
        status = 'processing'  # En cours, certains embeddings sont déjà générés
    else:
        status = 'pending'  # Aucun embedding généré encore
    
    # Logger pour déboguer
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"📊 Statut embedding pour note {note_id}: {status} (chunks: {total_chunks}, avec embeddings: {chunks_with_embeddings}, sans: {chunks_without_embeddings})")
    
    return EmbeddingStatus(
        note_id=note_id,
        total_chunks=total_chunks,
        chunks_with_embeddings=chunks_with_embeddings,
        chunks_without_embeddings=chunks_without_embeddings,
        status=status
    )

