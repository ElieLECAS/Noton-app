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
from pydantic import BaseModel

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
    """Supprimer une note"""
    success = delete_note(session, note_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )


@router.get("/notes/{note_id}/embedding-status", response_model=EmbeddingStatus)
async def get_note_embedding_status(
    note_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Vérifier l'état des embeddings d'une note (compatibilité frontend).
    
    Dans la nouvelle architecture, les embeddings sont toujours générés immédiatement.
    Cet endpoint retourne simplement si l'embedding est présent ou non.
    """
    note = get_note_by_id(session, note_id, current_user.id)
    if not note:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )
    
    # Dans la nouvelle architecture, on vérifie simplement si l'embedding existe
    has_embedding = note.embedding is not None
    status_value = 'completed' if has_embedding else 'none'
    
    return EmbeddingStatus(
        note_id=note_id,
        total_chunks=0,
        chunks_with_embeddings=1 if has_embedding else 0,
        chunks_without_embeddings=0 if has_embedding else 1,
        status=status_value
    )

