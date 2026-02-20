from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import FileResponse
from sqlmodel import Session
from app.database import get_session
from app.models.note import NoteCreate, NoteRead, NoteUpdate, Note
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services.note_service import (
    get_notes_by_project,
    get_note_by_id,
    create_note,
    update_note,
    delete_note
)
from app.services.document_service import process_document, save_uploaded_file, process_document_async
from pydantic import BaseModel
from pathlib import Path
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


@router.get("/notes/{note_id}/file")
async def get_note_file(
    note_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupérer le fichier source d'une note (PDF, etc.)"""
    note = session.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )
    
    if note.note_type != "document" or not note.source_file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Cette note n'a pas de fichier source"
        )
    
    file_path = Path(note.source_file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Le fichier source n'existe plus"
        )
    
    media_type = "application/pdf" if file_path.suffix.lower() == ".pdf" else "application/octet-stream"
    return FileResponse(
        path=str(file_path),
        filename=f"{note.title}{file_path.suffix}",
        media_type=media_type
    )


class NoteInfoResponse(BaseModel):
    id: int
    title: str
    note_type: str
    has_source_file: bool
    source_file_path: Optional[str] = None


@router.get("/notes/{note_id}/info", response_model=NoteInfoResponse)
async def get_note_info(
    note_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupérer les informations d'une note (type, fichier source)"""
    note = session.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )
    
    return NoteInfoResponse(
        id=note.id,
        title=note.title,
        note_type=note.note_type,
        has_source_file=bool(note.source_file_path),
        source_file_path=note.source_file_path if note.note_type == "document" else None
    )


@router.get("/images/{note_id}/{filename}")
async def get_note_image(
    note_id: int,
    filename: str,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Récupérer une image extraite d'un document (multimodal).
    
    Les images sont stockées dans media/images/{note_id}/ lors de l'extraction Docling.
    """
    note = session.get(Note, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Note non trouvée"
        )
    
    # Sécurité : empêcher path traversal
    safe_filename = Path(filename).name
    if safe_filename != filename or ".." in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nom de fichier invalide"
        )
    
    image_path = Path(f"media/images/{note_id}/{safe_filename}")
    if not image_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image non trouvée"
        )
    
    # Déterminer le type MIME
    suffix = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = mime_types.get(suffix, "image/png")
    
    return FileResponse(
        path=str(image_path),
        media_type=media_type,
        filename=safe_filename
    )


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


@router.post("/projects/{project_id}/documents", response_model=List[NoteRead], status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: int,
    files: List[UploadFile] = File(...),
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Uploader un ou plusieurs documents. Les documents seront traités en arrière-plan.
    Des notes sont créées immédiatement avec le statut 'pending' et le contenu "Traitement en cours...".
    L'upload est optimisé pour être non-bloquant et ne pas saturer les ressources.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Aucun fichier fourni"
        )
    
    import os
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    created_notes = []
    errors = []
    
    # Utiliser un pool de threads pour les opérations I/O lourdes (sauvegarde de fichiers)
    # Cela permet de ne pas bloquer le thread principal FastAPI
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=min(len(files), 4))  # Limiter à 4 workers max pour l'upload
    
    async def process_single_file(file: UploadFile):
        """Traiter un seul fichier de manière asynchrone"""
        filename = file.filename or "fichier_inconnu"
        try:
            # Lire le contenu du fichier (déjà asynchrone)
            file_content = await file.read()
            
            # Sauvegarder le fichier dans un thread séparé pour ne pas bloquer
            file_path = await loop.run_in_executor(
                executor,
                save_uploaded_file,
                file_content,
                filename
            )
            
            if not file_path:
                return (None, f"Erreur lors de la sauvegarde du fichier '{filename}'")
            
            # Créer une note immédiatement avec statut 'pending'
            # Utiliser le nom du fichier comme titre (sans l'extension)
            filename_without_ext = os.path.splitext(filename)[0]
            
            note_create = NoteCreate(
                title=filename_without_ext,
                content="⏳ Traitement en cours...",
                note_type="document",
                source_file_path=file_path,
                processing_status="pending",
                processing_progress=0,
            )
            
            logger.info(f"Création de la note pour le document '{filename}' dans le projet {project_id}")
            # La création de note doit être faite dans le thread principal pour éviter les problèmes de session
            # On retourne les données nécessaires et on créera la note dans le thread principal
            return (file_path, note_create, filename)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'upload du document '{filename}': {e}", exc_info=True)
            return (None, f"Erreur lors de l'upload de '{filename}': {str(e)}")
    
    # Traiter tous les fichiers en parallèle (mais avec une limite)
    tasks = [process_single_file(file) for file in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Nettoyer l'executor
    executor.shutdown(wait=False)
    
    # Créer les notes dans le thread principal (rapide, pas besoin de paralléliser)
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Exception lors du traitement d'un fichier: {result}", exc_info=True)
            errors.append(f"Exception: {str(result)}")
            continue
        
        if result[0] is None:  # Erreur
            errors.append(result[1])
            continue
        
        # Résultat valide : (file_path, note_create, filename)
        file_path, note_create, filename = result
        
        try:
            note = create_note(session, note_create, project_id, current_user.id)
            if not note:
                errors.append(f"Projet non trouvé pour le fichier '{filename}'")
                continue
            
            # Ajouter le traitement à la file d'attente (non-bloquant)
            # Cette fonction est déjà non-bloquante, elle ajoute juste à la queue
            process_document_async(note.id, file_path)
            
            created_notes.append(note)
            logger.info(f"Document '{filename}' uploadé, traitement en arrière-plan (note ID: {note.id})")
        except Exception as e:
            logger.error(f"Erreur lors de la création de la note pour '{filename}': {e}", exc_info=True)
            errors.append(f"Erreur lors de la création de la note pour '{filename}': {str(e)}")
    
    # Si aucun fichier n'a pu être traité
    if not created_notes:
        if errors:
            error_message = "; ".join(errors)
        else:
            error_message = "Aucun fichier n'a pu être uploadé"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message
        )
    
    # Si certains fichiers ont échoué, on retourne quand même les notes créées
    # mais on log les erreurs
    if errors:
        logger.warning(f"Certains fichiers n'ont pas pu être uploadés: {errors}")
    
    return [NoteRead.model_validate(note) for note in created_notes]


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

