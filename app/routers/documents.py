"""
Router pour l'upload et la gestion de documents.
Permet d'importer des fichiers (PDF, Excel, Word, images, etc.) comme notes.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlmodel import Session
from app.database import get_session
from app.models.note import Note, NoteRead
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services.project_service import get_project_by_id
from app.services.file_storage_service import get_file_storage_service
from app.services.docling_service import get_docling_service
from app.services.document_chunking_service import get_chunking_service
from app.models.document_chunk import DocumentChunk
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])


@router.post("/projects/{project_id}/notes/upload", response_model=NoteRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    project_id: int,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Importer un document (PDF, Excel, Word, image, etc.) comme note.
    Le document est parsé avec Docling, découpé en chunks intelligents,
    et les embeddings sont générés pour la recherche sémantique.
    
    Args:
        project_id: ID du projet
        file: Fichier uploadé
        title: Titre optionnel (sinon utilise le nom du fichier)
        current_user: Utilisateur authentifié
        session: Session de base de données
        
    Returns:
        Note créée avec le document
    """
    try:
        # 1. Vérifier que le projet appartient à l'utilisateur
        project = get_project_by_id(session, project_id, current_user.id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Projet non trouvé"
            )
        
        # 2. Lire le contenu du fichier
        file_content = await file.read()
        original_filename = file.filename or "document"
        
        logger.info(f"📤 Upload de {original_filename} ({len(file_content)} bytes) par utilisateur {current_user.id}")
        
        # 3. Vérifier le format supporté
        docling_service = get_docling_service()
        # On ne peut pas déterminer le MIME type sans sauvegarder d'abord
        # Donc on crée d'abord la note temporaire
        
        # 4. Créer une note temporaire pour obtenir un ID
        note_title = title or original_filename
        note = Note(
            title=note_title,
            content=f"Document importé: {original_filename}",
            note_type="document",
            project_id=project_id,
            user_id=current_user.id
        )
        
        session.add(note)
        session.commit()
        session.refresh(note)
        
        logger.info(f"📝 Note temporaire {note.id} créée")
        
        # 5. Sauvegarder le fichier
        file_storage = get_file_storage_service()
        try:
            file_path, mime_type = file_storage.save_uploaded_file(
                file_content,
                original_filename,
                current_user.id,
                project_id,
                note.id
            )
            
            # Vérifier si le format est supporté
            if not docling_service.is_supported_format(mime_type):
                # Supprimer la note et le fichier
                file_storage.delete_note_files(current_user.id, project_id, note.id)
                session.delete(note)
                session.commit()
                
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Format de fichier non supporté: {mime_type}"
                )
            
            # Mettre à jour la note avec les infos du fichier
            note.source_file_path = file_path
            note.source_file_type = mime_type
            session.add(note)
            session.commit()
            
            logger.info(f"💾 Fichier sauvegardé: {file_path} (type: {mime_type})")
            
        except ValueError as e:
            # Fichier trop volumineux ou autre erreur de validation
            session.delete(note)
            session.commit()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            # Erreur lors de la sauvegarde
            session.delete(note)
            session.commit()
            logger.error(f"❌ Erreur lors de la sauvegarde du fichier: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Erreur lors de la sauvegarde du fichier"
            )
        
        # 6. Parser le document avec Docling
        try:
            parsed_doc = docling_service.parse_document(file_path, mime_type)
            
            if not parsed_doc:
                raise Exception("Impossible de parser le document")
            
            # Mettre à jour le contenu de la note avec le contenu extrait
            note.content = parsed_doc.get_text_content()[:5000]  # Limiter pour l'affichage
            session.add(note)
            session.commit()
            
            logger.info(f"📄 Document parsé: {len(parsed_doc.elements)} éléments")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du parsing: {e}", exc_info=True)
            # On continue quand même, on utilisera le contenu par défaut
        
        # 7. Créer les chunks et embeddings
        try:
            chunking_service = get_chunking_service()
            
            # Créer les chunks
            chunks = chunking_service.chunk_note(note)
            logger.info(f"📦 {len(chunks)} chunks créés")
            
            # Générer les embeddings
            chunks = chunking_service.generate_embeddings_for_chunks(chunks)
            
            # Sauvegarder les chunks
            for chunk in chunks:
                session.add(chunk)
            
            session.commit()
            
            chunks_with_embeddings = sum(1 for c in chunks if c.embedding is not None)
            logger.info(f"✅ Document importé avec succès: {chunks_with_embeddings}/{len(chunks)} chunks avec embeddings")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création des chunks: {e}", exc_info=True)
            # On garde la note même si le chunking échoue
        
        session.refresh(note)
        return NoteRead.model_validate(note)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'upload du document: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur lors de l'import du document"
        )


@router.get("/documents/supported-formats")
async def get_supported_formats(
    current_user: UserRead = Depends(get_current_user)
):
    """
    Récupérer la liste des formats de fichiers supportés.
    """
    return {
        "formats": [
            {"name": "PDF", "mime_type": "application/pdf", "extensions": [".pdf"]},
            {"name": "Word", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "extensions": [".docx"]},
            {"name": "Excel", "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "extensions": [".xlsx"]},
            {"name": "PowerPoint", "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation", "extensions": [".pptx"]},
            {"name": "CSV", "mime_type": "text/csv", "extensions": [".csv"]},
            {"name": "Texte", "mime_type": "text/plain", "extensions": [".txt"]},
            {"name": "Markdown", "mime_type": "text/markdown", "extensions": [".md"]},
            {"name": "JSON", "mime_type": "application/json", "extensions": [".json"]},
            {"name": "Image PNG", "mime_type": "image/png", "extensions": [".png"]},
            {"name": "Image JPEG", "mime_type": "image/jpeg", "extensions": [".jpg", ".jpeg"]},
        ],
        "max_size_mb": 50
    }

