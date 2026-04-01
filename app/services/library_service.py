from typing import Optional
from sqlmodel import Session, select
from app.models.library import Library, LibraryRead, LibraryStats
from app.models.document import Document
from app.models.folder import Folder
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def get_or_create_user_library(session: Session, user_id: int) -> Library:
    """
    Récupère ou crée la bibliothèque globale partagée (commune à tous).
    Si elle n'existe pas, elle est créée.
    """
    statement = select(Library).where(Library.is_global == True)
    library = session.exec(statement).first()
    
    if not library:
        library = Library(
            name="Bibliothèque commune",
            user_id=None,
            is_global=True
        )
        session.add(library)
        session.commit()
        session.refresh(library)
        logger.info("Bibliothèque globale créée")
    
    return library


def get_library_by_id(session: Session, library_id: int, user_id: int) -> Optional[Library]:
    """Récupère une bibliothèque par son ID (toujours accessible si globale)."""
    library = session.get(Library, library_id)
    if not library:
        return None
    if library.is_global:
        return library
    if library.user_id == user_id:
        return library
    return None


def get_library_stats(session: Session, library_id: int, user_id: int) -> Optional[LibraryStats]:
    """Calcule les statistiques d'une bibliothèque."""
    library = get_library_by_id(session, library_id, user_id)
    if not library:
        return None
    
    total_documents = session.exec(
        select(Document).where(Document.library_id == library_id)
    ).all()
    
    total_folders = session.exec(
        select(Folder).where(Folder.library_id == library_id)
    ).all()
    
    total_size_mb = 0.0
    for doc in total_documents:
        if doc.source_file_path:
            try:
                import os
                if os.path.exists(doc.source_file_path):
                    total_size_mb += os.path.getsize(doc.source_file_path) / (1024 * 1024)
            except Exception:
                pass
    
    return LibraryStats(
        library_id=library_id,
        total_documents=len(total_documents),
        total_folders=len(total_folders),
        total_size_mb=round(total_size_mb, 2)
    )


def update_library(session: Session, library_id: int, name: str, user_id: int) -> Optional[Library]:
    """Met à jour le nom d'une bibliothèque."""
    library = get_library_by_id(session, library_id, user_id)
    if not library:
        return None
    
    library.name = name
    library.updated_at = datetime.utcnow()
    session.add(library)
    session.commit()
    session.refresh(library)
    
    return library
