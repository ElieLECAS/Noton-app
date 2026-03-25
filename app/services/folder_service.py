from typing import Optional, List
from sqlmodel import Session, select
from app.models.folder import Folder, FolderCreate, FolderUpdate, FolderRead, FolderWithContents
from app.models.document import Document
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_folder(
    session: Session,
    folder_create: FolderCreate,
    library_id: int,
    user_id: int
) -> Folder:
    """Crée un nouveau dossier."""
    folder = Folder(
        name=folder_create.name,
        parent_folder_id=folder_create.parent_folder_id,
        library_id=library_id,
        user_id=user_id
    )
    session.add(folder)
    session.commit()
    session.refresh(folder)
    logger.info(f"Dossier créé: {folder.name} (ID: {folder.id})")
    return folder


def get_folder_by_id(session: Session, folder_id: int, user_id: int) -> Optional[Folder]:
    """Récupère un dossier par son ID (bibliothèque globale partagée)."""
    statement = select(Folder).where(
        Folder.id == folder_id
    )
    return session.exec(statement).first()


def get_folders_by_parent(
    session: Session,
    parent_folder_id: Optional[int],
    library_id: int,
    user_id: int
) -> List[Folder]:
    """Récupère tous les sous-dossiers d'un dossier parent (ou dossiers racine si parent_folder_id est None)."""
    statement = select(Folder).where(
        Folder.library_id == library_id,
        Folder.parent_folder_id == parent_folder_id
    ).order_by(Folder.name)
    return list(session.exec(statement).all())


def get_folders_by_library(session: Session, library_id: int, user_id: int) -> List[Folder]:
    """Récupère tous les dossiers d'une bibliothèque."""
    statement = select(Folder).where(
        Folder.library_id == library_id
    ).order_by(Folder.name)
    return list(session.exec(statement).all())


def get_folder_path(session: Session, folder_id: int, user_id: int) -> List[FolderRead]:
    """
    Récupère le chemin complet d'un dossier (breadcrumb).
    Retourne la liste des dossiers parents du plus haut niveau au dossier actuel.
    """
    path = []
    current_folder = get_folder_by_id(session, folder_id, user_id)
    
    while current_folder:
        path.insert(0, FolderRead.model_validate(current_folder))
        if current_folder.parent_folder_id:
            current_folder = get_folder_by_id(session, current_folder.parent_folder_id, user_id)
        else:
            break
    
    return path


def get_folder_with_contents(
    session: Session,
    folder_id: int,
    user_id: int
) -> Optional[FolderWithContents]:
    """Récupère un dossier avec ses sous-dossiers et le compte de documents."""
    folder = get_folder_by_id(session, folder_id, user_id)
    if not folder:
        return None
    
    subfolders = get_folders_by_parent(session, folder_id, folder.library_id, user_id)
    
    document_count = session.exec(
        select(Document).where(
            Document.folder_id == folder_id
        )
    ).all()
    
    return FolderWithContents(
        **folder.model_dump(),
        subfolders=[FolderRead.model_validate(f) for f in subfolders],
        document_count=len(document_count)
    )


def move_folder(
    session: Session,
    folder_id: int,
    new_parent_id: Optional[int],
    user_id: int
) -> Optional[Folder]:
    """Déplace un dossier vers un nouveau parent."""
    folder = get_folder_by_id(session, folder_id, user_id)
    if not folder:
        return None
    
    if new_parent_id:
        new_parent = get_folder_by_id(session, new_parent_id, user_id)
        if not new_parent:
            return None
        
        if new_parent.library_id != folder.library_id:
            logger.error(f"Impossible de déplacer le dossier {folder_id} : les dossiers ne sont pas dans la même bibliothèque")
            return None
        
        current = new_parent
        while current:
            if current.id == folder_id:
                logger.error(f"Impossible de déplacer le dossier {folder_id} : déplacement circulaire détecté")
                return None
            current = get_folder_by_id(session, current.parent_folder_id, user_id) if current.parent_folder_id else None
    
    folder.parent_folder_id = new_parent_id
    folder.updated_at = datetime.utcnow()
    session.add(folder)
    session.commit()
    session.refresh(folder)
    
    return folder


def rename_folder(
    session: Session,
    folder_id: int,
    folder_update: FolderUpdate,
    user_id: int
) -> Optional[Folder]:
    """Renomme un dossier."""
    folder = get_folder_by_id(session, folder_id, user_id)
    if not folder:
        return None
    
    if folder_update.name:
        folder.name = folder_update.name
        folder.updated_at = datetime.utcnow()
        session.add(folder)
        session.commit()
        session.refresh(folder)
    
    return folder


def delete_folder(session: Session, folder_id: int, user_id: int) -> bool:
    """
    Supprime un dossier et tout son contenu (récursif).
    Supprime également tous les documents du dossier et de ses sous-dossiers.
    """
    # On utilise le service "new" qui gère aussi la suppression des chunks,
    # des associations et des fichiers source.
    from app.services.document_service_new import delete_document
    
    folder = get_folder_by_id(session, folder_id, user_id)
    if not folder:
        return False
    
    subfolders = get_folders_by_parent(session, folder_id, folder.library_id, user_id)
    for subfolder in subfolders:
        delete_folder(session, subfolder.id, user_id)
    
    documents = session.exec(
        select(Document).where(
            Document.folder_id == folder_id
        )
    ).all()
    
    for document in documents:
        delete_document(session, document.id, user_id)
    
    session.delete(folder)
    session.commit()
    
    logger.info(f"Dossier supprimé: {folder.name} (ID: {folder_id})")
    return True
