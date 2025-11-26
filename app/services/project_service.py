from typing import List, Optional
from sqlmodel import Session, select
from app.models.project import Project, ProjectCreate, ProjectUpdate
from app.models.note import Note
from app.models.user import User
from app.services.chunk_service import delete_chunks_for_note
import logging

logger = logging.getLogger(__name__)


def get_projects_by_user(session: Session, user_id: int) -> List[Project]:
    """Récupérer tous les projets d'un utilisateur"""
    statement = select(Project).where(Project.user_id == user_id).order_by(Project.updated_at.desc())
    return list(session.exec(statement).all())


def get_project_by_id(session: Session, project_id: int, user_id: int) -> Optional[Project]:
    """Récupérer un projet par son ID (vérifie que l'utilisateur en est propriétaire)"""
    project = session.get(Project, project_id)
    if project and project.user_id == user_id:
        return project
    return None


def create_project(session: Session, project_create: ProjectCreate, user_id: int) -> Project:
    """Créer un nouveau projet"""
    project = Project(
        **project_create.model_dump(),
        user_id=user_id
    )
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def update_project(session: Session, project_id: int, project_update: ProjectUpdate, user_id: int) -> Optional[Project]:
    """Mettre à jour un projet"""
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        return None
    
    update_data = project_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    
    from datetime import datetime
    project.updated_at = datetime.utcnow()
    
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


def delete_project(session: Session, project_id: int, user_id: int) -> bool:
    """Supprimer un projet et toutes ses notes associées (avec leurs chunks)"""
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        return False
    
    try:
        # Récupérer toutes les notes du projet
        statement = select(Note).where(Note.project_id == project_id)
        notes = list(session.exec(statement).all())
        
        # Supprimer les chunks et les notes associées
        for note in notes:
            # Supprimer les chunks de la note (sans commit pour transaction atomique)
            delete_chunks_for_note(session, note.id, commit=False)
            # Supprimer la note
            session.delete(note)
        
        logger.debug(f"Supprimé {len(notes)} notes et leurs chunks pour le projet {project_id}")
        
        # Supprimer le projet
        session.delete(project)
        
        # Commit atomique : soit tout est supprimé, soit rien
        session.commit()
        
        logger.info(f"🗑️ Projet {project_id} et toutes ses notes supprimés avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du projet {project_id}: {e}", exc_info=True)
        session.rollback()
        raise

