from typing import List, Optional
from sqlmodel import Session, select
from app.models.project import Project, ProjectCreate, ProjectUpdate
from app.models.user import User


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
    """Supprimer un projet"""
    project = get_project_by_id(session, project_id, user_id)
    if not project:
        return False
    
    session.delete(project)
    session.commit()
    return True

