from typing import Optional
from sqlmodel import Session, select
from app.models.user import User
from app.models.role import Role
from app.models.user_role import UserRole
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def ensure_user_has_admin_role(session: Session, user: User) -> bool:
    """
    Attribue le rôle admin à l'utilisateur si son email correspond à ADMIN_EMAIL.
    Retourne True si le rôle a été attribué, False sinon.
    """
    if not settings.ADMIN_EMAIL:
        return False
    
    if user.email.lower() != settings.ADMIN_EMAIL.lower():
        return False
    
    # Vérifier si l'utilisateur a déjà le rôle admin
    admin_role = session.exec(select(Role).where(Role.name == "admin")).first()
    if not admin_role:
        logger.warning("Rôle admin introuvable dans la base")
        return False
    
    existing_user_role = session.exec(
        select(UserRole).where(
            UserRole.user_id == user.id,
            UserRole.role_id == admin_role.id
        )
    ).first()
    
    if existing_user_role:
        return True
    
    # Attribuer le rôle admin
    user_role = UserRole(
        user_id=user.id,
        role_id=admin_role.id
    )
    session.add(user_role)
    session.commit()
    
    logger.info(f"Rôle admin attribué automatiquement à {user.email}")
    return True


def ensure_user_has_default_role(session: Session, user: User) -> bool:
    """
    Attribue le rôle 'member' par défaut à un utilisateur s'il n'a aucun rôle.
    Retourne True si un rôle a été attribué, False sinon.
    """
    existing_roles = session.exec(
        select(UserRole).where(UserRole.user_id == user.id)
    ).all()
    
    if existing_roles:
        return False
    
    member_role = session.exec(select(Role).where(Role.name == "member")).first()
    if not member_role:
        logger.warning("Rôle member introuvable dans la base")
        return False
    
    user_role = UserRole(
        user_id=user.id,
        role_id=member_role.id
    )
    session.add(user_role)
    session.commit()
    
    logger.info(f"Rôle member attribué par défaut à {user.email}")
    return True


def get_user_permissions(session: Session, user_id: int) -> set[str]:
    """
    Récupère toutes les permissions effectives d'un utilisateur
    (via tous ses rôles).
    """
    from app.models.role_permission import RolePermission
    from app.models.permission import Permission
    
    user_roles = session.exec(
        select(UserRole).where(UserRole.user_id == user_id)
    ).all()
    
    if not user_roles:
        return set()
    
    role_ids = [ur.role_id for ur in user_roles]
    
    role_permissions = session.exec(
        select(RolePermission).where(RolePermission.role_id.in_(role_ids))
    ).all()
    
    permission_ids = [rp.permission_id for rp in role_permissions]
    
    if not permission_ids:
        return set()
    
    permissions = session.exec(
        select(Permission).where(Permission.id.in_(permission_ids))
    ).all()
    
    return {perm.code for perm in permissions}


def user_has_permission(session: Session, user_id: int, permission_code: str) -> bool:
    """Vérifie si un utilisateur a une permission spécifique."""
    permissions = get_user_permissions(session, user_id)
    return permission_code in permissions
