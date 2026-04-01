from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from app.database import get_session
from app.models.user import User, UserRead, UserCreate
from app.models.role import Role, RoleCreate, RoleRead, RoleUpdate
from app.models.permission import Permission, PermissionCreate, PermissionRead
from app.models.user_role import UserRole, UserRoleCreate, UserRoleRead
from app.models.role_permission import RolePermission, RolePermissionCreate, RolePermissionRead
from app.routers.auth import get_current_user, require_permission
from app.services.auth_service import get_password_hash
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class UserWithRoles(BaseModel):
    """Utilisateur avec ses rôles."""
    id: int
    username: str
    email: str
    created_at: str
    roles: List[str]


class RoleWithPermissions(BaseModel):
    """Rôle avec ses permissions."""
    id: int
    name: str
    description: str | None
    is_system: bool
    permissions: List[str]


class AssignRoleRequest(BaseModel):
    """Assigner un rôle à un utilisateur."""
    user_id: int
    role_id: int


class AssignPermissionRequest(BaseModel):
    """Assigner une permission à un rôle."""
    role_id: int
    permission_id: int


# ==================== USERS ====================

@router.get("/users", response_model=List[UserWithRoles])
async def list_users(
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Liste tous les utilisateurs avec leurs rôles."""
    users = session.exec(select(User)).all()
    result = []
    
    for user in users:
        user_roles = session.exec(
            select(UserRole).where(UserRole.user_id == user.id)
        ).all()
        role_ids = [ur.role_id for ur in user_roles]
        roles = session.exec(
            select(Role).where(Role.id.in_(role_ids))
        ).all() if role_ids else []
        
        result.append(UserWithRoles(
            id=user.id,
            username=user.username,
            email=user.email,
            created_at=user.created_at.isoformat(),
            roles=[role.name for role in roles]
        ))
    
    return result


@router.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user_admin(
    user_create: UserCreate,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Créer un nouvel utilisateur (admin)."""
    from app.services.auth_service import create_user
    try:
        user = create_user(session, user_create)
        return UserRead.model_validate(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Supprimer un utilisateur."""
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de supprimer son propre compte"
        )
    
    # Supprimer les rôles associés
    user_roles = session.exec(select(UserRole).where(UserRole.user_id == user_id)).all()
    for ur in user_roles:
        session.delete(ur)
    
    session.delete(user)
    session.commit()


@router.post("/users/assign-role", response_model=UserRoleRead)
async def assign_role_to_user(
    request: AssignRoleRequest,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Assigner un rôle unique à un utilisateur (remplace les rôles existants)."""
    user = session.get(User, request.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Utilisateur non trouvé"
        )
    
    role = session.get(Role, request.role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rôle non trouvé"
        )
    
    existing_roles = session.exec(
        select(UserRole).where(UserRole.user_id == request.user_id)
    ).all()

    for existing_role in existing_roles:
        if existing_role.role_id == request.role_id:
            return UserRoleRead.model_validate(existing_role)
        session.delete(existing_role)

    user_role = UserRole(
        user_id=request.user_id,
        role_id=request.role_id,
        assigned_by=current_user.id
    )
    session.add(user_role)
    session.commit()
    session.refresh(user_role)
    
    return UserRoleRead.model_validate(user_role)


@router.delete("/users/{user_id}/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_role_from_user(
    user_id: int,
    role_id: int,
    current_user: UserRead = Depends(require_permission("config.manage_users")),
    session: Session = Depends(get_session)
):
    """Retirer un rôle d'un utilisateur."""
    user_role = session.exec(
        select(UserRole).where(
            UserRole.user_id == user_id,
            UserRole.role_id == role_id
        )
    ).first()
    
    if not user_role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Association utilisateur-rôle non trouvée"
        )
    
    session.delete(user_role)
    session.commit()


# ==================== ROLES ====================

@router.get("/roles", response_model=List[RoleWithPermissions])
async def list_roles(
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Liste tous les rôles avec leurs permissions."""
    roles = session.exec(select(Role)).all()
    result = []
    
    for role in roles:
        role_perms = session.exec(
            select(RolePermission).where(RolePermission.role_id == role.id)
        ).all()
        perm_ids = [rp.permission_id for rp in role_perms]
        permissions = session.exec(
            select(Permission).where(Permission.id.in_(perm_ids))
        ).all() if perm_ids else []
        
        result.append(RoleWithPermissions(
            id=role.id,
            name=role.name,
            description=role.description,
            is_system=role.is_system,
            permissions=[perm.code for perm in permissions]
        ))
    
    return result


@router.post("/roles", response_model=RoleRead, status_code=status.HTTP_201_CREATED)
async def create_role(
    role_create: RoleCreate,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Créer un nouveau rôle."""
    existing = session.exec(select(Role).where(Role.name == role_create.name)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Un rôle avec ce nom existe déjà"
        )
    
    role = Role(
        name=role_create.name,
        description=role_create.description,
        is_system=False
    )
    session.add(role)
    session.commit()
    session.refresh(role)
    
    return RoleRead.model_validate(role)


@router.put("/roles/{role_id}", response_model=RoleRead)
async def update_role(
    role_id: int,
    role_update: RoleUpdate,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Mettre à jour un rôle."""
    role = session.get(Role, role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rôle non trouvé"
        )
    
    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de modifier un rôle système"
        )
    
    update_data = role_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(role, key, value)
    
    session.add(role)
    session.commit()
    session.refresh(role)
    
    return RoleRead.model_validate(role)


@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(
    role_id: int,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Supprimer un rôle."""
    role = session.get(Role, role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rôle non trouvé"
        )
    
    if role.is_system:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de supprimer un rôle système"
        )
    
    # Supprimer les associations utilisateurs
    user_roles = session.exec(select(UserRole).where(UserRole.role_id == role_id)).all()
    for ur in user_roles:
        session.delete(ur)
    
    # Supprimer les permissions associées
    role_perms = session.exec(select(RolePermission).where(RolePermission.role_id == role_id)).all()
    for rp in role_perms:
        session.delete(rp)
    
    session.delete(role)
    session.commit()


@router.post("/roles/assign-permission", response_model=RolePermissionRead)
async def assign_permission_to_role(
    request: AssignPermissionRequest,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Assigner une permission à un rôle."""
    role = session.get(Role, request.role_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Rôle non trouvé"
        )
    
    permission = session.get(Permission, request.permission_id)
    if not permission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission non trouvée"
        )
    
    # Vérifier si l'association existe déjà
    existing = session.exec(
        select(RolePermission).where(
            RolePermission.role_id == request.role_id,
            RolePermission.permission_id == request.permission_id
        )
    ).first()
    
    if existing:
        return RolePermissionRead.model_validate(existing)
    
    role_perm = RolePermission(
        role_id=request.role_id,
        permission_id=request.permission_id
    )
    session.add(role_perm)
    session.commit()
    session.refresh(role_perm)
    
    return RolePermissionRead.model_validate(role_perm)


@router.delete("/roles/{role_id}/permissions/{permission_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_permission_from_role(
    role_id: int,
    permission_id: int,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Retirer une permission d'un rôle."""
    role_perm = session.exec(
        select(RolePermission).where(
            RolePermission.role_id == role_id,
            RolePermission.permission_id == permission_id
        )
    ).first()
    
    if not role_perm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Association rôle-permission non trouvée"
        )
    
    session.delete(role_perm)
    session.commit()


# ==================== PERMISSIONS ====================

@router.get("/permissions", response_model=List[PermissionRead])
async def list_permissions(
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Liste toutes les permissions."""
    permissions = session.exec(select(Permission)).all()
    return [PermissionRead.model_validate(p) for p in permissions]


@router.post("/permissions", response_model=PermissionRead, status_code=status.HTTP_201_CREATED)
async def create_permission(
    permission_create: PermissionCreate,
    current_user: UserRead = Depends(require_permission("config.manage_roles")),
    session: Session = Depends(get_session)
):
    """Créer une nouvelle permission."""
    existing = session.exec(select(Permission).where(Permission.code == permission_create.code)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Une permission avec ce code existe déjà"
        )
    
    permission = Permission(
        code=permission_create.code,
        name=permission_create.name,
        description=permission_create.description,
        category=permission_create.category,
        is_system=False
    )
    session.add(permission)
    session.commit()
    session.refresh(permission)
    
    return PermissionRead.model_validate(permission)
