from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .user_role import UserRole
    from .role_permission import RolePermission


class Role(SQLModel, table=True):
    """Rôle utilisateur pour le système RBAC."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    is_system: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    user_roles: List["UserRole"] = Relationship(back_populates="role")
    role_permissions: List["RolePermission"] = Relationship(back_populates="role")


class RoleCreate(SQLModel):
    """Schéma de création pour un rôle."""
    name: str
    description: Optional[str] = None


class RoleRead(SQLModel):
    """Schéma de lecture pour un rôle."""
    id: int
    name: str
    description: Optional[str]
    is_system: bool
    created_at: datetime


class RoleUpdate(SQLModel):
    """Schéma de mise à jour pour un rôle."""
    name: Optional[str] = None
    description: Optional[str] = None
