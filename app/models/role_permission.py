from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .role import Role
    from .permission import Permission


class RolePermission(SQLModel, table=True):
    """Association entre rôle et permission."""
    id: Optional[int] = Field(default=None, primary_key=True)
    role_id: int = Field(foreign_key="role.id", index=True)
    permission_id: int = Field(foreign_key="permission.id", index=True)
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    
    role: "Role" = Relationship(back_populates="role_permissions")
    permission: "Permission" = Relationship(back_populates="role_permissions")


class RolePermissionCreate(SQLModel):
    """Schéma de création pour une association rôle-permission."""
    role_id: int
    permission_id: int


class RolePermissionRead(SQLModel):
    """Schéma de lecture pour une association rôle-permission."""
    id: int
    role_id: int
    permission_id: int
    assigned_at: datetime
