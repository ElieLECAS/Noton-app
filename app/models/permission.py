from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .role_permission import RolePermission


class Permission(SQLModel, table=True):
    """Permission pour le système RBAC."""
    id: Optional[int] = Field(default=None, primary_key=True)
    code: str = Field(unique=True, index=True, max_length=100)
    name: str = Field(max_length=200)
    description: Optional[str] = Field(default=None, max_length=500)
    category: str = Field(max_length=50)
    is_system: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    role_permissions: List["RolePermission"] = Relationship(back_populates="permission")


class PermissionCreate(SQLModel):
    """Schéma de création pour une permission."""
    code: str
    name: str
    description: Optional[str] = None
    category: str


class PermissionRead(SQLModel):
    """Schéma de lecture pour une permission."""
    id: int
    code: str
    name: str
    description: Optional[str]
    category: str
    is_system: bool
    created_at: datetime
