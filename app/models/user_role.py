from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
    from .role import Role


class UserRole(SQLModel, table=True):
    """Association entre utilisateur et rôle."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    role_id: int = Field(foreign_key="role.id", index=True)
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_by: Optional[int] = Field(default=None, foreign_key="user.id")
    
    user: "User" = Relationship(
        back_populates="user_roles",
        sa_relationship_kwargs={"foreign_keys": "[UserRole.user_id]"},
    )
    role: "Role" = Relationship(back_populates="user_roles")


class UserRoleCreate(SQLModel):
    """Schéma de création pour une association utilisateur-rôle."""
    user_id: int
    role_id: int


class UserRoleRead(SQLModel):
    """Schéma de lecture pour une association utilisateur-rôle."""
    id: int
    user_id: int
    role_id: int
    assigned_at: datetime
    assigned_by: Optional[int]
