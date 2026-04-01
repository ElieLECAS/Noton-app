from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .user_role import UserRole


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True, max_length=150)
    email: str = Field(unique=True, index=True, max_length=255)
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    user_roles: List["UserRole"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"foreign_keys": "[UserRole.user_id]"},
    )


class UserCreate(SQLModel):
    username: str
    email: str
    password: str


class UserRead(SQLModel):
    id: int
    username: str
    email: str
    created_at: datetime
    roles: List[str] = []
    permissions: List[str] = []


class UserLogin(SQLModel):
    username: str
    password: str

