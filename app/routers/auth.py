from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlmodel import Session
from app.database import get_session
from app.models.user import UserCreate, UserLogin, UserRead
from app.services.auth_service import (
    create_user,
    authenticate_user,
    create_access_token,
    get_user_by_id,
    decode_token
)
from app.config import settings

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session)
) -> UserRead:
    """Dependency pour obtenir l'utilisateur courant depuis le token JWT"""
    token = credentials.credentials if credentials else request.cookies.get("authToken")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token manquant",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide"
        )
    user_id_str = payload.get("sub")
    if user_id_str is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide"
        )
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide"
        )
    user = get_user_by_id(session, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Utilisateur non trouvé"
        )
    
    # Enrichir avec les rôles et permissions
    from app.services.authorization_service import get_user_permissions
    from app.models.user_role import UserRole
    from app.models.role import Role
    from sqlmodel import select
    
    user_roles_db = session.exec(select(UserRole).where(UserRole.user_id == user.id)).all()
    role_ids = [ur.role_id for ur in user_roles_db]
    roles = session.exec(select(Role).where(Role.id.in_(role_ids))).all() if role_ids else []
    role_names = [role.name for role in roles]
    
    permissions = get_user_permissions(session, user.id)
    
    user_read = UserRead.model_validate(user)
    user_read.roles = role_names
    user_read.permissions = list(permissions)
    
    return user_read


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="authToken",
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        path="/",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


def _clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(key="authToken", path="/")


@router.post("/register", response_model=dict)
async def register(
    user_create: UserCreate,
    response: Response,
    session: Session = Depends(get_session)
):
    """Inscription d'un nouvel utilisateur"""
    try:
        user = create_user(session, user_create)
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=access_token_expires
        )
        _set_auth_cookie(response, access_token)
        return {
            "token_type": "bearer",
            "user": UserRead.model_validate(user)
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=dict)
async def login(
    user_login: UserLogin,
    response: Response,
    session: Session = Depends(get_session)
):
    """Connexion d'un utilisateur"""
    user = authenticate_user(session, user_login.username, user_login.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username ou mot de passe incorrect"
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)}, expires_delta=access_token_expires
    )
    _set_auth_cookie(response, access_token)
    return {
        "token_type": "bearer",
        "user": UserRead.model_validate(user)
    }


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(response: Response):
    """Déconnexion: suppression du cookie de session."""
    _clear_auth_cookie(response)


@router.get("/me", response_model=UserRead)
async def get_current_user_info(current_user: UserRead = Depends(get_current_user)):
    """Obtenir les informations de l'utilisateur courant"""
    return current_user


def require_permission(permission_code: str):
    """
    Dépendance pour vérifier qu'un utilisateur a une permission spécifique.
    Usage: current_user: UserRead = Depends(require_permission("space.create"))
    """
    def permission_checker(current_user: UserRead = Depends(get_current_user)) -> UserRead:
        if permission_code not in current_user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission requise: {permission_code}"
            )
        return current_user
    return permission_checker

