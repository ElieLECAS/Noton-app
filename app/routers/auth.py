from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
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
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: Session = Depends(get_session)
) -> UserRead:
    """Dependency pour obtenir l'utilisateur courant depuis le token JWT"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token manquant",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    print(f"DEBUG: Token reçu: {token[:20]}...")  # Log les 20 premiers caractères
    payload = decode_token(token)
    print(f"DEBUG: Payload décodé: {payload}")  # Log le payload
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
    return UserRead.model_validate(user)


@router.post("/register", response_model=dict)
async def register(user_create: UserCreate, session: Session = Depends(get_session)):
    """Inscription d'un nouvel utilisateur"""
    try:
        user = create_user(session, user_create)
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=access_token_expires
        )
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserRead.model_validate(user)
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=dict)
async def login(user_login: UserLogin, session: Session = Depends(get_session)):
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
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserRead.model_validate(user)
    }


@router.get("/me", response_model=UserRead)
async def get_current_user_info(current_user: UserRead = Depends(get_current_user)):
    """Obtenir les informations de l'utilisateur courant"""
    return current_user

