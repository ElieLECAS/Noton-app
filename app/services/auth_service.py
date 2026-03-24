from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlmodel import Session, select
from app.config import settings
from app.models.user import User, UserCreate, UserLogin

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifier un mot de passe"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hasher un mot de passe"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Créer un token JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    # S'assurer que c'est une string (python-jose peut retourner bytes)
    if isinstance(encoded_jwt, bytes):
        return encoded_jwt.decode('utf-8')
    return encoded_jwt


def get_user_by_username(session: Session, username: str) -> Optional[User]:
    """Récupérer un utilisateur par son username"""
    statement = select(User).where(User.username == username)
    return session.exec(statement).first()


def get_user_by_id(session: Session, user_id: int) -> Optional[User]:
    """Récupérer un utilisateur par son ID"""
    return session.get(User, user_id)


def create_user(session: Session, user_create: UserCreate) -> User:
    """Créer un nouvel utilisateur"""
    # Vérifier si l'utilisateur existe déjà
    existing_user = get_user_by_username(session, user_create.username)
    if existing_user:
        raise ValueError("Username already exists")
    
    existing_email = session.exec(select(User).where(User.email == user_create.email)).first()
    if existing_email:
        raise ValueError("Email already exists")
    
    # Créer l'utilisateur
    user = User(
        username=user_create.username,
        email=user_create.email,
        password_hash=get_password_hash(user_create.password)
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    
    # Attribution automatique admin si ADMIN_EMAIL correspond
    from app.services.authorization_service import ensure_user_has_admin_role, ensure_user_has_default_role
    ensure_user_has_admin_role(session, user)
    ensure_user_has_default_role(session, user)
    
    return user


def authenticate_user(session: Session, username: str, password: str) -> Optional[User]:
    """Authentifier un utilisateur"""
    user = get_user_by_username(session, username)
    if not user:
        return None
    if not verify_password(password, user.password_hash):
        return None
    
    # Attribution automatique admin si ADMIN_EMAIL correspond (au cas où l'email serait changé)
    from app.services.authorization_service import ensure_user_has_admin_role, ensure_user_has_default_role
    ensure_user_has_admin_role(session, user)
    ensure_user_has_default_role(session, user)
    
    return user


def decode_token(token: str) -> Optional[dict]:
    """Décoder un token JWT"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError as e:
        print(f"DEBUG: Erreur de décodage JWT: {e}")
        print(f"DEBUG: SECRET_KEY utilisé: {settings.SECRET_KEY[:20]}...")
        print(f"DEBUG: ALGORITHM utilisé: {settings.ALGORITHM}")
        return None

