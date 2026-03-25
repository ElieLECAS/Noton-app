from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session
import time
from typing import Optional

from app.database import get_session, create_db_and_tables, engine
from app.routers import auth, chat, conversations, kag, library, spaces, admin
from app.config import settings
from app.services.auth_service import decode_token, get_user_by_id
import logging

# Configurer le logging pour voir les messages INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME, description="Application de prise de notes avec chatbot IA")

# Configuration CORS - Debug
import os
env_cors = os.getenv("CORS_ALLOWED_ORIGINS")
logger.info(f"DEBUG CORS - Variable d'environnement brute: {repr(env_cors)}")
logger.info(f"DEBUG CORS - settings.CORS_ALLOWED_ORIGINS: {repr(settings.CORS_ALLOWED_ORIGINS)}")
logger.info(f"DEBUG CORS - Type: {type(settings.CORS_ALLOWED_ORIGINS)}")

if settings.CORS_ALLOWED_ORIGINS:
    logger.info(f"✅ Configuration CORS : origines autorisées = {settings.CORS_ALLOWED_ORIGINS}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Si aucune origine n'est spécifiée, autoriser toutes les origines (développement uniquement)
    logger.warning("⚠️ CORS : Aucune origine spécifiée, toutes les origines sont autorisées (mode développement)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Monter les routers
app.include_router(auth.router)
app.include_router(library.router)
app.include_router(spaces.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(kag.router)
app.include_router(admin.router)

# Configuration des templates
templates = Jinja2Templates(directory="app/templates")

# Ajouter le contexte global pour tous les templates
templates.env.globals["app_name"] = settings.APP_NAME
templates.env.globals["model_fast"] = {
    "provider": "mistral",
    "model": settings.MODEL_FAST
}

# Servir les fichiers statiques
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except:
    pass  # Le dossier static peut ne pas exister


@app.on_event("startup")
async def startup_event():
    """Créer les tables au démarrage."""
    create_db_and_tables()
    
    # Initialiser le système RBAC (permissions + rôles)
    try:
        from app.services.rbac_seed_service import seed_rbac_system
        with Session(engine) as session:
            seed_rbac_system(session)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation RBAC: {e}")
    
    # Tester l'initialisation du modèle d'embeddings HuggingFace
    try:
        from app.services.embedding_service import generate_embedding
        logger.info("Test d'initialisation du modèle d'embeddings HuggingFace...")
        # Test rapide avec un texte court
        test_embedding = generate_embedding("test")
        if test_embedding:
            logger.info("✅ Modèle d'embeddings HuggingFace initialisé et prêt")
        else:
            logger.warning("⚠️ Impossible de générer un embedding de test")
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors de l'initialisation du modèle d'embeddings: {e}")
        # Ne pas bloquer le démarrage si le modèle n'est pas disponible
    
    # Démarrer les workers pour la génération d'embeddings en arrière-plan
    try:
        from app.services.chunk_service import _ensure_embedding_workers
        _ensure_embedding_workers()
        logger.info("Workers d'embeddings démarrés")
    except Exception as e:
        logger.error(f"Erreur lors du démarrage des workers d'embeddings: {e}")
    
    # Démarrer les workers pour le traitement de documents en arrière-plan
    try:
        from app.services.document_service_new import _ensure_document_workers
        _ensure_document_workers()
        logger.info("Workers de traitement de documents démarrés")
    except Exception as e:
        logger.error(f"Erreur lors du démarrage des workers de traitement de documents: {e}")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request, session: Session = Depends(get_session)):
    """Page d'accueil (choix des espaces)."""
    if _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("home_spaces.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, session: Session = Depends(get_session)):
    """Page de connexion"""
    if not _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, session: Session = Depends(get_session)):
    """Page d'inscription"""
    if not _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/library", response_class=HTMLResponse)
async def library_page(request: Request, session: Session = Depends(get_session)):
    """Page bibliothèque générale."""
    if _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("library.html", {"request": request})


@app.get("/spaces/{space_id}", response_class=HTMLResponse)
async def space_detail_page(request: Request, space_id: int, session: Session = Depends(get_session)):
    """Page de discussion dans un espace."""
    if _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("space_detail.html", {"request": request, "space_id": space_id})


@app.get("/spaces/{space_id}/kag-graph", response_class=HTMLResponse)
async def space_kag_graph_page(request: Request, space_id: int, session: Session = Depends(get_session)):
    """Page de visualisation du graphe KAG d'un espace."""
    if _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("space_kag_graph.html", {"request": request, "space_id": space_id})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, session: Session = Depends(get_session)):
    """Page d'administration (gestion users/rôles)."""
    if _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}


def _extract_bearer_token_from_request(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return request.cookies.get("authToken")


def _is_request_authenticated(request: Request, session: Session) -> bool:
    token = _extract_bearer_token_from_request(request)
    if not token:
        return False
    payload = decode_token(token)
    if payload is None:
        return False
    user_id_str = payload.get("sub")
    if user_id_str is None:
        return False
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        return False
    return get_user_by_id(session, user_id) is not None


def _redirect_if_unauthenticated(request: Request, session: Session) -> bool:
    return not _is_request_authenticated(request, session)

