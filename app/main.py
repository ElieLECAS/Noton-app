from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session
import time

from app.database import get_session, create_db_and_tables
from app.routers import auth, projects, notes, chat, documents
from app.models import User, Project, Note
from app.services.scheduler_service import init_scheduler, start_scheduler, stop_scheduler
from app.config import settings
import logging

# Configurer le logging pour voir les messages INFO
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME, description="Application de prise de notes avec chatbot Ollama")

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
app.include_router(projects.router)
app.include_router(notes.router)
app.include_router(chat.router)
app.include_router(documents.router)

# Configuration des templates
templates = Jinja2Templates(directory="app/templates")

# Ajouter le contexte global pour tous les templates
templates.env.globals["app_name"] = settings.APP_NAME
templates.env.globals["model_private"] = {
    "provider": settings.MODEL_PRIVATE_PROVIDER,
    "model": settings.MODEL_PRIVATE_NAME
}
templates.env.globals["model_fast"] = {
    "provider": settings.MODEL_FAST_PROVIDER,
    "model": settings.MODEL_FAST_NAME
}
templates.env.globals["model_powerful"] = {
    "provider": settings.MODEL_POWERFUL_PROVIDER,
    "model": settings.MODEL_POWERFUL_NAME
}

# Servir les fichiers statiques
try:
    app.mount("/static", StaticFiles(directory="app/static"), name="static")
except:
    pass  # Le dossier static peut ne pas exister


@app.on_event("startup")
async def startup_event():
    """Créer les tables au démarrage et initialiser le scheduler"""
    create_db_and_tables()
    
    # Initialiser et démarrer le scheduler
    try:
        init_scheduler()
        start_scheduler()
        logger.info("✅ Scheduler initialisé et démarré")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du scheduler: {e}")
    
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
        from app.services.document_service import _ensure_document_workers
        _ensure_document_workers()
        logger.info("Workers de traitement de documents démarrés")
    except Exception as e:
        logger.error(f"Erreur lors du démarrage des workers de traitement de documents: {e}")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Page d'accueil - redirige vers login si non authentifié"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Page de connexion"""
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Page d'inscription"""
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/projects/{project_id}", response_class=HTMLResponse)
async def project_detail_page(request: Request, project_id: int):
    """Page de détail d'un projet"""
    return templates.TemplateResponse("project_detail.html", {"request": request})


@app.get("/notes/{note_id}/edit", response_class=HTMLResponse)
async def note_edit_page(request: Request, note_id: int):
    """Page d'édition d'une note"""
    return templates.TemplateResponse("note_edit.html", {"request": request})


@app.get("/projects/{project_id}/kag-graph", response_class=HTMLResponse)
async def project_kag_graph_page(request: Request, project_id: int):
    """Page de visualisation du graphe KAG d'un projet"""
    return templates.TemplateResponse("project_kag_graph.html", {"request": request})


@app.on_event("shutdown")
async def shutdown_event():
    """Arrêter le scheduler à l'arrêt de l'application"""
    try:
        stop_scheduler()
        logger.info("Scheduler arrêté proprement")
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt du scheduler: {e}")


@app.get("/studio", response_class=HTMLResponse)
async def studio_page(request: Request):
    """Page du studio d'agents"""
    return templates.TemplateResponse("studio.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}

