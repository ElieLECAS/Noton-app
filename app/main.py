from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session
import time

from app.database import get_session, create_db_and_tables
from app.routers import auth, projects, notes, chat, conversations
from app.models import User, Project, Note
from app.services.faiss_service import get_faiss_manager
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

# Monter les routers
app.include_router(auth.router)
app.include_router(projects.router)
app.include_router(notes.router)
app.include_router(chat.router)
app.include_router(conversations.router)

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
    """Créer les tables au démarrage et initialiser FAISS"""
    create_db_and_tables()
    
    # Initialiser FAISS et charger les embeddings depuis la base de données
    try:
        logger.info("Initialisation de FAISS...")
        faiss_manager = get_faiss_manager()
        faiss_manager.initialize()
        
        # Charger tous les embeddings depuis la DB
        # On charge à la demande lors de la première recherche pour éviter de bloquer le démarrage
        logger.info("FAISS initialisé. Les embeddings seront chargés à la demande.")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de FAISS: {e}")
        # Ne pas bloquer le démarrage si FAISS échoue
    
    # Tester la connexion à Ollama pour les embeddings
    try:
        from app.services.embedding_service import generate_embedding
        logger.info("Test de connexion à Ollama pour les embeddings...")
        # Test rapide avec un texte court
        test_embedding = generate_embedding("test")
        if test_embedding:
            logger.info("✅ Ollama est prêt pour générer les embeddings")
        else:
            logger.warning("⚠️ Impossible de générer un embedding de test avec Ollama")
    except Exception as e:
        logger.warning(f"⚠️ Erreur lors du test de connexion à Ollama: {e}")
        # Ne pas bloquer le démarrage si Ollama n'est pas disponible
    
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}

