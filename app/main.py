from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session
from typing import Optional

from app.database import get_session, create_db_and_tables, engine
from app.routers import auth, chat, conversations, library, spaces, admin, guided_trees
from app.config import settings
from app.services.auth_service import decode_token, get_user_by_id
from app.models.user import UserRead
import logging
import time

from app.logging_config import setup_app_logging

setup_app_logging()
logger = logging.getLogger(__name__)

# Fichier dédié : logs/library_document_processing.log (pipeline bibliothèque / espaces)
try:
    from app.library_document_logging import setup_library_document_file_logging

    _lib_log_path = setup_library_document_file_logging()
    logger.info("Journal documents bibliothèque/espaces : %s", _lib_log_path)
except Exception as e:
    logger.warning("Initialisation journal bibliothèque/espaces ignorée : %s", e)

# LangSmith — observabilité RAG
try:
    from app.tracing import init_langsmith
    init_langsmith()
except Exception as e:
    logger.warning("Initialisation LangSmith ignorée : %s", e)

app = FastAPI(
    title=settings.APP_NAME,
    description="Assistant documentaire RAG multimodal (menuiserie PROFERM)",
)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    """Trace chaque requête HTTP dans les logs du conteneur web."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if request.url.path != "/health":
        logger.info(
            "%s %s -> %s (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    return response


# Configuration CORS. Origines explicites uniquement (jamais de wildcard) : l'app sert
# son propre front en same-origin, donc l'absence d'origine CORS n'impacte pas l'UI, et
# on évite d'exposer l'API à n'importe quel site. Renseigner CORS_ALLOWED_ORIGINS pour
# autoriser un front séparé (les credentials ne sont activés que dans ce cas).
if settings.CORS_ALLOWED_ORIGINS:
    logger.info("CORS : origines autorisées = %s", settings.CORS_ALLOWED_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    logger.warning(
        "CORS : aucune origine configurée (CORS_ALLOWED_ORIGINS vide) — API restreinte au "
        "même origine. Renseigner la variable si un front séparé doit consommer l'API."
    )

# Monter les routers
app.include_router(auth.router)
app.include_router(library.router)
app.include_router(spaces.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(admin.router)
app.include_router(guided_trees.router)

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
    """Créer les tables au démarrage + journaliser la config effective."""
    create_db_and_tables()

    # Config effective : matrice de features sur une ligne + garde-fous d'incohérence.
    # Les migrations Alembic sont exécutées AVANT uvicorn par la commande du conteneur
    # (docker-compose `web`/`worker`), point unique pour éviter les courses multi-réplica.
    logger.info(settings.feature_summary())
    for warning in settings.coherence_warnings():
        logger.warning("[config] %s", warning)

    # Initialiser le système RBAC (permissions + rôles)
    try:
        from app.services.rbac_seed_service import seed_rbac_system
        with Session(engine) as session:
            seed_rbac_system(session)
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation RBAC: {e}")
    
    # Précharger le modèle ColPali en arrière-plan si activé
    if settings.COLPALI_ENABLED:
        import threading
        def preload_colpali():
            try:
                from app.services.colpali_service import get_colpali_model
                logger.info("Début du préchargement en arrière-plan du modèle ColPali...")
                get_colpali_model()
            except Exception as e:
                logger.error(f"Erreur lors du préchargement en arrière-plan du modèle ColPali: {e}")

        threading.Thread(target=preload_colpali, name="colpali-preload", daemon=True).start()

    # Précharger le cross-encoder de reranking (évite la latence de chargement
    # sur la 1re requête ; singleton, jamais rechargé ensuite).
    if settings.RERANKER_ENABLED and settings.RERANKER_PROVIDER == "local":
        import threading
        def preload_reranker():
            try:
                from app.services.reranker_service import warmup_cross_encoder
                logger.info("Début du préchargement en arrière-plan du cross-encoder de reranking...")
                warmup_cross_encoder()
            except Exception as e:
                logger.error(f"Erreur lors du préchargement en arrière-plan du cross-encoder: {e}")

        threading.Thread(target=preload_reranker, name="reranker-preload", daemon=True).start()
    
    # Workers threads (embeddings + documents) uniquement si thread ou hybrid (repli Celery)
    try:
        from app.services.task_dispatch import should_start_thread_workers

        if should_start_thread_workers():
            from app.services.chunk_service import _ensure_embedding_workers
            from app.services.document_service_new import _ensure_document_workers

            _ensure_embedding_workers()
            _ensure_document_workers()
            logger.info("Workers de traitement de documents (threads) démarrés")
        else:
            logger.info(
                "TASK_BACKEND_MODE=%s : pas de workers threads sur le process web (Celery)",
                settings.TASK_BACKEND_MODE,
            )
    except Exception as e:
        logger.error(f"Erreur lors du démarrage des workers threads: {e}")



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
    user = _get_authenticated_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("library.html", {"request": request, "user": user})


@app.get("/spaces/{space_id}", response_class=HTMLResponse)
async def space_detail_page(request: Request, space_id: int, session: Session = Depends(get_session)):
    """Page de discussion dans un espace."""
    user = _get_authenticated_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("space_detail.html", {"request": request, "space_id": space_id, "user": user})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, session: Session = Depends(get_session)):
    """Page d'administration (gestion users/rôles)."""
    if _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/admin/sav-trees", response_class=HTMLResponse)
async def admin_sav_trees_page(request: Request, session: Session = Depends(get_session)):
    """Builder des Arbres SAV (symptômes, éditeur outline, publication)."""
    if _redirect_if_unauthenticated(request, session):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("admin_sav_trees.html", {"request": request})


@app.get("/feedbacks", response_class=HTMLResponse)
async def feedbacks_page(request: Request, session: Session = Depends(get_session)):
    """Page contenant les feedbacks de l'utilisateur."""
    user = _get_authenticated_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("feedbacks.html", {"request": request, "user": user})



@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": time.time()}


def _extract_bearer_token_from_request(request: Request) -> Optional[str]:
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header.split(" ", 1)[1].strip()
    return request.cookies.get("authToken")


def _get_authenticated_user(request: Request, session: Session) -> Optional[UserRead]:
    token = _extract_bearer_token_from_request(request)
    if not token:
        return None
    payload = decode_token(token)
    if payload is None:
        return None
    user_id_str = payload.get("sub")
    if user_id_str is None:
        return None
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        return None
    
    user = get_user_by_id(session, user_id)
    if not user:
        return None
    
    # Mapper les rôles pour le template
    roles = [ur.role.name for ur in user.user_roles] if hasattr(user, "user_roles") else []
    return UserRead(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        roles=roles
    )


def _is_request_authenticated(request: Request, session: Session) -> bool:
    return _get_authenticated_user(request, session) is not None


def _redirect_if_unauthenticated(request: Request, session: Session) -> bool:
    return not _is_request_authenticated(request, session)
