"""LIA — l'assistant documentaire de PROFERM, adossé au wiki.

Quatre écrans : le chat (``/``), l'assistant vocal (``/vocal``), le wiki (``/wiki``, graphe +
lecteur) et l'administration. Le wiki est chargé au démarrage et rechargé dès qu'un fichier
change (voir wiki_service).
"""
import asyncio
import logging
import time
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

from app.config import settings
from app.database import create_db_and_tables, engine, get_session
from app.logging_config import setup_app_logging
from app.models.user import UserRead
from app.routers import admin, auth, chat, conversations, vocal, wiki
from app.services.auth_service import decode_token, get_user_by_id

setup_app_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="Assistant documentaire PROFERM — réponses sourcées sur le wiki interne (CAG).",
)


@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    """Trace chaque requête HTTP dans les logs du conteneur web."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if request.url.path != "/health":
        logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# CORS : origines explicites uniquement. L'app sert son propre front en same-origin ; sans
# variable, l'API reste restreinte au même origine.
if settings.CORS_ALLOWED_ORIGINS:
    logger.info("CORS : origines autorisées = %s", settings.CORS_ALLOWED_ORIGINS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(vocal.router)
app.include_router(wiki.router)
app.include_router(admin.router)

templates = Jinja2Templates(directory="app/templates")
templates.env.globals["app_name"] = settings.APP_NAME

app.mount("/static", StaticFiles(directory="app/static"), name="static")

from app.services.wiki_service import wiki_root
wiki_assets_path = wiki_root() / "wiki" / "assets"
wiki_assets_path.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(wiki_assets_path)), name="wiki-assets")


@app.on_event("startup")
async def startup_event():
    """Tables, RBAC, puis chargement du wiki (le premier tour ne doit pas l'attendre)."""
    create_db_and_tables()
    try:
        from app.services.rbac_seed_service import seed_rbac_system

        with Session(engine) as session:
            seed_rbac_system(session)
    except Exception as exc:  # noqa: BLE001
        logger.error("Initialisation RBAC en échec : %s", exc)

    try:
        from app.services.wiki_service import get_snapshot

        snapshot = get_snapshot()
        logger.info(
            "Wiki prêt : %d pages, ~%s tokens de prompt, modèle %s, raisonnement %s",
            len(snapshot.concept_pages),
            f"{snapshot.estimated_tokens:,}".replace(",", " "),
            settings.MODEL_FAST,
            settings.GENERATION_REASONING_EFFORT or "aucun",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Wiki indisponible au démarrage : %s", exc)
    logger.info(
        "Voix : transcription %s, synthèse %s, voix %s",
        settings.VOCAL_MODELE_TRANSCRIPTION,
        settings.VOCAL_MODELE_SYNTHESE,
        settings.VOCAL_VOIX,
    )
    if not settings.MISTRAL_API_KEY:
        logger.warning("MISTRAL_API_KEY est vide : le chat répondra une erreur.")
    else:
        from app.services.vocal_service import prechauffer_attentes

        # En arrière-plan : le premier tour vocal ne doit pas attendre la synthèse des phrases
        # d'attente, et le démarrage ne doit pas attendre l'API.
        asyncio.create_task(prechauffer_attentes())


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request, session: Session = Depends(get_session)):
    """Le chat — l'écran d'accueil."""
    user = _get_authenticated_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})


@app.get("/vocal", response_class=HTMLResponse)
async def vocal_page(request: Request, session: Session = Depends(get_session)):
    """L'assistant vocal : on parle, LIA cherche dans le wiki et répond de vive voix."""
    user = _get_authenticated_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("vocal.html", {"request": request, "user": user})


@app.get("/wiki", response_class=HTMLResponse)
async def wiki_page(request: Request, session: Session = Depends(get_session)):
    """Le wiki : graphe et lecteur de pages."""
    user = _get_authenticated_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("wiki.html", {"request": request, "user": user})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, session: Session = Depends(get_session)):
    if _get_authenticated_user(request, session):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, session: Session = Depends(get_session)):
    if _get_authenticated_user(request, session):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request, session: Session = Depends(get_session)):
    if not _get_authenticated_user(request, session):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/feedbacks", response_class=HTMLResponse)
async def feedbacks_page(request: Request, session: Session = Depends(get_session)):
    user = _get_authenticated_user(request, session)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("feedbacks.html", {"request": request, "user": user})


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


# ---------------------------------------------------------------------------
# Authentification des pages (cookie ou Bearer)
# ---------------------------------------------------------------------------


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
    roles = [ur.role.name for ur in user.user_roles] if hasattr(user, "user_roles") else []
    return UserRead(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
        roles=roles,
    )
