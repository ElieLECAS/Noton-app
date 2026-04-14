"""
Fixtures pytest : environnement, PostgreSQL, RBAC, client HTTP authentifié.

DATABASE_URL : utiliser une base dédiée aux tests (ex. noton_test).
Priorité : variable d'environnement DATABASE_URL, sinon PYTEST_DATABASE_URL,
sinon postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/noton_test
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Generator
from unittest import mock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

# Garantit l'import du package "app" en local et dans Docker.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Doit s'exécuter avant tout import de app.* (settings / engine).
# ---------------------------------------------------------------------------
os.environ.setdefault(
    "SECRET_KEY",
    "pytest-secret-key-do-not-use-in-production-min-32-chars",
)

def _is_test_db_name(name: str | None) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return lowered.endswith("_test") or "_pytest_" in lowered


def _build_isolated_test_url() -> str:
    """
    Force une DB de tests isolée, différente de la DB de l'app.
    Si PYTEST_DATABASE_URL est défini, on l'utilise uniquement si son nom contient
    un marqueur de test explicite.
    """
    explicit_pytest_url = os.environ.get("PYTEST_DATABASE_URL")
    if explicit_pytest_url:
        parsed = make_url(explicit_pytest_url)
        if not _is_test_db_name(parsed.database):
            raise RuntimeError(
                "PYTEST_DATABASE_URL doit cibler une base de test "
                "(nom finissant par _test ou contenant _pytest_)."
            )
        return explicit_pytest_url

    base_url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/noton",
    )
    try:
        parsed = make_url(base_url)
        db_name = parsed.database or "noton"
        test_name = db_name if db_name.endswith("_test") else f"{db_name}_test"
        # PostgreSQL limite les noms d'objets à 63 caractères.
        test_name = test_name[:63]
        return parsed.set(database=test_name).render_as_string(hide_password=False)
    except Exception:
        return "postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/noton_test"


os.environ["DATABASE_URL"] = _build_isolated_test_url()
os.environ.setdefault("TASK_BACKEND_MODE", "celery")
os.environ.setdefault("MISTRAL_API_KEY", "pytest-mistral-key")

from starlette.testclient import TestClient
from sqlmodel import Session, select

from app.database import engine
from app.embedding_config import EMBEDDING_DIMENSION
from app.main import app
from app.models.role import Role
from app.models.user import User
from app.models.user_role import UserRole
from app.services.auth_service import create_access_token, get_password_hash
from app.services.rbac_seed_service import seed_rbac_system
from sqlmodel import SQLModel


def _ensure_database_exists() -> None:
    target_url = os.environ["DATABASE_URL"]
    parsed = make_url(target_url)
    target_db = parsed.database
    if not target_db:
        return

    admin_db = "postgres" if target_db != "postgres" else "template1"
    admin_url = parsed.set(database=admin_db).render_as_string(hide_password=False)

    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db"),
                {"db": target_db},
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{target_db}"'))
    finally:
        admin_engine.dispose()


def _truncate_all_tables() -> None:
    with engine.connect() as conn:
        table_names = conn.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' ORDER BY tablename"
            )
        ).scalars().all()
        if table_names:
            table_list = ", ".join(f'"{name}"' for name in table_names)
            conn.execute(text(f"TRUNCATE TABLE {table_list} RESTART IDENTITY CASCADE"))
            conn.commit()


def _sync_knowledgeentity_confidence_score_column() -> None:
    """
    create_all ne modifie pas les tables existantes : une base déjà peuplée peut
    manquer des colonnes ajoutées après coup (ex. migration add_ke_confidence).
    """
    try:
        with engine.connect() as conn:
            has_table = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = 'knowledgeentity'"
                )
            ).scalar()
            if not has_table:
                return
            conn.execute(
                text(
                    "ALTER TABLE knowledgeentity "
                    "ADD COLUMN IF NOT EXISTS confidence_score DOUBLE PRECISION"
                )
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_knowledgeentity_confidence_score "
                    "ON knowledgeentity (confidence_score)"
                )
            )
            conn.commit()
    except Exception:
        # Ne pas bloquer la suite si le schéma diffère (SQLite, etc.)
        pass


def assign_role(session: Session, user_id: int, role_name: str) -> None:
    for ur in session.exec(select(UserRole).where(UserRole.user_id == user_id)).all():
        session.delete(ur)
    session.commit()
    role = session.exec(select(Role).where(Role.name == role_name)).first()
    if role is None:
        raise RuntimeError(f"Rôle {role_name!r} introuvable — exécuter seed_rbac_system")
    session.add(UserRole(user_id=user_id, role_id=role.id))
    session.commit()


def create_test_user(session: Session, role_name: str) -> User:
    suffix = uuid.uuid4().hex[:10]
    user = User(
        username=f"u_{role_name}_{suffix}",
        email=f"{role_name}_{suffix}@pytest.example.com",
        password_hash=get_password_hash("pytest-pass-123"),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    assign_role(session, user.id, role_name)
    session.refresh(user)
    return user


def bearer_headers(user_id: int) -> dict:
    token = create_access_token(
        data={"sub": str(user_id)},
        expires_delta=timedelta(hours=2),
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session", autouse=True)
def _patch_embedding_startup() -> Generator[None, None, None]:
    fake = [0.0] * EMBEDDING_DIMENSION
    with mock.patch(
        "app.services.embedding_service.generate_embedding",
        return_value=fake,
    ):
        yield


@pytest.fixture(scope="session", autouse=True)
def _init_db() -> Generator[None, None, None]:
    parsed = make_url(os.environ["DATABASE_URL"])
    if not _is_test_db_name(parsed.database):
        raise RuntimeError(
            "Refus d'exécuter les tests sur une base non identifiée comme base de test."
        )

    _ensure_database_exists()
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
    except Exception:
        pass
    SQLModel.metadata.create_all(engine)
    _sync_knowledgeentity_confidence_score_column()
    _truncate_all_tables()
    with Session(engine) as session:
        seed_rbac_system(session)
    yield
    _truncate_all_tables()


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


@pytest.fixture
def client(_init_db) -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_headers(db_session: Session) -> dict:
    user = create_test_user(db_session, "admin")
    return bearer_headers(user.id)


@pytest.fixture
def lecteur_headers(db_session: Session) -> dict:
    user = create_test_user(db_session, "lecteur")
    return bearer_headers(user.id)


@pytest.fixture
def responsable_headers(db_session: Session) -> dict:
    user = create_test_user(db_session, "responsable")
    return bearer_headers(user.id)
