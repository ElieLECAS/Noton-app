"""Moteur SQL et session.

Le schéma est posé par la migration unique (``app/alembic/versions/lia_wiki_schema.py``) ;
``create_all`` au démarrage est le filet de sécurité de la maison — idempotent, il ne touche
pas une table qui existe déjà.
"""
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

engine = create_engine(settings.DATABASE_URL, echo=settings.DATABASE_ECHO)


def get_session():
    """Dépendance FastAPI : une session par requête."""
    with Session(engine) as session:
        yield session


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
