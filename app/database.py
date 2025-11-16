from sqlmodel import SQLModel, create_engine, Session
from app.config import settings

# Créer le moteur de base de données
engine = create_engine(settings.DATABASE_URL, echo=True)


def get_session():
    """Dependency pour obtenir une session de base de données"""
    with Session(engine) as session:
        yield session


def create_db_and_tables():
    """Créer toutes les tables"""
    SQLModel.metadata.create_all(engine)

