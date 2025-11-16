from sqlmodel import SQLModel, create_engine, Session, text
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Créer le moteur de base de données
engine = create_engine(settings.DATABASE_URL, echo=True)


def get_session():
    """Dependency pour obtenir une session de base de données"""
    with Session(engine) as session:
        yield session


def fix_collation_version():
    """Corriger le mismatch de version de collation PostgreSQL"""
    try:
        # ALTER DATABASE doit être exécuté avec autocommit=True (pas dans une transaction)
        with engine.connect() as conn:
            conn = conn.execution_options(autocommit=True)
            conn.execute(text("ALTER DATABASE noton REFRESH COLLATION VERSION"))
        logger.info("Version de collation PostgreSQL rafraîchie avec succès")
    except Exception as e:
        # Ne pas bloquer le démarrage si ça échoue (peut déjà être fait ou autre erreur)
        logger.warning(f"Impossible de rafraîchir la version de collation (peut être déjà fait): {e}")


def create_db_and_tables():
    """Créer toutes les tables"""
    # Corriger le mismatch de collation avant de créer les tables
    fix_collation_version()
    SQLModel.metadata.create_all(engine)

