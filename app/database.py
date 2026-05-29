from sqlmodel import SQLModel, create_engine, Session, text
from app.config import settings
import logging
import re

logger = logging.getLogger(__name__)

# echo=False par défaut : echo=True rejoue chaque SQL dans les logs (INSERT/UPDATE d'embeddings = milliers de floats).
engine = create_engine(settings.DATABASE_URL, echo=settings.DATABASE_ECHO)


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


def reset_db_sequences():
    """Resets all PostgreSQL sequences to match the maximum ID in the tables, preventing UniqueViolation errors."""
    try:
        with engine.connect() as conn:
            conn = conn.execution_options(autocommit=True)
            
            # Find all sequences by inspecting column defaults and using pg_get_serial_sequence
            query = """
            SELECT 
                table_name, 
                column_name, 
                column_default,
                pg_get_serial_sequence('public.' || table_name, column_name) AS pg_seq
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
              AND (column_default LIKE 'nextval%' OR identity_generation IS NOT NULL);
            """
            result = conn.execute(text(query))
            for row in result:
                table_name = row[0]
                column_name = row[1]
                column_default = row[2]
                pg_seq = row[3]
                
                # Determine sequence name
                sequence_name = None
                if pg_seq:
                    sequence_name = pg_seq
                elif column_default and 'nextval' in column_default:
                    # Extract sequence name from nextval('sequence_name'::regclass)
                    match = re.search(r"nextval\('([^']+)'", column_default)
                    if match:
                        sequence_name = match.group(1)
                
                if not sequence_name:
                    # Construct default name fallback
                    sequence_name = f"{table_name}_{column_name}_seq"
                
                logger.info(f"Resetting sequence {sequence_name} for table {table_name}({column_name})...")
                
                reset_query = f"""
                SELECT setval(
                    '{sequence_name}', 
                    COALESCE((SELECT MAX("{column_name}") FROM "{table_name}"), 1), 
                    COALESCE((SELECT MAX("{column_name}") FROM "{table_name}") IS NOT NULL, false)
                );
                """
                conn.execute(text(reset_query))
        logger.info("Successfully reset all PostgreSQL sequences.")
    except Exception as e:
        logger.warning(f"Could not reset PostgreSQL sequences: {e}", exc_info=True)


def create_db_and_tables():
    """Créer toutes les tables"""
    # Corriger le mismatch de collation avant de créer les tables
    fix_collation_version()
    SQLModel.metadata.create_all(engine)
    # Réinitialiser les séquences d'ID pour éviter les erreurs d'unicité (UniqueViolation)
    reset_db_sequences()

