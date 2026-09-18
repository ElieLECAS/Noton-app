"""La migration unique : rejouable sur une base déjà en place, et elle nettoie l'ancien schéma.

C'est le chemin de déploiement d'une base migrée par l'ancienne chaîne : la commande du
conteneur efface la table de version (``stamp --purge base``) puis rejoue ``upgrade head``.
Les tables conservées gardent leurs lignes ; les colonnes et tables du système remplacé
disparaissent.
"""
from __future__ import annotations

from alembic import command
from sqlalchemy import inspect, text

from app.database import engine
from tests.conftest import alembic_config

EXPECTED_TABLES = {
    "alembic_version", "user", "role", "permission", "userrole", "rolepermission",
    "conversation", "message", "message_feedback",
}


def test_single_migration_is_idempotent_and_drops_legacy_schema(_init_db):
    with engine.connect() as conn:
        roles_before = conn.execute(text("SELECT COUNT(*) FROM role")).scalar()
        # Un reste de l'ancien système : une table et deux colonnes que le modèle ne connaît plus.
        conn.execute(text("CREATE TABLE IF NOT EXISTS documentchunk (id SERIAL PRIMARY KEY)"))
        conn.execute(text("ALTER TABLE conversation ADD COLUMN IF NOT EXISTS query_context JSONB"))
        conn.execute(text("ALTER TABLE message ADD COLUMN IF NOT EXISTS provider VARCHAR(50)"))
        conn.commit()

    cfg = alembic_config()
    command.stamp(cfg, "base", purge=True)
    command.upgrade(cfg, "head")

    with engine.connect() as conn:
        inspector = inspect(conn)
        assert set(inspector.get_table_names()) == EXPECTED_TABLES
        assert "query_context" not in {c["name"] for c in inspector.get_columns("conversation")}
        assert "provider" not in {c["name"] for c in inspector.get_columns("message")}
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "lia_wiki_schema"
        # Les données des tables conservées survivent au rejeu.
        assert conn.execute(text("SELECT COUNT(*) FROM role")).scalar() == roles_before
        assert roles_before >= 3
