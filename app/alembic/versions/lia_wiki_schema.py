"""Le schéma de LIA, en une seule migration.

Huit tables : comptes et rôles (``user``, ``role``, ``permission``, ``userrole``,
``rolepermission``), conversations (``conversation``, ``message``) et retours utilisateurs
(``message_feedback``). Tout le reste — bibliothèque, chunks, embeddings pgvector, KAG, arbres
SAV, espaces, gammes, journal d'audit admin, colonnes de compréhension de question — appartient
au système remplacé et est supprimé ici, ainsi que les extensions ``vector`` et ``pg_trgm``.

Idempotente (``IF NOT EXISTS`` / ``IF EXISTS`` partout) : ``create_all`` tourne aussi au
démarrage, et une base migrée par l'ancienne chaîne (51 révisions, supprimées le 18/09/2026)
est reprise par ``alembic stamp --purge base && alembic upgrade head`` — la commande du
conteneur le fait d'elle-même quand ``upgrade head`` ne reconnaît pas la révision inscrite.
Les comptes, conversations, messages et retours sont conservés.

Irréversible : les tables supprimées n'ont plus ni modèle ni code.

Revision ID: lia_wiki_schema
Revises: —
Create Date: 2026-09-18
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "lia_wiki_schema"
down_revision = None
branch_labels = None
depends_on = None

KEEP = frozenset(
    {
        "alembic_version",
        "user",
        "role",
        "permission",
        "userrole",
        "rolepermission",
        "conversation",
        "message",
        "message_feedback",
    }
)

# Colonnes des tables conservées qui appartenaient au système remplacé.
DROP_COLUMNS = {
    "conversation": ["space_id", "query_context"],
    "message": ["response_variants", "selected_variant", "provider"],
    "message_feedback": ["space_id", "chunk_ids", "auto_faq_generated", "auto_faq_content"],
}

TABLES = [
    """
    CREATE TABLE IF NOT EXISTS "user" (
        id SERIAL PRIMARY KEY,
        username VARCHAR(150) NOT NULL,
        email VARCHAR(255) NOT NULL,
        password_hash VARCHAR NOT NULL,
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS role (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        description VARCHAR(500),
        is_system BOOLEAN NOT NULL,
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS permission (
        id SERIAL PRIMARY KEY,
        code VARCHAR(100) NOT NULL,
        name VARCHAR(200) NOT NULL,
        description VARCHAR(500),
        category VARCHAR(50) NOT NULL,
        is_system BOOLEAN NOT NULL,
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS userrole (
        id SERIAL PRIMARY KEY,
        user_id INTEGER NOT NULL REFERENCES "user"(id),
        role_id INTEGER NOT NULL REFERENCES role(id),
        assigned_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        assigned_by INTEGER REFERENCES "user"(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rolepermission (
        id SERIAL PRIMARY KEY,
        role_id INTEGER NOT NULL REFERENCES role(id),
        permission_id INTEGER NOT NULL REFERENCES permission(id),
        assigned_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversation (
        id SERIAL PRIMARY KEY,
        title VARCHAR(200) NOT NULL,
        user_id INTEGER NOT NULL REFERENCES "user"(id) ON DELETE CASCADE,
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
        updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS message (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
        role VARCHAR(50) NOT NULL,
        content TEXT NOT NULL,
        model VARCHAR(100),
        sources TEXT,
        metadata_json JSONB,
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS message_feedback (
        id SERIAL PRIMARY KEY,
        message_id INTEGER REFERENCES message(id) ON DELETE SET NULL,
        user_id INTEGER NOT NULL REFERENCES "user"(id),
        is_positive BOOLEAN NOT NULL,
        comment TEXT,
        query_text TEXT NOT NULL,
        response_text TEXT,
        category VARCHAR(100),
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
        updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
        CONSTRAINT uq_feedback_message_user UNIQUE (message_id, user_id)
    )
    """,
]

INDEXES = [
    'CREATE UNIQUE INDEX IF NOT EXISTS ix_user_username ON "user" (username)',
    'CREATE UNIQUE INDEX IF NOT EXISTS ix_user_email ON "user" (email)',
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_role_name ON role (name)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ix_permission_code ON permission (code)",
    "CREATE INDEX IF NOT EXISTS ix_userrole_user_id ON userrole (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_userrole_role_id ON userrole (role_id)",
    "CREATE INDEX IF NOT EXISTS ix_rolepermission_role_id ON rolepermission (role_id)",
    "CREATE INDEX IF NOT EXISTS ix_rolepermission_permission_id ON rolepermission (permission_id)",
    "CREATE INDEX IF NOT EXISTS ix_conversation_user_id ON conversation (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_conversation_updated_at ON conversation (updated_at)",
    "CREATE INDEX IF NOT EXISTS ix_message_conversation_id ON message (conversation_id)",
    "CREATE INDEX IF NOT EXISTS ix_message_created_at ON message (created_at)",
    "CREATE INDEX IF NOT EXISTS ix_message_feedback_message_id ON message_feedback (message_id)",
    "CREATE INDEX IF NOT EXISTS ix_message_feedback_user_id ON message_feedback (user_id)",
]


def upgrade() -> None:
    conn = op.get_bind()

    for ddl in TABLES:
        conn.execute(text(ddl))
    for ddl in INDEXES:
        conn.execute(text(ddl))

    # Toute table du schéma public qui n'est pas dans KEEP appartient au système remplacé.
    rows = conn.execute(text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")).fetchall()
    for (name,) in rows:
        if name not in KEEP:
            conn.execute(text(f'DROP TABLE IF EXISTS "{name}" CASCADE'))

    for table, columns in DROP_COLUMNS.items():
        for column in columns:
            conn.execute(text(f'ALTER TABLE IF EXISTS "{table}" DROP COLUMN IF EXISTS "{column}"'))

    conn.execute(text("DROP EXTENSION IF EXISTS vector CASCADE"))
    conn.execute(text("DROP EXTENSION IF EXISTS pg_trgm CASCADE"))


def downgrade() -> None:
    raise RuntimeError("Migration irréversible : les tables supprimées n'ont plus de modèle ni de code.")
