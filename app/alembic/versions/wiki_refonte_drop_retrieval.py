"""Refonte wiki (2026-09-18) : LIA répond depuis le wiki, tout le reste disparaît.

Supprime les tables de la bibliothèque (documents, chunks, dossiers, espaces), du
retriever (entités KAG, catégories, synthèses), des arbres SAV et des gammes, ainsi que les
colonnes qui les référençaient sur ``conversation`` et ``message_feedback``. L'extension
pgvector n'a plus de consommateur.

Idempotente (``IF EXISTS`` partout) : ``create_all`` tourne aussi au démarrage, et la
migration peut être rejouée sur une base déjà migrée. Irréversible : les données de ces
tables n'ont plus de code pour les lire.

Revision ID: wiki_refonte_drop_retrieval
Revises: remove_ctx_enrichment_chunks
Create Date: 2026-09-18
"""
from __future__ import annotations

from alembic import op
from sqlalchemy import text

revision = "wiki_refonte_drop_retrieval"
down_revision = "remove_ctx_enrichment_chunks"
branch_labels = None
depends_on = None

# Ce qui reste : comptes, rôles, conversations, retours utilisateurs, journal d'audit.
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
        "adminauditlog",
    }
)

DROP_COLUMNS = {
    "conversation": ["space_id"],
    "message_feedback": ["space_id", "chunk_ids", "auto_faq_generated", "auto_faq_content"],
}


def upgrade() -> None:
    conn = op.get_bind()
    for table, columns in DROP_COLUMNS.items():
        for column in columns:
            conn.execute(text(f'ALTER TABLE IF EXISTS "{table}" DROP COLUMN IF EXISTS "{column}"'))

    # Toute table du schéma public qui n'est pas dans KEEP appartient au système remplacé
    # (bibliothèque, retriever, KAG, guidé, gammes, tables héritées des anciennes migrations).
    rows = conn.execute(
        text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    ).fetchall()
    for (name,) in rows:
        if name in KEEP:
            continue
        conn.execute(text(f'DROP TABLE IF EXISTS "{name}" CASCADE'))

    conn.execute(text("DROP EXTENSION IF EXISTS vector CASCADE"))


def downgrade() -> None:
    raise RuntimeError(
        "Migration irréversible : les tables supprimées n'ont plus de modèle ni de code."
    )
