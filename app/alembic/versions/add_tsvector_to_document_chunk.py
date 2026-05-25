"""Add tsvector column and GIN index to documentchunk.

Revision ID: add_tsvector_to_document_chunk
Revises: drop_kag_tables
Create Date: 2026-05-20
"""

from alembic import op


revision = "add_tsvector_to_document_chunk"
down_revision = "drop_kag_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Ajouter la colonne tsvector générée automatiquement si elle n'existe pas
    op.execute(
        """
        ALTER TABLE documentchunk 
        ADD COLUMN IF NOT EXISTS tsv_content tsvector 
        GENERATED ALWAYS AS (to_tsvector('french', coalesce(content, ''))) STORED;
        """
    )
    # Créer l'index GIN pour accélérer la recherche textuelle
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_documentchunk_tsv_content 
        ON documentchunk USING gin (tsv_content);
        """
    )


def downgrade() -> None:
    # Supprimer l'index et la colonne
    op.execute("DROP INDEX IF EXISTS ix_documentchunk_tsv_content;")
    op.execute("ALTER TABLE documentchunk DROP COLUMN IF EXISTS tsv_content;")
