"""Vérification indexation BM25 (tsv_content) et diagnostic chunks vides.

Revision ID: improve_bm25_indexing
Revises: add_document_classification
Create Date: 2026-06-18
"""

from alembic import op


revision = "improve_bm25_indexing"
down_revision = "add_document_classification"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'documentchunk' AND column_name = 'tsv_content'
            ) THEN
                ALTER TABLE documentchunk
                ADD COLUMN tsv_content tsvector
                GENERATED ALWAYS AS (to_tsvector('french', coalesce(content, ''))) STORED;

                CREATE INDEX idx_documentchunk_tsv_content
                ON documentchunk USING GIN (tsv_content);
            END IF;
        END $$;
        """
    )
    op.execute(
        """
        DO $$
        DECLARE
            empty_count INTEGER;
        BEGIN
            SELECT COUNT(*) INTO empty_count
            FROM documentchunk
            WHERE is_leaf = true AND (content IS NULL OR content = '');

            IF empty_count > 0 THEN
                RAISE WARNING 'Attention: % chunks L1 avec content vide (BM25 inutilisable)', empty_count;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    pass
