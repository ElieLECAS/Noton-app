"""Add KAG space constraints and indexes.

Revision ID: add_kag_space_constraints
Revises: add_library_architecture
Create Date: 2026-03-26
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "add_kag_space_constraints"
down_revision = "add_library_architecture"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Unicité métier pour éviter les doublons d'entité par espace.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledgeentity_space_name_normalized
        ON knowledgeentity (space_id, name_normalized)
        """
    )

    # Index trigram pour accélérer les fallback ILIKE sur name_normalized.
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledgeentity_name_normalized_trgm
        ON knowledgeentity USING gin (name_normalized gin_trgm_ops)
        """
    )

    # Garde-fou score relation [0, 1].
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'ck_chunkentityrelation_relevance_score_range'
            ) THEN
                ALTER TABLE chunkentityrelation
                ADD CONSTRAINT ck_chunkentityrelation_relevance_score_range
                CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0);
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'ck_chunkentityrelation_relevance_score_range'
            ) THEN
                ALTER TABLE chunkentityrelation
                DROP CONSTRAINT ck_chunkentityrelation_relevance_score_range;
            END IF;
        END
        $$;
        """
    )
    op.execute("DROP INDEX IF EXISTS ix_knowledgeentity_name_normalized_trgm")
    op.execute("DROP INDEX IF EXISTS uq_knowledgeentity_space_name_normalized")
