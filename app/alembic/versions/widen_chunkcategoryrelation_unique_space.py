"""widen chunkcategoryrelation unique constraint to include space_id

Un même chunk peut appartenir à plusieurs espaces : la contrainte d'unicité doit donc
porter sur (chunk_id, category_id, space_id), sinon l'insertion d'une catégorie pour un
document multi-espaces viole l'unicité et fait échouer toute la transaction (perte des
chunks d'enrichissement notamment).

Revision ID: widen_ccr_unique_space
Revises: add_match_symptoms_to_guidedtree
"""

from alembic import op


revision = "widen_ccr_unique_space"
down_revision = "add_match_symptoms_to_guidedtree"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_chunkcategoryrelation_chunk_category")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_chunkcategoryrelation_chunk_category_space
        ON chunkcategoryrelation (chunk_id, category_id, space_id)
        """
    )


def downgrade() -> None:
    # Dédupliquer (garder l'id le plus petit) avant de restaurer la contrainte 2-colonnes,
    # car des lignes multi-espaces violeraient l'ancienne unicité.
    op.execute(
        """
        DELETE FROM chunkcategoryrelation a
        USING chunkcategoryrelation b
        WHERE a.id > b.id
          AND a.chunk_id = b.chunk_id
          AND a.category_id = b.category_id
        """
    )
    op.execute("DROP INDEX IF EXISTS uq_chunkcategoryrelation_chunk_category_space")
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_chunkcategoryrelation_chunk_category
        ON chunkcategoryrelation (chunk_id, category_id)
        """
    )
