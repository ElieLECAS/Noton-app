"""make chunkcategoryrelation document-level (drop space_id)

La catégorisation est intrinsèque au contenu du document, pas à l'espace. On supprime
``space_id`` du lien : une seule ligne par (chunk_id, category_id). Le scope par espace est
dérivé à la requête via la table d'appartenance ``documentspace``.

Revision ID: ccr_doc_level_categories
Revises: widen_ccr_unique_space
"""

from alembic import op


revision = "ccr_doc_level_categories"
down_revision = "widen_ccr_unique_space"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Dédupliquer les lignes multi-espaces (garder l'id le plus petit par chunk+catégorie).
    op.execute(
        """
        DELETE FROM chunkcategoryrelation a
        USING chunkcategoryrelation b
        WHERE a.id > b.id
          AND a.chunk_id = b.chunk_id
          AND a.category_id = b.category_id
        """
    )
    # 2. Retirer la contrainte d'unicité incluant space_id et l'index space_id.
    op.execute("DROP INDEX IF EXISTS uq_chunkcategoryrelation_chunk_category_space")
    op.execute("DROP INDEX IF EXISTS ix_chunkcategoryrelation_space_id")
    # 3. Supprimer la colonne space_id.
    op.execute("ALTER TABLE chunkcategoryrelation DROP COLUMN IF EXISTS space_id")
    # 4. Rétablir l'unicité au niveau document (chunk, catégorie).
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_chunkcategoryrelation_chunk_category
        ON chunkcategoryrelation (chunk_id, category_id)
        """
    )


def downgrade() -> None:
    # Ré-introduire space_id (best-effort) : on l'affecte au premier espace de chaque document.
    op.execute(
        "ALTER TABLE chunkcategoryrelation ADD COLUMN IF NOT EXISTS space_id INTEGER"
    )
    op.execute(
        """
        UPDATE chunkcategoryrelation ccr
        SET space_id = ds.space_id
        FROM (
            SELECT document_id, MIN(space_id) AS space_id
            FROM document_space GROUP BY document_id
        ) ds
        WHERE ds.document_id = ccr.document_id AND ccr.space_id IS NULL
        """
    )
    op.execute("DROP INDEX IF EXISTS uq_chunkcategoryrelation_chunk_category")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_chunkcategoryrelation_space_id "
        "ON chunkcategoryrelation (space_id)"
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_chunkcategoryrelation_chunk_category_space
        ON chunkcategoryrelation (chunk_id, category_id, space_id)
        """
    )
