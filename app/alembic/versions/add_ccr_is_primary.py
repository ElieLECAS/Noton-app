"""add is_primary to chunkcategoryrelation

Marque la catégorie dominante d'un chunk (axe task/symptom). Idempotent : la colonne
peut déjà exister via SQLModel.create_all au démarrage.

Revision ID: add_ccr_is_primary
Revises: add_space_theme_synthesis
"""

import sqlalchemy as sa
from alembic import op


revision = "add_ccr_is_primary"
down_revision = "add_space_theme_synthesis"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("chunkcategoryrelation")}
    if "is_primary" not in columns:
        op.add_column(
            "chunkcategoryrelation",
            sa.Column(
                "is_primary",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
        )


def downgrade() -> None:
    op.execute("ALTER TABLE chunkcategoryrelation DROP COLUMN IF EXISTS is_primary")
