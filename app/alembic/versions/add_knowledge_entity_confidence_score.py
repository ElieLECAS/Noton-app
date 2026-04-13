"""add confidence_score to knowledgeentity

Revision ID: add_ke_confidence
Revises: add_entity_alias_rel
"""

from alembic import op
import sqlalchemy as sa


revision = "add_ke_confidence"
down_revision = "add_entity_alias_rel"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "knowledgeentity",
        sa.Column("confidence_score", sa.Float(), nullable=True),
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledgeentity_confidence_score "
        "ON knowledgeentity (confidence_score)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_knowledgeentity_confidence_score")
    op.drop_column("knowledgeentity", "confidence_score")
