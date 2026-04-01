"""add is_paid column to document

Revision ID: add_document_is_paid
Revises: add_library_architecture
Create Date: 2026-03-23 10:00:00.000000
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_document_is_paid"
down_revision = "add_library_architecture"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    cols_result = conn.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'document'"
        )
    )
    existing_cols = {row[0] for row in cols_result}
    if "is_paid" not in existing_cols:
        op.add_column(
            "document",
            sa.Column("is_paid", sa.Boolean(), nullable=False, server_default=sa.false()),
        )
        op.alter_column("document", "is_paid", server_default=None)


def downgrade() -> None:
    conn = op.get_bind()
    cols_result = conn.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'document'"
        )
    )
    existing_cols = {row[0] for row in cols_result}
    if "is_paid" in existing_cols:
        op.drop_column("document", "is_paid")
