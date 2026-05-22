"""add sources to message

Revision ID: add_sources_to_message
Revises: add_hnsw_and_remove_legacy
Create Date: 2026-05-22
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_sources_to_message"
down_revision = "add_hnsw_and_remove_legacy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    cols_result = conn.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'message'"
        )
    )
    existing_cols = {row[0] for row in cols_result}
    if "sources" not in existing_cols:
        op.add_column(
            "message",
            sa.Column("sources", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    cols_result = conn.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'message'"
        )
    )
    existing_cols = {row[0] for row in cols_result}
    if "sources" in existing_cols:
        op.drop_column("message", "sources")
