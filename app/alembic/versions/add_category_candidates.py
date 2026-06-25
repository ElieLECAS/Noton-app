"""add category_candidate table (extensible controlled vocabulary)

Revision ID: add_category_candidates
Revises: add_axis_to_document_categories
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "add_category_candidates"
down_revision = "add_axis_to_document_categories"
branch_labels = None
depends_on = None


def _existing_tables() -> set[str]:
    conn = op.get_bind()
    inspector = inspect(conn)
    return set(inspector.get_table_names(schema="public"))


def upgrade() -> None:
    if "categorycandidate" in _existing_tables():
        return
    op.create_table(
        "categorycandidate",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("axis", sa.String(length=32), nullable=False, server_default="symptom"),
        sa.Column("proposed_description", sa.Text(), nullable=False, server_default=""),
        sa.Column("occurrence_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("first_seen_document_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_index("ix_categorycandidate_slug", "categorycandidate", ["slug"], unique=True)
    op.create_index("ix_categorycandidate_axis", "categorycandidate", ["axis"])
    op.create_index("ix_categorycandidate_status", "categorycandidate", ["status"])


def downgrade() -> None:
    if "categorycandidate" not in _existing_tables():
        return
    op.execute("DROP INDEX IF EXISTS ix_categorycandidate_status")
    op.execute("DROP INDEX IF EXISTS ix_categorycandidate_axis")
    op.execute("DROP INDEX IF EXISTS ix_categorycandidate_slug")
    op.drop_table("categorycandidate")
