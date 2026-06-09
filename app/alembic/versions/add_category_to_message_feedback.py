"""add category to message feedback

Revision ID: add_category_to_message_feedback
Revises: cascade_delete_space_feedback
Create Date: 2026-06-09
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "add_category_to_message_feedback"
down_revision = "cascade_delete_space_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = {c["name"] for c in inspect(bind).get_columns("message_feedback")}
    if "category" not in existing:
        op.add_column(
            "message_feedback",
            sa.Column("category", sa.String(length=100), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("message_feedback", "category")
