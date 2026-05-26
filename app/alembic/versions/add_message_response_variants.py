"""add message response variants

Revision ID: add_message_response_variants
Revises: alter_message_feedback_on_delete
Create Date: 2026-05-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "add_message_response_variants"
down_revision = "alter_message_feedback_on_delete"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("message", sa.Column("response_variants", JSONB(), nullable=True))
    op.add_column(
        "message",
        sa.Column("selected_variant", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("message", "selected_variant")
    op.drop_column("message", "response_variants")
