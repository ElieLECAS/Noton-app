"""add faq to message feedback

Revision ID: add_faq_to_message_feedback
Revises: add_message_response_variants
Create Date: 2026-05-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "add_faq_to_message_feedback"
down_revision = "add_message_response_variants"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = {c["name"] for c in inspect(bind).get_columns("message_feedback")}
    if "auto_faq_generated" not in existing:
        op.add_column(
            "message_feedback",
            sa.Column(
                "auto_faq_generated",
                sa.Boolean(),
                server_default="false",
                nullable=False,
            ),
        )
    if "auto_faq_content" not in existing:
        op.add_column(
            "message_feedback",
            sa.Column("auto_faq_content", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column("message_feedback", "auto_faq_content")
    op.drop_column("message_feedback", "auto_faq_generated")
