"""add message feedback table

Revision ID: add_message_feedback
Revises: add_sources_to_message
Create Date: 2026-05-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = "add_message_feedback"
down_revision = "add_sources_to_message"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "message_feedback",
        sa.Column("id", sa.Integer(), nullable=False, primary_key=True),
        sa.Column("message_id", sa.Integer(), sa.ForeignKey("message.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("space_id", sa.Integer(), sa.ForeignKey("space.id"), nullable=False),
        sa.Column("is_positive", sa.Boolean(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("chunk_ids", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    # Unique constraint
    op.create_unique_constraint(
        "uq_feedback_message_user", "message_feedback", ["message_id", "user_id"]
    )
    # Indexes
    op.create_index(
        "ix_message_feedback_message_id", "message_feedback", ["message_id"]
    )
    op.create_index(
        "ix_message_feedback_user_id", "message_feedback", ["user_id"]
    )
    op.create_index(
        "ix_message_feedback_space_id", "message_feedback", ["space_id"]
    )
    op.create_index(
        "ix_message_feedback_space_positive", "message_feedback", ["space_id", "is_positive"]
    )


def downgrade() -> None:
    op.drop_index("ix_message_feedback_space_positive", table_name="message_feedback")
    op.drop_index("ix_message_feedback_space_id", table_name="message_feedback")
    op.drop_index("ix_message_feedback_user_id", table_name="message_feedback")
    op.drop_index("ix_message_feedback_message_id", table_name="message_feedback")
    op.drop_constraint("uq_feedback_message_user", "message_feedback")
    op.drop_table("message_feedback")
