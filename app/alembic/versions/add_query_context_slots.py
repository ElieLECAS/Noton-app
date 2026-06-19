"""Migration query_context sur Conversation et metadata_json sur Message."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "add_query_context_slots"
down_revision = "improve_bm25_indexing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation",
        sa.Column("query_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "message",
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("message", "metadata_json")
    op.drop_column("conversation", "query_context")
