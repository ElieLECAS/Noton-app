"""cascade delete space feedback

Revision ID: cascade_delete_space_feedback
Revises: add_faq_to_message_feedback
Create Date: 2026-06-08
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "cascade_delete_space_feedback"
down_revision = "add_faq_to_message_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Drop existing foreign key constraint
    op.execute("""
        ALTER TABLE message_feedback 
        DROP CONSTRAINT IF EXISTS message_feedback_space_id_fkey
    """)
    
    # 2. Add new foreign key constraint with ON DELETE CASCADE
    op.create_foreign_key(
        "message_feedback_space_id_fkey",
        "message_feedback",
        "space",
        ["space_id"],
        ["id"],
        ondelete="CASCADE"
    )


def downgrade() -> None:
    # 1. Drop new foreign key constraint
    op.execute("""
        ALTER TABLE message_feedback 
        DROP CONSTRAINT IF EXISTS message_feedback_space_id_fkey
    """)
    
    # 2. Re-add old foreign key constraint (no ondelete, defaults to restrict/no action)
    op.create_foreign_key(
        "message_feedback_space_id_fkey",
        "message_feedback",
        "space",
        ["space_id"],
        ["id"]
    )
