"""alter message feedback on delete

Revision ID: alter_message_feedback_on_delete
Revises: add_message_feedback
Create Date: 2026-05-25
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "alter_message_feedback_on_delete"
down_revision = "add_message_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Add response_text column
    op.add_column("message_feedback", sa.Column("response_text", sa.Text(), nullable=True))
    
    # 2. Make message_id nullable
    op.alter_column("message_feedback", "message_id", existing_type=sa.Integer(), nullable=True)
    
    # 3. Drop existing foreign key constraint
    op.execute("""
        ALTER TABLE message_feedback 
        DROP CONSTRAINT IF EXISTS message_feedback_message_id_fkey
    """)
    
    # 4. Add new foreign key constraint with ON DELETE SET NULL
    op.create_foreign_key(
        "message_feedback_message_id_fkey",
        "message_feedback",
        "message",
        ["message_id"],
        ["id"],
        ondelete="SET NULL"
    )


def downgrade() -> None:
    # 1. Drop new foreign key constraint
    op.execute("""
        ALTER TABLE message_feedback 
        DROP CONSTRAINT IF EXISTS message_feedback_message_id_fkey
    """)
    
    # 2. Re-add old foreign key constraint with ON DELETE CASCADE
    op.create_foreign_key(
        "message_feedback_message_id_fkey",
        "message_feedback",
        "message",
        ["message_id"],
        ["id"],
        ondelete="CASCADE"
    )
    
    # 3. Make message_id non-nullable
    op.alter_column("message_feedback", "message_id", existing_type=sa.Integer(), nullable=False)
    
    # 4. Drop response_text column
    op.drop_column("message_feedback", "response_text")
