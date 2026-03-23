"""add note processing_progress

Revision ID: add_note_processing_progress
Revises: add_notechunk_hierarchy
Create Date: 2026-02-19 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "add_note_processing_progress"
down_revision = "add_notechunk_hierarchy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'note'
            """
        )
    )
    existing_cols = {row[0] for row in result}

    if "processing_progress" not in existing_cols:
        op.add_column("note", sa.Column("processing_progress", sa.Integer(), nullable=True))

    # Backfill cohérent avec processing_status existant
    conn.execute(
        sa.text(
            """
            UPDATE note
            SET processing_progress = CASE
                WHEN processing_status = 'completed' THEN 100
                WHEN processing_status = 'pending' THEN 0
                WHEN processing_status = 'processing' THEN COALESCE(processing_progress, 10)
                WHEN processing_status = 'failed' THEN COALESCE(processing_progress, 0)
                ELSE COALESCE(processing_progress, 0)
            END
            WHERE processing_progress IS NULL
            """
        )
    )


def downgrade() -> None:
    op.drop_column("note", "processing_progress")
