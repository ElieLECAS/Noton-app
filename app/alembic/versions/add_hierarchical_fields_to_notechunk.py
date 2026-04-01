"""add hierarchical fields to notechunk for llamaindex

Revision ID: add_notechunk_hierarchy
Revises: add_task_run_log_schedule
Create Date: 2026-02-18 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "add_notechunk_hierarchy"
down_revision = "add_task_run_log_schedule"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    table_exists = conn.execute(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'notechunk'
            )
            """
        )
    ).scalar()
    if not table_exists:
        print("add_notechunk_hierarchy: table notechunk absente, migration ignorée.")
        return

    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'notechunk'
            """
        )
    )
    existing_cols = {row[0] for row in result}

    if "text" not in existing_cols:
        op.add_column("notechunk", sa.Column("text", sa.Text(), nullable=True))
    if "node_id" not in existing_cols:
        op.add_column("notechunk", sa.Column("node_id", sa.String(length=255), nullable=True))
    if "parent_node_id" not in existing_cols:
        op.add_column("notechunk", sa.Column("parent_node_id", sa.String(length=255), nullable=True))
    if "is_leaf" not in existing_cols:
        op.add_column("notechunk", sa.Column("is_leaf", sa.Boolean(), nullable=False, server_default=sa.true()))
    if "hierarchy_level" not in existing_cols:
        op.add_column("notechunk", sa.Column("hierarchy_level", sa.Integer(), nullable=False, server_default="0"))
    if "metadata_json" not in existing_cols:
        op.add_column("notechunk", sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    if "metadata_" not in existing_cols:
        op.add_column("notechunk", sa.Column("metadata_", postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    # Backfill pour compatibilité: text = content sur l'historique.
    conn.execute(sa.text("UPDATE notechunk SET text = content WHERE text IS NULL"))

    op.execute("CREATE INDEX IF NOT EXISTS ix_notechunk_node_id ON notechunk (node_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_notechunk_parent_node_id ON notechunk (parent_node_id)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_notechunk_is_leaf ON notechunk (is_leaf)")
    op.execute("CREATE INDEX IF NOT EXISTS ix_notechunk_hierarchy_level ON notechunk (hierarchy_level)")


def downgrade() -> None:
    conn = op.get_bind()
    table_exists = conn.execute(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'notechunk'
            )
            """
        )
    ).scalar()
    if not table_exists:
        return

    op.execute("DROP INDEX IF EXISTS ix_notechunk_hierarchy_level")
    op.execute("DROP INDEX IF EXISTS ix_notechunk_is_leaf")
    op.execute("DROP INDEX IF EXISTS ix_notechunk_parent_node_id")
    op.execute("DROP INDEX IF EXISTS ix_notechunk_node_id")

    op.drop_column("notechunk", "metadata_")
    op.drop_column("notechunk", "metadata_json")
    op.drop_column("notechunk", "hierarchy_level")
    op.drop_column("notechunk", "is_leaf")
    op.drop_column("notechunk", "parent_node_id")
    op.drop_column("notechunk", "node_id")
    op.drop_column("notechunk", "text")
