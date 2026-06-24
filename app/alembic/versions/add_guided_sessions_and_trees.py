"""add guided sessions and authored trees (procedural guidance engine)

Revision ID: add_guided_sessions_and_trees
Revises: add_document_categories
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "add_guided_sessions_and_trees"
down_revision = "add_document_categories"
branch_labels = None
depends_on = None


def _existing_tables() -> set[str]:
    conn = op.get_bind()
    inspector = inspect(conn)
    return set(inspector.get_table_names(schema="public"))


def upgrade() -> None:
    existing = _existing_tables()

    if "guidedsession" not in existing:
        op.create_table(
            "guidedsession",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("conversation_id", sa.Integer(), nullable=False),
            sa.Column("space_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("topic", sa.String(length=300), nullable=False, server_default=""),
            sa.Column("flow_kind", sa.String(length=20), nullable=False, server_default="howto"),
            sa.Column("source_mode", sa.String(length=20), nullable=False, server_default="dynamic"),
            sa.Column("authored_tree_id", sa.Integer(), nullable=True),
            sa.Column("current_node_key", sa.String(length=120), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
            sa.Column("path", sa.JSON(), nullable=True),
            sa.Column("accumulated_signals", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.ForeignKeyConstraint(["conversation_id"], ["conversation.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["space_id"], ["space.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_guidedsession_conversation_id", "guidedsession", ["conversation_id"])
        op.create_index("ix_guidedsession_space_id", "guidedsession", ["space_id"])
        op.create_index("ix_guidedsession_authored_tree_id", "guidedsession", ["authored_tree_id"])
        op.create_index("ix_guidedsession_status", "guidedsession", ["status"])

    if "guidedtree" not in existing:
        op.create_table(
            "guidedtree",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("slug", sa.String(length=120), nullable=False),
            sa.Column("title", sa.String(length=300), nullable=False),
            sa.Column("flow_kind", sa.String(length=20), nullable=False, server_default="diagnostic"),
            sa.Column("space_id", sa.Integer(), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("match_keywords", sa.JSON(), nullable=True),
            sa.Column("match_categories", sa.JSON(), nullable=True),
            sa.Column("root_node_key", sa.String(length=120), nullable=False, server_default="root"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.text("now()")),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("uq_guidedtree_slug", "guidedtree", ["slug"], unique=True)
        op.create_index("ix_guidedtree_space_id", "guidedtree", ["space_id"])
        op.create_index("ix_guidedtree_is_active", "guidedtree", ["is_active"])

    if "guidedtreenode" not in existing:
        op.create_table(
            "guidedtreenode",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("tree_id", sa.Integer(), nullable=False),
            sa.Column("node_key", sa.String(length=120), nullable=False),
            sa.Column("step_type", sa.String(length=20), nullable=False, server_default="instruction"),
            sa.Column("message", sa.Text(), nullable=False, server_default=""),
            sa.Column("is_terminal", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column("termination_type", sa.String(length=20), nullable=True),
            sa.Column("retrieval_categories", sa.JSON(), nullable=True),
            sa.Column("retrieval_entities", sa.JSON(), nullable=True),
            sa.Column("step_number_hint", sa.Integer(), nullable=True),
            sa.Column("choices", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(["tree_id"], ["guidedtree.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("tree_id", "node_key", name="uq_guidedtreenode_tree_node"),
        )
        op.create_index("ix_guidedtreenode_tree_id", "guidedtreenode", ["tree_id"])


def downgrade() -> None:
    existing = _existing_tables()
    if "guidedtreenode" in existing:
        op.execute("DROP INDEX IF EXISTS ix_guidedtreenode_tree_id")
        op.drop_table("guidedtreenode")
    if "guidedtree" in existing:
        op.execute("DROP INDEX IF EXISTS ix_guidedtree_is_active")
        op.execute("DROP INDEX IF EXISTS ix_guidedtree_space_id")
        op.execute("DROP INDEX IF EXISTS uq_guidedtree_slug")
        op.drop_table("guidedtree")
    if "guidedsession" in existing:
        op.execute("DROP INDEX IF EXISTS ix_guidedsession_status")
        op.execute("DROP INDEX IF EXISTS ix_guidedsession_authored_tree_id")
        op.execute("DROP INDEX IF EXISTS ix_guidedsession_space_id")
        op.execute("DROP INDEX IF EXISTS ix_guidedsession_conversation_id")
        op.drop_table("guidedsession")
