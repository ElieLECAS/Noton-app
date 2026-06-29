"""add space_theme_synthesis cache table

Cache des synthèses de carte mentale (CAG) par nœud d'arbre thématique, invalidé via
``content_hash`` (empreinte de l'ensemble des chunks injectés).

Revision ID: add_space_theme_synthesis
Revises: ccr_doc_level_categories
"""

import sqlalchemy as sa
from alembic import op


revision = "add_space_theme_synthesis"
down_revision = "ccr_doc_level_categories"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent : la table peut déjà exister via SQLModel.create_all au démarrage.
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "spacethemesynthesis" not in inspector.get_table_names():
        op.create_table(
            "spacethemesynthesis",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column("space_id", sa.Integer(), nullable=False),
            sa.Column("node_key", sa.String(length=128), nullable=False),
            sa.Column("axis", sa.String(length=32), nullable=False, server_default="task"),
            sa.Column("content_hash", sa.String(length=64), nullable=False, server_default=""),
            sa.Column("node_label", sa.String(length=300), nullable=False, server_default=""),
            sa.Column("synthesis_markdown", sa.Text(), nullable=False, server_default=""),
            sa.Column("sources_json", sa.JSON(), nullable=True),
            sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("truncated", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("model", sa.String(length=100), nullable=False, server_default=""),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )

    existing = {ix["name"] for ix in inspector.get_indexes("spacethemesynthesis")} if (
        "spacethemesynthesis" in inspector.get_table_names()
    ) else set()
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_spacethemesynthesis_space_id "
        "ON spacethemesynthesis (space_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_spacethemesynthesis_node_key "
        "ON spacethemesynthesis (node_key)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_space_theme_synthesis_node "
        "ON spacethemesynthesis (space_id, node_key, axis)"
    )
    _ = existing  # documentation : indices déjà présents tolérés (IF NOT EXISTS)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_space_theme_synthesis_node")
    op.execute("DROP INDEX IF EXISTS ix_spacethemesynthesis_node_key")
    op.execute("DROP INDEX IF EXISTS ix_spacethemesynthesis_space_id")
    op.execute("DROP TABLE IF EXISTS spacethemesynthesis")
