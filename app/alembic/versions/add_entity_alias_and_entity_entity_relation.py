"""add entity_entity_relation and entity_alias tables

Revision ID: add_entity_alias_rel
Revises: add_kag_space_constraints
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "add_entity_alias_rel"
down_revision = "add_kag_space_constraints"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names(schema="public"))

    if "entityalias" not in existing_tables:
        op.create_table(
            "entityalias",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("space_id", sa.Integer(), sa.ForeignKey("space.id", ondelete="CASCADE"), nullable=False),
            sa.Column("entity_id", sa.Integer(), sa.ForeignKey("knowledgeentity.id", ondelete="CASCADE"), nullable=False),
            sa.Column("alias_normalized", sa.String(length=500), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entityalias_space_id ON entityalias (space_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entityalias_entity_id ON entityalias (entity_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entityalias_alias_normalized ON entityalias (alias_normalized)"
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_entityalias_space_alias
        ON entityalias (space_id, alias_normalized)
        """
    )

    if "entityentityrelation" not in existing_tables:
        op.create_table(
            "entityentityrelation",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("space_id", sa.Integer(), sa.ForeignKey("space.id", ondelete="CASCADE"), nullable=False),
            sa.Column("entity_a_id", sa.Integer(), sa.ForeignKey("knowledgeentity.id", ondelete="CASCADE"), nullable=False),
            sa.Column("entity_b_id", sa.Integer(), sa.ForeignKey("knowledgeentity.id", ondelete="CASCADE"), nullable=False),
            sa.Column("relation_type", sa.String(length=64), nullable=False, server_default="co_occurs"),
            sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column(
                "source_chunk_id",
                sa.Integer(),
                sa.ForeignKey("documentchunk.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("confidence", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entityentityrelation_space_id ON entityentityrelation (space_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entityentityrelation_entity_a_id ON entityentityrelation (entity_a_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entityentityrelation_entity_b_id ON entityentityrelation (entity_b_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_entityentityrelation_relation_type ON entityentityrelation (relation_type)"
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_entityentityrelation_space_pair_type
        ON entityentityrelation (space_id, entity_a_id, entity_b_id, relation_type)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_entityentityrelation_space_pair_type")
    op.execute("DROP INDEX IF EXISTS ix_entityentityrelation_relation_type")
    op.execute("DROP INDEX IF EXISTS ix_entityentityrelation_entity_b_id")
    op.execute("DROP INDEX IF EXISTS ix_entityentityrelation_entity_a_id")
    op.execute("DROP INDEX IF EXISTS ix_entityentityrelation_space_id")
    op.execute("DROP TABLE IF EXISTS entityentityrelation")

    op.execute("DROP INDEX IF EXISTS uq_entityalias_space_alias")
    op.execute("DROP INDEX IF EXISTS ix_entityalias_alias_normalized")
    op.execute("DROP INDEX IF EXISTS ix_entityalias_entity_id")
    op.execute("DROP INDEX IF EXISTS ix_entityalias_space_id")
    op.execute("DROP TABLE IF EXISTS entityalias")
