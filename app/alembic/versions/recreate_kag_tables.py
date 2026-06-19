"""Recreate KAG knowledge graph tables (entities, relations, aliases).

Revision ID: recreate_kag_tables
Revises: add_query_context_slots
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy import inspect

from app.embedding_config import EMBEDDING_DIMENSION


revision = "recreate_kag_tables"
down_revision = "add_query_context_slots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    existing = set(inspector.get_table_names(schema="public"))

    if "knowledgeentity" not in existing:
        op.create_table(
            "knowledgeentity",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("space_id", sa.Integer(), sa.ForeignKey("space.id", ondelete="CASCADE"), nullable=False),
            sa.Column("name", sa.String(length=500), nullable=False),
            sa.Column("name_normalized", sa.String(length=500), nullable=False),
            sa.Column("entity_type", sa.String(length=100), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("mention_count", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True),
            sa.Column("confidence_score", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_knowledgeentity_space_id", "knowledgeentity", ["space_id"])
        op.create_index("ix_knowledgeentity_name", "knowledgeentity", ["name"])
        op.create_index("ix_knowledgeentity_name_normalized", "knowledgeentity", ["name_normalized"])
        op.create_index("ix_knowledgeentity_entity_type", "knowledgeentity", ["entity_type"])
        op.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledgeentity_space_name_type
            ON knowledgeentity (space_id, name_normalized, entity_type)
            """
        )

    if "chunkentityrelation" not in existing:
        op.create_table(
            "chunkentityrelation",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "chunk_id",
                sa.Integer(),
                sa.ForeignKey("documentchunk.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "entity_id",
                sa.Integer(),
                sa.ForeignKey("knowledgeentity.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("space_id", sa.Integer(), sa.ForeignKey("space.id", ondelete="CASCADE"), nullable=False),
            sa.Column("relation_role", sa.String(length=32), nullable=False, server_default="mention"),
            sa.Column("relevance_score", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("context_snippet", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_chunkentityrelation_chunk_id", "chunkentityrelation", ["chunk_id"])
        op.create_index("ix_chunkentityrelation_entity_id", "chunkentityrelation", ["entity_id"])
        op.create_index("ix_chunkentityrelation_space_id", "chunkentityrelation", ["space_id"])
        op.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_chunkentityrelation_chunk_entity_role
            ON chunkentityrelation (chunk_id, entity_id, relation_role)
            """
        )

    if "entityalias" not in existing:
        op.create_table(
            "entityalias",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("space_id", sa.Integer(), sa.ForeignKey("space.id", ondelete="CASCADE"), nullable=False),
            sa.Column(
                "entity_id",
                sa.Integer(),
                sa.ForeignKey("knowledgeentity.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("alias_normalized", sa.String(length=500), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_entityalias_space_id", "entityalias", ["space_id"])
        op.create_index("ix_entityalias_entity_id", "entityalias", ["entity_id"])
        op.create_index("ix_entityalias_alias_normalized", "entityalias", ["alias_normalized"])
        op.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_entityalias_space_alias
            ON entityalias (space_id, alias_normalized)
            """
        )

    if "entityentityrelation" not in existing:
        op.create_table(
            "entityentityrelation",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("space_id", sa.Integer(), sa.ForeignKey("space.id", ondelete="CASCADE"), nullable=False),
            sa.Column(
                "entity_a_id",
                sa.Integer(),
                sa.ForeignKey("knowledgeentity.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "entity_b_id",
                sa.Integer(),
                sa.ForeignKey("knowledgeentity.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("relation_type", sa.String(length=64), nullable=False, server_default="co_occurs"),
            sa.Column("relation_label", sa.Text(), nullable=True),
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
        op.create_index("ix_entityentityrelation_space_id", "entityentityrelation", ["space_id"])
        op.create_index("ix_entityentityrelation_entity_a_id", "entityentityrelation", ["entity_a_id"])
        op.create_index("ix_entityentityrelation_entity_b_id", "entityentityrelation", ["entity_b_id"])
        op.create_index("ix_entityentityrelation_relation_type", "entityentityrelation", ["relation_type"])
        op.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_entityentityrelation_space_pair_type
            ON entityentityrelation (space_id, entity_a_id, entity_b_id, relation_type)
            """
        )

    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_knowledgeentity_name_normalized_trgm
        ON knowledgeentity USING gin (name_normalized gin_trgm_ops)
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'ck_chunkentityrelation_relevance_score_range'
            ) THEN
                ALTER TABLE chunkentityrelation
                ADD CONSTRAINT ck_chunkentityrelation_relevance_score_range
                CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0);
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_knowledgeentity_name_normalized_trgm")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'ck_chunkentityrelation_relevance_score_range'
            ) THEN
                ALTER TABLE chunkentityrelation
                DROP CONSTRAINT ck_chunkentityrelation_relevance_score_range;
            END IF;
        END
        $$;
        """
    )
    for table in (
        "chunkentityrelation",
        "entityalias",
        "entityentityrelation",
        "knowledgeentity",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
