"""add KAG knowledge entity and relation tables

Revision ID: add_kag_tables
Revises: add_notechunk_hierarchy
Create Date: 2026-02-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector
from app.embedding_config import EMBEDDING_DIMENSION


revision = "add_kag_tables"
down_revision = "update_embedding_dim_1024_bge_m3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    
    result = conn.execute(
        sa.text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name IN ('knowledgeentity', 'chunkentityrelation')
            """
        )
    )
    existing_tables = {row[0] for row in result}
    
    if "knowledgeentity" not in existing_tables:
        op.create_table(
            "knowledgeentity",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(length=500), nullable=False),
            sa.Column("name_normalized", sa.String(length=500), nullable=False),
            sa.Column("entity_type", sa.String(length=100), nullable=False),
            sa.Column("project_id", sa.Integer(), sa.ForeignKey("project.id", ondelete="CASCADE"), nullable=False),
            sa.Column("mention_count", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_knowledgeentity_name", "knowledgeentity", ["name"])
        op.create_index("ix_knowledgeentity_name_normalized", "knowledgeentity", ["name_normalized"])
        op.create_index("ix_knowledgeentity_entity_type", "knowledgeentity", ["entity_type"])
        op.create_index("ix_knowledgeentity_project_id", "knowledgeentity", ["project_id"])
        op.create_index(
            "ix_knowledgeentity_project_name_normalized",
            "knowledgeentity",
            ["project_id", "name_normalized"],
            unique=True,
        )
    
    if "chunkentityrelation" not in existing_tables:
        op.create_table(
            "chunkentityrelation",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("chunk_id", sa.Integer(), sa.ForeignKey("notechunk.id", ondelete="CASCADE"), nullable=False),
            sa.Column("entity_id", sa.Integer(), sa.ForeignKey("knowledgeentity.id", ondelete="CASCADE"), nullable=False),
            sa.Column("relevance_score", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("project_id", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_chunkentityrelation_chunk_id", "chunkentityrelation", ["chunk_id"])
        op.create_index("ix_chunkentityrelation_entity_id", "chunkentityrelation", ["entity_id"])
        op.create_index("ix_chunkentityrelation_project_id", "chunkentityrelation", ["project_id"])
        op.create_index(
            "ix_chunkentityrelation_chunk_entity",
            "chunkentityrelation",
            ["chunk_id", "entity_id"],
            unique=True,
        )


def downgrade() -> None:
    op.drop_index("ix_chunkentityrelation_chunk_entity", table_name="chunkentityrelation")
    op.drop_index("ix_chunkentityrelation_project_id", table_name="chunkentityrelation")
    op.drop_index("ix_chunkentityrelation_entity_id", table_name="chunkentityrelation")
    op.drop_index("ix_chunkentityrelation_chunk_id", table_name="chunkentityrelation")
    op.drop_table("chunkentityrelation")
    
    op.drop_index("ix_knowledgeentity_project_name_normalized", table_name="knowledgeentity")
    op.drop_index("ix_knowledgeentity_project_id", table_name="knowledgeentity")
    op.drop_index("ix_knowledgeentity_entity_type", table_name="knowledgeentity")
    op.drop_index("ix_knowledgeentity_name_normalized", table_name="knowledgeentity")
    op.drop_index("ix_knowledgeentity_name", table_name="knowledgeentity")
    op.drop_table("knowledgeentity")
