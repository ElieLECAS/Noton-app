"""add library architecture with spaces and documents

Revision ID: add_library_architecture
Revises: add_kag_tables
Create Date: 2026-03-23 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector
from app.embedding_config import EMBEDDING_DIMENSION


revision = "add_library_architecture"
down_revision = "add_kag_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    
    result = conn.execute(
        sa.text(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name IN ('library', 'folder', 'space', 'document', 'document_space', 'documentchunk')
            """
        )
    )
    existing_tables = {row[0] for row in result}
    
    if "library" not in existing_tables:
        op.create_table(
            "library",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(length=200), nullable=False, server_default="Ma Bibliothèque"),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_library_user_id", "library", ["user_id"])
    
    if "folder" not in existing_tables:
        op.create_table(
            "folder",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("parent_folder_id", sa.Integer(), sa.ForeignKey("folder.id", ondelete="CASCADE"), nullable=True),
            sa.Column("library_id", sa.Integer(), sa.ForeignKey("library.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_folder_parent_folder_id", "folder", ["parent_folder_id"])
        op.create_index("ix_folder_library_id", "folder", ["library_id"])
        op.create_index("ix_folder_user_id", "folder", ["user_id"])
    
    if "space" not in existing_tables:
        op.create_table(
            "space",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("color", sa.String(length=20), nullable=True),
            sa.Column("icon", sa.String(length=50), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_space_user_id", "space", ["user_id"])

    # Backfill spaces from legacy projects to keep FK integrity when
    # project_id -> space_id columns are migrated.
    conn.execute(
        sa.text(
            """
            INSERT INTO space (id, name, description, user_id, color, icon, created_at, updated_at)
            SELECT p.id, p.title, p.description, p.user_id, NULL, NULL, p.created_at, p.updated_at
            FROM project p
            WHERE NOT EXISTS (
                SELECT 1 FROM space s WHERE s.id = p.id
            )
            """
        )
    )
    
    if "document" not in existing_tables:
        op.create_table(
            "document",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("title", sa.String(length=200), nullable=False),
            sa.Column("content", sa.Text(), nullable=True),
            sa.Column("document_type", sa.String(length=50), nullable=False, server_default="written"),
            sa.Column("source_file_path", sa.String(length=500), nullable=True),
            sa.Column("processing_status", sa.String(length=50), nullable=False, server_default="completed"),
            sa.Column("processing_progress", sa.Integer(), nullable=True, server_default="100"),
            sa.Column("folder_id", sa.Integer(), sa.ForeignKey("folder.id", ondelete="SET NULL"), nullable=True),
            sa.Column("library_id", sa.Integer(), sa.ForeignKey("library.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
            sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True),
        )
        op.create_index("ix_document_folder_id", "document", ["folder_id"])
        op.create_index("ix_document_library_id", "document", ["library_id"])
        op.create_index("ix_document_user_id", "document", ["user_id"])

    # Ensure at least one library exists for users owning legacy notes.
    conn.execute(
        sa.text(
            """
            INSERT INTO library (name, user_id, created_at, updated_at)
            SELECT 'Bibliothèque migrée', n.user_id, NOW(), NOW()
            FROM note n
            GROUP BY n.user_id
            HAVING NOT EXISTS (
                SELECT 1 FROM library l WHERE l.user_id = n.user_id
            )
            """
        )
    )

    # Backfill documents from legacy notes with same IDs to preserve
    # notechunk.note_id -> documentchunk.document_id references.
    conn.execute(
        sa.text(
            """
            INSERT INTO document (
                id, title, content, document_type, source_file_path,
                processing_status, processing_progress, folder_id,
                library_id, user_id, created_at, updated_at, embedding
            )
            SELECT
                n.id,
                n.title,
                n.content,
                n.note_type,
                n.source_file_path,
                n.processing_status,
                n.processing_progress,
                NULL,
                l.id,
                n.user_id,
                n.created_at,
                n.updated_at,
                n.embedding
            FROM note n
            JOIN library l ON l.user_id = n.user_id
            WHERE NOT EXISTS (
                SELECT 1 FROM document d WHERE d.id = n.id
            )
            """
        )
    )
    
    if "document_space" not in existing_tables:
        op.create_table(
            "document_space",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("document_id", sa.Integer(), sa.ForeignKey("document.id", ondelete="CASCADE"), nullable=False),
            sa.Column("space_id", sa.Integer(), sa.ForeignKey("space.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
            sa.Column("added_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        )
        op.create_index("ix_document_space_document_id", "document_space", ["document_id"])
        op.create_index("ix_document_space_space_id", "document_space", ["space_id"])
        op.create_index("ix_document_space_user_id", "document_space", ["user_id"])
        op.create_unique_constraint("uq_document_space_document_id_space_id", "document_space", ["document_id", "space_id"])
    
    if "documentchunk" not in existing_tables:
        op.create_table(
            "documentchunk",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("document_id", sa.Integer(), sa.ForeignKey("document.id", ondelete="CASCADE"), nullable=False),
            sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("text", sa.Text(), nullable=True),
            sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True),
            sa.Column("start_char", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("end_char", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("node_id", sa.String(length=200), nullable=True),
            sa.Column("parent_node_id", sa.String(length=200), nullable=True),
            sa.Column("is_leaf", sa.Boolean(), nullable=False, server_default="true"),
            sa.Column("hierarchy_level", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("metadata_json", postgresql.JSONB(), nullable=True),
            sa.Column("metadata_", postgresql.JSONB(), nullable=True),
        )
        op.create_index("ix_documentchunk_document_id", "documentchunk", ["document_id"])
        op.create_index("ix_documentchunk_node_id", "documentchunk", ["node_id"])
        op.create_index("ix_documentchunk_parent_node_id", "documentchunk", ["parent_node_id"])
        op.create_index("ix_documentchunk_is_leaf", "documentchunk", ["is_leaf"])
        op.create_index("ix_documentchunk_hierarchy_level", "documentchunk", ["hierarchy_level"])

    # Backfill document chunks from legacy notechunk to preserve chunk IDs used
    # by chunkentityrelation before re-pointing FK.
    conn.execute(
        sa.text(
            """
            INSERT INTO documentchunk (
                id, document_id, chunk_index, content, text, embedding,
                start_char, end_char, node_id, parent_node_id, is_leaf,
                hierarchy_level, metadata_json, metadata_
            )
            SELECT
                nc.id, nc.note_id, nc.chunk_index, nc.content, nc.text, nc.embedding,
                nc.start_char, nc.end_char, nc.node_id, nc.parent_node_id, nc.is_leaf,
                nc.hierarchy_level, nc.metadata_json, nc.metadata_
            FROM notechunk nc
            WHERE NOT EXISTS (
                SELECT 1 FROM documentchunk dc WHERE dc.id = nc.id
            )
            """
        )
    )
    
    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'knowledgeentity' AND column_name = 'space_id'
            """
        )
    )
    has_space_id = len(list(result)) > 0
    
    if not has_space_id:
        result = conn.execute(
            sa.text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'knowledgeentity' AND column_name = 'project_id'
                """
            )
        )
        has_project_id = len(list(result)) > 0
        
        if has_project_id:
            op.drop_constraint("knowledgeentity_project_id_fkey", "knowledgeentity", type_="foreignkey")
            op.drop_index("ix_knowledgeentity_project_id", "knowledgeentity")
            op.alter_column("knowledgeentity", "project_id", new_column_name="space_id")
            op.create_foreign_key("knowledgeentity_space_id_fkey", "knowledgeentity", "space", ["space_id"], ["id"], ondelete="CASCADE")
            op.create_index("ix_knowledgeentity_space_id", "knowledgeentity", ["space_id"])
    
    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'chunkentityrelation' AND column_name = 'space_id'
            """
        )
    )
    has_space_id = len(list(result)) > 0
    
    if not has_space_id:
        result = conn.execute(
            sa.text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'chunkentityrelation' AND column_name = 'project_id'
                """
            )
        )
        has_project_id = len(list(result)) > 0
        
        if has_project_id:
            op.drop_index("ix_chunkentityrelation_project_id", "chunkentityrelation")
            op.alter_column("chunkentityrelation", "project_id", new_column_name="space_id")
            op.create_index("ix_chunkentityrelation_space_id", "chunkentityrelation", ["space_id"])
        
        result = conn.execute(
            sa.text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'chunkentityrelation' AND column_name = 'chunk_id'
                """
            )
        )
        rows = list(result)
        if rows:
            conn.execute(sa.text("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.table_constraints
                        WHERE constraint_name = 'chunkentityrelation_chunk_id_fkey'
                        AND table_name = 'chunkentityrelation'
                    ) THEN
                        ALTER TABLE chunkentityrelation DROP CONSTRAINT chunkentityrelation_chunk_id_fkey;
                    END IF;
                END $$;
            """))
            
            op.create_foreign_key("chunkentityrelation_chunk_id_fkey", "chunkentityrelation", "documentchunk", ["chunk_id"], ["id"], ondelete="CASCADE")
    
    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'conversation' AND column_name = 'space_id'
            """
        )
    )
    has_space_id = len(list(result)) > 0
    
    if not has_space_id:
        result = conn.execute(
            sa.text(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'conversation' AND column_name = 'project_id'
                """
            )
        )
        has_project_id = len(list(result)) > 0
        
        if has_project_id:
            conn.execute(sa.text("DELETE FROM conversation WHERE project_id IS NULL"))
            
            op.drop_constraint("conversation_project_id_fkey", "conversation", type_="foreignkey")
            op.drop_index("ix_conversation_project_id", "conversation")
            op.alter_column("conversation", "project_id", new_column_name="space_id", nullable=False)
            op.create_foreign_key("conversation_space_id_fkey", "conversation", "space", ["space_id"], ["id"], ondelete="CASCADE")
            op.create_index("ix_conversation_space_id", "conversation", ["space_id"])


def downgrade() -> None:
    conn = op.get_bind()
    
    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'conversation' AND column_name = 'space_id'
            """
        )
    )
    has_space_id = len(list(result)) > 0
    
    if has_space_id:
        op.drop_constraint("conversation_space_id_fkey", "conversation", type_="foreignkey")
        op.drop_index("ix_conversation_space_id", "conversation")
        op.alter_column("conversation", "space_id", new_column_name="project_id", nullable=True)
        op.create_foreign_key("conversation_project_id_fkey", "conversation", "project", ["project_id"], ["id"], ondelete="CASCADE")
        op.create_index("ix_conversation_project_id", "conversation", ["project_id"])
    
    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'chunkentityrelation' AND column_name = 'space_id'
            """
        )
    )
    has_space_id = len(list(result)) > 0
    
    if has_space_id:
        op.drop_index("ix_chunkentityrelation_space_id", "chunkentityrelation")
        op.alter_column("chunkentityrelation", "space_id", new_column_name="project_id")
        op.create_index("ix_chunkentityrelation_project_id", "chunkentityrelation", ["project_id"])
    
    result = conn.execute(
        sa.text(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'knowledgeentity' AND column_name = 'space_id'
            """
        )
    )
    has_space_id = len(list(result)) > 0
    
    if has_space_id:
        op.drop_constraint("knowledgeentity_space_id_fkey", "knowledgeentity", type_="foreignkey")
        op.drop_index("ix_knowledgeentity_space_id", "knowledgeentity")
        op.alter_column("knowledgeentity", "space_id", new_column_name="project_id")
        op.create_foreign_key("knowledgeentity_project_id_fkey", "knowledgeentity", "project", ["project_id"], ["id"], ondelete="CASCADE")
        op.create_index("ix_knowledgeentity_project_id", "knowledgeentity", ["project_id"])
    
    op.drop_table("documentchunk")
    op.drop_table("document_space")
    op.drop_table("document")
    op.drop_table("space")
    op.drop_table("folder")
    op.drop_table("library")
