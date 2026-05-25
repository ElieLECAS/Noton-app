"""add hnsw and remove legacy

Revision ID: add_hnsw_and_remove_legacy
Revises: add_tsvector_to_document_chunk
Create Date: 2026-05-21
"""

from alembic import op
import sqlalchemy as sa

revision = "add_hnsw_and_remove_legacy"
down_revision = "add_tsvector_to_document_chunk"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Créer l'index HNSW sur documentchunk.embedding
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_documentchunk_embedding_hnsw 
        ON documentchunk USING hnsw (embedding vector_cosine_ops);
        """
    )

    # 2. Supprimer les anciennes tables avec contraintes (de la plus dépendante à la moins dépendante)
    op.execute("DROP TABLE IF EXISTS notechunk CASCADE;")
    op.execute("DROP TABLE IF EXISTS note CASCADE;")
    op.execute("DROP TABLE IF EXISTS project CASCADE;")


def downgrade() -> None:
    # On supprime au moins l'index en cas de retour arrière
    op.execute("DROP INDEX IF EXISTS ix_documentchunk_embedding_hnsw;")
