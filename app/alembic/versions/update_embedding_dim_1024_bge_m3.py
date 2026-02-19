"""update embedding dimension to 1024 for BGE-M3

Revision ID: update_embedding_dim_1024_bge_m3
Revises: add_note_processing_progress
Create Date: 2026-02-19 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "update_embedding_dim_1024_bge_m3"
down_revision: Union[str, None] = "add_note_processing_progress"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Changer la dimension des embeddings de 768 à 1024 pour BGE-M3.
    Les embeddings existants sont remis à NULL (non convertibles).
    """
    connection = op.get_bind()
    result = connection.execute(
        text(
            """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'notechunk'
        )
        """
        )
    )
    notechunk_exists = result.scalar()

    op.execute("UPDATE note SET embedding = NULL WHERE embedding IS NOT NULL")
    if notechunk_exists:
        op.execute("UPDATE notechunk SET embedding = NULL WHERE embedding IS NOT NULL")

    op.execute("ALTER TABLE note DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE note ADD COLUMN embedding vector(1024)")

    if notechunk_exists:
        op.execute("ALTER TABLE notechunk DROP COLUMN IF EXISTS embedding")
        op.execute("ALTER TABLE notechunk ADD COLUMN embedding vector(1024)")

    op.execute("DROP INDEX IF EXISTS note_embedding_idx")
    op.execute(
        "CREATE INDEX IF NOT EXISTS note_embedding_idx ON note USING hnsw (embedding vector_cosine_ops)"
    )

    if notechunk_exists:
        op.execute("DROP INDEX IF EXISTS notechunk_embedding_idx")
        op.execute(
            "CREATE INDEX IF NOT EXISTS notechunk_embedding_idx ON notechunk USING hnsw (embedding vector_cosine_ops)"
        )


def downgrade() -> None:
    """
    Revenir à 768 dimensions.
    Les embeddings existants sont remis à NULL (non convertibles).
    """
    connection = op.get_bind()
    result = connection.execute(
        text(
            """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = 'notechunk'
        )
        """
        )
    )
    notechunk_exists = result.scalar()

    op.execute("UPDATE note SET embedding = NULL WHERE embedding IS NOT NULL")
    if notechunk_exists:
        op.execute("UPDATE notechunk SET embedding = NULL WHERE embedding IS NOT NULL")

    op.execute("ALTER TABLE note DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE note ADD COLUMN embedding vector(768)")

    if notechunk_exists:
        op.execute("ALTER TABLE notechunk DROP COLUMN IF EXISTS embedding")
        op.execute("ALTER TABLE notechunk ADD COLUMN embedding vector(768)")

    op.execute("DROP INDEX IF EXISTS note_embedding_idx")
    op.execute(
        "CREATE INDEX IF NOT EXISTS note_embedding_idx ON note USING hnsw (embedding vector_cosine_ops)"
    )

    if notechunk_exists:
        op.execute("DROP INDEX IF EXISTS notechunk_embedding_idx")
        op.execute(
            "CREATE INDEX IF NOT EXISTS notechunk_embedding_idx ON notechunk USING hnsw (embedding vector_cosine_ops)"
        )
