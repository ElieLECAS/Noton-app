"""update embedding dimension to 384 for FastEmbed

Revision ID: update_embedding_dim_1536
Revises: add_note_source_file_path
Create Date: 2024-01-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'update_embedding_dim_1536'
down_revision: Union[str, None] = 'add_note_source_file_path'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Changer la dimension des embeddings à 384 pour correspondre à FastEmbed BAAI/bge-small-en-v1.5.
    Note: Les embeddings existants seront supprimés car on ne peut pas les convertir.
    Ils seront régénérés automatiquement avec le nouveau modèle FastEmbed.
    """
    # Supprimer les anciens embeddings de la table note
    op.execute("UPDATE note SET embedding = NULL WHERE embedding IS NOT NULL")
    
    # Supprimer les anciens embeddings de la table notechunk si elle existe
    op.execute("UPDATE notechunk SET embedding = NULL WHERE embedding IS NOT NULL")
    
    # Supprimer les anciennes colonnes vector (peu importe la dimension)
    op.execute("ALTER TABLE note DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE notechunk DROP COLUMN IF EXISTS embedding")
    
    # Recréer avec vector(384) pour FastEmbed
    op.execute("ALTER TABLE note ADD COLUMN embedding vector(384)")
    op.execute("ALTER TABLE notechunk ADD COLUMN embedding vector(384)")
    
    # Recréer les index HNSW avec la nouvelle dimension
    op.execute('DROP INDEX IF EXISTS note_embedding_idx')
    op.execute('CREATE INDEX IF NOT EXISTS note_embedding_idx ON note USING hnsw (embedding vector_cosine_ops)')
    
    op.execute('DROP INDEX IF EXISTS notechunk_embedding_idx')
    op.execute('CREATE INDEX IF NOT EXISTS notechunk_embedding_idx ON notechunk USING hnsw (embedding vector_cosine_ops)')


def downgrade() -> None:
    """
    Revenir à 1536 dimensions (pour OpenAI).
    """
    # Supprimer les embeddings existants
    op.execute("UPDATE note SET embedding = NULL WHERE embedding IS NOT NULL")
    op.execute("UPDATE notechunk SET embedding = NULL WHERE embedding IS NOT NULL")
    
    # Supprimer les colonnes vector(384)
    op.execute("ALTER TABLE note DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE notechunk DROP COLUMN IF EXISTS embedding")
    
    # Recréer avec vector(1536)
    op.execute("ALTER TABLE note ADD COLUMN embedding vector(1536)")
    op.execute("ALTER TABLE notechunk ADD COLUMN embedding vector(1536)")
    
    # Recréer les index HNSW avec la dimension 1536
    op.execute('DROP INDEX IF EXISTS note_embedding_idx')
    op.execute('CREATE INDEX IF NOT EXISTS note_embedding_idx ON note USING hnsw (embedding vector_cosine_ops)')
    
    op.execute('DROP INDEX IF EXISTS notechunk_embedding_idx')
    op.execute('CREATE INDEX IF NOT EXISTS notechunk_embedding_idx ON notechunk USING hnsw (embedding vector_cosine_ops)')

