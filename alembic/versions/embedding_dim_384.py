"""change embedding dimension to 384

Revision ID: embedding_dim_384
Revises: add_note_chunk_table
Create Date: 2024-01-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'embedding_dim_384'
down_revision: Union[str, None] = 'add_note_chunk_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Changer la dimension des embeddings de 768 à 384.
    Note: Les embeddings existants seront supprimés car on ne peut pas les convertir.
    Ils seront régénérés automatiquement avec le nouveau modèle plus léger.
    """
    # Supprimer les anciens embeddings de la table note (on les régénérera avec le nouveau modèle)
    op.execute("UPDATE note SET embedding = NULL WHERE embedding IS NOT NULL")
    
    # Supprimer les anciens embeddings de la table notechunk
    op.execute("UPDATE notechunk SET embedding = NULL WHERE embedding IS NOT NULL")
    
    # Supprimer les anciennes colonnes vector(768)
    op.execute("ALTER TABLE note DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE notechunk DROP COLUMN IF EXISTS embedding")
    
    # Recréer avec vector(384)
    op.execute("ALTER TABLE note ADD COLUMN embedding vector(384)")
    op.execute("ALTER TABLE notechunk ADD COLUMN embedding vector(384)")
    
    # Recréer les index HNSW avec la nouvelle dimension
    op.execute('DROP INDEX IF EXISTS note_embedding_idx')
    op.execute('CREATE INDEX IF NOT EXISTS note_embedding_idx ON note USING hnsw (embedding vector_cosine_ops)')
    
    op.execute('DROP INDEX IF EXISTS notechunk_embedding_idx')
    op.execute('CREATE INDEX IF NOT EXISTS notechunk_embedding_idx ON notechunk USING hnsw (embedding vector_cosine_ops)')


def downgrade() -> None:
    """
    Revenir à 768 dimensions (nécessite de régénérer les embeddings avec le modèle all-mpnet-base-v2).
    """
    # Supprimer les embeddings existants
    op.execute("UPDATE note SET embedding = NULL WHERE embedding IS NOT NULL")
    op.execute("UPDATE notechunk SET embedding = NULL WHERE embedding IS NOT NULL")
    
    # Supprimer les colonnes vector(384)
    op.execute("ALTER TABLE note DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE notechunk DROP COLUMN IF EXISTS embedding")
    
    # Recréer avec vector(768)
    op.execute("ALTER TABLE note ADD COLUMN embedding vector(768)")
    op.execute("ALTER TABLE notechunk ADD COLUMN embedding vector(768)")
    
    # Recréer les index HNSW avec la dimension 768
    op.execute('DROP INDEX IF EXISTS note_embedding_idx')
    op.execute('CREATE INDEX IF NOT EXISTS note_embedding_idx ON note USING hnsw (embedding vector_cosine_ops)')
    
    op.execute('DROP INDEX IF EXISTS notechunk_embedding_idx')
    op.execute('CREATE INDEX IF NOT EXISTS notechunk_embedding_idx ON notechunk USING hnsw (embedding vector_cosine_ops)')

