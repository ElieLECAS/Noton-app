"""add note embedding

Revision ID: add_note_embedding
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_note_embedding'
down_revision: Union[str, None] = 'initial_schema'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Activer l'extension pgvector
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Ajouter la colonne embedding à la table note
    # Utiliser le type SQL directement pour Alembic
    op.execute("ALTER TABLE note ADD COLUMN embedding vector(768)")
    
    # Créer un index HNSW pour la recherche rapide
    op.execute('CREATE INDEX IF NOT EXISTS note_embedding_idx ON note USING hnsw (embedding vector_cosine_ops)')


def downgrade() -> None:
    # Supprimer l'index
    op.drop_index('note_embedding_idx', table_name='note')
    
    # Supprimer la colonne embedding
    op.drop_column('note', 'embedding')
    
    # Note: On ne supprime pas l'extension vector car elle pourrait être utilisée ailleurs

