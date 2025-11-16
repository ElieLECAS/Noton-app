"""add note chunk table

Revision ID: add_note_chunk_table
Revises: add_note_embedding
Create Date: 2024-01-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_note_chunk_table'
down_revision: Union[str, None] = 'add_note_embedding'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Créer la table note_chunk
    op.execute("""
        CREATE TABLE IF NOT EXISTS notechunk (
            id SERIAL PRIMARY KEY,
            note_id INTEGER NOT NULL REFERENCES note(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL,
            embedding vector(768),
            start_char INTEGER NOT NULL DEFAULT 0,
            end_char INTEGER NOT NULL DEFAULT 0
        )
    """)
    
    # Créer un index sur note_id pour les recherches rapides
    op.create_index('notechunk_note_id_idx', 'notechunk', ['note_id'])
    
    # Créer un index HNSW sur l'embedding pour la recherche sémantique
    op.execute('CREATE INDEX IF NOT EXISTS notechunk_embedding_idx ON notechunk USING hnsw (embedding vector_cosine_ops)')


def downgrade() -> None:
    # Supprimer les index
    op.drop_index('notechunk_embedding_idx', table_name='notechunk')
    op.drop_index('notechunk_note_id_idx', table_name='notechunk')
    
    # Supprimer la table
    op.drop_table('notechunk')

