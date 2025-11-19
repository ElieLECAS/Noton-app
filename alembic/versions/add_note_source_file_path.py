"""add note source_file_path

Revision ID: add_note_source_file_path
Revises: remove_note_chunk_table
Create Date: 2024-01-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_note_source_file_path'
down_revision: Union[str, None] = 'remove_note_chunk_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ajouter la colonne source_file_path à la table note
    op.add_column('note', sa.Column('source_file_path', sa.String(), nullable=True))


def downgrade() -> None:
    # Supprimer la colonne source_file_path
    op.drop_column('note', 'source_file_path')

