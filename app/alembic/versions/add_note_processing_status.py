"""add note processing_status

Revision ID: add_note_processing_status
Revises: update_embedding_dim_1536
Create Date: 2024-01-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_note_processing_status'
down_revision: Union[str, None] = 'update_embedding_dim_1536'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ajouter la colonne processing_status à la table note avec valeur par défaut 'completed'
    op.add_column('note', sa.Column('processing_status', sa.String(), nullable=False, server_default='completed'))


def downgrade() -> None:
    # Supprimer la colonne processing_status
    op.drop_column('note', 'processing_status')

