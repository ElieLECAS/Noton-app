"""clean orphan messages

Revision ID: clean_orphan_messages
Revises: add_conversations_messages
Create Date: 2025-12-10 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'clean_orphan_messages'
down_revision = 'add_conversations_messages'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Supprimer tous les messages qui n'ont pas de conversation_id valide
    # (messages orphelins créés avant l'ajout de la fonctionnalité de conversations)
    op.execute("""
        DELETE FROM message 
        WHERE conversation_id IS NULL 
        OR conversation_id NOT IN (SELECT id FROM conversation)
    """)


def downgrade() -> None:
    # Pas de rollback possible pour la suppression de données
    pass

