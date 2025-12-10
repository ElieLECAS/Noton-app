"""add conversations and messages tables

Revision ID: add_conversations_messages
Revises: update_embedding_dim_768_nomic
Create Date: 2025-12-10 12:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_conversations_messages'
down_revision = 'update_embedding_dim_768_nomic'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Créer la table conversation
    op.create_table(
        'conversation',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('project_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['project_id'], ['project.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_conversation_user_id'), 'conversation', ['user_id'], unique=False)
    op.create_index(op.f('ix_conversation_project_id'), 'conversation', ['project_id'], unique=False)
    op.create_index(op.f('ix_conversation_updated_at'), 'conversation', ['updated_at'], unique=False)
    
    # Créer la table message
    op.create_table(
        'message',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=True),
        sa.Column('provider', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversation.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_message_conversation_id'), 'message', ['conversation_id'], unique=False)
    op.create_index(op.f('ix_message_created_at'), 'message', ['created_at'], unique=False)


def downgrade() -> None:
    # Supprimer les tables dans l'ordre inverse
    op.drop_index(op.f('ix_message_created_at'), table_name='message')
    op.drop_index(op.f('ix_message_conversation_id'), table_name='message')
    op.drop_table('message')
    
    op.drop_index(op.f('ix_conversation_updated_at'), table_name='conversation')
    op.drop_index(op.f('ix_conversation_project_id'), table_name='conversation')
    op.drop_index(op.f('ix_conversation_user_id'), table_name='conversation')
    op.drop_table('conversation')

