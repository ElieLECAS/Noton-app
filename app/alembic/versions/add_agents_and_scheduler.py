"""add agents and scheduler tables

Revision ID: add_agents_scheduler
Revises: clean_orphan_messages
Create Date: 2026-02-02 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_agents_scheduler'
down_revision = 'clean_orphan_messages'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Créer la table agent
    op.create_table(
        'agent',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('personality', sa.Text(), nullable=False),
        sa.Column('model_preset', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agent_user_id'), 'agent', ['user_id'], unique=False)
    
    # Créer la table agent_task
    op.create_table(
        'agenttask',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('agent_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('instruction', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['agent_id'], ['agent.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agenttask_agent_id'), 'agenttask', ['agent_id'], unique=False)
    
    # Créer la table scheduled_job
    op.create_table(
        'scheduledjob',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('agent_id', sa.Integer(), nullable=False),
        sa.Column('task_ids', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('cron_expression', sa.String(length=100), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('last_run_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['agent_id'], ['agent.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_scheduledjob_user_id'), 'scheduledjob', ['user_id'], unique=False)
    op.create_index(op.f('ix_scheduledjob_agent_id'), 'scheduledjob', ['agent_id'], unique=False)
    op.create_index(op.f('ix_scheduledjob_enabled'), 'scheduledjob', ['enabled'], unique=False)


def downgrade() -> None:
    # Supprimer les tables dans l'ordre inverse des dépendances
    op.drop_index(op.f('ix_scheduledjob_enabled'), table_name='scheduledjob')
    op.drop_index(op.f('ix_scheduledjob_agent_id'), table_name='scheduledjob')
    op.drop_index(op.f('ix_scheduledjob_user_id'), table_name='scheduledjob')
    op.drop_table('scheduledjob')
    
    op.drop_index(op.f('ix_agenttask_agent_id'), table_name='agenttask')
    op.drop_table('agenttask')
    
    op.drop_index(op.f('ix_agent_user_id'), table_name='agent')
    op.drop_table('agent')
