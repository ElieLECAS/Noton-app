"""add task_run_log and schedule fields

Revision ID: add_task_run_log_schedule
Revises: add_agents_scheduler
Create Date: 2026-02-02 19:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_task_run_log_schedule'
down_revision = 'add_agents_scheduler'
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Créer la table task_run_log seulement si elle n'existe pas
    result = conn.execute(sa.text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'taskrunlog')"
    ))
    table_exists = result.scalar()
    if not table_exists:
        op.create_table(
            'taskrunlog',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('scheduled_job_id', sa.Integer(), nullable=False),
            sa.Column('agent_task_id', sa.Integer(), nullable=False),
            sa.Column('task_name', sa.String(length=200), nullable=False),
            sa.Column('output', sa.Text(), nullable=True),
            sa.Column('error', sa.Text(), nullable=True),
            sa.Column('run_at', sa.DateTime(), nullable=False),
            sa.Column('user_id', sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(['scheduled_job_id'], ['scheduledjob.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['agent_task_id'], ['agenttask.id'], ondelete='CASCADE'),
            sa.ForeignKeyConstraint(['user_id'], ['user.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_taskrunlog_scheduled_job_id'), 'taskrunlog', ['scheduled_job_id'], unique=False)
        op.create_index(op.f('ix_taskrunlog_run_at'), 'taskrunlog', ['run_at'], unique=False)

    # Ajouter les colonnes schedule_* à scheduledjob si elles n'existent pas
    cols_result = conn.execute(sa.text(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'scheduledjob'"
    ))
    existing_cols = {row[0] for row in cols_result}
    if 'schedule_hour' not in existing_cols:
        op.add_column('scheduledjob', sa.Column('schedule_hour', sa.Integer(), nullable=True))
    if 'schedule_minute' not in existing_cols:
        op.add_column('scheduledjob', sa.Column('schedule_minute', sa.Integer(), nullable=True))
    if 'schedule_days' not in existing_cols:
        op.add_column('scheduledjob', sa.Column('schedule_days', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Backfill : parser les cron_expression existantes pour remplir schedule_*
    # Format attendu : "minute hour * * day_of_week"
    result = conn.execute(sa.text("SELECT id, cron_expression FROM scheduledjob"))
    
    for row in result:
        job_id, cron_expr = row
        try:
            # Parser le cron (format: "minute hour * * day_of_week")
            parts = cron_expr.split()
            if len(parts) >= 2:
                minute = int(parts[0])
                hour = int(parts[1])
                # Jours : si parts[4] existe et contient des chiffres/virgules
                days = [0,1,2,3,4,5,6]  # Par défaut tous les jours
                if len(parts) >= 5 and parts[4] != '*':
                    # Parser day_of_week : peut être "0-6", "0,1,2", "0-4", etc.
                    day_str = parts[4]
                    if ',' in day_str:
                        days = [int(d) for d in day_str.split(',')]
                    elif '-' in day_str:
                        start, end = day_str.split('-')
                        days = list(range(int(start), int(end) + 1))
                    else:
                        days = [int(day_str)]
                
                # Mettre à jour
                conn.execute(
                    sa.text("UPDATE scheduledjob SET schedule_hour = :hour, schedule_minute = :minute, schedule_days = :days WHERE id = :id"),
                    {"hour": hour, "minute": minute, "days": str(days).replace("'", '"'), "id": job_id}
                )
            else:
                # Cron invalide, utiliser valeurs par défaut
                conn.execute(
                    sa.text("UPDATE scheduledjob SET schedule_hour = 0, schedule_minute = 0, schedule_days = '[0,1,2,3,4,5,6]' WHERE id = :id"),
                    {"id": job_id}
                )
        except Exception as e:
            # En cas d'erreur de parsing, utiliser valeurs par défaut
            print(f"Erreur parsing cron pour job {job_id}: {e}, utilisation valeurs par défaut")
            conn.execute(
                sa.text("UPDATE scheduledjob SET schedule_hour = 0, schedule_minute = 0, schedule_days = '[0,1,2,3,4,5,6]' WHERE id = :id"),
                {"id": job_id}
            )
    
    # Rendre les colonnes non-nullables si elles le sont encore
    for col in ('schedule_hour', 'schedule_minute', 'schedule_days'):
        r = conn.execute(sa.text(
            "SELECT is_nullable FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'scheduledjob' AND column_name = :name"
        ), {"name": col})
        row = r.fetchone()
        if row and row[0] == 'YES':
            op.alter_column('scheduledjob', col, nullable=False)


def downgrade() -> None:
    # Supprimer les colonnes schedule_* de scheduledjob
    op.drop_column('scheduledjob', 'schedule_days')
    op.drop_column('scheduledjob', 'schedule_minute')
    op.drop_column('scheduledjob', 'schedule_hour')
    
    # Supprimer la table taskrunlog
    op.drop_index(op.f('ix_taskrunlog_run_at'), table_name='taskrunlog')
    op.drop_index(op.f('ix_taskrunlog_scheduled_job_id'), table_name='taskrunlog')
    op.drop_table('taskrunlog')
