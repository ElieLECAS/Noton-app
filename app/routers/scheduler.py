from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete
from sqlmodel import Session, select
from app.models.user import UserRead
from app.models.agent import Agent
from app.models.scheduled_job import ScheduledJob, ScheduledJobCreate, ScheduledJobRead, ScheduledJobUpdate, cron_from_schedule
from app.models.task_run_log import TaskRunLog, TaskRunLogRead
from app.routers.auth import get_current_user
from app.database import get_session
from datetime import datetime
import logging

from app.services.scheduler_service import run_scheduled_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.post("/jobs/{job_id}/run")
async def run_job_now(
    job_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Lancer immédiatement l'exécution d'un job planifié."""
    job = session.get(ScheduledJob, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    await run_scheduled_job(job_id)
    return {"ok": True, "message": "Job lancé"}


@router.get("/jobs", response_model=List[ScheduledJobRead])
async def list_scheduled_jobs(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Lister les jobs planifiés de l'utilisateur"""
    statement = select(ScheduledJob, Agent.name).join(
        Agent, ScheduledJob.agent_id == Agent.id
    ).where(
        ScheduledJob.user_id == current_user.id
    ).order_by(ScheduledJob.created_at.desc())
    
    results = session.exec(statement).all()
    
    jobs = []
    for job, agent_name in results:
        job_read = ScheduledJobRead.from_orm(job)
        job_read.agent_name = agent_name
        jobs.append(job_read)
    
    return jobs


@router.post("/jobs", response_model=ScheduledJobRead)
async def create_scheduled_job(
    job: ScheduledJobCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Créer un nouveau job planifié"""
    # Vérifier que l'agent appartient à l'utilisateur
    agent = session.get(Agent, job.agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    # Valider qu'au moins un jour est sélectionné
    if not job.schedule_days:
        raise HTTPException(status_code=400, detail="Au moins un jour doit être sélectionné")
    
    # Calculer l'expression cron depuis les champs schedule_*
    cron_expression = cron_from_schedule(job.schedule_hour, job.schedule_minute, job.schedule_days)
    
    db_job = ScheduledJob(
        user_id=current_user.id,
        agent_id=job.agent_id,
        task_ids=job.task_ids,
        cron_expression=cron_expression,
        schedule_hour=job.schedule_hour,
        schedule_minute=job.schedule_minute,
        schedule_days=job.schedule_days,
        enabled=job.enabled
    )
    session.add(db_job)
    session.commit()
    session.refresh(db_job)
    
    job_read = ScheduledJobRead.from_orm(db_job)
    job_read.agent_name = agent.name
    
    logger.info(f"Job planifié créé: {db_job.id} pour l'utilisateur {current_user.id}")
    return job_read


@router.get("/jobs/{job_id}", response_model=ScheduledJobRead)
async def get_scheduled_job(
    job_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupérer un job planifié spécifique"""
    job = session.get(ScheduledJob, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    # Récupérer le nom de l'agent
    agent = session.get(Agent, job.agent_id)
    
    job_read = ScheduledJobRead.from_orm(job)
    job_read.agent_name = agent.name if agent else None
    
    return job_read


@router.put("/jobs/{job_id}", response_model=ScheduledJobRead)
async def update_scheduled_job(
    job_id: int,
    job_update: ScheduledJobUpdate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Mettre à jour un job planifié"""
    job = session.get(ScheduledJob, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    # Si l'agent_id change, vérifier que le nouvel agent appartient à l'utilisateur
    if job_update.agent_id is not None:
        agent = session.get(Agent, job_update.agent_id)
        if not agent or agent.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Agent non trouvé")
        job.agent_id = job_update.agent_id
    
    if job_update.task_ids is not None:
        job.task_ids = job_update.task_ids
    
    # Si les champs schedule_* changent, recalculer le cron
    schedule_changed = False
    if job_update.schedule_hour is not None:
        job.schedule_hour = job_update.schedule_hour
        schedule_changed = True
    if job_update.schedule_minute is not None:
        job.schedule_minute = job_update.schedule_minute
        schedule_changed = True
    if job_update.schedule_days is not None:
        if not job_update.schedule_days:
            raise HTTPException(status_code=400, detail="Au moins un jour doit être sélectionné")
        job.schedule_days = job_update.schedule_days
        schedule_changed = True
    
    if schedule_changed:
        job.cron_expression = cron_from_schedule(job.schedule_hour, job.schedule_minute, job.schedule_days)
    
    if job_update.enabled is not None:
        job.enabled = job_update.enabled
    
    job.updated_at = datetime.utcnow()
    
    session.add(job)
    session.commit()
    session.refresh(job)
    
    # Récupérer le nom de l'agent
    agent = session.get(Agent, job.agent_id)
    
    job_read = ScheduledJobRead.from_orm(job)
    job_read.agent_name = agent.name if agent else None
    
    return job_read


@router.delete("/jobs/{job_id}")
async def delete_scheduled_job(
    job_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Supprimer un job planifié (et ses logs d'exécution)"""
    job = session.get(ScheduledJob, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    # Supprimer d'abord les logs d'exécution liés au job (contrainte FK)
    session.execute(delete(TaskRunLog).where(TaskRunLog.scheduled_job_id == job_id))
    session.delete(job)
    session.commit()
    
    logger.info(f"Job planifié supprimé: {job_id}")
    return {"ok": True, "message": "Job planifié supprimé"}


@router.get("/jobs/{job_id}/runs", response_model=List[TaskRunLogRead])
async def get_job_runs(
    job_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupérer l'historique des exécutions d'un job"""
    # Vérifier que le job appartient à l'utilisateur
    job = session.get(ScheduledJob, job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Job non trouvé")
    
    # Récupérer les runs triés par date décroissante (les plus récents en premier)
    statement = select(TaskRunLog).where(
        TaskRunLog.scheduled_job_id == job_id
    ).order_by(TaskRunLog.run_at.desc()).limit(100)  # Limiter à 100 derniers runs
    
    runs = session.exec(statement).all()
    return runs
