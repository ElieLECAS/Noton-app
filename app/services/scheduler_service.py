"""
Service de planification des tâches d'agents avec APScheduler
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlmodel import Session, select
from app.models.scheduled_job import ScheduledJob
from app.models.task_run_log import TaskRunLog
from app.models.agent_task import AgentTask
from app.services.agent_runner import execute_agent_task
from app.database import engine
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Instance globale du scheduler
scheduler = None


def init_scheduler():
    """Initialiser le scheduler APScheduler"""
    global scheduler
    scheduler = AsyncIOScheduler()
    logger.info("Scheduler APScheduler initialisé")


def start_scheduler():
    """Démarrer le scheduler"""
    global scheduler
    if scheduler and not scheduler.running:
        scheduler.start()
        logger.info("✅ Scheduler APScheduler démarré")
        
        # Ajouter un job pour synchroniser les scheduled jobs toutes les minutes
        scheduler.add_job(
            sync_scheduled_jobs,
            trigger='interval',
            minutes=1,
            id='sync_scheduled_jobs',
            replace_existing=True
        )


def stop_scheduler():
    """Arrêter le scheduler"""
    global scheduler
    if scheduler and scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler APScheduler arrêté")


async def sync_scheduled_jobs():
    """
    Synchroniser les scheduled jobs depuis la base de données vers APScheduler
    Cette fonction est appelée périodiquement pour détecter les nouveaux jobs ou modifications
    """
    with Session(engine) as session:
        # Récupérer tous les jobs actifs
        statement = select(ScheduledJob).where(ScheduledJob.enabled == True)
        jobs = session.exec(statement).all()
        
        # Liste des IDs de jobs actifs
        active_job_ids = {f"scheduled_job_{job.id}" for job in jobs}
        
        # Supprimer les jobs APScheduler qui ne sont plus actifs
        current_jobs = scheduler.get_jobs()
        for current_job in current_jobs:
            if current_job.id.startswith("scheduled_job_") and current_job.id not in active_job_ids:
                scheduler.remove_job(current_job.id)
                logger.info(f"Job APScheduler supprimé: {current_job.id}")
        
        # Ajouter ou mettre à jour les jobs actifs
        for job in jobs:
            job_id = f"scheduled_job_{job.id}"
            
            # Vérifier si le job existe déjà
            existing_job = scheduler.get_job(job_id)
            
            if existing_job:
                # Supprimer le job existant pour le recréer avec la config actuelle
                scheduler.remove_job(job_id)
            
            # Créer (ou recréer) le job APScheduler
            try:
                trigger = CronTrigger.from_crontab(job.cron_expression)
                scheduler.add_job(
                    run_scheduled_job,
                    trigger=trigger,
                    args=[job.id],
                    id=job_id,
                    replace_existing=True
                )
                logger.info(f"Job APScheduler ajouté: {job_id} avec cron '{job.cron_expression}'")
            except Exception as e:
                logger.error(f"Erreur lors de l'ajout du job {job_id}: {e}")


async def run_scheduled_job(job_id: int):
    """
    Exécuter un scheduled job : exécuter toutes ses tâches
    
    Args:
        job_id: ID du ScheduledJob dans la base de données
    """
    logger.info(f"Exécution du scheduled job {job_id}")
    
    with Session(engine) as session:
        # Charger le job
        job = session.get(ScheduledJob, job_id)
        
        if not job:
            logger.error(f"Scheduled job {job_id} non trouvé")
            return
        
        if not job.enabled:
            logger.warning(f"Scheduled job {job_id} est désactivé, arrêt de l'exécution")
            return
        
        # Exécuter chaque tâche
        run_at = datetime.utcnow()
        for task_id in job.task_ids:
            logger.info(f"Exécution de la tâche {task_id} pour l'agent {job.agent_id}")
            
            # Récupérer le nom de la tâche
            task = session.get(AgentTask, task_id)
            task_name = task.name if task else f"Tâche {task_id}"
            
            try:
                # Ajouter un contexte avec la date/heure actuelle
                input_context = f"Exécution planifiée à {run_at.strftime('%Y-%m-%d %H:%M:%S')} UTC"
                
                result = await execute_agent_task(
                    session=session,
                    agent_id=job.agent_id,
                    task_id=task_id,
                    input_context=input_context
                )
                
                # Enregistrer le résultat dans TaskRunLog
                out = result.get("output")
                err = result.get("error")
                # Si erreur, ne pas sauver output vide qui cache l'erreur
                task_run_log = TaskRunLog(
                    scheduled_job_id=job.id,
                    agent_task_id=task_id,
                    task_name=task_name,
                    output=out if out is not None and not err else None,
                    error=err if err else None,
                    run_at=run_at,
                    user_id=job.user_id
                )
                session.add(task_run_log)
                session.commit()
                
                if result.get("error"):
                    logger.error(f"Erreur lors de l'exécution de la tâche {task_id}: {result['error']}")
                else:
                    logger.info(f"Tâche {task_id} exécutée avec succès")
                    out = (result.get("output") or "")[:200]
                    if out:
                        logger.debug(f"Résultat: {out}...")
                
            except Exception as e:
                logger.error(f"Exception lors de l'exécution de la tâche {task_id}: {e}", exc_info=True)
                # Enregistrer l'erreur dans TaskRunLog
                try:
                    task_run_log = TaskRunLog(
                        scheduled_job_id=job.id,
                        agent_task_id=task_id,
                        task_name=task_name,
                        output=None,
                        error=str(e),
                        run_at=run_at,
                        user_id=job.user_id
                    )
                    session.add(task_run_log)
                    session.commit()
                except Exception as log_error:
                    logger.error(f"Impossible d'enregistrer l'erreur dans TaskRunLog: {log_error}")
        
        # Mettre à jour last_run_at
        job.last_run_at = datetime.utcnow()
        session.add(job)
        session.commit()
        
        logger.info(f"Scheduled job {job_id} terminé")
