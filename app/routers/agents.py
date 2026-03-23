from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, func
from app.models.user import UserRead
from app.models.agent import Agent, AgentCreate, AgentRead, AgentUpdate
from app.models.agent_task import AgentTask, AgentTaskCreate, AgentTaskRead, AgentTaskUpdate
from app.routers.auth import get_current_user
from app.database import get_session
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("", response_model=List[AgentRead])
async def list_agents(
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Lister les agents de l'utilisateur"""
    query = select(
        Agent,
        func.count(AgentTask.id).label("task_count")
    ).outerjoin(AgentTask).where(
        Agent.user_id == current_user.id
    ).group_by(Agent.id).order_by(Agent.created_at.desc())
    
    results = session.exec(query).all()
    
    agents = []
    for agent, task_count in results:
        agent_read = AgentRead.from_orm(agent)
        agent_read.task_count = task_count or 0
        agents.append(agent_read)
    
    return agents


@router.post("", response_model=AgentRead)
async def create_agent(
    agent: AgentCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Créer un nouvel agent"""
    db_agent = Agent(
        user_id=current_user.id,
        name=agent.name,
        personality=agent.personality,
        model_preset=agent.model_preset
    )
    session.add(db_agent)
    session.commit()
    session.refresh(db_agent)
    
    agent_read = AgentRead.from_orm(db_agent)
    agent_read.task_count = 0
    
    logger.info(f"Agent créé: {db_agent.id} pour l'utilisateur {current_user.id}")
    return agent_read


@router.get("/{agent_id}", response_model=AgentRead)
async def get_agent(
    agent_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Récupérer un agent spécifique"""
    agent = session.get(Agent, agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    # Compter les tâches
    task_count = session.exec(
        select(func.count(AgentTask.id)).where(AgentTask.agent_id == agent_id)
    ).first()
    
    agent_read = AgentRead.from_orm(agent)
    agent_read.task_count = task_count or 0
    
    return agent_read


@router.put("/{agent_id}", response_model=AgentRead)
async def update_agent(
    agent_id: int,
    agent_update: AgentUpdate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Mettre à jour un agent"""
    agent = session.get(Agent, agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    if agent_update.name is not None:
        agent.name = agent_update.name
    if agent_update.personality is not None:
        agent.personality = agent_update.personality
    if agent_update.model_preset is not None:
        agent.model_preset = agent_update.model_preset
    
    agent.updated_at = datetime.utcnow()
    
    session.add(agent)
    session.commit()
    session.refresh(agent)
    
    # Compter les tâches
    task_count = session.exec(
        select(func.count(AgentTask.id)).where(AgentTask.agent_id == agent_id)
    ).first()
    
    agent_read = AgentRead.from_orm(agent)
    agent_read.task_count = task_count or 0
    
    return agent_read


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Supprimer un agent"""
    agent = session.get(Agent, agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    # Supprimer l'agent (les tâches et jobs associés seront supprimés en cascade)
    session.delete(agent)
    session.commit()
    
    logger.info(f"Agent supprimé: {agent_id}")
    return {"ok": True, "message": "Agent supprimé"}


# Routes pour les tâches des agents

@router.get("/{agent_id}/tasks", response_model=List[AgentTaskRead])
async def list_agent_tasks(
    agent_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Lister les tâches d'un agent"""
    # Vérifier que l'agent appartient à l'utilisateur
    agent = session.get(Agent, agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    statement = select(AgentTask).where(
        AgentTask.agent_id == agent_id
    ).order_by(AgentTask.created_at)
    
    tasks = session.exec(statement).all()
    return tasks


@router.post("/{agent_id}/tasks", response_model=AgentTaskRead)
async def create_agent_task(
    agent_id: int,
    task: AgentTaskCreate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Créer une nouvelle tâche pour un agent"""
    # Vérifier que l'agent appartient à l'utilisateur
    agent = session.get(Agent, agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    db_task = AgentTask(
        agent_id=agent_id,
        name=task.name,
        instruction=task.instruction
    )
    session.add(db_task)
    session.commit()
    session.refresh(db_task)
    
    logger.info(f"Tâche créée: {db_task.id} pour l'agent {agent_id}")
    return db_task


@router.put("/{agent_id}/tasks/{task_id}", response_model=AgentTaskRead)
async def update_agent_task(
    agent_id: int,
    task_id: int,
    task_update: AgentTaskUpdate,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Mettre à jour une tâche"""
    # Vérifier que l'agent appartient à l'utilisateur
    agent = session.get(Agent, agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    # Vérifier que la tâche appartient à l'agent
    task = session.get(AgentTask, task_id)
    if not task or task.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    if task_update.name is not None:
        task.name = task_update.name
    if task_update.instruction is not None:
        task.instruction = task_update.instruction
    
    session.add(task)
    session.commit()
    session.refresh(task)
    
    return task


@router.delete("/{agent_id}/tasks/{task_id}")
async def delete_agent_task(
    agent_id: int,
    task_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Supprimer une tâche"""
    # Vérifier que l'agent appartient à l'utilisateur
    agent = session.get(Agent, agent_id)
    if not agent or agent.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Agent non trouvé")
    
    # Vérifier que la tâche appartient à l'agent
    task = session.get(AgentTask, task_id)
    if not task or task.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    session.delete(task)
    session.commit()
    
    logger.info(f"Tâche supprimée: {task_id}")
    return {"ok": True, "message": "Tâche supprimée"}
