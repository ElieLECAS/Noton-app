"""
Service pour exécuter les tâches des agents avec LangGraph
"""
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from sqlmodel import Session
from app.models.agent import Agent
from app.models.agent_task import AgentTask
from app.services.ollama_service import chat as ollama_chat
from app.services.openai_service import chat as openai_chat
from app.services.mistral_service import chat as mistral_chat
from app.config import settings, get_model_for_preset
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentState(Dict):
    """État pour le graphe d'exécution d'agent"""
    agent_id: int
    task_id: int
    agent_personality: str
    task_instruction: str
    model: str
    provider: str
    input: Optional[str]
    output: Optional[str]
    error: Optional[str]


async def load_agent_and_task(state: AgentState) -> AgentState:
    """Nœud : Charger l'agent et la tâche depuis la base de données"""
    logger.info(f"Chargement de l'agent {state['agent_id']} et de la tâche {state['task_id']}")
    
    # Note : nous devons passer la session depuis l'extérieur ou utiliser une session contextuelle
    # Pour l'instant, nous supposons que agent_personality et task_instruction sont déjà chargés
    # Cette fonction serait étendue pour charger depuis la DB si nécessaire
    
    return state


async def run_task(state: AgentState) -> AgentState:
    """Nœud : Exécuter la tâche en appelant le LLM avec la personnalité de l'agent"""
    logger.info(f"=== Exécution de la tâche {state['task_id']} pour l'agent {state['agent_id']} ===")
    logger.info(f"State reçu: provider={state.get('provider')!r}, model={state.get('model')!r}")
    
    try:
        # Construire le contexte avec la personnalité de l'agent
        context = [
            {"role": "system", "content": state["agent_personality"]},
            {"role": "user", "content": state["task_instruction"]}
        ]
        
        # Ajouter l'input si fourni (ex: date, contexte supplémentaire)
        if state.get("input"):
            context.append({"role": "user", "content": f"Contexte supplémentaire : {state['input']}"})
        
        # Même logique que le chat : provider/model du state (définis depuis get_model_for_preset)
        provider = (state.get("provider") or "ollama")
        if not isinstance(provider, str):
            provider = "ollama"
        provider = provider.strip().lower()
        model = state.get("model")
        if model is None or (isinstance(model, str) and not model.strip()):
            model = (settings.MODEL_FAST_NAME or "").strip()
        else:
            model = str(model).strip()
        
        logger.info(f"run_task: provider={provider!r} model={model!r}")
        
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                output = "Erreur : OPENAI_API_KEY non configurée. Configurez-la pour utiliser le modèle prédéfini (OpenAI)."
                logger.error(f"OPENAI_API_KEY manquante pour task {state['task_id']}")
            else:
                model_openai = model or (settings.OPENAI_MODEL[0] if settings.OPENAI_MODEL else "gpt-4o-mini")
                logger.info(f"Appel OpenAI avec modèle: {model_openai!r}")
                logger.info(f"Context: {len(context)} messages")
                response = await openai_chat("", model_openai, context)
                logger.info(f"Réponse OpenAI reçue: {response.keys() if isinstance(response, dict) else type(response)}")
                if "choices" in response and len(response["choices"]) > 0:
                    output = response["choices"][0]["message"].get("content") or ""
                    logger.info(f"Content extrait: {len(output)} caractères")
                else:
                    output = "Erreur : réponse OpenAI vide"
                    logger.warning(f"Réponse OpenAI sans choices ou vide: {response}")
        elif provider == "mistral":
            if not settings.MISTRAL_API_KEY:
                output = "Erreur : MISTRAL_API_KEY non configurée. Configurez-la pour utiliser le modèle prédéfini (Mistral)."
                logger.error(f"MISTRAL_API_KEY manquante pour task {state['task_id']}")
            else:
                model_mistral = model or "mistral-small-latest"
                logger.info(f"Appel Mistral avec modèle: {model_mistral!r}")
                logger.info(f"Context: {len(context)} messages")
                response = await mistral_chat("", model_mistral, context)
                logger.info(f"Réponse Mistral reçue: {response.keys() if isinstance(response, dict) else type(response)}")
                if "choices" in response and len(response["choices"]) > 0:
                    output = response["choices"][0]["message"].get("content") or ""
                    logger.info(f"Content extrait: {len(output)} caractères")
                else:
                    output = "Erreur : réponse Mistral vide"
                    logger.warning(f"Réponse Mistral sans choices ou vide: {response}")
        else:
            # Ollama uniquement quand le provider prédéfini est ollama
            model_ollama = model or settings.MODEL_FAST_NAME or "llama3.2:1b"
            logger.info(f"Appel Ollama avec modèle: {model_ollama!r}")
            response = await ollama_chat("", model_ollama, context)
            logger.info(f"Réponse Ollama reçue: {response.keys() if isinstance(response, dict) else type(response)}")
            msg = response.get("message") or {}
            output = msg.get("content")
            if output is None:
                output = response.get("response") or ""
            if output is None:
                output = "Erreur : réponse Ollama vide"
                logger.warning(f"Réponse Ollama vide: {response}")
            else:
                output = str(output).strip() or ""
                logger.info(f"Content Ollama: {len(output)} caractères")
        
        state["output"] = output or ""
        logger.info(f"State output final: {len(state['output'])} caractères")
        state["error"] = None
        logger.info(f"Tâche {state['task_id']} exécutée avec succès")
        
    except Exception as e:
        error_msg = f"Erreur : {type(e).__name__}: {str(e)}"
        logger.error(f"EXCEPTION lors de l'exécution de la tâche {state['task_id']}: {error_msg}", exc_info=True)
        state["error"] = error_msg
        state["output"] = None
    
    return state


def create_agent_graph() -> StateGraph:
    """Créer le graphe d'exécution d'agent avec LangGraph"""
    workflow = StateGraph(AgentState)
    
    # Ajouter les nœuds
    workflow.add_node("load", load_agent_and_task)
    workflow.add_node("run_task", run_task)
    
    # Définir les arêtes
    workflow.set_entry_point("load")
    workflow.add_edge("load", "run_task")
    workflow.add_edge("run_task", END)
    
    # Compiler le graphe
    return workflow.compile()


# Instance globale du graphe
agent_graph = create_agent_graph()


async def execute_agent_task(
    session: Session,
    agent_id: int,
    task_id: int,
    input_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exécuter une tâche d'agent via LangGraph
    
    Args:
        session: Session de base de données
        agent_id: ID de l'agent
        task_id: ID de la tâche
        input_context: Contexte supplémentaire optionnel (ex: date, événement)
    
    Returns:
        Dict contenant output, error, etc.
    """
    # Charger l'agent et la tâche depuis la DB
    agent = session.get(Agent, agent_id)
    task = session.get(AgentTask, task_id)
    
    if not agent:
        logger.error(f"Agent {agent_id} non trouvé")
        return {"error": f"Agent {agent_id} non trouvé", "output": None}
    
    if not task:
        logger.error(f"Tâche {task_id} non trouvée")
        return {"error": f"Tâche {task_id} non trouvée", "output": None}
    
    if task.agent_id != agent_id:
        logger.error(f"La tâche {task_id} n'appartient pas à l'agent {agent_id}")
        return {"error": "Tâche non associée à cet agent", "output": None}
    
    # Même source que le chat : preset → provider + model (config)
    preset = (agent.model_preset or "fast").strip().lower()
    preset_config = get_model_for_preset(agent.model_preset)
    provider_raw = preset_config.get("provider")
    model_raw = preset_config.get("model")
    provider = (provider_raw if provider_raw is not None else "ollama").strip().lower()
    model = (model_raw if model_raw is not None else "").strip()
    if not model and provider == "ollama":
        model = (settings.MODEL_FAST_NAME or "llama3.2:1b").strip()
    
    logger.info(f"Agent preset={preset!r} → provider={provider!r} model={model!r}")
    
    # Préparer l'état initial
    initial_state: AgentState = {
        "agent_id": agent_id,
        "task_id": task_id,
        "agent_personality": agent.personality,
        "task_instruction": task.instruction,
        "model": model,
        "provider": provider,
        "input": input_context,
        "output": None,
        "error": None
    }
    
    # Exécuter le graphe
    logger.info(f"Démarrage de l'exécution de la tâche {task_id} pour l'agent {agent_id}")
    logger.info(f"Initial state: provider={initial_state.get('provider')!r}, model={initial_state.get('model')!r}")
    result_state = await agent_graph.ainvoke(initial_state)
    logger.info(f"Result state reçu: type={type(result_state)}, keys={result_state.keys() if isinstance(result_state, dict) else 'N/A'}")
    # LangGraph peut retourner le state dans une clé (ex: __root__) selon la config
    if isinstance(result_state, dict) and "output" not in result_state and len(result_state) == 1:
        result_state = next(iter(result_state.values()), result_state)
    output = result_state.get("output") if isinstance(result_state, dict) else None
    error = result_state.get("error") if isinstance(result_state, dict) else None
    return {
        "agent_id": agent_id,
        "task_id": task_id,
        "output": (output if output is not None else "") if not error else None,
        "error": error,
        "timestamp": datetime.utcnow().isoformat()
    }
