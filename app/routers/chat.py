from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.database import get_session
from app.services.ollama_service import get_available_models as get_ollama_models, chat as ollama_chat, chat_stream as ollama_chat_stream
from app.services.openai_service import get_available_models as get_openai_models, chat as openai_chat, chat_stream as openai_chat_stream
from app.config import settings
from app.services.semantic_search_service import search_relevant_notes, search_relevant_passages
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "ollama"  # "ollama" ou "openai"
    context: Optional[List[dict]] = None


@router.get("/ollama/models", response_model=List[str])
async def list_ollama_models():
    """Récupérer la liste des modèles Ollama disponibles"""
    models = await get_ollama_models()
    return models


@router.get("/openai/models", response_model=List[str])
async def list_openai_models():
    """Récupérer uniquement le modèle OpenAI configuré dans .env"""
    # Retourner uniquement le modèle configuré dans le .env si disponible
    # On retourne le modèle même si seulement OPENAI_MODEL est défini (pour l'affichage)
    if settings.OPENAI_MODEL:
        logger.info(f"Modèle OpenAI configuré: {settings.OPENAI_MODEL}")
        return [settings.OPENAI_MODEL]
    else:
        logger.warning("OPENAI_MODEL n'est pas configuré dans les variables d'environnement")
        return []


@router.get("/providers/models")
async def list_all_models():
    """Récupérer tous les modèles disponibles par provider"""
    ollama_models = await get_ollama_models()
    openai_models = await get_openai_models()
    
    return {
        "ollama": ollama_models,
        "openai": openai_models
    }


@router.post("/chat")
async def send_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI)"""
    try:
        if request.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise HTTPException(status_code=400, detail="OpenAI API key n'est pas configurée")
            response = await openai_chat(request.message, request.model, request.context)
            # OpenAI retourne {"choices": [{"message": {"content": "..."}}]}
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"].get("content", "")
                return {"message": {"content": content}}
            return response
        else:  # ollama par défaut
            response = await ollama_chat(request.message, request.model, request.context)
            # Ollama retourne {"message": {"role": "assistant", "content": "..."}, ...}
            if "message" in response and "content" in response["message"]:
                return {"message": {"content": response["message"]["content"]}}
            return response
    except HTTPException:
        raise
    except Exception as e:
        provider_name = "OpenAI" if request.provider == "openai" else "Ollama"
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel à {provider_name}: {str(e)}")


@router.post("/chat/stream")
async def stream_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI) avec streaming"""
    async def generate():
        try:
            if request.provider == "openai":
                if not settings.OPENAI_API_KEY:
                    error_msg = "OpenAI API key n'est pas configurée"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                async for line in openai_chat_stream(request.message, request.model, request.context):
                    if line.strip():
                        # OpenAI renvoie déjà au format Ollama via le service
                        yield f"data: {line}\n\n"
            else:  # ollama par défaut
                async for line in ollama_chat_stream(request.message, request.model, request.context):
                    if line.strip():
                        # Ollama renvoie des lignes JSON, on les renvoie telles quelles en SSE
                        yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


class ProjectChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "ollama"  # "ollama" ou "openai"
    context: Optional[List[dict]] = None


def build_semantic_context_from_passages(passages: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les passages pertinents trouvés par recherche sémantique"""
    system_message = {
        "role": "system",
        "content": (
            "Tu es un assistant IA amical et professionnel qui aide à analyser et résumer les notes d'un projet.\n\n"
            "RÈGLES IMPORTANTES DE CONVERSATION :\n"
            "- Pour les salutations simples (bonjour, salut, bonsoir, etc.), réponds de manière naturelle et amicale "
            "sans mentionner les passages ou le projet. Exemple : 'Bonjour ! Comment puis-je vous aider aujourd'hui ?'\n"
            "- Pour les questions générales ou de conversation, réponds naturellement sans forcer l'utilisation des passages.\n"
            "- Utilise les passages fournis UNIQUEMENT lorsque la question de l'utilisateur nécessite des informations "
            "spécifiques sur le projet ou les notes.\n"
            "- Si les passages ne sont pas pertinents pour répondre à la question, réponds simplement sans les mentionner.\n"
            "- Sois concis et naturel dans tes réponses.\n\n"
            "QUAND UTILISER LES PASSAGES :\n"
            "- Questions sur le contenu du projet, les notes, les décisions prises, etc.\n"
            "- Demandes de résumé, d'analyse ou d'explication sur le projet\n"
            "- Questions techniques ou spécifiques nécessitant des informations du projet\n\n"
            "FORMAT DE RÉPONSE (uniquement pour les réponses longues ou structurées) :\n"
            "- Utilise le Markdown pour améliorer la lisibilité :\n"
            "  * **Titres** (## pour les sections principales, ### pour les sous-sections)\n"
            "  * **Listes à puces** (- ou *) ou **listes numérotées** (1. 2. 3.)\n"
            "  * **Gras** pour les points importants\n"
            "  * *Italique* pour les termes techniques\n"
            "  * `Blocs de code` pour les termes spécifiques\n"
            "- Pour les réponses courtes ou simples, réponds naturellement sans formatage excessif.\n\n"
        )
    }
    
    passages_content = []
    
    if passages:
        system_message["content"] += (
            "PASSAGES DISPONIBLES DU PROJET (à utiliser uniquement si pertinents pour répondre à la question) :\n\n"
        )
        
        for i, passage_data in enumerate(passages, 1):
            passage = passage_data['passage']
            score = passage_data.get('score', 0.0)
            note_title = passage_data.get('note_title', 'Note sans titre')
            
            # Construire le passage avec métadonnées
            passage_text = f"[Passage {i}] (Pertinence: {score:.2f})\n{passage}\n"
            passages_content.append(passage_text)
        
        system_message["content"] += "\n---\n".join(passages_content)
        system_message["content"] += (
            f"\n\n(Note : {len(passages)} passages disponibles. Utilise-les uniquement si nécessaire pour répondre à la question.)"
        )
    
    return [system_message]


def build_semantic_context(note_results: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les notes pertinentes trouvées par recherche sémantique RAG"""
    system_message = {
        "role": "system",
        "content": "Tu es un assistant IA qui aide à analyser et résumer les notes d'un projet. Tu as accès aux notes les plus pertinentes du projet ci-dessous (trouvées par recherche sémantique RAG). Réponds aux questions de l'utilisateur en te basant sur ces notes.\n\n"
    }
    
    notes_content = []
    
    for result in note_results:
        note = result['note']
        score = result.get('score', 0.0)
        
        # Construire le contenu de la note
        note_text = f"Note: {note.title}\n"
        if note.content:
            note_text += f"Contenu: {note.content}\n"
        note_text += f"Pertinence: {score:.2f}\n"
        note_text += "---\n"
        notes_content.append(note_text)
    
    if notes_content:
        system_message["content"] += "\n".join(notes_content)
        system_message["content"] += f"\n\n(Note: {len(note_results)} note(s) pertinente(s) trouvée(s) sur la base de votre question)"
    else:
        system_message["content"] += "Aucune note pertinente trouvée pour votre question."
    
    return [system_message]


@router.post("/projects/{project_id}/chat/stream")
async def stream_project_chat_message(
    project_id: int,
    request: ProjectChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI) avec streaming et contexte enrichi des passages pertinents du projet"""
    # Recherche sémantique RAG AVANCÉE au niveau des passages
    # Analyse TOUTES les notes du projet et retourne les PASSAGES les plus pertinents
    passages = search_relevant_passages(
        session=session,
        project_id=project_id,
        query_text=request.message,
        user_id=current_user.id,
        k=15,  # 15 passages pertinents (environ 500 caractères chacun)
        passage_size=500
    )
    
    # Construire le contexte enrichi avec les passages pertinents
    project_context = build_semantic_context_from_passages(passages)
    
    # Ajouter le contexte de conversation existant si fourni
    if request.context:
        full_context = project_context + request.context
    else:
        full_context = project_context
    
    # Ajouter le message utilisateur actuel
    full_context.append({"role": "user", "content": request.message})
    
    async def generate():
        try:
            if request.provider == "openai":
                if not settings.OPENAI_API_KEY:
                    error_msg = "OpenAI API key n'est pas configurée"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                # Passer None comme message car il est déjà dans le contexte
                async for line in openai_chat_stream("", request.model, full_context):
                    if line.strip():
                        yield f"data: {line}\n\n"
            else:  # ollama par défaut
                # Passer None comme message car il est déjà dans le contexte
                async for line in ollama_chat_stream("", request.model, full_context):
                    if line.strip():
                        yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

