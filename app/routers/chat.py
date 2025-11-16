from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.database import get_session
from app.services.ollama_service import get_available_models, chat, chat_stream
from app.services.semantic_search_service import search_relevant_notes, search_relevant_passages
import json

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    model: str
    context: Optional[List[dict]] = None


@router.get("/ollama/models", response_model=List[str])
async def list_ollama_models():
    """Récupérer la liste des modèles Ollama disponibles"""
    models = await get_available_models()
    return models


@router.post("/chat")
async def send_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user)
):
    """Envoyer un message au chatbot Ollama"""
    try:
        response = await chat(request.message, request.model, request.context)
        # Ollama retourne {"message": {"role": "assistant", "content": "..."}, ...}
        if "message" in response and "content" in response["message"]:
            return {"message": {"content": response["message"]["content"]}}
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel à Ollama: {str(e)}")


@router.post("/chat/stream")
async def stream_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user)
):
    """Envoyer un message au chatbot Ollama avec streaming"""
    async def generate():
        try:
            async for line in chat_stream(request.message, request.model, request.context):
                if line.strip():
                    # Ollama renvoie des lignes JSON, on les renvoie telles quelles en SSE
                    yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


class ProjectChatRequest(BaseModel):
    message: str
    model: str
    context: Optional[List[dict]] = None


def build_semantic_context_from_passages(passages: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les passages pertinents trouvés par recherche sémantique"""
    system_message = {
        "role": "system",
        "content": (
            "Tu es un assistant IA qui aide à analyser et résumer les notes d'un projet. "
            "Tu as accès aux PASSAGES les plus pertinents extraits de TOUTES les notes du projet "
            "(trouvés par recherche sémantique RAG avancée). "
            "Ces passages proviennent de différentes notes et contiennent les informations les plus pertinentes "
            "pour répondre à la question de l'utilisateur.\n\n"
            "Réponds aux questions de l'utilisateur en te basant sur ces passages. "
            "Tu peux combiner des informations de différents passages pour donner une réponse complète.\n\n"
            "IMPORTANT - FORMAT DE RÉPONSE :\n"
            "Tu DOIS TOUJOURS formater tes réponses en Markdown pour une meilleure lisibilité :\n"
            "- Utilise des **titres** (## pour les sections principales, ### pour les sous-sections)\n"
            "- Utilise des **listes à puces** (- ou *) ou **listes numérotées** (1. 2. 3.) pour énumérer des éléments\n"
            "- Utilise **le gras** pour mettre en évidence les points importants\n"
            "- Utilise *l'italique* pour les termes techniques ou les emphases\n"
            "- Utilise des `blocs de code` pour les termes spécifiques\n"
            "- Structure ta réponse avec des paragraphes clairs séparés par des lignes vides\n"
            "- Pour les résumés longs, utilise des sections avec des titres\n\n"
        )
    }
    
    passages_content = []
    
    if passages:
        system_message["content"] += "PASSAGES PERTINENTS EXTRAITS DU PROJET :\n\n"
        
        for i, passage_data in enumerate(passages, 1):
            passage = passage_data['passage']
            score = passage_data.get('score', 0.0)
            note_title = passage_data.get('note_title', 'Note sans titre')
            
            # Construire le passage avec métadonnées
            passage_text = f"[Passage {i}] (Pertinence: {score:.2f})\n{passage}\n"
            passages_content.append(passage_text)
        
        system_message["content"] += "\n---\n".join(passages_content)
        system_message["content"] += (
            f"\n\n📊 Statistiques : {len(passages)} passages pertinents extraits et analysés. "
            f"Toutes les notes du projet ont été examinées pour trouver les informations les plus pertinentes."
        )
    else:
        system_message["content"] += "Aucun passage pertinent trouvé dans le projet."
    
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
    """Envoyer un message au chatbot Ollama avec streaming et contexte enrichi des passages pertinents du projet"""
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
            # Passer None comme message car il est déjà dans le contexte
            async for line in chat_stream("", request.model, full_context):
                if line.strip():
                    yield f"data: {line}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

