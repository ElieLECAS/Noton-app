from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.database import get_session
from app.services.ollama_service import get_available_models, chat, chat_stream
from app.services.semantic_search_service import search_relevant_chunks
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


def build_semantic_context(chunk_results: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les chunks pertinents trouvés par recherche sémantique RAG"""
    system_message = {
        "role": "system",
        "content": "Tu es un assistant IA qui aide à analyser et résumer les notes d'un projet. Tu as accès aux passages les plus pertinents des notes du projet ci-dessous (trouvés par recherche sémantique RAG). Réponds aux questions de l'utilisateur en te basant sur ces passages.\n\n"
    }
    
    chunks_content = []
    seen_notes = set()  # Pour éviter de répéter le titre de la note
    
    for result in chunk_results:
        chunk = result['chunk']
        note = result['note']
        score = result.get('score', 0.0)
        
        # Ajouter le titre de la note seulement la première fois qu'on la voit
        note_header = ""
        if note.id not in seen_notes:
            note_header = f"Note: {note.title}\n"
            seen_notes.add(note.id)
        
        chunk_text = f"{note_header}Passage: {chunk.content}\n"
        chunk_text += "---\n"
        chunks_content.append(chunk_text)
    
    if chunks_content:
        system_message["content"] += "\n".join(chunks_content)
        system_message["content"] += f"\n\n(Note: {len(chunk_results)} passage(s) pertinent(s) trouvé(s) sur la base de votre question parmi {len(seen_notes)} note(s))"
    else:
        system_message["content"] += "Aucun passage pertinent trouvé pour votre question."
    
    return [system_message]


@router.post("/projects/{project_id}/chat/stream")
async def stream_project_chat_message(
    project_id: int,
    request: ProjectChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Envoyer un message au chatbot Ollama avec streaming et contexte enrichi des chunks pertinents du projet"""
    # Recherche sémantique RAG au niveau des chunks
    # Retourne les passages les plus pertinents (chunks) au lieu des notes complètes
    chunk_results = search_relevant_chunks(
        session=session,
        project_id=project_id,
        query_text=request.message,
        user_id=current_user.id,
        k=10  # 10 chunks pertinents
    )
    
    # Construire le contexte enrichi avec seulement les chunks pertinents
    project_context = build_semantic_context(chunk_results)
    
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

