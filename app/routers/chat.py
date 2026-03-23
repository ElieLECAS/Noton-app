from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.database import get_session
from app.services.ollama_service import get_available_models as get_ollama_models, chat as ollama_chat, chat_stream as ollama_chat_stream
from app.services.openai_service import (
    get_available_models as get_openai_models,
    chat as openai_chat,
    chat_stream as openai_chat_stream,
    generate_image as openai_generate_image,
)
from app.services.mistral_service import (
    chat as mistral_chat,
    chat_stream as mistral_chat_stream,
)
from app.config import settings
from app.services.semantic_search_service import search_relevant_notes, search_relevant_passages
from app.services.chat_tools import get_available_tools
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.agent import Agent
from app.models.note import Note
from app.models.space import Space
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)

# Nombre de passages RAG renvoyés au LLM (configurable via RAG_TOP_K)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "ollama"  # "ollama", "openai" ou "mistral"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None  # ID de la conversation (optionnel pour compatibilité)
    agent_id: Optional[int] = None  # ID de l'agent (optionnel)


class GenerateImageRequest(BaseModel):
    prompt: str
    conversation_id: Optional[int] = None


@router.get("/ollama/models", response_model=List[str])
async def list_ollama_models():
    """Récupérer la liste des modèles Ollama disponibles"""
    models = await get_ollama_models()
    return models


@router.get("/openai/models", response_model=List[str])
async def list_openai_models():
    """Récupérer les modèles OpenAI configurés dans .env"""
    # Retourner les modèles configurés dans le .env si disponibles
    if settings.OPENAI_MODEL:
        logger.info(f"Modèles OpenAI configurés: {settings.OPENAI_MODEL}")
        return settings.OPENAI_MODEL
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
        "openai": openai_models,
        # Les modèles Mistral sont généralement configurés côté client via les presets/env,
        # on ne récupère donc pas dynamiquement la liste ici.
    }


@router.post("/chat")
async def send_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI). Si BRAVE_SEARCH_API_KEY est configurée, le modèle peut utiliser la recherche web."""
    try:
        tools = get_available_tools(include_brave_search=bool(settings.BRAVE_SEARCH_API_KEY))
        if request.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise HTTPException(status_code=400, detail="OpenAI API key n'est pas configurée")
            response = await openai_chat(request.message, request.model, request.context, tools=tools or None)
            # OpenAI retourne {"choices": [{"message": {"content": "..."}}]}
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"].get("content", "")
                return {"message": {"content": content}}
            return response
        elif request.provider == "mistral":
            if not settings.MISTRAL_API_KEY:
                raise HTTPException(status_code=400, detail="Mistral API key n'est pas configurée")
            response = await mistral_chat(request.message, request.model, request.context, tools=tools or None)
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"].get("content", "")
                return {"message": {"content": content}}
            return response
        else:  # ollama par défaut
            response = await ollama_chat(request.message, request.model, request.context, tools=tools or None)
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
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI) avec streaming"""
    # Charger l'agent si agent_id est fourni
    agent = None
    if request.agent_id:
        agent = session.get(Agent, request.agent_id)
        if not agent or agent.user_id != current_user.id:
            logger.warning(f"Agent {request.agent_id} non trouvé ou non autorisé pour l'utilisateur {current_user.id}")
            agent = None
    
    # Sauvegarder le message utilisateur si conversation_id est fourni
    if request.conversation_id:
        try:
            user_message = Message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.message,
                model=None,
                provider=None
            )
            session.add(user_message)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message utilisateur: {e}")
    
    # Construire le contexte avec la personnalité de l'agent si disponible
    full_context = []
    if agent:
        full_context.append({"role": "system", "content": agent.personality})
    
    # Ajouter le contexte existant
    if request.context:
        full_context.extend(request.context)
    
    # Variable pour accumuler la réponse de l'assistant
    assistant_response = []
    tools = get_available_tools(include_brave_search=bool(settings.BRAVE_SEARCH_API_KEY))
    use_tools = bool(tools)

    async def generate():
        try:
            if use_tools:
                # Avec tools : appel non-streaming (boucle tool_calls) puis on simule le stream pour l'UX
                full_messages = full_context + [{"role": "user", "content": request.message}]
                if request.provider == "openai":
                    if not settings.OPENAI_API_KEY:
                        error_msg = "OpenAI API key n'est pas configurée"
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    response = await openai_chat("", request.model, full_messages, tools=tools)
                    content = ""
                    if "choices" in response and response["choices"]:
                        content = (response["choices"][0].get("message") or {}).get("content") or ""
                elif request.provider == "mistral":
                    if not settings.MISTRAL_API_KEY:
                        error_msg = "Mistral API key n'est pas configurée"
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    response = await mistral_chat("", request.model, full_messages, tools=tools)
                    content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                else:
                    response = await ollama_chat("", request.model, full_messages, tools=tools)
                    content = (response.get("message") or {}).get("content") or ""
                # Simuler le streaming par chunks pour garder l'effet de frappe côté client
                chunk_size = 25
                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            else:
                if request.provider == "openai":
                    if not settings.OPENAI_API_KEY:
                        error_msg = "OpenAI API key n'est pas configurée"
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    async for line in openai_chat_stream(request.message, request.model, full_context):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    assistant_response.append(data["message"]["content"])
                            except Exception:
                                pass
                            yield f"data: {line}\n\n"
                elif request.provider == "mistral":
                    if not settings.MISTRAL_API_KEY:
                        error_msg = "Mistral API key n'est pas configurée"
                        yield f"data: {json.dumps({'error': error_msg})}\n\n"
                        return
                    # Pas d'API streaming dédiée utilisée ici : on simule le stream à partir de la réponse complète
                    response = await mistral_chat(request.message, request.model, full_context)
                    content = ""
                    if "choices" in response and response["choices"]:
                        content = (response["choices"][0].get("message") or {}).get("content") or ""
                    chunk_size = 25
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i : i + chunk_size]
                        assistant_response.append(chunk)
                        yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"
                else:
                    async for line in ollama_chat_stream(request.message, request.model, full_context):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    assistant_response.append(data["message"]["content"])
                            except Exception:
                                pass
                            yield f"data: {line}\n\n"
            
            # Sauvegarder la réponse de l'assistant si conversation_id est fourni
            if request.conversation_id and assistant_response:
                try:
                    complete_response = "".join(assistant_response)
                    assistant_message = Message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=complete_response,
                        model=request.model,
                        provider=request.provider
                    )
                    session.add(assistant_message)
                    
                    # Mettre à jour la date de modification de la conversation
                    conversation = session.get(Conversation, request.conversation_id)
                    if conversation:
                        conversation.updated_at = datetime.utcnow()
                        session.add(conversation)
                    
                    session.commit()
                    logger.info(f"Réponse de l'assistant sauvegardée dans la conversation {request.conversation_id}")
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde de la réponse de l'assistant: {e}")
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/chat/generate-image")
async def chat_generate_image(
    request: GenerateImageRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Générer une image via OpenAI DALL-E 3 à partir d'un prompt texte."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key n'est pas configurée. La génération d'images nécessite une clé OpenAI.",
        )
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Le prompt ne peut pas être vide.")

    # Sauvegarder le message utilisateur si conversation_id est fourni
    if request.conversation_id:
        try:
            user_message = Message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.prompt.strip(),
                model=None,
                provider=None,
            )
            session.add(user_message)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message utilisateur: {e}")

    try:
        images = await openai_generate_image(prompt=request.prompt.strip())
    except Exception as e:
        logger.exception("Erreur génération image OpenAI: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Erreur lors de la génération d'image: {str(e)}",
        )

    if not images or "url" not in images[0]:
        raise HTTPException(
            status_code=502,
            detail="Aucune image retournée par l'API OpenAI.",
        )

    image_url = images[0]["url"]
    # Contenu markdown pour affichage dans le chat (image cliquable)
    content = f"![Image générée]({image_url})\n\n*Prompt : {request.prompt.strip()}*"

    # Sauvegarder la réponse assistant si conversation_id est fourni
    if request.conversation_id:
        try:
            assistant_message = Message(
                conversation_id=request.conversation_id,
                role="assistant",
                content=content,
                model=settings.OPENAI_IMAGE_MODEL,
                provider="openai",
            )
            session.add(assistant_message)
            conv = session.get(Conversation, request.conversation_id)
            if conv:
                conv.updated_at = datetime.utcnow()
                session.add(conv)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message assistant (image): {e}")

    return {"url": image_url, "content": content}


class ProjectChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "ollama"  # "ollama" ou "openai"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None  # ID de la conversation (optionnel pour compatibilité)
    agent_id: Optional[int] = None  # ID de l'agent (optionnel)


class SpaceChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "ollama"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None
    agent_id: Optional[int] = None


def build_semantic_context_from_passages(passages: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les passages pertinents trouvés par recherche sémantique"""
    # Préprompt court pour limiter les input tokens
    system_message = {
        "role": "system",
        "content": (
            "Expert Technico-Commercial PROFERM. Réponds uniquement à partir des passages ci-dessous. "
            "Donnée absente (Uw, prix, garantie…) → indique que la doc ne le précise pas. "
            "Règles : bref et poli ; pas de marques concurrentes sauf si citées ; contradiction → source la plus spécifique. "
            "Format : tableaux Markdown pour valeurs techniques ; citations [1], [2] pour les affirmations (pas en salutations)."
        ),
    }
    
    passages_content = []
    
    if passages:
        system_message["content"] += "\n\nPASSAGES :\n\n"
        
        for i, passage_data in enumerate(passages, 1):
            passage = passage_data['passage']
            score = passage_data.get('score', 0.0)
            note_title = passage_data.get('note_title', 'Note sans titre')

            # Pour les chunks image, injecter l'URL accessible par le frontend
            if (
                passage_data.get('is_image_chunk')
                and passage_data.get('image_filename')
                and passage_data.get('note_id')
            ):
                image_url = f"/api/images/{passage_data['note_id']}/{passage_data['image_filename']}"
                caption = passage_data.get('caption', '') or 'Image du document'
                passage_text = (
                    f"[{i}] ({score:.2f}) [IMAGE]\n{note_title}\n{passage}\n"
                    f">>> Image : ![{caption}]({image_url}) [{i}]\n"
                )
            else:
                passage_text = f"[{i}] ({score:.2f}) {note_title}\n{passage}\n"
            passages_content.append(passage_text)
        
        system_message["content"] += "\n---\n".join(passages_content)
        system_message["content"] += f"\n\n({len(passages)} passages.)"
    
    return [system_message]


def build_semantic_context(note_results: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les notes pertinentes trouvées par recherche sémantique RAG"""
    # Préprompt court pour limiter les input tokens
    system_message = {
        "role": "system",
        "content": (
            "Expert Technique PROFERM. Réponds uniquement à partir des notes ci-dessous. "
            "Infos absentes (Uw, garantie…) → doc ne le précise pas. "
            "Contradiction → privilégie la source la plus spécifique. Markdown (tableaux, gras), cite le titre de la note.\n\n"
            "NOTES :"
        ),
    }
    
    notes_content = []
    
    for result in note_results:
        note = result['note']
        score = result.get('score', 0.0)
        
        note_text = f"{note.title} ({score:.2f})\n"
        if note.content:
            note_text += f"{note.content}\n"
        note_text += "---\n"
        notes_content.append(note_text)
    
    if notes_content:
        system_message["content"] += "\n".join(notes_content)
        system_message["content"] += f"\n\n({len(note_results)} note(s))."
    else:
        system_message["content"] += "\nAucune note pertinente."
    
    return [system_message]


@router.post("/projects/{project_id}/chat/stream")
async def stream_project_chat_message(
    project_id: int,
    request: ProjectChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Envoyer un message au chatbot (Ollama ou OpenAI) avec streaming et contexte enrichi des passages pertinents du projet"""
    # Charger l'agent si agent_id est fourni
    agent = None
    if request.agent_id:
        agent = session.get(Agent, request.agent_id)
        if not agent or agent.user_id != current_user.id:
            logger.warning(f"Agent {request.agent_id} non trouvé ou non autorisé pour l'utilisateur {current_user.id}")
            agent = None
    
    # Sauvegarder le message utilisateur si conversation_id est fourni
    if request.conversation_id:
        try:
            user_message = Message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.message,
                model=None,
                provider=None
            )
            session.add(user_message)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du message utilisateur: {e}")
    
    # Recherche sémantique RAG AVANCÉE au niveau des passages
    # Analyse TOUTES les notes du projet et retourne les PASSAGES les plus pertinents
    passages = search_relevant_passages(
        session=session,
        project_id=project_id,
        query_text=request.message,
        user_id=current_user.id,
        k=RAG_TOP_K,
        passage_size=500
    )
    
    # Construire le contexte enrichi avec les passages pertinents
    project_context = build_semantic_context_from_passages(passages)
    
    # Préfixer avec la personnalité de l'agent si disponible
    if agent:
        full_context = [{"role": "system", "content": agent.personality}] + project_context
    else:
        full_context = project_context
    
    # Ajouter le contexte de conversation existant si fourni
    if request.context:
        full_context = full_context + request.context
    
    # Ajouter le message utilisateur actuel
    full_context.append({"role": "user", "content": request.message})
    
    # Variable pour accumuler la réponse de l'assistant
    assistant_response = []
    
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
                        # Accumuler la réponse
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                assistant_response.append(data["message"]["content"])
                        except Exception:
                            pass
                        yield f"data: {line}\n\n"
            elif request.provider == "mistral":
                if not settings.MISTRAL_API_KEY:
                    error_msg = "Mistral API key n'est pas configurée"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                # Pas d'API streaming dédiée utilisée ici : on simule le stream à partir de la réponse complète
                response = await mistral_chat("", request.model, full_context)
                content = ""
                if "choices" in response and response["choices"]:
                    content = (response["choices"][0].get("message") or {}).get("content") or ""
                chunk_size = 25
                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"
            else:  # ollama par défaut
                # Passer None comme message car il est déjà dans le contexte
                async for line in ollama_chat_stream("", request.model, full_context):
                    if line.strip():
                        # Accumuler la réponse
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                assistant_response.append(data["message"]["content"])
                        except Exception:
                            pass
                        yield f"data: {line}\n\n"
            
            # Sauvegarder la réponse de l'assistant si conversation_id est fourni
            if request.conversation_id and assistant_response:
                try:
                    complete_response = "".join(assistant_response)
                    assistant_message = Message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=complete_response,
                        model=request.model,
                        provider=request.provider
                    )
                    session.add(assistant_message)
                    
                    # Mettre à jour la date de modification de la conversation
                    conversation = session.get(Conversation, request.conversation_id)
                    if conversation:
                        conversation.updated_at = datetime.utcnow()
                        session.add(conversation)
                    
                    session.commit()
                    logger.info(f"Réponse de l'assistant sauvegardée dans la conversation {request.conversation_id}")
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde de la réponse de l'assistant: {e}")
            
            # Envoyer les sources utilisées pour les citations
            if passages:
                note_ids = list({p.get("note_id") for p in passages if p.get("note_id")})
                notes = (
                    session.exec(select(Note).where(Note.id.in_(note_ids))).all()
                    if note_ids
                    else []
                )
                has_source_file_by_note = {
                    n.id: (n.note_type == "document" and bool(n.source_file_path))
                    for n in notes
                }
                sources_data = []
                for i, p in enumerate(passages, 1):
                    passage_raw = p.get("passage_raw", p.get("passage", ""))
                    excerpt = (passage_raw[:200] + "...") if len(passage_raw) > 200 else passage_raw
                    source_item = {
                        "index": i,
                        "note_id": p.get("note_id"),
                        "note_title": p.get("note_title", "Note sans titre"),
                        "chunk_id": p.get("chunk_id"),
                        "excerpt": excerpt,
                        "passage_full": passage_raw,
                        "score": round(p.get("score", 0.0), 2),
                        "page_no": p.get("page_no"),
                        "section": p.get("section"),
                        "has_source_file": has_source_file_by_note.get(p.get("note_id"), False),
                    }
                    # Multimodal : ajouter les infos image si présentes
                    if p.get("is_image_chunk"):
                        source_item["is_image_chunk"] = True
                        source_item["image_path"] = p.get("image_path")
                        source_item["image_filename"] = p.get("image_filename")
                        source_item["caption"] = p.get("caption", "")
                    sources_data.append(source_item)
                yield f"data: {json.dumps({'sources': sources_data})}\n\n"
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/spaces/{space_id}/chat/stream")
async def stream_space_chat_message(
    space_id: int,
    request: SpaceChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Chat streaming scoped aux documents accessibles dans un espace."""
    # Espace chat: imposer le FastModel configuré en environnement.
    forced_provider = (settings.MODEL_FAST_PROVIDER or "mistral").strip().lower()
    forced_model = (settings.MODEL_FAST_NAME or "mistral-small-latest").strip()

    space = session.get(Space, space_id)
    if not space or space.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Espace non trouvé")

    # Charger l'agent si fourni
    agent = None
    if request.agent_id:
        agent = session.get(Agent, request.agent_id)
        if not agent or agent.user_id != current_user.id:
            agent = None

    if request.conversation_id:
        try:
            user_message = Message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.message,
                model=None,
                provider=None,
            )
            session.add(user_message)
            session.commit()
        except Exception as e:
            logger.error(f"Erreur sauvegarde message utilisateur (space chat): {e}")

    # Retrieval simple lexical sur chunks des documents de l'espace
    terms = [t.strip().lower() for t in request.message.split() if len(t.strip()) >= 3][:8]
    chunk_stmt = (
        select(DocumentChunk, Document.title, Document.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .join(DocumentSpace, DocumentSpace.document_id == Document.id)
        .where(
            DocumentSpace.space_id == space_id,
            Document.user_id == current_user.id,
            DocumentChunk.is_leaf == True,
        )
        .limit(300)
    )
    rows = session.exec(chunk_stmt).all()
    passages = []
    for chunk, document_title, document_id in rows:
        content = (chunk.content or "").strip()
        if not content:
            continue
        lowered = content.lower()
        score = sum(1 for t in terms if t in lowered)
        if terms and score == 0:
            continue
        passages.append({
            "document_id": document_id,
            "document_title": document_title or "Document sans titre",
            "passage": content[:1200],
            "score": float(score),
        })
    passages.sort(key=lambda x: x["score"], reverse=True)
    passages = passages[:RAG_TOP_K]

    rag_system = {
        "role": "system",
        "content": "Tu es LIA. Réponds en t'appuyant uniquement sur les passages fournis pour cet espace."
    }
    if passages:
        rag_system["content"] += "\n\nPASSAGES:\n" + "\n\n---\n\n".join(
            f"[{i+1}] {p['document_title']}\n{p['passage']}" for i, p in enumerate(passages)
        )
    else:
        rag_system["content"] += "\n\nAucun passage trouvé dans cet espace pour cette requête."

    full_context = []
    if agent:
        full_context.append({"role": "system", "content": agent.personality})
    full_context.append(rag_system)
    if request.context:
        full_context.extend(request.context)
    full_context.append({"role": "user", "content": request.message})

    assistant_response: List[str] = []

    async def generate():
        try:
            if forced_provider == "openai":
                if not settings.OPENAI_API_KEY:
                    yield f"data: {json.dumps({'error': 'OpenAI API key non configurée'})}\n\n"
                    return
                async for line in openai_chat_stream("", forced_model, full_context):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                assistant_response.append(data["message"]["content"])
                        except Exception:
                            pass
                        yield f"data: {line}\n\n"
            elif forced_provider == "mistral":
                if not settings.MISTRAL_API_KEY:
                    yield f"data: {json.dumps({'error': 'Mistral API key non configurée'})}\n\n"
                    return
                response = await mistral_chat("", forced_model, full_context)
                content = ""
                if "choices" in response and response["choices"]:
                    content = (response["choices"][0].get("message") or {}).get("content") or ""
                for i in range(0, len(content), 25):
                    chunk = content[i:i+25]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            else:
                async for line in ollama_chat_stream("", request.model, full_context):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                assistant_response.append(data["message"]["content"])
                        except Exception:
                            pass
                        yield f"data: {line}\n\n"

            if request.conversation_id and assistant_response:
                try:
                    complete_response = "".join(assistant_response)
                    assistant_message = Message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=complete_response,
                        model=forced_model,
                        provider=forced_provider,
                    )
                    session.add(assistant_message)
                    conv = session.get(Conversation, request.conversation_id)
                    if conv:
                        conv.updated_at = datetime.utcnow()
                        session.add(conv)
                    session.commit()
                except Exception as e:
                    logger.error(f"Erreur sauvegarde réponse assistant (space chat): {e}")

            if passages:
                sources_data = [
                    {
                        "index": i + 1,
                        "document_id": p["document_id"],
                        "document_title": p["document_title"],
                        "excerpt": (p["passage"][:200] + "...") if len(p["passage"]) > 200 else p["passage"],
                        "score": p["score"],
                    }
                    for i, p in enumerate(passages)
                ]
                yield f"data: {json.dumps({'sources': sources_data})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

