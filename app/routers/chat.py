from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.database import get_session, engine
from app.services.mistral_service import (
    chat as mistral_chat,
    chat_stream as mistral_chat_stream,
)
from app.config import settings
from app.services.semantic_search_service import search_relevant_notes, search_relevant_passages
from app.services.chat_tools import get_available_tools
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.note import Note
from app.models.space import Space
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.space_search_service import search_relevant_passages as search_space_passages
from app.services.space_service import get_space_by_id
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


def _persist_assistant_reply(
    conversation_id: int,
    content: str,
    model: str,
    provider: str,
) -> None:
    """Écrit la réponse assistant hors session de la requête (StreamingResponse ferme souvent la session injectée avant la fin du générateur)."""
    with Session(engine) as s:
        s.add(
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                model=model,
                provider=provider,
            )
        )
        conv = s.get(Conversation, conversation_id)
        if conv:
            conv.updated_at = datetime.utcnow()
            s.add(conv)
        s.commit()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


# Nombre de passages RAG renvoyés au LLM (configurable via RAG_TOP_K).
# Défaut 8 : avec 1 passage, le modèle comble avec des généralisations faux catalogue (tableaux inventés, ✓/✗).
RAG_TOP_K = _int_env("RAG_TOP_K", 8)
# Paramétrage en dur du chat "espaces"
SPACE_CHAT_MAX_TOKENS = 3000
SPACE_CHAT_TEMPERATURE = 0.1
SPACE_CHAT_TOP_P = None
SPACE_CHAT_SYSTEM_PROMPT = (
    "Tu es LIA, assistante PROFERM pour les collaborateurs et les clients. "
    "Tu reponds en francais, avec un ton humain, professionnel, clair et orienté solution. "
    "Tu donnes des reponses directes, concretes et operationnelles, sans mentionner le fonctionnement interne de recherche ni la provenance des informations. "
    "Tu peux utiliser des tableaux Markdown quand cela aide la lisibilite (comparatif, synthese, options, dimensions, performances, disponibilite). "
    "N'invente jamais de caracteristique, prix, delai, compatibilite ou disponibilite. "
    "Si une information n'est pas disponible ou reste incertaine, dis-le simplement en une phrase courte et propose la meilleure action de verification. "
    "Evite les formulations defensives, les avertissements inutiles et les redites. "
    "Structure par defaut en sections courtes avec listes a puces, et termine par une question utile uniquement si cela fait avancer l'echange."
)

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "mistral"  # Conservé pour compatibilité, ignoré (modèle fast unique)
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None  # ID de la conversation (optionnel pour compatibilité)


@router.get("/providers/models")
async def list_all_models():
    """Retourne le modèle chat unique configuré."""
    return {
        "mistral": [settings.MODEL_FAST],
    }


@router.post("/chat")
async def send_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user)
):
    """Envoyer un message au chatbot avec le modèle fast unique."""
    try:
        tools = get_available_tools(include_brave_search=bool(settings.BRAVE_SEARCH_API_KEY))
        if not settings.MISTRAL_API_KEY:
            raise HTTPException(status_code=400, detail="Mistral API key n'est pas configurée")
        response = await mistral_chat(request.message, settings.MODEL_FAST, request.context, tools=tools or None)
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"].get("content", "")
            return {"message": {"content": content}}
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel au modèle fast: {str(e)}")


@router.post("/chat/stream")
async def stream_chat_message(
    request: ChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Envoyer un message au chatbot avec streaming (modèle fast unique)."""
    
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
    
    # Construire le contexte.
    full_context = []

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
                if not settings.MISTRAL_API_KEY:
                    error_msg = "Mistral API key n'est pas configurée"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                response = await mistral_chat("", settings.MODEL_FAST, full_messages, tools=tools)
                content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                # Simuler le streaming par chunks pour garder l'effet de frappe côté client
                chunk_size = 25
                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"
            else:
                if not settings.MISTRAL_API_KEY:
                    error_msg = "Mistral API key n'est pas configurée"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return
                async for raw_chunk in mistral_chat_stream("", settings.MODEL_FAST, full_context):
                    try:
                        parsed = json.loads(raw_chunk)
                    except json.JSONDecodeError:
                        continue
                    chunk = (parsed.get("message") or {}).get("content") or ""
                    if not chunk:
                        continue
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"

            if request.conversation_id and assistant_response:
                try:
                    _persist_assistant_reply(
                        request.conversation_id,
                        "".join(assistant_response),
                        settings.MODEL_FAST,
                        "mistral",
                    )
                    logger.info(
                        "Réponse de l'assistant sauvegardée dans la conversation %s",
                        request.conversation_id,
                    )
                except Exception:
                    logger.exception(
                        "Erreur lors de la sauvegarde de la réponse de l'assistant"
                    )

            yield f"data: {json.dumps({'done': True})}\n\n"
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


class ProjectChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "mistral"  # Conservé pour compatibilité, ignoré
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None  # ID de la conversation (optionnel pour compatibilité)


class SpaceChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "mistral"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None


def build_space_context_from_passages(passages: List[dict]) -> dict:
    """
    Construit le contexte système à partir des passages RAG + KAG rerankés.
    Format unifié pour le LLM (comme build_semantic_context_from_passages).
    """
    system_message = {
        "role": "system",
        "content": SPACE_CHAT_SYSTEM_PROMPT,
    }

    if passages:
        system_message["content"] += "\n\nPASSAGES :\n\n"
        passages_content = []
        for i, passage_data in enumerate(passages, 1):
            passage = passage_data['passage']
            score = passage_data.get('score', 0.0)
            document_title = passage_data.get('document_title', 'Document sans titre')
            passage_text = f"[{i}] ({score:.2f}) {document_title}\n{passage}\n"
            passages_content.append(passage_text)
        system_message["content"] += "\n---\n".join(passages_content)
        system_message["content"] += f"\n\n({len(passages)} passages.)"
    else:
        system_message["content"] += "\n\nAucun passage trouvé dans cet espace pour cette requête."

    return system_message


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
    """Envoyer un message au chatbot avec streaming et contexte enrichi des passages pertinents du projet."""
    
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
            if not settings.MISTRAL_API_KEY:
                error_msg = "Mistral API key n'est pas configurée"
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                return
            async for raw_chunk in mistral_chat_stream("", settings.MODEL_FAST, full_context):
                try:
                    parsed = json.loads(raw_chunk)
                except json.JSONDecodeError:
                    continue
                chunk = (parsed.get("message") or {}).get("content") or ""
                if not chunk:
                    continue
                assistant_response.append(chunk)
                yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"

            if request.conversation_id and assistant_response:
                try:
                    _persist_assistant_reply(
                        request.conversation_id,
                        "".join(assistant_response),
                        settings.MODEL_FAST,
                        "mistral",
                    )
                    logger.info(
                        "Réponse de l'assistant sauvegardée dans la conversation %s",
                        request.conversation_id,
                    )
                except Exception:
                    logger.exception(
                        "Erreur lors de la sauvegarde de la réponse de l'assistant (projet)"
                    )

            # Envoyer les sources utilisées pour les citations
            if passages:
                note_ids = list({p.get("note_id") for p in passages if p.get("note_id")})
                with Session(engine) as src_session:
                    notes = (
                        src_session.exec(select(Note).where(Note.id.in_(note_ids))).all()
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

            yield f"data: {json.dumps({'done': True})}\n\n"
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
    # Espace chat: imposer le modèle fast unique configuré.
    forced_provider = "mistral"
    forced_model = settings.MODEL_FAST

    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(status_code=404, detail="Espace non trouvé")

    if request.conversation_id:
        conversation = session.get(Conversation, request.conversation_id)
        if (
            not conversation
            or conversation.user_id != current_user.id
            or conversation.space_id != space_id
        ):
            raise HTTPException(status_code=404, detail="Conversation non trouvée")
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

    # Recherche sémantique RAG + KAG complète (identique au pipeline projet)
    passages = search_space_passages(
        session=session,
        space_id=space_id,
        query_text=request.message,
        user_id=current_user.id,
        k=RAG_TOP_K,
    )

    # Construire le contexte système à partir des passages rerankés
    space_context = build_space_context_from_passages(passages)

    full_context = []
    full_context.append(space_context)
    if request.context:
        full_context.extend(request.context)
    full_context.append({"role": "user", "content": request.message})

    assistant_response: List[str] = []

    async def generate():
        try:
            if not settings.MISTRAL_API_KEY:
                yield f"data: {json.dumps({'error': 'Mistral API key non configurée'})}\n\n"
                return
            async for raw_chunk in mistral_chat_stream(
                "",
                forced_model,
                full_context,
                max_tokens=SPACE_CHAT_MAX_TOKENS,
                temperature=SPACE_CHAT_TEMPERATURE,
                top_p=SPACE_CHAT_TOP_P,
            ):
                try:
                    parsed = json.loads(raw_chunk)
                except json.JSONDecodeError:
                    continue
                chunk = (parsed.get("message") or {}).get("content") or ""
                if not chunk:
                    continue
                assistant_response.append(chunk)
                yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"
            # Persister et envoyer les sources avant `done` : le client peut annuler la lecture
            # dès `done`, ce qui coupait le générateur avant commit / événements suivants.
            if request.conversation_id and assistant_response:
                try:
                    complete_response = "".join(assistant_response)
                    _persist_assistant_reply(
                        request.conversation_id,
                        complete_response,
                        forced_model,
                        forced_provider,
                    )
                    logger.info(
                        "Réponse assistant sauvegardée (space chat), conversation %s",
                        request.conversation_id,
                    )
                except Exception:
                    logger.exception("Erreur sauvegarde réponse assistant (space chat)")

            if passages:
                sources_data = [
                    {
                        "index": i + 1,
                        "document_id": p["document_id"],
                        "document_title": p["document_title"],
                        "excerpt": (p["passage_raw"][:200] + "...") if len(p.get("passage_raw", "")) > 200 else p.get("passage_raw", p.get("passage", "")),
                        "score": round(p["score"], 2),
                        "page_no": p.get("page_no"),
                        "section": p.get("section"),
                    }
                    for i, p in enumerate(passages)
                ]
                yield f"data: {json.dumps({'sources': sources_data})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

