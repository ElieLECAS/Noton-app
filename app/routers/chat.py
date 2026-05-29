from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select
from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.database import get_session, engine
from app.services.mistral_service import (
    MistralRateLimitError,
    chat as mistral_chat,
    chat_stream as mistral_chat_stream,
)
from app.config import settings

async def chat_wrapper(
    message: str,
    model: str,
    context: Optional[List[dict]] = None,
    **kwargs
) -> dict:
    if settings.LLM_PROVIDER == "ollama":
        from app.services.ollama_service import chat as ollama_chat
        return await ollama_chat(message=message, model=model, context=context)
    else:
        return await mistral_chat(message=message, model=model, context=context, **kwargs)

async def chat_stream_wrapper(
    message: str,
    model: str,
    context: Optional[List[dict]] = None,
):
    if settings.LLM_PROVIDER == "ollama":
        from app.services.ollama_service import chat_stream as ollama_chat_stream
        async for chunk in ollama_chat_stream(message=message, model=model, context=context):
            yield chunk
    else:
        async for chunk in mistral_chat_stream(message=message, model=model, context=context):
            yield chunk
from app.services.chat_tools import get_available_tools
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.space import Space
from app.models.document import Document
from app.models.document_space import DocumentSpace
from app.services.space_service import get_space_by_id
from app.tracing import trace_run, trace_pipeline
from datetime import datetime
import json
import logging
import os
import httpx

logger = logging.getLogger(__name__)


def _coerce_positive_int(value) -> Optional[int]:
    """Convertit une valeur en int de page (>0), sinon None."""
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_page_from_passage(passage: dict) -> Optional[int]:
    """Résout la page depuis un passage (page_no, page_start, page_label, page_idx)."""
    return (
        _coerce_positive_int(passage.get("page_no"))
        or _coerce_positive_int(passage.get("page_start"))
        or _coerce_positive_int(passage.get("page_label"))
        or _coerce_positive_int(passage.get("page_idx"))
    )



def _persist_assistant_reply(
    conversation_id: int,
    content: str,
    model: str,
    provider: str,
    sources: Optional[str] = None,
) -> int:
    """Écrit la réponse assistant hors session de la requête (StreamingResponse ferme souvent la session injectée avant la fin du générateur) et retourne son ID."""
    with Session(engine) as s:
        msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            model=model,
            provider=provider,
            sources=sources,
        )
        s.add(msg)
        conv = s.get(Conversation, conversation_id)
        if conv:
            conv.updated_at = datetime.utcnow()
            s.add(conv)
        s.commit()
        s.refresh(msg)
        return msg.id


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


# Nombre de pages ColPali renvoyées au LLM (configurable via RAG_TOP_K).
RAG_TOP_K = _int_env("RAG_TOP_K", 3)
COLPALI_MAX_IMAGES = _int_env("COLPALI_MAX_IMAGES", 3)
SPACE_CHAT_MAX_TOKENS = 1200
SPACE_CHAT_TEMPERATURE = 0.7
SPACE_CHAT_TOP_P = None
SPACE_HISTORY_MAX_CHARS = _int_env("SPACE_HISTORY_MAX_CHARS", 8000)
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"

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
        if settings.LLM_PROVIDER != "ollama" and not settings.MISTRAL_API_KEY:
            raise HTTPException(status_code=400, detail="Mistral API key n'est pas configurée")
        response = await chat_wrapper(request.message, settings.MODEL_FAST, request.context, tools=tools or None)
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

    # Ajouter le contexte existant (limité aux 10 derniers messages pour la performance)
    if request.context:
        full_context.extend(request.context[-10:])
    
    # Variable pour accumuler la réponse de l'assistant
    assistant_response = []
    tools = get_available_tools(include_brave_search=bool(settings.BRAVE_SEARCH_API_KEY))
    use_tools = bool(tools)

    async def generate():
        error_msg_to_yield = None
        try:
            if use_tools:
                # Avec tools : appel non-streaming (boucle tool_calls) puis on simule le stream pour l'UX
                full_messages = full_context + [{"role": "user", "content": request.message}]
                if settings.LLM_PROVIDER != "ollama" and not settings.MISTRAL_API_KEY:
                    error_msg_to_yield = "Mistral API key n'est pas configurée"
                    return
                response = await chat_wrapper("", settings.MODEL_FAST, full_messages, tools=tools)
                content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                # Simuler le streaming par chunks pour garder l'effet de frappe côté client
                chunk_size = 25
                for i in range(0, len(content), chunk_size):
                    chunk = content[i : i + chunk_size]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"
            else:
                if settings.LLM_PROVIDER != "ollama" and not settings.MISTRAL_API_KEY:
                    error_msg_to_yield = "Mistral API key n'est pas configurée"
                    return
                async for raw_chunk in chat_stream_wrapper("", settings.MODEL_FAST, full_context):
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

        except MistralRateLimitError as e:
            logger.warning("Limite de débit Mistral (stream_chat_message): %s", e)
            error_msg_to_yield = str(e)
        except Exception as e:
            logger.exception("Erreur dans le générateur stream_chat_message")
            error_msg_to_yield = str(e)
        
        if error_msg_to_yield:
            yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")



class SpaceChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "mistral"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None


def _truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"


def _sanitize_context_messages(messages: Optional[List[dict]], *, max_messages: int = 10) -> List[dict]:
    if not messages:
        return []
    cleaned: List[dict] = []
    used_chars = 0
    for msg in reversed(messages[-max_messages:]):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in {"system", "user", "assistant"}:
            continue
        content_raw = msg.get("content", "")
        if isinstance(content_raw, str):
            content = content_raw
        else:
            content = json.dumps(content_raw, ensure_ascii=False)
        if not content.strip():
            continue

        remaining = SPACE_HISTORY_MAX_CHARS - used_chars
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = _truncate_text(content, remaining)
        cleaned.append({"role": role, "content": content})
        used_chars += len(content)

    cleaned.reverse()
    return cleaned


def _load_conversation_context(
    session: Session,
    conversation_id: int,
    *,
    max_messages: int = 12,
) -> List[dict]:
    rows = (
        session.exec(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc())
            .limit(max_messages)
        )
        .all()
    )
    rows = list(reversed(rows))
    return _sanitize_context_messages(
        [{"role": r.role, "content": r.content} for r in rows if r.content],
        max_messages=max_messages,
    )


def _render_colpali_page_images(
    session: Session,
    passages: List[dict],
    *,
    max_images: int,
) -> List[str]:
    """Rend les pages PDF ColPali en PNG base64 pour le LLM vision."""
    import base64
    from app.services.multimodal_page_service import render_pdf_page_png

    user_images: List[str] = []
    seen_pages: set = set()

    for passage in passages:
        doc_id = passage.get("document_id")
        page_no = passage.get("page_no") or passage.get("page_start")
        if doc_id is None or page_no is None:
            continue
        page_key = (doc_id, page_no)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        if len(user_images) >= max_images:
            break
        try:
            doc_obj = session.get(Document, doc_id)
            if doc_obj and doc_obj.source_file_path and os.path.exists(doc_obj.source_file_path):
                logger.info("Rendu ColPali page %s du document %s", page_no, doc_id)
                png_bytes = render_pdf_page_png(doc_obj.source_file_path, int(page_no) - 1, dpi=150)
                user_images.append(base64.b64encode(png_bytes).decode("utf-8"))
        except Exception as e:
            logger.error("Erreur rendu page %s (document %s): %s", page_no, doc_id, e)

    return user_images


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

    # Pass 1 : Recherche technique (exclut FAQ correctives)
    with trace_run(
        "technical_retrieval",
        run_type="retriever",
        inputs={"query": request.message, "space_id": space_id, "k": RAG_TOP_K},
        tags=["rag", "technical", "space"],
    ) as retrieval_run:
        from app.services.space_search_service import search_technical_passages
        retrieval = await search_technical_passages(
            session=session,
            space_id=space_id,
            query_text=request.message,
            user_id=current_user.id,
            k=RAG_TOP_K,
        )
        doc_passages = retrieval["passages"]
        retrieval_status = retrieval["status"]
        retrieval_reason = retrieval.get("reason")
        
        retrieval_run.end(outputs={
            "status": retrieval_status,
            "reason": retrieval_reason,
            "nb_passages": len(doc_passages),
            "passages": [
                {
                    "document_title": p.get("document_title"),
                    "chunk_id": p.get("chunk_id"),
                    "score": round(float(p.get("score", 0)), 4),
                    "page_no": p.get("page_no"),
                }
                for p in doc_passages
            ],
        })

    user_images = _render_colpali_page_images(
        session,
        doc_passages,
        max_images=COLPALI_MAX_IMAGES,
    )

    full_context: List[dict] = []
    if request.conversation_id:
        conversation_context = _load_conversation_context(
            session,
            request.conversation_id,
            max_messages=12,
        )
    elif request.context:
        conversation_context = _sanitize_context_messages(request.context, max_messages=10)
    else:
        conversation_context = []

    while conversation_context and conversation_context[-1].get("role") == "user":
        conversation_context.pop()
    full_context.extend(conversation_context)

    user_msg: dict = {"role": "user", "content": request.message}
    if user_images:
        user_msg["images"] = user_images
    full_context.append(user_msg)

    llm_model = (
        settings.VISION_MODEL
        if user_images and settings.LLM_PROVIDER == "mistral"
        else settings.MODEL_FAST
        if user_images and settings.LLM_PROVIDER == "ollama"
        else forced_model
    )

    _pipeline_inputs_space = {
        "query": request.message,
        "space_id": space_id,
        "user_id": current_user.id,
        "model": llm_model,
        "nb_pages": len(doc_passages),
        "nb_images": len(user_images),
    }

    assistant_response: List[str] = []

    async def generate():
        error_msg_to_yield = None
        try:
            if not doc_passages:
                static_reply = (
                    "Aucune page pertinente trouvée dans les documents de cet espace pour cette question."
                )
                chunk_size = 25
                for i in range(0, len(static_reply), chunk_size):
                    chunk = static_reply[i : i + chunk_size]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"

                assistant_message_id = None
                if request.conversation_id:
                    try:
                        assistant_message_id = _persist_assistant_reply(
                            request.conversation_id,
                            static_reply,
                            llm_model,
                            forced_provider,
                            None,
                        )
                    except Exception:
                        logger.exception("Erreur sauvegarde réponse statique assistant (space chat)")

                yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id})}\n\n"
                return

            with trace_pipeline(
                "space_chat_pipeline",
                inputs=_pipeline_inputs_space,
                tags=["chat", "space", "colpali"],
            ) as pipeline_run:
                if settings.LLM_PROVIDER != "ollama" and not settings.MISTRAL_API_KEY:
                    raise ValueError("Mistral API key non configurée")

                final_response = ""
                with trace_run(
                    "colpali_generation",
                    run_type="llm",
                    inputs={
                        "model": llm_model,
                        "nb_images": len(user_images),
                        "nb_pages": len(doc_passages),
                    },
                    tags=["llm", "colpali", "space"],
                ) as gen_run:
                    try:
                        llm_res = await chat_wrapper(
                            "",
                            llm_model,
                            full_context,
                            max_tokens=SPACE_CHAT_MAX_TOKENS,
                            temperature=SPACE_CHAT_TEMPERATURE,
                            top_p=SPACE_CHAT_TOP_P,
                        )
                    except httpx.HTTPStatusError as exc:
                        if exc.response is not None and exc.response.status_code == 400:
                            logger.warning(
                                "LLM 400, fallback question seule (space_id=%s, conv_id=%s)",
                                space_id,
                                request.conversation_id,
                            )
                            fallback_context = [user_msg]
                            llm_res = await chat_wrapper(
                                "",
                                llm_model,
                                fallback_context,
                                max_tokens=SPACE_CHAT_MAX_TOKENS,
                                temperature=SPACE_CHAT_TEMPERATURE,
                                top_p=SPACE_CHAT_TOP_P,
                            )
                        else:
                            raise
                    if "choices" in llm_res and len(llm_res["choices"]) > 0:
                        final_response = llm_res["choices"][0]["message"].get("content", "").strip()
                    gen_run.end(outputs={"response_chars": len(final_response)})

                if not final_response:
                    final_response = "Je n'ai pas pu générer de réponse."

                chunk_size = 25
                for i in range(0, len(final_response), chunk_size):
                    chunk = final_response[i : i + chunk_size]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"

                pipeline_run.end(outputs={
                    "nb_pages": len(doc_passages),
                    "nb_images": len(user_images),
                    "response_chars": len(final_response),
                })

                sources_data = []
                if doc_passages:
                    doc_ids = list({p.get("document_id") for p in doc_passages if p.get("document_id")})
                    with Session(engine) as src_session:
                        docs = (
                            src_session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
                            if doc_ids
                            else []
                        )
                        has_file_by_doc = {
                            d.id: (d.document_type == "document" and bool(d.source_file_path))
                            for d in docs
                        }

                    for i, p in enumerate(doc_passages):
                        did = p.get("document_id")
                        resolved_page = _resolve_page_from_passage(p)
                        sources_data.append(
                            {
                                "index": i + 1,
                                "document_id": did,
                                "document_title": p.get("document_title"),
                                "chunk_id": p.get("chunk_id"),
                                "chunk_index": p.get("chunk_index"),
                                "excerpt": f"Page {resolved_page}" if resolved_page else "",
                                "passage_full": "",
                                "score": round(float(p.get("score", 0)), 2),
                                "page_no": resolved_page,
                                "page_start": p.get("page_start"),
                                "page_end": p.get("page_end"),
                                "has_source_file": has_file_by_doc.get(did, False),
                            }
                        )

                assistant_message_id = None
                if request.conversation_id and assistant_response:
                    try:
                        complete_response = "".join(assistant_response)
                        sources_json = json.dumps(sources_data) if sources_data else None
                        assistant_message_id = _persist_assistant_reply(
                            request.conversation_id,
                            complete_response,
                            llm_model,
                            forced_provider,
                            sources_json,
                        )
                    except Exception:
                        logger.exception("Erreur sauvegarde réponse assistant (space chat)")

                if sources_data:
                    yield f"data: {json.dumps({'sources': sources_data})}\n\n"
                yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id})}\n\n"

        except MistralRateLimitError as e:
            logger.warning("Limite de débit Mistral (stream_space_chat_message): %s", e)
            error_msg_to_yield = str(e)
        except Exception as e:
            logger.exception("Erreur dans le générateur stream_space_chat_message")
            error_msg_to_yield = str(e)

        if error_msg_to_yield:
            yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

