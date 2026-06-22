from typing import List, Optional, Literal
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
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
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.space_search_service import search_relevant_passages as search_space_passages
from app.services.space_service import get_space_by_id
from app.services.illustration_service import determine_and_crop_illustration, has_significant_visuals
from app.tracing import trace_run, trace_pipeline
from datetime import datetime
import json
import logging
import os
import httpx
import re

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


def _resolve_page_from_chunk(chunk: DocumentChunk) -> Optional[int]:
    """
    Résout la page d'un chunk en fusionnant metadata_json + metadata_.
    metadata_json prime mais on garde le fallback legacy.
    """
    merged_meta = {}
    if isinstance(chunk.metadata_, dict):
        merged_meta.update(chunk.metadata_)
    if isinstance(chunk.metadata_json, dict):
        merged_meta.update(chunk.metadata_json)
    return (
        _coerce_positive_int(merged_meta.get("page_no"))
        or _coerce_positive_int(merged_meta.get("page_start"))
        or _coerce_positive_int(merged_meta.get("page_label"))
        or _coerce_positive_int(merged_meta.get("page_idx"))
    )


def _persist_assistant_reply(
    conversation_id: int,
    content: str,
    model: str,
    provider: str,
    sources: Optional[str] = None,
    metadata_json: Optional[dict] = None,
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
            metadata_json=metadata_json,
        )
        s.add(msg)
        conv = s.get(Conversation, conversation_id)
        if conv:
            conv.updated_at = datetime.utcnow()
            s.add(conv)
        s.commit()
        s.refresh(msg)
        return msg.id


def _save_conversation_query_context(conversation_id: int, query_context: dict) -> None:
    with Session(engine) as s:
        conv = s.get(Conversation, conversation_id)
        if conv:
            conv.query_context = query_context
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


# Nombre de passages RAG renvoyés au LLM (configurable via RAG_TOP_K / settings).
RAG_TOP_K = _int_env("RAG_TOP_K", settings.RAG_TOP_K)
# Paramétrage en dur du chat "espaces"
SPACE_CHAT_MAX_TOKENS = 1200
SPACE_CHAT_TEMPERATURE = 0.0
SPACE_CHAT_TOP_P = None
SPACE_CONTEXT_MAX_CHARS = _int_env("SPACE_CONTEXT_MAX_CHARS", 18000)
SPACE_CONTEXT_MAX_PASSAGE_CHARS = _int_env(
    "SPACE_CONTEXT_MAX_PASSAGE_CHARS", settings.SPACE_CONTEXT_MAX_PASSAGE_CHARS
)
SPACE_HISTORY_MAX_CHARS = _int_env("SPACE_HISTORY_MAX_CHARS", 8000)
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"
SPACE_CHAT_SYSTEM_PROMPT = (
    "Tu es LIA, l'assistante experte de PROFERM. Ton rôle est d'accompagner les collaborateurs et les clients de manière chaleureuse, professionnelle et précise sur nos produits et services.\n"
    "Identité : Tu parles au nom de PROFERM. Quand tu dis 'nous' ou 'nos gammes', tu fais référence aux produits PROFERM. Les documents des fournisseurs (Technal, Profine, Askey, Roto, etc.) concernent nos partenaires et doivent être présentés comme tels.\n"
    "Ton & Style de discussion : Réponds sous forme de discussion fluide, naturelle et en prose. Privilégie une vraie conversation chaleureuse plutôt que d'aligner systématiquement des listes. Sois agréable dans tes échanges. Salue courtoisement l'utilisateur si c'est le début de la conversation, mais supprime tout texte superflu (évite les formules de salutation répétées ou de politesse de fin systématiques).\n"
    "Concision stricte : Limite drastiquement la longueur de tes réponses. Réponds de manière extrêmement directe, concise et minimaliste. Si la question demande une pièce, un outil ou une valeur, donne uniquement cette information avec sa citation, sans rajouter les étapes de montage adjacentes ni d'autres détails techniques non sollicités.\n"
    "Puces & Tableaux : Priorise la prose. N'utilise les listes à puces que si c'est réellement justifié (par exemple pour énumérer des éléments simples où la prose nuirait à la lisibilité). Utilise les tableaux Markdown pour présenter clairement les données techniques ou les comparaisons complexes sans répéter ou paraphraser les informations du tableau dans le texte qui l'accompagne.\n"
    "Filtrage des informations : Réponds exclusivement à la question posée. Si l'information n'est pas dans le chunk spécifique à la section demandée, ne complète pas avec des données d'autres sections. Réponds exactement au périmètre de la question posée sans proposer d'informations complémentaires non sollicitées.\n"
    "Désambiguïsation & Contextualisation automatique : Sois extrêmement vigilante avec les dénominations de gammes (ex : Perform 70 vs Perform 76), les versions de produits (ex : standard vs renforcée) et les configurations spécifiques (ex : seuil PMR vs seuil standard). Ne les confonds jamais et ne mélange pas leurs composants ou instructions. Si une information ou un composant varie selon la gamme, la version ou la configuration, présente systématiquement et automatiquement la distinction ou les différents cas de figure applicables selon les données du contexte, sans demander de précision ou de clarification à l'utilisateur.\n"
    "Citations strictes et obligatoires : Pour chaque fait technique, mesure, tolérance ou instruction que tu mentionnes, cite obligatoirement le nom exact du document et son numéro de page sous la forme [Nom du document, page X] (par exemple : [Notice de pose LUMEAL GA, page 8]). Si l'extrait ne contient pas de numéro de page précis, mentionne simplement le nom du document [Nom du document]. N'invente jamais de numéros de pages ou de noms de documents.\n"
    "Interdiction d'halluciner, de surinterpréter et d'assembler des informations : Ne fais aucune extrapolation, supposition, spéculation ou généralisation. Ne cherche pas à deviner ou à enjoliver. Ne combine/colle JAMAIS des références de produits (ex: T141019), des cotes (ex: 300 mm) ou des dimensions (ex: 2.40 m) issues de phrases ou de sections différentes pour fabriquer une spécification qui n'est pas explicitement écrite telle quelle. Si le texte ne contient pas l'association directe et exacte demandée pour le composant spécifique, réponds obligatoirement : 'La notice ne précise pas [la mesure ou la spécification] pour cette pièce' au lieu d'extrapoler ou de proposer des valeurs standards du bâtiment.\n"
    "### RÈGLE DE SÉCURITÉ STRICTE : GROUNDING TECHNIQUE ET GESTES\n"
    "- Interdiction absolue d'enrichir, d'interpréter, de paraphraser ou d'extrapoler les faits, valeurs numériques, cinématiques, gestes techniques ou étapes de montage (ex: imaginer des angles, rotations, clics) à partir de tes propres connaissances ou de ta propre interprétation.\n"
    "- Si une consigne technique, une cote ou une étape de montage est demandée, tu dois restituer STRICTEMENT et MOT POUR MOT les verbes d'action et les composants textuels fournis dans les chunks (ex: 'Mettre en contact', 'Clipper l'autre côté').\n"
    "- En l'absence de détails explicites et exacts dans le contexte, n'invente rien, refuse d'extrapoler et dis : 'La notice ne précise pas [ce détail]'. Privilégie une concision totale plutôt que du jargon métier extrapolé.\n"
    "Règle d'or : Hard Grounding strict. Tu dois te limiter exclusivement aux faits décrits de manière explicite dans le contexte fourni (les PASSAGES) et à leurs liaisons directes. Si l'information recherchée est absente du contexte ou incertaine, indique-le clairement et propose une étape de vérification sans essayer de deviner.\n"
    "### RÉSOLUTION DES CONFLITS ET SPÉCIFICITÉ DES SOURCES :\n"
    "- En cas de contradictions ou d'informations divergentes entre plusieurs passages (ex. dimensions d'outils ou instructions différentes), privilégie toujours le passage issu du document le plus spécifique au sujet de la question (ex. la notice dédiée au 'report de charge' pour une question sur le report de charge, plutôt qu'une notice de montage générale). Les passages sont classés par ordre de pertinence décroissante."
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


@router.get("/chats/illustrations/{filename}")
async def get_illustration(filename: str):
    """Sert une image d'illustration détourée depuis le cache."""
    cache_dir = os.path.join(os.path.dirname(settings.LANCED_DB_DIR), "illustration_cache")
    file_path = os.path.join(cache_dir, filename)
    
    # Sécurité basique contre le path traversal
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(status_code=400, detail="Nom de fichier invalide")
        
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Illustration non trouvée")
        
    return FileResponse(file_path, media_type="image/png")


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



class SlotActionRequest(BaseModel):
    field: str
    value: str = ""
    action: Literal["fill", "skip"] = "fill"


class SpaceChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "mistral"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None
    slot_action: Optional[SlotActionRequest] = None


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
        used_chars = 0
        for i, passage_data in enumerate(passages, 1):
            passage = str(passage_data.get("passage") or "")
            if not passage:
                continue
            passage = _truncate_text(passage, SPACE_CONTEXT_MAX_PASSAGE_CHARS)
            score = passage_data.get('score', 0.0)
            document_title = passage_data.get('document_title', 'Document sans titre')
            
            page_no = passage_data.get("page_no")
            page_start = passage_data.get("page_start")
            page_end = passage_data.get("page_end")
            page_info = ""
            if page_no is not None:
                page_info = f", page {page_no}"
            elif page_start is not None:
                if page_end is not None and page_end != page_start:
                    page_info = f", pages {page_start}-{page_end}"
                else:
                    page_info = f", page {page_start}"
            
            passage_text = f"[{i}] ({score:.2f}) {document_title}{page_info}\n{passage}\n"
            if used_chars + len(passage_text) > SPACE_CONTEXT_MAX_CHARS:
                break
            passages_content.append(passage_text)
            used_chars += len(passage_text)
        system_message["content"] += "\n---\n".join(passages_content)
        system_message["content"] += f"\n\n({len(passages_content)} passages.)"
    else:
        system_message["content"] += "\n\nAucun passage trouvé dans cet espace pour cette requête."

    return system_message

@router.post("/spaces/{space_id}/chat/stream")
async def stream_space_chat_message(
    space_id: int,
    request: SpaceChatRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Chat streaming scoped aux documents accessibles dans un espace."""
    logger.info(
        "Chat espace démarré — space_id=%s user_id=%s message_len=%d",
        space_id,
        current_user.id,
        len(request.message or ""),
    )
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

    conversation_context: List[dict] = []
    if request.conversation_id:
        conversation_context = _load_conversation_context(
            session,
            request.conversation_id,
            max_messages=12,
        )
    elif request.context:
        conversation_context = _sanitize_context_messages(request.context, max_messages=10)

    retrieval_queries = None
    rag_user_message = request.message
    lw_result = None

    if settings.QUERY_UNDERSTANDING_ENABLED:
        logger.info("[chat] Étape 1/5 — lightweight query understanding")
        from app.services.lightweight_query_understanding import run_lightweight_understanding

        persisted_context = None
        if request.conversation_id:
            conv_for_ctx = session.get(Conversation, request.conversation_id)
            if conv_for_ctx:
                persisted_context = conv_for_ctx.query_context

        with trace_run(
            "lightweight_query_understanding",
            run_type="chain",
            inputs={"query": request.message, "space_id": space_id},
            tags=["query_understanding", "lightweight", "space"],
        ) as qu_run:
            lw_result = await run_lightweight_understanding(
                user_message=request.message,
                history=conversation_context,
                session=session,
                persisted_context=persisted_context,
            )
            qu_run.end(outputs={
                "route": lw_result.route,
                "ready_for_retrieval": lw_result.ready_for_retrieval,
                "signals": lw_result.signals.model_dump() if lw_result.signals else None,
            })

        if request.conversation_id and lw_result.query_context:
            _save_conversation_query_context(request.conversation_id, lw_result.query_context)

        routing_decision_decision = lw_result.route
    else:
        logger.info("[chat] Étape 1/4 — routage requête (Mistral)")
        from app.services.query_reasoning_service import decide_retrieval_route

        with trace_run(
            "query_routing",
            run_type="chain",
            inputs={"query": request.message, "space_id": space_id},
            tags=["routing", "space"],
        ) as routing_run:
            routing_decision = await decide_retrieval_route(request.message)
            routing_run.end(outputs={
                "decision": routing_decision.decision,
                "reasoning": routing_decision.reasoning,
            })
            logger.info(
                "Query routing decision: %s (reason: %s)",
                routing_decision.decision,
                routing_decision.reasoning,
            )
        routing_decision_decision = routing_decision.decision

    if routing_decision_decision == "direct":
        direct_context = list(conversation_context)
        # Éliminer tout message utilisateur en suspens à la fin de l'historique
        while direct_context and direct_context[-1].get("role") == "user":
            direct_context.pop()

        system_message = {
            "role": "system",
            "content": SPACE_CHAT_SYSTEM_PROMPT + "\n\n(Note : Aucune recherche documentaire n'est nécessaire pour ce message. Réponds de manière polie et directe à la requête de l'utilisateur.)",
        }
        
        full_context_draft = [system_message]
        full_context_draft.extend(direct_context)
        full_context_draft.append({"role": "user", "content": request.message})

        _pipeline_inputs_space = {
            "query": request.message,
            "space_id": space_id,
            "user_id": current_user.id,
            "model": forced_model,
            "routing": "direct",
        }

        assistant_response = []

        async def generate_direct():
            error_msg_to_yield = None
            try:
                with trace_pipeline(
                    "space_chat_pipeline_direct",
                    inputs=_pipeline_inputs_space,
                    tags=["chat", "space", "direct"],
                ) as pipeline_run:
                    if settings.LLM_PROVIDER != "ollama" and not settings.MISTRAL_API_KEY:
                        raise ValueError("Mistral API key non configurée")

                    with trace_run(
                        "stream_generation_direct",
                        run_type="llm",
                        inputs={
                            "model": forced_model,
                            "messages": [
                                {"role": m.get("role"), "content": str(m.get("content", ""))}
                                for m in full_context_draft
                            ]
                        },
                        tags=["llm", "stream", "space", "direct"]
                    ) as stream_run:
                        async for raw_chunk in chat_stream_wrapper(
                            message="",
                            model=forced_model,
                            context=full_context_draft,
                        ):
                            try:
                                parsed = json.loads(raw_chunk)
                            except json.JSONDecodeError:
                                continue
                            content = (parsed.get("message") or {}).get("content") or ""
                            if not content:
                                continue
                            assistant_response.append(content)
                            yield f"data: {json.dumps({'message': {'content': content}})}\n\n"
                        
                        final_response = "".join(assistant_response)
                        stream_run.end(outputs={"response": final_response})

                    pipeline_run.end(outputs={
                        "response_chars": len(final_response),
                    })

                    assistant_message_id = None
                    if request.conversation_id and assistant_response:
                        try:
                            assistant_message_id = _persist_assistant_reply(
                                request.conversation_id,
                                final_response,
                                forced_model,
                                forced_provider,
                                None,
                            )
                            logger.info(
                                "Réponse assistant directe sauvegardée (space chat), conversation %s",
                                request.conversation_id,
                            )
                        except Exception:
                            logger.exception("Erreur sauvegarde réponse directe assistant (space chat)")

                    yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id})}\n\n"

            except MistralRateLimitError as e:
                logger.warning("Limite de débit Mistral (stream_space_chat_message direct): %s", e)
                error_msg_to_yield = str(e)
            except Exception as e:
                logger.exception("Erreur dans le générateur stream_space_chat_message (direct)")
                error_msg_to_yield = str(e)
            
            if error_msg_to_yield:
                yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"

        return StreamingResponse(generate_direct(), media_type="text/event-stream")

    if settings.QUERY_UNDERSTANDING_ENABLED and lw_result and not lw_result.ready_for_retrieval:
        clarification = lw_result.clarification
        if clarification:
            logger.info(
                "[chat] Clarification — phase=%s vague=%s (pas de RAG)",
                clarification.phase,
                clarification.phase == "awaiting_vague_clarification",
            )

            async def generate_clarification():
                question = clarification.question
                yield f"data: {json.dumps({'message': {'content': question}})}\n\n"

                assistant_message_id = None
                if request.conversation_id:
                    try:
                        assistant_message_id = _persist_assistant_reply(
                            request.conversation_id,
                            question,
                            forced_model,
                            forced_provider,
                            None,
                            metadata_json={"clarification_type": "vague_request"},
                        )
                    except Exception:
                        logger.exception("Erreur sauvegarde clarification assistant (space chat)")

                yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id})}\n\n"

            return StreamingResponse(generate_clarification(), media_type="text/event-stream")

    if settings.QUERY_UNDERSTANDING_ENABLED and lw_result and lw_result.ready_for_retrieval:
        retrieval_queries = lw_result.retrieval_queries
        rag_user_message = (
            lw_result.query_context.get("enriched_user_message")
            or lw_result.query_context.get("original_user_message")
            or request.message
        )

    step_label = "2/5" if settings.QUERY_UNDERSTANDING_ENABLED else "2/4"
    logger.info("[chat] Étape %s — retrieval hybride (ColPali + pgvector + BM25)", step_label)
    with trace_run(
        "technical_retrieval",
        run_type="retriever",
        inputs={
            "query": rag_user_message,
            "space_id": space_id,
            "k": RAG_TOP_K,
            "retrieval_queries": retrieval_queries.model_dump() if retrieval_queries else None,
        },
        tags=["rag", "technical", "space"],
    ) as retrieval_run:
        from app.services.space_search_service import search_technical_passages
        retrieval = await search_technical_passages(
            session=session,
            space_id=space_id,
            query_text=rag_user_message,
            user_id=current_user.id,
            k=RAG_TOP_K,
            queries=retrieval_queries,
        )
        doc_passages = retrieval["passages"]
        retrieval_status = retrieval["status"]
        retrieval_reason = retrieval.get("reason")
        retrieval_images = retrieval.get("images") or []
        dynamic_k = retrieval.get("dynamic_k")
        rerank_status = retrieval.get("rerank_status")
        
        retrieval_run.end(outputs={
            "status": retrieval_status,
            "reason": retrieval_reason,
            "dynamic_k": dynamic_k,
            "rerank_status": rerank_status,
            "nb_passages": len(doc_passages),
            "passages": [
                {
                    "document_title": p.get("document_title"),
                    "chunk_id": p.get("chunk_id"),
                    "score": round(float(p.get("score", 0)), 4),
                    "page_no": p.get("page_no"),
                    "section": p.get("section"),
                    "passage_preview": (p.get("passage_raw") or p.get("passage", ""))[:300],
                }
                for p in doc_passages
            ],
        })

    from app.services.rag_generation_service import (
        enrich_colpali_passages_with_pymupdf,
        is_vision_model,
        render_page_images_for_passages_async,
        build_rag_user_message,
    )

    doc_passages = enrich_colpali_passages_with_pymupdf(session, doc_passages)

    if settings.QUERY_UNDERSTANDING_ENABLED and lw_result and lw_result.signals:
        from app.services.retrieval_boost_service import apply_soft_boosts_to_passages

        doc_passages = apply_soft_boosts_to_passages(
            session=session,
            passages=doc_passages,
            signals=lw_result.signals,
        )

    if settings.QUERY_UNDERSTANDING_ENABLED and lw_result and lw_result.signals.primary_source:
        from app.services.query_reasoning_service import QueryIntent
        from app.services.space_search_service import refine_with_source_authority

        intent_obj = QueryIntent(
            intent=lw_result.signals.intent or "generic",
            primary_source=lw_result.signals.primary_source,
            reasoning="lightweight extraction",
            confidence=lw_result.signals.confidence,
        )
        doc_passages = refine_with_source_authority(doc_passages, rag_user_message, intent_obj)

    logger.info(
        "[chat] Étape %s — contexte RAG (%d passages, status=%s, dynamic_k=%s, rerank=%s)",
        "3/5" if settings.QUERY_UNDERSTANDING_ENABLED else "3/4",
        len(doc_passages),
        retrieval_status,
        dynamic_k,
        rerank_status,
    )

    # Construire le contexte système à partir des passages techniques
    # Si low confidence : injecter un prompt spécial pour forcer la clarification
    if retrieval_status == "low_confidence_clarification":
        space_context_draft = build_space_context_from_passages(doc_passages)
        # Ajouter une instruction de clarification forcée après les passages
        space_context_draft["content"] += (
            "\n\n⚠️ IMPORTANT : Les passages ci-dessus sont ambigus ou de faible pertinence. "
            "Ne déduis PAS de réponse définitive. Tu dois poser à l'utilisateur une question "
            "précise de clarification basée uniquement sur le contenu de ces 1-2 passages."
        )
        logger.info(
            "Low confidence détectée : prompt forcé à demander clarification (status=%s, reason=%s)",
            retrieval_status,
            retrieval_reason,
        )
    else:
        space_context_draft = build_space_context_from_passages(doc_passages)

    full_context_draft = []
    full_context_draft.append(space_context_draft)

    rag_history = list(conversation_context)
    while rag_history and rag_history[-1].get("role") == "user":
        rag_history.pop()

    full_context_draft.extend(rag_history)

    user_images: List[str] = []
    if doc_passages and is_vision_model(forced_model):
        if retrieval_images:
            user_images = retrieval_images[: settings.RAG_MAX_IMAGES]
            logger.info(
                "[stream_space_chat_message] %d image(s) PNG du pipeline multimodal pour %s",
                len(user_images),
                forced_model,
            )
        else:
            user_images = await render_page_images_for_passages_async(
                session,
                doc_passages,
                max_pages=settings.RAG_MAX_IMAGES,
                needs_image_only=not settings.RAG_RENDER_ALL_IMAGES,
            )
            logger.info(
                "[stream_space_chat_message] %d image(s) PNG rendues (legacy) pour %s",
                len(user_images),
                forced_model,
            )
    elif doc_passages:
        logger.info(
            "[stream_space_chat_message] Pas d'images (modèle non vision: %s)",
            forced_model,
        )

    user_msg = build_rag_user_message(
        rag_user_message,
        images_b64=user_images or None,
    )
    full_context_draft.append(user_msg)

    logger.info(
        "[chat] Étape %s — génération réponse stream (model=%s)",
        "4/5" if settings.QUERY_UNDERSTANDING_ENABLED else "4/4",
        forced_model,
    )

    _pipeline_inputs_space = {
        "query": rag_user_message,
        "space_id": space_id,
        "user_id": current_user.id,
        "model": forced_model,
        "nb_doc_passages": len(doc_passages),
    }

    assistant_response: List[str] = []

    async def generate():
        error_msg_to_yield = None
        try:
            if not doc_passages:
                static_reply = "Je ne trouve pas de réponse à votre question dans les documents disponibles dans cet espace car aucune source n'est jugée suffisamment pertinente (seuil minimum de 75%)."
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
                            forced_model,
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
                tags=["chat", "space", "rag", "kag"],
            ) as pipeline_run:
                if settings.LLM_PROVIDER != "ollama" and not settings.MISTRAL_API_KEY:
                    raise ValueError("Mistral API key non configurée")

                with trace_run(
                    "stream_generation",
                    run_type="llm",
                    inputs={
                        "model": forced_model,
                        "messages": [
                            {"role": m.get("role"), "content": str(m.get("content", ""))}
                            for m in full_context_draft
                        ]
                    },
                    tags=["llm", "stream", "space"]
                ) as stream_run:
                    fallback_triggered = False
                    try:
                        async for raw_chunk in chat_stream_wrapper(
                            message="",
                            model=forced_model,
                            context=full_context_draft,
                        ):
                            try:
                                parsed = json.loads(raw_chunk)
                            except json.JSONDecodeError:
                                continue
                            content = (parsed.get("message") or {}).get("content") or ""
                            if not content:
                                continue
                            assistant_response.append(content)
                            yield f"data: {json.dumps({'message': {'content': content}})}\n\n"
                    except httpx.HTTPStatusError as exc:
                        if exc.response is not None and exc.response.status_code == 400:
                            fallback_triggered = True
                        else:
                            raise
                    
                    if fallback_triggered and not assistant_response:
                        logger.warning("Mistral 400 on initial stream, falling back without history")
                        fallback_context = [space_context_draft, {"role": "user", "content": request.message}]
                        try:
                            async for raw_chunk in chat_stream_wrapper(
                                message="",
                                model=forced_model,
                                context=fallback_context,
                            ):
                                try:
                                    parsed = json.loads(raw_chunk)
                                except json.JSONDecodeError:
                                    continue
                                content = (parsed.get("message") or {}).get("content") or ""
                                if not content:
                                    continue
                                assistant_response.append(content)
                                yield f"data: {json.dumps({'message': {'content': content}})}\n\n"
                        except httpx.HTTPStatusError as fallback_exc:
                            if fallback_exc.response is not None and fallback_exc.response.status_code == 400:
                                logger.warning("Mistral 400 persistent, falling back to minimal context without RAG")
                                async for raw_chunk in chat_stream_wrapper(
                                    message="",
                                    model=forced_model,
                                    context=[{"role": "user", "content": request.message}],
                                ):
                                    try:
                                        parsed = json.loads(raw_chunk)
                                    except json.JSONDecodeError:
                                        continue
                                    content = (parsed.get("message") or {}).get("content") or ""
                                    if not content:
                                        continue
                                    assistant_response.append(content)
                                    yield f"data: {json.dumps({'message': {'content': content}})}\n\n"
                            else:
                                raise
                    
                    final_response = "".join(assistant_response)
                    stream_run.end(outputs={"response": final_response})

                pipeline_run.end(outputs={
                    "nb_doc_passages": len(doc_passages),
                    "nb_faq_passages": 0,
                    "response_chars": len(final_response),
                })

                # Combiner les passages docs pour les sources
                all_passages = doc_passages
                sources_data = []
                if all_passages:
                    doc_ids = list({p.get("document_id") for p in all_passages if p.get("document_id")})
                    with Session(engine) as src_session:
                        docs = (
                            src_session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
                            if doc_ids
                            else []
                        )
                        has_file_by_doc = {
                            d.id: bool(d.source_file_path)
                            for d in docs
                        }
                        # Fallback "profondeur": reconstruire une page fiable depuis les chunks
                        # si le passage n'a pas de page exploitable.
                        candidate_chunk_ids = list(
                            {
                                cid
                                for p in all_passages
                                for cid in [p.get("source_leaf_chunk_id"), p.get("chunk_id")]
                                if isinstance(cid, int)
                            }
                        )
                        chunk_by_id = {}
                        if candidate_chunk_ids:
                            chunk_rows = src_session.exec(
                                select(DocumentChunk).where(DocumentChunk.id.in_(candidate_chunk_ids))
                            ).all()
                            chunk_by_id = {c.id: c for c in chunk_rows}

                        chunk_by_doc_and_index = {}
                        doc_chunk_indexes = {
                            (p.get("document_id"), p.get("chunk_index"))
                            for p in all_passages
                            if p.get("document_id") is not None
                            and isinstance(p.get("chunk_index"), int)
                        }
                        for did, cidx in doc_chunk_indexes:
                            row = src_session.exec(
                                select(DocumentChunk)
                                .where(
                                    DocumentChunk.document_id == did,
                                    DocumentChunk.chunk_index == cidx,
                                )
                                .order_by(DocumentChunk.is_leaf.desc(), DocumentChunk.id.desc())
                            ).first()
                            if row:
                                chunk_by_doc_and_index[(did, cidx)] = row

                    for i, p in enumerate(all_passages):
                        did = p.get("document_id")
                        raw = p.get("passage_raw", p.get("passage", ""))
                        resolved_page = _resolve_page_from_passage(p)
                        resolved_page_start = _coerce_positive_int(p.get("page_start"))
                        resolved_page_end = _coerce_positive_int(p.get("page_end"))

                        if resolved_page is None:
                            fallback_chunk = None
                            leaf_chunk_id = p.get("source_leaf_chunk_id")
                            chunk_id = p.get("chunk_id")
                            if isinstance(leaf_chunk_id, int):
                                fallback_chunk = chunk_by_id.get(leaf_chunk_id)
                            if fallback_chunk is None and isinstance(chunk_id, int):
                                fallback_chunk = chunk_by_id.get(chunk_id)
                            if fallback_chunk is None:
                                fallback_chunk = chunk_by_doc_and_index.get(
                                    (did, p.get("chunk_index"))
                                )
                            if fallback_chunk is not None:
                                resolved_page = _resolve_page_from_chunk(fallback_chunk)
                                if resolved_page_start is None:
                                    resolved_page_start = _resolve_page_from_chunk(fallback_chunk)

                        sources_data.append(
                            {
                                "index": i + 1,
                                "document_id": did,
                                "document_title": p["document_title"],
                                "chunk_id": p.get("chunk_id"),
                                "source_leaf_chunk_id": p.get("source_leaf_chunk_id"),
                                "chunk_index": p.get("chunk_index"),
                                "excerpt": (raw[:200] + "...") if len(raw or "") > 200 else raw,
                                "passage_full": raw,
                                "score": round(p["score"], 2),
                                "page_no": resolved_page,
                                "page_start": resolved_page_start,
                                "page_end": resolved_page_end,
                                "section": p.get("section"),
                                "has_source_file": has_file_by_doc.get(did, False),
                            }
                        )
                    logger.info(
                        "Space chat sources built: %s",
                        [
                            {
                                "idx": s.get("index"),
                                "doc": s.get("document_id"),
                                "chunk_index": s.get("chunk_index"),
                                "page_no": s.get("page_no"),
                                "page_start": s.get("page_start"),
                                "page_end": s.get("page_end"),
                            }
                            for s in sources_data
                        ],
                    )

                complete_response = "".join(assistant_response)
                
                # Détecter si un schéma ou illustration est pertinent pour compléter la réponse
                # Stratégie multi-page : on collecte toutes les pages candidates, on les
                # pré-filtre (visuels significatifs), puis on itère en ordre de pertinence
                # et on s'arrête au premier succès. Max 3 tentatives.
                MAX_ILLUSTRATION_PAGES = 3
                illustration_data = None
                
                # a) Collecter TOUTES les pages citées dans la réponse (regex amélioré)
                cited_pages: List[Dict[str, Any]] = []  # [{"doc_id": ..., "page_no": ..., "source": ...}]
                seen_page_keys = set()
                
                # Patterns multiples pour attraper différents formats de citation
                citation_patterns = [
                    # "[Nom du document, page X]" ou "[Nom, pages X-Y]"
                    r"\[([^\]]+?),\s*pages?\s+(\d+)(?:\s*[-–]\s*\d+)?\]",
                    # "(voir Notice X, p.12)" ou "(Notice X, p. 12)"
                    r"\((?:voir\s+)?([^\)]+?),\s*p\.?\s*(\d+)\)",
                    # "[Source, p.12]"
                    r"\[([^\]]+?),\s*p\.?\s*(\d+)\]",
                ]
                
                for pattern in citation_patterns:
                    citation_matches = re.findall(pattern, complete_response, re.IGNORECASE)
                    for doc_title_match, page_no_str in citation_matches:
                        try:
                            page_no_cited = int(page_no_str)
                            # Trouver le passage correspondant dans sources_data
                            for s in sources_data:
                                p_no = s.get("page_no") or s.get("page_start")
                                p_title = s.get("document_title") or ""
                                if p_no == page_no_cited and (
                                    doc_title_match.lower() in p_title.lower()
                                    or p_title.lower() in doc_title_match.lower()
                                ):
                                    did = s.get("document_id")
                                    pk = (did, p_no)
                                    if pk not in seen_page_keys and did:
                                        seen_page_keys.add(pk)
                                        cited_pages.append({
                                            "doc_id": did,
                                            "page_no": p_no,
                                            "document_title": p_title,
                                            "source": "citation",
                                            "score": s.get("score", 0.0),
                                        })
                                    break
                        except ValueError:
                            continue
                
                # b) Ajouter les pages de sources_data non encore citées (fallback par score)
                for s in sources_data:
                    did = s.get("document_id")
                    pno = s.get("page_no") or s.get("page_start")
                    if did and pno and has_file_by_doc.get(did):
                        pk = (did, pno)
                        if pk not in seen_page_keys:
                            seen_page_keys.add(pk)
                            cited_pages.append({
                                "doc_id": did,
                                "page_no": pno,
                                "document_title": s.get("document_title") or "Document",
                                "source": "passage_fallback",
                                "score": s.get("score", 0.0),
                            })
                
                # Trier : citations d'abord, puis par score décroissant
                cited_pages.sort(key=lambda p: (0 if p["source"] == "citation" else 1, -p["score"]))
                
                logger.info(
                    f"Illustration candidate pages ({len(cited_pages)} total, max {MAX_ILLUSTRATION_PAGES} attempts): "
                    f"{[(p['doc_id'], p['page_no'], p['source'], p['score']) for p in cited_pages[:MAX_ILLUSTRATION_PAGES+2]]}"
                )
                
                # c) Itérer avec pré-filtre et arrêt au premier succès
                attempts = 0
                for page_candidate in cited_pages:
                    if attempts >= MAX_ILLUSTRATION_PAGES:
                        logger.info(f"Reached max illustration attempts ({MAX_ILLUSTRATION_PAGES}). Stopping.")
                        break
                    
                    did = page_candidate["doc_id"]
                    pno = page_candidate["page_no"]
                    doc_title_ill = page_candidate["document_title"]
                    
                    try:
                        with Session(engine) as ill_session:
                            doc_obj = ill_session.get(Document, did)
                            if not doc_obj or not doc_obj.source_file_path or not os.path.exists(doc_obj.source_file_path):
                                logger.debug(f"Skipping page {pno} of doc {did}: file not found")
                                continue
                            
                            pdf_path = doc_obj.source_file_path
                            
                            # Pré-filtre rapide : la page a-t-elle des visuels significatifs ?
                            if not has_significant_visuals(pdf_path, pno):
                                logger.info(f"Page {pno} of doc {did} has no significant visuals. Skipping.")
                                continue
                            
                            # Appel au modèle de vision (comptabilisé comme tentative)
                            attempts += 1
                            logger.info(
                                f"Illustration attempt {attempts}/{MAX_ILLUSTRATION_PAGES}: "
                                f"page {pno} of '{doc_title_ill}' (source: {page_candidate['source']})"
                            )
                            
                            yield f"data: {json.dumps({'status': 'cropping', 'attempt': attempts, 'max_attempts': MAX_ILLUSTRATION_PAGES, 'page_no': pno, 'document_title': doc_title_ill})}\n\n"
                            
                            illustration_data = await determine_and_crop_illustration(
                                query=request.message,
                                answer=complete_response,
                                pdf_path=pdf_path,
                                page_no=pno,
                                doc_title=doc_title_ill
                            )
                            
                            if illustration_data:
                                logger.info(
                                    f"Illustration found on page {pno} of '{doc_title_ill}' "
                                    f"(attempt {attempts}). Stopping search."
                                )
                                break
                            else:
                                logger.info(f"Vision model rejected page {pno} of doc {did}. Trying next.")
                    except Exception as ill_err:
                        logger.error(f"Failed to check/crop illustration on page {pno} of doc {did}: {ill_err}", exc_info=True)
                        
                # Si illustration trouvée, l'injecter dans sources_data
                if illustration_data:
                    # Trouver le doc_id du candidat qui a produit l'illustration
                    ill_doc_id = None
                    for pc in cited_pages:
                        if pc["page_no"] == illustration_data.get("page_no") and pc["document_title"] == illustration_data.get("document_title"):
                            ill_doc_id = pc["doc_id"]
                            break
                    
                    ill_source = {
                        "is_cropped_illustration": True,
                        "url": illustration_data["url"],
                        "title": illustration_data["title"],
                        "page_no": illustration_data["page_no"],
                        "document_title": illustration_data["document_title"],
                        "document_id": ill_doc_id,
                    }
                    sources_data.append(ill_source)
                    logger.info(f"Illustration added to sources_data: {ill_source}")

                # Persister et envoyer les sources avant `done` : le client peut annuler la lecture
                # dès `done`, ce qui coupait le générateur avant commit / événements suivants.
                assistant_message_id = None
                if request.conversation_id and assistant_response:
                    try:
                        sources_json = json.dumps(sources_data) if sources_data else None
                        assistant_message_id = _persist_assistant_reply(
                            request.conversation_id,
                            complete_response,
                            forced_model,
                            forced_provider,
                            sources_json,
                        )
                        logger.info(
                            "Réponse assistant sauvegardée (space chat), conversation %s avec %s sources",
                            request.conversation_id,
                            len(sources_data) if sources_data else 0,
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

