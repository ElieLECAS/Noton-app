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
from app.services.chat_tools import get_available_tools
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.space import Space
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.space_search_service import search_relevant_passages as search_space_passages
from app.services.space_service import get_space_by_id
from app.tracing import trace_run, trace_pipeline
from datetime import datetime
import json
import logging
import os

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


# Nombre de passages RAG renvoyés au LLM (configurable via RAG_TOP_K).
# Défaut 8 : avec 1 passage, le modèle comble avec des généralisations faux catalogue (tableaux inventés, ✓/✗).
RAG_TOP_K = _int_env("RAG_TOP_K", 8)
# Paramétrage en dur du chat "espaces"
SPACE_CHAT_MAX_TOKENS = 1200
SPACE_CHAT_TEMPERATURE = 0.1
SPACE_CHAT_TOP_P = None
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"
SPACE_CHAT_SYSTEM_PROMPT = (
    "Tu es LIA, l'assistante experte de PROFERM. Ton rôle est d'accompagner les collaborateurs et les clients de manière chaleureuse, professionnelle et précise sur nos produits et services.\n"
    "Identité : Tu parles au nom de PROFERM. Quand tu dis 'nous' ou 'nos gammes', tu fais référence aux produits PROFERM. Les documents des fournisseurs (Technal, Profine, Askey, Roto, etc.) concernent nos partenaires et doivent être présentés comme tels.\n"
    "Ton & Style de discussion : Réponds sous forme de discussion fluide, naturelle et en prose. Privilégie une vraie conversation chaleureuse plutôt que d'aligner systématiquement des listes. Sois agréable dans tes échanges. Salue courtoisement l'utilisateur si c'est le début de la conversation, mais supprime tout texte superflu (évite les formules de salutation répétées ou de politesse de fin systématiques).\n"
    "Concision stricte : Limite drastiquement la longueur de tes réponses. Reste très synthétique, privilégie la qualité de l'explication courte à la quantité de texte, et va directement au but sans longs paragraphes d'introduction ou de conclusion.\n"
    "Puces & Tableaux : Priorise la prose. N'utilise les listes à puces que si c'est réellement justifié (par exemple pour énumérer des éléments simples où la prose nuirait à la lisibilité). Utilise les tableaux Markdown pour présenter clairement les données techniques ou les comparaisons complexes sans répéter ou paraphraser les informations du tableau dans le texte qui l'accompagne.\n"
    "Filtrage des informations : Réponds exactement au périmètre de la question posée sans proposer d'informations complémentaires non sollicitées.\n"
    "Désambiguïsation : Sois extrêmement vigilante avec les dénominations de gammes proches (ex: Perform 70 vs Perform 76). Ne les confonds jamais. Si une requête est ambiguë, demande poliment une précision.\n"
    "Règle d'or : Ne jamais inventer de données. Si l'information recherchée est absente du contexte fourni, indique-le avec courtoisie et propose une étape de vérification."
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
                if not settings.MISTRAL_API_KEY:
                    error_msg_to_yield = "Mistral API key n'est pas configurée"
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
                    error_msg_to_yield = "Mistral API key n'est pas configurée"
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
                    "section": p.get("section"),
                    "passage_preview": (p.get("passage_raw") or p.get("passage", ""))[:300],
                }
                for p in doc_passages
            ],
        })

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
    if request.context:
        full_context_draft.extend(request.context[-10:])
    full_context_draft.append({"role": "user", "content": request.message})

    _pipeline_inputs_space = {
        "query": request.message,
        "space_id": space_id,
        "user_id": current_user.id,
        "model": forced_model,
        "nb_doc_passages": len(doc_passages),
    }

    assistant_response: List[str] = []

    async def generate():
        error_msg_to_yield = None
        try:
            with trace_pipeline(
                "space_chat_pipeline",
                inputs=_pipeline_inputs_space,
                tags=["chat", "space", "rag", "kag"],
            ) as pipeline_run:
                if not settings.MISTRAL_API_KEY:
                    raise ValueError("Mistral API key non configurée")

                # Étape 1 : Génération du Brouillon de Réponse (sans FAQ)
                draft_response = ""
                with trace_run(
                    "draft_generation",
                    run_type="llm",
                    inputs={
                        "model": forced_model,
                        "messages": [
                            {"role": m.get("role"), "content": str(m.get("content", ""))}
                            for m in full_context_draft
                        ]
                    },
                    tags=["llm", "draft", "space"]
                ) as draft_run:
                    draft_res = await mistral_chat(
                        "",
                        forced_model,
                        full_context_draft,
                        max_tokens=SPACE_CHAT_MAX_TOKENS,
                        temperature=SPACE_CHAT_TEMPERATURE,
                        top_p=SPACE_CHAT_TOP_P,
                    )
                    if "choices" in draft_res and len(draft_res["choices"]) > 0:
                        draft_response = draft_res["choices"][0]["message"].get("content", "").strip()
                    draft_run.end(outputs={"draft_response": draft_response})

                # Pass 2 : Recherche FAQ correctives post-brouillon (si activée)
                final_response = draft_response
                faq_passages = []
                
                if draft_response and settings.FAQ_POST_DRAFT_ENABLED:
                    with trace_run(
                        "faq_corrective_retrieval",
                        run_type="retriever",
                        inputs={
                            "query": request.message,
                            "space_id": space_id,
                            "draft_preview": draft_response[:200],
                        },
                        tags=["retrieval", "faq_corrective", "post_draft"],
                    ) as faq_retrieval_run:
                        from app.services.space_search_service import search_corrective_faq_passages
                        faq_result = await search_corrective_faq_passages(
                            session=session,
                            space_id=space_id,
                            query_text=request.message,
                            user_id=current_user.id,
                            draft_response=draft_response,
                            k=settings.FAQ_TOP_K,
                        )
                        faq_passages = faq_result.get("passages", [])
                        faq_status = faq_result.get("status")
                        faq_reason = faq_result.get("reason")
                        
                        faq_retrieval_run.end(outputs={
                            "status": faq_status,
                            "reason": faq_reason,
                            "nb_faq": len(faq_passages),
                        })

                # Étape 3 : Critique/Correction si FAQ correctives pertinentes trouvées
                if faq_passages and draft_response:
                    from app.services.chat_critique_service import (
                        build_critique_messages,
                        resolve_critique_final,
                    )

                    faq_content_list = []
                    for i, p in enumerate(faq_passages, 1):
                        raw = p.get("passage_raw") or p.get("passage", "")
                        faq_content_list.append(f"FAQ {i}:\n{raw}")
                    faq_formatted_text = "\n---\n".join(faq_content_list)
                    critique_messages = build_critique_messages(draft_response, faq_formatted_text)

                    with trace_run(
                        "critique_generation",
                        run_type="llm",
                        inputs={
                            "model": forced_model,
                            "draft_response": draft_response[:200],
                            "nb_faq": len(faq_passages),
                        },
                        tags=["llm", "critique", "space"]
                    ) as critique_run:
                        critique_res = await mistral_chat(
                            "",
                            forced_model,
                            critique_messages,
                            max_tokens=SPACE_CHAT_MAX_TOKENS,
                            temperature=0,
                            response_format={"type": "json_object"},
                        )
                        raw_critique = ""
                        if "choices" in critique_res and len(critique_res["choices"]) > 0:
                            raw_critique = critique_res["choices"][0]["message"].get("content", "").strip()
                        final_response = resolve_critique_final(raw_critique, draft_response)
                        critique_run.end(outputs={
                            "final_response": final_response,
                            "raw_critique_preview": raw_critique[:300],
                        })

                if not final_response:
                    final_response = "Je n'ai pas pu générer de réponse."

                # Simuler le streaming par chunks pour garder l'effet de frappe côté client
                chunk_size = 25
                for i in range(0, len(final_response), chunk_size):
                    chunk = final_response[i : i + chunk_size]
                    assistant_response.append(chunk)
                    yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"

                pipeline_run.end(outputs={
                    "nb_doc_passages": len(doc_passages),
                    "nb_faq_passages": len(faq_passages),
                    "response_chars": len(final_response),
                })

                # Combiner les passages docs + FAQ pour les sources
                all_passages = doc_passages + faq_passages
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
                            d.id: (d.document_type == "document" and bool(d.source_file_path))
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

                # Persister et envoyer les sources avant `done` : le client peut annuler la lecture
                # dès `done`, ce qui coupait le générateur avant commit / événements suivants.
                assistant_message_id = None
                if request.conversation_id and assistant_response:
                    try:
                        complete_response = "".join(assistant_response)
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

