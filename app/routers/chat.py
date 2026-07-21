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
    temperature: Optional[float] = None,
    **kwargs
) -> dict:
    # Température de génération : 0.0 par défaut (hard grounding strict).
    temp = settings.SPACE_CHAT_TEMPERATURE if temperature is None else temperature
    if settings.LLM_PROVIDER == "ollama":
        from app.services.ollama_service import chat as ollama_chat
        return await ollama_chat(message=message, model=model, context=context)
    else:
        return await mistral_chat(message=message, model=model, context=context, temperature=temp, **kwargs)

async def chat_stream_wrapper(
    message: str,
    model: str,
    context: Optional[List[dict]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    # Température de génération : 0.0 par défaut (hard grounding strict).
    temp = settings.SPACE_CHAT_TEMPERATURE if temperature is None else temperature
    # Plafond de tokens de réponse : override explicite > SPACE_CHAT_MAX_TOKENS > plancher 2048.
    # Le plancher évite le repli global MAX_COMPLETION_TOKENS (1024) qui coupait les réponses
    # procédurales longues (SPACE_CHAT_MAX_TOKENS arrive souvent à None via un env vide).
    tokens = max_tokens if max_tokens is not None else (settings.SPACE_CHAT_MAX_TOKENS or 2048)

    # Reasoning natif (mistral-small) : bascule high/none via GENERATION_REASONING_EFFORT.
    # En "high", le modèle produit un ThinkChunk (masqué du stream par mistral_chat_stream)
    # avant la réponse ; on relève le plancher de tokens car le thinking consomme le budget.
    reasoning_kwargs = {}
    if settings.GENERATION_REASONING_EFFORT == "high":
        reasoning_kwargs["reasoning_effort"] = "high"
        tokens = max(tokens, settings.GENERATION_REASONING_MAX_TOKENS)

    if settings.LLM_PROVIDER == "ollama":
        from app.services.ollama_service import chat_stream as ollama_chat_stream
        async for chunk in ollama_chat_stream(message=message, model=model, context=context):
            yield chunk
    else:
        async for chunk in mistral_chat_stream(
            message=message, model=model, context=context, temperature=temp, max_tokens=tokens,
            **reasoning_kwargs,
        ):
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
from app.services.illustration_service import extract_reference_illustration
from app.tracing import trace_run, trace_pipeline
from datetime import datetime
import asyncio
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
            merged = dict(query_context or {})
            previous = conv.query_context or {}
            # L'ancre documentaire (current_documents) est écrite par
            # _update_conversation_documents APRÈS le retrieval. Le query_context produit
            # par la compréhension ne la contient jamais : sans cette préservation, chaque
            # tour l'efface AVANT le retrieval — et un tour qui n'atteint pas le retrieval
            # (clarification, low-confidence, erreur) la perdait définitivement.
            if "current_documents" not in merged and previous.get("current_documents"):
                merged["current_documents"] = previous["current_documents"]
            conv.query_context = merged
            conv.updated_at = datetime.utcnow()
            s.add(conv)
            s.commit()


def _update_conversation_documents(conversation_id: int, document_ids: List[int]) -> None:
    """Persiste les documents d'ancre (sujet documentaire courant) dans query_context.

    Fusionne dans le contexte existant sans l'écraser ; réassigne le dict entier pour
    déclencher le suivi de modification JSON de SQLAlchemy.
    """
    with Session(engine) as s:
        conv = s.get(Conversation, conversation_id)
        if conv:
            qc = dict(conv.query_context or {})
            qc["current_documents"] = [int(d) for d in document_ids]
            conv.query_context = qc
            conv.updated_at = datetime.utcnow()
            s.add(conv)
            s.commit()
            # Log de cycle de vie de l'ancre (diagnostic continuité conversation).
            logger.info(
                "[chat] ancre mise à jour — conversation=%s docs=%s",
                conversation_id,
                qc["current_documents"],
            )


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
# Chat "espaces" : température et plafond de tokens de réponse viennent de settings
# (SPACE_CHAT_TEMPERATURE / SPACE_CHAT_MAX_TOKENS), transmis par chat_stream_wrapper.
# Budget contexte relevé pour tirer parti de la fenêtre 256k de Mistral Large : plusieurs
# pages entières tiennent dans le contexte texte. Ajustables par env si modèle plus court.
SPACE_CONTEXT_MAX_CHARS = _int_env("SPACE_CONTEXT_MAX_CHARS", 80000)
SPACE_CONTEXT_MAX_PASSAGE_CHARS = _int_env(
    "SPACE_CONTEXT_MAX_PASSAGE_CHARS", settings.SPACE_CONTEXT_MAX_PASSAGE_CHARS
)
SPACE_HISTORY_MAX_CHARS = _int_env("SPACE_HISTORY_MAX_CHARS", 16000)
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"
# Prompt système du chat d'espace — 4 blocs hiérarchisés (identité, contexte, politique
# de réponse, grounding). Refonte 2026-07-03 : vocabulaire aligné sur le CAG (le modèle
# reçoit des DOCUMENTS avec en-têtes, plus des « chunks »), et politique de clarification
# explicite — l'ancienne consigne « sans demander de précision » forçait le modèle à
# choisir un produit à la place de l'utilisateur (cf. dérive INNOSLIDE).
SPACE_CHAT_SYSTEM_PROMPT = (
    "Tu es LIA, l'assistante technique experte de PROFERM (menuiserie, volets roulants). "
    "Tu accompagnes collaborateurs et clients avec chaleur, professionnalisme et précision. "
    "Tu parles au nom de PROFERM : « nous », « nos gammes » = produits PROFERM ; les documents "
    "des fournisseurs (Technal, Profine, Askey, Roto, etc.) concernent nos partenaires et sont présentés comme tels.\n"
    "\n"
    "### CONTEXTE FOURNI\n"
    "Tu reçois des DOCUMENTS complets ou en extrait étendu, chacun avec un en-tête (source, gamme, "
    "matériau, type) et des marqueurs [page N], classés par pertinence décroissante. "
    "Avant d'attribuer une valeur, une cote ou une consigne à une gamme/produit, vérifie l'en-tête "
    "du document : ne transfère JAMAIS une information d'une gamme vers une autre (ex. Perform 70 ≠ Perform 76, "
    "seuil PMR ≠ seuil standard, version standard ≠ renforcée). En cas d'informations contradictoires entre "
    "documents, le document le plus spécifique au sujet de la question prime.\n"
    "\n"
    "### SUJET DEMANDÉ (première décision, avant tout)\n"
    "Identifie le sujet EXACT de la question. Dans une question relationnelle "
    "(« X compatible avec Y », « quel X pour Y », « X adapté à Y »), le sujet est X ; "
    "Y n'est qu'un filtre. Ta réponse porte sur X : liste/décris les X trouvés. "
    "Ne produis JAMAIS de présentation non demandée de Y (pas de fiche sur le 6111 quand "
    "on te demande les seuils compatibles avec le 6111).\n"
    "\n"
    "### FORMAT ADAPTATIF (choisi par toi, à la fin, jamais à l'avance)\n"
    "Le format le plus COURT qui répond complètement est le bon :\n"
    "- lookup d'une référence nue (« 6111 », « profil 76180 ») → fiche structurée sourcée "
    "(type, dimensions, compatibilités, usage) ;\n"
    "- question relationnelle → liste des éléments trouvés avec leurs références et sources ;\n"
    "- question procédurale (« comment poser... ») → étapes ordonnées ;\n"
    "- question ponctuelle (une cote, une norme, un fait) → 1 à 3 phrases, PAS de fiche ;\n"
    "- comparaison explicite → tableau court.\n"
    "Si la réponse dépend d'une gamme/version que l'utilisateur n'a pas précisée et que les "
    "documents en couvrent PLUSIEURS : ne choisis pas à sa place — cas courts (2-3 lignes chacun) "
    "→ présente-les brièvement ; sinon pose UNE question de clarification listant les options.\n"
    "\n"
    "### GROUNDING DUR (le bloc COUVERTURE fait foi)\n"
    "Un bloc « COUVERTURE DE LA RECHERCHE » figure dans le contexte : c'est un rapport factuel "
    "mesuré, il PRIME sur ton impression de savoir.\n"
    "- Statut « vide » → réponds que les documents de l'espace ne couvrent pas cette question, "
    "et arrête-toi là.\n"
    "- Référence marquée ABSENTE → dis qu'elle n'est pas documentée dans cet espace. "
    "INTERDICTION d'utiliser les valeurs d'une référence voisine (6110 ≠ 6111).\n"
    "- Information absente ou incertaine → dis-le (« La notice ne précise pas [ce détail] ») "
    "et propose une vérification. Jamais de connaissance générale pour combler un trou factuel.\n"
    "\n"
    "### ANTI-DIGRESSION\n"
    "Aucune information non nécessaire à LA question posée : pas de section « À noter » hors "
    "sujet, pas de caractéristiques non demandées, pas de fiche complète quand on demande UN "
    "attribut, pas de rappel de références citées en passant.\n"
    "\n"
    "### GROUNDING STRICT (sécurité)\n"
    "- Fonde-toi EXCLUSIVEMENT sur les faits explicites des documents fournis et leurs liaisons directes. "
    "Aucune extrapolation, supposition ou généralisation depuis tes connaissances générales.\n"
    "- Consignes techniques, cotes, gestes de montage : restitue STRICTEMENT et MOT POUR MOT les verbes "
    "d'action et composants des documents (ex. « Mettre en contact », « Clipper l'autre côté »). "
    "N'imagine jamais d'angles, rotations ou clics absents du texte.\n"
    "- Ne combine JAMAIS des références (ex. T141019), cotes (ex. 300 mm) ou dimensions (ex. 2,40 m) "
    "issues de phrases ou sections différentes pour fabriquer une spécification qui n'est pas écrite telle quelle.\n"
    "\n"
    "### IMAGES\n"
    "Quand une référence produit (ex. TGY3702, TMX13) ou un schéma pertinent est disponible, une "
    "illustration est jointe AUTOMATIQUEMENT à ta réponse. Ne prétends JAMAIS que tu n'as pas accès "
    "aux images, aux photos ou aux catalogues : présente l'information dont tu disposes, l'illustration "
    "apparaît d'elle-même. Si aucune illustration n'est disponible pour la pièce demandée, dis simplement "
    "que la notice ne fournit pas de visuel pour cette référence — sans nier ta capacité à en montrer.\n"
    "\n"
    "### STYLE\n"
    "Concis par défaut : vise la réponse la plus courte qui couvre exactement ce qui est demandé. "
    "Prose fluide et naturelle ; listes à puces seulement si elles servent la lisibilité ; tableaux Markdown "
    "UNIQUEMENT si les documents fournissent eux-mêmes des données tabulaires (dimensions, compatibilités) — "
    "ne transforme pas une explication en plan à sections numérotées. "
    "Ne dessine JAMAIS de schéma ASCII ou de diagramme improvisé pour illustrer un raisonnement : "
    "un schéma texte n'est légitime que s'il retranscrit un schéma réellement présent dans le document. "
    "Salue courtoisement en début de conversation, sans formules répétées ensuite. "
    "AUCUNE citation dans le corps du texte : n'écris jamais [Nom du document, page X] ni de numéro de page — "
    "les sources sont affichées automatiquement sous ta réponse."
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


class GuidedChoiceRequest(BaseModel):
    """Choix sélectionné par l'utilisateur dans un aiguillage procédural."""
    value: str
    label: str = ""
    free_text: bool = False


class SpaceChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "mistral"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None
    slot_action: Optional[SlotActionRequest] = None
    guided_choice: Optional[GuidedChoiceRequest] = None


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


# Sources UI par document packé : implémentation dans le packer (partagée avec la fiche
# technique) ; ré-exportée ici car chat.py et ses tests l'utilisent sous ce nom.
from app.services.context_packer_service import (  # noqa: E402
    build_document_sources as _build_document_sources,
)


def _build_generation_context(
    session: Session,
    doc_passages: List[dict],
    anchor_document_ids: Optional[List[int]] = None,
    intent: Optional[str] = None,
) -> dict:
    """Contexte système de génération : CAG (documents entiers, fenêtre 256k) si activé,
    sinon fallback historique (passages tronqués). Les documents ANCRÉS (sujet courant de
    la conversation) sont toujours inclus dans le contexte CAG — garantie de continuité.
    Le budget de packing s'adapte à l'intent (CAG_BUDGET_BY_INTENT)."""
    if settings.CAG_ENABLED:
        from app.services.context_packer_service import build_cag_context

        return build_cag_context(
            session,
            doc_passages,
            system_prompt=SPACE_CHAT_SYSTEM_PROMPT,
            anchor_document_ids=anchor_document_ids,
            intent=intent,
        )
    return build_space_context_from_passages(doc_passages)


def _guided_streaming_response(
    gtr,
    conversation_id: int,
    forced_model: str,
    forced_provider: str,
) -> StreamingResponse:
    """Réponse SSE d'un tour de guidage (partagée par la reprise active et le nouveau
    départ). Streame le message de l'étape, les sources, persiste, puis émet l'étape
    structurée. Factorisé (P0.4) pour éviter la duplication entre les deux points d'entrée."""

    async def generate_guided():
        error_msg_to_yield = None
        try:
            # 0. Réflexion du routeur (si génération dynamique avec reasoning) : faux-streamée
            #    dans la bulle « réflexion », masquée à l'arrivée du message de l'étape.
            thinking_text = getattr(gtr, "thinking", "") or ""
            if thinking_text:
                for i in range(0, len(thinking_text), 60):
                    yield f"data: {json.dumps({'thinking': thinking_text[i:i+60]})}\n\n"

            message_text = gtr.message_text or ""
            # 1. Streamer le message de l'étape (effet machine à écrire)
            chunk_size = 40
            for i in range(0, len(message_text), chunk_size):
                chunk = message_text[i : i + chunk_size]
                yield f"data: {json.dumps({'message': {'content': chunk}})}\n\n"

            # 2. Sources (citations → PDF)
            if gtr.sources:
                yield f"data: {json.dumps({'sources': gtr.sources})}\n\n"

            # 3. Persister le message assistant (session fraîche, cf. piège SSE)
            assistant_message_id = None
            try:
                assistant_message_id = _persist_assistant_reply(
                    conversation_id,
                    message_text,
                    forced_model,
                    forced_provider,
                    json.dumps(gtr.sources, ensure_ascii=False) if gtr.sources else None,
                    metadata_json={"guided_step": gtr.step},
                )
            except Exception:
                logger.exception("Erreur sauvegarde étape guidée (space chat)")

            # 4. Émettre l'étape structurée (choix interactifs)
            step_event = dict(gtr.step)
            step_event["guided_session_id"] = gtr.guided_session_id
            step_event["message_id"] = assistant_message_id
            yield f"data: {json.dumps({'step': step_event})}\n\n"

            yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id})}\n\n"
        except Exception as e:
            logger.exception("Erreur dans le générateur stream_space_chat_message (guided)")
            error_msg_to_yield = str(e)

        if error_msg_to_yield:
            yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"

    return StreamingResponse(generate_guided(), media_type="text/event-stream")


def _guided_anchor_ids(persisted_qc: Optional[dict]) -> List[int]:
    """Documents d'ancre du sujet courant (biaisent le retrieval de chaque étape guidée)."""
    if not (settings.CONVERSATION_ANCHOR_ENABLED and isinstance(persisted_qc, dict)):
        return []
    return [
        int(d)
        for d in (persisted_qc.get("current_documents") or [])
        if isinstance(d, (int, str)) and str(d).isdigit()
    ]


async def _stream_llm_to_sse(
    context: List[dict],
    *,
    model: str,
    max_tokens: Optional[int],
    source_filter,
    sink: List[str],
):
    """Streame une génération Mistral en événements SSE, filtre le bloc <sources> et
    accumule le texte AFFICHÉ dans ``sink``. Factorisé (P0.2) pour dédupliquer les
    tentatives full/eco/minimal ; propage httpx.HTTPStatusError à l'appelant (fallback)."""
    async for raw_chunk in chat_stream_wrapper(
        message="", model=model, context=context, max_tokens=max_tokens
    ):
        try:
            parsed = json.loads(raw_chunk)
        except json.JSONDecodeError:
            continue
        # Reasoning : relayer le thinking comme événement distinct (le client l'affiche
        # dans une bulle « réflexion » puis la masque à l'arrivée de la réponse). Ni filtré
        # <sources>, ni accumulé dans le sink (ce n'est pas la réponse persistée).
        thinking = parsed.get("thinking")
        if thinking:
            yield f"data: {json.dumps({'thinking': thinking})}\n\n"
            continue
        content = (parsed.get("message") or {}).get("content") or ""
        if not content:
            continue
        if source_filter is not None:
            content = source_filter.feed(content)
            if not content:
                continue
        sink.append(content)
        yield f"data: {json.dumps({'message': {'content': content}})}\n\n"


def _build_eco_context(
    session: Session,
    doc_passages: List[dict],
    anchor_document_ids: Optional[List[int]],
    user_message: str,
) -> List[dict]:
    """Contexte de SECOURS minimal après un Mistral 400 (P0.2) : CAG re-packé à un petit
    budget (sans images ni historique) + question. Cible la cause probable (contexte trop
    volumineux) au lieu de rejouer le contexte massif à l'identique."""
    if settings.CAG_ENABLED and doc_passages:
        from app.services.context_packer_service import build_cag_context

        eco_system = build_cag_context(
            session,
            doc_passages,
            system_prompt=SPACE_CHAT_SYSTEM_PROMPT,
            token_budget=settings.CAG_ECO_TOKEN_BUDGET,
            max_documents=settings.CAG_ECO_MAX_DOCUMENTS,
            anchor_document_ids=anchor_document_ids,
        )
    else:
        eco_system = {"role": "system", "content": SPACE_CHAT_SYSTEM_PROMPT}
    return [eco_system, {"role": "user", "content": user_message}]


def _persist_reply_with_retry(
    conversation_id: int,
    text: str,
    model: str,
    provider: str,
    sources_json: Optional[str],
    *,
    metadata_json: Optional[dict] = None,
    attempts: int = 2,
) -> Optional[int]:
    """Persiste la réponse assistant avec un petit retry (P0.2) : une écriture DB qui
    échoue ne doit pas perdre silencieusement la réponse déjà affichée à l'utilisateur."""
    last_exc = None
    for i in range(max(1, attempts)):
        try:
            return _persist_assistant_reply(
                conversation_id, text, model, provider, sources_json,
                metadata_json=metadata_json,
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Persistance réponse échouée (tentative %d): %s", i + 1, exc)
    logger.error("Persistance réponse ABANDONNÉE après %d tentatives: %s", attempts, last_exc)
    return None


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

    # ——— Aiguillage procédural (guidage SAV / chantier) ———
    # Court-circuite le pipeline one-shot quand la demande relève d'un guidage pas-à-pas,
    # ou quand un parcours guidé est déjà actif sur la conversation (reprise).
    # Gardé par GUIDED_FLOW_ENABLED : zéro impact quand le flag est désactivé.
    # 1) REPRISE d'un parcours actif : 0 appel LLM (l'utilisateur répond à une étape en
    #    cours). Le NOUVEAU DÉPART (décision is_guided) est traité APRÈS la compréhension
    #    fusionnée, dont il RÉUTILISE la décision guidée — un seul appel LLM avant le
    #    retrieval (P0.4), au lieu d'un decide_guided_mode dédié qui doublait l'appel.
    guided_active_state = None
    guided_persisted_qc = None
    if settings.GUIDED_FLOW_ENABLED and request.conversation_id:
        from app.services.guided_flow_service import load_active_guided_state, run_guided_turn

        conv_for_guided = session.get(Conversation, request.conversation_id)
        guided_persisted_qc = conv_for_guided.query_context if conv_for_guided else None
        guided_active_state = load_active_guided_state(guided_persisted_qc)

        if guided_active_state:
            anchors = _guided_anchor_ids(guided_persisted_qc)
            logger.info("[chat] Mode guidé — REPRISE parcours actif (anchors=%s)", anchors)
            gtr = await run_guided_turn(
                session=session,
                space_id=space_id,
                user_id=current_user.id,
                conversation_id=request.conversation_id,
                user_message=request.message,
                guided_choice=request.guided_choice.model_dump() if request.guided_choice else None,
                history=conversation_context,
                active_state=guided_active_state,
                product_named=True,  # produit déjà traité en reprise
                anchor_document_ids=anchors or None,
            )
            return _guided_streaming_response(
                gtr, request.conversation_id, forced_model, forced_provider
            )

    # ——— Fiche technique : ROUTE SUPPRIMÉE (refonte routage 2026-07-21, C2) ———
    # L'ancien fast-path regex court-circuitait la compréhension et figeait le format
    # AVANT d'avoir lu la question (« SEUILS compatibles avec 6111 » → fiche 6111).
    # Ses deux valeurs sont relogées dans la voie unique :
    #   - résolution par code → chunk pinning (C6) + BM25 sur la question autonome ;
    #   - format fiche → règle FORMAT ADAPTATIF du prompt (décidée par le reasoning).

    retrieval_queries = None
    retrieval_query_groups = None
    rag_user_message = request.message
    # Texte utilisé pour la RECHERCHE documentaire (peut différer du message de
    # génération : reformulation history-aware en question autonome).
    retrieval_query_text = request.message
    lw_result = None
    # Documents d'ancre du sujet courant (réutilisés pour biaiser le retrieval de ce tour).
    anchor_document_ids: List[int] = []

    if settings.QUERY_UNDERSTANDING_ENABLED:
        logger.info("[chat] Étape 1/5 — lightweight query understanding")
        from app.services.lightweight_query_understanding import run_lightweight_understanding

        persisted_context = None
        if request.conversation_id:
            conv_for_ctx = session.get(Conversation, request.conversation_id)
            if conv_for_ctx:
                persisted_context = conv_for_ctx.query_context

        # Fallback robuste (C1) : compréhension en échec → route=search avec le message
        # brut. Chercher ne coûte presque rien et ne peut pas inventer — c'est le défaut
        # le plus sûr ; jamais de mur pour l'utilisateur.
        try:
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
                    "topic_shift": lw_result.topic_shift,
                    "signals": lw_result.signals.model_dump() if lw_result.signals else None,
                })
        except Exception as qu_err:
            logger.error("[chat] compréhension en échec → fallback route=search : %s", qu_err)
            lw_result = None

        # Changement de sujet détecté : le retrieval/génération de ce tour repart des
        # signaux FRAÎCHEMENT extraits (déjà le cas), sans réintégration de l'ancien sujet
        # (assurée en amont par le condense topic_shift-aware et _node_merge_context).
        if lw_result and lw_result.topic_shift:
            logger.info("[chat] topic_shift détecté — le contexte du tour précédent n'est pas réutilisé")

        # Ancre documentaire : hors changement de sujet, on réutilise les documents du sujet
        # courant (mémorisés au tour précédent) pour biaiser le retrieval → la conversation
        # reste sur le même produit d'un tour à l'autre.
        if (
            settings.CONVERSATION_ANCHOR_ENABLED
            and lw_result
            and not lw_result.topic_shift
            and persisted_context
        ):
            anchor_document_ids = [
                int(d)
                for d in (persisted_context.get("current_documents") or [])
                if d is not None
            ]
            if anchor_document_ids:
                logger.info("[chat] ancre documentaire active — docs=%s", anchor_document_ids)

        if request.conversation_id and lw_result and lw_result.query_context:
            _save_conversation_query_context(request.conversation_id, lw_result.query_context)

        routing_decision_decision = lw_result.route if lw_result else "rag"
    else:
        # C1 (refonte routage 2026-07-21) : plus de routeur LLM parallèle quand la
        # compréhension est désactivée — route=search directe avec le message brut.
        # (L'ancien decide_retrieval_route autonome doublait l'appel et les chemins.)
        logger.info("[chat] Compréhension désactivée — route=search par défaut")
        routing_decision_decision = "rag"

    # ——— NOUVEAU DÉPART guidé (décision portée par la compréhension fusionnée, P0.4) ———
    # Réutilise la décision guidée du fused (0 appel LLM supplémentaire) ; retombe sur
    # decide_guided_mode uniquement si le fused ne l'a pas produite (QU off / fused échoué).
    if (
        settings.GUIDED_FLOW_ENABLED
        and request.conversation_id
        and not guided_active_state
    ):
        from app.services.guided_flow_service import run_guided_turn
        from app.services.query_reasoning_service import resolve_guided_mode

        guided_topic_hint = ""
        if lw_result and lw_result.query_context:
            guided_topic_hint = (
                lw_result.query_context.get("current_topic")
                or lw_result.query_context.get("standalone_question")
                or ""
            )
        guided_decision = await resolve_guided_mode(
            request.message,
            conversation_context,
            fused_guided=(lw_result.guided if lw_result else None),
            topic=guided_topic_hint,
        )
        # Intention ambiguë (« régler la hauteur » = ajuster OU dimensionner) : on n'entre
        # PAS en guidé, le one-shot pose la clarification (politique de réponse n°2).
        if guided_decision.is_guided and guided_decision.needs_intent_clarification:
            logger.info("[chat] Guidé différé — intention ambiguë : clarification via one-shot")
        elif guided_decision.is_guided:
            anchors = _guided_anchor_ids(guided_persisted_qc)
            logger.info(
                "[chat] Mode guidé — NOUVEAU départ flow_kind=%s topic=%r product_named=%s anchors=%s",
                guided_decision.flow_kind,
                guided_decision.topic or guided_topic_hint,
                guided_decision.product_named,
                anchors,
            )
            gtr = await run_guided_turn(
                session=session,
                space_id=space_id,
                user_id=current_user.id,
                conversation_id=request.conversation_id,
                user_message=request.message,
                guided_choice=request.guided_choice.model_dump() if request.guided_choice else None,
                history=conversation_context,
                active_state=None,
                flow_kind=guided_decision.flow_kind,
                topic=guided_decision.topic or guided_topic_hint,
                symptom=guided_decision.detected_symptom,
                product_named=guided_decision.product_named,
                anchor_document_ids=anchors or None,
            )
            return _guided_streaming_response(
                gtr, request.conversation_id, forced_model, forced_provider
            )

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
                        # C5 (refonte 2026-07-21) : même moteur de stream que la voie RAG
                        # (_stream_llm_to_sse) — relais thinking, parsing et erreurs uniformes.
                        async for sse_event in _stream_llm_to_sse(
                            full_context_draft,
                            model=forced_model,
                            max_tokens=None,
                            source_filter=None,
                            sink=assistant_response,
                        ):
                            yield sse_event

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
        retrieval_query_groups = lw_result.query_groups if len(lw_result.query_groups) > 1 else None
        rag_user_message = (
            lw_result.query_context.get("enriched_user_message")
            or lw_result.query_context.get("original_user_message")
            or request.message
        )
        # Recherche documentaire : privilégie la question autonome reformulée
        # (résout les messages de suivi type "et le tgy3834 ?"), sinon fallback.
        retrieval_query_text = (
            lw_result.query_context.get("standalone_question")
            or rag_user_message
        )
        if retrieval_query_groups:
            logger.info(
                "[chat] Multi-query strategy=%s groups=%s",
                lw_result.query_strategy,
                [g.label for g in retrieval_query_groups],
            )

    step_label = "2/5" if settings.QUERY_UNDERSTANDING_ENABLED else "2/4"
    logger.info("[chat] Étape %s — retrieval hybride (ColPali + pgvector + BM25 + KAG)", step_label)
    import time as _time
    _t_retrieval_start = _time.perf_counter()
    with trace_run(
        "technical_retrieval",
        run_type="retriever",
        inputs={
            "query": retrieval_query_text,
            "space_id": space_id,
            "k": RAG_TOP_K,
            "retrieval_queries": retrieval_queries.model_dump() if retrieval_queries else None,
        },
        tags=["rag", "technical", "space"],
    ) as retrieval_run:
        from app.services.space_search_service import search_technical_passages

        async def _do_retrieval():
            return await search_technical_passages(
                session=session,
                space_id=space_id,
                query_text=retrieval_query_text,
                user_id=current_user.id,
                k=RAG_TOP_K,
                queries=retrieval_queries,
                signals=lw_result.signals if lw_result and lw_result.signals else None,
                query_groups=retrieval_query_groups,
                anchor_document_ids=anchor_document_ids or None,
            )

        # Budget temps global (P0.3) : un canal qui freeze (ColPali CPU, MaxSim LanceDB) ne
        # doit pas bloquer indéfiniment — au-delà du budget, dégradation gracieuse (0 passage).
        try:
            if settings.RETRIEVAL_TIMEOUT_S and settings.RETRIEVAL_TIMEOUT_S > 0:
                retrieval = await asyncio.wait_for(
                    _do_retrieval(), timeout=settings.RETRIEVAL_TIMEOUT_S
                )
            else:
                retrieval = await _do_retrieval()
        except asyncio.TimeoutError:
            logger.error(
                "[chat] retrieval au-delà du budget %.0fs → dégradation gracieuse (0 passage)",
                settings.RETRIEVAL_TIMEOUT_S,
            )
            retrieval = {
                "passages": [], "images": [], "status": "degraded_timeout",
                "reason": "retrieval_timeout",
            }
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

    # [PERF] Bucket macro n°1 : tout le retrieval (encode ColPali + MaxSim + 4 retrievers
    # + rerank MiniLM + rerank vision). À comparer au bucket génération plus bas.
    logger.info(
        "[PERF][chat] retrieval TOTAL %.2fs — %d passages (rerank=%s)",
        _time.perf_counter() - _t_retrieval_start,
        len(doc_passages),
        rerank_status,
    )

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

    # L'autorité de source (primary_source) est déjà appliquée par
    # apply_soft_boosts_to_passages ci-dessus, de façon proportionnelle à l'étendue
    # des scores. L'ancien refine_with_source_authority ajoutait un SECOND boost
    # (0.8·confidence, échelle ambiguë) sur le même critère → double-comptage supprimé.

    # Mémorise les documents dominants de ce tour comme ancre du sujet courant (réutilisée
    # pour biaiser le retrieval du prochain tour, tant qu'il n'y a pas de changement de sujet).
    if settings.CONVERSATION_ANCHOR_ENABLED and request.conversation_id and doc_passages:
        from app.services.retrieval_boost_service import compute_anchor_documents

        new_anchor = compute_anchor_documents(
            doc_passages, max_docs=settings.CONVERSATION_ANCHOR_MAX_DOCS
        )
        if new_anchor:
            _update_conversation_documents(request.conversation_id, new_anchor)

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
        space_context_draft = _build_generation_context(
            session,
            doc_passages,
            anchor_document_ids=anchor_document_ids or None,
            intent=(lw_result.signals.intent if lw_result and lw_result.signals else None),
        )
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
        space_context_draft = _build_generation_context(
            session,
            doc_passages,
            anchor_document_ids=anchor_document_ids or None,
            intent=(lw_result.signals.intent if lw_result and lw_result.signals else None),
        )

    # ——— Chunk pinning (C6) + bloc COUVERTURE (C3) ———
    # Placés en FIN de message système (zone de forte attention, comme le fil de
    # conversation) : extraits de référence VERBATIM pour les codes demandés, puis
    # rapport factuel de couverture (le grounding cesse d'être déclaratif).
    from app.services.coverage_service import (
        build_coverage_block,
        extract_message_reference_codes,
    )

    requested_codes = extract_message_reference_codes(
        retrieval_query_text,
        request.message,
        *(
            (lw_result.signals.detected_references or [])
            if lw_result and lw_result.signals
            else []
        ),
    )

    pinned_codes: List[str] = []
    if requested_codes and settings.KAG_ENABLED:
        try:
            from app.services.kag_graph_service import build_pinned_reference_block

            pinned_block, pinned_codes = build_pinned_reference_block(
                session, space_id, requested_codes
            )
            if pinned_block:
                space_context_draft["content"] += "\n\n" + pinned_block
                logger.info("[chat] chunk pinning — codes épinglés=%s", pinned_codes)
        except Exception as pin_err:
            logger.warning("[chat] chunk pinning ignoré : %s", pin_err)

    coverage_block = build_coverage_block(
        context_text=space_context_draft.get("content") or "",
        requested_codes=requested_codes,
        doc_passages=doc_passages,
        pinned_codes=pinned_codes,
        retrieval_status=retrieval_status,
    )
    space_context_draft["content"] += "\n\n" + coverage_block

    # Fil de la conversation : sujet courant, entités en focus et demande reformulée,
    # injectés à la FIN du message système (donc juste avant l'historique et le message
    # utilisateur). Sans ce bloc, un suivi elliptique ("tu as ses dimensions ?") arrive
    # après ~100k tokens de documents CAG et le modèle perd le référent — il répond sur
    # n'importe quel élément du contexte au lieu du sujet de la conversation.
    if lw_result:
        from app.services.conversation_state_service import format_generation_state_block

        conversation_thread_block = format_generation_state_block(
            lw_result.query_context,
            standalone_question=lw_result.query_context.get("standalone_question"),
            original_message=request.message,
        )
        if conversation_thread_block:
            space_context_draft["content"] += "\n\n" + conversation_thread_block

    full_context_draft = []
    full_context_draft.append(space_context_draft)

    # Changement de sujet : on n'envoie PAS l'historique de l'ancien sujet à la génération,
    # sinon le modèle reste ancré dessus. Les passages RAG + le message courant suffisent.
    if lw_result and lw_result.topic_shift:
        rag_history: List[dict] = []
        logger.info("[chat] topic_shift — historique de génération élagué (nouveau sujet)")
    else:
        rag_history = list(conversation_context)
    while rag_history and rag_history[-1].get("role") == "user":
        rag_history.pop()

    full_context_draft.extend(rag_history)

    user_images: List[str] = []
    user_image_captions: List[dict] = []
    cag_documents_ctx: List[dict] = list(space_context_draft.get("cag_documents") or [])
    if doc_passages and is_vision_model(forced_model):
        if settings.CAG_ENABLED and cag_documents_ctx:
            # Alignement texte/visuel : PNG UNIQUEMENT pour des pages réellement packées
            # dans le contexte CAG (et légendées), pas pour les passages top-k bruts.
            from app.services.context_packer_service import select_cag_images

            user_images, user_image_captions = await asyncio.to_thread(
                select_cag_images,
                session,
                cag_documents_ctx,
                doc_passages,
            )
            logger.info(
                "[stream_space_chat_message] %d image(s) PNG alignées sur le contexte CAG pour %s",
                len(user_images),
                forced_model,
            )
        elif retrieval_images:
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

    # Sandwich anti « lost in the middle » : rappel final de tâche (+ question autonome)
    # en toute fin de contexte, après les ~10-100k tokens de documents.
    task_reminder = None
    if settings.CAG_ENABLED and doc_passages:
        from app.services.rag_generation_service import build_cag_task_reminder

        task_reminder = build_cag_task_reminder(
            request.message,
            standalone_question=(
                lw_result.query_context.get("standalone_question") if lw_result else None
            ),
        )

    user_msg = build_rag_user_message(
        rag_user_message,
        images_b64=user_images or None,
        image_captions=user_image_captions or None,
        task_reminder=task_reminder,
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
                    # Bloc machine <sources>{...}</sources> émis en fin de réponse CAG :
                    # filtré du stream (jamais affiché), parsé pour les sources UI.
                    from app.services.stream_source_filter import SourcesTagStreamFilter

                    source_filter = SourcesTagStreamFilter() if settings.CAG_ENABLED else None
                    # Plafond de réponse relevé en mode CAG (procédures complètes).
                    gen_max_tokens = settings.CAG_MAX_COMPLETION_TOKENS if settings.CAG_ENABLED else None

                    # Tentatives dégressives face à un Mistral 400 (souvent = contexte trop
                    # gros) : contexte complet → CAG RE-PACKÉ en budget eco → question nue.
                    # Contextes construits PARESSEUSEMENT (callables) : le cas normal (succès
                    # au 1er essai) ne doit JAMAIS payer le coût d'un repackaging CAG "eco"
                    # inutile (SQL + calcul de tokens) à chaque requête.
                    # Le filtre <sources> est RÉINITIALISÉ à chaque tentative (P0.2) pour ne
                    # pas hériter d'un état de capture partiel de la tentative précédente.
                    stream_attempts = [
                        ("full", lambda: full_context_draft, gen_max_tokens),
                        (
                            "eco",
                            lambda: _build_eco_context(
                                session, doc_passages,
                                anchor_document_ids or None, request.message,
                            ),
                            gen_max_tokens,
                        ),
                        ("minimal", lambda: [{"role": "user", "content": request.message}], None),
                    ]
                    # [PERF] Bucket macro n°2 : génération (latence 1er token = prefill).
                    _t_gen_start = _time.perf_counter()
                    for _label, _ctx_fn, _mt in stream_attempts:
                        # On ne bascule en fallback QUE si rien n'a encore été émis.
                        if assistant_response:
                            break
                        if _label != "full":
                            logger.warning("Mistral 400 → tentative de secours '%s'", _label)
                            if settings.CAG_ENABLED:
                                source_filter = SourcesTagStreamFilter()
                        try:
                            async for _sse in _stream_llm_to_sse(
                                _ctx_fn(),
                                model=forced_model,
                                max_tokens=_mt,
                                source_filter=source_filter,
                                sink=assistant_response,
                            ):
                                yield _sse
                            break  # génération réussie
                        except httpx.HTTPStatusError as exc:
                            if (
                                exc.response is not None
                                and exc.response.status_code == 400
                                and not assistant_response
                            ):
                                continue  # tenter le niveau de secours suivant
                            raise
                    logger.info(
                        "[PERF][chat] génération TOTAL %.2fs — %d chars",
                        _time.perf_counter() - _t_gen_start,
                        sum(len(c) for c in assistant_response),
                    )

                    # Fin de stream : relâcher un éventuel texte retenu à tort (balise
                    # <sources> jamais complétée) pour ne rien perdre de la réponse.
                    if source_filter is not None:
                        _tail = source_filter.finalize()
                        if _tail:
                            assistant_response.append(_tail)
                            yield f"data: {json.dumps({'message': {'content': _tail}})}\n\n"

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
                has_file_by_doc = {}
                if settings.CAG_ENABLED and cag_documents_ctx:
                    # Sources par DOCUMENT : reflète le contexte réellement packé (CAG),
                    # filtré par le bloc <sources> du modèle quand il est exploitable.
                    used_pages_by_index = (
                        {u["doc"]: u["pages"] for u in source_filter.used_documents}
                        if source_filter is not None
                        else {}
                    )
                    sources_data = _build_document_sources(cag_documents_ctx, used_pages_by_index)
                    has_file_by_doc = {
                        s["document_id"]: bool(s.get("has_source_file"))
                        for s in sources_data
                        if s.get("document_id") is not None
                    }
                    logger.info(
                        "Space chat sources (doc-level): %s",
                        [
                            (s.get("index"), s.get("document_id"), s.get("used_pages"))
                            for s in sources_data
                        ],
                    )
                elif all_passages:
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
                
                # Illustration ancrée sur la RÉFÉRENCE (code profilé) demandée et présente comme
                # texte sur une page citée. Sans code cible ancrable → aucune image (abstention).
                MAX_ILLUSTRATION_PAGES = 3
                illustration_data = None

                # Codes de référence cibles, dérivés de la demande utilisateur (signaux + message)
                _ref_pattern = re.compile(
                    getattr(settings, "ILLUSTRATION_REFERENCE_PATTERN", r"\b\d{3,5}[A-Za-z]?\b")
                )
                target_codes: List[str] = []
                _seen_codes: set = set()
                _signal_texts: List[str] = []
                if lw_result and lw_result.signals:
                    _signal_texts.extend(lw_result.signals.detected_references or [])
                    _signal_texts.extend(lw_result.signals.entity_texts or [])
                _signal_texts.append(request.message or "")
                for _txt in _signal_texts:
                    for _code in _ref_pattern.findall(_txt or ""):
                        _k = _code.strip().lower()
                        if _k and _k not in _seen_codes:
                            _seen_codes.add(_k)
                            target_codes.append(_code.strip())

                # Tie-breaker : prioriser les codes présents dans les visual_labels des chunks cités
                if target_codes:
                    try:
                        _label_codes: set = set()
                        _chunk_ids = {
                            cid
                            for s in sources_data
                            for cid in (s.get("source_leaf_chunk_id"), s.get("chunk_id"))
                            if isinstance(cid, int)
                        }
                        if _chunk_ids:
                            with Session(engine) as _lbl_session:
                                _rows = _lbl_session.exec(
                                    select(DocumentChunk).where(DocumentChunk.id.in_(_chunk_ids))
                                ).all()
                                for _c in _rows:
                                    for _lbl in ((_c.metadata_json or {}).get("visual_labels") or []):
                                        for _code in _ref_pattern.findall(str(_lbl)):
                                            _label_codes.add(_code.strip().lower())
                        if _label_codes:
                            target_codes.sort(key=lambda c: 0 if c.strip().lower() in _label_codes else 1)
                    except Exception as _lbl_err:
                        logger.debug("Illustration: priorisation visual_labels ignorée: %s", _lbl_err)
                    logger.info("Illustration: codes cibles (priorité) = %s", target_codes)
                else:
                    logger.info("Illustration: aucun code de référence ancrable → pas d'illustration.")
                
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
                
                # Itérer : l'ancre texte sert de pré-filtre ; arrêt au premier succès.
                # Si aucun code cible, on ne tente rien (abstention stricte).
                attempts = 0
                for page_candidate in (cited_pages if target_codes else []):
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

                            # L'ancre texte (search_for) dans extract_reference_illustration sert de pré-filtre.
                            attempts += 1
                            logger.info(
                                f"Illustration attempt {attempts}/{MAX_ILLUSTRATION_PAGES}: "
                                f"page {pno} of '{doc_title_ill}' (source: {page_candidate['source']})"
                            )
                            
                            yield f"data: {json.dumps({'status': 'cropping', 'attempt': attempts, 'max_attempts': MAX_ILLUSTRATION_PAGES, 'page_no': pno, 'document_title': doc_title_ill})}\n\n"
                            
                            illustration_data = await extract_reference_illustration(
                                targets=target_codes,
                                pdf_path=pdf_path,
                                page_no=pno,
                                doc_title=doc_title_ill,
                            )

                            if illustration_data:
                                logger.info(
                                    "Illustration trouvée: '%s' page %s de '%s' (tentative %s).",
                                    illustration_data.get("reference"), pno, doc_title_ill, attempts,
                                )
                                break
                            else:
                                logger.info(f"Aucune illustration ancrable sur page {pno} de doc {did}. Page suivante.")
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

                # Vérification post-génération (P2, 2026-07-20) : le texte a déjà streamé au
                # client à ce stade — ce contrôle ne bloque PAS l'affichage, il détecte et
                # trace les réponses hors-sujet ou hallucinées (cf. plan_p2_generation_
                # small_verification_2026-07-20.md §2). N'échoue jamais la persistance.
                verification_result = None
                if request.conversation_id and assistant_response and space_context_draft.get("content"):
                    try:
                        from app.services.response_verification_service import verify_response

                        verification_result = await verify_response(
                            question=retrieval_query_text,
                            response_text=complete_response,
                            context_text=space_context_draft["content"],
                            model=forced_model,
                        )
                    except Exception as verif_err:
                        logger.warning("Vérification post-génération ignorée: %s", verif_err)

                # Persister et envoyer les sources avant `done` : le client peut annuler la lecture
                # dès `done`, ce qui coupait le générateur avant commit / événements suivants.
                assistant_message_id = None
                if request.conversation_id and assistant_response:
                    sources_json = json.dumps(sources_data) if sources_data else None
                    assistant_message_id = _persist_reply_with_retry(
                        request.conversation_id,
                        complete_response,
                        forced_model,
                        forced_provider,
                        sources_json,
                        metadata_json=(
                            {"verification": verification_result} if verification_result else None
                        ),
                    )
                    logger.info(
                        "Réponse assistant sauvegardée (space chat), conversation %s avec %s sources",
                        request.conversation_id,
                        len(sources_data) if sources_data else 0,
                    )

                if sources_data:
                    yield f"data: {json.dumps({'sources': sources_data})}\n\n"
                yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id})}\n\n"

        except MistralRateLimitError as e:
            logger.warning("Limite de débit Mistral (stream_space_chat_message): %s", e)
            error_msg_to_yield = str(e)
        except Exception as e:
            logger.exception("Erreur dans le générateur stream_space_chat_message")
            error_msg_to_yield = str(e)
            # Réponse partiellement streamée puis interrompue (P0.2) : persister le partiel
            # AVEC un marqueur de troncature — sinon l'utilisateur voit du texte à l'écran
            # que rien ne conserve en base (et le prochain tour perd ce contexte).
            partial = "".join(assistant_response).strip()
            if request.conversation_id and partial:
                _persist_reply_with_retry(
                    request.conversation_id,
                    partial,
                    forced_model,
                    forced_provider,
                    None,
                    metadata_json={"response_truncated": True, "error": str(e)[:300]},
                )

        if error_msg_to_yield:
            yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

