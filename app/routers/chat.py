from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Literal
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
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
    # Température de génération : SPACE_CHAT_TEMPERATURE (0.2 par défaut, grounding strict).
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
    # Température de génération : SPACE_CHAT_TEMPERATURE (0.2 par défaut, grounding strict).
    temp = settings.SPACE_CHAT_TEMPERATURE if temperature is None else temperature
    # Plafond de tokens de réponse : override explicite > SPACE_CHAT_MAX_TOKENS > plancher 2048.
    # Le plancher évite le repli global MAX_COMPLETION_TOKENS (1024) qui coupait les réponses
    # procédurales longues (SPACE_CHAT_MAX_TOKENS arrive souvent à None via un env vide).
    tokens = max_tokens if max_tokens is not None else (settings.SPACE_CHAT_MAX_TOKENS or 2048)

    # Reasoning natif : bascule high/none via GENERATION_REASONING_EFFORT. En "high", le
    # modèle produit un ThinkChunk (masqué du stream par mistral_chat_stream) avant la
    # réponse ; on relève le plancher de tokens car le thinking consomme le budget.
    #
    # UNIQUEMENT pour les modèles qui séparent leur réflexion du texte : la demander à un
    # modèle qui ne le fait pas la fait atterrir DANS la réponse, sous les yeux de
    # l'utilisateur (mistral-medium, 2026-08-26).
    from app.services.rag_generation_service import supports_structured_reasoning

    reasoning_kwargs = {}
    if settings.GENERATION_REASONING_EFFORT == "high":
        if supports_structured_reasoning(model):
            reasoning_kwargs["reasoning_effort"] = "high"
            tokens = max(tokens, settings.GENERATION_REASONING_MAX_TOKENS)
        else:
            logger.info(
                "[génération] reasoning non demandé : %s ne sépare pas sa réflexion du "
                "texte de réponse (elle finirait affichée à l'utilisateur).",
                model,
            )

    # Graine d'échantillonnage : rend deux générations identiques comparables. Sans elle,
    # une différence de réponse entre deux essais peut venir du hasard et non du changement
    # de retriever qu'on cherche à évaluer.
    if settings.GENERATION_SEED is not None:
        reasoning_kwargs["random_seed"] = settings.GENERATION_SEED

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
            # Périmètre de recherche confirmé (current_scope) : même logique de préservation
            # que l'ancre documentaire — écrit après le retrieval, il serait sinon effacé par
            # la sauvegarde du query_context de compréhension au tour suivant.
            if "current_scope" not in merged and previous.get("current_scope"):
                merged["current_scope"] = previous["current_scope"]
            conv.query_context = merged
            conv.updated_at = datetime.utcnow()
            s.add(conv)
            s.commit()


def _update_conversation_scope(conversation_id: int, scope: dict) -> None:
    """Persiste le périmètre de recherche confirmé (current_scope) dans query_context.

    Fusion non destructive (comme l'ancre documentaire) : réassigne le dict entier pour
    déclencher le suivi de modification JSON de SQLAlchemy.
    """
    with Session(engine) as s:
        conv = s.get(Conversation, conversation_id)
        if conv:
            qc = dict(conv.query_context or {})
            qc["current_scope"] = dict(scope or {})
            conv.query_context = qc
            conv.updated_at = datetime.utcnow()
            s.add(conv)
            s.commit()


def _update_conversation_documents(
    conversation_id: int, document_ids: List[int], intent: Optional[str] = None
) -> None:
    """Persiste les documents d'ancre (sujet documentaire courant) dans query_context.

    Fusionne dans le contexte existant sans l'écraser ; réassigne le dict entier pour
    déclencher le suivi de modification JSON de SQLAlchemy.

    ``intent`` mémorise l'intention qui a produit cette ancre : au tour suivant, un
    changement d'intention (référence → installation → SAV) signifie que le bon TYPE de
    document a changé, et l'ancre perd alors sa garantie de packing.
    """
    with Session(engine) as s:
        conv = s.get(Conversation, conversation_id)
        if conv:
            qc = dict(conv.query_context or {})
            qc["current_documents"] = [int(d) for d in document_ids]
            qc["current_documents_intent"] = (intent or "").strip() or None
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


class ScopeChoiceRequest(BaseModel):
    """Périmètre confirmé par l'utilisateur via la carte (champ → valeur ; '' = peu importe)."""
    values: Dict[str, str] = Field(default_factory=dict)


class SavStartInline(BaseModel):
    """Démarrage d'un Arbre SAV via le flux de chat (bouton / picker / chip)."""
    tree_slug: str = ""
    tree_id: Optional[int] = None
    entry_node_key: Optional[str] = None


class SpaceChatRequest(BaseModel):
    message: str
    model: str
    provider: str = "mistral"
    context: Optional[List[dict]] = None
    conversation_id: Optional[int] = None
    slot_action: Optional[SlotActionRequest] = None
    guided_choice: Optional[GuidedChoiceRequest] = None
    scope_choice: Optional[ScopeChoiceRequest] = None
    sav_start: Optional[SavStartInline] = None


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
    elected_document_ids: Optional[List[int]] = None,
    pinned_pages: Optional[Dict[int, List[int]]] = None,
) -> dict:
    """Contexte système de génération : CAG (documents entiers, fenêtre 256k) si activé,
    sinon fallback historique (passages tronqués). Les documents ANCRÉS (sujet courant de
    la conversation) sont toujours inclus dans le contexte CAG — garantie de continuité.
    Le budget de packing s'adapte à l'intent (CAG_BUDGET_BY_INTENT).

    ``elected_document_ids``/``pinned_pages`` (B6) : élection du juge de suffisance —
    documents élus packés en tête, pages citées promues seeds de la fenêtre gloutonne."""
    if settings.CAG_ENABLED:
        from app.services.context_packer_service import build_cag_context

        return build_cag_context(
            session,
            doc_passages,
            system_prompt=SPACE_CHAT_SYSTEM_PROMPT,
            anchor_document_ids=anchor_document_ids,
            intent=intent,
            elected_document_ids=elected_document_ids,
            pinned_pages=pinned_pages,
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


def _scope_streaming_response(card: dict, conversation_id: Optional[int]) -> StreamingResponse:
    """Réponse SSE d'une proposition de périmètre (human-in-the-loop) : émet l'événement
    `scope_proposal` (carte à boutons) puis termine le tour. AUCUN retrieval n'est lancé —
    le clic de l'utilisateur ré-entrera le pipeline avec ``scope_choice`` au tour suivant."""

    async def generate_scope():
        try:
            payload = dict(card)
            payload["conversation_id"] = conversation_id
            yield f"data: {json.dumps({'scope_proposal': payload})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.exception("Erreur dans le générateur de proposition de périmètre")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate_scope(), media_type="text/event-stream")


def _normalize_code(s: str) -> str:
    """Réduit une chaîne à ses caractères alphanumériques minuscules (pour comparer des codes)."""
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _passages_contain_codes(passages: List[dict], codes: List[str]) -> bool:
    """True si au moins un code demandé (≥3 car.) apparaît, normalisé, dans un passage."""
    norm_codes = [c for c in (_normalize_code(x) for x in (codes or [])) if len(c) >= 3]
    if not norm_codes:
        return False
    blob = " ".join(
        _normalize_code(p.get("passage_raw") or p.get("passage") or "") for p in (passages or [])
    )
    return any(c in blob for c in norm_codes)


async def _stream_llm_to_sse(
    context: List[dict],
    *,
    model: str,
    max_tokens: Optional[int],
    source_filter,
    sink: List[str],
    reasoning_sink: Optional[List[str]] = None,
    hold_messages: bool = False,
):
    """Streame une génération Mistral en événements SSE, filtre le bloc <sources> et
    accumule le texte AFFICHÉ dans ``sink``. Factorisé (P0.2) pour dédupliquer les
    tentatives full/eco/minimal ; propage httpx.HTTPStatusError à l'appelant (fallback).

    ``reasoning_sink`` (optionnel) accumule le thinking natif du modèle pour la trace de
    génération (bouton « cheminement »). Il n'est PAS la réponse persistée.

    ``hold_messages`` (B7c, vérification bloquante) : le texte est accumulé dans ``sink``
    mais AUCUN événement message n'est émis — l'appelant vérifie puis rejoue le tampon.
    Les événements thinking continuent de streamer (bulle « réflexion »)."""
    async for raw_chunk in chat_stream_wrapper(
        message="", model=model, context=context, max_tokens=max_tokens
    ):
        try:
            parsed = json.loads(raw_chunk)
        except json.JSONDecodeError:
            continue
        # Reasoning : relayer le thinking comme événement distinct (le client l'affiche
        # dans une bulle « réflexion » puis la masque à l'arrivée de la réponse). Ni filtré
        # <sources>, ni accumulé dans le sink de réponse ; capturé à part pour la trace.
        thinking = parsed.get("thinking")
        if thinking:
            if reasoning_sink is not None:
                reasoning_sink.append(thinking)
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
        if hold_messages:
            continue
        yield f"data: {json.dumps({'message': {'content': content}})}\n\n"


def _build_eco_context(
    session: Session,
    doc_passages: List[dict],
    anchor_document_ids: Optional[List[int]],
    user_message: str,
) -> List[dict]:
    """Contexte de SECOURS minimal après un Mistral 400 (P0.2) : CAG re-packé à un petit
    budget (sans images ni historique) + question. Cible la cause probable (contexte trop
    volumineux) au lieu de rejouer le contexte massif à l'identique.

    Ce repli abandonne les images. En mode CAG_IMAGE_ONLY il faut donc RÉTABLIR le texte :
    sinon il ne reste qu'un manifeste, le modèle répond sans aucun document — et invente
    avec assurance (constaté le 2026-08-26 : réponse plausible mais entièrement fausse).
    """
    if settings.CAG_ENABLED and doc_passages:
        from app.services.context_packer_service import build_cag_context

        with _forced_text_context_if_image_only():
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


@contextmanager
def _forced_text_context_if_image_only(reason: str = "repli sans images"):
    """Rétablit temporairement le texte documentaire quand CAG_IMAGE_ONLY est actif.

    Le mode image-only n'est tenable que si des images partent RÉELLEMENT. Dès qu'elles
    sautent — repli de secours, modèle non multimodal — il ne reste qu'un manifeste : le
    modèle répond alors sans aucun document, et invente avec assurance (constaté le
    2026-08-26 avec mistral-medium, absent de la liste des modèles vision).
    """
    if not settings.CAG_IMAGE_ONLY:
        yield
        return
    logger.warning(
        "[CAG] CAG_IMAGE_ONLY actif mais %s — le texte documentaire est RÉTABLI pour ne "
        "pas répondre sur un contexte vide.",
        reason,
    )
    settings.CAG_IMAGE_ONLY = False
    try:
        yield
    finally:
        settings.CAG_IMAGE_ONLY = True


async def _attempt_repair_generation(
    *,
    session: Session,
    space_id: int,
    full_context_draft: List[dict],
    draft_text: str,
    verification_result: dict,
    model: str,
    allowed_document_ids: Optional[List[int]],
) -> Optional[dict]:
    """Réparation d'une réponse non ancrée (B7c) — UNE régénération contrainte.

    Deux leviers, selon ce que la vérification a trouvé :
      - codes non étayés → recherche SQL ciblée des extraits faisant autorité pour ces
        codes (chunk pinning) : s'ils existent, le modèle reçoit la vérité verbatim ; s'ils
        n'existent pas, consigne d'écrire explicitement l'absence.
      - claims non étayés (cotes/normes) → consigne de suppression/aveu.

    Retourne {"text", "source_filter"} ou None si la réparation n'a rien produit.
    N'échoue jamais l'appelant."""
    from app.services.stream_source_filter import SourcesTagStreamFilter

    claims = list(verification_result.get("unsupported_claims") or [])
    codes = list(verification_result.get("unsupported_codes") or [])
    if not claims:
        return None

    repair_system = dict(full_context_draft[0]) if full_context_draft else None
    if not repair_system or repair_system.get("role") != "system":
        return None

    if codes:
        try:
            from app.services.page_retrieval_service import get_space_document_ids
            from app.services.reference_pinning_service import build_pinned_reference_block

            pin_doc_ids = allowed_document_ids or get_space_document_ids(
                session, space_id, document_filter="technical"
            )
            pinned_block, pinned = build_pinned_reference_block(session, pin_doc_ids, codes)
            if pinned_block:
                repair_system["content"] = (
                    repair_system["content"]
                    + "\n\n### EXTRAITS DE RÉFÉRENCE (réparation — font foi)\n"
                    + pinned_block
                )
                logger.info("[verify] réparation — extraits épinglés pour %s", pinned)
        except Exception as pin_err:  # noqa: BLE001
            logger.warning("[verify] épinglage de réparation ignoré : %s", pin_err)

    repair_instruction = (
        "CONTRÔLE QUALITÉ : les éléments suivants de ta réponse précédente ne figurent "
        "dans AUCUN des documents fournis : "
        + ", ".join(str(c) for c in claims[:8])
        + ". Régénère ta réponse en t'appuyant EXCLUSIVEMENT sur les documents du "
        "contexte. Pour tout élément introuvable, écris explicitement « les documents "
        "fournis ne précisent pas ... » au lieu de l'affirmer. N'invente aucune référence "
        "ni valeur. Conserve la ligne <sources> exigée en fin de réponse."
    )

    repair_context = [repair_system] + [dict(m) for m in full_context_draft[1:]]
    repair_context.append({"role": "assistant", "content": draft_text})
    repair_context.append({"role": "user", "content": repair_instruction})

    repair_filter = SourcesTagStreamFilter() if settings.CAG_ENABLED else None
    repair_sink: List[str] = []
    try:
        async for _ in _stream_llm_to_sse(
            repair_context,
            model=model,
            max_tokens=settings.CAG_MAX_COMPLETION_TOKENS if settings.CAG_ENABLED else None,
            source_filter=repair_filter,
            sink=repair_sink,
            hold_messages=True,
        ):
            pass
    except Exception as repair_err:  # noqa: BLE001
        logger.warning("[verify] régénération de réparation en échec : %s", repair_err)
        return None
    if repair_filter is not None:
        tail = repair_filter.finalize()
        if tail:
            repair_sink.append(tail)
    repaired_text = "".join(repair_sink).strip()
    if not repaired_text:
        return None
    return {"text": repaired_text, "source_filter": repair_filter}


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


# Plafond de taille du reasoning persisté dans la trace (le CoT natif peut être volumineux
# ; on borne pour ne pas gonfler message.metadata_json ni le payload SSE de l'événement done).
_TRACE_REASONING_MAX_CHARS = 8000
_TRACE_PASSAGES_MAX = 12


def _summarize_passages_for_trace(sources_data: Optional[List[dict]]) -> List[dict]:
    """Résumé compact des passages retenus pour la trace (titre, score, page, section).
    Réutilise ``sources_data`` déjà construit (aucun accès DB), ignore l'illustration."""
    summary: List[dict] = []
    for s in sources_data or []:
        if s.get("is_cropped_illustration"):
            continue
        summary.append({
            "document_title": s.get("document_title"),
            "document_id": s.get("document_id"),
            "score": s.get("score"),
            "page_no": s.get("page_no") or s.get("page_start"),
            "section": s.get("section"),
            "used_pages": s.get("used_pages"),
        })
        if len(summary) >= _TRACE_PASSAGES_MAX:
            break
    return summary


def _build_generation_trace(
    *,
    route: str,
    lw_result=None,
    retrieval_status: Optional[str] = None,
    retrieval_reason: Optional[str] = None,
    dynamic_k: Optional[int] = None,
    rerank_status: Optional[str] = None,
    nb_passages: Optional[int] = None,
    anchor_document_ids: Optional[List[int]] = None,
    requested_codes=None,
    pinned_codes: Optional[List[str]] = None,
    reasoning_parts: Optional[List[str]] = None,
    passages_summary: Optional[List[dict]] = None,
    verification: Optional[dict] = None,
    retry_info: Optional[dict] = None,
    anchor_boost: Optional[dict] = None,
    anchor_intent_changed: bool = False,
    cag_documents: Optional[List[dict]] = None,
    loop_info: Optional[dict] = None,
) -> dict:
    """Assemble le « cheminement » de génération persisté dans message.metadata_json['trace']
    et renvoyé dans l'événement SSE final. Alimente le bouton d'inspection côté UI.

    Tout le contenu doit rester JSON-sérialisable (colonne JSON + json.dumps du SSE)."""
    qc = (lw_result.query_context if lw_result else {}) or {}
    signals = lw_result.signals if (lw_result and lw_result.signals) else None

    reasoning_text = ""
    if reasoning_parts:
        reasoning_text = "".join(reasoning_parts).strip()
        if len(reasoning_text) > _TRACE_REASONING_MAX_CHARS:
            reasoning_text = reasoning_text[:_TRACE_REASONING_MAX_CHARS] + "…"

    trace: dict = {
        "route": route,
        "topic_shift": (bool(lw_result.topic_shift) if lw_result else None),
        "standalone_question": qc.get("standalone_question"),
        "current_topic": qc.get("current_topic") or qc.get("llm_current_topic"),
        "signals": (signals.model_dump() if signals else None),
        "reasoning": (reasoning_text or None),
        "reasoning_effort": settings.GENERATION_REASONING_EFFORT,
        "generation_temperature": settings.SPACE_CHAT_TEMPERATURE,
        "model": settings.MODEL_FAST,
    }

    if route == "rag":
        trace["retrieval"] = {
            "status": retrieval_status,
            "reason": retrieval_reason,
            "dynamic_k": dynamic_k,
            "rerank_status": rerank_status,
            "nb_passages": nb_passages,
            "anchor_document_ids": list(anchor_document_ids or []),
            "kag_enabled": settings.KAG_ENABLED,
            # Un tour biaisé par le sujet courant, ou dont les passages viennent d'un
            # retry sur une autre requête, doit être identifiable sans lire les logs.
            "anchor_boost": anchor_boost,
            "anchor_intent_changed": bool(anchor_intent_changed),
            "retry": retry_info,
        }
        # Documents réellement DONNÉS au modèle (≠ documents cités par lui) : sans ça,
        # un contexte à 3 documents s'affichait avec 1 seul passage mobilisé.
        if cag_documents:
            trace["packed_documents"] = [
                {
                    "index": d.get("index"),
                    "document_id": d.get("document_id"),
                    "document_title": d.get("document_title"),
                    "election_score": d.get("election_score"),
                    "score": d.get("score"),
                    "pages": d.get("pages"),
                    "seed_pages": d.get("seed_pages"),
                    "full_document": d.get("full_document"),
                }
                for d in cag_documents
            ]
        trace["kag"] = {
            "requested_codes": list(requested_codes or []),
            "pinned_codes": list(pinned_codes or []),
        }
        trace["passages"] = passages_summary or []
        # Boucle agentique (B8) : rounds du juge, actions de relance, élection — la
        # boucle doit être rejouable depuis la trace, pas depuis les logs.
        if loop_info:
            trace["loop"] = loop_info

    if verification:
        trace["verification"] = verification

    return trace


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
        # Confirmation de périmètre (scope_choice) : le message a déjà été persisté au
        # tour qui a produit la carte — ne pas le dédoubler. Un démarrage d'Arbre SAV
        # (sav_start) persiste son propre libellé plus bas.
        if not request.scope_choice and not request.sav_start:
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

    # ——— Arbre SAV : DÉMARRAGE explicite (bouton / picker / chip / deep-link) ———
    # 100 % déterministe : l'utilisateur a cliqué, aucune classification LLM en jeu.
    if request.sav_start and request.conversation_id:
        from app.models.guided_tree import GuidedTree
        from app.services.guided_flow_service import start_guided_session

        sav_tree = None
        if request.sav_start.tree_id is not None:
            sav_tree = session.get(GuidedTree, request.sav_start.tree_id)
        elif request.sav_start.tree_slug:
            sav_tree = session.exec(
                select(GuidedTree).where(GuidedTree.slug == request.sav_start.tree_slug)
            ).first()
        try:
            session.add(
                Message(
                    conversation_id=request.conversation_id,
                    role="user",
                    content=f"🔧 Diagnostic SAV : {sav_tree.title if sav_tree else request.sav_start.tree_slug}",
                    model=None,
                    provider=None,
                )
            )
            session.commit()
        except Exception:
            logger.exception("Erreur persistance message de démarrage SAV")

        gtr = start_guided_session(
            session,
            space_id=space_id,
            user_id=current_user.id,
            conversation_id=request.conversation_id,
            tree_slug=request.sav_start.tree_slug,
            tree_id=request.sav_start.tree_id,
            entry_node_key=request.sav_start.entry_node_key,
        )
        return _guided_streaming_response(
            gtr, request.conversation_id, forced_model, forced_provider
        )

    # ——— Arbre SAV : REPRISE d'un parcours actif (refonte 2026-07-30) ———
    # Traversée déterministe du snapshot publié — zéro retrieval, zéro LLM (sauf mapping
    # d'une réponse tapée sur les choix du nœud). Le guidé ne démarre plus JAMAIS depuis
    # une classification du message libre : le RAG répond toujours (voir chip plus bas).
    guided_active_state = None
    if request.conversation_id:
        from app.services.guided_flow_service import load_active_guided_state, run_guided_turn

        conv_for_guided = session.get(Conversation, request.conversation_id)
        guided_active_state = load_active_guided_state(
            conv_for_guided.query_context if conv_for_guided else None
        )

        if guided_active_state:
            logger.info("[chat] Arbre SAV — reprise du parcours actif")
            gtr = await run_guided_turn(
                session=session,
                space_id=space_id,
                user_id=current_user.id,
                conversation_id=request.conversation_id,
                user_message=request.message,
                guided_choice=request.guided_choice.model_dump() if request.guided_choice else None,
                active_state=guided_active_state,
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
    rag_user_message = request.message
    # Texte utilisé pour la RECHERCHE documentaire (peut différer du message de
    # génération : reformulation history-aware en question autonome).
    retrieval_query_text = request.message
    lw_result = None
    # Documents d'ancre du sujet courant (réutilisés pour biaiser le retrieval de ce tour).
    anchor_document_ids: List[int] = []
    # Vrai quand l'intention du tour diffère de celle qui a produit l'ancre : celle-ci
    # garde alors son boost de recherche mais perd sa garantie de packing (cf. A2).
    anchor_intent_changed = False

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
            # ——— L'ancre ne survit pas à un changement d'INTENTION ———
            # Le sujet peut rester identique (topic_shift=false) alors que le bon TYPE de
            # document change : « quelle référence ? » se répond dans un catalogue de
            # conception, « comment l'installer ? » dans un catalogue de fabrication.
            # Sans ce garde-fou, le document du tour précédent était packé de force et
            # celui qui contenait la procédure n'avait plus de place (cas mesuré 27/07).
            _prev_intent = (persisted_context.get("current_documents_intent") or "").strip()
            _this_intent = (
                (lw_result.signals.intent or "").strip()
                if (lw_result and lw_result.signals)
                else ""
            )
            if (
                anchor_document_ids
                and settings.CONVERSATION_ANCHOR_INTENT_GUARD
                and _prev_intent
                and _this_intent
                and _prev_intent != _this_intent
            ):
                anchor_intent_changed = True
                logger.info(
                    "[chat] ancre dégradée — intention %s → %s (boost conservé, packing forcé retiré)",
                    _prev_intent,
                    _this_intent,
                )
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

    # ——— Invitation Arbre SAV (déterministe — jamais à la place de la réponse) ———
    # Si le message a une connotation SAV (symptôme détecté / vocabulaire client /
    # intention troubleshooting) ET qu'un arbre PUBLIÉ matche via l'index d'entrée,
    # un chip « Lancer le diagnostic » est émis SOUS la réponse RAG. Sinon, la demande
    # alimente le backlog des angles morts (GuidedGap). Le RAG répond dans tous les cas.
    sav_suggestion: Optional[dict] = None
    if request.conversation_id and not guided_active_state:
        try:
            _sig = lw_result.signals if lw_result else None
            _detected = str(getattr(_sig, "detected_symptom", "") or "") if _sig else ""
            _freeform = str(getattr(_sig, "symptom_freeform", "") or "") if _sig else ""
            _intent = str(getattr(_sig, "intent", "") or "") if _sig else ""
            if _detected or _freeform or _intent == "troubleshooting":
                from app.services.guided_entry_index_service import match_entry
                from app.services.guided_gap_service import record_gap

                _entry = match_entry(
                    session,
                    space_id=space_id,
                    query_text=request.message,
                    detected_symptom=_detected,
                )
                if _entry is not None:
                    sav_suggestion = {
                        "tree_slug": _entry.tree_slug,
                        "tree_title": _entry.tree_title,
                        "entry_node_key": _entry.entry_node_key,
                        "symptom_slug": _entry.symptom_slug,
                        "method": _entry.method,
                        "score": round(float(_entry.score), 3),
                    }
                    logger.info(
                        "[chat] Arbre SAV — invitation %s (méthode=%s score=%.2f)",
                        _entry.tree_slug,
                        _entry.method,
                        _entry.score,
                    )
                else:
                    record_gap(
                        session,
                        space_id=space_id,
                        detected_symptom=_detected,
                        query_text=request.message,
                        conversation_id=request.conversation_id,
                    )
        except Exception:
            logger.exception("[chat] suggestion Arbre SAV en échec (non bloquant)")

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
        reasoning_parts_direct: List[str] = []

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
                            reasoning_sink=reasoning_parts_direct,
                        ):
                            yield sse_event

                        final_response = "".join(assistant_response)
                        stream_run.end(outputs={"response": final_response})

                    pipeline_run.end(outputs={
                        "response_chars": len(final_response),
                    })

                    direct_trace = _build_generation_trace(
                        route="direct",
                        lw_result=lw_result,
                        reasoning_parts=reasoning_parts_direct,
                    )
                    assistant_message_id = None
                    if request.conversation_id and assistant_response:
                        try:
                            assistant_message_id = _persist_assistant_reply(
                                request.conversation_id,
                                final_response,
                                forced_model,
                                forced_provider,
                                None,
                                metadata_json={"trace": direct_trace},
                            )
                            logger.info(
                                "Réponse assistant directe sauvegardée (space chat), conversation %s",
                                request.conversation_id,
                            )
                        except Exception:
                            logger.exception("Erreur sauvegarde réponse directe assistant (space chat)")

                    yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id, 'trace': direct_trace})}\n\n"

            except MistralRateLimitError as e:
                logger.warning("Limite de débit Mistral (stream_space_chat_message direct): %s", e)
                error_msg_to_yield = str(e)
            except Exception as e:
                logger.exception("Erreur dans le générateur stream_space_chat_message (direct)")
                error_msg_to_yield = str(e)
            
            if error_msg_to_yield:
                yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"

        return StreamingResponse(generate_direct(), media_type="text/event-stream")

    if settings.QUERY_UNDERSTANDING_ENABLED and lw_result and lw_result.ready_for_retrieval:
        retrieval_queries = lw_result.retrieval_queries
        rag_user_message = (
            lw_result.query_context.get("original_user_message") or request.message
        )
        # Recherche documentaire : privilégie la question autonome reformulée
        # (résout les messages de suivi type "et le tgy3834 ?"), sinon fallback.
        retrieval_query_text = (
            lw_result.query_context.get("standalone_question")
            or rag_user_message
        )

    # ——— Périmètre de recherche confirmé (human-in-the-loop) ———
    # Décide QUOI/OÙ chercher AVANT le retrieval. Gardé par SCOPE_CONFIRMATION_ENABLED +
    # SCOPE_MODE (off = aucun impact). Trois issues : carte à confirmer (stop du tour),
    # confirmation reçue (applique le choix), ou périmètre déterminé (applique en silence).
    allowed_document_ids: Optional[List[int]] = None
    scope_choice_confirmed = False
    if (
        settings.SCOPE_CONFIRMATION_ENABLED
        and settings.SCOPE_MODE != "off"
        and lw_result
    ):
        from app.services.scope_resolver_service import (
            PROPOSE_CARD,
            build_scope_card,
            compute_allowed_document_ids,
            compute_space_scope_stats,
            resolve_scope,
        )

        _persisted_qc = None
        if request.conversation_id:
            _conv_scope = session.get(Conversation, request.conversation_id)
            _persisted_qc = _conv_scope.query_context if _conv_scope else None
        _inherited = (_persisted_qc or {}).get("current_scope") if _persisted_qc else None

        try:
            if request.scope_choice is not None:
                # L'utilisateur vient de valider la carte : appliquer son choix directement.
                _chosen = {k: v for k, v in (request.scope_choice.values or {}).items() if v}
                allowed_document_ids = (
                    compute_allowed_document_ids(session, space_id, _chosen) if _chosen else None
                )
                if request.conversation_id:
                    _update_conversation_scope(request.conversation_id, _chosen)
                # Un périmètre explicitement confirmé prime sur le sujet mémorisé : sans
                # cette purge, les documents du tour précédent (souvent ceux que la carte
                # sert justement à écarter) restaient ancrés et repackés de force.
                scope_choice_confirmed = True
                logger.info("[chat] Périmètre confirmé par l'utilisateur : %s", _chosen)
            else:
                _stats = compute_space_scope_stats(session, space_id)
                _resolution = resolve_scope(
                    lw_result.signals.model_dump() if lw_result.signals else {},
                    space_stats=_stats,
                    inherited_scope=_inherited,
                    topic_shift=bool(lw_result.topic_shift),
                    intent=(lw_result.signals.intent if lw_result.signals else None),
                )
                if _resolution.decision == PROPOSE_CARD:
                    logger.info(
                        "[chat] Périmètre — carte proposée (champs demandés: %s)",
                        [f.key for f in _resolution.asked_fields()],
                    )
                    return _scope_streaming_response(
                        build_scope_card(_resolution), request.conversation_id
                    )
                _applied = _resolution.applied_scope
                allowed_document_ids = (
                    compute_allowed_document_ids(session, space_id, _applied) if _applied else None
                )
                if request.conversation_id and _applied:
                    _update_conversation_scope(request.conversation_id, _applied)
        except Exception as _scope_err:
            logger.warning("[chat] résolution de périmètre ignorée (erreur): %s", _scope_err)
            allowed_document_ids = None

    # ——— L'ancre documentaire ne contourne plus le périmètre ———
    # L'ancre boostait le retrieval ET forçait le packing de ses documents SANS jamais être
    # croisée avec le périmètre : les documents (souvent faux) d'un tour raté occupaient
    # alors tous les slots CAG, et cliquer sur la carte de filtres ne changeait rien.
    if anchor_document_ids and scope_choice_confirmed:
        anchor_document_ids = []
        if request.conversation_id:
            _update_conversation_documents(request.conversation_id, [])
        logger.info("[chat] Périmètre confirmé → ancre documentaire purgée")
    elif anchor_document_ids and allowed_document_ids is not None:
        _allowed = {int(d) for d in allowed_document_ids}
        _kept = [d for d in anchor_document_ids if int(d) in _allowed]
        if len(_kept) != len(anchor_document_ids):
            logger.info(
                "[chat] Ancre restreinte au périmètre — %d/%d document(s) conservé(s)",
                len(_kept),
                len(anchor_document_ids),
            )
        anchor_document_ids = _kept

    # Ancre utilisée pour le PACKING (distincte de celle du retrieval) : quand l'intention
    # a changé, le document du tour précédent garde son boost de recherche mais perd son
    # slot CAG réservé — sinon il occupe la place du document qui porte vraiment la réponse.
    cag_anchor_document_ids = [] if anchor_intent_changed else list(anchor_document_ids or [])

    step_label = "2/5" if settings.QUERY_UNDERSTANDING_ENABLED else "2/4"
    logger.info("[chat] Étape %s — retrieval hybride (ColPali + BM25)", step_label)
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

        # Sentinelle des overrides de relance (B5) : None est une valeur légitime
        # (« pas de queries pré-dérivées ») — la sentinelle distingue « inchangé ».
        _UNSET = object()

        async def _do_retrieval(
            allowed: Optional[List[int]],
            *,
            query_text: Optional[str] = None,
            anchors=_UNSET,
            k: Optional[int] = None,
            queries=_UNSET,
        ):
            # Les overrides servent aux relances de la boucle agentique (B5) : requête
            # réécrite par le juge, désancrage, k élargi. Sans override, comportement
            # strictement identique à l'historique.
            _anchors = (anchor_document_ids or None) if anchors is _UNSET else (anchors or None)
            return await search_technical_passages(
                session=session,
                space_id=space_id,
                query_text=query_text if query_text is not None else retrieval_query_text,
                user_id=current_user.id,
                k=k or RAG_TOP_K,
                queries=retrieval_queries if queries is _UNSET else queries,
                signals=lw_result.signals if lw_result and lw_result.signals else None,
                anchor_document_ids=_anchors,
                allowed_document_ids=allowed,
            )

        async def _run_retrieval(allowed: Optional[List[int]], **overrides):
            # Budget temps global (P0.3) : un canal qui freeze (ColPali CPU, MaxSim LanceDB)
            # ne doit pas bloquer indéfiniment — au-delà du budget, dégradation gracieuse.
            try:
                if settings.RETRIEVAL_TIMEOUT_S and settings.RETRIEVAL_TIMEOUT_S > 0:
                    return await asyncio.wait_for(
                        _do_retrieval(allowed, **overrides), timeout=settings.RETRIEVAL_TIMEOUT_S
                    )
                return await _do_retrieval(allowed, **overrides)
            except asyncio.TimeoutError:
                logger.error(
                    "[chat] retrieval au-delà du budget %.0fs → dégradation gracieuse (0 passage)",
                    settings.RETRIEVAL_TIMEOUT_S,
                )
                return {
                    "passages": [], "images": [], "status": "degraded_timeout",
                    "reason": "retrieval_timeout",
                }

        retrieval = await _run_retrieval(allowed_document_ids)
        # Élargissement automatique : un périmètre trop étroit (aucun document) ne doit
        # JAMAIS produire un « pas documenté » à tort → on relance sans périmètre.
        if allowed_document_ids is not None and retrieval.get("reason") == "scope_empty":
            logger.info(
                "[chat] Périmètre vide (0 document) → élargissement automatique à tout l'espace"
            )
            retrieval = await _run_retrieval(None)
        doc_passages = retrieval["passages"]
        retrieval_status = retrieval["status"]
        retrieval_reason = retrieval.get("reason")
        retrieval_images = retrieval.get("images") or []
        dynamic_k = retrieval.get("dynamic_k")
        rerank_status = retrieval.get("rerank_status")

        # ——— Retry automatique sur RÉFÉRENCE introuvable (1 passe) ———
        # Le retriever est sensible à la formulation (« référence X » vs « crémone X ») :
        # si la question porte un code et qu'AUCUN passage ne le contient, on relance UNE
        # recherche ciblée sur le code seul (recall élevé). S'il reste introuvable, on
        # demandera une précision à l'utilisateur (cf. bloc génération) au lieu d'un
        # « non documenté » en cul-de-sac.
        reference_not_found: List[str] = []
        retry_trace: Dict[str, Any] = {"triggered": False}
        try:
            from app.services.coverage_service import extract_message_reference_codes

            _req_codes = extract_message_reference_codes(
                retrieval_query_text,
                request.message,
                *(
                    (lw_result.signals.detected_references or [])
                    if (lw_result and lw_result.signals)
                    else []
                ),
            )
        except Exception:
            _req_codes = []

        if _req_codes and not _passages_contain_codes(doc_passages, _req_codes):
            code_query = " ".join(_req_codes)
            # Cheminement (T2) : sans cette trace, rien ne distingue un tour à un retrieval
            # d'un tour à deux — alors que le retry interroge le retriever avec une AUTRE
            # question (les codes seuls) et que ce sont SES passages qui sont packés.
            retry_trace = {
                "triggered": True,
                "requested_codes": list(_req_codes),
                "retry_query": code_query,
                "standalone_question": retrieval_query_text,
                "hits_before": list(retrieval.get("top_hits") or []),
            }
            logger.info(
                "[chat] Référence(s) %s absente(s) des passages → retry recherche ciblée",
                _req_codes,
            )
            try:
                retry = await search_technical_passages(
                    session=session,
                    space_id=space_id,
                    query_text=code_query,
                    user_id=current_user.id,
                    k=RAG_TOP_K,
                    signals=lw_result.signals if lw_result and lw_result.signals else None,
                    anchor_document_ids=anchor_document_ids or None,
                    allowed_document_ids=allowed_document_ids,
                )
            except Exception as _retry_err:
                logger.warning("[chat] retry ciblé référence échoué : %s", _retry_err)
                retry = None
            if retry and _passages_contain_codes(retry.get("passages") or [], _req_codes):
                logger.info("[chat] Retry ciblé — référence(s) trouvée(s), passages remplacés")
                retrieval = retry
                doc_passages = retrieval["passages"]
                retrieval_status = retrieval["status"]
                retrieval_reason = retrieval.get("reason")
                retrieval_images = retrieval.get("images") or []
                retry_trace.update(
                    {
                        "outcome": "found",
                        "passages_replaced": True,
                        "hits_after": list(retrieval.get("top_hits") or []),
                        "warning": (
                            "Le classement packé est celui du retry (requête « "
                            f"{code_query} »), pas celui de la question de l'utilisateur."
                        ),
                    }
                )
            else:
                reference_not_found = list(_req_codes)
                # Ce que le retry a RAMENÉ compte autant que son échec : sans ces hits, la
                # trace laisse croire qu'il n'a rien trouvé, alors qu'il a bien ramené des
                # pages — simplement aucune ne portait le code.
                retry_trace.update(
                    {
                        "outcome": "not_found",
                        "passages_replaced": False,
                        "hits_after": list((retry or {}).get("top_hits") or []),
                    }
                )
                logger.info(
                    "[chat] Retry ciblé — référence(s) %s toujours introuvable(s) → clarification",
                    _req_codes,
                )

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

    # ——— Boucle agentique (B4/B5, plan 2026-07-29) : juge de suffisance + relances ———
    # AVANT la génération, un modèle distinct lit les DOSSIERS CANDIDATS (pack-juge) et
    # statue : élire (documents + pages → packing) ou relancer la recherche (reformulation,
    # filtres, désancrage), sous deadline dure. Mode shadow = verdicts tracés, jamais
    # actionnés. Flag off = pipeline strictement inchangé. Un juge défaillant (timeout,
    # JSON invalide, preuve introuvable) ne dégrade JAMAIS le tour : on génère comme
    # aujourd'hui, l'échec est tracé.
    loop_trace: Optional[Dict[str, Any]] = None
    judge_elected_ids: List[int] = []
    judge_pinned_pages: Dict[int, List[int]] = {}
    judge_note_block: Optional[str] = None
    loop_exhausted_missing: Optional[str] = None
    if settings.AGENTIC_LOOP_ENABLED and settings.CAG_ENABLED and doc_passages:
        from app.services.coverage_service import (
            coverage_status as _loop_coverage_status,
            extract_message_reference_codes as _loop_extract_codes,
        )
        from app.services.retrieval_judge_service import (
            apply_judge_action,
            build_judge_note_block,
            build_judge_pack,
            judge_candidates,
            passages_pool_key,
        )
        from app.services.slot_catalog import expected_content_for_intent

        _loop_mode = "active" if settings.AGENTIC_LOOP_MODE == "active" else "shadow"
        loop_trace = {"mode": _loop_mode, "rounds": [], "deadline_hit": False}
        _loop_deadline = _time.monotonic() + max(5.0, settings.LOOP_DEADLINE_S)
        _loop_intent = lw_result.signals.intent if (lw_result and lw_result.signals) else None
        _expected_content = expected_content_for_intent(_loop_intent)
        try:
            _loop_codes = _loop_extract_codes(
                retrieval_query_text,
                request.message,
                *(
                    (lw_result.signals.detected_references or [])
                    if (lw_result and lw_result.signals)
                    else []
                ),
            )
        except Exception:  # noqa: BLE001
            _loop_codes = []
        _max_rounds = 1 + (max(0, settings.JUDGE_MAX_RETRIES) if _loop_mode == "active" else 0)
        _loop_query = retrieval_query_text
        _loop_allowed = allowed_document_ids
        _loop_anchor_override = _UNSET
        _loop_k: Optional[int] = None
        _prev_pool = passages_pool_key(doc_passages)
        _prev_missing: Optional[str] = None

        try:
            for _round in range(1, _max_rounds + 1):
                judge_pack = await asyncio.to_thread(
                    build_judge_pack, session, doc_passages, intent=_loop_intent
                )
                _judge_images: List[str] = []
                if settings.JUDGE_IMAGES_MODE == "always" or (
                    settings.JUDGE_IMAGES_MODE == "auto"
                    and _loop_intent in ("installation", "troubleshooting")
                ):
                    try:
                        from app.services.context_packer_service import select_cag_images

                        _judge_images, _ = await asyncio.to_thread(
                            select_cag_images,
                            session,
                            judge_pack["cag_documents"],
                            doc_passages,
                            max_images=settings.JUDGE_MAX_IMAGES,
                        )
                    except Exception as _ji_err:  # noqa: BLE001
                        logger.warning("[loop] images du juge ignorées : %s", _ji_err)

                _coverage_line = None
                if _loop_codes:
                    _cov = _loop_coverage_status(
                        context_text=judge_pack["context_text"],
                        requested_codes=_loop_codes,
                        doc_passages=doc_passages,
                    )
                    _missing_codes = _cov.get("missing_codes") or []
                    _coverage_line = (
                        ("ABSENTES des dossiers : " + ", ".join(_missing_codes))
                        if _missing_codes
                        else ("toutes présentes dans les dossiers : " + ", ".join(_loop_codes))
                    )

                verdict = await judge_candidates(
                    question=_loop_query,
                    intent=_loop_intent,
                    expected_content=_expected_content,
                    judge_pack=judge_pack,
                    coverage_line=_coverage_line,
                    images=_judge_images or None,
                    round_index=_round,
                    previous_missing=_prev_missing,
                )
                _round_trace = {
                    "round": _round,
                    "query": _loop_query,
                    "status": verdict.get("status"),
                    "status_reason": verdict.get("status_reason"),
                    "verdict": verdict.get("verdict"),
                    "confidence": verdict.get("confidence"),
                    "evidence": (verdict.get("evidence") or "")[:300],
                    "evidence_verified": verdict.get("evidence_verified"),
                    "missing": (verdict.get("missing") or "")[:300],
                    "elected": [
                        {
                            "document_id": e.get("document_id"),
                            "document_index": e.get("document_index"),
                            "pages": e.get("pages"),
                            "role": e.get("role"),
                        }
                        for e in (verdict.get("elected") or [])
                    ],
                    "candidates": [
                        {"document_id": d.get("document_id"), "index": d.get("index")}
                        for d in (judge_pack.get("cag_documents") or [])
                    ],
                    "images": len(_judge_images),
                    "duration_ms": verdict.get("duration_ms"),
                    "model": verdict.get("model"),
                }
                loop_trace["rounds"].append(_round_trace)

                if _loop_mode == "shadow":
                    break
                if verdict["status"] != "ok":
                    break
                if verdict["verdict"] == "sufficient":
                    judge_elected_ids = [e["document_id"] for e in verdict["elected"]]
                    judge_pinned_pages = {
                        e["document_id"]: e["pages"]
                        for e in verdict["elected"]
                        if e.get("pages")
                    }
                    judge_note_block = build_judge_note_block(verdict)
                    break

                # Verdict « insuffisant » → relance informée, si budget et action utile.
                _prev_missing = verdict.get("missing") or None
                if _round >= _max_rounds:
                    loop_exhausted_missing = _prev_missing or "information demandée introuvable"
                    break
                if _time.monotonic() > _loop_deadline:
                    loop_trace["deadline_hit"] = True
                    loop_exhausted_missing = _prev_missing or "information demandée introuvable"
                    break
                _action = apply_judge_action(verdict, current_query=_loop_query)
                if not _action:
                    _round_trace["relaunch"] = "no_action"
                    loop_exhausted_missing = _prev_missing or "information demandée introuvable"
                    break
                _round_trace["action"] = _action.get("label")
                _loop_query = _action.get("query_text") or _loop_query
                if _action.get("widen_scope"):
                    _loop_allowed = None
                if _action.get("restrict_document_id"):
                    _loop_allowed = [int(_action["restrict_document_id"])]
                if _action.get("drop_anchor"):
                    _loop_anchor_override = []
                if _action.get("raise_k"):
                    _loop_k = max(RAG_TOP_K, min(2 * RAG_TOP_K, settings.RERANK_POOL))

                _relaunch = await _run_retrieval(
                    _loop_allowed,
                    query_text=_loop_query,
                    anchors=_loop_anchor_override,
                    k=_loop_k,
                    queries=None,
                )
                _new_passages = _relaunch.get("passages") or []
                if not _new_passages:
                    _round_trace["relaunch"] = "empty"
                    loop_exhausted_missing = _prev_missing or "information demandée introuvable"
                    break
                _new_passages = enrich_colpali_passages_with_pymupdf(session, _new_passages)
                if settings.QUERY_UNDERSTANDING_ENABLED and lw_result and lw_result.signals:
                    from app.services.retrieval_boost_service import (
                        apply_soft_boosts_to_passages as _loop_boosts,
                    )

                    _new_passages = _loop_boosts(
                        session=session, passages=_new_passages, signals=lw_result.signals
                    )
                _new_pool = passages_pool_key(_new_passages)
                if _new_pool == _prev_pool:
                    _round_trace["relaunch"] = "no_progress"
                    loop_exhausted_missing = _prev_missing or "information demandée introuvable"
                    break
                _round_trace["relaunch"] = "replaced"
                _prev_pool = _new_pool
                doc_passages = _new_passages
                retrieval = _relaunch
                retrieval_status = _relaunch["status"]
                retrieval_reason = _relaunch.get("reason")
                retrieval_images = _relaunch.get("images") or []
                dynamic_k = _relaunch.get("dynamic_k")
                rerank_status = _relaunch.get("rerank_status")
        except Exception as _loop_err:  # noqa: BLE001
            logger.exception("[loop] boucle agentique interrompue — génération inchangée")
            loop_trace["error"] = str(_loop_err)[:200]

    # Mémorise les documents dominants de ce tour comme ancre du sujet courant (réutilisée
    # pour biaiser le retrieval du prochain tour, tant qu'il n'y a pas de changement de sujet).
    if settings.CONVERSATION_ANCHOR_ENABLED and request.conversation_id and doc_passages:
        from app.services.retrieval_boost_service import compute_anchor_documents

        new_anchor = compute_anchor_documents(
            doc_passages, max_docs=settings.CONVERSATION_ANCHOR_MAX_DOCS
        )
        if new_anchor:
            _update_conversation_documents(
                request.conversation_id,
                new_anchor,
                intent=(lw_result.signals.intent if (lw_result and lw_result.signals) else None),
            )

    logger.info(
        "[chat] Étape %s — contexte RAG (%d passages, status=%s, dynamic_k=%s, rerank=%s)",
        "3/5" if settings.QUERY_UNDERSTANDING_ENABLED else "3/4",
        len(doc_passages),
        retrieval_status,
        dynamic_k,
        rerank_status,
    )

    # Garde-fou du mode 100 % PNG : sans modèle multimodal, AUCUNE image ne partira et le
    # contexte se réduirait à un manifeste — le modèle répondrait de mémoire. On rebascule
    # donc sur le texte pour toute la construction du contexte de ce tour.
    _image_only_impossible = settings.CAG_IMAGE_ONLY and not is_vision_model(forced_model)

    @contextmanager
    def _context_mode():
        if _image_only_impossible:
            with _forced_text_context_if_image_only(
                f"le modèle {forced_model} n'accepte pas d'images"
            ):
                yield
        else:
            yield

    # Construire le contexte système à partir des passages techniques
    # Si low confidence : injecter un prompt spécial pour forcer la clarification
    if retrieval_status == "low_confidence_clarification":
        with _context_mode():
            space_context_draft = _build_generation_context(
                session,
                doc_passages,
                anchor_document_ids=cag_anchor_document_ids or None,
                intent=(lw_result.signals.intent if lw_result and lw_result.signals else None),
                elected_document_ids=judge_elected_ids or None,
                pinned_pages=judge_pinned_pages or None,
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
        with _context_mode():
            space_context_draft = _build_generation_context(
                session,
                doc_passages,
                anchor_document_ids=cag_anchor_document_ids or None,
                intent=(lw_result.signals.intent if lw_result and lw_result.signals else None),
                elected_document_ids=judge_elected_ids or None,
                pinned_pages=judge_pinned_pages or None,
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

    # Épinglage par référence : recherche SQL directe du chunk faisant autorité pour un
    # code demandé (densité de spécification), sans graphe d'entités.
    pinned_codes: List[str] = []
    if requested_codes:
        try:
            from app.services.page_retrieval_service import get_space_document_ids
            from app.services.reference_pinning_service import (
                build_pinned_reference_block,
            )

            pin_doc_ids = allowed_document_ids or get_space_document_ids(
                session, space_id, document_filter="technical"
            )
            pinned_block, pinned_codes = build_pinned_reference_block(
                session, pin_doc_ids, requested_codes
            )
            if pinned_block:
                space_context_draft["content"] += "\n\n" + pinned_block
                logger.info("[chat] épinglage référence — codes=%s", pinned_codes)
        except Exception as pin_err:
            logger.warning("[chat] épinglage référence ignoré : %s", pin_err)

    coverage_block = build_coverage_block(
        context_text=space_context_draft.get("content") or "",
        requested_codes=requested_codes,
        doc_passages=doc_passages,
        pinned_codes=pinned_codes,
        retrieval_status=retrieval_status,
    )
    space_context_draft["content"] += "\n\n" + coverage_block

    # ——— Note du juge (B6) / aveu structuré (B5) ———
    # Même zone de forte attention (fin du message système) que la COUVERTURE. La note
    # pointe les pages VALIDÉES par le contrôle documentaire ; l'aveu remplace le
    # « je comble le trou » par un constat explicite de ce qui manque.
    if judge_note_block:
        space_context_draft["content"] += "\n\n" + judge_note_block
    elif loop_exhausted_missing:
        space_context_draft["content"] += (
            "\n\n⚠️ CONTRÔLE DOCUMENTAIRE (fait foi) : la recherche a été relancée sans "
            "trouver l'information demandée (" + loop_exhausted_missing + "). "
            "Dis explicitement ce que les documents fournis contiennent d'utile et ce qui "
            "manque. Ne comble JAMAIS le manque par déduction, connaissance générale ou "
            "référence voisine ; propose à l'utilisateur UNE précision courte qui "
            "permettrait de relancer la recherche."
        )
        logger.info(
            "[loop] relances épuisées → génération en aveu structuré (manque : %s)",
            loop_exhausted_missing,
        )

    # Référence introuvable même après retry ciblé (cf. bloc retrieval) : au lieu d'un
    # « non documenté » en cul-de-sac, demander UNE précision pour relancer la recherche.
    if reference_not_found:
        _suppliers_present = ""
        try:
            from app.services.scope_resolver_service import compute_space_scope_stats

            _sup = sorted(compute_space_scope_stats(session, space_id).get("supplier") or [])
            if _sup:
                _suppliers_present = " L'espace couvre : " + ", ".join(_sup) + "."
        except Exception:
            pass
        space_context_draft["content"] += (
            "\n\n⚠️ IMPORTANT : la ou les référence(s) "
            + ", ".join(reference_not_found)
            + " n'apparaissent dans AUCUN passage ci-dessus, même après une recherche ciblée. "
            "Ne réponds PAS « non documentée » comme réponse finale. À la place, demande à "
            "l'utilisateur UNE précision courte qui permettrait de relancer la recherche : "
            "le FOURNISSEUR/la marque concernée, la gamme, ou une reformulation avec plus de "
            "contexte." + _suppliers_present
        )
        logger.info(
            "[chat] Référence(s) %s introuvable(s) → génération orientée clarification",
            reference_not_found,
        )

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
    # Accumule le reasoning natif du modèle (thinking) pour la trace de génération.
    reasoning_parts: List[str] = []

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
                
                static_trace = _build_generation_trace(
                    route="rag",
                    lw_result=lw_result,
                    retrieval_status=retrieval_status,
                    retrieval_reason=retrieval_reason,
                    dynamic_k=dynamic_k,
                    rerank_status=rerank_status,
                    nb_passages=0,
                    anchor_document_ids=anchor_document_ids,
                    requested_codes=requested_codes,
                    pinned_codes=pinned_codes,
                )
                assistant_message_id = None
                if request.conversation_id:
                    try:
                        assistant_message_id = _persist_assistant_reply(
                            request.conversation_id,
                            static_reply,
                            forced_model,
                            forced_provider,
                            None,
                            metadata_json={"trace": static_trace},
                        )
                    except Exception:
                        logger.exception("Erreur sauvegarde réponse statique assistant (space chat)")

                yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id, 'trace': static_trace})}\n\n"
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

                    # ——— Vérification bloquante (B7c) : mode tampon ———
                    # Le texte est généré SANS être émis, vérifié (codes + cotes contre le
                    # contexte complet + juge LLM distinct), réparé une fois si besoin,
                    # PUIS rejoué au client. VERIFY_BLOCKING=false = flux historique.
                    verify_active = bool(
                        settings.VERIFY_ENABLED
                        and request.conversation_id
                        and space_context_draft.get("content")
                    )
                    buffer_mode = bool(verify_active and settings.VERIFY_BLOCKING)

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
                                reasoning_sink=reasoning_parts,
                                hold_messages=buffer_mode,
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
                            if not buffer_mode:
                                yield f"data: {json.dumps({'message': {'content': _tail}})}\n\n"

                    final_response = "".join(assistant_response)
                    stream_run.end(outputs={"response": final_response})

                # ——— Gate de vérification (B7c) — le texte n'a PAS encore été émis ———
                # En mode 100 % PNG le contexte ne contient plus de texte : l'ancrage
                # compare la réponse à un manifeste et déclarerait inventée TOUTE valeur
                # lue sur une image. On désactive donc le gate — le test tourne sans filet,
                # ce qui est acceptable pour une expérimentation, jamais en production.
                verification_result = None
                if settings.CAG_IMAGE_ONLY and buffer_mode and assistant_response:
                    logger.warning(
                        "[verify] CAG_IMAGE_ONLY actif — vérification d'ancrage DÉSACTIVÉE "
                        "(pas de texte à confronter à la réponse)"
                    )
                elif buffer_mode and assistant_response:
                    # Pages citées par le modèle via <sources> : le juge doit voir en
                    # priorité ce sur quoi la réponse s'appuie.
                    _cited_pages_early: Dict[int, List[int]] = {}
                    if source_filter is not None and cag_documents_ctx:
                        _docs_by_index = {
                            int(d.get("index")): d
                            for d in cag_documents_ctx
                            if d.get("index") is not None
                        }
                        for _u in source_filter.used_documents or []:
                            _doc = _docs_by_index.get(_u.get("doc"))
                            _pages = [p for p in (_u.get("pages") or []) if isinstance(p, int)]
                            if _doc and _pages:
                                _cited_pages_early.setdefault(
                                    int(_doc["document_id"]), []
                                ).extend(_pages)
                    try:
                        from app.services.response_verification_service import verify_response

                        verification_result = await verify_response(
                            question=retrieval_query_text,
                            response_text="".join(assistant_response),
                            context_text=space_context_draft["content"],
                            model=settings.effective_verify_model,
                            document_blocks=space_context_draft.get("cag_document_blocks"),
                            cag_documents=space_context_draft.get("cag_documents"),
                            cited_pages=_cited_pages_early,
                        )
                    except Exception as _verif_err:  # noqa: BLE001
                        logger.warning(
                            "Vérification bloquante en échec — émission sans gate : %s",
                            _verif_err,
                        )
                        verification_result = None

                    if verification_result is not None:
                        if verification_result.get("ok"):
                            verification_result["action"] = "passed"
                        else:
                            verification_result["action"] = "flagged"
                            for _repair_round in range(max(0, settings.VERIFY_MAX_REPAIRS)):
                                _repair = await _attempt_repair_generation(
                                    session=session,
                                    space_id=space_id,
                                    full_context_draft=full_context_draft,
                                    draft_text="".join(assistant_response),
                                    verification_result=verification_result,
                                    model=forced_model,
                                    allowed_document_ids=allowed_document_ids,
                                )
                                if not _repair:
                                    break
                                # Re-vérification PROGRAMMATIQUE seulement (codes + cotes,
                                # contexte complet) : rapide, insensible au juge LLM.
                                from app.services.response_verification_service import (
                                    check_grounding,
                                    check_reference_grounding,
                                )

                                _rep_text = _repair["text"]
                                _rep_claims = check_grounding(
                                    _rep_text, space_context_draft["content"]
                                )
                                if settings.VERIFY_CODE_GROUNDING:
                                    for _c in check_reference_grounding(
                                        _rep_text, space_context_draft["content"]
                                    ):
                                        if _c not in _rep_claims:
                                            _rep_claims.append(_c)
                                _before_claims = len(
                                    verification_result.get("unsupported_claims") or []
                                )
                                if _rep_claims and len(_rep_claims) >= _before_claims:
                                    logger.warning(
                                        "[verify] réparation sans progrès (%s) — texte "
                                        "initial conservé, réponse marquée",
                                        _rep_claims,
                                    )
                                    break
                                assistant_response.clear()
                                assistant_response.append(_rep_text)
                                if _repair.get("source_filter") is not None:
                                    source_filter = _repair["source_filter"]
                                verification_result["repaired"] = True
                                verification_result["unsupported_after_repair"] = _rep_claims
                                verification_result["action"] = (
                                    "repaired" if not _rep_claims else "flagged"
                                )
                                logger.info(
                                    "[verify] réponse réparée — non étayés %d → %d",
                                    _before_claims,
                                    len(_rep_claims),
                                )
                                break
                    final_response = "".join(assistant_response)

                # Replay du tampon (mode buffer) : effet machine à écrire simulé, le texte
                # émis est le texte VÉRIFIÉ (réparé le cas échéant).
                if buffer_mode:
                    _replay_text = "".join(assistant_response)
                    for _i in range(0, len(_replay_text), 60):
                        yield f"data: {json.dumps({'message': {'content': _replay_text[_i:_i + 60]}})}\n\n"

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

                # Vérification post-génération ADVISORY (P2, 2026-07-20 ; refonte B7) : en
                # mode non bloquant le texte a déjà streamé — ce contrôle détecte et trace.
                # En mode bloquant (buffer), la vérification a DÉJÀ eu lieu avant émission
                # (gate B7c ci-dessus) : on ne la rejoue pas. Modèle DISTINCT de la
                # génération (effective_verify_model) ; VERIFY_ENABLED=false = zéro appel.
                if (
                    verification_result is None
                    and verify_active
                    and assistant_response
                    and not settings.CAG_IMAGE_ONLY  # cf. gate B7c : rien à confronter
                ):
                    try:
                        from app.services.response_verification_service import verify_response

                        # Pages réellement citées par le modèle (<sources>) : le juge doit
                        # voir EN PRIORITÉ ce sur quoi la réponse s'appuie, pas la tête du
                        # contexte. Le contexte complet reste passé pour le contrôle
                        # programmatique, qui le scanne sans troncature.
                        _cited_pages: dict = {}
                        for _src in sources_data or []:
                            _sid = _src.get("document_id")
                            _used = [p for p in (_src.get("used_pages") or []) if isinstance(p, int)]
                            if _sid is not None and _used:
                                _cited_pages.setdefault(int(_sid), []).extend(_used)

                        verification_result = await verify_response(
                            question=retrieval_query_text,
                            response_text=complete_response,
                            context_text=space_context_draft["content"],
                            model=settings.effective_verify_model,
                            document_blocks=space_context_draft.get("cag_document_blocks"),
                            cag_documents=space_context_draft.get("cag_documents"),
                            cited_pages=_cited_pages,
                        )
                        if verification_result is not None:
                            verification_result["action"] = (
                                "passed" if verification_result.get("ok") else "detected_only"
                            )
                    except Exception as verif_err:
                        logger.warning("Vérification post-génération ignorée: %s", verif_err)

                # Trace de génération (« cheminement ») : route, signaux, retrieval, KAG,
                # reasoning et vérification, persistée dans metadata_json et renvoyée au client
                # pour le bouton d'inspection. La clé verification reste aussi au niveau racine
                # de metadata_json (compat lecture existante).
                generation_trace = _build_generation_trace(
                    route="rag",
                    lw_result=lw_result,
                    retrieval_status=retrieval_status,
                    retrieval_reason=retrieval_reason,
                    dynamic_k=dynamic_k,
                    rerank_status=rerank_status,
                    nb_passages=len(doc_passages),
                    anchor_document_ids=anchor_document_ids,
                    requested_codes=requested_codes,
                    pinned_codes=pinned_codes,
                    reasoning_parts=reasoning_parts,
                    passages_summary=_summarize_passages_for_trace(sources_data),
                    verification=verification_result,
                    retry_info=retry_trace,
                    anchor_boost=retrieval.get("anchor_boost"),
                    anchor_intent_changed=anchor_intent_changed,
                    cag_documents=space_context_draft.get("cag_documents"),
                    loop_info=loop_trace,
                )

                # Persister et envoyer les sources avant `done` : le client peut annuler la lecture
                # dès `done`, ce qui coupait le générateur avant commit / événements suivants.
                assistant_message_id = None
                if request.conversation_id and assistant_response:
                    sources_json = json.dumps(sources_data) if sources_data else None
                    reply_metadata: dict = {"trace": generation_trace}
                    if verification_result:
                        reply_metadata["verification"] = verification_result
                    assistant_message_id = _persist_reply_with_retry(
                        request.conversation_id,
                        complete_response,
                        forced_model,
                        forced_provider,
                        sources_json,
                        metadata_json=reply_metadata,
                    )
                    logger.info(
                        "Réponse assistant sauvegardée (space chat), conversation %s avec %s sources",
                        request.conversation_id,
                        len(sources_data) if sources_data else 0,
                    )

                if sources_data:
                    yield f"data: {json.dumps({'sources': sources_data})}\n\n"
                # Invitation Arbre SAV : émise APRÈS la réponse, jamais à sa place.
                if sav_suggestion:
                    yield f"data: {json.dumps({'sav_suggestion': sav_suggestion})}\n\n"
                yield f"data: {json.dumps({'done': True, 'message_id': assistant_message_id, 'trace': generation_trace})}\n\n"

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


# ---------------------------------------------------------------------------
# Arbre SAV — entrées explicites (bouton / picker / chip / deep-link)
# ---------------------------------------------------------------------------


class SavStartRequest(BaseModel):
    conversation_id: int
    tree_slug: str = ""
    tree_id: Optional[int] = None
    entry_node_key: Optional[str] = None


@router.get("/spaces/{space_id}/sav/entries")
async def get_sav_entries(
    space_id: int,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Contenu du picker « Diagnostic SAV » : arbres publiés de l'espace, par symptôme."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(status_code=404, detail="Espace non trouvé")
    from app.services.guided_entry_index_service import list_sav_entries

    return {"entries": list_sav_entries(session, space_id)}


@router.post("/spaces/{space_id}/sav/start")
async def start_sav_diagnostic(
    space_id: int,
    request: SavStartRequest,
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Démarre un parcours d'arbre SAV publié (100 % déterministe) et streame la
    première étape avec le même contrat SSE que le chat."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(status_code=404, detail="Espace non trouvé")
    conversation = session.get(Conversation, request.conversation_id)
    if (
        not conversation
        or conversation.user_id != current_user.id
        or conversation.space_id != space_id
    ):
        raise HTTPException(status_code=404, detail="Conversation non trouvée")

    from app.services.guided_flow_service import start_guided_session

    gtr = start_guided_session(
        session,
        space_id=space_id,
        user_id=current_user.id,
        conversation_id=request.conversation_id,
        tree_slug=request.tree_slug,
        tree_id=request.tree_id,
        entry_node_key=request.entry_node_key,
    )
    return _guided_streaming_response(
        gtr, request.conversation_id, settings.MODEL_FAST, "mistral"
    )


@router.post("/spaces/{space_id}/sav/photo")
async def upload_sav_photo(
    space_id: int,
    conversation_id: int = Form(...),
    file: UploadFile = File(...),
    current_user: UserRead = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    """Photo client pendant un diagnostic : stockée sur la session guidée active,
    jointe au récapitulatif d'escalade."""
    space = get_space_by_id(session, space_id, current_user.id)
    if not space:
        raise HTTPException(status_code=404, detail="Espace non trouvé")
    conversation = session.get(Conversation, conversation_id)
    if not conversation or conversation.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation non trouvée")

    from app.services.guided_flow_service import load_active_guided_state
    from app.models.guided_session import GuidedSession

    state = load_active_guided_state(conversation.query_context)
    if not state:
        raise HTTPException(status_code=409, detail="Aucun diagnostic actif sur cette conversation")
    gsession = session.get(GuidedSession, int(state["active_session_id"]))
    if gsession is None:
        raise HTTPException(status_code=404, detail="Session de diagnostic introuvable")

    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Photo trop volumineuse (max 15 Mo)")

    from app.services.document_service_new import save_uploaded_file

    path = save_uploaded_file(content, file.filename or "photo.jpg", upload_dir="media/sav_photos")
    if not path:
        raise HTTPException(status_code=500, detail="Échec de sauvegarde de la photo")

    files = list(gsession.uploaded_files or [])
    files.append(
        {
            "path": path,
            "node_key": gsession.current_node_key,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
    )
    gsession.uploaded_files = files
    gsession.updated_at = datetime.utcnow()
    session.add(gsession)
    session.commit()
    return {"ok": True, "count": len(files)}

