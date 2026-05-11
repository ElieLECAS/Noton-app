from typing import Dict, List, Optional
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
from app.tracing import trace_run, trace_pipeline
from datetime import datetime
import json
import logging
import os
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
# Plan A : 8 → 4. Avec la correction A1 (feuille au lieu du parent entier),
# 4 passages bien ciblés suffisent et le LLM hallucine beaucoup moins.
RAG_TOP_K = _int_env("RAG_TOP_K", 4)


def _space_chat_max_tokens() -> int:
    """Priorité : OVERRIDE → variable compose → settings → défaut plan qualité."""
    o = os.getenv("SPACE_CHAT_MAX_TOKENS_OVERRIDE")
    if o and str(o).strip():
        try:
            return max(1, int(o))
        except ValueError:
            pass
    o = os.getenv("SPACE_CHAT_MAX_TOKENS")
    if o and str(o).strip():
        try:
            return max(1, int(o))
        except ValueError:
            pass
    if settings.SPACE_CHAT_MAX_TOKENS is not None:
        return max(1, int(settings.SPACE_CHAT_MAX_TOKENS))
    return 1500


def _space_chat_temperature() -> float:
    o = os.getenv("SPACE_CHAT_TEMPERATURE_OVERRIDE")
    if o and str(o).strip():
        try:
            return float(o)
        except ValueError:
            pass
    o = os.getenv("SPACE_CHAT_TEMPERATURE")
    if o and str(o).strip():
        try:
            return float(o)
        except ValueError:
            pass
    return float(settings.SPACE_CHAT_TEMPERATURE)


# Paramétrage chat "espaces" (docker-compose / .env / settings)
SPACE_CHAT_MAX_TOKENS = _space_chat_max_tokens()
SPACE_CHAT_TEMPERATURE = _space_chat_temperature()
SPACE_CHAT_TOP_P = None
TRACE_VERBOSE_TEXT = os.getenv("TRACE_VERBOSE_TEXT", "false").lower() == "true"
SPACE_AGENTIC_STEPBACK_ENABLED = os.getenv("SPACE_AGENTIC_STEPBACK_ENABLED", "true").lower() == "true"
SPACE_AGENTIC_MIN_SCORE = float(os.getenv("SPACE_AGENTIC_MIN_SCORE", "0.12"))
SPACE_AGENTIC_MIN_CITED_PASSAGES = int(os.getenv("SPACE_AGENTIC_MIN_CITED_PASSAGES", "2"))
SPACE_AGENTIC_MAX_TURNS = max(1, int(os.getenv("SPACE_AGENTIC_MAX_TURNS", "2")))
SPACE_AGENTIC_DECOMPOSE_MAX_SUBQS = max(1, int(os.getenv("SPACE_AGENTIC_DECOMPOSE_MAX_SUBQS", "3")))

# --- Plan A : system prompt « grounded strict » -----------------------------
# - Réponse UNIQUEMENT à partir des PASSAGES (pas de connaissance externe).
# - Citations [n] obligatoires pour chaque affirmation factuelle.
# - Si l'info n'est pas dans les passages → réponse "non disponible dans la
#   documentation fournie".
# - Identité métier sortie du prompt (configurable via SPACE_ASSISTANT_NAME et
#   SPACE_ASSISTANT_ORG). Évite le biais "PROFERM" hardcodé qui pousse le LLM
#   à compléter par sa base interne.
SPACE_ASSISTANT_NAME = os.getenv("SPACE_ASSISTANT_NAME", "l'assistant documentaire")
SPACE_ASSISTANT_ORG = os.getenv("SPACE_ASSISTANT_ORG", "").strip()

_ORG_LINE = (
    f"Tu réponds pour le compte de {SPACE_ASSISTANT_ORG}. "
    if SPACE_ASSISTANT_ORG
    else ""
)

SPACE_CHAT_SYSTEM_PROMPT = (
    f"Tu es {SPACE_ASSISTANT_NAME}. {_ORG_LINE}"
    "Tu réponds en français, de façon claire, factuelle et concise.\n\n"
    "RÈGLES STRICTES — à suivre sans exception :\n"
    "1) Tu réponds UNIQUEMENT à partir des PASSAGES fournis ci-dessous. "
    "N'utilise jamais de connaissance externe, ne fais aucune supposition, "
    "n'invente aucune donnée (chiffres, dimensions, références, normes).\n"
    "2) Pour chaque affirmation factuelle, cite le ou les numéros de passage "
    "correspondants entre crochets, ex : « ... Uw = 1,2 W/m².K [1][3] ».\n"
    "3) Si l'information demandée n'est PAS présente dans les passages, "
    "réponds exactement : « L'information n'est pas présente dans la "
    "documentation fournie. » et propose éventuellement une reformulation "
    "ou un point à vérifier. N'essaie pas de répondre quand même.\n"
    "4) Si plusieurs passages se contredisent, signale-le explicitement et "
    "cite chacune des sources.\n"
    "5) Distingue rigoureusement les références techniques proches "
    "(ex. Perform 70 vs Perform 76, version A vs B). En cas de doute, "
    "demande une précision plutôt que d'extrapoler.\n"
    "6) Lorsque l'information est dans un tableau ou une liste du passage, "
    "reprends-la telle quelle (cellules, colonnes, intitulés de lignes). "
    "Pour les marques, gammes et désignations fournisseurs, cite exactement "
    "le libellé des passages sans le généraliser.\n"
    "7) Format : réponse courte (3 à 8 lignes) par défaut, listes ou "
    "tableaux Markdown uniquement quand cela ajoute de la clarté technique. "
    "Ne mentionne jamais le fonctionnement interne de la recherche, du RAG, "
    "des chunks, du reranker, etc."
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

        except Exception as e:
            logger.exception("Erreur dans le générateur stream_chat_message")
            error_msg_to_yield = str(e)
        
        if error_msg_to_yield:
            yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"
    
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


def _build_space_system_prompt_from_settings(space: Optional[Space]) -> str:
    """
    B4/B5: construit le prompt système à partir des settings d'espace stockés en BDD.
    """
    prompt = SPACE_CHAT_SYSTEM_PROMPT
    if space is None:
        return prompt
    cfg = getattr(space, "settings_json", None)
    if not isinstance(cfg, dict):
        return prompt
    persona = str(cfg.get("system_prompt_persona") or "").strip()
    if persona:
        prompt = persona
    enabled_sources = cfg.get("enabled_sources")
    if isinstance(enabled_sources, list) and enabled_sources:
        src = [str(s).strip() for s in enabled_sources if str(s).strip()]
        if src:
            prompt += (
                "\n\nContraintes de source (espace): "
                + ", ".join(src)
                + ". Si une source est hors liste, indique qu'elle n'est pas autorisée dans cet espace."
            )
    return prompt


def build_space_context_from_passages(passages: List[dict], system_prompt: Optional[str] = None) -> dict:
    """
    Construit le contexte système à partir des passages RAG + KAG rerankés.
    Format unifié pour le LLM (comme build_semantic_context_from_passages).
    """
    system_message = {
        "role": "system",
        "content": system_prompt or SPACE_CHAT_SYSTEM_PROMPT,
    }

    if passages:
        system_message["content"] += "\n\nPASSAGES (numérotés [1], [2], … — à citer dans la réponse) :\n\n"
        passages_content = []
        for i, passage_data in enumerate(passages, 1):
            passage = passage_data['passage']
            score = passage_data.get('score', 0.0)
            document_title = passage_data.get('document_title', 'Document sans titre')
            page_no = passage_data.get('page_no')
            section = passage_data.get('section') or ""
            header_bits = [f"[{i}]", f"score={score:.2f}", f"doc={document_title}"]
            if page_no:
                header_bits.append(f"p.{page_no}")
            if section:
                header_bits.append(f"section={section}")
            header = " | ".join(header_bits)
            passage_text = f"{header}\n{passage}\n"
            passages_content.append(passage_text)
        system_message["content"] += "\n---\n".join(passages_content)
        system_message["content"] += (
            f"\n\n({len(passages)} passage(s) disponible(s). "
            "Réponds uniquement à partir de ces passages, en citant les numéros utilisés.)"
        )
    else:
        system_message["content"] += (
            "\n\nAucun passage pertinent n'a été trouvé dans la documentation "
            "pour cette requête. Réponds exactement : « L'information n'est pas "
            "présente dans la documentation fournie. » et propose une reformulation."
        )

    return system_message


def _estimate_evidence_quality(passages: List[dict]) -> dict:
    """
    Heuristique légère de suffisance des preuves (Plan C).
    """
    if not passages:
        return {"grade": "weak", "top_score": 0.0, "strong_count": 0}
    scores = [float(p.get("score") or 0.0) for p in passages]
    top = max(scores) if scores else 0.0
    strong_count = sum(1 for s in scores if s >= SPACE_AGENTIC_MIN_SCORE)
    if top >= (SPACE_AGENTIC_MIN_SCORE + 0.10) and strong_count >= SPACE_AGENTIC_MIN_CITED_PASSAGES:
        grade = "strong"
    elif strong_count >= 1:
        grade = "medium"
    else:
        grade = "weak"
    return {"grade": grade, "top_score": top, "strong_count": strong_count}


def _build_stepback_query(query: str) -> str:
    """
    Reformulation « step-back » sans LLM (Plan C pragmatique).
    Retire les identifiants ultra-spécifiques et garde l'intention métier.
    """
    if not query or not query.strip():
        return query
    q = query.strip()
    # Supprime ponctuation forte et normalise espaces
    q = re.sub(r"[\(\)\[\]\{\}:;,_]", " ", q)
    # Retire les codes très spécifiques type "76171", "A*4", refs alphanum longues
    q = re.sub(r"\b[A-Za-z]*\d{3,}[A-Za-z0-9\-_/]*\b", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) < 12:
        return query
    return q


def _merge_agentic_passages(primary: List[dict], secondary: List[dict], k: int) -> List[dict]:
    """
    Fusionne deux jeux de passages en conservant les meilleurs et en dédupliquant
    sur (document_id, chunk_id/source_leaf_chunk_id).
    """
    merged = []
    seen = set()

    def _key(p: dict):
        did = p.get("document_id")
        cid = p.get("source_leaf_chunk_id") or p.get("chunk_id")
        return (did, cid, p.get("chunk_index"))

    for p in primary + secondary:
        key = _key(p)
        if key in seen:
            continue
        seen.add(key)
        merged.append(p)

    merged.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return merged[:k]


def _plan_agentic_strategy(query: str) -> str:
    q = (query or "").lower()
    if any(tok in q for tok in [" vs ", " versus ", " comparer ", "différence", "difference"]):
        return "decompose"
    if any(tok in q for tok in [" impact ", " dépend", "depend", "cause", "lien entre"]):
        return "decompose"
    return "step_back"


def _decompose_subqueries(query: str) -> List[str]:
    if not query:
        return []
    parts = re.split(r"\b(?:et|ainsi que|versus|vs|ou)\b", query, flags=re.IGNORECASE)
    sub = [p.strip(" ,;:.") for p in parts if p and len(p.strip()) >= 10]
    if len(sub) < 2:
        return [query]
    return sub[:SPACE_AGENTIC_DECOMPOSE_MAX_SUBQS]


async def _agentic_retrieve_space_passages(
    *,
    session: Session,
    space_id: int,
    user_id: int,
    original_query: str,
    k: int,
) -> tuple[List[dict], Dict]:
    """
    C1/C3: state machine agentique courte:
      Plan -> Retrieve -> Critique -> Refine (max 2 tours) -> stop.
    """
    if not SPACE_AGENTIC_STEPBACK_ENABLED:
        base = await search_space_passages(
            session=session,
            space_id=space_id,
            query_text=original_query,
            user_id=user_id,
            k=k,
        )
        return base[:k], {
            "iteration": 1,
            "query_current": original_query,
            "strategy": "single",
            "evidence_grade": _estimate_evidence_quality(base).get("grade"),
            "stop_reason": "agentic_disabled",
            "history": [],
        }

    state = {
        "iteration": 1,
        "query_current": original_query,
        "strategy": _plan_agentic_strategy(original_query),
        "evidence_grade": "weak",
        "stop_reason": None,
        "history": [],
    }
    aggregate: List[dict] = []
    seen_turn_queries = set()

    while state["iteration"] <= SPACE_AGENTIC_MAX_TURNS:
        q = (state["query_current"] or "").strip()
        if not q:
            state["stop_reason"] = "empty_query"
            break
        if q.lower() in seen_turn_queries:
            state["stop_reason"] = "no_new_evidence"
            break
        seen_turn_queries.add(q.lower())

        # Retrieve
        if state["strategy"] == "decompose":
            sub_passages: List[dict] = []
            for sq in _decompose_subqueries(q):
                p = await search_space_passages(
                    session=session,
                    space_id=space_id,
                    query_text=sq,
                    user_id=user_id,
                    k=k,
                )
                sub_passages = _merge_agentic_passages(sub_passages, p, k)
            turn_passages = sub_passages
        else:
            turn_passages = await search_space_passages(
                session=session,
                space_id=space_id,
                query_text=q,
                user_id=user_id,
                k=k,
            )

        aggregate = _merge_agentic_passages(aggregate, turn_passages, k)
        evidence = _estimate_evidence_quality(aggregate)
        state["evidence_grade"] = evidence["grade"]
        state["history"].append(
            {
                "iteration": state["iteration"],
                "query": q,
                "strategy": state["strategy"],
                "nb_passages": len(turn_passages),
                "evidence": evidence,
            }
        )

        # Critique + stop criteria
        if evidence["grade"] == "strong":
            state["stop_reason"] = "enough_evidence"
            break
        if state["iteration"] >= SPACE_AGENTIC_MAX_TURNS:
            state["stop_reason"] = "max_turns"
            break

        # Refine
        if state["strategy"] == "decompose":
            # Après une décomposition, on fait un step-back court.
            state["strategy"] = "step_back"
            state["query_current"] = _build_stepback_query(q)
        else:
            state["strategy"] = "decompose" if _plan_agentic_strategy(q) == "decompose" else "step_back"
            state["query_current"] = _build_stepback_query(q)
        state["iteration"] += 1

    return aggregate[:k], state


def build_semantic_context_from_passages(passages: List[dict]) -> List[dict]:
    """Construire le contexte enrichi avec les passages pertinents trouvés par recherche sémantique"""
    # Préprompt court pour limiter les input tokens
    system_message = {
        "role": "system",
        "content": (
            "Tu es LIA, l'experte PROFERM. Réponds uniquement à partir des passages ci-dessous. "
            "Priorise toujours les solutions PROFERM si elles sont présentes. "
            "Vigilance : Ne confonds pas les gammes similaires (ex: Perform 70 vs 76). Précise la version si ambigu. "
            "Donnée absente → indique que la doc ne le précise pas. "
            "Règles : bref et poli ; cite les citations [1], [2]. "
            "Identité : 'Nous' = PROFERM. Fournisseurs = partenaires."
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
            "Tu es LIA, experte technique PROFERM. Réponds uniquement à partir des notes ci-dessous. "
            "Vigilance gammes : Distingue bien les versions (ex: Perform 70 vs 76). "
            "Privilégie PROFERM. Fournisseurs (Technal, etc.) = partenaires. "
            "Infos absentes → indique-le. "
            "Markdown (tableaux), cite le titre de la note.\n\n"
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
    with trace_run(
        "rag_retrieval",
        run_type="retriever",
        inputs={"query": request.message, "project_id": project_id, "k": RAG_TOP_K},
        tags=["rag", "project"],
    ) as retrieval_run:
        passages = await search_relevant_passages(
            session=session,
            project_id=project_id,
            query_text=request.message,
            user_id=current_user.id,
            k=RAG_TOP_K,
            passage_size=500,
        )
        retrieval_run.end(outputs={
            "nb_passages": len(passages),
            "passages": [
                {
                    "note_title": p.get("note_title"),
                    "chunk_id": p.get("chunk_id"),
                    "score": round(float(p.get("score", 0)), 4),
                    "page_no": p.get("page_no"),
                    "section": p.get("section"),
                    "passage_preview": (p.get("passage_raw") or p.get("passage", ""))[:300],
                }
                for p in passages
            ],
        })

    # Construire le contexte enrichi avec les passages pertinents
    project_context = build_semantic_context_from_passages(passages)
    
    full_context = project_context
    
    # Ajouter le contexte de conversation existant si fourni (limité aux 10 derniers messages)
    if request.context:
        full_context = full_context + request.context[-10:]
    
    # Ajouter le message utilisateur actuel
    full_context.append({"role": "user", "content": request.message})

    # Trace root pipeline projet
    _pipeline_inputs = {
        "query": request.message,
        "project_id": project_id,
        "user_id": current_user.id,
        "model": settings.MODEL_FAST,
        "nb_passages": len(passages),
    }
    
    # Variable pour accumuler la réponse de l'assistant
    assistant_response = []
    
    async def generate():
        error_msg_to_yield = None
        try:
            with trace_pipeline(
                "project_chat_pipeline",
                inputs=_pipeline_inputs,
                tags=["chat", "project", "rag"],
            ) as pipeline_run:
                if not settings.MISTRAL_API_KEY:
                    raise ValueError("Mistral API key n'est pas configurée")

                # Trace context building
                with trace_run(
                    "context_building",
                    run_type="chain",
                    inputs={
                        "nb_passages": len(passages),
                        "system_prompt_preview": (full_context[0].get("content", "") if full_context else "")[:200],
                    },
                    tags=["context"],
                ) as ctx_run:
                    ctx_run.end(outputs={
                        "nb_messages_context": len(full_context),
                        "context_chars": sum(len(str(m.get("content", ""))) for m in full_context),
                    })

                # Trace LLM generation
                with trace_run(
                    "llm_generation",
                    run_type="llm",
                    inputs={
                        "model": settings.MODEL_FAST,
                        "provider": "mistral",
                        "messages": [
                            (
                                {"role": m.get("role"), "content": str(m.get("content", ""))}
                                if TRACE_VERBOSE_TEXT
                                else {"role": m.get("role"), "content_preview": str(m.get("content", ""))[:300]}
                            )
                            for m in full_context
                        ],
                    },
                    tags=["llm", "mistral", "streaming"],
                ) as llm_run:
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

                    full_response = "".join(assistant_response)
                    llm_run.end(outputs={
                        "response_chars": len(full_response),
                        "response_preview": full_response[:500],
                    })

                pipeline_run.end(outputs={
                    "nb_passages_used": len(passages),
                    "response_chars": len("".join(assistant_response)),
                })

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
            logger.exception("Erreur dans le générateur stream_project_chat_message")
            error_msg_to_yield = str(e)
        
        if error_msg_to_yield:
            yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"
    
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
    with trace_run(
        "rag_kag_retrieval",
        run_type="retriever",
        inputs={"query": request.message, "space_id": space_id, "k": RAG_TOP_K},
        tags=["rag", "kag", "space"],
    ) as retrieval_run:
        passages, agentic_state = await _agentic_retrieve_space_passages(
            session=session,
            space_id=space_id,
            user_id=current_user.id,
            original_query=request.message,
            k=RAG_TOP_K,
        )
        retrieval_run.end(outputs={
            "nb_passages": len(passages),
            "agentic_state": {
                "iteration": agentic_state.get("iteration"),
                "strategy": agentic_state.get("strategy"),
                "evidence_grade": agentic_state.get("evidence_grade"),
                "stop_reason": agentic_state.get("stop_reason"),
                "history": agentic_state.get("history", [])[-3:],
            },
            "passages": [
                {
                    "document_title": p.get("document_title"),
                    "chunk_id": p.get("chunk_id"),
                    "score": round(float(p.get("score", 0)), 4),
                    "page_no": p.get("page_no"),
                    "section": p.get("section"),
                    "passage_preview": (p.get("passage_raw") or p.get("passage", ""))[:300],
                }
                for p in passages
            ],
        })

    # Construire le contexte système à partir des passages rerankés
    system_prompt = _build_space_system_prompt_from_settings(space)
    space_context = build_space_context_from_passages(passages, system_prompt=system_prompt)

    full_context = []
    full_context.append(space_context)
    if request.context:
        full_context.extend(request.context[-10:])
    full_context.append({"role": "user", "content": request.message})

    _pipeline_inputs_space = {
        "query": request.message,
        "space_id": space_id,
        "user_id": current_user.id,
        "model": forced_model,
        "nb_passages": len(passages),
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

                with trace_run(
                    "llm_generation",
                    run_type="llm",
                    inputs={
                        "model": forced_model,
                        "provider": "mistral",
                        "max_tokens": SPACE_CHAT_MAX_TOKENS,
                        "temperature": SPACE_CHAT_TEMPERATURE,
                        "messages": [
                            (
                                {"role": m.get("role"), "content": str(m.get("content", ""))}
                                if TRACE_VERBOSE_TEXT
                                else {"role": m.get("role"), "content_preview": str(m.get("content", ""))[:300]}
                            )
                            for m in full_context
                        ],
                    },
                    tags=["llm", "mistral", "streaming", "space"],
                ) as llm_run:
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

                    full_response = "".join(assistant_response)
                    llm_run.end(outputs={
                        "response_chars": len(full_response),
                        "response_preview": full_response[:500],
                    })

                pipeline_run.end(outputs={
                    "nb_passages_used": len(passages),
                    "response_chars": len("".join(assistant_response)),
                })

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
                    doc_ids = list({p.get("document_id") for p in passages if p.get("document_id")})
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
                                for p in passages
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
                            for p in passages
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

                    sources_data = []
                    for i, p in enumerate(passages):
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
                    yield f"data: {json.dumps({'sources': sources_data})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.exception("Erreur dans le générateur stream_space_chat_message")
            error_msg_to_yield = str(e)
        
        if error_msg_to_yield:
            yield f"data: {json.dumps({'error': error_msg_to_yield})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

