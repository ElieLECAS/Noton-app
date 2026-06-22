"""
Graphe LangGraph de query understanding : slot filling, clarification, requêtes multi-retriever.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph

from app.config import settings
from app.services.mistral_service import chat
from app.services.query_reasoning_service import decide_retrieval_route
from app.services.slot_catalog import (
    OPTIONAL_FIELDS,
    build_slot_prompt,
    detect_optional_skip_from_message,
    empty_slots,
    get_label,
    is_valid_slot_value,
    next_missing_optional,
    next_missing_required,
    REQUIRED_FIELDS,
    parse_content_categories,
    serialize_content_categories,
    suggested_categories_for_intent,
)

logger = logging.getLogger(__name__)


class SlotAction(BaseModel):
    field: str
    value: str = ""
    action: Literal["fill", "skip"] = "fill"


class RetrievalQueries(BaseModel):
    colpali: str
    semantic: str
    lexical: str
    reasoning: str = ""
    slots_used: Dict[str, Any] = Field(default_factory=dict)


class ClarificationResult(BaseModel):
    question: str
    slot_prompt: Optional[Dict[str, Any]] = None
    phase: str
    pending_field: str = ""


class QueryUnderstandingResult(BaseModel):
    route: str = "rag"
    ready_for_retrieval: bool = False
    clarification: Optional[ClarificationResult] = None
    retrieval_queries: Optional[RetrievalQueries] = None
    query_context: Dict[str, Any] = Field(default_factory=dict)
    slots: Dict[str, Optional[str]] = Field(default_factory=dict)


class QueryUnderstandingState(TypedDict, total=False):
    user_message: str
    history: List[Dict[str, str]]
    slot_action: Optional[Dict[str, Any]]
    persisted_context: Optional[Dict[str, Any]]
    route: str
    route_reasoning: str
    slots: Dict[str, Optional[str]]
    skipped_optional: List[str]
    ready_for_retrieval: bool
    clarification: Optional[Dict[str, Any]]
    retrieval_queries: Optional[Dict[str, Any]]
    phase: str
    pending_field: Optional[str]
    reset_context: bool
    original_user_message: str
    enriched_user_message: str
    awaiting_vague_clarification: bool


EXTRACT_SYSTEM_PROMPT = """Tu es un extracteur de slots pour un assistant RAG industriel menuiserie (PROFERM).
Analyse le message utilisateur et l'historique pour remplir les champs connus.

CHAMPS POSSIBLES (ne remplir que si explicitement mentionné ou clairement déductible) :
- intent : specification | installation | regulatory | product_selection | troubleshooting | documentation
- product_family : fenetres | portes | coulissants
- material : pvc | aluminium | hybride (matériau, PAS la gamme Proferm)
- product_range : perform | lumine | hybride (gamme Proferm) | textural
- supplier : kommerling | profine | askey | roto | technal | soprofen | proferm
- content_categories : liste de slugs séparés par virgule parmi :
  mounting | hardware_adjustment | sealing | drilling_constraints | dimensions_tolerances |
  load_capacity | material_profile | glazing | parts_references | product_range |
  regulatory | warranty | certification | commercial | product_comparison | troubleshooting
  (choisir selon le type de contenu recherché ; plusieurs valeurs possibles, ex. "mounting,sealing")

RÈGLES :
1. Ne pas inventer de valeurs absentes du dialogue.
2. "hybride" comme matériau ≠ gamme Hybride Proferm — distinguer selon le contexte.
3. Si l'utilisateur change clairement de sujet (nouvelle question métier sans lien), mets new_question=true.
4. Si l'utilisateur répond à une clarification en cours, new_question=false.
5. Si l'utilisateur répond « je ne sais pas » / « je sais pas » à une question sur gamme, fournisseur ou catégories, ne remplis pas ce slot (laisse null).
6. Pour content_categories, déduis les catégories pertinentes depuis l'intent si possible (ex. installation → mounting,hardware_adjustment).

Retourne UNIQUEMENT un JSON :
{
  "slots": { "intent": null ou valeur, "product_family": null, ... },
  "new_question": false,
  "reasoning": "..."
}
"""

GENERATE_QUERIES_SYSTEM_PROMPT = """Tu génères 3 requêtes de recherche optimisées pour un RAG hybride menuiserie.

Entrées : slots validés + question utilisateur.

SORTIE JSON obligatoire :
{
  "colpali": "...",
  "semantic": "...",
  "lexical": "...",
  "reasoning": "...",
  "slots_used": { ... }
}

RÈGLES PAR CANAL :
- colpali : phrase descriptive 20-45 mots, vocabulaire documentaire (notice, fiche, catalogue, schéma, tableau dimensions). ColPali matche texte ET visuel dans les patches PDF.
- semantic : question naturelle complète avec synonymes pour embedding vectoriel.
- lexical : mots-clés courts (max ~12 tokens), refs exactes, pas de phrase interrogative, optimisé BM25/tsquery.

Injecte toujours les slots remplis. N'invente pas de codes produits absents des slots ou du message.
"""

VAGUENESS_ASSESS_SYSTEM_PROMPT = """Tu évalues si une demande utilisateur est suffisamment précise pour lancer une recherche documentaire RAG (menuiserie PROFERM).

Tu reçois : la demande (éventuellement enrichie de précisions), les slots structurants déjà collectés, et l'historique récent.

Une demande est TROP VAGUE (too_vague=true) si une recherche documentaire risque de renvoyer des passages génériques ou hors-sujet, par exemple :
- Dépannage / « souci », « problème » sans symptôme concret (jeu, casse, ne clippe pas, mauvaise cote…)
- Pose / montage / réglage sans préciser l'action attendue ou le composant visé quand le contexte l'exige
- Question générique (« quelle est la cote ? », « comment faire ? ») sans élément technique identifiable
- Slots remplis (matériau, gamme…) mais la formulation reste trop courte pour orienter la recherche
- Références produit ambiguës ou absentes alors qu'elles changeraient la notice (ex. variante Perform, type de dormant)

Une demande est SUFFISAMMENT PRÉCISE (too_vague=false) si :
- Le besoin métier est clair (dimension précise, étape de pose nommée, référence produit, norme, symptôme décrit)
- L'utilisateur a déjà apporté des précisions qui levent l'ambiguïté principale
- La question cible un document ou une information retrouvable sans autre clarification

Si too_vague=true, rédige clarification_question : 2 à 4 phrases en français, ton professionnel et chaleureux, invitant l'utilisateur à DEVELOPPER sa demande (symptôme, composant, contexte, référence). Pas de listes de boutons. Pose 1 à 2 questions ouvertes ciblées.

Retourne UNIQUEMENT un JSON :
{
  "too_vague": true ou false,
  "reasoning": "courte explication",
  "clarification_question": "texte ou chaîne vide si too_vague=false"
}
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_query_context() -> Dict[str, Any]:
    return {
        "slots": empty_slots(),
        "skipped_optional": [],
        "phase": "collecting_required",
        "pending_field": None,
        "retrieval_queries": None,
        "original_user_message": "",
        "enriched_user_message": "",
        "awaiting_vague_clarification": False,
        "updated_at": _utc_now_iso(),
    }


def _full_request_text(state: QueryUnderstandingState) -> str:
    enriched = (state.get("enriched_user_message") or "").strip()
    original = (state.get("original_user_message") or state.get("user_message") or "").strip()
    return enriched or original


def _merge_slot_dict(
    base: Dict[str, Optional[str]],
    incoming: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    merged = dict(base)
    for key in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        val = incoming.get(key)
        if val is None:
            continue
        if key == "content_categories":
            if isinstance(val, list):
                serialized = serialize_content_categories([str(v) for v in val])
            else:
                serialized = serialize_content_categories(parse_content_categories(str(val)))
            if serialized and is_valid_slot_value(key, serialized):
                merged[key] = serialized
            continue
        if str(val).strip():
            candidate = str(val).strip().lower()
            if is_valid_slot_value(key, candidate):
                merged[key] = candidate
    return merged


async def _node_route_decision(state: QueryUnderstandingState) -> Dict[str, Any]:
    decision = await decide_retrieval_route(state.get("user_message", ""), state.get("history"))
    return {
        "route": decision.decision,
        "route_reasoning": decision.reasoning,
    }


def _node_merge_slots(state: QueryUnderstandingState) -> Dict[str, Any]:
    persisted = state.get("persisted_context") or {}
    ctx = _default_query_context()
    if persisted:
        ctx["slots"] = _merge_slot_dict(empty_slots(), persisted.get("slots") or {})
        ctx["skipped_optional"] = list(persisted.get("skipped_optional") or [])
        ctx["phase"] = persisted.get("phase") or "collecting_required"
        ctx["pending_field"] = persisted.get("pending_field")
        ctx["original_user_message"] = persisted.get("original_user_message") or ""
        ctx["enriched_user_message"] = persisted.get("enriched_user_message") or ""
        ctx["awaiting_vague_clarification"] = bool(persisted.get("awaiting_vague_clarification"))

    slots = dict(ctx["slots"])
    skipped = list(ctx["skipped_optional"])
    reset_context = False
    original_message = ctx.get("original_user_message") or state.get("user_message", "")
    enriched_message = ctx.get("enriched_user_message") or ""
    awaiting_vague = bool(ctx.get("awaiting_vague_clarification"))
    phase = ctx.get("phase") or "collecting_required"

    slot_action = state.get("slot_action")
    user_message = (state.get("user_message") or "").strip()

    if (
        awaiting_vague
        and user_message
        and not slot_action
        and phase == "awaiting_vague_clarification"
    ):
        base = enriched_message or original_message
        enriched_message = f"{base}\n\nPrécision utilisateur : {user_message}"

    if slot_action:
        field = slot_action.get("field", "")
        action = slot_action.get("action", "fill")
        if action == "skip" and field in OPTIONAL_FIELDS:
            if field not in skipped:
                skipped.append(field)
        elif action == "fill" and field:
            value = str(slot_action.get("value", "")).strip().lower()
            if is_valid_slot_value(field, value):
                slots[field] = value
    else:
        if user_message:
            pending = ctx.get("pending_field")
            skip_field = detect_optional_skip_from_message(user_message, pending)
            if skip_field and skip_field not in skipped:
                skipped.append(skip_field)
        if not ctx.get("original_user_message"):
            original_message = state.get("user_message", "")

    return {
        "slots": slots,
        "skipped_optional": skipped,
        "original_user_message": original_message,
        "enriched_user_message": enriched_message,
        "awaiting_vague_clarification": awaiting_vague,
        "reset_context": reset_context,
        "phase": phase,
        "pending_field": ctx.get("pending_field"),
    }


async def _node_extract_slots(state: QueryUnderstandingState) -> Dict[str, Any]:
    slots = dict(state.get("slots") or empty_slots())
    skipped = list(state.get("skipped_optional") or [])
    reset_context = state.get("reset_context", False)
    original_message = state.get("original_user_message") or state.get("user_message", "")
    enriched_message = state.get("enriched_user_message") or ""
    awaiting_vague = bool(state.get("awaiting_vague_clarification"))

    slot_action = state.get("slot_action")
    if slot_action:
        return {
            "slots": slots,
            "skipped_optional": skipped,
            "reset_context": reset_context,
            "original_user_message": original_message,
            "enriched_user_message": enriched_message,
            "awaiting_vague_clarification": awaiting_vague,
        }

    user_message = (state.get("user_message") or "").strip()
    if not user_message:
        return {
            "slots": slots,
            "skipped_optional": skipped,
            "reset_context": reset_context,
            "original_user_message": original_message,
            "enriched_user_message": enriched_message,
            "awaiting_vague_clarification": awaiting_vague,
        }

    history_snippet = ""
    history = state.get("history") or []
    if history:
        lines = []
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[:300]
            lines.append(f"{role}: {content}")
        history_snippet = "\n".join(lines)

    current_slots_json = json.dumps({k: v for k, v in slots.items() if v}, ensure_ascii=False)
    prompt = (
        f"Message utilisateur : '{user_message}'\n"
        f"Historique récent :\n{history_snippet or '(vide)'}\n"
        f"Slots déjà remplis : {current_slots_json}\n"
        "Extrais les nouveaux slots depuis le message."
    )

    try:
        response = await chat(
            "",
            model=settings.MODEL_FAST,
            context=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)
        incoming = data.get("slots") or {}
        slots = _merge_slot_dict(slots, incoming)
        if data.get("new_question") is True:
            reset_context = True
            slots = _merge_slot_dict(empty_slots(), incoming)
            skipped = []
            original_message = user_message
            enriched_message = ""
            awaiting_vague = False
    except Exception as exc:
        logger.error("[query_understanding] extract_slots failed: %s", exc)

    if reset_context:
        enriched_message = ""

    return {
        "slots": slots,
        "skipped_optional": skipped,
        "reset_context": reset_context,
        "original_user_message": original_message,
        "enriched_user_message": enriched_message,
        "awaiting_vague_clarification": False if reset_context else awaiting_vague,
    }


def _node_validate_and_next(state: QueryUnderstandingState) -> Dict[str, Any]:
    slots = state.get("slots") or empty_slots()
    skipped = state.get("skipped_optional") or []

    missing_required = next_missing_required(slots)
    if missing_required:
        return {
            "ready_for_retrieval": False,
            "phase": "collecting_required",
            "pending_field": missing_required,
        }

    missing_optional = next_missing_optional(slots, skipped)
    if missing_optional:
        return {
            "ready_for_retrieval": False,
            "phase": "collecting_optional",
            "pending_field": missing_optional,
        }

    return {
        "ready_for_retrieval": True,
        "phase": "ready",
        "pending_field": None,
    }


def _node_build_clarification(state: QueryUnderstandingState) -> Dict[str, Any]:
    field = state.get("pending_field") or next_missing_required(state.get("slots") or {})
    phase = state.get("phase") or "collecting_required"
    allow_skip = phase == "collecting_optional"
    slot_prompt = build_slot_prompt(field, phase=phase, allow_skip=allow_skip)
    question = slot_prompt["question"]

    return {
        "clarification": {
            "question": question,
            "slot_prompt": slot_prompt,
            "phase": phase,
            "pending_field": field,
        },
        "ready_for_retrieval": False,
    }


async def _node_generate_retrieval_queries(state: QueryUnderstandingState) -> Dict[str, Any]:
    slots = state.get("slots") or empty_slots()
    user_message = _full_request_text(state)
    filled = {k: v for k, v in slots.items() if v}
    labels = {k: get_label(k, v) for k, v in filled.items()}

    prompt = (
        f"Question utilisateur : '{user_message}'\n"
        f"Slots validés : {json.dumps(filled, ensure_ascii=False)}\n"
        f"Labels FR : {json.dumps(labels, ensure_ascii=False)}\n"
        "Génère les 3 requêtes optimisées."
    )

    try:
        response = await chat(
            "",
            model=settings.MODEL_FAST,
            context=[
                {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)
        queries = RetrievalQueries(
            colpali=str(data.get("colpali") or user_message).strip(),
            semantic=str(data.get("semantic") or user_message).strip(),
            lexical=str(data.get("lexical") or user_message).strip(),
            reasoning=str(data.get("reasoning") or ""),
            slots_used=data.get("slots_used") or filled,
        )
    except Exception as exc:
        logger.error("[query_understanding] generate_queries failed: %s", exc)
        fallback = user_message or " ".join(labels.values())
        queries = RetrievalQueries(
            colpali=fallback,
            semantic=fallback,
            lexical=fallback,
            reasoning=f"fallback: {exc}",
            slots_used=filled,
        )

    return {"retrieval_queries": queries.model_dump()}


async def _node_assess_vagueness(state: QueryUnderstandingState) -> Dict[str, Any]:
    """Évalue si la demande est assez précise pour le RAG ; sinon demande textuelle d'approfondissement."""
    full_request = _full_request_text(state)
    slots = state.get("slots") or empty_slots()
    filled = {k: v for k, v in slots.items() if v}
    labels = {k: get_label(k, v) for k, v in filled.items()}

    history_snippet = ""
    history = state.get("history") or []
    if history:
        lines = []
        for msg in history[-8:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[:400]
            lines.append(f"{role}: {content}")
        history_snippet = "\n".join(lines)

    prompt = (
        f"Demande utilisateur (complète) :\n{full_request}\n\n"
        f"Slots structurants : {json.dumps(filled, ensure_ascii=False)}\n"
        f"Labels FR : {json.dumps(labels, ensure_ascii=False)}\n\n"
        f"Historique récent :\n{history_snippet or '(vide)'}\n\n"
        "Évalue si cette demande est trop vague pour une recherche documentaire pertinente."
    )

    try:
        response = await chat(
            "",
            model=settings.MODEL_FAST,
            context=[
                {"role": "system", "content": VAGUENESS_ASSESS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)
        too_vague = bool(data.get("too_vague"))
        reasoning = str(data.get("reasoning") or "")
        question = str(data.get("clarification_question") or "").strip()
    except Exception as exc:
        logger.error("[query_understanding] assess_vagueness failed: %s", exc)
        too_vague = False
        reasoning = f"fallback: {exc}"
        question = ""

    logger.info(
        "[query_understanding] assess_vagueness — too_vague=%s reason=%s request=%r",
        too_vague,
        reasoning[:120],
        full_request[:120],
    )

    if too_vague and question:
        return {
            "ready_for_retrieval": False,
            "phase": "awaiting_vague_clarification",
            "awaiting_vague_clarification": True,
            "enriched_user_message": full_request,
            "clarification": {
                "question": question,
                "slot_prompt": None,
                "phase": "awaiting_vague_clarification",
                "pending_field": "",
            },
        }

    return {
        "ready_for_retrieval": True,
        "phase": "ready",
        "awaiting_vague_clarification": False,
        "enriched_user_message": full_request,
        "clarification": None,
    }


def _after_route(state: QueryUnderstandingState) -> str:
    if state.get("route") == "direct":
        return "end_direct"
    return "merge_slots"


def _after_validate(state: QueryUnderstandingState) -> str:
    if state.get("ready_for_retrieval"):
        return "assess_vagueness"
    return "build_clarification"


def _after_assess_vagueness(state: QueryUnderstandingState) -> str:
    if state.get("ready_for_retrieval"):
        return "generate_queries"
    return "end_vague_clarification"


def _build_graph():
    graph = StateGraph(QueryUnderstandingState)
    graph.add_node("route_decision", _node_route_decision)
    graph.add_node("merge_slots", _node_merge_slots)
    graph.add_node("extract_slots", _node_extract_slots)
    graph.add_node("validate_and_next", _node_validate_and_next)
    graph.add_node("build_clarification", _node_build_clarification)
    graph.add_node("assess_vagueness", _node_assess_vagueness)
    graph.add_node("generate_retrieval_queries", _node_generate_retrieval_queries)

    graph.set_entry_point("route_decision")
    graph.add_conditional_edges(
        "route_decision",
        _after_route,
        {"end_direct": END, "merge_slots": "merge_slots"},
    )
    graph.add_edge("merge_slots", "extract_slots")
    graph.add_edge("extract_slots", "validate_and_next")
    graph.add_conditional_edges(
        "validate_and_next",
        _after_validate,
        {
            "build_clarification": "build_clarification",
            "assess_vagueness": "assess_vagueness",
        },
    )
    graph.add_conditional_edges(
        "assess_vagueness",
        _after_assess_vagueness,
        {
            "generate_queries": "generate_retrieval_queries",
            "end_vague_clarification": END,
        },
    )
    graph.add_edge("build_clarification", END)
    graph.add_edge("generate_retrieval_queries", END)
    return graph.compile()


_GRAPH = None


def get_query_understanding_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def _build_query_context_from_state(state: QueryUnderstandingState) -> Dict[str, Any]:
    return {
        "slots": state.get("slots") or empty_slots(),
        "skipped_optional": state.get("skipped_optional") or [],
        "phase": state.get("phase") or "collecting_required",
        "pending_field": state.get("pending_field"),
        "retrieval_queries": state.get("retrieval_queries"),
        "original_user_message": state.get("original_user_message") or "",
        "enriched_user_message": state.get("enriched_user_message") or "",
        "awaiting_vague_clarification": bool(state.get("awaiting_vague_clarification")),
        "updated_at": _utc_now_iso(),
    }


async def run_query_understanding(
    *,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    persisted_context: Optional[Dict[str, Any]] = None,
    slot_action: Optional[SlotAction] = None,
) -> QueryUnderstandingResult:
    """Exécute le graphe de query understanding et retourne le résultat structuré."""
    initial: QueryUnderstandingState = {
        "user_message": user_message,
        "history": history or [],
        "persisted_context": persisted_context,
        "slot_action": slot_action.model_dump() if slot_action else None,
    }

    graph = get_query_understanding_graph()
    final_state = await graph.ainvoke(initial)

    route = final_state.get("route") or "rag"
    if route == "direct":
        return QueryUnderstandingResult(route="direct", ready_for_retrieval=False)

    query_context = _build_query_context_from_state(final_state)

    if final_state.get("ready_for_retrieval") and final_state.get("retrieval_queries"):
        rq_data = final_state["retrieval_queries"]
        return QueryUnderstandingResult(
            route="rag",
            ready_for_retrieval=True,
            retrieval_queries=RetrievalQueries(**rq_data),
            query_context=query_context,
            slots=query_context["slots"],
        )

    clarification_data = final_state.get("clarification")
    if clarification_data:
        return QueryUnderstandingResult(
            route="rag",
            ready_for_retrieval=False,
            clarification=ClarificationResult(**clarification_data),
            query_context=query_context,
            slots=query_context["slots"],
        )

    return QueryUnderstandingResult(
        route="rag",
        ready_for_retrieval=False,
        query_context=query_context,
        slots=query_context["slots"],
    )
