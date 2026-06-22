"""
Query understanding léger : extraction automatique de signaux, clarification si vague, requêtes multi-retriever.
Remplace le slot filling obligatoire par une compréhension souple (boosts, pas filtres).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sqlmodel import Session

from app.config import settings
from app.services.mistral_service import chat
from app.services.query_reasoning_service import decide_retrieval_route
from app.services.query_signals_schemas import (
    LightweightQuerySignals,
    build_extract_signals_prompt,
    parse_and_validate_signals,
    to_lightweight_signals,
)
from app.services.query_understanding_graph import (
    GENERATE_QUERIES_SYSTEM_PROMPT,
    ClarificationResult,
    RetrievalQueries,
    VAGUENESS_ASSESS_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


class LightweightQueryResult(BaseModel):
    route: str = "rag"
    ready_for_retrieval: bool = False
    clarification: Optional[ClarificationResult] = None
    retrieval_queries: Optional[RetrievalQueries] = None
    signals: LightweightQuerySignals = Field(default_factory=LightweightQuerySignals)
    query_context: Dict[str, Any] = Field(default_factory=dict)


class LightweightState(TypedDict, total=False):
    user_message: str
    history: List[Dict[str, str]]
    session: Session
    persisted_context: Optional[Dict[str, Any]]
    route: str
    route_reasoning: str
    signals: Dict[str, Any]
    ready_for_retrieval: bool
    clarification: Optional[Dict[str, Any]]
    retrieval_queries: Optional[Dict[str, Any]]
    original_user_message: str
    enriched_user_message: str
    awaiting_vague_clarification: bool


def _full_request_text(state: LightweightState) -> str:
    enriched = (state.get("enriched_user_message") or "").strip()
    original = (state.get("original_user_message") or state.get("user_message") or "").strip()
    return enriched or original


async def _node_route_decision(state: LightweightState) -> Dict[str, Any]:
    decision = await decide_retrieval_route(state.get("user_message", ""), state.get("history"))
    return {
        "route": decision.decision,
        "route_reasoning": decision.reasoning,
    }


async def _node_extract_signals(state: LightweightState) -> Dict[str, Any]:
    session = state["session"]
    user_message = (state.get("user_message") or "").strip()
    history = state.get("history") or []

    history_snippet = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[:300]
            lines.append(f"{role}: {content}")
        history_snippet = "\n".join(lines)

    prompt = (
        f"Message utilisateur : '{user_message}'\n"
        f"Historique récent :\n{history_snippet or '(vide)'}\n"
        "Extrais les signaux de la demande."
    )

    signals_dict: Dict[str, Any] = {}
    try:
        response = await chat(
            "",
            model=settings.MODEL_FAST,
            context=[
                {"role": "system", "content": build_extract_signals_prompt(session)},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        raw = json.loads(content)
        extraction = parse_and_validate_signals(raw, session=session)
        signals = to_lightweight_signals(extraction)
        signals_dict = signals.model_dump()
        logger.info(
            "[lightweight_qu] extract_signals — intent=%s entities=%d categories=%s",
            signals.intent,
            len(signals.entity_texts),
            signals.inferred_categories,
        )
    except Exception as exc:
        logger.error("[lightweight_qu] extract_signals failed: %s", exc)
        signals_dict = LightweightQuerySignals().model_dump()

    return {"signals": signals_dict}


async def _node_assess_vagueness(state: LightweightState) -> Dict[str, Any]:
    full_request = _full_request_text(state)
    signals = state.get("signals") or {}

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
        f"Signaux extraits : {json.dumps(signals, ensure_ascii=False)}\n\n"
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
        question = str(data.get("clarification_question") or "").strip()
    except Exception as exc:
        logger.error("[lightweight_qu] assess_vagueness failed: %s", exc)
        too_vague = False
        question = ""

    logger.info(
        "[lightweight_qu] assess_vagueness — too_vague=%s request=%r",
        too_vague,
        full_request[:120],
    )

    if too_vague and question:
        return {
            "ready_for_retrieval": False,
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
        "awaiting_vague_clarification": False,
        "enriched_user_message": full_request,
        "clarification": None,
    }


async def _node_generate_queries(state: LightweightState) -> Dict[str, Any]:
    signals = state.get("signals") or {}
    user_message = _full_request_text(state)

    prompt = (
        f"Question utilisateur : '{user_message}'\n"
        f"Signaux extraits : {json.dumps(signals, ensure_ascii=False)}\n"
        "Génère les 3 requêtes optimisées. Injecte les entités et références détectées."
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
        entity_texts = signals.get("entity_texts") or []
        refs = signals.get("detected_references") or []
        lexical_extra = " ".join(entity_texts[:8] + refs[:4])
        base_lexical = str(data.get("lexical") or user_message).strip()
        if lexical_extra and lexical_extra.lower() not in base_lexical.lower():
            base_lexical = f"{base_lexical} {lexical_extra}".strip()

        queries = RetrievalQueries(
            colpali=str(data.get("colpali") or user_message).strip(),
            semantic=str(data.get("semantic") or user_message).strip(),
            lexical=base_lexical,
            reasoning=str(data.get("reasoning") or ""),
            slots_used=data.get("slots_used") or signals,
        )
    except Exception as exc:
        logger.error("[lightweight_qu] generate_queries failed: %s", exc)
        entity_texts = signals.get("entity_texts") or []
        fallback = user_message
        if entity_texts:
            fallback = f"{user_message} {' '.join(entity_texts[:6])}"
        queries = RetrievalQueries(
            colpali=fallback,
            semantic=fallback,
            lexical=fallback,
            reasoning=f"fallback: {exc}",
            slots_used=signals,
        )

    return {"retrieval_queries": queries.model_dump()}


def _node_merge_context(state: LightweightState) -> Dict[str, Any]:
    persisted = state.get("persisted_context") or {}
    original = persisted.get("original_user_message") or state.get("user_message", "")
    enriched = persisted.get("enriched_user_message") or ""
    awaiting_vague = bool(persisted.get("awaiting_vague_clarification"))

    user_message = (state.get("user_message") or "").strip()
    if awaiting_vague and user_message and persisted.get("phase") == "awaiting_vague_clarification":
        base = enriched or original
        enriched = f"{base}\n\nPrécision utilisateur : {user_message}"

    return {
        "original_user_message": original or user_message,
        "enriched_user_message": enriched,
        "awaiting_vague_clarification": awaiting_vague,
    }


def _after_route(state: LightweightState) -> str:
    if state.get("route") == "direct":
        return "end_direct"
    return "merge_context"


def _after_assess_vagueness(state: LightweightState) -> str:
    if state.get("ready_for_retrieval"):
        return "generate_queries"
    return "end_clarification"


def _build_graph():
    graph = StateGraph(LightweightState)
    graph.add_node("route_decision", _node_route_decision)
    graph.add_node("merge_context", _node_merge_context)
    graph.add_node("extract_signals", _node_extract_signals)
    graph.add_node("assess_vagueness", _node_assess_vagueness)
    graph.add_node("generate_queries", _node_generate_queries)

    graph.set_entry_point("route_decision")
    graph.add_conditional_edges(
        "route_decision",
        _after_route,
        {"end_direct": END, "merge_context": "merge_context"},
    )
    graph.add_edge("merge_context", "extract_signals")
    graph.add_edge("extract_signals", "assess_vagueness")
    graph.add_conditional_edges(
        "assess_vagueness",
        _after_assess_vagueness,
        {"generate_queries": "generate_queries", "end_clarification": END},
    )
    graph.add_edge("generate_queries", END)
    return graph.compile()


_GRAPH = None


def _get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


async def run_lightweight_understanding(
    *,
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    session: Session,
    persisted_context: Optional[Dict[str, Any]] = None,
) -> LightweightQueryResult:
    initial_state: LightweightState = {
        "user_message": user_message,
        "history": history or [],
        "session": session,
        "persisted_context": persisted_context,
    }

    final_state = await _get_graph().ainvoke(initial_state)

    if final_state.get("route") == "direct":
        return LightweightQueryResult(route="direct", ready_for_retrieval=False)

    signals_data = final_state.get("signals") or {}
    signals = LightweightQuerySignals(**signals_data) if signals_data else LightweightQuerySignals()

    query_context = {
        "signals": signals.model_dump(),
        "original_user_message": final_state.get("original_user_message") or user_message,
        "enriched_user_message": final_state.get("enriched_user_message") or "",
        "awaiting_vague_clarification": bool(final_state.get("awaiting_vague_clarification")),
        "phase": (
            "awaiting_vague_clarification"
            if final_state.get("awaiting_vague_clarification")
            else "ready"
        ),
    }

    if final_state.get("ready_for_retrieval") and final_state.get("retrieval_queries"):
        return LightweightQueryResult(
            route="rag",
            ready_for_retrieval=True,
            retrieval_queries=RetrievalQueries(**final_state["retrieval_queries"]),
            signals=signals,
            query_context=query_context,
        )

    clarification_data = final_state.get("clarification")
    if clarification_data:
        return LightweightQueryResult(
            route="rag",
            ready_for_retrieval=False,
            clarification=ClarificationResult(**clarification_data),
            signals=signals,
            query_context=query_context,
        )

    return LightweightQueryResult(
        route="rag",
        ready_for_retrieval=False,
        signals=signals,
        query_context=query_context,
    )
