"""
Query understanding léger : extraction automatique de signaux, clarification si vague, requêtes multi-retriever.
Remplace le slot filling obligatoire par une compréhension souple (boosts, pas filtres).
"""
from __future__ import annotations

import asyncio
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
    QueryGroup,
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
    query_strategy: str = "single"
    query_groups: List[QueryGroup] = Field(default_factory=list)
    # True quand le dernier message change de sujet vs l'historique : le condense ne
    # réintègre alors pas l'ancien sujet et l'appelant n'hérite pas des signaux passés.
    topic_shift: bool = False


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
    standalone_question: str
    topic_shift: bool
    query_strategy: str
    query_groups: List[Dict[str, Any]]


def _full_request_text(state: LightweightState) -> str:
    enriched = (state.get("enriched_user_message") or "").strip()
    original = (state.get("original_user_message") or state.get("user_message") or "").strip()
    return enriched or original


def _search_text(state: LightweightState) -> str:
    """Texte à utiliser pour la RECHERCHE documentaire.

    Privilégie la question autonome reformulée (history-aware) si disponible,
    sinon retombe sur la demande complète. Ne change pas le texte affiché ni
    celui envoyé au LLM de génération de réponse.
    """
    standalone = (state.get("standalone_question") or "").strip()
    return standalone or _full_request_text(state)


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
            model=settings.MODEL_QUERY_UNDERSTANDING,
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

    # Mode rapide : on saute l'appel LLM de vagueness et on file au retrieval.
    if not settings.QUERY_VAGUENESS_CHECK_ENABLED:
        return {
            "ready_for_retrieval": True,
            "awaiting_vague_clarification": False,
            "enriched_user_message": full_request,
            "clarification": None,
        }

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
            model=settings.MODEL_QUERY_UNDERSTANDING,
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


PLAN_MULTI_QUERY_SYSTEM_PROMPT = """Tu analyses une requête utilisateur pour un assistant documentaire menuiserie (PROFERM) et décides d'une stratégie de recherche multi-requêtes.

3 stratégies possibles :
- "single" : requête unitaire (un sujet, un produit, un concept clair) → 1 seul groupe
- "decomposed" : requête comparative ou multi-sujets orthogonaux (Alu vs PVC, pose + étanchéité, plusieurs gammes distinctes) → 2 ou 3 groupes thématiques
- "reformulated" : requête précise mais formulée de façon ambiguë ou avec plusieurs terminologies équivalentes → 2 ou 3 paraphrases du même sujet

RÈGLES :
- Préfère toujours "single" pour toute requête portant sur un seul sujet ou produit.
- "decomposed" uniquement si la question oppose ou juxtapose explicitement des thèmes documentairement indépendants (max 3 groupes).
- "reformulated" si le terme principal a des synonymes métier importants (joint/étanchéité, montage/pose/fixation).
- group.label : 2-4 mots, distinctif (ex : "Gamme Aluminium", "Réglementation DTU", "Pose & Fixation").
- group.focus : orientation de recherche courte, vocabulaire documentaire, 10-20 mots max.

Retourne UNIQUEMENT un JSON :
{
  "strategy": "single" | "decomposed" | "reformulated",
  "reasoning": "explication courte",
  "groups": [
    { "label": "...", "focus": "..." }
  ]
}
"""


CONDENSE_QUESTION_SYSTEM_PROMPT = """Tu reformules le DERNIER message d'un utilisateur en une question AUTONOME, à partir de l'historique de conversation, pour un assistant documentaire menuiserie (PROFERM).

But : la reformulation servira UNIQUEMENT à une recherche documentaire. Elle doit être compréhensible sans l'historique.

RÈGLES STRICTES :
- Résous les références implicites ("et le X ?", "et celui-ci", "sa pose", "cette gamme") en réintégrant le contexte du sujet précédent (gamme, produit, système).
- Le SUJET PRINCIPAL est ce que demande le dernier message. S'il introduit une nouvelle référence ou un nouveau produit (ex : un nouveau code produit), CE nouvel élément doit être le cœur de la question — ne le noie pas sous l'ancien sujet.
- Conserve tel quel tout code/référence produit du dernier message (ne corrige pas, ne modifie pas la casse des références techniques).
- Si le dernier message est déjà autonome, renvoie-le quasiment inchangé.
- Reste concis (une seule question, ≤ 30 mots). Pas de préambule.

Retourne UNIQUEMENT un JSON :
{ "standalone_question": "..." }
"""


async def _node_condense_question(state: LightweightState) -> Dict[str, Any]:
    full_request = _full_request_text(state)
    history = state.get("history") or []

    # Pas d'historique OU condensation désactivée → la demande est déjà autonome.
    if not settings.QUERY_CONDENSE_ENABLED or not history:
        return {"standalone_question": full_request}

    lines = []
    for msg in history[-6:]:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))[:400]
        lines.append(f"{role}: {content}")
    history_snippet = "\n".join(lines)

    prompt = (
        f"Historique récent :\n{history_snippet or '(vide)'}\n\n"
        f"Dernier message utilisateur :\n{full_request}\n\n"
        "Reformule ce dernier message en une question autonome pour la recherche documentaire."
    )

    standalone = full_request
    try:
        response = await chat(
            "",
            model=settings.MODEL_QUERY_UNDERSTANDING,
            context=[
                {"role": "system", "content": CONDENSE_QUESTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)
        candidate = str(data.get("standalone_question") or "").strip()
        if candidate:
            standalone = candidate
    except Exception as exc:
        logger.error("[lightweight_qu] condense_question failed: %s", exc)
        standalone = full_request

    logger.info(
        "[lightweight_qu] condense_question — %r → %r",
        full_request[:80],
        standalone[:120],
    )
    return {"standalone_question": standalone}


FUSED_UNDERSTANDING_EXTRA_PROMPT = """

======================================================================
EN PLUS des signaux ci-dessus, produis AUSSI, dans le MÊME objet JSON, ces champs de
pilotage de la recherche documentaire :

A. "route" : "direct" ou "rag".
   - "direct" : salutation, remerciement, question sur ton identité/rôle/capacités, ou
     bavardage sans lien avec un produit ou un document.
   - "rag" : question technique, produit, gamme, norme, pose, réglage, réparation,
     fournisseur — ou EN CAS DE DOUTE.

B. "topic_shift" : true / false.
   - true si le DERNIER message change de sujet par rapport à l'historique : nouveau
     produit / gamme / thème sans lien, OU demande explicite d'oublier/changer de sujet
     ("oublie", "autre chose", "passons à", "sinon parle-moi de").
   - true aussi s'il n'y a PAS d'historique.
   - false s'il poursuit clairement le même sujet ("et le X ?", "sa pose", "et en PVC ?"
     quand cela s'applique au même produit).

C. "standalone_question" : reformulation AUTONOME du dernier message pour la recherche.
   - Si topic_shift = false : résous les références implicites ("et le X ?", "celui-ci",
     "sa pose") en réintégrant le sujet de l'historique.
   - Si topic_shift = true : NE réintègre PAS l'historique ; rends simplement le dernier
     message grammaticalement autonome, sans y coller l'ancien sujet.
   - Conserve tel quel tout code / référence du dernier message. Une seule question, <= 30 mots.

D. "too_vague" : true si la demande reste trop vague pour une recherche pertinente MÊME en
   tenant compte de l'historique. Si true, remplis "clarification_question" (question courte,
   en français). Sinon too_vague = false et "clarification_question" = null.

Le JSON final = TOUS les champs de signaux ci-dessus + "route" + "topic_shift" +
"standalone_question" + "too_vague" + "clarification_question". Aucun autre champ.
"""


def _history_snippet(history: List[Dict[str, str]], *, turns: int, cap: int) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-turns:]:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))[:cap]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _node_fused_understand(state: LightweightState) -> Dict[str, Any]:
    """Route + signaux + condense + vagueness + topic_shift en UN appel LLM.

    Remplace les nœuds route_decision / extract_signals / assess_vagueness /
    condense_question quand QUERY_FUSED_UNDERSTANDING_ENABLED est actif. Le petit modèle
    de compréhension traite tout en une passe JSON, ce qui supprime 3-4 allers-retours
    séquentiels avant le retrieval.
    """
    session = state["session"]
    full_request = _full_request_text(state)
    history = state.get("history") or []
    history_snippet = _history_snippet(history, turns=6, cap=400)

    system_prompt = build_extract_signals_prompt(session) + FUSED_UNDERSTANDING_EXTRA_PROMPT
    user_prompt = (
        f"Historique récent :\n{history_snippet or '(vide)'}\n\n"
        f"Dernier message utilisateur : '{full_request}'\n\n"
        "Extrais les signaux de CE dernier message, puis produis route, topic_shift, "
        "standalone_question, too_vague et clarification_question."
    )

    # Valeurs de repli (si l'appel LLM échoue, on ne bloque jamais le pipeline).
    route = "rag"
    topic_shift = not history
    standalone = full_request
    too_vague = False
    clarification_q = ""
    signals_dict = LightweightQuerySignals().model_dump()

    try:
        response = await chat(
            "",
            model=settings.MODEL_QUERY_UNDERSTANDING,
            context=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        raw = json.loads(content)

        route = str(raw.get("route") or "rag").strip().lower()
        if route not in ("direct", "rag"):
            route = "rag"
        # Pas d'historique → toujours un nouveau sujet, quel que soit le LLM.
        topic_shift = bool(raw.get("topic_shift")) if history else True
        candidate = str(raw.get("standalone_question") or "").strip()
        if candidate:
            standalone = candidate
        too_vague = bool(raw.get("too_vague"))
        clarification_q = str(raw.get("clarification_question") or "").strip()

        # parse_and_validate_signals ignore les clés méta (route, topic_shift, …).
        extraction = parse_and_validate_signals(raw, session=session)
        signals_dict = to_lightweight_signals(extraction).model_dump()
    except Exception as exc:
        logger.error("[lightweight_qu] fused_understand failed: %s", exc)

    if not settings.QUERY_VAGUENESS_CHECK_ENABLED:
        too_vague = False

    logger.info(
        "[lightweight_qu] fused_understand — route=%s topic_shift=%s too_vague=%s standalone=%r",
        route,
        topic_shift,
        too_vague,
        standalone[:100],
    )

    if route == "direct":
        return {
            "route": route,
            "route_reasoning": "fused",
            "signals": signals_dict,
            "topic_shift": topic_shift,
        }

    if too_vague and clarification_q:
        return {
            "route": route,
            "route_reasoning": "fused",
            "signals": signals_dict,
            "topic_shift": topic_shift,
            "standalone_question": standalone,
            "ready_for_retrieval": False,
            "awaiting_vague_clarification": True,
            "enriched_user_message": full_request,
            "clarification": {
                "question": clarification_q,
                "slot_prompt": None,
                "phase": "awaiting_vague_clarification",
                "pending_field": "",
            },
        }

    return {
        "route": route,
        "route_reasoning": "fused",
        "signals": signals_dict,
        "topic_shift": topic_shift,
        "standalone_question": standalone,
        "ready_for_retrieval": True,
        "awaiting_vague_clarification": False,
        "enriched_user_message": full_request,
        "clarification": None,
    }


async def _node_plan_multi_query(state: LightweightState) -> Dict[str, Any]:
    user_message = _search_text(state)

    # Mode rapide : pas de planification multi-requêtes → un seul groupe, 0 appel LLM.
    if not settings.QUERY_MULTI_GROUP_ENABLED:
        logger.info("[lightweight_qu] plan_multi_query — désactivé (mode rapide), strategy=single")
        return {
            "query_strategy": "single",
            "query_groups": [{"label": "Recherche principale", "focus": user_message}],
        }

    signals = state.get("signals") or {}

    entity_texts = [
        e if isinstance(e, str) else e.get("text", "")
        for e in (signals.get("entities") or [])
    ]
    prompt = (
        f"Question utilisateur : '{user_message}'\n"
        f"Signaux extraits : intent={signals.get('intent')}, "
        f"material_hint={signals.get('material_hint')}, "
        f"entities={entity_texts[:8]}\n"
        "Décide la stratégie de recherche et liste les groupes."
    )

    try:
        response = await chat(
            "",
            model=settings.MODEL_QUERY_UNDERSTANDING,
            context=[
                {"role": "system", "content": PLAN_MULTI_QUERY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response["choices"][0]["message"].get("content", "{}")
        data = json.loads(content)
        strategy = str(data.get("strategy") or "single").strip()
        groups_raw = data.get("groups") or []
        if not isinstance(groups_raw, list) or not groups_raw:
            strategy = "single"
            groups_raw = [{"label": "Recherche principale", "focus": user_message}]
        groups_raw = groups_raw[:3]
    except Exception as exc:
        logger.error("[lightweight_qu] plan_multi_query failed: %s", exc)
        strategy = "single"
        groups_raw = [{"label": "Recherche principale", "focus": user_message}]

    logger.info(
        "[lightweight_qu] plan_multi_query — strategy=%s groups=%s",
        strategy,
        [g.get("label") for g in groups_raw],
    )
    return {"query_strategy": strategy, "query_groups": groups_raw}


async def _generate_one_group_queries(
    user_message: str,
    signals: Dict[str, Any],
    label: str,
    focus: str,
    history_snippet: str = "",
) -> RetrievalQueries:
    history_block = (
        f"Historique récent (contexte uniquement) :\n{history_snippet}\n"
        if history_snippet
        else ""
    )
    prompt = (
        f"Question utilisateur (autonome) : '{user_message}'\n"
        f"{history_block}"
        f"Groupe : {label}\nFocus : {focus}\n"
        f"Signaux extraits : {json.dumps(signals, ensure_ascii=False)}\n"
        "Génère les 3 requêtes optimisées pour ce groupe spécifique. "
        "Chaque requête doit refléter le focus du groupe, pas la question générale dans son ensemble. "
        "Le sujet de la question autonome prime : n'introduis pas de sujet issu uniquement de l'historique."
    )
    try:
        response = await chat(
            "",
            model=settings.MODEL_QUERY_UNDERSTANDING,
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
        base_lexical = str(data.get("lexical") or focus).strip()
        if lexical_extra and lexical_extra.lower() not in base_lexical.lower():
            base_lexical = f"{base_lexical} {lexical_extra}".strip()
        return RetrievalQueries(
            colpali=str(data.get("colpali") or focus).strip(),
            semantic=str(data.get("semantic") or focus).strip(),
            lexical=base_lexical,
            reasoning=str(data.get("reasoning") or ""),
            slots_used=data.get("slots_used") or signals,
        )
    except Exception as exc:
        logger.error("[lightweight_qu] generate_queries group=%r failed: %s", label, exc)
        return RetrievalQueries(
            colpali=focus,
            semantic=focus,
            lexical=focus,
            reasoning=f"fallback: {exc}",
            slots_used=signals,
        )


async def _node_generate_queries(state: LightweightState) -> Dict[str, Any]:
    signals = state.get("signals") or {}
    user_message = _search_text(state)
    groups_raw = state.get("query_groups") or [{"label": "Recherche principale", "focus": user_message}]

    history = state.get("history") or []
    history_snippet = ""
    if history:
        lines = []
        for msg in history[-4:]:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[:300]
            lines.append(f"{role}: {content}")
        history_snippet = "\n".join(lines)

    tasks = [
        _generate_one_group_queries(
            user_message,
            signals,
            g.get("label", f"Groupe {i + 1}"),
            g.get("focus", user_message),
            history_snippet,
        )
        for i, g in enumerate(groups_raw)
    ]
    results: List[RetrievalQueries] = list(await asyncio.gather(*tasks))

    query_groups_out = [
        {
            "label": groups_raw[i].get("label", f"Groupe {i + 1}"),
            "focus": groups_raw[i].get("focus", user_message),
            "queries": r.model_dump(),
        }
        for i, r in enumerate(results)
    ]

    logger.info(
        "[lightweight_qu] generate_queries — %d groupe(s) générés",
        len(query_groups_out),
    )
    return {
        "query_groups": query_groups_out,
        "retrieval_queries": results[0].model_dump() if results else None,
    }


def _node_merge_context(state: LightweightState) -> Dict[str, Any]:
    """Prépare le message à comprendre pour ce tour.

    IMPORTANT — on n'hérite du message/contexte du tour précédent QUE lorsqu'on reprend
    une clarification (le tour N-1 a posé une question de précision et l'utilisateur y
    répond maintenant). En dehors de ce cas, le message courant EST le sujet : réutiliser
    l'``original_user_message`` persisté ferait « coller » la conversation au tout premier
    message et lui ferait perdre le fil dès qu'on enchaîne ou change de sujet.
    """
    persisted = state.get("persisted_context") or {}
    user_message = (state.get("user_message") or "").strip()
    awaiting_vague = bool(persisted.get("awaiting_vague_clarification"))
    in_vague_continuation = (
        awaiting_vague
        and user_message
        and persisted.get("phase") == "awaiting_vague_clarification"
    )

    if in_vague_continuation:
        original = persisted.get("original_user_message") or user_message
        base = persisted.get("enriched_user_message") or original
        enriched = f"{base}\n\nPrécision utilisateur : {user_message}"
        return {
            "original_user_message": original,
            "enriched_user_message": enriched,
            "awaiting_vague_clarification": True,
        }

    return {
        "original_user_message": user_message,
        "enriched_user_message": "",
        "awaiting_vague_clarification": False,
    }


def _after_route(state: LightweightState) -> str:
    if state.get("route") == "direct":
        return "end_direct"
    return "merge_context"


def _after_assess_vagueness(state: LightweightState) -> str:
    if state.get("ready_for_retrieval"):
        return "condense_question"
    return "end_clarification"


def _after_fused(state: LightweightState) -> str:
    if state.get("route") == "direct":
        return "end_direct"
    if not state.get("ready_for_retrieval"):
        return "end_clarification"
    return "plan_multi_query"


def _build_graph_legacy():
    """Graphe multi-nœuds historique (route → signaux → vagueness → condense → …)."""
    graph = StateGraph(LightweightState)
    graph.add_node("route_decision", _node_route_decision)
    graph.add_node("merge_context", _node_merge_context)
    graph.add_node("extract_signals", _node_extract_signals)
    graph.add_node("assess_vagueness", _node_assess_vagueness)
    graph.add_node("condense_question", _node_condense_question)
    graph.add_node("plan_multi_query", _node_plan_multi_query)
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
        {"condense_question": "condense_question", "end_clarification": END},
    )
    graph.add_edge("condense_question", "plan_multi_query")
    graph.add_edge("plan_multi_query", "generate_queries")
    graph.add_edge("generate_queries", END)
    return graph.compile()


def _build_graph_fused():
    """Graphe fusionné : merge_context → 1 appel LLM (route+signaux+condense+vagueness)."""
    graph = StateGraph(LightweightState)
    graph.add_node("merge_context", _node_merge_context)
    graph.add_node("fused_understand", _node_fused_understand)
    graph.add_node("plan_multi_query", _node_plan_multi_query)
    graph.add_node("generate_queries", _node_generate_queries)

    graph.set_entry_point("merge_context")
    graph.add_edge("merge_context", "fused_understand")
    graph.add_conditional_edges(
        "fused_understand",
        _after_fused,
        {
            "end_direct": END,
            "end_clarification": END,
            "plan_multi_query": "plan_multi_query",
        },
    )
    graph.add_edge("plan_multi_query", "generate_queries")
    graph.add_edge("generate_queries", END)
    return graph.compile()


_GRAPH_LEGACY = None
_GRAPH_FUSED = None


def _get_graph():
    """Sélectionne le graphe selon le flag (lu à l'exécution → testable/basculable)."""
    global _GRAPH_LEGACY, _GRAPH_FUSED
    if settings.QUERY_FUSED_UNDERSTANDING_ENABLED:
        if _GRAPH_FUSED is None:
            _GRAPH_FUSED = _build_graph_fused()
        return _GRAPH_FUSED
    if _GRAPH_LEGACY is None:
        _GRAPH_LEGACY = _build_graph_legacy()
    return _GRAPH_LEGACY


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

    topic_shift = bool(final_state.get("topic_shift"))
    standalone_q = final_state.get("standalone_question") or ""

    # Sujet courant persistant (fil conducteur) : réinitialisé au message courant sur
    # changement de sujet, sinon conservé d'un tour à l'autre. Sert de repère de continuité
    # (télémétrie + base pour une future mémoire de conversation).
    persisted_topic = (persisted_context or {}).get("current_topic") if persisted_context else None
    if topic_shift or not persisted_topic:
        current_topic = standalone_q or final_state.get("original_user_message") or user_message
    else:
        current_topic = persisted_topic

    query_context = {
        "signals": signals.model_dump(),
        "original_user_message": final_state.get("original_user_message") or user_message,
        "enriched_user_message": final_state.get("enriched_user_message") or "",
        "standalone_question": standalone_q,
        "awaiting_vague_clarification": bool(final_state.get("awaiting_vague_clarification")),
        "topic_shift": topic_shift,
        "current_topic": current_topic,
        "phase": (
            "awaiting_vague_clarification"
            if final_state.get("awaiting_vague_clarification")
            else "ready"
        ),
    }

    if final_state.get("ready_for_retrieval") and final_state.get("retrieval_queries"):
        parsed_groups: List[QueryGroup] = []
        for g in (final_state.get("query_groups") or []):
            queries_data = g.get("queries")
            if queries_data:
                try:
                    parsed_groups.append(
                        QueryGroup(
                            label=g.get("label", ""),
                            focus=g.get("focus", ""),
                            queries=RetrievalQueries(**queries_data),
                        )
                    )
                except Exception:
                    pass

        return LightweightQueryResult(
            route="rag",
            ready_for_retrieval=True,
            retrieval_queries=RetrievalQueries(**final_state["retrieval_queries"]),
            signals=signals,
            query_context=query_context,
            query_strategy=final_state.get("query_strategy") or "single",
            query_groups=parsed_groups,
            topic_shift=topic_shift,
        )

    clarification_data = final_state.get("clarification")
    if clarification_data:
        return LightweightQueryResult(
            route="rag",
            ready_for_retrieval=False,
            clarification=ClarificationResult(**clarification_data),
            signals=signals,
            query_context=query_context,
            topic_shift=topic_shift,
        )

    return LightweightQueryResult(
        route="rag",
        ready_for_retrieval=False,
        signals=signals,
        query_context=query_context,
        topic_shift=topic_shift,
    )
