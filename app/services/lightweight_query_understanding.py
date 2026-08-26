"""
Query understanding : UN appel LLM fusionné, puis construction déterministe des requêtes.

Chemin unique (2026-08-26) :
    fused_understand   → 1 appel LLM : route + signaux + question autonome + topic_shift
    build_queries_fast → 0 appel LLM : requêtes colpali / lexical

Trois implémentations coexistaient auparavant (graphe LangGraph dédié, graphe legacy
multi-nœuds, graphe fusionné) dont une seule s'exécutait. Les deux autres ont été
supprimées, ainsi que la chaîne multi-groupe et le contrôle de vagueness — inatteignables
en configuration par défaut. Voir docs/plan_query_understanding_p0_p1_2026-08-26.md.
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
from app.services.conversation_state_service import (
    build_conversation_state,
    format_state_facts,
)
from app.services.mistral_service import chat
from app.services.query_schemas import RetrievalQueries
from app.services.query_signals_schemas import (
    LightweightQuerySignals,
    build_extract_signals_prompt,
    parse_and_validate_signals,
    to_lightweight_signals,
)

logger = logging.getLogger(__name__)


async def _understanding_chat(*args, **kwargs) -> Dict[str, Any]:
    """Appel LLM de compréhension avec BUDGET TEMPS applicatif (P0.3).

    ``chat`` a déjà un timeout HTTP (120 s) mais c'est trop long avant un retrieval :
    on borne l'appel à QUERY_UNDERSTANDING_TIMEOUT_S et on laisse l'appelant retomber
    sur ses valeurs de repli en cas de dépassement (jamais de blocage indéfini)."""
    timeout = settings.QUERY_UNDERSTANDING_TIMEOUT_S
    if timeout and timeout > 0:
        return await asyncio.wait_for(chat(*args, **kwargs), timeout=timeout)
    return await chat(*args, **kwargs)


class LightweightQueryResult(BaseModel):
    route: str = "rag"
    ready_for_retrieval: bool = False
    retrieval_queries: Optional[RetrievalQueries] = None
    signals: LightweightQuerySignals = Field(default_factory=LightweightQuerySignals)
    query_context: Dict[str, Any] = Field(default_factory=dict)
    # True quand le dernier message change de sujet vs l'historique : le condense ne
    # réintègre alors pas l'ancien sujet et l'appelant n'hérite pas des signaux passés.
    topic_shift: bool = False


class LightweightState(TypedDict, total=False):
    user_message: str
    history: List[Dict[str, str]]
    session: Session
    persisted_context: Optional[Dict[str, Any]]
    route: str
    signals: Dict[str, Any]
    ready_for_retrieval: bool
    retrieval_queries: Optional[Dict[str, Any]]
    standalone_question: str
    topic_shift: bool
    # Étiquette de sujet courant proposée par la compréhension fusionnée (peut être vide).
    llm_current_topic: str


def _search_text(state: LightweightState) -> str:
    """Texte à utiliser pour la RECHERCHE documentaire.

    Privilégie la question autonome reformulée (history-aware) si disponible, sinon
    retombe sur le message brut. Ne change ni le texte affiché ni celui envoyé au LLM
    de génération de réponse.
    """
    standalone = (state.get("standalone_question") or "").strip()
    return standalone or (state.get("user_message") or "").strip()


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

D. "current_topic" : étiquette COURTE (≤ 12 mots) du sujet courant de la conversation
   APRÈS ce message : produit/référence principal + angle abordé
   (ex : "dormant 6101 — dimensions", "réglage compression ouvrant PVC").
   - Si topic_shift = false : fais ÉVOLUER le sujet persistant fourni (même produit,
     nouvelle facette) au lieu de le recopier tel quel.
   - Si topic_shift = true : repars du seul dernier message.

Un bloc « État de la conversation » peut être fourni AVANT l'historique : c'est le fil
persistant des tours précédents (sujet courant, références en focus). Utilise-le en
priorité pour résoudre les références implicites du dernier message ("ses", "celui-ci",
"et le…") et pour juger topic_shift — il est plus fiable qu'un historique tronqué.

Le JSON final = TOUS les champs de signaux ci-dessus + "route" + "topic_shift" +
"standalone_question" + "current_topic". Aucun autre champ.
"""


def _compact_message_content(content: str) -> str:
    """Compacte un message pour les extraits d'historique des prompts de compréhension.

    Les réponses structurées (fiches techniques, tableaux Markdown) tronquées
    brutalement à N caractères deviennent du bruit : on retire les lignes de tableau
    et la décoration Markdown, puis on aplatit — le budget de caractères garde ainsi
    l'essentiel sémantique (sujet, références) au lieu de pipes et de tirets.
    """
    kept: List[str] = []
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Lignes de tableau Markdown (rangées et séparateurs) : purement décoratives
        # dans un extrait tronqué.
        if stripped.startswith("|") or set(stripped) <= {"-", "|", ":", " "}:
            continue
        kept.append(stripped.lstrip("#*-• ").strip())
    return " ".join(k for k in kept if k)


def _history_snippet(history: List[Dict[str, str]], *, turns: int, cap: int) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-turns:]:
        role = msg.get("role", "user")
        content = _compact_message_content(str(msg.get("content", "")))[:cap]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _node_fused_understand(state: LightweightState) -> Dict[str, Any]:
    """Route + signaux + question autonome + topic_shift en UN appel LLM.

    Le petit modèle de compréhension traite tout en une passe JSON : c'est le seul
    appel LLM avant le retrieval.
    """
    session = state["session"]
    full_request = (state.get("user_message") or "").strip()
    history = state.get("history") or []
    history_snippet = _history_snippet(history, turns=6, cap=400)

    # Fil persistant de la conversation (sujet courant + entités en focus) : plus fiable
    # qu'un historique tronqué pour résoudre les suivis elliptiques, et seul survivant
    # quand le tour précédent est passé par un fast-path (fiche technique).
    state_facts = format_state_facts(state.get("persisted_context"))
    state_block = (
        f"État de la conversation (fil persistant) :\n{state_facts}\n\n" if state_facts else ""
    )

    # La décision guidée par LLM est SUPPRIMÉE (refonte Arbre SAV 2026-07-30) : le RAG
    # répond toujours ; l'entrée en diagnostic est déterministe (bouton / chip via
    # guided_entry_index_service). detected_symptom reste porté par les signaux.
    system_prompt = build_extract_signals_prompt(session) + FUSED_UNDERSTANDING_EXTRA_PROMPT
    user_prompt = (
        f"{state_block}"
        f"Historique récent :\n{history_snippet or '(vide)'}\n\n"
        f"Dernier message utilisateur : '{full_request}'\n\n"
        "Extrais les signaux de CE dernier message, puis produis route, topic_shift, "
        "standalone_question et current_topic."
    )

    # Valeurs de repli (si l'appel LLM échoue, on ne bloque jamais le pipeline).
    route = "rag"
    topic_shift = not history
    standalone = full_request
    llm_current_topic = ""
    signals_dict = LightweightQuerySignals().model_dump()

    # Appel + validation JSON avec 1 retry (P0.3) : un JSON tronqué/invalide ou sans clé
    # "route" est réessayé une fois avant de retomber sur les valeurs de repli — au lieu
    # d'un fallback silencieux non diagnostiqué.
    raw: Optional[Dict[str, Any]] = None
    for attempt in range(2):
        try:
            response = await _understanding_chat(
                "",
                model=settings.MODEL_QUERY_UNDERSTANDING,
                context=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response["choices"][0]["message"].get("content", "{}")
            candidate_raw = json.loads(content)
            if not isinstance(candidate_raw, dict) or "route" not in candidate_raw:
                raise ValueError("JSON fused sans clé 'route'")
            raw = candidate_raw
            break
        except asyncio.TimeoutError:
            logger.warning("[lightweight_qu] fused_understand timeout (tentative %d)", attempt + 1)
            break  # ne pas re-tenter un appel déjà trop lent
        except Exception as exc:
            logger.error("[lightweight_qu] fused_understand invalide (tentative %d): %s", attempt + 1, exc)

    if raw is not None:
        route = str(raw.get("route") or "rag").strip().lower()
        if route not in ("direct", "rag"):
            route = "rag"
        # Pas d'historique → toujours un nouveau sujet, quel que soit le LLM.
        topic_shift = bool(raw.get("topic_shift")) if history else True
        candidate = str(raw.get("standalone_question") or "").strip()
        if candidate:
            standalone = candidate
        llm_current_topic = str(raw.get("current_topic") or "").strip()

        # parse_and_validate_signals ignore les clés méta (route, topic_shift, …).
        try:
            extraction = parse_and_validate_signals(raw, session=session)
            signals_dict = to_lightweight_signals(extraction).model_dump()
        except Exception as exc:
            logger.error("[lightweight_qu] fused signaux invalides: %s", exc)

    logger.info(
        "[lightweight_qu] fused_understand — route=%s topic_shift=%s standalone=%r",
        route,
        topic_shift,
        standalone[:100],
    )

    if route == "direct":
        return {
            "route": route,
            "signals": signals_dict,
            "topic_shift": topic_shift,
        }

    return {
        "route": route,
        "signals": signals_dict,
        "topic_shift": topic_shift,
        "standalone_question": standalone,
        "llm_current_topic": llm_current_topic,
        "ready_for_retrieval": True,
    }


def _node_build_queries_fast(state: LightweightState) -> Dict[str, Any]:
    """Construit les requêtes retriever SANS appel LLM.

    La question autonome reformulée par le fused sert de requête visuelle (ColPali) ;
    le canal lexical (BM25) y ajoute les entités et références extraites, qui pèsent
    dans le tsvector. Zéro appel LLM → un seul appel de compréhension au total.
    """
    signals = state.get("signals") or {}
    search_text = _search_text(state)
    entity_texts = [str(t) for t in (signals.get("entity_texts") or []) if str(t).strip()]
    refs = [str(r) for r in (signals.get("detected_references") or []) if str(r).strip()]
    lexical_extra = " ".join(entity_texts[:8] + refs[:4]).strip()
    lexical = f"{search_text} {lexical_extra}".strip() if lexical_extra else search_text

    rq = RetrievalQueries(
        colpali=search_text,
        lexical=lexical,
        reasoning="deterministic (fused, 0 LLM)",
        slots_used=signals,
    )
    logger.info("[lightweight_qu] build_queries_fast — requêtes déterministes (0 appel LLM)")
    return {"retrieval_queries": rq.model_dump()}


def _after_fused(state: LightweightState) -> str:
    if state.get("route") == "direct":
        return "end_direct"
    return "build_queries_fast"


_GRAPH = None


def _build_graph():
    graph = StateGraph(LightweightState)
    graph.add_node("fused_understand", _node_fused_understand)
    graph.add_node("build_queries_fast", _node_build_queries_fast)

    graph.set_entry_point("fused_understand")
    graph.add_conditional_edges(
        "fused_understand",
        _after_fused,
        {"end_direct": END, "build_queries_fast": "build_queries_fast"},
    )
    graph.add_edge("build_queries_fast", END)
    return graph.compile()


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

    topic_shift = bool(final_state.get("topic_shift"))
    standalone_q = final_state.get("standalone_question") or ""

    # Fil conducteur de la conversation (sujet courant + entités en focus) : mis à jour
    # chaque tour, réinitialisé sur changement de sujet. Réinjecté dans la compréhension
    # du tour suivant ET dans le contexte de génération (bloc « fil de la conversation »).
    conversation_state = build_conversation_state(
        persisted_context,
        topic_shift=topic_shift,
        llm_topic=final_state.get("llm_current_topic"),
        fallback_topic=standalone_q or user_message,
        new_entities=signals.entity_texts,
    )

    query_context = {
        "signals": signals.model_dump(),
        "original_user_message": user_message,
        "standalone_question": standalone_q,
        "topic_shift": topic_shift,
        **conversation_state,
    }

    retrieval_queries = final_state.get("retrieval_queries")
    if final_state.get("ready_for_retrieval") and retrieval_queries:
        return LightweightQueryResult(
            route="rag",
            ready_for_retrieval=True,
            retrieval_queries=RetrievalQueries(**retrieval_queries),
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
