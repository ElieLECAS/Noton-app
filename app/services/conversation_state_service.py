"""État conversationnel persistant — le « fil » de la discussion.

Problème résolu : la continuité de conversation reposait sur deux mécanismes
partiels — l'ancre DOCUMENTAIRE (current_documents, niveau document : insuffisant
quand un document couvre des dizaines de références) et un ``current_topic``
calculé/persisté mais jamais réinjecté nulle part. Résultat : un suivi elliptique
(« tu as ses dimensions ? ») perdait son référent dès que le tour précédent était
passé par un fast-path (fiche technique) ou que le contexte CAG (~100k tokens)
noyait l'historique.

Ce service centralise un état de conversation COMPACT, entretenu à chaque tour et
stocké dans ``Conversation.query_context`` :
  - ``current_topic``    : étiquette courte du sujet courant (mise à jour chaque tour,
                           réinitialisée sur changement de sujet) ;
  - ``focus_entities``   : références/entités récemment au centre de la discussion
                           (les plus récentes d'abord, plafonnées, remises à zéro sur
                           changement de sujet).

Cet état est réinjecté aux TROIS points qui en ont besoin :
  1. compréhension de requête (résolution des références implicites au condense) ;
  2. retrieval (les entités alimentent déjà les requêtes lexicales via les signaux) ;
  3. génération (bloc « fil de la conversation » dans le message système, pour que le
     modèle résolve « ses / celui-ci / et le… » même avec un très grand contexte).

Aucune référence produit ni cas particulier codé en dur : uniquement de la mécanique
de fusion/rendu d'état.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

# Plafond d'entités gardées en focus : le sujet courant d'une conversation technique
# tient sur une poignée de références ; au-delà, le bloc devient du bruit de prompt.
FOCUS_ENTITIES_MAX = 8


def _dedupe_keep_order(texts: Iterable[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for t in texts:
        text = str(t or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def merge_focus_entities(
    previous: Optional[List[str]],
    new_entities: Optional[List[str]],
    *,
    topic_shift: bool,
    cap: int = FOCUS_ENTITIES_MAX,
) -> List[str]:
    """Fusionne les entités en focus : les nouvelles d'abord (récence), puis les
    anciennes, dédupliquées et plafonnées. Changement de sujet → on repart des
    seules entités du tour courant."""
    fresh = _dedupe_keep_order(new_entities or [])
    if topic_shift:
        return fresh[:cap]
    return _dedupe_keep_order(fresh + list(previous or []))[:cap]


def build_conversation_state(
    persisted: Optional[Dict[str, Any]],
    *,
    topic_shift: bool,
    llm_topic: Optional[str] = None,
    fallback_topic: str = "",
    new_entities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calcule l'état de conversation du tour courant (à fusionner dans query_context).

    - ``llm_topic`` : étiquette de sujet proposée par la compréhension de ce tour
      (prioritaire : elle fait évoluer le sujet au fil des facettes abordées) ;
    - ``fallback_topic`` : repli quand le LLM n'en propose pas (question autonome
      ou message brut) ;
    - sans changement de sujet, le sujet persistant est conservé si rien de mieux.
    """
    persisted = persisted or {}
    previous_topic = str(persisted.get("current_topic") or "").strip()
    llm_topic = str(llm_topic or "").strip()

    if topic_shift or not previous_topic:
        current_topic = llm_topic or str(fallback_topic or "").strip()
    else:
        current_topic = llm_topic or previous_topic

    focus_entities = merge_focus_entities(
        persisted.get("focus_entities"),
        new_entities,
        topic_shift=topic_shift,
    )

    return {
        "current_topic": current_topic,
        "focus_entities": focus_entities,
    }


def format_state_facts(query_context: Optional[Dict[str, Any]]) -> str:
    """Rendu compact de l'état (sujet + entités), pour les prompts de COMPRÉHENSION.

    Retourne "" si l'état est vide (premier tour) : aucun bloc n'est alors injecté,
    comportement identique à avant.
    """
    qc = query_context or {}
    topic = str(qc.get("current_topic") or "").strip()
    entities = [str(e).strip() for e in (qc.get("focus_entities") or []) if str(e).strip()]

    lines: List[str] = []
    if topic:
        lines.append(f"Sujet courant : {topic}")
    if entities:
        lines.append(f"Références / entités en focus : {', '.join(entities)}")
    return "\n".join(lines)


def format_generation_state_block(
    query_context: Optional[Dict[str, Any]],
    *,
    standalone_question: Optional[str] = None,
    original_message: Optional[str] = None,
) -> str:
    """Bloc « fil de la conversation » injecté dans le message système de GÉNÉRATION.

    Donne au modèle le référent des pronoms/ellipses du dernier message, même quand
    l'historique est loin derrière un très grand contexte documentaire (CAG). La
    question reformulée n'est incluse que si elle apporte quelque chose (différente
    du message brut).
    """
    facts = format_state_facts(query_context)
    if not facts:
        return ""

    lines: List[str] = ["FIL DE LA CONVERSATION :", facts]

    standalone = str(standalone_question or "").strip()
    original = str(original_message or "").strip()
    if standalone and standalone.lower() != original.lower():
        lines.append(
            f"Le dernier message de l'utilisateur, replacé dans ce contexte, signifie : « {standalone} »"
        )

    lines.append(
        "Les pronoms et références implicites du dernier message (« ses », « celui-ci », "
        "« et le… ») renvoient à ce sujet courant, sauf si le message introduit "
        "explicitement un autre sujet. Réponds au dernier message dans la continuité "
        "de ce fil : ne dérive pas vers d'autres éléments du contexte documentaire "
        "qui ne répondent pas à la question posée."
    )
    return "\n".join(lines)
