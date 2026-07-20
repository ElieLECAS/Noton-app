"""Vérification post-génération — détecte les réponses hors-sujet ou hallucinées.

Deux contrôles complémentaires, exécutés APRÈS la génération complète (le texte a
déjà streamé au client — voir docs/plan_p2_generation_small_verification_2026-07-20.md
§2.3 pour la contrainte UX) :

  1. ``check_grounding`` — programmatique, zéro LLM : extrait les normes/cotes/références
     citées dans la réponse et vérifie leur présence LITTÉRALE dans le contexte packé.
     A détecté 100% des fabrications du cas réel du 20/07 (normes NF inventées, calcul
     arithmétique présenté comme une cote du document).
  2. ``judge_relevance`` — un appel LLM court (modèle configurable, température 0.0) juge
     si la réponse répond réellement à la question posée et si elle semble s'appuyer sur
     le contexte fourni (angle mort du contrôle programmatique : le hors-sujet confiant).

Point d'entrée : ``verify_response``. Ne bloque jamais la génération ; le résultat est
loggé et destiné à ``Message.metadata_json["verification"]``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contrôle 1 — programmatique (zéro LLM)
# ---------------------------------------------------------------------------

_NORM_PATTERNS = (
    re.compile(r"\bNF\s?[A-Z]{0,3}\s?[\d][\d\.\-]*\b", re.IGNORECASE),
    re.compile(r"\bDTU\s?\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bFD\s?DTU\s?\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bEN\s?\d+(?:-\d+)?\b", re.IGNORECASE),
)

# Cotes numériques avec unité (mm/cm/m/kg/N/°). Le \b final évite de capturer un
# fragment de nombre plus long (ex. "12" dans "1200").
_MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:mm|cm|m|kg|N|°)\b", re.IGNORECASE
)

# Longueur mini pour qu'une chaîne extraite soit considérée comme une affirmation
# vérifiable (évite de signaler des faux positifs sur des fragments trop courts).
_MIN_CLAIM_LEN = 3


def _normalize_for_match(value: str) -> str:
    """Normalise une chaîne pour comparaison tolérante (espaces, virgule/point, casse)."""
    value = value.replace(",", ".")
    value = re.sub(r"\s+", "", value)
    return value.strip().lower()


def extract_verifiable_claims(text: str) -> List[str]:
    """Extrait les normes et cotes numériques citées dans un texte, dédupliquées."""
    if not text:
        return []
    claims: List[str] = []
    seen = set()
    for pattern in (*_NORM_PATTERNS, _MEASUREMENT_PATTERN):
        for match in pattern.finditer(text):
            raw = match.group(0).strip()
            if len(raw) < _MIN_CLAIM_LEN:
                continue
            key = _normalize_for_match(raw)
            if key and key not in seen:
                seen.add(key)
                claims.append(raw)
    return claims


def check_grounding(response_text: str, context_text: str) -> List[str]:
    """Retourne les affirmations (normes/cotes) de `response_text` ABSENTES de
    `context_text`. Liste vide = pas de problème détecté par ce contrôle.

    LIMITE CONNUE (validée sur cas réel 2026-07-20) : ce contrôle est un test de
    présence littérale, pas d'attribution. Une norme/cote RÉELLEMENT présente dans
    le document mais RATTACHÉE À TORT à un fait différent (ex. une norme sur les
    charges mécaniques citée pour justifier une hauteur de poignée) n'est PAS
    détectée ici — seule une invention totale (chaîne absente du document) l'est.
    C'est le rôle du juge LLM (judge_relevance) de tenter d'attraper ce cas plus
    subtil, sans garantie non plus."""
    claims = extract_verifiable_claims(response_text)
    if not claims:
        return []
    normalized_context = _normalize_for_match(context_text or "")
    unsupported = [
        claim for claim in claims
        if _normalize_for_match(claim) not in normalized_context
    ]
    return unsupported


# ---------------------------------------------------------------------------
# Contrôle 2 — jugement LLM (pertinence / grounding global)
# ---------------------------------------------------------------------------

_VERIFICATION_SYSTEM_PROMPT = """Tu es un contrôleur qualité qui vérifie une réponse d'assistant technique AVANT qu'elle soit archivée.

Tu reçois : la question de l'utilisateur, la réponse générée par l'assistant, et un extrait du contexte documentaire qui a servi à la générer.

Juge deux choses :
1. answers_question : la réponse traite-t-elle la demande de façon APPROPRIÉE au vu du contexte ?
   - Une abstention HONNÊTE est appropriée : si le contexte ne contient PAS l'information demandée
     et que la réponse le dit clairement (« le document ne précise pas... »), alors answers_question=true.
     Ne pénalise PAS une abstention justifiée — c'est le comportement attendu, pas une erreur.
   - answers_question=false UNIQUEMENT si la réponse part sur un sujet VOISIN en le présentant comme
     la réponse, esquive la question alors que le contexte y répond, ou noie la réponse sous du hors-sujet.
2. grounded : les affirmations de la réponse s'appuient-elles sur le contexte fourni, sans invention
   de valeurs, normes ou détails absents ? Une abstention est grounded par nature.

Sur grounded, sois strict : en cas de doute sur une valeur/norme inventée, grounded=false.

RETOURNE UNIQUEMENT un objet JSON valide avec exactement ces champs :
{
  "answers_question": true|false,
  "grounded": true|false,
  "issues": ["description courte de chaque problème détecté, vide si aucun"]
}
Aucun texte avant ou après le JSON."""

# Troncature de l'extrait de contexte envoyé au juge (coût/latence ; le contrôle
# programmatique, lui, scanne le contexte COMPLET sans troncature).
_CONTEXT_EXCERPT_CHARS = 20000


def build_verification_messages(
    question: str, response_text: str, context_text: str
) -> List[Dict[str, str]]:
    excerpt = (context_text or "")[:_CONTEXT_EXCERPT_CHARS]
    user_content = (
        f"Question de l'utilisateur :\n{question}\n\n"
        f"Réponse générée par l'assistant :\n{response_text}\n\n"
        f"Extrait du contexte documentaire utilisé :\n{excerpt}\n\n"
        "Juge la réponse selon les règles du système et retourne le JSON demandé."
    )
    return [
        {"role": "system", "content": _VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_verification_json(raw: str) -> Dict[str, Any]:
    """Parse la sortie JSON du juge. Repli permissif (answers_question=True,
    grounded=True) si le parsing échoue — un échec technique ne doit pas se
    traduire par un faux signal d'alerte."""
    fallback = {"answers_question": True, "grounded": True, "issues": [], "parse_error": True}
    if not raw or not raw.strip():
        return fallback
    content = raw.strip()
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        content = match.group(0)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("[verification] JSON invalide côté juge, repli permissif")
        return fallback
    if not isinstance(data, dict):
        return fallback
    return {
        "answers_question": bool(data.get("answers_question", True)),
        "grounded": bool(data.get("grounded", True)),
        "issues": [str(i) for i in (data.get("issues") or []) if str(i).strip()],
        "parse_error": False,
    }


async def judge_relevance(
    question: str,
    response_text: str,
    context_text: str,
    *,
    model: str,
) -> Dict[str, Any]:
    """Appel LLM de jugement (température 0.0). N'échoue jamais l'appelant :
    en cas d'erreur API, repli permissif + log."""
    from app.services.mistral_service import chat

    try:
        result = await chat(
            "",
            model=model,
            context=build_verification_messages(question, response_text, context_text),
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = (result.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return parse_verification_json(raw)
    except Exception as exc:
        logger.warning("[verification] Échec appel juge LLM (%s) — repli permissif", exc)
        return {"answers_question": True, "grounded": True, "issues": [], "parse_error": True}


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


async def verify_response(
    *,
    question: str,
    response_text: str,
    context_text: str,
    model: str,
) -> Dict[str, Any]:
    """Orchestre les deux contrôles. Ne lève jamais — un échec de vérification
    ne doit pas faire échouer la persistance de la réponse déjà affichée.

    Retourne un dict prêt à stocker dans Message.metadata_json["verification"] :
    {
      "ok": bool,                       # False si un problème a été détecté
      "unsupported_claims": [...],      # contrôle 1
      "answers_question": bool,         # contrôle 2
      "grounded": bool,                 # contrôle 2
      "issues": [...],                  # contrôle 2
    }
    """
    unsupported = check_grounding(response_text, context_text)

    llm_result = await judge_relevance(question, response_text, context_text, model=model)

    ok = not unsupported and llm_result["answers_question"] and llm_result["grounded"]

    result = {
        "ok": ok,
        "unsupported_claims": unsupported,
        "answers_question": llm_result["answers_question"],
        "grounded": llm_result["grounded"],
        "issues": llm_result["issues"],
    }

    if not ok:
        logger.warning(
            "[verification] Problème détecté — unsupported=%s answers_question=%s "
            "grounded=%s issues=%s | question=%r",
            unsupported,
            llm_result["answers_question"],
            llm_result["grounded"],
            llm_result["issues"],
            question[:120],
        )

    return result
