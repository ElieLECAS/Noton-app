"""
Critique FAQ post-brouillon : sortie JSON structurée + sanitisation anti fuite méta-texte.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CRITIQUE_SYSTEM_PROMPT = """Tu es un contrôleur qualité de l'assistante LIA.
Tu compares une réponse brouillon aux FAQ correctives officielles.

Règles :
- Si le brouillon est conforme aux FAQ (aucune contradiction, omission ou déformation), decision = "unchanged".
- Si le brouillon contient une erreur signalée par une FAQ, decision = "corrected" et réécris final_message.
- final_message doit être le message utilisateur final uniquement : ton professionnel, chaleureux, concis.
- Pas de commentaire, pas de préambule, pas de guillemets autour du message.
- Si decision = "unchanged", laisse final_message vide.

RETOURNE UNIQUEMENT un objet JSON valide avec exactement ces champs :
{
  "decision": "unchanged" | "corrected",
  "final_message": "..."
}
Aucun texte avant ou après le JSON."""

_META_MARKERS = (
    "la réponse temporaire est correcte",
    "voici la réponse à envoyer",
    "sans modification",
    "conforme aux faq",
    "conforme à la faq",
    "conforme aux données",
    "ne contredit",
    "réponse à envoyer à l'utilisateur",
)

_MIN_USEFUL_LENGTH = 20


def build_critique_messages(draft_response: str, faq_formatted_text: str) -> List[Dict[str, str]]:
    """Construit les messages Mistral pour l'étape critique FAQ (sortie JSON)."""
    user_content = f"""Brouillon de l'assistante :
{draft_response}

FAQ correctives officielles :
{faq_formatted_text}

Compare le brouillon aux FAQ et retourne le JSON demandé."""
    return [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _strip_outer_quotes(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'", "«", "»"):
        return text[1:-1].strip()
    return text


def _extract_quoted_block(text: str) -> Optional[str]:
    """Extrait le contenu après un préambule type « Voici la réponse... » entre guillemets."""
    match = re.search(
        r"(?:voici la réponse[^:]*:|sans modification\s*:)\s*[\"«](.+?)[\"»]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return None


def _contains_meta_markers(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _META_MARKERS)


def sanitize_critique_output(raw: str, draft_response: str) -> str:
    """
    Nettoie une sortie critique texte libre (filet de sécurité).
    Retourne draft_response si le résultat reste invalide ou méta.
    """
    if not raw or not raw.strip():
        return draft_response

    text = raw.strip()

    # Retirer un bloc méta en tête avant la vraie réponse quotée
    quoted = _extract_quoted_block(text)
    if quoted:
        text = quoted

    # Supprimer les lignes méta en début de texte
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    skipping_meta = True
    for line in lines:
        line_stripped = line.strip()
        if skipping_meta and line_stripped and _contains_meta_markers(line_stripped):
            continue
        if skipping_meta and line_stripped.lower().startswith("voici la réponse"):
            continue
        skipping_meta = False
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()
    text = _strip_outer_quotes(text)

    if not text or len(text) < _MIN_USEFUL_LENGTH or _contains_meta_markers(text):
        return draft_response

    return text


def parse_critique_json(raw: str, draft_response: str) -> str:
    """
    Parse la réponse JSON du LLM critique.
    - unchanged → brouillon exact
    - corrected → final_message sanitizé
    - échec parse → sanitize du raw, sinon brouillon
    """
    if not raw or not raw.strip():
        return draft_response

    content = raw.strip()

    # Extraire un objet JSON si entouré de texte parasite
    json_match = re.search(r"\{[\s\S]*\}", content)
    if json_match:
        content = json_match.group(0)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Critique FAQ : JSON invalide, fallback sanitize")
        return sanitize_critique_output(raw, draft_response)

    if not isinstance(data, dict):
        return sanitize_critique_output(raw, draft_response)

    decision = str(data.get("decision", "")).strip().lower()
    final_message = data.get("final_message")

    if decision == "unchanged":
        return draft_response

    if decision == "corrected":
        if not isinstance(final_message, str) or not final_message.strip():
            logger.warning("Critique FAQ : corrected sans final_message, fallback brouillon")
            return draft_response
        sanitized = sanitize_critique_output(final_message.strip(), draft_response)
        return sanitized

    logger.warning("Critique FAQ : decision inconnue '%s', fallback sanitize", decision)
    return sanitize_critique_output(raw, draft_response)


def resolve_critique_final(raw_llm_output: str, draft_response: str) -> str:
    """Point d'entrée : parse JSON puis sanitize, repli sur le brouillon."""
    return parse_critique_json(raw_llm_output, draft_response)
