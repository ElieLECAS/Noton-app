"""
Génération de texte technique RAG-friendly à partir des feedbacks utilisateur.
Uniquement à partir de la question, de la réponse assistant et du commentaire — sans enrichissement externe.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal, TYPE_CHECKING

from app.config import settings
from app.services.mistral_service import chat as mistral_chat

if TYPE_CHECKING:
    from app.models.message_feedback import MessageFeedback

logger = logging.getLogger(__name__)

FEEDBACK_KNOWLEDGE_CONTENT_TYPE = "feedback_knowledge"
FeedbackKnowledgeMode = Literal["corrective", "enrichment"]

# Sortie courte : titre + 1–3 phrases + mots-clés
FEEDBACK_KNOWLEDGE_MAX_TOKENS = 320

_STRICT_RULES = """
RÈGLES IMPÉRATIVES (prioritaires sur tout le reste) :
1. Utilise UNIQUEMENT les informations présentes dans les trois blocs « Question », « Réponse assistant » et « Commentaire utilisateur » ci-dessous.
2. N'invente AUCUNE référence produit, norme, marque, valeur numérique ou procédure absente de ces blocs.
3. Ne recopie PAS la réponse assistant en entier : synthétise question + correction/précision en 1 à 3 phrases techniques maximum.
4. En cas de contradiction, le commentaire utilisateur prime sur la réponse assistant.
5. Ne traite QUE le point soulevé par le commentaire ; n'élargis pas le sujet.
6. Si une information manque, ne la déduis pas — omets-la ou écris « non précisé ».
"""

_OUTPUT_FORMAT = """
Format de sortie (Markdown uniquement, sans préambule, sans bloc ```markdown) :

# [Titre court — 5 à 12 mots, sujet technique issu de la question]

[Corps : 1 à 3 phrases maximum, denses et exploitables pour la recherche (RAG).
- Phrase(s) = réinterprétation technique de la question + correction ou précision utilisateur.
- Inclure les références produit (ex. T411005, T141037) et l'ordre de montage/pose quand le commentaire ou la réponse les précisent.
- Pas de listes numérotées longues, pas de sections ## Question / ## Réponse.
- Style notice technique : verbes d'action, séquence claire, vocabulaire métier.]

**Mots-clés :** [5 à 10 termes ou expressions extraits des blocs, séparés par des virgules — pas de termes inventés]
"""


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n?(.*?)\n?```\s*$", t, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def build_feedback_knowledge_prompt(
    feedback: "MessageFeedback",
    *,
    mode: FeedbackKnowledgeMode,
) -> str:
    """Construit le prompt Mistral — sources strictement limitées au feedback."""
    query = (feedback.query_text or "").strip() or "(non fournie)"
    response = (feedback.response_text or "").strip() or "(non fournie)"
    comment = (feedback.comment or "").strip() or "(non fourni)"

    sources = f"""Question utilisateur :
{query}

Réponse assistant :
{response}

Commentaire utilisateur :
{comment}
"""

    if mode == "corrective":
        role = (
            "Tu rédiges une fiche technique courte pour une base documentaire (RAG). "
            "L'utilisateur a signalé une erreur dans la réponse assistant."
        )
        task = (
            "Produis 1 à 3 phrases qui expriment la BONNE procédure ou le BON fait technique, "
            "en intégrant la correction du commentaire. "
            "Exemple de forme attendue : « Le joint T411005 se pose sur la tranche du vitrage "
            "avant d'emboîter le montant central T141037 sur les supports T401005 ; "
            "ne pas le placer dans la feuillure du dormant. » "
            "Ne reformule pas toute la réponse assistant si le commentaire ne corrige qu'un point précis."
        )
    else:
        role = (
            "Tu rédiges une fiche technique courte pour une base documentaire (RAG). "
            "L'utilisateur a validé la réponse et ajoute une précision."
        )
        task = (
            "Produis 1 à 3 phrases qui conservent l'essentiel de la réponse assistant "
            "et intègrent UNIQUEMENT la précision du commentaire, sans la développer au-delà de ce qui est dit."
        )

    return f"{role}\n\n{_STRICT_RULES}\n\n{task}\n\n{sources}\n{_OUTPUT_FORMAT}"


def feedback_knowledge_mode(feedback: "MessageFeedback") -> FeedbackKnowledgeMode:
    return "enrichment" if feedback.is_positive else "corrective"


async def generate_feedback_knowledge_content_async(feedback: "MessageFeedback") -> str:
    """Appelle Mistral et retourne le markdown généré (sources feedback uniquement)."""
    mode = feedback_knowledge_mode(feedback)
    prompt = build_feedback_knowledge_prompt(feedback, mode=mode)
    response = await mistral_chat(
        prompt,
        settings.MODEL_FAST,
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=FEEDBACK_KNOWLEDGE_MAX_TOKENS,
    )
    generated = ""
    if response and "choices" in response and len(response["choices"]) > 0:
        generated = response["choices"][0]["message"].get("content", "").strip()
    generated = _strip_code_fences(generated)
    if not generated:
        raise ValueError("Mistral n'a généré aucun contenu pour le feedback")
    return generated


def generate_feedback_knowledge_content(feedback: "MessageFeedback") -> str:
    """Version synchrone pour tâches Celery."""
    return asyncio.run(generate_feedback_knowledge_content_async(feedback))
