"""Fiche technique — réponse NATURELLE et sourcée à une demande par référence.

Problème résolu : l'équipe Qualité tape une référence ("profil 6101", "seuil 76180",
"parle-moi de la gamme Textural"). Une référence n'a pas de sens sémantique fort (son
embedding est du bruit) mais une valeur lexicale exacte : on veut donc un retrieval
orienté référence (boost lexical/BM25), puis une réponse claire.

Refonte 2026-07-03 : l'ancien service produisait un GABARIT déterministe rigide (liste de
caractéristiques sourcées page par page) + un dump du graphe KAG (« Entités liées :
6104 (8 mentions)… ») + un pied « Fiabilité ». Résultat mécanique et peu naturel. On
remplace ça par une génération LLM en prose, sur le CONTEXTE CAG (documents entiers,
en-têtes gamme/matériau) — exactement l'infrastructure du chat — avec des sources
au niveau DOCUMENT (« Documents consultés »), cohérentes avec le reste de l'app.

Couches :
  0. DÉTECTION  — ``detect_reference_query`` : regex, zéro LLM, décide du fast-path.
  1. RÉSOLUTION — ``_resolve_passages`` : pages qui citent la référence (boost lexical).
  2. GÉNÉRATION — contexte CAG + LLM (prose naturelle, grounding strict, ton naturel).
  3. SOURCES    — au niveau document (``build_document_sources``).

Point d'entrée : ``build_fiche_technique``. Gardé par settings.FICHE_TECHNIQUE_ENABLED.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlmodel import Session

from app.config import settings
from app.services.mistral_service import chat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Couche 0 — Détection de « référence nue »
# ---------------------------------------------------------------------------

# Marqueurs de question en langage naturel : si présents, ce n'est PAS une référence nue
# → on laisse le pipeline RAG / guidé traiter (comment poser, etc.). NB : "quelle/quel"
# ne bloque PAS ici volontairement — "quelle est la référence 6101 ?" reste une fiche.
_NL_QUESTION_MARKERS = (
    "comment", "pourquoi", "peux-tu", "peux tu", "explique", "expliquer",
    "différence", "difference", "que faire", "à quoi", "a quoi",
)

# Marqueurs documentaires : leur présence autorise une requête un peu plus longue à rester
# en mode fiche ("notice de montage seuil 76180", "parle-moi de la gamme Textural").
_DOC_LOOKUP_MARKERS = (
    "fiche", "notice", "réf", "ref ", "reference", "référence", "profil", "profilé",
    "profile", "seuil", "vis", "gabarit", "joint", "traverse", "dormant", "ouvrant",
    "montage", "cote", "cotes", "spéc", "spec", "quincaillerie", "gamme", "parle-moi",
    "parle moi", "présente", "presente", "caractéristique", "caracteristique",
)


class ReferenceQuery(BaseModel):
    """Résultat de la détection : la ou les références et le contexte utile."""

    references: List[str] = Field(default_factory=list)
    primary: str = ""
    raw_message: str = ""


def _looks_like_year(token: str) -> bool:
    """Un jeton à 4 chiffres dans une plage d'années (millésimes de documents).

    Le corpus est saturé de millésimes (2023-06_DEPLIANT, catalogue 2024…) : sans ce
    filtre, "référentiel gammes 2024" déclencherait une fiche à tort.
    """
    if len(token) != 4 or not token.isdigit():
        return False
    return 1990 <= int(token) <= 2035


def _extract_references(message: str) -> List[str]:
    """Extrait les références techniques (numériques + alphanumériques), dédupliquées."""
    refs: List[str] = []
    seen: set = set()
    for pattern in (settings.FICHE_REFERENCE_ALNUM_PATTERN, settings.FICHE_REFERENCE_PATTERN):
        try:
            for match in re.findall(pattern, message):
                token = str(match).strip()
                key = token.lower()
                if token and key not in seen and not _looks_like_year(token):
                    seen.add(key)
                    refs.append(token)
        except re.error as exc:  # motif env mal formé → on ignore ce motif
            logger.warning("[fiche] motif de référence invalide (%s): %s", pattern, exc)
    return refs


def detect_reference_query(message: str) -> Optional[ReferenceQuery]:
    """Décide si `message` est une demande par référence justifiant le fast-path fiche.

    Deux voies :
      - une RÉFÉRENCE technique (6101, A076, 76180…) hors question conversationnelle ;
      - une demande de PRÉSENTATION de gamme/produit nommé ("parle-moi de la gamme
        Textural") portée par un marqueur documentaire, même sans code chiffré.
    Conservateur : en cas de doute → None (pipeline RAG normal). Aucun appel LLM.
    """
    text = (message or "").strip()
    if not text:
        return None

    lowered = text.lower()

    # 1. Une question ponctuée ("… ?") ou procédurale ("comment", "pourquoi"…) n'est pas
    #    un lookup de fiche → pipeline RAG/guidé. Les demandes de présentation légitimes
    #    ("profil 6101", "parle-moi de la gamme Textural") ne portent pas de "?".
    if "?" in text or any(marker in lowered for marker in _NL_QUESTION_MARKERS):
        return None

    references = _extract_references(text)
    has_doc_marker = any(marker in lowered for marker in _DOC_LOOKUP_MARKERS)
    word_count = len(text.split())

    if references:
        # Référence présente : courte (référence nue) OU portée par un marqueur documentaire.
        if word_count > settings.FICHE_MAX_WORDS and not has_doc_marker:
            return None
        return ReferenceQuery(references=references, primary=references[0], raw_message=text)

    # 2. Pas de code chiffré, mais demande de présentation ("parle-moi de la gamme X") :
    #    on laisse passer si un marqueur documentaire est présent et la requête reste courte.
    if has_doc_marker and word_count <= settings.FICHE_MAX_WORDS:
        return ReferenceQuery(references=[], primary="", raw_message=text)

    return None


# ---------------------------------------------------------------------------
# Couche 1 — Résolution : passages qui citent la référence
# ---------------------------------------------------------------------------


async def _resolve_passages(
    *,
    session: Session,
    space_id: int,
    user_id: int,
    ref_query: ReferenceQuery,
) -> List[Dict[str, Any]]:
    """Récupère les passages mentionnant la référence via le retrieval hybride, en posant
    la référence comme requête (le canal lexical/BM25 fait remonter le code exact) et en
    injectant les références dans les signaux (boost lexical déjà câblé côté retrieval)."""
    from app.services.query_signals_schemas import LightweightQuerySignals
    from app.services.space_search_service import search_technical_passages

    query_text = ref_query.raw_message or ref_query.primary
    signals = LightweightQuerySignals(
        intent="documentation",
        detected_references=ref_query.references,
        entity_texts=ref_query.references,
    )

    try:
        retrieval = await search_technical_passages(
            session=session,
            space_id=space_id,
            query_text=query_text,
            user_id=user_id,
            k=settings.FICHE_TECHNIQUE_K,
            signals=signals,
        )
    except Exception as exc:
        logger.exception("[fiche] retrieval échoué: %s", exc)
        return []

    passages = retrieval.get("passages") or []
    logger.info(
        "[fiche] résolution — refs=%s → %d passage(s)",
        ref_query.references or ref_query.raw_message[:40],
        len(passages),
    )
    return passages


# ---------------------------------------------------------------------------
# Couche 2 — Génération naturelle (prose sur contexte CAG)
# ---------------------------------------------------------------------------

FICHE_SYSTEM_PROMPT = (
    "Tu es LIA, l'assistante technique experte de PROFERM (menuiserie PVC/aluminium). "
    "L'utilisateur demande des informations sur une référence, un profilé ou une gamme. "
    "Tu reçois des DOCUMENTS (avec en-tête source/gamme/matériau et marqueurs [page N]).\n"
    "\n"
    "Rédige une présentation NATURELLE, fluide et COURTE, comme le ferait un expert produit "
    "qui répond en quelques phrases, pas un fiche technique exhaustive :\n"
    "- Commence par une phrase qui situe la référence/gamme (ce que c'est, à quoi ça sert), "
    "si les documents le disent.\n"
    "- N'inclus QUE les informations directement utiles à ce qui est demandé "
    "(caractéristiques, dimensions/cotes, matériau, compatibilités, accessoires associés, "
    "règles de pose clés) — ne déroule pas systématiquement toutes les rubriques possibles. "
    "Prose pour l'essentiel ; une courte liste à puces UNIQUEMENT pour des cotes ou "
    "compatibilités qui s'y prêtent ; pas de sections numérotées ni de schéma improvisé.\n"
    "- Reste concis et va à l'essentiel. N'invente aucune valeur : restitue cotes, "
    "diamètres et références MOT POUR MOT depuis les documents. Vérifie l'en-tête avant "
    "d'attribuer une info à une gamme (ne mélange jamais deux gammes).\n"
    "\n"
    "TON : direct et professionnel, PAS de gabarit rigide, PAS de section « Fiabilité », "
    "PAS de liste d'« entités liées », PAS de disclaimer systématique du type « aucune "
    "désignation explicite ». Si une information courante manque, tu peux le signaler en UNE "
    "phrase à la fin, sobrement — mais seulement si c'est vraiment pertinent pour l'utilisateur. "
    "N'écris aucun marqueur de source entre crochets dans le texte : les sources sont "
    "affichées automatiquement sous ta réponse."
)


def _fiche_user_message(reference_label: str) -> str:
    """Message utilisateur de la fiche (présentation d'une référence/gamme)."""
    if reference_label:
        return (
            f"Présente la référence « {reference_label} » à partir des documents ci-dessus : "
            "ce que c'est et ses informations techniques utiles."
        )
    return (
        "Présente le produit / la gamme demandé(e) à partir des documents ci-dessus : "
        "ce que c'est et ses informations techniques utiles."
    )


async def _generate_fiche_prose(reference_label: str, system_context: str) -> str:
    """Appelle le LLM (sync) pour produire la présentation. Conservé pour compat ;
    le routeur privilégie désormais prepare_fiche_technique + génération streamée."""
    response = await chat(
        "",
        model=settings.MODEL_FAST,
        context=[
            {"role": "system", "content": system_context},
            {"role": "user", "content": _fiche_user_message(reference_label)},
        ],
        temperature=0.2,
        max_tokens=settings.FICHE_MAX_TOKENS,
    )
    return (response["choices"][0]["message"].get("content") or "").strip()


class FicheResult(BaseModel):
    """Sortie prête à streamer par le routeur."""

    markdown: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    # Étiquette de sujet pour le fil de conversation (ex. "profil 6101", "gamme Textural").
    topic: str = ""


class FichePrepared(BaseModel):
    """Contexte de génération fiche PRÉPARÉ (sans texte généré) — permet au routeur de
    streamer la génération avec reasoning/thinking au lieu d'un chat() sync + faux-stream."""

    system_content: str
    user_message: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    topic: str = ""


async def prepare_fiche_technique(
    *,
    session: Session,
    space_id: int,
    user_id: int,
    ref_query: ReferenceQuery,
) -> Optional["FichePrepared"]:
    """Comme build_fiche_technique mais SANS générer le texte : retrieval orienté référence
    → contexte CAG → sources/topic. Retourne le contexte (system + user message) pour une
    génération STREAMÉE par le routeur. None si aucun passage trouvé (→ pipeline RAG normal)."""
    from app.services.context_packer_service import build_cag_context, build_document_sources

    passages = await _resolve_passages(
        session=session, space_id=space_id, user_id=user_id, ref_query=ref_query,
    )
    if not passages:
        return None

    system_message = build_cag_context(
        session, passages, system_prompt=FICHE_SYSTEM_PROMPT,
        intent="documentation", emit_sources_tag=False,
    )
    cag_documents = system_message.get("cag_documents") or []
    reference_label = ref_query.primary or ref_query.raw_message
    return FichePrepared(
        system_content=system_message["content"],
        user_message=_fiche_user_message(reference_label),
        sources=build_document_sources(cag_documents, {}),
        topic=(ref_query.primary or ref_query.raw_message or "").strip()[:120],
    )


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------


async def build_fiche_technique(
    *,
    session: Session,
    space_id: int,
    user_id: int,
    ref_query: ReferenceQuery,
) -> Optional[FicheResult]:
    """Pipeline : retrieval orienté référence → contexte CAG (documents entiers) →
    génération naturelle → sources par document.

    Renvoie None si aucun passage n'est trouvé : le routeur retombe alors sur le pipeline
    RAG normal (message « rien trouvé »)."""
    from app.services.context_packer_service import build_cag_context, build_document_sources

    passages = await _resolve_passages(
        session=session,
        space_id=space_id,
        user_id=user_id,
        ref_query=ref_query,
    )
    if not passages:
        return None

    # Contexte documentaire = même infrastructure que le chat (documents entiers, en-têtes),
    # budget "documentation" (petit) ; on n'exige PAS le bloc <sources> (les sources UI
    # affichent directement les documents packés).
    system_message = build_cag_context(
        session,
        passages,
        system_prompt=FICHE_SYSTEM_PROMPT,
        intent="documentation",
        emit_sources_tag=False,
    )
    cag_documents = system_message.get("cag_documents") or []

    reference_label = ref_query.primary or ref_query.raw_message
    try:
        markdown = await _generate_fiche_prose(reference_label, system_message["content"])
    except Exception as exc:
        logger.exception("[fiche] génération échouée: %s", exc)
        return None

    if not markdown:
        return None

    sources = build_document_sources(cag_documents, {})
    topic = (ref_query.primary or ref_query.raw_message or "").strip()[:120]

    return FicheResult(markdown=markdown, sources=sources, topic=topic)
