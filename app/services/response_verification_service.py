"""Contrôle programmatique de sortie du lecteur — zéro LLM.

Le lecteur agentique (``reader_agent_service``) répond après avoir lu : le pack initial
puis ce que ses outils lui ont rendu. Ce module confronte sa réponse à **exactement cette
matière** (pack ∪ résultats d'outils) — le lecteur et le contrôle lisent la même chose,
ce qui n'était pas le cas de l'ancien juge LLM (texte des pages vs images vues).

Trois contrôles, tous littéraux et insensibles au modèle :

  1. ``check_grounding`` — normes / cotes / teintes RAL citées dans la réponse et absentes
     du texte lu. A détecté 100 % des fabrications du cas réel du 20/07.
  2. ``check_reference_grounding`` / ``unsupported_reference_codes`` — CODES PRODUITS
     alphanumériques (TGY3710, 9F67…) absents du texte lu ET de la question : la
     référence inventée par analogie de numérotation meurt ici.
  3. ``evidence_in_pack`` — les citations ``<evidence>`` du lecteur doivent se retrouver
     mot pour mot dans ce qu'il a lu (accents, casse, espaces tolérés ; découpe sur les
     ellipses). Contrôle SOUPLE : une citation introuvable est signalée, pas bloquante —
     une valeur lue sur une planche muette n'a pas de texte à citer.

Point d'entrée : ``check_reader_output``. Quand le verdict est KO, il produit aussi le
message de retour destiné au lecteur (message ``user`` de contrôle) : « ces éléments ne
figurent dans aucune page lue : … ; vérifie avec chercher_code / lire_pages, ou écris
explicitement l'absence ». Le lecteur relit ; il ne réécrit pas à l'aveugle.

L'ancien juge LLM (``judge_relevance`` / ``verify_response``) a été retiré le 2026-09-02 :
un petit modèle qui ne voit pas les images et lit un extrait plafonné ne peut pas juger
à la place de celui qui a lu — et ses verdicts « aveugles » finissaient marqués SUSPECT.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Contrôle 1 — normes / cotes / RAL
# ---------------------------------------------------------------------------

_NORM_PATTERNS = (
    re.compile(r"\bNF\s?[A-Z]{0,3}\s?[\d][\d\.\-]*\b", re.IGNORECASE),
    re.compile(r"\bDTU\s?\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bFD\s?DTU\s?\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bEN\s?\d+(?:-\d+)?\b", re.IGNORECASE),
    # Teintes RAL. Cas réel du 01/09 : une liste de RAL (1015, 7035, 9005…) absents de
    # TOUT le corpus est partie à l'utilisateur. Écrite « RAL 9016 » avec une espace, elle
    # échappait au contrôle des codes produits (qui exige lettres et chiffres contigus) et
    # le contrôle des normes ne connaissait pas le RAL. La normalisation retire les espaces :
    # « RAL 9016 » et « RAL9016 » se comparent à l'identique côté réponse et côté contexte.
    re.compile(r"\bRAL\s?\d{4}\b", re.IGNORECASE),
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
    le document mais RATTACHÉE À TORT à un fait différent n'est PAS détectée ici —
    seule une invention totale (chaîne absente du document) l'est."""
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
# Contrôle 2 — codes produits
# ---------------------------------------------------------------------------

# Un code produit vérifiable côté RÉPONSE doit porter au moins une lettre : les codes
# purement numériques (« 6111 », « 155 ») sont indiscernables des quantités, années et
# numéros de page (« page 111 ») dans un texte généré — le contrôle ne doit jamais
# accuser à tort. TGY3702 / 9F67 / RAL9016, eux, sont des références sans ambiguïté.
_CODE_HAS_LETTER = re.compile(r"[A-Za-z]")


def extract_response_reference_codes(text: str, *, limit: int = 16) -> List[str]:
    """Codes produits ALPHANUMÉRIQUES cités dans une réponse générée, dédupliqués."""
    from app.services.reference_codes import REF_CODE_RE

    codes: List[str] = []
    seen: set = set()
    for match in REF_CODE_RE.finditer(text or ""):
        tok = match.group(0).strip().upper()
        if not tok or not _CODE_HAS_LETTER.search(tok):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        codes.append(tok)
        if len(codes) >= limit:
            break
    return codes


def unsupported_reference_codes(
    response_text: str,
    context_text: str,
    *,
    question: Optional[str] = None,
    user_message: Optional[str] = None,
) -> List[str]:
    """Codes de la réponse absents du contexte ET absents de la question.

    Un code que l'utilisateur a lui-même tapé (« la rallonge TGY3704 ») n'est pas une
    invention du modèle, même si le texte lu ne le contient pas — cas fréquent des
    planches CAO, où la référence n'existe qu'en image.
    """
    unsupported = check_reference_grounding(response_text, context_text)
    if not unsupported:
        return []
    asked = " ".join(t for t in (question, user_message) if t)
    if not asked:
        return unsupported
    from app.services.reference_codes import code_in_text

    return [code for code in unsupported if not code_in_text(code, asked)]


def check_reference_grounding(response_text: str, context_text: str) -> List[str]:
    """Codes produits de ``response_text`` ABSENTS de ``context_text``.

    Présence testée avec frontières alphanumériques (« TGY371 » ne matche pas dans
    « TGY3710 »), insensible à la casse. C'est LE contrôle qui attrape le cas TGY3710 :
    référence inventée, plausible, citée avec assurance — chaîne introuvable dans ce
    que le lecteur a lu."""
    from app.services.reference_codes import code_in_text

    return [
        code
        for code in extract_response_reference_codes(response_text)
        if not code_in_text(code, context_text or "")
    ]


# ---------------------------------------------------------------------------
# Contrôle 3 — citations verbatim
# ---------------------------------------------------------------------------

# Longueur minimale (caractères normalisés) d'une citation pour valoir preuve : en
# dessous, n'importe quel fragment (« TGY », « 3 points ») serait retrouvé par hasard.
_EVIDENCE_MIN_CHARS = 12
# Longueur minimale d'un segment de citation (découpe sur les ellipses).
_EVIDENCE_SEGMENT_MIN_CHARS = 20
_ELLIPSIS_SPLIT = re.compile(r"…|\.\.\.|\[\.\.\.\]|\[…\]")


def _normalize_for_evidence(value: str) -> str:
    """Normalisation tolérante pour la preuve : accents retirés, espaces repliés,
    guillemets/puces neutralisés, casse ignorée."""
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("’", "'").replace("«", '"').replace("»", '"')
    value = re.sub(r"[\s ]+", " ", value)
    return value.strip().lower()


def evidence_in_pack(evidence: str, pack_text: str) -> bool:
    """La citation est-elle LITTÉRALEMENT dans le texte lu ?

    Tolérances : accents/casse/espaces, et découpe sur les ellipses (« … ») — chaque
    segment substantiel doit être retrouvé."""
    ev = _normalize_for_evidence(evidence)
    if len(ev) < _EVIDENCE_MIN_CHARS:
        return False
    pack = _normalize_for_evidence(pack_text)
    if not pack:
        return False
    if ev in pack:
        return True
    segments = [
        seg.strip(" \"'.,;:-")
        for seg in _ELLIPSIS_SPLIT.split(ev)
        if len(seg.strip(" \"'.,;:-")) >= _EVIDENCE_SEGMENT_MIN_CHARS
    ]
    if not segments:
        return False
    return all(seg in pack for seg in segments)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def check_reader_output(
    *,
    response_text: str,
    evidence_text: str,
    question: Optional[str] = None,
    user_message: Optional[str] = None,
    citations: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Confronte la réponse du lecteur à ce qu'il a lu. Ne lève jamais.

    Retourne un dict prêt pour ``Message.metadata_json["verification"]`` :
    {
      "ok": bool,                     # False si un code/une cote/une norme est non étayé
      "unsupported_claims": [...],    # normes, cotes, RAL puis codes (fusionnés)
      "unsupported_codes": [...],     # codes produits seuls
      "citations_total": int,
      "citations_unverified": [...],  # citations <evidence> introuvables (souple)
      "feedback": str | None,         # message de contrôle à renvoyer au lecteur si KO
      "action": None,                 # posé par l'appelant : passed | repaired | flagged
    }
    """
    evidence_text = evidence_text or ""
    unsupported = check_grounding(response_text, evidence_text)
    unsupported_codes = unsupported_reference_codes(
        response_text, evidence_text, question=question, user_message=user_message
    )
    for code in unsupported_codes:
        if code not in unsupported:
            unsupported.append(code)

    from app.services.stream_source_filter import strip_citation_location

    cites = [c for c in (citations or []) if c and c.strip()]
    unverified = [
        c for c in cites if not evidence_in_pack(strip_citation_location(c), evidence_text)
    ]

    ok = not unsupported
    feedback: Optional[str] = None
    if not ok:
        bits = [
            "CONTRÔLE DOCUMENTAIRE (automatique, fait foi) : les éléments suivants de ta "
            "réponse ne figurent dans AUCUNE page que tu as lue : "
            + ", ".join(str(c) for c in unsupported[:10])
            + "."
        ]
        if unsupported_codes:
            bits.append(
                "Pour chaque référence, vérifie son existence avec chercher_code (ou relis la "
                "page avec lire_pages / zoomer). Si elle n'existe pas dans les documents, "
                "écris explicitement que les documents ne la mentionnent pas — ne la déduis "
                "jamais d'une numérotation voisine."
            )
        else:
            bits.append(
                "Relis la page qui porte la valeur (lire_pages, ou zoomer si elle est trop "
                "petite) et cite-la exactement ; sinon écris que les documents ne précisent "
                "pas cette valeur."
            )
        if unverified:
            bits.append(
                f"{len(unverified)} citation(s) de ton bloc <evidence> n'ont pas été retrouvées "
                "mot pour mot : copie les phrases exactes des pages lues."
            )
        bits.append(
            "Puis réponds à nouveau, complètement, en conservant les blocs <sources> et "
            "<evidence> en fin de réponse."
        )
        feedback = " ".join(bits)
        logger.warning(
            "[contrôle] réponse non étayée — %s | question=%r", unsupported, (question or "")[:120]
        )
    elif unverified:
        logger.info(
            "[contrôle] %d/%d citation(s) <evidence> non retrouvée(s) mot pour mot (souple)",
            len(unverified),
            len(cites),
        )

    return {
        "ok": ok,
        "unsupported_claims": unsupported,
        "unsupported_codes": unsupported_codes,
        "citations_total": len(cites),
        "citations_unverified": unverified,
        "feedback": feedback,
        "action": None,
    }
