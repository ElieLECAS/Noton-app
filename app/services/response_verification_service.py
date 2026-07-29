"""Vérification post-génération — détecte les réponses hors-sujet ou hallucinées.

Deux contrôles complémentaires, exécutés APRÈS la génération complète (le texte a
déjà streamé au client — voir docs/plan_p2_generation_small_verification_2026-07-20.md
§2.3 pour la contrainte UX) :

  1. ``check_grounding`` — programmatique, zéro LLM : extrait les normes/cotes citées dans
     la réponse et vérifie leur présence LITTÉRALE dans le contexte packé. A détecté 100%
     des fabrications du cas réel du 20/07 (normes NF inventées, calcul arithmétique
     présenté comme une cote du document).
  1bis. ``check_reference_grounding`` (B7a, plan 2026-07-29) — même principe pour les
     CODES PRODUITS alphanumériques de la réponse (TGY3710, 9F67…). Cas réel du 29/07 :
     une référence inventée (TGY3710) ne pouvait être signalée que par le juge LLM en
     texte libre — le contrôle programmatique ne connaissait que normes et cotes.
  2. ``judge_relevance`` — un appel LLM court (modèle configurable, température 0.0) juge
     si la réponse répond réellement à la question posée et si elle semble s'appuyer sur
     le contexte fourni (angle mort du contrôle programmatique : le hors-sujet confiant).
     Tri-état ``judge_status`` : "ok" (verdict exploitable) | "unknown" (échec technique —
     parse/API). Un échec d'infra n'est NI un blanc-seing NI une alerte.

Point d'entrée : ``verify_response``. Ce module ne bloque jamais par lui-même ; c'est
l'appelant (chat.py, mode VERIFY_BLOCKING) qui décide quoi faire du résultat, destiné à
``Message.metadata_json["verification"]``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

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


def check_reference_grounding(response_text: str, context_text: str) -> List[str]:
    """Codes produits de ``response_text`` ABSENTS de ``context_text`` (B7a).

    Présence testée avec frontières alphanumériques (« TGY371 » ne matche pas dans
    « TGY3710 »), insensible à la casse. Le contexte passé est le contexte COMPLET
    (jamais tronqué), donc ce contrôle est insensible au plafond du juge LLM.
    C'est LE contrôle qui attrape le cas TGY3710 : référence inventée, plausible,
    citée avec assurance — chaîne introuvable dans les documents packés."""
    from app.services.reference_codes import code_in_text

    return [
        code
        for code in extract_response_reference_codes(response_text)
        if not code_in_text(code, context_text or "")
    ]


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

RÈGLE D'ABSENCE — le contexte peut t'être fourni PARTIELLEMENT (un manifeste liste alors
TOUS les documents réellement donnés à l'assistant, suivi du contenu le plus pertinent) :
  - Ne conclus JAMAIS « le contexte ne contient pas X » ou « le contexte ne parle que de
    la gamme Y » sur la base d'un extrait partiel. Un document listé au manifeste FAIT
    PARTIE du contexte même si son texte n'apparaît pas ci-dessous.
  - Si une affirmation de la réponse renvoie à un document du manifeste dont le texte ne
    t'est pas montré, considère-la comme NON VÉRIFIABLE — pas comme inventée — et
    signale-le dans issues sans passer grounded à false pour autant.

RETOURNE UNIQUEMENT un objet JSON valide avec exactement ces champs :
{
  "answers_question": true|false,
  "grounded": true|false,
  "issues": ["description courte de chaque problème détecté, vide si aucun"]
}
Aucun texte avant ou après le JSON."""

# Repère de page inséré par le packer (``_render_document_block``) : sert à découper un
# bloc document en unités de page pour le remplissage par les preuves.
_PAGE_MARKER = re.compile(r"^\[page (\d+)(?: — ★[^\]]*)?\]$", re.MULTILINE)


def _document_manifest(cag_documents: List[Dict[str, Any]]) -> str:
    """Liste de TOUS les documents packés — jamais tronquée.

    C'est le garde-fou structurel : sans lui, un juge qui ne reçoit que le début du
    contexte conclut « le contexte concerne exclusivement la gamme X » alors qu'un
    document de la gamme Y était packé plus loin (cas réel du 27/07). ~200 caractères
    par document, donc négligeable devant le budget d'extrait.
    """
    if not cag_documents:
        return ""
    lines = [
        "DOCUMENTS RÉELLEMENT FOURNIS À L'ASSISTANT "
        "(manifeste COMPLET — aucun document n'est omis de cette liste) :"
    ]
    for doc in cag_documents:
        pages = doc.get("pages") or []
        span = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else (str(pages[0]) if pages else "—")
        seeds = doc.get("seed_pages") or []
        scope = "document complet" if doc.get("full_document") else "extrait"
        detail = f"pages {span} ({scope}, {len(pages)} page(s))"
        if seeds:
            detail += f" · pages retrouvées par la recherche : {', '.join(str(s) for s in seeds)}"
        lines.append(f"  [{doc.get('index')}] {doc.get('document_title') or 'Sans titre'} — {detail}")
    return "\n".join(lines)


def _split_block_into_pages(block: str) -> List[Tuple[Optional[int], str]]:
    """Découpe un bloc document en (page_no, texte). L'en-tête précède la 1re page."""
    markers = list(_PAGE_MARKER.finditer(block))
    if not markers:
        return [(None, block)]
    units: List[Tuple[Optional[int], str]] = []
    header = block[: markers[0].start()].strip()
    if header:
        units.append((None, header))
    for idx, match in enumerate(markers):
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(block)
        units.append((int(match.group(1)), block[match.start() : end].strip()))
    return units


def build_verification_context(
    document_blocks: List[str],
    cag_documents: List[Dict[str, Any]],
    *,
    cited_pages: Optional[Dict[int, List[int]]] = None,
    max_chars: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Assemble le contexte soumis au juge et le rapport de couverture.

    Trois différences avec l'ancien ``context_text[:20000]`` :

    * on part des **blocs documents seuls** — le prompt système (5 000+ caractères de
      consignes) n'a aucune valeur probante pour un jugement de grounding ;
    * le **manifeste** de tous les documents packés est toujours présent ;
    * sous plafond, le remplissage suit les **preuves** (pages citées par la réponse, puis
      pages retrouvées par la recherche, puis le reste) et non l'ordre du contexte, si
      bien que ce qui fonde la réponse est montré en priorité.

    Retourne ``(texte, {"coverage", "truncated", "chars_total", "chars_sent", "documents"})``.
    """
    limit = settings.VERIFICATION_CONTEXT_MAX_CHARS if max_chars is None else max_chars
    manifest = _document_manifest(cag_documents) if settings.VERIFICATION_INCLUDE_MANIFEST else ""
    blocks = [b for b in (document_blocks or []) if b]
    total = sum(len(b) for b in blocks)

    report: Dict[str, Any] = {
        "chars_total": total,
        "documents": len(cag_documents or []),
        "manifest": bool(manifest),
    }

    def _finish(body: str, truncated: bool) -> Tuple[str, Dict[str, Any]]:
        text = f"{manifest}\n\n{body}" if manifest else body
        report["chars_sent"] = len(body)
        report["truncated"] = truncated
        # Bornée à 1.0 : le corps assemblé porte des séparateurs absents du total brut.
        report["coverage"] = round(min(1.0, len(body) / total), 3) if total else 1.0
        return text, report

    if not blocks:
        return _finish("", False)

    # Cas nominal : tout tient (limite nulle = illimité).
    if limit <= 0 or total <= limit:
        return _finish("\n\n".join(blocks), False)

    # Sous plafond : remplissage par les preuves d'abord.
    cited = {int(k): {int(p) for p in v} for k, v in (cited_pages or {}).items()}
    doc_by_index = {d.get("index"): d for d in (cag_documents or [])}

    units: List[Tuple[int, int, int, str]] = []  # (priorité, ordre doc, page, texte)
    for position, block in enumerate(blocks):
        doc = doc_by_index.get(position + 1) or {}
        doc_id = int(doc.get("document_id") or -1)
        seeds = {int(s) for s in (doc.get("seed_pages") or [])}
        cited_here = cited.get(doc_id, set())
        for page_no, text in _split_block_into_pages(block):
            if page_no is None:
                priority = 0  # en-tête de document : identité produit, toujours en premier
            elif page_no in cited_here:
                priority = 1
            elif page_no in seeds:
                priority = 2
            else:
                priority = 3
            units.append((priority, position, page_no or 0, text))

    units.sort(key=lambda u: (u[0], u[1], u[2]))
    kept: List[Tuple[int, int, str]] = []
    used = 0
    for priority, position, page_no, text in units:
        if used + len(text) + 2 > limit and kept:
            continue
        kept.append((position, page_no, text))
        used += len(text) + 2

    kept.sort(key=lambda u: (u[0], u[1]))
    body = "\n\n".join(text for _, _, text in kept)
    return _finish(body, True)


def build_verification_messages(
    question: str, response_text: str, context_text: str
) -> List[Dict[str, str]]:
    """Messages du juge. ``context_text`` est déjà assemblé et borné par
    ``build_verification_context`` — aucune troncature supplémentaire ici."""
    user_content = (
        f"Question de l'utilisateur :\n{question}\n\n"
        f"Réponse générée par l'assistant :\n{response_text}\n\n"
        f"Contexte documentaire fourni à l'assistant :\n{context_text}\n\n"
        "Juge la réponse selon les règles du système et retourne le JSON demandé."
    )
    return [
        {"role": "system", "content": _VERIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parse_verification_json(raw: str) -> Dict[str, Any]:
    """Parse la sortie JSON du juge — TRI-ÉTAT (B0, plan 2026-07-29).

    Un parsing en échec ne produit plus un blanc-seing (answers_question=True,
    grounded=True) mais un verdict ``judge_status="unknown"`` : l'appelant sait que le
    juge LLM n'a PAS statué et ne doit ni bloquer ni blanchir sur cette base — seul le
    contrôle programmatique fait alors foi. Les booléens restent à True par compat
    d'affichage (ils ne signifient rien quand judge_status != "ok")."""
    fallback = {
        "answers_question": True,
        "grounded": True,
        "issues": [],
        "parse_error": True,
        "judge_status": "unknown",
    }
    if not raw or not raw.strip():
        return fallback
    content = raw.strip()
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        content = match.group(0)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("[verification] JSON invalide côté juge — verdict unknown")
        return fallback
    if not isinstance(data, dict):
        return fallback
    return {
        "answers_question": bool(data.get("answers_question", True)),
        "grounded": bool(data.get("grounded", True)),
        "issues": [str(i) for i in (data.get("issues") or []) if str(i).strip()],
        "parse_error": False,
        "judge_status": "ok",
    }


async def judge_relevance(
    question: str,
    response_text: str,
    context_text: str,
    *,
    model: str,
) -> Dict[str, Any]:
    """Appel LLM de jugement (température 0.0). N'échoue jamais l'appelant :
    en cas d'erreur API le verdict est ``judge_status="unknown"`` — ni alerte ni
    blanc-seing (tri-état B0)."""
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
        logger.warning("[verification] Échec appel juge LLM (%s) — verdict unknown", exc)
        return {
            "answers_question": True,
            "grounded": True,
            "issues": [],
            "parse_error": True,
            "judge_status": "unknown",
        }


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


async def verify_response(
    *,
    question: str,
    response_text: str,
    context_text: str,
    model: str,
    document_blocks: Optional[List[str]] = None,
    cag_documents: Optional[List[Dict[str, Any]]] = None,
    cited_pages: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, Any]:
    """Orchestre les deux contrôles. Ne lève jamais — un échec de vérification
    ne doit pas faire échouer la persistance de la réponse déjà affichée.

    ``context_text`` reste le contexte COMPLET : le contrôle programmatique le scanne
    intégralement. Le juge LLM, lui, reçoit un contexte assemblé par
    ``build_verification_context`` à partir de ``document_blocks`` — sans le prompt
    système, avec le manifeste de tous les documents packés, et rempli par les preuves.

    Retourne un dict prêt à stocker dans Message.metadata_json["verification"] :
    {
      "ok": bool,                       # False si un problème a été détecté
      "unsupported_claims": [...],      # contrôle 1 (contexte complet)
      "answers_question": bool,         # contrôle 2
      "grounded": bool,                 # contrôle 2
      "issues": [...],                  # contrôle 2
      "context": {...},                 # couverture du contexte montré au juge
      "judge_suspect": bool,            # verdict LLM à prendre avec réserve
      "judge_suspect_reason": str|None,
    }
    """
    unsupported = check_grounding(response_text, context_text)
    # B7a : codes produits de la réponse absents du contexte COMPLET. Programmatique,
    # insensible à toute troncature — quand cette liste est non vide, aucun verdict LLM
    # (ni judge_suspect) ne peut blanchir la réponse.
    unsupported_codes: List[str] = []
    if settings.VERIFY_CODE_GROUNDING:
        unsupported_codes = check_reference_grounding(response_text, context_text)
        for code in unsupported_codes:
            if code not in unsupported:
                unsupported.append(code)

    if document_blocks:
        judge_context, coverage = build_verification_context(
            document_blocks, cag_documents or [], cited_pages=cited_pages
        )
    else:
        # Repli (contexte non CAG) : on borne au même plafond, faute de structure.
        limit = settings.VERIFICATION_CONTEXT_MAX_CHARS
        full = context_text or ""
        judge_context = full if limit <= 0 else full[:limit]
        coverage = {
            "chars_total": len(full),
            "chars_sent": len(judge_context),
            "coverage": round(len(judge_context) / len(full), 3) if full else 1.0,
            "truncated": bool(limit > 0 and len(full) > limit),
            "documents": 0,
            "manifest": False,
        }

    llm_result = await judge_relevance(question, response_text, context_text=judge_context, model=model)

    # Verdict LLM à prendre avec réserve (J6) : un juge qui n'a pas tout vu ne peut pas
    # conclure à une invention, et une contradiction avec le contrôle programmatique —
    # qui, lui, a lu 100 % du contexte — doit être signalée plutôt qu'affichée à égalité.
    # Tri-état : un juge en échec technique (judge_status="unknown") n'est PAS négatif —
    # il n'a simplement pas statué, et seul le programmatique décide.
    judge_status = llm_result.get("judge_status", "ok")
    judge_negative = judge_status == "ok" and not (
        llm_result["answers_question"] and llm_result["grounded"]
    )
    suspect_reason: Optional[str] = None
    if judge_negative and coverage.get("truncated"):
        suspect_reason = (
            f"le juge n'a vu que {int(round(coverage.get('coverage', 0) * 100))} % du contexte"
        )
    elif judge_negative and extract_verifiable_claims(response_text) and not unsupported:
        suspect_reason = (
            "le contrôle programmatique (contexte complet) confirme toutes les valeurs citées"
        )
    judge_suspect = suspect_reason is not None

    if judge_suspect or judge_status != "ok":
        # Le programmatique fait alors seul foi : il a lu l'intégralité du contexte.
        ok = not unsupported
    else:
        ok = not unsupported and llm_result["answers_question"] and llm_result["grounded"]

    result = {
        "ok": ok,
        "unsupported_claims": unsupported,
        "unsupported_codes": unsupported_codes,
        "answers_question": llm_result["answers_question"],
        "grounded": llm_result["grounded"],
        "issues": llm_result["issues"],
        "context": coverage,
        "judge_status": judge_status,
        "judge_suspect": judge_suspect,
        "judge_suspect_reason": suspect_reason,
    }

    if judge_suspect:
        logger.info(
            "[verification] Verdict LLM marqué SUSPECT (%s) — couverture=%.0f%% issues=%s",
            suspect_reason,
            coverage.get("coverage", 0) * 100,
            llm_result["issues"],
        )
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
