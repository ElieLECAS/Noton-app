"""Juge de suffisance pré-génération (E4) et actions de relance (E5).

Plan boucle agentique 2026-07-29 (docs/plan_boucle_agentique_2026-07-29.md, lots B3-B5).

Le principe : « le chunk trouve, la page juge, la fenêtre génère ». Avant toute
génération, un modèle DISTINCT du générateur (``MODEL_JUDGE``) lit des DOSSIERS
CANDIDATS — les passages regroupés par document, pages voisines incluses, métadonnées en
tête — et répond à trois questions : bon document ? bon TYPE de contenu pour l'intention ?
l'information demandée littéralement présente ? Son verdict est structuré et borné à un
espace d'actions fermé : élire (documents + pages) ou relancer (reformulation, filtres).

Anti-complaisance MÉCANIQUE, pas seulement prompt (leçon du juge auto-complaisant,
audit 2026-07-28) : un verdict « sufficient » n'est actionnable que si sa citation
``evidence`` est retrouvée LITTÉRALEMENT dans le pack — vérification par code,
insensible au modèle. Symétriquement, un échec d'infra (timeout, JSON invalide) n'est
pas un verdict : tri-état ``status="unknown"`` → l'orchestrateur génère comme aujourd'hui.

Fonctions pures (parse/normalisation/actions) séparées de l'appel LLM → testables sans DB.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional

from sqlmodel import Session

from app.config import settings

logger = logging.getLogger(__name__)

# Longueur minimale (caractères normalisés) d'une citation pour valoir preuve : en
# dessous, n'importe quel fragment (« TGY », « 3 points ») serait retrouvé par hasard.
_EVIDENCE_MIN_CHARS = 12
# Longueur minimale d'un segment de citation (découpe sur les ellipses du juge).
_EVIDENCE_SEGMENT_MIN_CHARS = 20

JUDGE_SYSTEM_PROMPT = """Tu es un CONTRÔLEUR DOCUMENTAIRE pour un fabricant de menuiseries (fenêtres, portes, coulissants, volets roulants).
Ta mission : décider si les documents candidats permettent de répondre LITTÉRALEMENT à la question — AVANT toute génération. Ta posture par défaut est le REFUS : en cas de doute, verdict "insufficient".

Tu reçois : la QUESTION, l'INTENTION, le TYPE DE CONTENU ATTENDU, l'état des RÉFÉRENCES demandées, et des DOSSIERS CANDIDATS (un par document : en-tête métadonnées puis extraits de pages ; « ★ » signale une page retrouvée par la recherche ; un manifeste liste TOUS les documents fournis).

RÈGLES DURES :
1. "sufficient" UNIQUEMENT si tu peux citer dans "evidence" une phrase EXACTE, copiée mot pour mot d'un dossier, qui porte l'information demandée. Pas de paraphrase, pas de reformulation.
2. Une page qui MENTIONNE une référence sans donner l'information demandée ne suffit pas : un tableau de composition ou une nomenclature ne répond PAS à une question de pose/montage ; une notice de pose ne répond PAS à une question de choix de référence.
3. Ne déduis JAMAIS une référence ou une valeur par analogie de numérotation (TGY3702 ≠ TGY3710) ni depuis une gamme voisine.
4. RÈGLE D'ABSENCE : les extraits sont parfois partiels — un document listé au manifeste fait partie du contexte même si son texte n'est pas montré ci-dessous. Ne conclus jamais qu'un document ne contient pas X au seul motif que son extrait ne le montre pas ; propose plutôt une action de recherche ciblée sur ce document.
5. Si "insufficient" : dis ce qui manque dans "missing" et propose UNE action dans "next_action", la plus spécifique possible :
   - "rewritten_query" : reformulation courte et précise en vocabulaire métier (pose, montage, réglage, rallonge, crémone, nomenclature...), DIFFÉRENTE de la question actuelle ;
   - "restrict_to_document_index" : chercher dans CE document candidat (bon document identifié, mauvaises pages montrées) ;
   - "widen_scope" : true si le périmètre de recherche actuel semble trop étroit ;
   - "drop_anchor" : true si les candidats semblent hérités du tour précédent et inadaptés au TYPE d'information demandé par CETTE question ;
   - "raise_k" : true pour ramener plus de passages.

RETOURNE UNIQUEMENT un objet JSON valide, sans texte avant ni après :
{
  "verdict": "sufficient" | "insufficient",
  "confidence": 0.0,
  "evidence": "citation exacte (≤ 300 caractères) si sufficient, sinon \\"\\"",
  "elected": [{"document_index": 1, "pages": [110, 111], "role": "steps"}],
  "missing": "ce qui manque, en clair, si insufficient",
  "next_action": {"rewritten_query": "", "widen_scope": false, "drop_anchor": false, "restrict_to_document_index": null, "raise_k": false}
}
"role" ∈ steps | reference | support. "elected" liste les documents à donner au générateur (le meilleur d'abord), même quand une seule page suffit."""


# ---------------------------------------------------------------------------
# Pack-juge (B3) — dossiers candidats
# ---------------------------------------------------------------------------


def build_judge_pack(
    session: Session,
    passages: List[Dict[str, Any]],
    *,
    intent: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble les DOSSIERS CANDIDATS montrés au juge.

    Deux packs par tour (décision d'archi n°3 du plan) : le pack-juge est LARGE et peu
    profond — jusqu'à ``JUDGE_MAX_CANDIDATE_DOCS`` documents, extraits remplis par les
    preuves sous ``JUDGE_CONTEXT_MAX_CHARS`` — là où le pack-génération est étroit et
    profond. Réutilise le packer CAG (cache feuilles TTL → 2e appel quasi gratuit) puis
    l'assembleur evidence-first du juge de vérification (manifeste complet inclus).
    """
    from app.services.context_packer_service import build_cag_context
    from app.services.response_verification_service import build_verification_context

    pack = build_cag_context(
        session,
        passages,
        system_prompt="",
        token_budget=settings.CAG_TOKEN_BUDGET,
        max_documents=max(1, settings.JUDGE_MAX_CANDIDATE_DOCS),
        intent=intent,
        emit_sources_tag=False,
    )
    cag_documents = list(pack.get("cag_documents") or [])
    context_text, coverage = build_verification_context(
        list(pack.get("cag_document_blocks") or []),
        cag_documents,
        max_chars=settings.JUDGE_CONTEXT_MAX_CHARS,
    )
    return {
        "context_text": context_text,
        "coverage": coverage,
        "cag_documents": cag_documents,
    }


def build_judge_messages(
    *,
    question: str,
    intent: Optional[str],
    expected_content: str,
    judge_context: str,
    coverage_line: Optional[str] = None,
    round_index: int = 1,
    previous_missing: Optional[str] = None,
    images: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Messages de l'appel juge. Les images de pages (PNG b64) partent dans le message
    user via la clé ``images`` — convertie par ``mistral_service._clean_messages``."""
    parts = [
        f"QUESTION : {question}",
        f"INTENTION : {intent or 'inconnue'}",
        f"TYPE DE CONTENU ATTENDU : {expected_content}",
    ]
    if coverage_line:
        parts.append(f"RÉFÉRENCES DEMANDÉES : {coverage_line}")
    if round_index > 1 and previous_missing:
        parts.append(
            f"RELANCE n°{round_index - 1} — au tour précédent tu avais jugé insuffisant "
            f"(manquait : {previous_missing}). Les dossiers ci-dessous viennent de la "
            "NOUVELLE recherche."
        )
    parts.append(f"DOSSIERS CANDIDATS :\n{judge_context}")
    parts.append("Rends ton verdict JSON.")
    user_msg: Dict[str, Any] = {"role": "user", "content": "\n\n".join(parts)}
    if images:
        user_msg["images"] = list(images)
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, user_msg]


# ---------------------------------------------------------------------------
# Parsing + contrôles mécaniques du verdict (purs)
# ---------------------------------------------------------------------------


def parse_judge_json(raw: str) -> Optional[Dict[str, Any]]:
    """Parse la sortie du juge ; None si inexploitable (→ verdict unknown)."""
    if not raw or not raw.strip():
        return None
    content = raw.strip()
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        content = match.group(0)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _normalize_for_evidence(value: str) -> str:
    """Normalisation tolérante pour la preuve : accents retirés, espaces repliés,
    guillemets/puces neutralisés, casse ignorée."""
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("’", "'").replace("«", '"').replace("»", '"')
    value = re.sub(r"[\s ]+", " ", value)
    return value.strip().lower()


_ELLIPSIS_SPLIT = re.compile(r"…|\.\.\.|\[\.\.\.\]|\[…\]")


def evidence_in_pack(evidence: str, pack_text: str) -> bool:
    """La citation du juge est-elle LITTÉRALEMENT dans le pack qu'il a lu ?

    C'est le verrou anti-complaisance : un « sufficient » dont la preuve est introuvable
    est dégradé (cf. ``normalize_verdict``). Tolérances : accents/casse/espaces, et
    découpe sur les ellipses (« … ») — chaque segment substantiel doit être retrouvé."""
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


def _coerce_pages(raw: Any) -> List[int]:
    pages: List[int] = []
    for p in raw or []:
        try:
            val = int(p)
        except (TypeError, ValueError):
            continue
        if val > 0 and val not in pages:
            pages.append(val)
    return pages


def normalize_verdict(
    parsed: Optional[Dict[str, Any]],
    cag_documents: List[Dict[str, Any]],
    pack_text: str,
    *,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Verdict final : parsing validé + contrôles mécaniques.

    Tri-état ``status`` :
      - "ok"      → verdict exploitable ("sufficient" avec preuve vérifiée, ou "insufficient") ;
      - "unknown" → le juge n'a pas statué de façon fiable (parse, verdict invalide,
        confiance sous le seuil, ou « sufficient » sans preuve retrouvable). L'orchestrateur
        génère alors comme aujourd'hui — un juge défaillant ne dégrade jamais le tour.
    """
    min_confidence = (
        settings.JUDGE_MIN_CONFIDENCE if min_confidence is None else min_confidence
    )
    result: Dict[str, Any] = {
        "status": "unknown",
        "status_reason": None,
        "verdict": None,
        "confidence": 0.0,
        "evidence": "",
        "evidence_verified": False,
        "missing": "",
        "elected": [],
        "next_action": None,
    }
    if parsed is None:
        result["status_reason"] = "parse_error"
        return result

    verdict = str(parsed.get("verdict") or "").strip().lower()
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    result["confidence"] = confidence
    result["evidence"] = str(parsed.get("evidence") or "").strip()
    result["missing"] = str(parsed.get("missing") or "").strip()

    # Élection résolue via les dossiers candidats (document_index → document_id) : un
    # index inconnu est écarté — le juge ne peut pas élire un document qu'on ne lui a
    # pas montré.
    doc_by_index = {int(d.get("index")): d for d in cag_documents if d.get("index") is not None}
    elected: List[Dict[str, Any]] = []
    for entry in parsed.get("elected") or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("document_index"))
        except (TypeError, ValueError):
            continue
        doc = doc_by_index.get(idx)
        if not doc:
            continue
        elected.append(
            {
                "document_index": idx,
                "document_id": int(doc.get("document_id")),
                "document_title": doc.get("document_title"),
                "pages": _coerce_pages(entry.get("pages")),
                "role": str(entry.get("role") or "support").strip().lower(),
            }
        )
    result["elected"] = elected

    action_raw = parsed.get("next_action") or {}
    if isinstance(action_raw, dict):
        restrict_idx = action_raw.get("restrict_to_document_index")
        try:
            restrict_idx = int(restrict_idx) if restrict_idx is not None else None
        except (TypeError, ValueError):
            restrict_idx = None
        restrict_doc = doc_by_index.get(restrict_idx) if restrict_idx is not None else None
        result["next_action"] = {
            "rewritten_query": str(action_raw.get("rewritten_query") or "").strip(),
            "widen_scope": bool(action_raw.get("widen_scope")),
            "drop_anchor": bool(action_raw.get("drop_anchor")),
            "restrict_to_document_index": restrict_idx if restrict_doc else None,
            "restrict_to_document_id": (
                int(restrict_doc.get("document_id")) if restrict_doc else None
            ),
            "raise_k": bool(action_raw.get("raise_k")),
        }

    if verdict not in ("sufficient", "insufficient"):
        result["status_reason"] = "invalid_verdict"
        return result
    result["verdict"] = verdict

    if confidence < min_confidence:
        result["status_reason"] = "low_confidence"
        return result

    if verdict == "sufficient":
        result["evidence_verified"] = evidence_in_pack(result["evidence"], pack_text)
        if not result["evidence_verified"]:
            # Anti-complaisance : un positif sans preuve retrouvable n'est NI une élection
            # NI un motif de relance — il vaut « pas de juge » et se voit dans la trace.
            result["status_reason"] = "evidence_not_found"
            return result

    result["status"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Appel LLM (E4)
# ---------------------------------------------------------------------------


async def judge_candidates(
    *,
    question: str,
    intent: Optional[str],
    expected_content: str,
    judge_pack: Dict[str, Any],
    coverage_line: Optional[str] = None,
    images: Optional[List[str]] = None,
    round_index: int = 1,
    previous_missing: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Appel du juge (modèle ``MODEL_JUDGE``, température 0.0, JSON strict, timeout).

    N'échoue jamais l'appelant : toute erreur d'infra produit ``status="unknown"``."""
    from app.services.mistral_service import chat

    model = model or settings.MODEL_JUDGE
    timeout_s = timeout_s if timeout_s is not None else settings.JUDGE_TIMEOUT_S
    messages = build_judge_messages(
        question=question,
        intent=intent,
        expected_content=expected_content,
        judge_context=judge_pack.get("context_text") or "",
        coverage_line=coverage_line,
        round_index=round_index,
        previous_missing=previous_missing,
        images=images,
    )
    started = time.perf_counter()
    parsed: Optional[Dict[str, Any]] = None
    infra_reason: Optional[str] = None
    try:
        call = chat(
            "",
            model=model,
            context=messages,
            temperature=0.0,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        if timeout_s and timeout_s > 0:
            response = await asyncio.wait_for(call, timeout=timeout_s)
        else:
            response = await call
        raw = (response.get("choices") or [{}])[0].get("message", {}).get("content", "")
        parsed = parse_judge_json(raw)
        if parsed is None:
            infra_reason = "parse_error"
    except asyncio.TimeoutError:
        infra_reason = "timeout"
        logger.warning("[judge] timeout après %.0fs — verdict unknown", timeout_s)
    except Exception as exc:  # noqa: BLE001
        infra_reason = "api_error"
        logger.warning("[judge] échec appel juge (%s) — verdict unknown", exc)

    verdict = normalize_verdict(
        parsed,
        judge_pack.get("cag_documents") or [],
        judge_pack.get("context_text") or "",
    )
    # La cause d'INFRA prime sur l'artefact de normalisation : un timeout ou une erreur
    # API produit parsed=None, que normalize_verdict étiquette « parse_error » faute de
    # mieux — la vraie cause est celle de l'exception.
    if verdict["status"] == "unknown" and infra_reason:
        verdict["status_reason"] = infra_reason
    verdict["model"] = model
    verdict["duration_ms"] = int((time.perf_counter() - started) * 1000)
    logger.info(
        "[judge] round=%d status=%s verdict=%s confiance=%.2f preuve_vérifiée=%s élus=%s (%d ms)",
        round_index,
        verdict["status"],
        verdict["verdict"],
        verdict["confidence"],
        verdict["evidence_verified"],
        [e["document_id"] for e in verdict["elected"]],
        verdict["duration_ms"],
    )
    return verdict


# ---------------------------------------------------------------------------
# Actions de relance (E5) + aides d'orchestration (pures)
# ---------------------------------------------------------------------------


def _normalize_query(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def apply_judge_action(
    verdict: Dict[str, Any],
    *,
    current_query: str,
) -> Optional[Dict[str, Any]]:
    """Traduit le ``next_action`` du juge en paramètres de recherche.

    Retourne None quand rien d'exploitable ne CHANGE (même requête, aucun levier) :
    relancer la même recherche redonnerait les mêmes nomenclatures en tête — le biais de
    densité de référence est déterministe. L'appelant traite None comme « pas de progrès »."""
    action = verdict.get("next_action") or {}
    if not isinstance(action, dict):
        return None

    rewritten = (action.get("rewritten_query") or "").strip()
    if rewritten and _normalize_query(rewritten) == _normalize_query(current_query):
        rewritten = ""

    resolved = {
        "query_text": rewritten or None,
        "widen_scope": bool(action.get("widen_scope")),
        "drop_anchor": bool(action.get("drop_anchor")),
        "restrict_document_id": action.get("restrict_to_document_id"),
        "raise_k": bool(action.get("raise_k")),
    }
    if not any(
        (
            resolved["query_text"],
            resolved["widen_scope"],
            resolved["drop_anchor"],
            resolved["restrict_document_id"],
            resolved["raise_k"],
        )
    ):
        return None

    bits = []
    if resolved["query_text"]:
        bits.append(f"requête → « {resolved['query_text']} »")
    if resolved["restrict_document_id"]:
        bits.append(f"restreint au document {resolved['restrict_document_id']}")
    if resolved["widen_scope"]:
        bits.append("périmètre élargi")
    if resolved["drop_anchor"]:
        bits.append("ancre abandonnée")
    if resolved["raise_k"]:
        bits.append("k augmenté")
    resolved["label"] = ", ".join(bits)
    return resolved


def passages_pool_key(passages: List[Dict[str, Any]]) -> frozenset:
    """Empreinte du pool (doc, page ancre) : deux relances qui ramènent le même pool ne
    progressent pas — la boucle s'arrête au lieu de payer un juge pour rien."""
    return frozenset(
        (p.get("document_id"), p.get("page_no") or p.get("page_start"))
        for p in passages or []
        if p.get("document_id") is not None
    )


def build_judge_note_block(verdict: Dict[str, Any]) -> Optional[str]:
    """Bloc « NOTE DE RECHERCHE » injecté dans le contexte de génération (B6) : le
    générateur sait OÙ le contrôle documentaire a validé la réponse."""
    elected = verdict.get("elected") or []
    if not elected:
        return None
    lines = ["### NOTE DE RECHERCHE (contrôle documentaire, fait foi)"]
    role_labels = {
        "steps": "étapes/procédure",
        "reference": "références",
        "support": "complément",
    }
    for e in elected:
        pages = e.get("pages") or []
        pages_txt = (
            " — pages " + ", ".join(str(p) for p in pages) if pages else ""
        )
        lines.append(
            f"- Document {e.get('document_index')} « {e.get('document_title') or '?'} »"
            f" ({role_labels.get(e.get('role'), e.get('role') or 'complément')}){pages_txt}"
        )
    evidence = (verdict.get("evidence") or "").strip()
    if evidence:
        lines.append(f'- Preuve vérifiée : « {evidence[:300]} »')
    lines.append(
        "Réponds d'abord depuis ces documents et pages. Ne complète jamais par une "
        "référence ou une valeur absente des documents fournis."
    )
    return "\n".join(lines)
