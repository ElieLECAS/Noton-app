"""Évaluation de la GÉNÉRATION : poser une question au vrai pipeline, noter la réponse.

Deux appelants, une seule implémentation : le runner en ligne de commande
(``app/scripts/eval_golden_generation.py``) et la page « Qualité RAG » de
l'administration. Ce qu'on mesure doit être identique des deux côtés, sinon les chiffres
d'un rapport et ceux de l'écran ne veulent plus dire la même chose.

Le tour de chat est appelé en HTTP sur l'endpoint réel (``/api/spaces/{id}/chat/stream``),
pas reconstitué à partir des services. Le tour vit dans ``stream_space_chat_message``
(compréhension → périmètre → retrieval → élection → pack de lecture → boucle → contrôle →
sources) ; le rejouer par morceaux mesurerait une pipeline qui n'existe pas.

Le scoring est LITTÉRAL et sans LLM. Un juge qui note une réponse sur une planche cotée
sans voir la planche est aveugle ; une comparaison de valeurs, elle, ne se trompe pas.
Quatre formes d'attendu : valeur (avec valeurs pièges), liste, texte, abstention.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

VERDICTS = ("juste", "ambigu", "faux", "abstention_ok", "abstention_ko", "erreur")

_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
# Séparateur de milliers : « 3 500 mm » dans le document ≡ « 3500 » dans le jeu.
_THOUSANDS_RE = re.compile(r"(?<=\d)[\s   ](?=\d{3}(?!\d))")
# Frontières d'un nombre. La ponctuation de fin de phrase ne bloque PAS le match
# (« la référence est 76373. ») ; seule une suite décimale l'invalide (« 30 » ≠ « 30.5 »).
_NUM_LEFT = r"(?<!\d)(?<![\d][.,])"
_NUM_RIGHT = r"(?!\d)(?![.,]\d)"


def normalize_text(value: str) -> str:
    """Minuscules, accents retirés, espaces repliés, milliers recollés."""
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("«", '"').replace("»", '"')
    text = re.sub(r"[\s   ]+", " ", text)
    return _THOUSANDS_RE.sub("", text).strip().lower()


def strip_accents(value: str) -> str:
    """Accents retirés SANS toucher à la casse — pour les motifs d'expression régulière.

    Les passer par ``normalize_text`` les mettrait en minuscules et casserait leurs classes
    de caractères (``\\S`` deviendrait ``\\s``, qui ne matche que des espaces).
    """
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace("’", "'").replace("«", '"').replace("»", '"')


def value_present(value: str, text: str) -> bool:
    """La valeur figure-t-elle dans le texte, avec des frontières qui excluent les voisins ?"""
    needle = (value or "").strip()
    if not needle:
        return False
    haystack = normalize_text(text)
    if _NUMERIC_RE.match(needle):
        for variant in {needle, needle.replace(",", "."), needle.replace(".", ",")}:
            if re.search(_NUM_LEFT + re.escape(variant.lower()) + _NUM_RIGHT, haystack):
                return True
        return False
    norm = normalize_text(needle)
    return re.search(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])", haystack) is not None


def regex_any_matches(patterns: Sequence[str], text: str) -> Optional[str]:
    """Première expression régulière qui matche le texte normalisé (None si aucune)."""
    haystack = normalize_text(text)
    for pattern in patterns or []:
        try:
            if re.search(strip_accents(pattern), haystack, re.IGNORECASE):
                return pattern
        except re.error:
            continue
    return None


def score_answer(attendu: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """Confronte une réponse à l'attendu. Fonction pure, ne lève jamais."""
    kind = str((attendu or {}).get("type") or "valeur")
    interdits = list((attendu or {}).get("interdits") or [])
    interdits_presents = [v for v in interdits if value_present(v, answer)]
    regex_patterns = (attendu or {}).get("regex_any") or []

    if kind == "abstention":
        matched = regex_any_matches(regex_patterns, answer)
        return {
            "verdict": "abstention_ok" if matched else "abstention_ko",
            "trouves": [], "manquants": [],
            "interdits_presents": interdits_presents, "regex": matched,
        }

    if kind == "texte":
        matched = regex_any_matches(regex_patterns, answer)
        return {
            "verdict": "juste" if matched else "faux",
            "trouves": [], "manquants": [],
            "interdits_presents": interdits_presents, "regex": matched,
        }

    if kind == "liste":
        elements = (attendu or {}).get("elements") or []
        minimum = int((attendu or {}).get("min") or len(elements) or 1)
        trouves = [e for e in elements if value_present(e, answer)]
        return {
            "verdict": "juste" if len(trouves) >= minimum and elements else "faux",
            "trouves": trouves,
            "manquants": [e for e in elements if e not in trouves],
            "interdits_presents": interdits_presents, "regex": None,
        }

    valeurs = (attendu or {}).get("valeurs") or []
    mode = str((attendu or {}).get("mode") or "tous")
    trouves = [v for v in valeurs if value_present(v, answer)]
    manquants = [v for v in valeurs if v not in trouves]
    # Sans valeur attendue, rien n'est vérifié : attendu mal écrit, jamais un succès.
    ok = bool(valeurs) and (len(trouves) == len(valeurs) if mode == "tous" else bool(trouves))
    if ok and regex_patterns:
        # La regex QUALIFIE la valeur (« 4 » n'a de sens qu'avec « 6 pans ») : condition ET.
        ok = regex_any_matches(regex_patterns, answer) is not None
    if not ok:
        verdict = "faux"
    elif interdits_presents:
        # Bonne valeur ET valeur piège : à relire à la main plutôt qu'à compter d'un côté.
        # C'est la signature de la confusion de cote sur une planche.
        verdict = "ambigu"
    else:
        verdict = "juste"
    return {
        "verdict": verdict, "trouves": trouves, "manquants": manquants,
        "interdits_presents": interdits_presents,
        "regex": regex_any_matches(regex_patterns, answer) if regex_patterns else None,
    }


# ---------------------------------------------------------------------------
# Pages : attendues, citées, vues
# ---------------------------------------------------------------------------


def expected_pairs(preuve: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for item in preuve or []:
        did = item.get("document_id")
        for page in item.get("pages") or []:
            try:
                out.append((int(did), int(page)))
            except (TypeError, ValueError):
                continue
    return out


def pages_from_sources(sources: Sequence[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Couples (document, page) que l'interface affiche comme sources de la réponse."""
    out: List[Tuple[int, int]] = []
    for src in sources or []:
        did = src.get("document_id")
        if did is None:
            continue
        pages = src.get("used_pages") or src.get("pages") or []
        if not pages:
            for key in ("page_no", "page_start"):
                if isinstance(src.get(key), int):
                    pages = [src[key]]
                    break
        for page in pages:
            try:
                out.append((int(did), int(page)))
            except (TypeError, ValueError):
                continue
    return out


def pages_packed(trace: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Pages du manifeste du pack — mesure de rappel du retriever, pas de ce qui a été vu."""
    out: List[Tuple[int, int]] = []
    for doc in (trace or {}).get("packed_documents") or []:
        did = doc.get("document_id")
        if did is None:
            continue
        for page in doc.get("pages") or doc.get("seed_pages") or []:
            try:
                out.append((int(did), int(page)))
            except (TypeError, ValueError):
                continue
    return out


def pages_seen_as_image(trace: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Pages dont le modèle a réellement eu l'IMAGE : lectures du pack, PNG joints, outils.

    Seule mesure qui sépare une réponse lue d'une réponse devinée : sur ce corpus, les
    cotes n'existent que sur le dessin.
    """
    out: List[Tuple[int, int]] = []
    for img in (trace or {}).get("images") or []:
        try:
            out.append((int(img["document_id"]), int(img["page_no"])))
        except (KeyError, TypeError, ValueError):
            continue
    for item in ((trace or {}).get("reading") or {}).get("details") or []:
        if item.get("etat") == "non_lue":
            continue
        try:
            out.append((int(item["document_id"]), int(item["page_no"])))
        except (KeyError, TypeError, ValueError):
            continue
    for rnd in ((trace or {}).get("loop") or {}).get("rounds") or []:
        for call in rnd.get("calls") or []:
            for pair in call.get("pages_read") or []:
                try:
                    out.append((int(pair[0]), int(pair[1])))
                except (TypeError, ValueError, IndexError):
                    continue
    return out


# ---------------------------------------------------------------------------
# Un tour de chat réel
# ---------------------------------------------------------------------------


async def ask_chat(
    client,
    *,
    base_url: str,
    space_id: int,
    token: str,
    message: str,
    model: str,
    scope_choice: Optional[Dict[str, str]] = None,
    timeout_s: float = 240.0,
) -> Dict[str, Any]:
    """Un tour complet sur l'endpoint SSE réel. Sans ``conversation_id`` : aucun historique,
    aucune persistance — chaque question est indépendante, condition pour comparer deux
    exécutions."""
    payload: Dict[str, Any] = {"message": message, "model": model, "provider": "mistral"}
    if scope_choice is not None:
        payload["scope_choice"] = {"values": scope_choice}

    result: Dict[str, Any] = {
        "answer": "", "sources": [], "trace": {}, "stages": [], "thinking": [],
        "scope_proposal": None, "error": None, "first_token_s": None, "total_s": None,
    }
    chunks: List[str] = []
    t0 = time.perf_counter()
    async with client.stream(
        "POST",
        f"{base_url}/api/spaces/{space_id}/chat/stream",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout_s,
    ) as response:
        if response.status_code != 200:
            body = (await response.aread()).decode(errors="replace")
            result["error"] = f"HTTP {response.status_code}: {body[:300]}"
            result["total_s"] = time.perf_counter() - t0
            return result
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            raw = line.split("data:", 1)[1].strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                result["error"] = str(event["error"])[:400]
            elif event.get("scope_proposal"):
                result["scope_proposal"] = event["scope_proposal"]
            elif event.get("stage"):
                result["stages"].append(event["stage"].get("label"))
            elif event.get("thinking"):
                result["thinking"].append(event["thinking"])
            elif (event.get("message") or {}).get("content"):
                if result["first_token_s"] is None:
                    result["first_token_s"] = time.perf_counter() - t0
                chunks.append(event["message"]["content"])
            elif event.get("sources"):
                result["sources"] = event["sources"]
            elif event.get("done"):
                result["trace"] = event.get("trace") or {}
    result["answer"] = "".join(chunks)
    result["total_s"] = time.perf_counter() - t0
    return result


async def run_question(
    client,
    entry: Dict[str, Any],
    *,
    base_url: str,
    space_id: int,
    token: str,
    model: str,
    timeout_s: float = 240.0,
) -> Dict[str, Any]:
    """Pose la question, franchit la carte de périmètre si elle s'affiche, note la réponse.

    ``entry`` accepte les deux formes : ``preuve`` (golden brut) ou ``pages_attendues``
    portant un ``document_id`` (golden normalisé pour l'administration).
    """
    out = await ask_chat(
        client, base_url=base_url, space_id=space_id, token=token,
        message=entry["question"], model=model, timeout_s=timeout_s,
    )
    scope_asked = out.get("scope_proposal") is not None
    if scope_asked and not out["answer"]:
        # Carte de confirmation du périmètre : on répond « peu importe » (aucun filtre),
        # le pire cas pour le retriever, donc la mesure la plus honnête.
        out = await ask_chat(
            client, base_url=base_url, space_id=space_id, token=token,
            message=entry["question"], model=model, scope_choice={}, timeout_s=timeout_s,
        )

    answer = out["answer"]
    if out["error"] and not answer:
        score = {"verdict": "erreur", "trouves": [], "manquants": [],
                 "interdits_presents": [], "regex": None}
    else:
        score = score_answer(entry.get("attendu") or {}, answer)

    preuve = entry.get("preuve") or entry.get("pages_attendues") or []
    expected = expected_pairs(preuve)
    cited = pages_from_sources(out["sources"])
    packed = pages_packed(out["trace"])
    seen = pages_seen_as_image(out["trace"])
    page_exacte = any(p in cited for p in expected)
    doc_cite = bool({d for d, _ in expected} & {d for d, _ in cited})
    loop = (out["trace"] or {}).get("loop") or {}
    verification = (out["trace"] or {}).get("verification") or {}
    retrieval = (out["trace"] or {}).get("retrieval") or {}
    reading = (out["trace"] or {}).get("reading") or {}

    return {
        "id": entry.get("id"),
        "question": entry["question"],
        "difficulte": entry.get("difficulte"),
        "tags": entry.get("tags") or [],
        "verdict": score["verdict"],
        "trouves": score["trouves"],
        "manquants": score["manquants"],
        "interdits_presents": score["interdits_presents"],
        "regex": score["regex"],
        "attendu": entry.get("attendu"),
        "verite": entry.get("verite"),
        "answer": answer,
        "answer_chars": len(answer),
        "page_citee": page_exacte,
        "doc_cite": doc_cite,
        "page_decalee": doc_cite and not page_exacte,
        "page_vue": any(p in seen for p in expected),
        "page_packee": any(p in packed for p in expected),
        "pages_attendues": [list(p) for p in expected],
        "pages_citees": [list(p) for p in sorted(set(cited))],
        "pages_vues": [list(p) for p in sorted(set(seen))],
        "pages_packees": [list(p) for p in sorted(set(packed))],
        "scope_card": scope_asked,
        "first_token_s": round(out["first_token_s"], 2) if out["first_token_s"] else None,
        "total_s": round(out["total_s"], 2) if out["total_s"] else None,
        "tool_calls": loop.get("tool_calls"),
        "tool_names": [
            call.get("tool")
            for rnd in (loop.get("rounds") or [])
            for call in (rnd.get("calls") or [])
        ],
        "rounds": len(loop.get("rounds") or []),
        "control_rounds": loop.get("control_rounds"),
        "stopped_by": loop.get("stopped_by"),
        "degraded": loop.get("degraded"),
        "verification_action": verification.get("action"),
        "unsupported_claims": verification.get("unsupported_claims") or [],
        "retrieval_status": retrieval.get("status"),
        "retrieval_retry": bool((retrieval.get("retry") or {}).get("triggered")),
        "nb_passages": retrieval.get("nb_passages"),
        "pages_lues": reading.get("pages_read"),
        "pages_repondent": reading.get("pages_answered"),
        "lecture_ms": reading.get("wall_ms"),
        "error": out["error"],
    }
