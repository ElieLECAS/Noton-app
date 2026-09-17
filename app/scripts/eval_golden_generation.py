"""Runner d'évaluation de la GÉNÉRATION — rejoue un golden set sur le vrai endpoint SSE.

Le golden retrieval (``eval_golden_retrieval.py``) mesure si la bonne PAGE est packée.
Celui-ci mesure l'étage d'après : **la réponse dit-elle la bonne valeur, et cite-t-elle la
page qui la porte ?** C'est la mesure qui manquait pour arbitrer entre la voie image-only
d'avant et le couple texte + image (audit ``docs/audit_rag_generation_2026-09-14.md``, P1).

Pourquoi passer par HTTP et non par des appels de services : le tour de chat vit dans
``stream_space_chat_message`` (compréhension → périmètre → retrieval → élection → pack →
génération → contrôle → sources). Le rejouer en important des morceaux mesurerait une
pipeline qui n'existe pas. On appelle donc l'endpoint réel et on lit le flux SSE, exactement
comme le navigateur.

Ce que le runner mesure, par question :
  * verdict      : juste | ambigu | faux | abstention_ok | abstention_ko | erreur
  * page_citee   : une page de preuve figure dans les sources renvoyées
  * page_vue     : son PNG était joint au contexte de génération
  * latence      : premier token, total
  * contrôle     : verdict du contrôle d'ancrage (passed / flagged)

Usage (dans le conteneur) ::

    docker compose exec web python -m app.scripts.eval_golden_generation
    docker compose exec web python -m app.scripts.eval_golden_generation --only g29_002,g29_013
    docker compose exec web python -m app.scripts.eval_golden_generation --tags parclose --repeat 3
    docker compose exec web python -m app.scripts.eval_golden_generation --label voie-rapide

``--repeat N`` rejoue chaque question N fois : c'est ainsi qu'on mesure la STABILITÉ (le
même modèle sur la même page peut rendre 30, 30, 27, 27, 30 selon la formulation — mesuré
le 13/09). Une question instable est signalée même si sa première réponse est juste.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_GOLDEN = "tests/fixtures/golden/space29_perform76_generation.json"
DEFAULT_BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:8000")
DEFAULT_OUT_DIR = "media/_eval"

VERDICTS = ("juste", "ambigu", "faux", "abstention_ok", "abstention_ko", "erreur")


# ---------------------------------------------------------------------------
# Normalisation et présence d'une valeur — fonctions PURES (testées sans réseau)
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
# Espaces de séparation des milliers : « 3 500 mm » doit matcher « 3500 ».
_THOUSANDS_RE = re.compile(r"(?<=\d)[\s   ](?=\d{3}(?!\d))")


def normalize_text(value: str) -> str:
    """Minuscules, accents retirés, espaces repliés, milliers recollés.

    La normalisation des milliers est indispensable : les documents écrivent « 3 500 mm »
    (espace fine) là où le golden porte « 3500 ». Sans elle, une bonne réponse est comptée
    fausse.
    """
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("«", '"').replace("»", '"')
    text = re.sub(r"[\s   ]+", " ", text)
    text = _THOUSANDS_RE.sub("", text)
    return text.strip().lower()


# Frontières d'un nombre. La ponctuation de fin de phrase ne doit PAS bloquer le match :
# « la référence est 76373. » contient bien 76373. Seule une SUITE décimale l'invalide
# (« 30 » ne doit pas être validé par « 30.5 », ni « 5 » par « 1.5 »).
_NUM_LEFT = r"(?<!\d)(?<![\d][.,])"
_NUM_RIGHT = r"(?!\d)(?![.,]\d)"


def strip_accents(value: str) -> str:
    """Accents et guillemets typographiques retirés, SANS toucher à la casse.

    Utilisé pour les motifs des expressions régulières : les passer par ``normalize_text``
    les mettrait en minuscules et casserait les classes de caractères (``\\S`` deviendrait
    ``\\s``, qui ne matche que des espaces — bug attrapé par le test d'auto-validation
    du golden).
    """
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace("’", "'").replace("«", '"').replace("»", '"')


def value_present(value: str, text: str) -> bool:
    """La valeur figure-t-elle dans le texte, avec des frontières qui excluent les voisins ?

    Deux régimes :
      * NOMBRE (« 16 », « 1.5 », « 76373 ») → frontières numériques : « 16 » ne matche ni
        « 116 » ni « 16.5 », la virgule décimale est acceptée (« 1,5 » ≡ « 1.5 »), et un
        point de fin de phrase ne bloque pas ;
      * MOT ou code alphanumérique (« NT1947 », « post-extrudé ») → frontières
        alphanumériques.
    """
    needle = (value or "").strip()
    if not needle:
        return False
    haystack = normalize_text(text)
    if _NUMERIC_RE.match(needle):
        variants = {needle, needle.replace(",", "."), needle.replace(".", ",")}
        for variant in variants:
            if re.search(_NUM_LEFT + re.escape(variant.lower()) + _NUM_RIGHT, haystack):
                return True
        return False
    norm = normalize_text(needle)
    return re.search(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])", haystack) is not None


def regex_any_matches(patterns: Sequence[str], text: str) -> Optional[str]:
    """Première regex qui matche le texte NORMALISÉ (None si aucune).

    Le texte est mis en minuscules et dépouillé de ses accents ; le motif n'est dépouillé
    que de SES accents (``strip_accents``) et comparé en IGNORECASE — jamais mis en
    minuscules, sous peine de casser ses classes de caractères.
    """
    haystack = normalize_text(text)
    for pattern in patterns or []:
        try:
            if re.search(strip_accents(pattern), haystack, re.IGNORECASE):
                return pattern
        except re.error:
            continue
    return None


def score_answer(attendu: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """Confronte une réponse à l'attendu du golden. Fonction pure, ne lève jamais.

    Retourne {verdict, trouves, manquants, interdits_presents, regex}.
    """
    kind = str(attendu.get("type") or "valeur")
    interdits = [v for v in (attendu.get("interdits") or [])]
    interdits_presents = [v for v in interdits if value_present(v, answer)]
    regex_patterns = attendu.get("regex_any") or []

    if kind == "abstention":
        matched = regex_any_matches(regex_patterns, answer)
        return {
            "verdict": "abstention_ok" if matched else "abstention_ko",
            "trouves": [],
            "manquants": [],
            "interdits_presents": interdits_presents,
            "regex": matched,
        }

    if kind == "texte":
        matched = regex_any_matches(regex_patterns, answer)
        return {
            "verdict": "juste" if matched else "faux",
            "trouves": [],
            "manquants": [],
            "interdits_presents": interdits_presents,
            "regex": matched,
        }

    if kind == "liste":
        elements = attendu.get("elements") or []
        minimum = int(attendu.get("min") or len(elements))
        trouves = [e for e in elements if value_present(e, answer)]
        manquants = [e for e in elements if e not in trouves]
        ok = len(trouves) >= minimum
        return {
            "verdict": "juste" if ok else "faux",
            "trouves": trouves,
            "manquants": manquants,
            "interdits_presents": interdits_presents,
            "regex": None,
        }

    # type "valeur"
    valeurs = attendu.get("valeurs") or []
    mode = str(attendu.get("mode") or "tous")
    trouves = [v for v in valeurs if value_present(v, answer)]
    manquants = [v for v in valeurs if v not in trouves]
    # Sans valeur attendue, rien n'est vérifié : c'est un attendu mal écrit, jamais un
    # succès (« toutes les valeurs de la liste vide sont présentes » serait vrai).
    ok = bool(valeurs) and (len(trouves) == len(valeurs) if mode == "tous" else bool(trouves))
    if ok and regex_patterns:
        # regex_any en complément d'une valeur : condition ET (elle qualifie la valeur,
        # ex. « 4 » n'a de sens qu'accompagné de « 6 pans »).
        ok = regex_any_matches(regex_patterns, answer) is not None
    if not ok:
        verdict = "faux"
    elif interdits_presents:
        # La bonne valeur est là, une valeur piège aussi : à relire à la main plutôt qu'à
        # compter juste (c'est la signature de la confusion de cote sur une planche).
        verdict = "ambigu"
    else:
        verdict = "juste"
    return {
        "verdict": verdict,
        "trouves": trouves,
        "manquants": manquants,
        "interdits_presents": interdits_presents,
        "regex": regex_any_matches(regex_patterns, answer) if regex_patterns else None,
    }


# ---------------------------------------------------------------------------
# Pages : citées (sources) et vues (pack + outils)
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
    """Couples (document_id, page) que l'UI affiche comme sources de la réponse.

    ``used_pages`` (pages déclarées par le modèle) prime ; à défaut, toutes les pages du
    document packé — sinon un tour dont le bloc ``<sources>`` est illisible paraîtrait
    ne citer aucune page.
    """
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
    """Pages présentes dans le MANIFESTE du pack (documents donnés au modèle).

    ``packed_documents`` liste TOUTES les pages retrouvées par la recherche, sans plafond :
    c'est une mesure de rappel du retriever, pas de ce que le modèle a vu. Pour « vu »,
    voir ``pages_seen_as_image``.
    """
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
    """Pages dont le modèle a réellement eu l'IMAGE : les PNG joints au contexte.

    C'est la seule mesure qui sépare une réponse lue d'une réponse devinée : sur ce corpus
    les cotes n'existent que sur le dessin.
    """
    out: List[Tuple[int, int]] = []
    for img in (trace or {}).get("images") or []:
        did, page = img.get("document_id"), img.get("page_no")
        if did is None or page is None:
            continue
        try:
            out.append((int(did), int(page)))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Appel HTTP du tour de chat
# ---------------------------------------------------------------------------


def mint_token(user_id: int) -> Tuple[str, str]:
    """Jeton JWT du même format que ``/api/auth/login``.

    ``sub`` porte l'IDENTIFIANT de l'utilisateur en chaîne : ``get_current_user`` le passe
    par ``int()`` et rejette tout le reste en 401 « Token invalide ».
    """
    from sqlmodel import Session

    from app.database import engine
    from app.models.user import User
    from app.services.auth_service import create_access_token

    with Session(engine) as session:
        user = session.get(User, user_id)
        if user is None:
            raise SystemExit(f"Utilisateur {user_id} introuvable.")
        token = create_access_token(
            data={"sub": str(user.id)}, expires_delta=timedelta(hours=6)
        )
        username = user.username
    return token, username


async def ask(
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
    """Un tour de chat complet. Retourne le texte, les sources, la trace et les temps.

    Sans ``conversation_id`` : aucun historique, aucune persistance — chaque question est
    indépendante, ce qui est la condition pour comparer deux exécutions.
    """
    payload: Dict[str, Any] = {"message": message, "model": model, "provider": "mistral"}
    if scope_choice is not None:
        payload["scope_choice"] = {"values": scope_choice}

    result: Dict[str, Any] = {
        "answer": "",
        "sources": [],
        "trace": {},
        "stages": [],
        "thinking": [],
        "scope_proposal": None,
        "error": None,
        "first_token_s": None,
        "total_s": None,
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
                continue
            if event.get("scope_proposal") or event.get("type") == "scope_proposal":
                result["scope_proposal"] = event.get("scope_proposal") or event
                continue
            if event.get("stage"):
                result["stages"].append(event["stage"].get("label"))
                continue
            if event.get("thinking"):
                result["thinking"].append(event["thinking"])
                continue
            content = (event.get("message") or {}).get("content")
            if content:
                if result["first_token_s"] is None:
                    result["first_token_s"] = time.perf_counter() - t0
                chunks.append(content)
                continue
            if event.get("sources"):
                result["sources"] = event["sources"]
                continue
            if event.get("done"):
                result["trace"] = event.get("trace") or {}
    result["answer"] = "".join(chunks)
    result["total_s"] = time.perf_counter() - t0
    return result


async def run_one(
    client,
    entry: Dict[str, Any],
    *,
    base_url: str,
    space_id: int,
    token: str,
    model: str,
    timeout_s: float,
) -> Dict[str, Any]:
    """Un essai : pose la question, franchit la carte de périmètre si elle apparaît, score."""
    out = await ask(
        client,
        base_url=base_url,
        space_id=space_id,
        token=token,
        message=entry["question"],
        model=model,
        timeout_s=timeout_s,
    )
    scope_asked = out.get("scope_proposal") is not None
    if scope_asked and not out["answer"]:
        # Carte de confirmation du périmètre : on répond « peu importe » (aucun filtre),
        # ce qui est le pire cas pour le retriever — donc la mesure la plus honnête.
        out = await ask(
            client,
            base_url=base_url,
            space_id=space_id,
            token=token,
            message=entry["question"],
            model=model,
            scope_choice={},
            timeout_s=timeout_s,
        )
    answer = out["answer"]
    if out["error"] and not answer:
        score = {"verdict": "erreur", "trouves": [], "manquants": [], "interdits_presents": [], "regex": None}
    else:
        score = score_answer(entry.get("attendu") or {}, answer)

    expected = expected_pairs(entry.get("preuve") or [])
    cited = pages_from_sources(out["sources"])
    packed = pages_packed(out["trace"])
    seen = pages_seen_as_image(out["trace"])
    expected_docs = {d for d, _ in expected}
    cited_docs = {d for d, _ in cited}
    # Le bon document cité mais la mauvaise page : défaut distinct d'une absence de
    # citation — un clic sur la source ouvre alors la mauvaise page. Cas observé : le
    # modèle recopie le numéro IMPRIMÉ sur la planche (page 5) au lieu du numéro de page
    # du PDF (page 8), ce qui décale systématiquement toutes ses citations.
    page_exacte = any(p in cited for p in expected)
    doc_cite = bool(expected_docs & cited_docs)
    loop = (out["trace"] or {}).get("loop") or {}
    verification = (out["trace"] or {}).get("verification") or {}
    retrieval = (out["trace"] or {}).get("retrieval") or {}

    return {
        "id": entry["id"],
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
        "error": out["error"],
    }


# ---------------------------------------------------------------------------
# Agrégation et rapport
# ---------------------------------------------------------------------------


def settings_snapshot() -> Dict[str, Any]:
    """Réglages qui changent le résultat — sans eux, deux rapports ne se comparent pas."""
    from app.config import settings

    keys = [
        "MODEL_FAST", "SPACE_CHAT_TEMPERATURE", "GENERATION_REASONING_EFFORT",
        "RAG_TOP_K", "CAG_MAX_DOCUMENTS", "CAG_IMAGE_DPI", "CAG_MAX_IMAGES",
        "GENERATION_MAX_IMAGES", "RERANKER_ENABLED",
        "COLPALI_GATING_ENABLED", "QUERY_UNDERSTANDING_ENABLED", "SCOPE_MODE",
        "CONVERSATION_ANCHOR_ENABLED", "GENERATION_SEED",
    ]
    return {k: getattr(settings, k, None) for k in keys}


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compte les verdicts, globalement et par difficulté/tag, plus les latences."""
    def counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        out = {v: 0 for v in VERDICTS}
        for r in rows:
            out[r["verdict"]] = out.get(r["verdict"], 0) + 1
        return out

    def rate(rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        good = sum(1 for r in rows if r["verdict"] in ("juste", "abstention_ok"))
        return round(100.0 * good / len(rows), 1)

    by_difficulty = {}
    for level in ("simple", "complexe"):
        rows = [r for r in results if r.get("difficulte") == level]
        if rows:
            by_difficulty[level] = {"n": len(rows), "reussite_pct": rate(rows), **counts(rows)}

    tags = sorted({t for r in results for t in r.get("tags") or []})
    by_tag = {}
    for tag in tags:
        rows = [r for r in results if tag in (r.get("tags") or [])]
        if len(rows) >= 2:
            by_tag[tag] = {"n": len(rows), "reussite_pct": rate(rows), **counts(rows)}

    firsts = [r["first_token_s"] for r in results if r.get("first_token_s")]
    totals = [r["total_s"] for r in results if r.get("total_s")]

    def pct(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(q * len(ordered)))
        return round(ordered[idx], 2)

    return {
        "n": len(results),
        "reussite_pct": rate(results),
        "verdicts": counts(results),
        "page_citee_pct": _pct_of(results, "page_citee"),
        "doc_cite_pct": _pct_of(results, "doc_cite"),
        "page_decalee_pct": _pct_of(results, "page_decalee"),
        "page_vue_pct": _pct_of(results, "page_vue"),
        "page_packee_pct": _pct_of(results, "page_packee"),
        "par_difficulte": by_difficulty,
        "par_tag": by_tag,
        "latence": {
            "premier_token_p50": pct(firsts, 0.5),
            "premier_token_p90": pct(firsts, 0.9),
            "total_p50": pct(totals, 0.5),
            "total_p90": pct(totals, 0.9),
            "total_max": round(max(totals), 2) if totals else None,
        },
        "boucle": {
            "appels_outils_moyen": round(statistics.mean([r["tool_calls"] or 0 for r in results]), 2) if results else 0,
            "tours_avec_outil": sum(1 for r in results if (r.get("tool_calls") or 0) > 0),
            "rounds_de_controle": sum(r.get("control_rounds") or 0 for r in results),
            "degrades": sum(1 for r in results if r.get("degraded")),
            "arrets": _tally(r.get("stopped_by") for r in results),
            "outils": _tally(t for r in results for t in r.get("tool_names") or []),
        },
        "controle": _tally(r.get("verification_action") for r in results),
        "cartes_perimetre": sum(1 for r in results if r.get("scope_card")),
        "erreurs": [r["id"] for r in results if r["verdict"] == "erreur"],
    }


def _pct_of(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(100.0 * sum(1 for r in rows if r.get(key)) / len(rows), 1)


def _tally(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        if v is None:
            continue
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def stability(all_runs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Questions dont le verdict change d'un essai à l'autre.

    C'est la mesure qui manque le plus : une réponse juste une fois sur deux n'est pas une
    réponse juste. Sans elle, un changement de pipeline peut sembler gagner alors qu'il n'a
    fait que retomber du bon côté du hasard.
    """
    unstable = []
    for qid, runs in all_runs.items():
        verdicts = [r["verdict"] for r in runs]
        if len(set(verdicts)) > 1:
            unstable.append(
                {
                    "id": qid,
                    "verdicts": verdicts,
                    "reponses": [r["answer"][:160] for r in runs],
                }
            )
    return {"n_questions_instables": len(unstable), "details": unstable}


def print_report(report: Dict[str, Any]) -> None:
    agg = report["agregat"]
    print()
    print("=" * 78)
    print(f"GOLDEN GÉNÉRATION — espace {report['space_id']} — {report['label']}")
    print("=" * 78)
    s = report["reglages"]
    print(f"modèle={s.get('MODEL_FAST')}  reasoning={s.get('GENERATION_REASONING_EFFORT')}  "
          f"dpi={s.get('CAG_IMAGE_DPI')}  images={s.get('GENERATION_MAX_IMAGES')}  "
          "")
    print(f"questions={agg['n']}  essais/question={report['repeat']}")
    print("-" * 78)
    v = agg["verdicts"]
    print(f"RÉUSSITE            {agg['reussite_pct']:>5.1f} %   "
          f"(juste {v['juste']} · abstention_ok {v['abstention_ok']})")
    print(f"  faux              {v['faux']:>5}")
    print(f"  ambigu            {v['ambigu']:>5}   (bonne valeur + valeur piège : à relire)")
    print(f"  abstention_ko     {v['abstention_ko']:>5}   (a répondu au lieu de s'abstenir)")
    print(f"  erreur            {v['erreur']:>5}")
    print(f"page de preuve VUE en image  {agg['page_vue_pct']:>5.1f} %   "
          f"(présente dans le pack : {agg['page_packee_pct']} %)")
    print(f"citation exacte              {agg['page_citee_pct']:>5.1f} %   "
          f"(bon document : {agg['doc_cite_pct']} % · page décalée : {agg['page_decalee_pct']} %)")
    print("-" * 78)
    for level, data in agg["par_difficulte"].items():
        print(f"{level:<20}{data['reussite_pct']:>5.1f} %   n={data['n']}")
    print("-" * 78)
    lat = agg["latence"]
    print(f"premier token       p50 {lat['premier_token_p50']}s   p90 {lat['premier_token_p90']}s")
    print(f"total               p50 {lat['total_p50']}s   p90 {lat['total_p90']}s   max {lat['total_max']}s")
    boucle = agg["boucle"]
    print(f"outils              {boucle['appels_outils_moyen']} appels/tour   "
          f"{boucle['tours_avec_outil']} tour(s) avec outil   "
          f"{boucle['rounds_de_controle']} round(s) de contrôle   "
          f"{boucle['degrades']} dégradé(s)")
    if boucle["outils"]:
        print(f"  détail outils     {boucle['outils']}")
    if boucle["arrets"]:
        print(f"  motifs d'arrêt    {boucle['arrets']}")
    if agg["controle"]:
        print(f"contrôle de sortie  {agg['controle']}")
    if agg["cartes_perimetre"]:
        print(f"cartes de périmètre {agg['cartes_perimetre']} (relancées sans filtre)")
    print("-" * 78)
    print("PAR THÈME")
    for tag, data in sorted(agg["par_tag"].items(), key=lambda kv: kv[1]["reussite_pct"]):
        print(f"  {tag:<16}{data['reussite_pct']:>5.1f} %   n={data['n']}")
    print("-" * 78)
    print("ÉCHECS")
    for r in report["resultats"]:
        if r["verdict"] in ("juste", "abstention_ok"):
            continue
        detail = ""
        if r["manquants"]:
            detail = f"manque {r['manquants']}"
        if r["interdits_presents"]:
            detail += f"  piège {r['interdits_presents']}"
        print(f"  [{r['verdict']:<14}] {r['id']}  {r['question'][:64]}")
        print(f"     {detail}  | page citée={r['page_citee']} vue={r['page_vue']} "
              f"| outils={r['tool_calls']} arrêt={r['stopped_by']} | {r['total_s']}s")
        print(f"     → {(r['answer'] or r['error'] or '').strip()[:220]}")
    stab = report.get("stabilite") or {}
    if stab.get("n_questions_instables"):
        print("-" * 78)
        print(f"INSTABLES ({stab['n_questions_instables']} question(s) au verdict changeant)")
        for item in stab["details"]:
            print(f"  {item['id']}: {item['verdicts']}")
    print("=" * 78)
    print(f"rapport complet : {report['chemin']}")


# ---------------------------------------------------------------------------
# Entrée
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    import httpx

    golden_path = Path(args.golden)
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = golden["entries"]
    space_id = int(args.space or golden.get("space_id"))

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        entries = [e for e in entries if e["id"] in wanted]
    if args.tags:
        wanted_tags = {s.strip() for s in args.tags.split(",") if s.strip()}
        entries = [e for e in entries if wanted_tags & set(e.get("tags") or [])]
    if args.difficulte:
        entries = [e for e in entries if e.get("difficulte") == args.difficulte]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        print("Aucune question sélectionnée.")
        return 1

    token, username = mint_token(args.user)
    from app.config import settings

    model = args.model or settings.MODEL_FAST
    print(f"[eval] {len(entries)} question(s) × {args.repeat} essai(s) — espace {space_id} "
          f"— utilisateur {username} — modèle {model}")
    print(f"[eval] endpoint {args.base_url}")

    results: List[Dict[str, Any]] = []
    all_runs: Dict[str, List[Dict[str, Any]]] = {}
    limits = httpx.Limits(max_connections=max(1, args.concurrency))
    async with httpx.AsyncClient(limits=limits) as client:
        semaphore = asyncio.Semaphore(max(1, args.concurrency))

        async def guarded(entry: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                for attempt in range(1, args.retries + 2):
                    res = await run_one(
                        client, entry,
                        base_url=args.base_url, space_id=space_id, token=token,
                        model=model, timeout_s=args.timeout,
                    )
                    # Une limite de débit n'est pas un échec de la pipeline : on réessaie.
                    if res["verdict"] != "erreur" or attempt > args.retries:
                        return res
                    wait = 8 * attempt
                    print(f"  [{entry['id']}] erreur ({str(res['error'])[:80]}) — "
                          f"nouvel essai dans {wait}s")
                    await asyncio.sleep(wait)
                return res

        marks = {"juste": "OK ", "abstention_ok": "OK ", "ambigu": "~~ ",
                 "faux": "KO ", "abstention_ko": "KO ", "erreur": "ERR"}
        for run_no in range(1, args.repeat + 1):
            done: List[Dict[str, Any]] = []
            # ``as_completed`` et non ``gather`` : sur une campagne de plusieurs dizaines de
            # minutes, chaque verdict doit s'afficher dès qu'il tombe — sinon rien ne
            # distingue une campagne qui avance d'une campagne bloquée.
            for coro in asyncio.as_completed([guarded(e) for e in entries]):
                res = await coro
                res["essai"] = run_no
                done.append(res)
                all_runs.setdefault(res["id"], []).append(res)
                print(
                    f"  {marks.get(res['verdict'], '?  ')} [{run_no}] {res['id']:<9} "
                    f"{res['verdict']:<14} vue={'oui' if res['page_vue'] else 'non'} "
                    f"cite={'oui' if res['page_citee'] else 'non'} "
                    f"outils={res['tool_calls']} {res['total_s']}s  "
                    f"{(res['answer'] or res['error'] or '')[:66].strip()}",
                    flush=True,
                )
            done.sort(key=lambda r: [e["id"] for e in entries].index(r["id"]))
            if run_no == 1:
                results = list(done)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"generation_{space_id}_{args.label}_{stamp}.json"
    report = {
        "label": args.label,
        "space_id": space_id,
        "golden": str(golden_path),
        "date": stamp,
        "repeat": args.repeat,
        "reglages": settings_snapshot(),
        "agregat": aggregate(results),
        "stabilite": stability(all_runs) if args.repeat > 1 else None,
        "resultats": results,
        "tous_les_essais": all_runs if args.repeat > 1 else None,
        "chemin": str(path),
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Évaluation de la génération sur un golden set.")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--space", type=int, default=None, help="Espace (défaut : celui du golden).")
    parser.add_argument("--user", type=int, default=1, help="Utilisateur dont on emprunte le jeton.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=None, help="Défaut : MODEL_FAST (le modèle de prod).")
    parser.add_argument("--only", default="", help="Identifiants séparés par des virgules.")
    parser.add_argument("--tags", default="", help="Thèmes séparés par des virgules.")
    parser.add_argument("--difficulte", default="", choices=["", "simple", "complexe"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1, help="Essais par question (stabilité).")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1, help="Reprises sur erreur (limite de débit).")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--label", default="courant", help="Nom de la configuration mesurée.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
