"""Runner d'évaluation de la GÉNÉRATION — rejoue un golden set sur le vrai chat (wiki, CAG).

Mesure, par question : **la réponse dit-elle la bonne valeur, et cite-t-elle une page du wiki
tirée du document de preuve ?** On appelle l'endpoint réel ``POST /api/chat/stream`` et on lit
le flux SSE exactement comme le navigateur — rejouer des morceaux de service mesurerait une
pipeline qui n'existe pas.

Le golden (``tests/fixtures/golden/space29_perform76_generation.json``, 62 questions) date du
système précédent : ses preuves sont des couples (document, page PDF). Le wiki n'a plus de
pages PDF, il a des pages markdown dont le frontmatter ``sources`` cite le PDF d'origine. La
citation est donc comptée exacte quand une page citée par le modèle est issue du document de
preuve (correspondance ``DOCUMENTS`` ci-dessous → fichier de ``raw/``).

Ce que le runner mesure :
  * verdict      : juste | ambigu | faux | abstention_ok | abstention_ko | erreur
  * doc_cite     : une page citée provient du document de preuve
  * citation_inconnue : le modèle a cité une page qui n'existe pas dans le wiki
  * latence      : premier token, total ; tokens et part en cache du prompt

Usage (dans le conteneur) ::

    docker compose exec web python -m app.scripts.eval_golden_generation
    docker compose exec web python -m app.scripts.eval_golden_generation --only g29_002,g29_013
    docker compose exec web python -m app.scripts.eval_golden_generation --tags parclose --repeat 3
    docker compose exec web python -m app.scripts.eval_golden_generation --label wiki-cag

``--repeat N`` rejoue chaque question N fois : c'est ainsi qu'on mesure la STABILITÉ — une
réponse juste une fois sur deux n'est pas une réponse juste.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_GOLDEN = "tests/fixtures/golden/space29_perform76_generation.json"
DEFAULT_BASE_URL = os.getenv("EVAL_BASE_URL", "http://localhost:8000")
DEFAULT_OUT_DIR = "logs/eval"

VERDICTS = ("juste", "ambigu", "faux", "abstention_ok", "abstention_ko", "erreur")

# Identifiants de documents du golden → fichier de raw/ (le frontmatter des pages du wiki
# cite ``raw/<fichier>``). Le catalogue général du golden est l'édition 2024 ; le wiki porte
# l'édition 2026 du même document : c'est elle qui est acceptée.
DOCUMENTS: Dict[int, str] = {
    438: "cahier-technique-perform76-2026-09-02-cc03.pdf",
    424: "catalogue-general-2026-01.pdf",
    427: "depliant-innoslide-2024-01.pdf",
    433: "depliant-innoslide-2024-01-a4-web.pdf",
}


# ---------------------------------------------------------------------------
# Normalisation et présence d'une valeur — fonctions PURES (testées sans réseau)
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
# Espaces de séparation des milliers : « 3 500 mm » doit matcher « 3500 ».
_THOUSANDS_RE = re.compile(r"(?<=\d)[\s   ](?=\d{3}(?!\d))")


def normalize_text(value: str) -> str:
    """Minuscules, accents retirés, espaces repliés, milliers recollés."""
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("«", '"').replace("»", '"')
    text = re.sub(r"[\s   ]+", " ", text)
    text = _THOUSANDS_RE.sub("", text)
    return text.strip().lower()


_NUM_LEFT = r"(?<!\d)(?<![\d][.,])"
_NUM_RIGHT = r"(?!\d)(?![.,]\d)"


def strip_accents(value: str) -> str:
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
        variants = {needle, needle.replace(",", "."), needle.replace(".", ",")}
        for variant in variants:
            if re.search(_NUM_LEFT + re.escape(variant.lower()) + _NUM_RIGHT, haystack):
                return True
        return False
    norm = normalize_text(needle)
    return re.search(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])", haystack) is not None


def regex_any_matches(patterns: Sequence[str], text: str) -> Optional[str]:
    haystack = normalize_text(text)
    for pattern in patterns or []:
        try:
            if re.search(strip_accents(pattern), haystack, re.IGNORECASE):
                return pattern
        except re.error:
            continue
    return None


def score_answer(attendu: Dict[str, Any], answer: str) -> Dict[str, Any]:
    """Confronte une réponse à l'attendu du golden. Fonction pure, ne lève jamais."""
    kind = str(attendu.get("type") or "valeur")
    interdits = [v for v in (attendu.get("interdits") or [])]
    interdits_presents = [v for v in interdits if value_present(v, answer)]
    regex_patterns = attendu.get("regex_any") or []

    if kind == "abstention":
        matched = regex_any_matches(regex_patterns, answer)
        return {"verdict": "abstention_ok" if matched else "abstention_ko", "trouves": [],
                "manquants": [], "interdits_presents": interdits_presents, "regex": matched}
    if kind == "texte":
        matched = regex_any_matches(regex_patterns, answer)
        return {"verdict": "juste" if matched else "faux", "trouves": [], "manquants": [],
                "interdits_presents": interdits_presents, "regex": matched}
    if kind == "liste":
        elements = attendu.get("elements") or []
        minimum = int(attendu.get("min") or len(elements))
        trouves = [e for e in elements if value_present(e, answer)]
        manquants = [e for e in elements if e not in trouves]
        return {"verdict": "juste" if len(trouves) >= minimum else "faux", "trouves": trouves,
                "manquants": manquants, "interdits_presents": interdits_presents, "regex": None}

    valeurs = attendu.get("valeurs") or []
    mode = str(attendu.get("mode") or "tous")
    trouves = [v for v in valeurs if value_present(v, answer)]
    manquants = [v for v in valeurs if v not in trouves]
    ok = bool(valeurs) and (len(trouves) == len(valeurs) if mode == "tous" else bool(trouves))
    if ok and regex_patterns:
        ok = regex_any_matches(regex_patterns, answer) is not None
    if not ok:
        verdict = "faux"
    elif interdits_presents:
        verdict = "ambigu"
    else:
        verdict = "juste"
    return {"verdict": verdict, "trouves": trouves, "manquants": manquants,
            "interdits_presents": interdits_presents,
            "regex": regex_any_matches(regex_patterns, answer) if regex_patterns else None}


# ---------------------------------------------------------------------------
# Pages du wiki ↔ documents de preuve
# ---------------------------------------------------------------------------


def wiki_pages_by_raw_file() -> Dict[str, List[str]]:
    """Pour chaque PDF de raw/, les pages du wiki dont le frontmatter le cite."""
    from app.services.wiki_service import get_snapshot

    index: Dict[str, List[str]] = {}
    for page in get_snapshot().concept_pages:
        for source in page.sources:
            resource = str(source.get("resource") or "")
            if resource.startswith("raw/"):
                index.setdefault(resource[4:], []).append(page.id)
    return index


def expected_docs(preuve: Sequence[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for item in preuve or []:
        try:
            out.append(int(item.get("document_id")))
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Appel HTTP du tour de chat
# ---------------------------------------------------------------------------


def mint_token(user_id: int):
    """Jeton JWT du même format que ``/api/auth/login``."""
    from sqlmodel import Session

    from app.database import engine
    from app.models.user import User
    from app.services.auth_service import create_access_token

    with Session(engine) as session:
        user = session.get(User, user_id)
        if user is None:
            raise SystemExit(f"Utilisateur {user_id} introuvable.")
        token = create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(hours=6))
        return token, user.username


async def ask(client, *, base_url: str, token: str, message: str, timeout_s: float) -> Dict[str, Any]:
    """Un tour de chat complet dans une conversation NEUVE (aucun historique), supprimée après.

    Chaque question est indépendante : c'est la condition pour comparer deux exécutions."""
    headers = {"Authorization": f"Bearer {token}"}
    result: Dict[str, Any] = {"answer": "", "sources": [], "anomalies": [], "trace": {},
                              "thinking_chars": 0, "error": None, "first_token_s": None, "total_s": None}
    created = await client.post(f"{base_url}/api/conversations", json={"title": "eval golden"},
                                headers=headers, timeout=30.0)
    if created.status_code != 201:
        result["error"] = f"création de conversation : HTTP {created.status_code}"
        return result
    conversation_id = created.json()["id"]
    chunks: List[str] = []
    t0 = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{base_url}/api/chat/stream",
            json={"message": message, "conversation_id": conversation_id},
            headers=headers, timeout=timeout_s,
        ) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                result["error"] = f"HTTP {response.status_code}: {body[:300]}"
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
                if event.get("thinking"):
                    result["thinking_chars"] += len(event["thinking"])
                    continue
                content = (event.get("message") or {}).get("content")
                if content:
                    if result["first_token_s"] is None:
                        result["first_token_s"] = time.perf_counter() - t0
                    chunks.append(content)
                    continue
                if event.get("sources") is not None:
                    result["sources"] = event["sources"]
                    result["anomalies"] = event.get("anomalies") or []
                    continue
                if event.get("done"):
                    result["trace"] = event.get("trace") or {}
    finally:
        result["answer"] = "".join(chunks)
        result["total_s"] = time.perf_counter() - t0
        try:
            await client.delete(f"{base_url}/api/conversations/{conversation_id}", headers=headers, timeout=30.0)
        except Exception:  # noqa: BLE001
            pass
    return result


async def run_one(client, entry: Dict[str, Any], *, base_url: str, token: str, timeout_s: float,
                  pages_by_file: Dict[str, List[str]]) -> Dict[str, Any]:
    out = await ask(client, base_url=base_url, token=token, message=entry["question"], timeout_s=timeout_s)
    answer = out["answer"]
    if out["error"] and not answer:
        score = {"verdict": "erreur", "trouves": [], "manquants": [], "interdits_presents": [], "regex": None}
    else:
        score = score_answer(entry.get("attendu") or {}, answer)

    docs = expected_docs(entry.get("preuve") or [])
    expected_pages = {p for d in docs for p in pages_by_file.get(DOCUMENTS.get(d, ""), [])}
    cited = [s["path"] for s in out["sources"] if s.get("exists")]
    unknown = [s["path"] for s in out["sources"] if not s.get("exists")]
    trace = out["trace"] or {}
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
        "thinking_chars": out["thinking_chars"],
        "doc_cite": bool(expected_pages & set(cited)),
        "pages_citees": cited,
        "pages_attendues": sorted(expected_pages),
        "citations_inconnues": unknown,
        "anomalies": [a["id"] for a in out["anomalies"]],
        "first_token_s": round(out["first_token_s"], 2) if out["first_token_s"] else None,
        "total_s": round(out["total_s"], 2) if out["total_s"] else None,
        "prompt_tokens": trace.get("prompt_tokens"),
        "cached_tokens": trace.get("cached_tokens"),
        "completion_tokens": trace.get("completion_tokens"),
        "error": out["error"],
    }


# ---------------------------------------------------------------------------
# Agrégation et rapport
# ---------------------------------------------------------------------------


def settings_snapshot() -> Dict[str, Any]:
    from app.config import settings
    from app.services.wiki_service import get_snapshot

    snap = get_snapshot()
    return {
        "MODEL_FAST": settings.MODEL_FAST,
        "GENERATION_REASONING_EFFORT": settings.GENERATION_REASONING_EFFORT,
        "CHAT_TEMPERATURE": settings.CHAT_TEMPERATURE,
        "CHAT_MAX_TOKENS": settings.CHAT_MAX_TOKENS,
        "wiki_pages": len(snap.concept_pages),
        "wiki_chars": snap.char_count,
        "wiki_cache_key": snap.cache_key,
    }


def _pct_of(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(100.0 * sum(1 for r in rows if r.get(key)) / len(rows), 1)


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    def counts(rows):
        out = {v: 0 for v in VERDICTS}
        for r in rows:
            out[r["verdict"]] = out.get(r["verdict"], 0) + 1
        return out

    def rate(rows):
        if not rows:
            return 0.0
        good = sum(1 for r in rows if r["verdict"] in ("juste", "abstention_ok"))
        return round(100.0 * good / len(rows), 1)

    by_difficulty = {}
    for level in ("simple", "complexe"):
        rows = [r for r in results if r.get("difficulte") == level]
        if rows:
            by_difficulty[level] = {"n": len(rows), "reussite_pct": rate(rows), **counts(rows)}
    by_tag = {}
    for tag in sorted({t for r in results for t in r.get("tags") or []}):
        rows = [r for r in results if tag in (r.get("tags") or [])]
        if len(rows) >= 2:
            by_tag[tag] = {"n": len(rows), "reussite_pct": rate(rows), **counts(rows)}

    def pct(values, q):
        if not values:
            return None
        ordered = sorted(values)
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 2)

    firsts = [r["first_token_s"] for r in results if r.get("first_token_s")]
    totals = [r["total_s"] for r in results if r.get("total_s")]
    prompts = [r["prompt_tokens"] for r in results if r.get("prompt_tokens")]
    cached = [r["cached_tokens"] for r in results if r.get("cached_tokens") is not None]
    return {
        "n": len(results),
        "reussite_pct": rate(results),
        "verdicts": counts(results),
        "doc_cite_pct": _pct_of(results, "doc_cite"),
        "tours_avec_citation_inconnue": sum(1 for r in results if r.get("citations_inconnues")),
        "tours_avec_anomalie_signalee": sum(1 for r in results if r.get("anomalies")),
        "par_difficulte": by_difficulty,
        "par_tag": by_tag,
        "latence": {
            "premier_token_p50": pct(firsts, 0.5), "premier_token_p90": pct(firsts, 0.9),
            "total_p50": pct(totals, 0.5), "total_p90": pct(totals, 0.9),
            "total_max": round(max(totals), 2) if totals else None,
        },
        "tokens": {
            "prompt_median": int(sorted(prompts)[len(prompts) // 2]) if prompts else None,
            "cache_pct_moyen": round(100.0 * sum(cached) / sum(prompts), 1) if prompts and cached else None,
            "tours_sans_cache": sum(1 for c in cached if c == 0),
        },
        "erreurs": [r["id"] for r in results if r["verdict"] == "erreur"],
    }


def stability(all_runs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    unstable = []
    for qid, runs in all_runs.items():
        verdicts = [r["verdict"] for r in runs]
        if len(set(verdicts)) > 1:
            unstable.append({"id": qid, "verdicts": verdicts, "reponses": [r["answer"][:160] for r in runs]})
    return {"n_questions_instables": len(unstable), "details": unstable}


def print_report(report: Dict[str, Any]) -> None:
    agg = report["agregat"]
    s = report["reglages"]
    print()
    print("=" * 78)
    print(f"GOLDEN GÉNÉRATION — wiki CAG — {report['label']}")
    print("=" * 78)
    print(f"modèle={s.get('MODEL_FAST')}  reasoning={s.get('GENERATION_REASONING_EFFORT')}  "
          f"wiki={s.get('wiki_pages')} pages / {s.get('wiki_chars')} car.  clé={s.get('wiki_cache_key')}")
    print(f"questions={agg['n']}  essais/question={report['repeat']}")
    print("-" * 78)
    v = agg["verdicts"]
    print(f"RÉUSSITE            {agg['reussite_pct']:>5.1f} %   (juste {v['juste']} · abstention_ok {v['abstention_ok']})")
    print(f"  faux              {v['faux']:>5}")
    print(f"  ambigu            {v['ambigu']:>5}   (bonne valeur + valeur piège : à relire)")
    print(f"  abstention_ko     {v['abstention_ko']:>5}   (a répondu au lieu de s'abstenir)")
    print(f"  erreur            {v['erreur']:>5}")
    print(f"document de preuve cité      {agg['doc_cite_pct']:>5.1f} %")
    print(f"citations inconnues          {agg['tours_avec_citation_inconnue']:>5} tour(s)   "
          f"anomalies signalées {agg['tours_avec_anomalie_signalee']} tour(s)")
    print("-" * 78)
    for level, data in agg["par_difficulte"].items():
        print(f"{level:<20}{data['reussite_pct']:>5.1f} %   n={data['n']}")
    print("-" * 78)
    lat, tok = agg["latence"], agg["tokens"]
    print(f"premier token       p50 {lat['premier_token_p50']}s   p90 {lat['premier_token_p90']}s")
    print(f"total               p50 {lat['total_p50']}s   p90 {lat['total_p90']}s   max {lat['total_max']}s")
    print(f"prompt              {tok['prompt_median']} tokens (médiane)   cache {tok['cache_pct_moyen']} %   "
          f"{tok['tours_sans_cache']} tour(s) sans cache")
    print("-" * 78)
    print("PAR THÈME")
    for tag, data in sorted(agg["par_tag"].items(), key=lambda kv: kv[1]["reussite_pct"]):
        print(f"  {tag:<16}{data['reussite_pct']:>5.1f} %   n={data['n']}")
    print("-" * 78)
    print("ÉCHECS")
    for r in report["resultats"]:
        if r["verdict"] in ("juste", "abstention_ok"):
            continue
        detail = f"manque {r['manquants']}" if r["manquants"] else ""
        if r["interdits_presents"]:
            detail += f"  piège {r['interdits_presents']}"
        print(f"  [{r['verdict']:<14}] {r['id']}  {r['question'][:64]}")
        print(f"     {detail}  | doc cité={r['doc_cite']} | {r['total_s']}s")
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
    pages_by_file = wiki_pages_by_raw_file()
    reglages = settings_snapshot()
    print(f"[eval] {len(entries)} question(s) × {args.repeat} essai(s) — utilisateur {username} "
          f"— modèle {reglages['MODEL_FAST']} — wiki {reglages['wiki_pages']} pages")
    print(f"[eval] endpoint {args.base_url}")

    results: List[Dict[str, Any]] = []
    all_runs: Dict[str, List[Dict[str, Any]]] = {}
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=max(1, args.concurrency))) as client:
        semaphore = asyncio.Semaphore(max(1, args.concurrency))

        async def guarded(entry):
            async with semaphore:
                for attempt in range(1, args.retries + 2):
                    res = await run_one(client, entry, base_url=args.base_url, token=token,
                                        timeout_s=args.timeout, pages_by_file=pages_by_file)
                    if res["verdict"] != "erreur" or attempt > args.retries:
                        return res
                    wait = 8 * attempt
                    print(f"  [{entry['id']}] erreur ({str(res['error'])[:80]}) — nouvel essai dans {wait}s")
                    await asyncio.sleep(wait)
                return res

        marks = {"juste": "OK ", "abstention_ok": "OK ", "ambigu": "~~ ", "faux": "KO ",
                 "abstention_ko": "KO ", "erreur": "ERR"}
        for run_no in range(1, args.repeat + 1):
            done: List[Dict[str, Any]] = []
            for coro in asyncio.as_completed([guarded(e) for e in entries]):
                res = await coro
                res["essai"] = run_no
                done.append(res)
                all_runs.setdefault(res["id"], []).append(res)
                cache = ""
                if res.get("prompt_tokens"):
                    cache = f"cache={round(100 * (res.get('cached_tokens') or 0) / res['prompt_tokens'])}% "
                print(f"  {marks.get(res['verdict'], '?  ')} [{run_no}] {res['id']:<9} {res['verdict']:<14} "
                      f"doc={'oui' if res['doc_cite'] else 'non'} {cache}{res['total_s']}s  "
                      f"{(res['answer'] or res['error'] or '')[:66].strip()}", flush=True)
            done.sort(key=lambda r: [e["id"] for e in entries].index(r["id"]))
            if run_no == 1:
                results = list(done)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"generation_wiki_{args.label}_{stamp}.json"
    report = {
        "label": args.label, "golden": str(golden_path), "date": stamp, "repeat": args.repeat,
        "reglages": reglages, "agregat": aggregate(results),
        "stabilite": stability(all_runs) if args.repeat > 1 else None,
        "resultats": results, "tous_les_essais": all_runs if args.repeat > 1 else None,
        "chemin": str(path),
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Évaluation de la génération (wiki CAG) sur un golden set.")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--user", type=int, default=1, help="Utilisateur dont on emprunte le jeton.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--only", default="", help="Identifiants séparés par des virgules.")
    parser.add_argument("--tags", default="", help="Thèmes séparés par des virgules.")
    parser.add_argument("--difficulte", default="", choices=["", "simple", "complexe"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1, help="Essais par question (stabilité).")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1, help="Reprises sur erreur (limite de débit).")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--label", default="wiki-cag", help="Nom de la configuration mesurée.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    return parser


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
