"""Sonde de GÉNÉRATION isolée du retriever — 15/09/2026.

Question d'Elie : « le retriever retourne très souvent la bonne page ; c'est en lisant ce
qu'il a en contexte (PNG et/ou texte extrait) que le modèle dit des bêtises ». On mesure
donc l'étage de lecture SEUL : pour chaque question du golden espace 29, on donne au
modèle la ou les pages de PREUVE (celles que le retriever aurait dû rendre) sous
plusieurs formes, avec un prompt neutre identique, et on note la réponse avec la même
fonction de score que le runner d'évaluation.

Variantes :
  png       : PNG des pages de preuve seuls (vanilla image only)
  png_frag  : PNG + fragments indexés de ces pages, rendus comme le packer les rend
  png_md    : PNG + markdown de page (media/page_markdown/<doc>.md, ou le cahier
              technique Perform76 pour le dossier 438 sans couche texte)
  md        : markdown de page seul, sans image
  frag      : fragments indexés seuls, sans image

Le dossier 438 n'existe plus en base locale : ses planches sont prises dans
media/_eval/probe_inputs/perform76_complet.pdf (page originale N = N-ième page sans
caractère de ce PDF composite) et son « markdown » dans perform76_cahier.md
(section « Page P » avec P = N - 3).

Usage (dans le conteneur) ::

    docker compose exec web python -m app.scripts.probe_generation_variants
    docker compose exec web python -m app.scripts.probe_generation_variants --only g29_001,g29_013
    docker compose exec web python -m app.scripts.probe_generation_variants --variants png,png_md
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import settings
from app.scripts.eval_golden_generation import score_answer

GOLDEN = "tests/fixtures/golden/space29_perform76_generation.json"
OUT_DIR = Path("media/_eval")
P76_PDF = Path("media/_eval/probe_inputs/perform76_complet.pdf")
P76_MD = Path("media/_eval/probe_inputs/perform76_cahier.md")
PAGE_MD_DIR = Path("media/page_markdown")
P76_DOC_ID = 438
ALL_VARIANTS = ("png", "png_frag", "png_md", "md", "frag")

SYSTEM_PROMPT = (
    "Tu es un technicien en menuiserie. On te fournit une ou plusieurs pages d'un document "
    "technique (image de la page et/ou texte de la page). Réponds à la question uniquement "
    "à partir de ces pages. Donne la ou les valeurs exactes telles qu'elles sont écrites, "
    "avec leur unité. Si les pages ne permettent pas de répondre, dis-le explicitement."
)


# ---------------------------------------------------------------------------
# Sources : PNG, fragments, markdown
# ---------------------------------------------------------------------------

_p76_zero_pages: Optional[List[int]] = None


def _p76_page_index(original_page: int) -> int:
    """Page originale N du dossier 438 → index (1-based) dans le PDF composite."""
    global _p76_zero_pages
    import fitz

    if _p76_zero_pages is None:
        doc = fitz.open(str(P76_PDF))
        _p76_zero_pages = [i + 1 for i, p in enumerate(doc) if len(p.get_text()) == 0]
        doc.close()
    return _p76_zero_pages[original_page - 1]


def render_png_b64(pdf_path: str, page_no: int, dpi: int) -> str:
    import fitz

    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi)
        return base64.b64encode(pix.tobytes("png")).decode("ascii")
    finally:
        doc.close()


def doc_pdf_path(session, doc_id: int) -> Optional[str]:
    from app.models.document import Document

    doc = session.get(Document, doc_id)
    if doc and doc.source_file_path and os.path.exists(doc.source_file_path):
        return doc.source_file_path
    return None


def page_png(session, doc_id: int, page: int, dpi: int) -> Optional[str]:
    if doc_id == P76_DOC_ID:
        return render_png_b64(str(P76_PDF), _p76_page_index(page), dpi)
    path = doc_pdf_path(session, doc_id)
    return render_png_b64(path, page, dpi) if path else None


def page_fragments(session, doc_id: int, page: int) -> Optional[str]:
    """Texte que le packer CAG rend pour cette page (chunks feuilles, ordre de lecture)."""
    from app.services.context_packer_service import _load_leaf_records

    records = [r for r in _load_leaf_records(session, doc_id) if r[0] == page]
    if not records:
        return None
    return f"[page {page}]\n" + "\n".join(text for _, _, text in records)


_md_cache: Dict[int, str] = {}


def page_markdown(doc_id: int, page: int) -> Optional[str]:
    if doc_id == P76_DOC_ID:
        printed = page - 3
        if printed < 1:
            return None
        text = P76_MD.read_text(encoding="utf-8")
        m = re.search(rf"(?ms)^## Page {printed} — .*?(?=^## |\Z)", text)
        if not m:
            return None
        conv = re.search(r"(?ms)^## Conventions\n.*?(?=^---)", text)
        return ((conv.group(0).strip() + "\n\n") if conv else "") + m.group(0).strip()
    path = PAGE_MD_DIR / f"{doc_id}.md"
    if not path.exists():
        return None
    if doc_id not in _md_cache:
        _md_cache[doc_id] = path.read_text(encoding="utf-8")
    text = _md_cache[doc_id]
    m = re.search(rf"(?ms)^## Page PDF {page} .*?(?=^## Page PDF |\Z)", text)
    if not m:
        return None
    conv = re.search(r"(?ms)^## Conventions de lecture\n.*?(?=^---)", text)
    return ((conv.group(0).strip() + "\n\n") if conv else "") + m.group(0).strip()


# ---------------------------------------------------------------------------
# Appel modèle
# ---------------------------------------------------------------------------


async def call_model(
    client: httpx.AsyncClient,
    *,
    question: str,
    text_blocks: List[str],
    images_b64: List[str],
    model: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []
    user_text = ""
    if text_blocks:
        user_text += "TEXTE DES PAGES :\n\n" + "\n\n".join(text_blocks) + "\n\n"
    if images_b64:
        user_text += f"({len(images_b64)} image(s) de page jointe(s).)\n\n"
    user_text += "QUESTION : " + question
    parts.append({"type": "text", "text": user_text})
    for img in images_b64:
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": parts},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    if seed is not None:
        payload["random_seed"] = seed
    base_url = (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/")
    headers = {
        "Authorization": f"Bearer {settings.MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }
    t0 = time.perf_counter()
    last_exc: Optional[Exception] = None
    for attempt in range(5):
        try:
            resp = await client.post(f"{base_url}/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            msg = (data.get("choices") or [{}])[0].get("message") or {}
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            usage = data.get("usage") or {}
            return {
                "answer": content,
                "seconds": round(time.perf_counter() - t0, 2),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            await asyncio.sleep(2.0 * (attempt + 1))
    return {"answer": "", "error": str(last_exc), "seconds": round(time.perf_counter() - t0, 2)}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_inputs(session, entry: Dict[str, Any], dpi: int) -> Dict[str, Any]:
    """Pour une question : PNG, fragments et markdown de chaque page de preuve."""
    pages: List[Tuple[int, int]] = []
    for p in entry.get("preuve") or []:
        did = int(p["document_id"])
        for pg in p.get("pages") or []:
            pages.append((did, int(pg)))
    pages = pages[:3]
    pngs, frags, mds = [], [], []
    for did, pg in pages:
        try:
            png = page_png(session, did, pg, dpi)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! PNG {did} p{pg} : {exc}", flush=True)
            png = None
        if png:
            pngs.append(png)
        frag = page_fragments(session, did, pg)
        if frag:
            frags.append(f"=== Document {did}, page {pg} (texte indexé) ===\n{frag}")
        md = page_markdown(did, pg)
        if md:
            mds.append(f"=== Document {did}, page {pg} (markdown) ===\n{md}")
    return {"pages": pages, "png": pngs, "frag": frags, "md": mds}


def variant_payload(variant: str, inputs: Dict[str, Any]) -> Optional[Tuple[List[str], List[str]]]:
    """(text_blocks, images) pour la variante, ou None si la matière manque."""
    png, frag, md = inputs["png"], inputs["frag"], inputs["md"]
    if variant == "png":
        return ([], png) if png else None
    if variant == "png_frag":
        return (frag, png) if (png and frag) else None
    if variant == "png_md":
        return (md, png) if (png and md) else None
    if variant == "md":
        return (md, []) if md else None
    if variant == "frag":
        return (frag, []) if frag else None
    raise ValueError(variant)


def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scored = [r for r in rows if r["verdict"] != "n/a"]
    ok = sum(1 for r in scored if r["verdict"] in ("juste", "abstention_ok"))
    secs = [r["seconds"] for r in scored if r.get("seconds")]
    toks = [r["prompt_tokens"] for r in scored if r.get("prompt_tokens")]
    return {
        "n": len(scored),
        "reussite_pct": round(100.0 * ok / len(scored), 1) if scored else None,
        "verdicts": {
            v: sum(1 for r in scored if r["verdict"] == v) for v in sorted({r["verdict"] for r in scored})
        },
        "latence_p50": round(statistics.median(secs), 1) if secs else None,
        "prompt_tokens_p50": round(statistics.median(toks)) if toks else None,
    }


async def main_async(args: argparse.Namespace) -> int:
    from sqlmodel import Session

    from app.database import engine

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    entries = golden["entries"]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        entries = [e for e in entries if e["id"] in wanted]
    entries = [e for e in entries if e.get("preuve")]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    model = args.model or settings.MODEL_FAST
    print(
        f"{len(entries)} question(s) × {variants} — modèle {model}, dpi {args.dpi}, seed {args.seed}",
        flush=True,
    )

    sem = asyncio.Semaphore(args.concurrency)
    results: List[Dict[str, Any]] = []

    with Session(engine) as session:
        prepared = []
        for e in entries:
            inputs = build_inputs(session, e, args.dpi)
            prepared.append((e, inputs))
            print(
                f"  {e['id']}: pages={inputs['pages']} png={len(inputs['png'])} "
                f"frag={len(inputs['frag'])} md={len(inputs['md'])}",
                flush=True,
            )

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=30.0)) as client:

        async def run_one(entry: Dict[str, Any], inputs: Dict[str, Any], variant: str) -> None:
            payload = variant_payload(variant, inputs)
            row: Dict[str, Any] = {
                "id": entry["id"],
                "variant": variant,
                "tags": entry.get("tags") or [],
                "difficulte": entry.get("difficulte"),
                "question": entry["question"],
                "pages": inputs["pages"],
            }
            if payload is None:
                row.update({"verdict": "n/a", "answer": ""})
                results.append(row)
                return
            text_blocks, images = payload
            async with sem:
                out = await call_model(
                    client,
                    question=entry["question"],
                    text_blocks=text_blocks,
                    images_b64=images,
                    model=model,
                    seed=args.seed,
                )
            row.update(out)
            row["text_chars"] = sum(len(t) for t in text_blocks)
            if out.get("error"):
                row["verdict"] = "erreur"
            else:
                row.update(score_answer(entry["attendu"], out["answer"]))
            results.append(row)
            print(
                f"  {entry['id']:8s} {variant:9s} → {row['verdict']:14s} {out.get('seconds')}s",
                flush=True,
            )

        tasks = [run_one(e, inp, v) for e, inp in prepared for v in variants]
        await asyncio.gather(*tasks)

    summary: Dict[str, Any] = {
        "par_variante": {v: _agg([r for r in results if r["variant"] == v]) for v in variants}
    }
    ids_all = {
        e["id"] for e, inp in prepared if all(variant_payload(v, inp) is not None for v in variants)
    }
    summary["sous_ensemble_complet"] = {
        "n_questions": len(ids_all),
        "par_variante": {
            v: _agg([r for r in results if r["variant"] == v and r["id"] in ids_all]) for v in variants
        },
    }
    summary["par_tag"] = {}
    for tag in ("cote", "parclose", "tableau", "texte", "catalogue", "code", "perform76", "innoslide"):
        summary["par_tag"][tag] = {
            v: _agg([r for r in results if r["variant"] == v and tag in r["tags"]]) for v in variants
        }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"probe_variants_{args.label}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "label": args.label,
                "date": stamp,
                "model": model,
                "dpi": args.dpi,
                "seed": args.seed,
                "system_prompt": SYSTEM_PROMPT,
                "summary": summary,
                "results": sorted(results, key=lambda r: (r["id"], r["variant"])),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print("\n=== RÉSUMÉ ===")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n→ {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--golden", default=GOLDEN)
    p.add_argument("--only", default=None)
    p.add_argument("--variants", default=",".join(ALL_VARIANTS))
    p.add_argument("--model", default=None)
    p.add_argument("--dpi", type=int, default=settings.CAG_IMAGE_DPI)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--label", default="lecture-isolee")
    return p


def main() -> int:
    return asyncio.run(main_async(build_parser().parse_args()))


if __name__ == "__main__":
    sys.exit(main())
