"""Sonde de LECTURE VISUELLE — que sait lire un modèle sur une planche cotée ?

Répond à une question de conception : ``lire_pages`` peut-il déléguer la lecture d'une
page à un modèle vision (PNG seul, sans texte extrait) et rendre une réponse TEXTE ?

On compare, sur la même page et la même question : page entière vs crop, small vs large.
Vérité terrain connue (doc 438 p.8, cotation bleue = épaisseur vitrage) :
    parclose 2452 → 16 mm     parclose 2636 → 30 mm

Usage :
    docker compose exec web python -m app.scripts.probe_vision_read --doc 438 --page 8
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlmodel import Session

from app.database import engine
from app.models.document import Document
from app.services.mistral_service import chat

OUT_DIR = "media/_probe_reader"

# (référence, valeur attendue) — la cotation bleue de la page 8 du doc 438.
GROUND_TRUTH: List[Tuple[str, str]] = [("2452", "16"), ("2636", "30")]

QUESTION = (
    "Sur cette planche, quelle est l'épaisseur de vitrage de la parclose {code} ? "
    "La légende indique « Cotation en bleu = épaisseur vitrage » : chaque schéma porte "
    "DEUX nombres, celui en BLEU (à gauche) et un autre en noir. Donne uniquement le "
    "nombre en bleu, au format « {code} = N mm »."
)


def render(pdf_path: str, page_no: int, dpi: int) -> str:
    from app.services.multimodal_page_service import render_page_png_cached

    png = render_page_png_cached(pdf_path, page_no, dpi=dpi)
    return base64.b64encode(png).decode("utf-8")


def render_crop(pdf_path: str, page_no: int, zone: str) -> Optional[str]:
    """Crop d'un quart de page, même code que l'outil ``zoomer``."""
    import fitz

    from app.services import illustration_service as ill

    with fitz.open(pdf_path) as pdf:
        pg = pdf[page_no - 1]
        W, H = pg.rect.width, pg.rect.height
        rects = {
            "haut-gauche": fitz.Rect(0, 0, W * 0.55, H * 0.55),
            "haut-droit": fitz.Rect(W * 0.45, 0, W, H * 0.55),
            "bas-gauche": fitz.Rect(0, H * 0.45, W * 0.55, H),
            "bas-droit": fitz.Rect(W * 0.45, H * 0.45, W, H),
        }
        img = ill._make_crop_image(pdf_path, page_no, rects[zone])
    if img is None:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def ask(model: str, b64: str, question: str) -> Tuple[str, int]:
    t0 = time.perf_counter()
    try:
        result: Dict[str, Any] = await chat(
            message="",
            model=model,
            context=[{"role": "user", "content": question, "images": [b64]}],
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001
        return f"!! {type(exc).__name__}: {exc}", int((time.perf_counter() - t0) * 1000)
    # ``chat`` rend la réponse brute de l'API (format chat completions).
    choice = (result.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):  # contenu multipart
        content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return " ".join(str(content or result).split()), int((time.perf_counter() - t0) * 1000)


def verdict(answer: str, expected: str) -> str:
    if answer.startswith("!!"):
        return "ERREUR"
    return "✓ JUSTE" if f"{expected} mm" in answer or f"= {expected}" in answer else "✗ FAUX"


async def main_async(doc_id: int, page_no: int, models: List[str], dpis: List[int]) -> None:
    with Session(engine) as session:
        doc = session.get(Document, doc_id)
        if doc is None or not doc.source_file_path or not os.path.exists(doc.source_file_path):
            print("!! document ou PDF introuvable")
            return
        pdf_path = doc.source_file_path
        title = doc.title

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Lecture visuelle — doc {doc_id} « {title} », page {page_no}")
    print(f"Vérité terrain : " + ", ".join(f"{c} = {v} mm" for c, v in GROUND_TRUTH))

    variants: List[Tuple[str, str]] = []
    for dpi in dpis:
        variants.append((f"page entière @ {dpi} dpi", render(pdf_path, page_no, dpi)))
    for zone in ("haut-gauche", "haut-droit"):
        crop = render_crop(pdf_path, page_no, zone)
        if crop:
            variants.append((f"crop {zone} @ 200 dpi", crop))

    for label, b64 in variants:
        size_kb = len(base64.b64decode(b64)) / 1024
        print("\n" + "═" * 78)
        print(f"  {label}  ({size_kb:.0f} Ko)")
        print("═" * 78)
        for model in models:
            for code, expected in GROUND_TRUTH:
                answer, ms = await ask(model, b64, QUESTION.format(code=code))
                print(f"  {model:24s} {code} → {verdict(answer, expected):8s} {ms:5d} ms")
                print(f"      « {answer[:200]} »")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=int, default=438)
    parser.add_argument("--page", type=int, default=8)
    parser.add_argument("--models", type=str, default="mistral-small-latest,mistral-large-latest")
    parser.add_argument("--dpis", type=str, default="220")
    args = parser.parse_args()

    asyncio.run(
        main_async(
            args.doc,
            args.page,
            [m.strip() for m in args.models.split(",") if m.strip()],
            [int(d) for d in args.dpis.split(",") if d.strip()],
        )
    )


if __name__ == "__main__":
    main()
