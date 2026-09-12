"""Sonde des outils du lecteur agentique — exécute les 5 outils sur un cas RÉEL.

Aucune logique nouvelle : on instancie le même ``ToolContext`` que le routeur et on
appelle les handlers produits par ``build_reader_tools``. Ce que le script imprime est
donc EXACTEMENT ce que le modèle reçoit dans son message ``tool``.

Les images rendues sont écrites en PNG sous ``media/_probe_reader/`` (monté côté hôte)
pour pouvoir vérifier À L'ŒIL qu'un zoom cadre bien ce qu'on croit — une cote lue de
travers ne se voit pas dans un log.

Usage (dans le conteneur) :
    docker compose exec web python -m app.scripts.probe_reader_tools \
        --space 28 --doc 438 --page 8 --code 2452 \
        --question "épaisseur de vitrage pour une parclose 2452"
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import os
import re
import time
from typing import Any, Dict, List, Optional

from sqlmodel import Session

from app.database import engine
from app.models.document import Document
from app.services.reader_agent_service import ToolResult
from app.services.reader_tools import ToolContext, build_reader_tools

OUT_DIR = "media/_probe_reader"
TEXT_PREVIEW = 2000


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------


def hr(title: str) -> None:
    print("\n" + "═" * 78)
    print(f"  {title}")
    print("═" * 78)


def sub(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 70 - len(title)))


def preview(text: str, limit: int = TEXT_PREVIEW) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[… {len(text) - limit} caractères coupés par la sonde]"


def save_images(result: ToolResult, tag: str) -> List[str]:
    """Écrit les PNG de l'outil sur disque et retourne les chemins."""
    paths: List[str] = []
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, img in enumerate(result.images or [], start=1):
        b64 = img.get("b64") or ""
        if not b64:
            continue
        path = os.path.join(
            OUT_DIR, f"{tag}_doc{img.get('document_id')}_p{img.get('page_no')}_{i}.png"
        )
        try:
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(b64))
        except Exception as exc:  # noqa: BLE001
            print(f"    ! écriture PNG échouée ({path}) : {exc}")
            continue
        size_kb = len(base64.b64decode(b64)) / 1024
        paths.append(path)
        print(
            f"    image {i} → {path}  ({size_kb:.0f} Ko, doc {img.get('document_id')} "
            f"p.{img.get('page_no')}{', ' + img['label'] if img.get('label') else ''})"
        )
    return paths


async def run_tool(tools: Dict[str, Any], name: str, args: Dict[str, Any], tag: str) -> Optional[ToolResult]:
    """Appelle un outil et imprime tout ce qu'il rend."""
    spec = tools.get(name)
    if spec is None:
        print(f"  !! outil inconnu : {name}")
        return None

    label = ""
    if spec.label is not None:
        try:
            label = spec.label(args)
        except Exception as exc:  # noqa: BLE001
            label = f"(libellé en échec : {exc})"

    sub(f"{name}({args})")
    print(f"  étape affichée à l'utilisateur : « {label} »")
    t0 = time.perf_counter()
    try:
        result = await spec.handler(args)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! EXCEPTION : {type(exc).__name__}: {exc}")
        return None
    ms = int((time.perf_counter() - t0) * 1000)

    print(
        f"  → {ms} ms | error={result.error} | {len(result.text)} car. de texte | "
        f"{len(result.images)} image(s) | pages_read={result.pages_read} | "
        f"documents={result.documents}"
    )
    evidence = result.text if result.evidence is None else result.evidence
    print(f"  → preuve versée au corpus : {len(evidence or '')} car.")
    if result.images:
        save_images(result, tag)
    print("\n  ┌─ TEXTE RENDU AU MODÈLE " + "─" * 45)
    for line in preview(result.text).splitlines():
        print("  │ " + line)
    print("  └" + "─" * 68)
    return result


# ---------------------------------------------------------------------------
# Contrôles bruts sur la page (hors outils)
# ---------------------------------------------------------------------------


def inspect_page(document_id: int, page_no: int, code: str) -> None:
    """Ce que la page contient VRAIMENT : texte indexé, texte PDF, ancre du zoom."""
    from app.services.context_packer_service import _load_leaf_records
    from app.services.rag_generation_service import extract_page_text_from_pdf
    from app.services.reference_codes import spec_density

    hr(f"ÉTAT BRUT DE LA PAGE — doc {document_id}, page {page_no}")

    with Session(engine) as session:
        doc = session.get(Document, document_id)
        if doc is None:
            print("  !! document introuvable")
            return
        print(f"  titre   : {doc.title}")
        print(f"  fichier : {doc.source_file_path} (existe={os.path.exists(doc.source_file_path or '')})")
        records = _load_leaf_records(session, document_id)

    indexed = "\n".join(txt for page, _, txt in records if page == page_no)
    sub("texte INDEXÉ de la page (chunks feuilles L1/L2)")
    print(f"  {len(indexed)} caractères | densité de spécifications = {spec_density(indexed)}")
    print(f"  contient « {code} » : {code.lower() in indexed.lower()}")
    for line in preview(indexed, 1200).splitlines():
        print("  │ " + line)

    pdf_text = ""
    if doc.source_file_path and os.path.exists(doc.source_file_path):
        try:
            pdf_text = extract_page_text_from_pdf(doc.source_file_path, page_no)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! extraction pymupdf échouée : {exc}")

    sub("texte NATIF pymupdf de la page (couche texte du PDF)")
    print(f"  {len(pdf_text)} caractères | densité de spécifications = {spec_density(pdf_text)}")
    print(f"  contient « {code} » : {code.lower() in pdf_text.lower()}")
    for line in preview(pdf_text, 1200).splitlines():
        print("  │ " + line)

    # Cotes présentes dans le texte : c'est CE QUE le contrôle de sortie sait vérifier.
    sub("cotes présentes dans le texte (ce que le contrôle de sortie peut confronter)")
    units = re.findall(r"\b\d+(?:[.,]\d+)?\s?(?:mm|cm|m|kg|N|°)\b", indexed + "\n" + pdf_text, re.IGNORECASE)
    print(f"  {len(units)} occurrence(s) : {sorted(set(units))[:40]}")

    # Ancre du zoom : sans elle, zoomer(autour_de=...) retombe sur une zone approximative.
    sub(f"ancre de zoom — page.search_for({code!r})")
    if not (doc.source_file_path and os.path.exists(doc.source_file_path)):
        print("  (pas de PDF source : zoom impossible)")
        return
    try:
        import fitz

        from app.services import illustration_service as ill

        with fitz.open(doc.source_file_path) as pdf:
            if page_no < 1 or page_no > len(pdf):
                print(f"  !! page hors document ({len(pdf)} pages)")
                return
            pg = pdf[page_no - 1]
            anchors = pg.search_for(code)
            print(f"  {len(anchors)} ancre(s) trouvée(s) pour « {code} »")
            for a in anchors[:5]:
                print(f"    rect=({a.x0:.0f},{a.y0:.0f})-({a.x1:.0f},{a.y1:.0f})")
            labels = ill.find_code_labels(pg, ill._reference_pattern())
            print(f"  {len(labels)} libellé(s) de code détecté(s) sur la page")
            print(f"    échantillon : {sorted({str(l[0]) for l in labels})[:20]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! inspection PDF échouée : {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Sonde
# ---------------------------------------------------------------------------


async def probe(
    *,
    space_id: int,
    user_id: int,
    document_id: int,
    page_no: int,
    code: str,
    question: str,
    with_search: bool,
) -> None:
    ctx = ToolContext(
        space_id=space_id,
        user_id=user_id,
        allowed_document_ids=None,
        signals=None,
        known_documents={},
        matched_pages_by_doc={},
    )
    tools = {t.name: t for t in build_reader_tools(ctx)}

    hr("OUTILS DISPONIBLES")
    for name, spec in tools.items():
        print(f"  {name:18s} max_images={spec.max_images}")

    hr("1. plan_du_document")
    await run_tool(tools, "plan_du_document", {"document_id": document_id}, "plan")

    hr("2. lire_pages — TEXTE SEUL")
    await run_tool(
        tools, "lire_pages", {"document_id": document_id, "pages": [page_no]}, "lire_texte"
    )

    hr("3. lire_pages — TEXTE + IMAGE")
    await run_tool(
        tools,
        "lire_pages",
        {"document_id": document_id, "pages": [page_no], "avec_images": True},
        "lire_image",
    )

    hr("4. zoomer — ancré sur le code")
    await run_tool(
        tools,
        "zoomer",
        {"document_id": document_id, "page": page_no, "autour_de": code},
        "zoom_code",
    )

    hr("5. zoomer — repli par quart de page")
    for zone in ("haut-gauche", "haut-droit", "bas-gauche", "bas-droit"):
        await run_tool(
            tools,
            "zoomer",
            {"document_id": document_id, "page": page_no, "zone": zone},
            f"zoom_{zone}",
        )

    hr("6. chercher_code — tout le périmètre")
    await run_tool(tools, "chercher_code", {"code": code}, "code_global")

    hr("7. chercher_code — borné au document")
    await run_tool(
        tools, "chercher_code", {"code": code, "document_id": document_id}, "code_doc"
    )

    if with_search:
        hr("8. rechercher — tout le périmètre (pipeline complet, lent)")
        await run_tool(tools, "rechercher", {"question": question}, "search_global")

        hr("9. rechercher — borné au document")
        await run_tool(
            tools, "rechercher", {"question": question, "document_id": document_id}, "search_doc"
        )
    else:
        hr("8-9. rechercher — SAUTÉ (--with-search pour l'inclure)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=int, default=28)
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--doc", type=int, default=438)
    parser.add_argument("--page", type=int, default=8)
    parser.add_argument("--code", type=str, default="2452")
    parser.add_argument(
        "--question",
        type=str,
        default="épaisseur de vitrage compatible avec la parclose 2452",
    )
    parser.add_argument(
        "--with-search",
        action="store_true",
        help="inclut l'outil rechercher (relance tout le retrieval, ~15 s par appel)",
    )
    args = parser.parse_args()

    print(f"Sonde des outils du lecteur — espace {args.space}, document {args.doc}, "
          f"page {args.page}, code {args.code}")
    print(f"Images écrites sous : {OUT_DIR}/")

    inspect_page(args.doc, args.page, args.code)
    asyncio.run(
        probe(
            space_id=args.space,
            user_id=args.user,
            document_id=args.doc,
            page_no=args.page,
            code=args.code,
            question=args.question,
            with_search=args.with_search,
        )
    )
    print("\nSonde terminée.")


if __name__ == "__main__":
    main()
