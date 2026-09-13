"""Sonde du pack de LECTURE — imprime EXACTEMENT ce que le générateur recevra.

Rejoue le tour 0 réel (retrieval, élection) puis construit le pack de lecture, sans appeler
le générateur. Sert à vérifier à l'œil qu'aucun texte indexé ne fuit dans le contexte et
que les pages utiles ont bien été lues sur l'image.

Usage (dans le conteneur) :
    docker compose exec web python -m app.scripts.probe_reading_pack \
        --space 28 --question "donne moi l'epaisseur de vitrage pour une parclose 2636"
"""
from __future__ import annotations

import argparse
import asyncio
import time

from sqlmodel import Session

from app.config import settings
from app.database import engine
from app.services.reading_pack_service import build_reading_pack


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", type=int, default=28)
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--question", required=True)
    parser.add_argument("--needle", default="")
    parser.add_argument("--full", action="store_true", help="imprimer le pack entier")
    args = parser.parse_args()

    from app.services.coverage_service import extract_message_reference_codes
    from app.services.space_search_service import search_technical_passages

    needle = args.needle
    if not needle:
        codes = extract_message_reference_codes(args.question)
        needle = codes[0] if codes else ""

    with Session(engine) as session:
        t0 = time.perf_counter()
        retrieval = await search_technical_passages(
            session=session,
            space_id=args.space,
            query_text=args.question,
            user_id=args.user,
            k=settings.RAG_TOP_K,
        )
        t_retrieval = time.perf_counter() - t0

        election = (retrieval.get("election") or {}).get("elected") or []
        elected = [int(d["document_id"]) for d in election if d.get("document_id") is not None]
        print(f"\nretrieval {t_retrieval:.1f}s — {len(retrieval.get('passages') or [])} passages, élus {elected}")

        t1 = time.perf_counter()
        pack = await build_reading_pack(
            session,
            retrieval.get("passages") or [],
            question=args.question,
            system_prompt="",
            needle=needle,
            elected_document_ids=elected or None,
            dpi=settings.CAG_IMAGE_DPI,
        )
        t_pack = time.perf_counter() - t1

    tr = pack.get("reading_trace") or {}
    print(
        f"pack de lecture {t_pack:.1f}s — {tr.get('pages_read')} page(s) lue(s) : "
        f"{tr.get('pages_answered')} répondent, {tr.get('pages_absent')} absentes, "
        f"{tr.get('pages_failed')} en échec"
    )
    for d in tr.get("details") or []:
        print(f"   doc {d['document_id']} p.{d['page_no']:<4} {d['etat']:<8} {d['ms']:>6}ms")

    contenu = pack.get("content") or ""
    print(f"\ncontexte : {len(contenu)} caractères, {len(pack.get('cag_documents') or [])} document(s)")
    print("=" * 96)
    print(contenu if args.full else contenu[:4000])
    if not args.full and len(contenu) > 4000:
        print(f"\n[… {len(contenu) - 4000} caractères coupés par la sonde, utilise --full]")


if __name__ == "__main__":
    asyncio.run(main())
