"""
Script de validation : compare extraction pymupdf4llm vs Mistral OCR sur le corpus docs/generale.

Usage (depuis la racine du projet) :
    python scripts/test_pdf_extraction.py
    python scripts/test_pdf_extraction.py --ocr     # forcer Mistral OCR (comparaison)
    python scripts/test_pdf_extraction.py --file "docs/generale/SOLEAL-GY-55-notice-installation-5832-002-052018-FR (1).pdf"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ajouter la racine au path pour les imports app.*
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Charger .env pour app.config (DATABASE_URL, etc.)
import os
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
# Valeurs minimales si .env incomplet (script de test uniquement)
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-scripts")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test extraction PDF hybride")
    p.add_argument("--ocr", action="store_true", help="Forcer Mistral OCR (bypass pymupdf4llm)")
    p.add_argument("--file", help="Tester un seul fichier PDF")
    p.add_argument("--preview", type=int, default=400, metavar="N", help="Nb chars aperçu (défaut: 400)")
    return p.parse_args()


def check_pymupdf4llm() -> bool:
    try:
        import pymupdf4llm  # noqa: F401
        return True
    except ImportError:
        print("[WARN] pymupdf4llm non installe — installer avec: pip install pymupdf4llm")
        return False


def test_pdf(pdf_path: Path, force_ocr: bool, preview_chars: int) -> dict:
    from app.services.pdf_extraction_service import (
        extract_markdown_from_pdf,
        extract_markdown_pymupdf4llm,
        has_extractable_text,
    )

    is_text = has_extractable_text(str(pdf_path))
    t0 = time.perf_counter()

    if force_ocr:
        try:
            from app.services.mistral_ocr_service import _ocr_pdf_page_by_page
            md = _ocr_pdf_page_by_page(str(pdf_path))
            method = "ocr (force)"
        except ImportError as exc:
            raise RuntimeError(
                "Mode --ocr necessite pdf2image et MISTRAL_API_KEY. "
                f"Import manquant: {exc}"
            ) from exc
    else:
        try:
            md, method = extract_markdown_from_pdf(str(pdf_path))
        except ImportError:
            # Test natif sans deps OCR (pdf2image, mistral)
            md = extract_markdown_pymupdf4llm(str(pdf_path))
            method = "native (direct)"

    elapsed = time.perf_counter() - t0

    # Indicateurs qualité simples
    heading_count = md.count("\n#")
    table_count = md.count("\n|")
    list_count = md.count("\n- ") + md.count("\n* ")
    page_markers = md.count("<!-- page:")

    return {
        "file": pdf_path.name,
        "is_text_pdf": is_text,
        "method": method,
        "elapsed_s": elapsed,
        "chars": len(md),
        "headings": heading_count,
        "table_rows": table_count,
        "list_items": list_count,
        "page_markers": page_markers,
        "preview": md[:preview_chars].replace("\n", " | "),
    }


def print_result(r: dict) -> None:
    icon = "[TXT]" if r["is_text_pdf"] else "[SCAN]"
    print(f"\n{icon}  {r['file']}")
    print(f"   Méthode    : {r['method']}")
    print(f"   Durée      : {r['elapsed_s']:.2f}s")
    print(f"   Longueur   : {r['chars']:,} chars")
    print(f"   Headings # : {r['headings']}")
    print(f"   Lignes |   : {r['table_rows']}")
    print(f"   Items -    : {r['list_items']}")
    print(f"   Marqueurs  : {r['page_markers']} × <!-- page:N -->")
    print(f"   Aperçu     : {r['preview'][:300]}...")


def main() -> None:
    args = parse_args()

    if not check_pymupdf4llm():
        sys.exit(1)

    if args.file:
        pdfs = [Path(args.file)]
        if not pdfs[0].exists():
            print(f"Fichier introuvable : {args.file}")
            sys.exit(1)
    else:
        corpus = ROOT / "docs" / "generale"
        pdfs = sorted(corpus.glob("*.pdf"))
        if not pdfs:
            print(f"Aucun PDF trouvé dans {corpus}")
            sys.exit(1)

    if args.ocr:
        print(f"\n[OCR] Mode comparaison : Mistral OCR force sur {len(pdfs)} PDF(s)")
        print("      (necessite MISTRAL_API_KEY dans l'environnement)")
    else:
        print(f"\n[TEST] Extraction hybride pymupdf4llm sur {len(pdfs)} PDF(s)")

    total_elapsed = 0.0
    native_count = 0
    ocr_count = 0

    for pdf in pdfs:
        try:
            r = test_pdf(pdf, force_ocr=args.ocr, preview_chars=args.preview)
            print_result(r)
            total_elapsed += r["elapsed_s"]
            if "native" in r["method"]:
                native_count += 1
            else:
                ocr_count += 1
        except Exception as exc:
            print(f"\n[ERR] {pdf.name} -> ERREUR : {exc}")

    if len(pdfs) > 1:
        print(f"\n{'='*60}")
        print(f"  Total    : {len(pdfs)} PDF(s) en {total_elapsed:.1f}s ({total_elapsed/len(pdfs):.2f}s/PDF)")
        if not args.ocr:
            print(f"  Natif    : {native_count} ({100*native_count//len(pdfs)}%)")
            print(f"  OCR      : {ocr_count} ({100*ocr_count//len(pdfs)}%)")


if __name__ == "__main__":
    main()
