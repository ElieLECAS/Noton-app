"""
Liste les documents non classés et suggère un fournisseur probable (sans auto-classifier).

Usage:
    python -m app.scripts.backfill_document_classification
    python -m app.scripts.backfill_document_classification --csv report.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sqlmodel import Session, select

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import engine
from app.models.document import Document
from app.services.document_service_new import infer_document_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Rapport documents non classés")
    parser.add_argument("--csv", type=str, default="", help="Chemin CSV de sortie")
    args = parser.parse_args()

    rows: list[dict] = []
    with Session(engine) as session:
        docs = session.exec(
            select(Document).where(Document.classification_status != "complete")
        ).all()
        for doc in docs:
            suggested = infer_document_source(
                file_path=doc.source_file_path,
                content=doc.content,
            )
            rows.append(
                {
                    "id": doc.id,
                    "title": doc.title,
                    "classification_status": doc.classification_status,
                    "current_supplier": doc.source or "",
                    "suggested_supplier": suggested,
                    "product_types": ",".join(doc.product_types or []),
                    "materials": ",".join(doc.materials or []),
                }
            )

    print(f"Documents non classés : {len(rows)}")
    for row in rows[:20]:
        print(
            f"  #{row['id']} {row['title'][:50]} "
            f"(fournisseur suggéré: {row['suggested_supplier']})"
        )
    if len(rows) > 20:
        print(f"  ... et {len(rows) - 20} autres")

    if args.csv:
        out = Path(args.csv)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        print(f"Rapport CSV : {out}")


if __name__ == "__main__":
    main()
