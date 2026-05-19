"""
Relance l'extraction KAG (fiches page + entités) pour des documents bibliothèque.

Exemples:
  python -m app.scripts.reindex_library_kag --document-id 243
  python -m app.scripts.reindex_library_kag --library-id 1
  python -m app.scripts.reindex_library_kag --all-completed
"""
from __future__ import annotations

import argparse
import logging

from sqlmodel import Session, select

from app.database import engine
from app.models.document import Document
from app.models.document_space import DocumentSpace
from app.services.chunk_service import run_kag_for_library_document

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Réindexation KAG documents bibliothèque.")
    parser.add_argument("--document-id", type=int, action="append", default=None)
    parser.add_argument("--library-id", type=int, default=None)
    parser.add_argument(
        "--all-completed",
        action="store_true",
        help="Tous les documents au statut completed",
    )
    args = parser.parse_args()

    with Session(engine) as session:
        doc_ids: list[int] = []
        if args.document_id:
            doc_ids.extend(args.document_id)
        elif args.library_id is not None:
            rows = session.exec(
                select(Document.id).where(
                    Document.library_id == args.library_id,
                    Document.processing_status == "completed",
                )
            ).all()
            doc_ids.extend(rows)
        elif args.all_completed:
            rows = session.exec(
                select(Document.id).where(Document.processing_status == "completed")
            ).all()
            doc_ids.extend(rows)
        else:
            parser.error("Spécifier --document-id, --library-id ou --all-completed")

        doc_ids = list(dict.fromkeys(doc_ids))
        if not doc_ids:
            print("Aucun document à traiter.")
            return

        print(f"KAG pour {len(doc_ids)} document(s)…")
        for document_id in doc_ids:
            spaces = session.exec(
                select(DocumentSpace.space_id).where(
                    DocumentSpace.document_id == document_id
                )
            ).all()
            if not spaces:
                print(f"  document_id={document_id} — aucun espace, ignoré")
                continue
            print(f"  document_id={document_id} — {len(spaces)} espace(s)")
            try:
                run_kag_for_library_document(document_id)
                print(f"  document_id={document_id} — OK")
            except Exception as exc:
                logger.exception("KAG échoué document_id=%s", document_id)
                print(f"  document_id={document_id} — ERREUR: {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
