from __future__ import annotations

import argparse

from sqlmodel import Session

from app.database import engine
from app.services.chunk_service import reindex_notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Réindexation hiérarchique des notes.")
    parser.add_argument("--project-id", type=int, default=None, help="Réindexer uniquement un projet")
    parser.add_argument(
        "--note-id",
        type=int,
        action="append",
        default=None,
        help="Réindexer une note précise (option répétable)",
    )
    args = parser.parse_args()

    with Session(engine) as session:
        count = reindex_notes(session, note_ids=args.note_id, project_id=args.project_id)
        print(f"Notes réindexées: {count}")


if __name__ == "__main__":
    main()
