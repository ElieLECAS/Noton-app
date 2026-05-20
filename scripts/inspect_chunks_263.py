"""Inspect chunks for document_id=263."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ.setdefault(
    "DATABASE_URL",
    f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'postgres')}@localhost:5435/"
    f"{os.getenv('POSTGRES_DB', 'noton')}",
)
os.environ.setdefault("SECRET_KEY", "inspect-script")

from sqlmodel import Session, select

from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

DOC_ID = 263

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    with Session(engine) as session:
        doc = session.get(Document, DOC_ID)
        if not doc:
            print(f"Document {DOC_ID} introuvable")
            return
        print(f"DOC id={doc.id} title={doc.title!r}")
        print(f"     source={getattr(doc, 'source', None)}")
        print(f"     status={getattr(doc, 'processing_status', None)}")

        chunks = list(
            session.exec(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == DOC_ID)
                .order_by(DocumentChunk.chunk_index)
            ).all()
        )
        print(f"\nNB CHUNKS: {len(chunks)}\n")

        for c in chunks:
            meta = dict(c.metadata_json or {})
            headers = [
                meta.get(f"Header {i}")
                for i in range(1, 7)
                if meta.get(f"Header {i}")
            ]
            heading = " > ".join(headers) or meta.get("parent_heading") or meta.get("heading") or "-"
            ver = meta.get("chunking_version", "?")
            content = c.content or ""
            print("=" * 70)
            print(
                f"chunk_index={c.chunk_index} id={c.id} is_leaf={c.is_leaf} "
                f"chars={len(content)} ver={ver}"
            )
            print(f"heading: {heading}")
            print("-" * 70)
            print(content[:1200])
            if len(content) > 1200:
                print(f"... [{len(content) - 1200} chars restants]")


if __name__ == "__main__":
    main()
