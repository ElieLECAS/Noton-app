from app.database import get_db
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from sqlmodel import select

def inspect():
    db = next(get_db())
    # List all documents in the DB
    docs = db.exec(select(Document)).all()
    print("=== DOCUMENTS ===")
    for d in docs:
        print(f"ID: {d.id}, Title: {d.title}, Filename: {d.filename}")
        
    # Check chunks for a specific document
    for d in docs:
        if "Montage" in d.title or "Roto" in d.title:
            chunks = db.exec(select(DocumentChunk).where(DocumentChunk.document_id == d.id)).all()
            print(f"\n=== CHUNKS FOR {d.title} (Total: {len(chunks)}) ===")
            pages = sorted(list(set(c.metadata_json.get("page_no") or c.metadata_json.get("page_start") or c.metadata_.get("page_no") or c.metadata_.get("page_start") for c in chunks if c.metadata_json or c.metadata_)))
            print("Pages available:", pages[:20], "... total pages count:", len(pages))

if __name__ == "__main__":
    inspect()
