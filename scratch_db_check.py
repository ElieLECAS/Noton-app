from sqlmodel import Session, select
from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

with Session(engine) as session:
    docs = session.exec(select(Document)).all()
    print(f"Total documents: {len(docs)}")
    for d in docs:
        print(f"Doc ID: {d.id}, Title: {d.title}, Source: {d.source_file_path}")
        chunks = session.exec(select(DocumentChunk).where(DocumentChunk.document_id == d.id)).all()
        print(f"  Total chunks: {len(chunks)}")
        if chunks:
            print("  First chunk fields:")
            c = chunks[0]
            print(f"    id: {c.id}")
            print(f"    chunk_index: {c.chunk_index}")
            print(f"    is_leaf: {c.is_leaf}")
            print(f"    content (len={len(c.content or '')}): {repr((c.content or '')[:100])}")
            print(f"    text (len={len(c.text or '')}): {repr((c.text or '')[:100])}")
            print(f"    metadata_json: {c.metadata_json}")
