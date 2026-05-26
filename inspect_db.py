from sqlmodel import Session, select
from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.message_feedback import MessageFeedback

def inspect_db():
    print("=== Feedbacks récents ===")
    with Session(engine) as session:
        feedbacks = session.exec(select(MessageFeedback).order_by(MessageFeedback.created_at.desc())).all()
        print(f"Total feedbacks: {len(feedbacks)}")
        for fb in feedbacks[:5]:
            print(f"ID: {fb.id} | Positif: {fb.is_positive} | Espace: {fb.space_id} | Chunks d'origine: {fb.chunk_ids}")
            print(f"  Question: {fb.query_text[:100]}...")
            print(f"  Commentaire: {fb.comment}")
            print(f"  Généré: {fb.auto_faq_generated}")
            print(f"  Contenu FAQ: {fb.auto_faq_content}")
            print("-" * 50)

        print("\n=== Documents FAQ virtuels ===")
        faq_docs = session.exec(select(Document).where(Document.title.like("FAQ Corrective%"))).all()
        print(f"Total documents FAQ: {len(faq_docs)}")
        for doc in faq_docs:
            spaces = session.exec(select(DocumentSpace).where(DocumentSpace.document_id == doc.id)).all()
            space_ids = [s.space_id for s in spaces]
            chunks = session.exec(select(DocumentChunk).where(DocumentChunk.document_id == doc.id)).all()
            print(f"Doc ID: {doc.id} | Title: {doc.title} | Espaces: {space_ids} | Chunks: {len(chunks)}")
            for chunk in chunks:
                print(f"  Chunk ID: {chunk.id} | Content: {chunk.content[:150]}...")
            print("-" * 50)

if __name__ == "__main__":
    inspect_db()
