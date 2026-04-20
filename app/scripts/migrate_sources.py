import os
import sys
from pathlib import Path

# Ajouter le répertoire racine au path pour les imports
sys.path.append(os.getcwd())

from sqlmodel import Session, select
from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.document_service_new import infer_document_source

def migrate():
    print("🚀 Démarrage de la migration des sources...")
    with Session(engine) as session:
        # 1. Récupérer tous les documents
        documents = session.exec(select(Document)).all()
        print(f"📄 Analyse de {len(documents)} documents...")
        
        updated_docs = 0
        updated_chunks = 0
        
        for doc in documents:
            # Même si doc.source est déjà rempli (ex: par un test récent), 
            # on le recalcule pour s'assurer de la cohérence.
            old_source = doc.source
            new_source = infer_document_source(
                file_path=doc.source_file_path, 
                content=doc.content
            )
            
            if new_source != old_source:
                doc.source = new_source
                session.add(doc)
                updated_docs += 1
                
                # 2. Mettre à jour les chunks associés
                chunks_stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
                chunks = session.exec(chunks_stmt).all()
                for chunk in chunks:
                    if chunk.source != new_source:
                        chunk.source = new_source
                        session.add(chunk)
                        updated_chunks += 1
            
            if updated_docs % 10 == 0 and updated_docs > 0:
                session.commit()
                print(f"✅ Progress: {updated_docs} documents traités...")

        session.commit()
        print("\n✨ Migration terminée !")
        print(f"📊 Résumé :")
        print(f"   - Documents mis à jour : {updated_docs}")
        print(f"   - Chunks mis à jour : {updated_chunks}")

if __name__ == "__main__":
    migrate()
