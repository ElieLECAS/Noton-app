from sqlmodel import Session, select, text
from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.services.space_search_service import _retrieve_leaves_sql, _retrieve_leaves_bm25_sql

def run_diagnostics():
    space_id = 19
    query_text = (
        "Lors du montage d'un seuil PY1100 pour une porte SOLEAL PY, "
        "quelle est la référence de l'ensemble \"pièces de liaison porte\" "
        "(indispensable pour l'obturation et l'étanchéité de la partie basse des montants) "
        "et quelle pièce spécifique (référence) doit être mise en butée contre le bouchon côté serrure ?"
    )
    
    print("=== DIAGNOSTIC RAG FAQ CORRECTIVE ===")
    
    with Session(engine) as session:
        # 1. Inspecter les documents FAQ en base
        faq_docs = session.exec(select(Document).where(Document.title.like("FAQ Corrective%"))).all()
        print(f"\n1. Documents FAQ trouvés en BDD ({len(faq_docs)}):")
        for doc in faq_docs:
            spaces = session.exec(select(DocumentSpace).where(DocumentSpace.document_id == doc.id)).all()
            space_ids = [s.space_id for s in spaces]
            chunks = session.exec(select(DocumentChunk).where(DocumentChunk.document_id == doc.id)).all()
            print(f"  - ID: {doc.id} | Title: '{doc.title}'")
            print(f"    Espaces associés (DocumentSpace): {space_ids}")
            print(f"    Nombre de chunks: {len(chunks)}")
            for c in chunks:
                has_emb = "Oui" if c.embedding is not None else "Non"
                emb_len = len(c.embedding) if c.embedding is not None else 0
                print(f"      * Chunk ID: {c.id} | is_leaf: {c.is_leaf} | Embedding: {has_emb} ({emb_len} dims)")
                print(f"        Contenu: {c.content[:200]}...")

        # 2. Exécuter la recherche vectorielle manuellement
        print(f"\n2. Exécution de _retrieve_leaves_sql pour l'espace {space_id}...")
        vector_results = _retrieve_leaves_sql(session, space_id, 1, query_text, candidate_k=80)
        print(f"Nombre de résultats vectoriels: {len(vector_results)}")
        
        faq_in_vector = [r for r in vector_results if r.node.metadata.get("document_title", "").startswith("FAQ Corrective")]
        print(f"\nPassages FAQ dans les résultats vectoriels ({len(faq_in_vector)}):")
        for r in faq_in_vector:
            print(f"  - Doc: '{r.node.metadata.get('document_title')}' | Score: {r.score:.4f} | Content: {r.node.text[:150]}...")
            
        print("\nTop 5 des résultats vectoriels globaux:")
        for i, r in enumerate(vector_results[:5], 1):
            print(f"  {i}. Doc: '{r.node.metadata.get('document_title')}' | Score: {r.score:.4f} | Content: {r.node.text[:100]}...")

        # 3. Exécuter la recherche lexicale manuellement
        print(f"\n3. Exécution de _retrieve_leaves_bm25_sql pour l'espace {space_id}...")
        try:
            lexical_results = _retrieve_leaves_bm25_sql(session, space_id, 1, query_text, candidate_k=80)
            print(f"Nombre de résultats lexicaux: {len(lexical_results)}")
            
            faq_in_lexical = [r for r in lexical_results if r.node.metadata.get("document_title", "").startswith("FAQ Corrective")]
            print(f"\nPassages FAQ dans les résultats lexicaux ({len(faq_in_lexical)}):")
            for r in faq_in_lexical:
                print(f"  - Doc: '{r.node.metadata.get('document_title')}' | Score: {r.score:.4f} | Content: {r.node.text[:150]}...")
                
            print("\nTop 5 des résultats lexicaux globaux:")
            for i, r in enumerate(lexical_results[:5], 1):
                print(f"  {i}. Doc: '{r.node.metadata.get('document_title')}' | Score: {r.score:.4f} | Content: {r.node.text[:100]}...")
        except Exception as e:
            print(f"Erreur recherche lexicale: {e}")

if __name__ == "__main__":
    run_diagnostics()
