import pytest
from unittest import mock
from sqlmodel import Session, select
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.chunk_entity_relation import ChunkEntityRelation
from app.models.knowledge_entity import KnowledgeEntity
from app.models.space import Space
from app.models.document_space import DocumentSpace
from app.services.kag_graph_service import process_kag_for_document_space
from app.services.space_search_service import _retrieve_parent_enriched_sql


@pytest.fixture
def mock_embedding_batch():
    # Mock generate_embeddings_batch pour retourner des embeddings bidons
    fake_embedding = [0.1] * 1024
    with mock.patch("app.services.embedding_service.generate_embeddings_batch", return_value=[fake_embedding, fake_embedding, fake_embedding, fake_embedding]):
        yield


def test_page_enrichment_pipeline(db_session: Session, mock_embedding_batch):
    # 1. Préparation des données d'entrée
    # Créer un espace de tests
    space = Space(name="Test Space", description="Espace de test pour enrichissement par page")
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    # Créer un document
    doc = Document(
        title="Notice Luméal",
        content="",
        document_type="document",
        processing_status="completed",
        processing_progress=100
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    # Lier le document à l'espace
    doc_space = DocumentSpace(document_id=doc.id, space_id=space.id)
    db_session.add(doc_space)
    db_session.commit()

    # Créer des chunks feuilles (leaves)
    leaf1 = DocumentChunk(
        document_id=doc.id,
        content="Le coulissant Luméal présente une isolation thermique renforcée.",
        text="Le coulissant Luméal présente une isolation thermique renforcée.",
        is_leaf=True,
        hierarchy_level=3,
        metadata_json={"page_no": 1, "page_start": 1, "document_title": "Notice Luméal"}
    )
    leaf2 = DocumentChunk(
        document_id=doc.id,
        content="Le vitrage isolant de Luméal doit avoir une épaisseur de 24mm.",
        text="Le vitrage isolant de Luméal doit avoir une épaisseur de 24mm.",
        is_leaf=True,
        hierarchy_level=3,
        metadata_json={"page_no": 1, "page_start": 1, "document_title": "Notice Luméal"}
    )
    leaf3 = DocumentChunk(
        document_id=doc.id,
        content="Le coulissant Soleal dispose de profils fins.",
        text="Le coulissant Soleal dispose de profils fins.",
        is_leaf=True,
        hierarchy_level=3,
        metadata_json={"page_no": 2, "page_start": 2, "document_title": "Notice Luméal"}
    )
    db_session.add(leaf1)
    db_session.add(leaf2)
    db_session.add(leaf3)
    db_session.commit()
    db_session.refresh(leaf1)
    db_session.refresh(leaf2)
    db_session.refresh(leaf3)

    # 2. Mocking du service d'extraction par page (LLM)
    mock_extract_responses = {
        1: {
            "summary": "Résumé de la page 1 concernant le coulissant Luméal et son vitrage de 24mm.",
            "qas": [
                {"question": "Quelle est l'épaisseur du vitrage Luméal ?", "answer": "24mm"}
            ],
            "entities": [
                {"name": "Luméal", "type": "produit", "importance": 0.9}
            ]
        },
        2: {
            "summary": "Résumé de la page 2 concernant le coulissant Soleal et ses profils fins.",
            "qas": [
                {"question": "Comment sont les profils du coulissant Soleal ?", "answer": "Ils sont fins."}
            ],
            "entities": [
                {"name": "Soleal", "type": "produit", "importance": 0.8}
            ]
        }
    }

    def side_effect_extract(page_content, context_hint=None):
        # Déterminer la page à partir du context_hint
        if "Page : 1" in context_hint:
            return mock_extract_responses[1]
        elif "Page : 2" in context_hint:
            return mock_extract_responses[2]
        return None

    # Appliquer le mock
    with mock.patch("app.services.kag_graph_service.generate_page_enrichment_and_entities_sync", side_effect=side_effect_extract):
        # Exécuter l'ingestion par page
        result = process_kag_for_document_space(db_session, doc.id, space.id)
        
        # Vérifications Ingestion
        assert result["chunks"] == 2  # 2 pages traitées avec succès
        assert result["entities"] >= 2  # Au moins 2 entités insérées (Luméal et Soleal)
        
        # Charger tous les chunks créés pour ce document
        all_chunks = db_session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).all()
        
        # On doit avoir 3 feuilles + 2 résumés de page + 2 Q&As = 7 chunks au total
        assert len(all_chunks) == 7
        
        # Vérifier les types des nouveaux chunks
        summary_chunks = [c for c in all_chunks if not c.is_leaf and c.metadata_json.get("chunk_type") == "page_summary"]
        qa_chunks = [c for c in all_chunks if not c.is_leaf and c.metadata_json.get("chunk_type") == "page_qa"]
        
        assert len(summary_chunks) == 2
        assert len(qa_chunks) == 2
        
        # Vérifier le contenu
        assert any("Résumé de la page 1" in c.content for c in summary_chunks)
        assert any("épaisseur du vitrage Luméal" in c.content for c in qa_chunks)

        # Vérifier que les entités sont créées
        lumeal_entity = db_session.exec(
            select(KnowledgeEntity).where(KnowledgeEntity.name == "Luméal", KnowledgeEntity.space_id == space.id)
        ).first()
        assert lumeal_entity is not None
        
        # Vérifier que l'entité est bien liée à TOUTES les feuilles de la page 1 ET au résumé de page
        relations = db_session.exec(
            select(ChunkEntityRelation).where(
                ChunkEntityRelation.entity_id == lumeal_entity.id,
                ChunkEntityRelation.space_id == space.id
            )
        ).all()
        
        # Relations avec leaf1, leaf2 et summary_chunk de page 1
        assert len(relations) == 3
        chunk_ids_related = {rel.chunk_id for rel in relations}
        assert leaf1.id in chunk_ids_related
        assert leaf2.id in chunk_ids_related

        # 3. Test du Retrieval et de l'Hydratation par page
        # Nous allons simuler une recherche où le résumé de page 1 match
        page_1_summary_chunk = next(c for c in summary_chunks if c.metadata_json.get("page_no") == 1)
        
        # Mock de session.execute pour simuler le retour vectoriel sur les chunks parents
        mock_row = mock.MagicMock(
            similarity_score=0.95,
            node_id=None,
            document_id=doc.id,
            metadata_json=page_1_summary_chunk.metadata_json
        )
        
        # Nous mockons le retour SQL parent de _retrieve_parent_enriched_sql
        with mock.patch("app.services.space_search_service.session.execute") as mock_execute:
            mock_execute.return_value = [mock_row]
            
            # Nous appelons _retrieve_parent_enriched_sql
            retrieved_nodes = _retrieve_parent_enriched_sql(
                session=db_session,
                space_id=space.id,
                user_id=1,
                query_text="Quelle est l'épaisseur du vitrage Luméal ?",
                candidate_k=10
            )
            
            # Vérifier l'hydratation par page
            # La recherche sur le résumé de page 1 doit hydrater et retourner les feuilles de la page 1 (leaf1 et leaf2) !
            assert len(retrieved_nodes) == 2
            retrieved_ids = {n.node.id_ for n in retrieved_nodes}
            assert f"chunk-{leaf1.id}" in retrieved_ids
            assert f"chunk-{leaf2.id}" in retrieved_ids
            assert f"chunk-{leaf3.id}" not in retrieved_ids  # Leaf 3 est sur page 2, ne doit pas être retournée
