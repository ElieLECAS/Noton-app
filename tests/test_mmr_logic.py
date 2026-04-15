import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from llama_index.core.schema import NodeWithScore, TextNode
from app.services.space_search_service import (
    _compute_mmr_with_parent_constraint,
    _fetch_embeddings_for_chunks,
    MMR_K,
    MMR_LAMBDA
)

def test_fetch_embeddings_for_chunks_sql_execution():
    """Vérifie que la requête SQL ANY(:ids) est bien formée."""
    session = MagicMock()
    chunk_ids = [101, 102, 103]
    
    # Mock du résultat de session.execute
    mock_result = [
        MagicMock(id=101, embedding=[0.1, 0.2]),
        MagicMock(id=102, embedding=[0.3, 0.4])
    ]
    session.execute.return_value = mock_result
    
    embeddings = _fetch_embeddings_for_chunks(session, chunk_ids)
    
    assert len(embeddings) == 2
    assert 101 in embeddings
    assert 102 in embeddings
    assert isinstance(embeddings[101], np.ndarray)
    assert np.allclose(embeddings[101], [0.1, 0.2])
    
    # Vérifie que ANY(:ids) a été utilisé
    args, _ = session.execute.call_args
    assert "ANY(:ids)" in str(args[0])
    assert args[1]["ids"] == chunk_ids

def test_mmr_selection_diversifies_redundant_results():
    """
    Vérifie que la MMR préfère un résultat différent à un résultat redondant.
    """
    query_emb = np.array([1.0, 0.0])
    
    # Candidat 1 : Très pertinent
    c1 = NodeWithScore(node=TextNode(id_="chunk-1", text="Docs sur le point A", metadata={"parent_node_id": "P1"}), score=0.9)
    # Candidat 2 : Très pertinent mais identique à C1 (redondant)
    c2 = NodeWithScore(node=TextNode(id_="chunk-2", text="Docs sur le point A bis", metadata={"parent_node_id": "P2"}), score=0.85)
    # Candidat 3 : Moins pertinent individuellement mais différent (diversité)
    c3 = NodeWithScore(node=TextNode(id_="chunk-3", text="Docs sur le point B", metadata={"parent_node_id": "P3"}), score=0.7)
    
    candidates = [c1, c2, c3]
    embeddings = {
        1: np.array([1.0, 0.0]),   # Parfaitement aligné (Query)
        2: np.array([0.99, 0.01]), # Quasi-identique à C1 (Redondant)
        3: np.array([0.5, 0.8]),   # Différent de C1 mais toujours pertinent
    }
    
    # On demande 2 passages. Sans MMR, on aurait C1 et C2. 
    # Avec MMR, on veut C1 et C3.
    selected = _compute_mmr_with_parent_constraint(
        query_embedding=query_emb,
        candidates=candidates,
        candidate_embeddings=embeddings,
        target_k=2,
        lambda_param=0.3  # Plus de poids à la diversité pour le test
    )
    
    assert len(selected) == 2
    selected_ids = [n.node.id_ for n in selected]
    assert "chunk-1" in selected_ids
    assert "chunk-3" in selected_ids
    assert "chunk-2" not in selected_ids

def test_mmr_respects_strict_parent_constraint():
    """Vérifie qu'un seul chunk par parent est sélectionné."""
    query_emb = np.array([1.0, 1.0])
    
    candidates = [
        NodeWithScore(node=TextNode(id_="chunk-1", metadata={"parent_node_id": "PARENT_A"}), score=0.9),
        NodeWithScore(node=TextNode(id_="chunk-2", metadata={"parent_node_id": "PARENT_A"}), score=0.8),
        NodeWithScore(node=TextNode(id_="chunk-3", metadata={"parent_node_id": "PARENT_B"}), score=0.7),
    ]
    
    embeddings = {
        1: np.array([1.0, 1.0]),
        2: np.array([1.0, 0.9]),
        3: np.array([0.1, 0.1]),
    }
    
    selected = _compute_mmr_with_parent_constraint(
        query_embedding=query_emb,
        candidates=candidates,
        candidate_embeddings=embeddings,
        target_k=5, # Plus que le nombre de parents
        lambda_param=0.5
    )
    
    # On ne doit avoir que 2 résultats car il n'y a que 2 parents uniques
    assert len(selected) == 2
    parent_ids = [n.node.metadata["parent_node_id"] for n in selected]
    assert len(set(parent_ids)) == 2
    assert "PARENT_A" in parent_ids
    assert "PARENT_B" in parent_ids

def test_mmr_fallback_missing_embeddings():
    """Vérifie que le système ne plante pas si des embeddings manquent."""
    query_emb = np.array([1.0, 0.0])
    candidates = [
        NodeWithScore(node=TextNode(id_="chunk-1"), score=0.9),
        NodeWithScore(node=TextNode(id_="chunk-2"), score=0.8),
    ]
    
    # Aucun embedding fourni
    selected = _compute_mmr_with_parent_constraint(
        query_embedding=query_emb,
        candidates=candidates,
        candidate_embeddings={},
        target_k=1
    )
    
    # Doit faire un fallback sur le premier candidat du pool
    assert len(selected) == 1
    assert selected[0].node.id_ == "chunk-1"

def test_mmr_lambda_influence():
    """Vérifie que lambda=1.0 se comporte comme un tri simple."""
    query_emb = np.array([1.0, 0.0])
    c1 = NodeWithScore(node=TextNode(id_="chunk-1"), score=0.9)
    c2 = NodeWithScore(node=TextNode(id_="chunk-2"), score=0.8)
    
    embeddings = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
    }
    
    selected = _compute_mmr_with_parent_constraint(
        query_embedding=query_emb,
        candidates=[c1, c2],
        candidate_embeddings=embeddings,
        target_k=2,
        lambda_param=1.0 # 100% pertinence, 0% diversité
    )
    
    assert selected[0].node.id_ == "chunk-1"
    assert selected[1].node.id_ == "chunk-2"
