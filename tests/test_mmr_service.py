"""
Tests unitaires pour mmr_service (diversité, max_per_parent).

Tests sans dépendance DB (mock de fetch_embeddings).
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from llama_index.core.schema import NodeWithScore, TextNode

from app.services import mmr_service


def test_cosine_similarity():
    """Similarité cosinus entre vecteurs."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [1.0, 0.0, 0.0]
    
    sim = mmr_service._cosine_similarity(vec1, vec2)
    assert abs(sim - 1.0) < 0.01  # Vecteurs identiques → sim=1
    
    vec3 = [0.0, 1.0, 0.0]
    sim2 = mmr_service._cosine_similarity(vec1, vec3)
    assert abs(sim2) < 0.01  # Vecteurs orthogonaux → sim=0


def test_compute_mmr_penalizes_duplicates():
    """MMR pénalise les documents trop similaires aux déjà sélectionnés."""
    # Embedding requête
    query_emb = [1.0, 0.0, 0.0]
    
    # Créer 3 candidats : 2 très similaires entre eux, 1 différent
    nodes = [
        NodeWithScore(
            node=TextNode(id_="chunk-1", text="doc1", metadata={"parent_node_id": "parent1"}),
            score=0.0,
        ),
        NodeWithScore(
            node=TextNode(id_="chunk-2", text="doc2", metadata={"parent_node_id": "parent1"}),
            score=0.0,
        ),
        NodeWithScore(
            node=TextNode(id_="chunk-3", text="doc3", metadata={"parent_node_id": "parent2"}),
            score=0.0,
        ),
    ]
    
    # Embeddings : chunk-1 et chunk-2 sont identiques (redondants)
    # chunk-3 est orthogonal (diversité)
    candidates = [
        (nodes[0], [1.0, 0.0, 0.0]),  # Très similaire à la requête
        (nodes[1], [1.0, 0.0, 0.0]),  # Identique à chunk-1 (doublon)
        (nodes[2], [0.0, 1.0, 0.0]),  # Orthogonal (apporte diversité)
    ]
    
    result = mmr_service.compute_mmr(
        query_emb,
        candidates,
        lambda_=0.5,  # Trade-off équilibré pertinence/diversité
        k=2,
        max_per_parent=5,
    )
    
    assert len(result) == 2
    # chunk-1 devrait être sélectionné en premier (meilleure similarité requête)
    assert result[0][0].node.id_ == "chunk-1"
    # chunk-3 devrait être sélectionné en second (diversité > chunk-2 doublon)
    assert result[1][0].node.id_ == "chunk-3"


def test_compute_mmr_respects_max_per_parent():
    """MMR respecte la contrainte max_per_parent."""
    query_emb = [1.0, 0.0, 0.0]
    
    # 4 candidats tous du même parent
    nodes = [
        NodeWithScore(
            node=TextNode(id_=f"chunk-{i}", text=f"doc{i}", metadata={"parent_node_id": "parent1"}),
            score=0.0,
        )
        for i in range(4)
    ]
    
    # Embeddings légèrement différents
    candidates = [
        (nodes[i], [1.0 - i * 0.1, i * 0.1, 0.0])
        for i in range(4)
    ]
    
    # Limiter à max 2 documents par parent
    result = mmr_service.compute_mmr(
        query_emb,
        candidates,
        lambda_=0.7,
        k=4,  # On demande 4 documents
        max_per_parent=2,  # Mais max 2 par parent
    )
    
    # Devrait s'arrêter à 2 car tous sont du même parent
    assert len(result) <= 2


def test_compute_mmr_distributes_across_parents():
    """MMR favorise la distribution entre plusieurs parents."""
    query_emb = [1.0, 0.0, 0.0]
    
    # 6 candidats : 3 du parent1, 3 du parent2
    nodes = []
    for parent_idx in range(2):
        for doc_idx in range(3):
            nodes.append(
                NodeWithScore(
                    node=TextNode(
                        id_=f"chunk-p{parent_idx}-{doc_idx}",
                        text=f"doc{doc_idx}",
                        metadata={"parent_node_id": f"parent{parent_idx}"}
                    ),
                    score=0.0,
                )
            )
    
    # Embeddings tous similaires à la requête
    candidates = [(n, [1.0, 0.0, 0.0]) for n in nodes]
    
    result = mmr_service.compute_mmr(
        query_emb,
        candidates,
        lambda_=0.5,
        k=4,
        max_per_parent=2,
    )
    
    # Devrait sélectionner 4 documents : 2 de chaque parent
    assert len(result) == 4
    parent_counts = {}
    for nws, _ in result:
        parent_id = nws.node.metadata.get("parent_node_id")
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
    
    # Chaque parent devrait avoir max 2 documents
    for count in parent_counts.values():
        assert count <= 2


def test_compute_mmr_high_lambda_favors_relevance():
    """Lambda élevé (→1) favorise la pertinence vs diversité."""
    query_emb = [1.0, 0.0, 0.0]
    
    nodes = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-2", text="doc2", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-3", text="doc3", metadata={}), score=0.0),
    ]
    
    # chunk-1 et chunk-2 très similaires à la requête, chunk-3 moins
    candidates = [
        (nodes[0], [1.0, 0.0, 0.0]),  # Très pertinent
        (nodes[1], [0.9, 0.1, 0.0]),  # Très pertinent (redondant avec chunk-1)
        (nodes[2], [0.3, 0.7, 0.0]),  # Moins pertinent mais diversifié
    ]
    
    # Lambda=0.95 → favorise fortement la pertinence
    result = mmr_service.compute_mmr(
        query_emb,
        candidates,
        lambda_=0.95,
        k=2,
        max_per_parent=5,
    )
    
    assert len(result) == 2
    # Devrait sélectionner les 2 plus pertinents même s'ils sont redondants
    selected_ids = {nws.node.id_ for nws, _ in result}
    assert "chunk-1" in selected_ids
    assert "chunk-2" in selected_ids  # Redondant mais pertinent


def test_compute_mmr_low_lambda_favors_diversity():
    """Lambda faible (→0) favorise la diversité vs pertinence."""
    query_emb = [1.0, 0.0, 0.0]
    
    nodes = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-2", text="doc2", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-3", text="doc3", metadata={}), score=0.0),
    ]
    
    # chunk-1 et chunk-2 très similaires à la requête, chunk-3 moins mais diversifié
    candidates = [
        (nodes[0], [1.0, 0.0, 0.0]),  # Très pertinent
        (nodes[1], [0.9, 0.1, 0.0]),  # Très pertinent (redondant avec chunk-1)
        (nodes[2], [0.3, 0.7, 0.0]),  # Moins pertinent mais diversifié
    ]
    
    # Lambda=0.2 → favorise fortement la diversité
    result = mmr_service.compute_mmr(
        query_emb,
        candidates,
        lambda_=0.2,
        k=2,
        max_per_parent=5,
    )
    
    assert len(result) == 2
    # Devrait sélectionner chunk-1 (premier) puis chunk-3 (diversité)
    selected_ids = [nws.node.id_ for nws, _ in result]
    assert selected_ids[0] == "chunk-1"
    assert selected_ids[1] == "chunk-3"  # Diversifié > redondant


def test_compute_mmr_empty_query_embedding():
    """Fallback si l'embedding requête est vide."""
    nodes = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-2", text="doc2", metadata={}), score=0.0),
    ]
    
    candidates = [(nodes[0], [1.0, 0.0]), (nodes[1], [0.0, 1.0])]
    
    # Query embedding vide → fallback ordre rerank
    result = mmr_service.compute_mmr(
        [],
        candidates,
        lambda_=0.7,
        k=2,
        max_per_parent=5,
    )
    
    assert len(result) == 2
    # Devrait garder l'ordre original
    assert result[0][0].node.id_ == "chunk-1"
    assert result[1][0].node.id_ == "chunk-2"


def test_compute_mmr_empty_candidates():
    """Gestion correcte de la liste vide."""
    result = mmr_service.compute_mmr(
        [1.0, 0.0, 0.0],
        [],
        lambda_=0.7,
        k=5,
        max_per_parent=5,
    )
    
    assert result == []


def test_compute_mmr_candidates_without_embeddings():
    """Fallback si les candidats n'ont pas d'embeddings."""
    nodes = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-2", text="doc2", metadata={}), score=0.0),
    ]
    
    # Candidats avec embeddings vides
    candidates = [(nodes[0], []), (nodes[1], [])]
    
    result = mmr_service.compute_mmr(
        [1.0, 0.0, 0.0],
        candidates,
        lambda_=0.7,
        k=2,
        max_per_parent=5,
    )
    
    # Fallback ordre rerank
    assert len(result) == 2
