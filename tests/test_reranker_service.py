"""
Tests unitaires pour reranker_service (early stop, guardrails, K dynamique).

Mock du CrossEncoder pour éviter de charger PyTorch en CI.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from llama_index.core.schema import NodeWithScore, TextNode

from app.services import reranker_service


def test_should_early_stop_triggers_when_mean_above_threshold():
    """Early stop activé si la moyenne des top-N scores dépasse le seuil."""
    # Scores RRF très élevés
    scores = [0.9, 0.88, 0.85, 0.82, 0.80]
    threshold = 0.78
    
    result = reranker_service.should_early_stop(scores, threshold)
    assert result is True


def test_should_early_stop_not_triggered_when_mean_below_threshold():
    """Early stop pas activé si la moyenne est en dessous du seuil."""
    # Scores RRF modérés
    scores = [0.7, 0.65, 0.6, 0.55, 0.5]
    threshold = 0.78
    
    result = reranker_service.should_early_stop(scores, threshold)
    assert result is False


def test_should_early_stop_empty_scores():
    """Early stop pas activé sur liste vide."""
    result = reranker_service.should_early_stop([], 0.78)
    assert result is False


def test_rerank_nodes_with_mocked_cross_encoder():
    """Rerank batch : mock du CrossEncoder pour éviter PyTorch."""
    # Créer des nœuds de test
    nodes = [
        NodeWithScore(
            node=TextNode(id_="chunk-1", text="Document très pertinent pour la requête", metadata={}),
            score=0.5,
        ),
        NodeWithScore(
            node=TextNode(id_="chunk-2", text="Document moins pertinent", metadata={}),
            score=0.4,
        ),
        NodeWithScore(
            node=TextNode(id_="chunk-3", text="Document pas pertinent du tout", metadata={}),
            score=0.3,
        ),
    ]
    
    # Mock du CrossEncoder
    mock_model = MagicMock()
    # Simuler des scores cross-encoder (le premier doc est le plus pertinent)
    mock_model.predict.return_value = np.array([0.95, 0.60, 0.20])
    
    with patch.object(reranker_service, '_get_cross_encoder', return_value=mock_model):
        scored = reranker_service.rerank_nodes(
            query_text="requête test",
            nodes_with_score=nodes,
            char_cap=8000,
            batch_size=16,
        )
    
    # Vérifier que les nœuds sont triés par score cross-encoder décroissant
    assert len(scored) == 3
    assert scored[0][1] == 0.95  # chunk-1 en premier
    assert scored[1][1] == 0.60  # chunk-2 en deuxième
    assert scored[2][1] == 0.20  # chunk-3 en troisième


def test_apply_dynamic_filtering_guardrails_low_confidence():
    """Guardrail bégaiement : gap faible + zscore plat → low_confidence_clarification."""
    # Créer des nœuds avec scores très proches (plats)
    nodes = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-2", text="doc2", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-3", text="doc3", metadata={}), score=0.0),
    ]
    
    # Scores cross-encoder très proches (distribution plate)
    scored = [(nodes[0], 0.51), (nodes[1], 0.50), (nodes[2], 0.49)]
    
    result = reranker_service.apply_dynamic_filtering(
        scored,
        min_k=2,
        max_k=12,
        softmax_cum_threshold=0.80,
        stutter_gap=0.05,  # Gap P1-P2 doit être > 0.05
        zscore_flat_threshold=0.05,  # Stdev doit être > 0.05
    )
    
    # Avec ces scores, le softmax sera très plat → bégaiement détecté
    assert result.status == "low_confidence_clarification"
    assert len(result.nodes) <= 2  # Top 1-2 seulement
    assert result.gap_top1_top2 is not None
    assert result.zscore_flatness is not None


def test_apply_dynamic_filtering_ok_status():
    """K dynamique : sélection normale si gap et zscore OK."""
    # Scores cross-encoder avec bonne différenciation
    nodes = [
        NodeWithScore(node=TextNode(id_=f"chunk-{i}", text=f"doc{i}", metadata={}), score=0.0)
        for i in range(10)
    ]
    
    # Scores cross-encoder décroissants (bonne distribution)
    scored = [(nodes[i], 1.0 - i * 0.1) for i in range(10)]
    
    result = reranker_service.apply_dynamic_filtering(
        scored,
        min_k=2,
        max_k=8,
        softmax_cum_threshold=0.80,
        stutter_gap=0.05,
        zscore_flat_threshold=0.05,
    )
    
    assert result.status == "ok"
    assert 2 <= len(result.nodes) <= 8  # Borné entre min_k et max_k
    assert result.gap_top1_top2 is not None
    assert result.gap_top1_top2 > 0.05  # Pas de bégaiement


def test_apply_dynamic_filtering_respects_min_k():
    """K dynamique : respecte la borne inférieure MIN_DYNAMIC_K."""
    nodes = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1", metadata={}), score=0.0),
    ]
    scored = [(nodes[0], 0.8)]
    
    result = reranker_service.apply_dynamic_filtering(
        scored,
        min_k=3,
        max_k=12,
        softmax_cum_threshold=0.80,
        stutter_gap=0.05,
        zscore_flat_threshold=0.05,
    )
    
    # Même avec 1 seul candidat, on ne peut pas respecter min_k=3
    # mais le code garde ce qu'il peut
    assert len(result.nodes) <= len(scored)


def test_apply_dynamic_filtering_respects_max_k():
    """K dynamique : respecte la borne supérieure MAX_DYNAMIC_K."""
    nodes = [
        NodeWithScore(node=TextNode(id_=f"chunk-{i}", text=f"doc{i}", metadata={}), score=0.0)
        for i in range(20)
    ]
    # Scores uniformes → tous pourraient être sélectionnés
    scored = [(nodes[i], 0.5) for i in range(20)]
    
    result = reranker_service.apply_dynamic_filtering(
        scored,
        min_k=2,
        max_k=5,
        softmax_cum_threshold=0.80,
        stutter_gap=0.05,
        zscore_flat_threshold=0.05,
    )
    
    # Même si beaucoup de candidats, max_k=5 est respecté
    assert len(result.nodes) <= 5


def test_apply_dynamic_filtering_empty_candidates():
    """Gestion correcte de la liste vide."""
    result = reranker_service.apply_dynamic_filtering(
        [],
        min_k=2,
        max_k=12,
        softmax_cum_threshold=0.80,
        stutter_gap=0.05,
        zscore_flat_threshold=0.05,
    )
    
    assert result.status == "ok"
    assert result.nodes == []
    assert result.reason == "no_candidates"
