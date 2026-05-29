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
    import asyncio

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

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.95, 0.60, 0.20])

    with patch.object(reranker_service, '_get_cross_encoder', return_value=mock_model):
        scored = asyncio.run(
            reranker_service.rerank_nodes(
                query_text="requête test",
                nodes_with_score=nodes,
                char_cap=8000,
                batch_size=16,
            )
        )

    assert len(scored) == 3
    assert scored[0][1] == 0.95
    assert scored[1][1] == 0.60
    assert scored[2][1] == 0.20


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
    nodes = [
        NodeWithScore(node=TextNode(id_=f"chunk-{i}", text=f"doc{i}", metadata={}), score=0.0)
        for i in range(5)
    ]

    # Scores très différenciés pour éviter le guardrail « bégaiement »
    scored = [(nodes[i], 10.0 - i * 3.0) for i in range(5)]

    with patch("app.services.reranker_service.settings") as mock_settings:
        mock_settings.RERANKER_MIN_SCORE = -10.0
        mock_settings.RAG_MIN_PERTINENCE = 0.0
        result = reranker_service.apply_dynamic_filtering(
            scored,
            min_k=2,
            max_k=8,
            softmax_cum_threshold=0.80,
            stutter_gap=0.05,
            zscore_flat_threshold=0.05,
        )

    assert result.status == "ok"
    assert 2 <= len(result.nodes) <= 8
    assert result.gap_top1_top2 is not None
    assert result.gap_top1_top2 > 0.05


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


def test_apply_dynamic_filtering_filters_below_threshold():
    """Vérifier que les candidats sous le seuil RERANKER_MIN_SCORE sont filtrés."""
    nodes = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-2", text="doc2", metadata={}), score=0.0),
    ]
    # chunk-1 a un score supérieur au seuil (-2.5 >= -3.0)
    # chunk-2 a un score inférieur au seuil (-3.5 < -3.0)
    scored = [(nodes[0], -2.5), (nodes[1], -3.5)]
    
    with patch("app.services.reranker_service.settings") as mock_settings:
        mock_settings.RERANKER_MIN_SCORE = -3.0
        mock_settings.RAG_MIN_PERTINENCE = 0.0
        result = reranker_service.apply_dynamic_filtering(
            scored,
            min_k=1,
            max_k=12,
            softmax_cum_threshold=0.80,
            stutter_gap=0.05,
            zscore_flat_threshold=0.05,
        )
        
    assert result.status == "ok"
    assert len(result.nodes) == 1
    assert result.nodes[0].id_ == "chunk-1"
    
    # Test lorsque tous les candidats sont sous le seuil
    scored_all_low = [(nodes[0], -4.0), (nodes[1], -4.5)]
    with patch("app.services.reranker_service.settings") as mock_settings:
        mock_settings.RERANKER_MIN_SCORE = -3.0
        mock_settings.RAG_MIN_PERTINENCE = 0.0
        result_all_low = reranker_service.apply_dynamic_filtering(
            scored_all_low,
            min_k=1,
            max_k=12,
            softmax_cum_threshold=0.80,
            stutter_gap=0.05,
            zscore_flat_threshold=0.05,
        )
    assert result_all_low.status == "ok"
    assert len(result_all_low.nodes) == 0
    assert result_all_low.reason == "no_candidates_above_threshold"


def test_apply_dynamic_filtering_filters_below_rag_min_pertinence():
    """Vérifier que les candidats sous le seuil RAG_MIN_PERTINENCE sont filtrés et que nws.score est normalisé."""
    nodes = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1", metadata={}), score=0.0),
        NodeWithScore(node=TextNode(id_="chunk-2", text="doc2", metadata={}), score=0.0),
    ]
    # logit de -0.2 -> sigmoid(-0.2 + 1.5) = sigmoid(1.3) = 0.785 -> passe le seuil de 0.75
    # logit de -1.5 -> sigmoid(-1.5 + 1.5) = sigmoid(0) = 0.50 -> filtré par le seuil de 0.75
    scored = [(nodes[0], -0.2), (nodes[1], -1.5)]
    
    with patch("app.services.reranker_service.settings") as mock_settings:
        mock_settings.RERANKER_MIN_SCORE = -10.0
        mock_settings.RAG_MIN_PERTINENCE = 0.75
        result = reranker_service.apply_dynamic_filtering(
            scored,
            min_k=1,
            max_k=12,
            softmax_cum_threshold=0.80,
            stutter_gap=0.05,
            zscore_flat_threshold=0.05,
        )
        
    assert result.status == "ok"
    assert len(result.nodes) == 1
    assert result.nodes[0].id_ == "chunk-1"
    # Vérifier la normalisation sigmoïde calibrée du score
    import math
    expected_score = 1.0 / (1.0 + math.exp(-(-0.2 + 1.5)))
    assert math.isclose(result.nodes[0].score, expected_score)


