"""
Tests unitaires pour le filtrage RRF dynamique (Option A).
"""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.space_search_service import (
    reciprocal_rank_fusion,
    filter_passages_by_rrf_score,
)


def test_reciprocal_rank_fusion_stores_raw_score():
    """Vérifie que reciprocal_rank_fusion stocke le score RRF brut dans les métadonnées."""
    vector_results = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1"), score=0.9),
        NodeWithScore(node=TextNode(id_="chunk-2", text="doc2"), score=0.8),
    ]
    lexical_results = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="doc1"), score=0.7),
        NodeWithScore(node=TextNode(id_="chunk-3", text="doc3"), score=0.6),
    ]

    fused = reciprocal_rank_fusion(vector_results, lexical_results, k=60, top_n=5)

    assert len(fused) > 0
    # Vérifier la présence du score RRF brut dans les métadonnées
    for nws in fused:
        assert "raw_rrf_score" in nws.node.metadata
        assert isinstance(nws.node.metadata["raw_rrf_score"], float)
        assert nws.node.metadata["raw_rrf_score"] > 0.0


def test_filter_passages_by_rrf_score_one_dominant():
    """Option A : Si un passage est ultra-dominant (gros écart), on ne garde que lui."""
    passages = [
        {"passage": "doc1", "raw_rrf_score": 0.032, "score": 0.9},  # Dominant (ex: 1er en vector + lexical)
        {"passage": "doc2", "raw_rrf_score": 0.016, "score": 0.5},  # Moitié moins bon
        {"passage": "doc3", "raw_rrf_score": 0.010, "score": 0.3},
    ]

    # Seuil = 0.032 * 0.70 = 0.0224. Seul doc1 est >= 0.0224.
    filtered = filter_passages_by_rrf_score(
        passages,
        enabled=True,
        min_k=1,
        max_k=10,
        factor=0.70,
    )

    assert len(filtered) == 1
    assert filtered[0]["passage"] == "doc1"


def test_filter_passages_by_rrf_score_multiple_relevant():
    """Option A : Si plusieurs passages ont des scores proches, on les garde tous."""
    passages = [
        {"passage": "doc1", "raw_rrf_score": 0.032, "score": 0.9},
        {"passage": "doc2", "raw_rrf_score": 0.030, "score": 0.8},
        {"passage": "doc3", "raw_rrf_score": 0.028, "score": 0.7},
        {"passage": "doc4", "raw_rrf_score": 0.016, "score": 0.4},
    ]

    # Seuil = 0.032 * 0.70 = 0.0224. doc1, doc2, doc3 sont >= 0.0224.
    filtered = filter_passages_by_rrf_score(
        passages,
        enabled=True,
        min_k=1,
        max_k=10,
        factor=0.70,
    )

    assert len(filtered) == 3
    assert [p["passage"] for p in filtered] == ["doc1", "doc2", "doc3"]


def test_filter_passages_by_rrf_score_respects_min_k():
    """Option A : Respecte la borne min_k même si le score est en dessous du seuil."""
    passages = [
        {"passage": "doc1", "raw_rrf_score": 0.032, "score": 0.9},
        {"passage": "doc2", "raw_rrf_score": 0.010, "score": 0.3},
    ]

    # Seuil = 0.032 * 0.70 = 0.0224.
    # Si min_k = 2, on doit quand même garder doc2.
    filtered = filter_passages_by_rrf_score(
        passages,
        enabled=True,
        min_k=2,
        max_k=10,
        factor=0.70,
    )

    assert len(filtered) == 2


def test_filter_passages_by_rrf_score_respects_max_k():
    """Option A : Respecte la borne max_k même si plus de documents dépassent le seuil."""
    passages = [
        {"passage": f"doc{i}", "raw_rrf_score": 0.030, "score": 0.8}
        for i in range(15)
    ]

    # Tous dépassent le seuil (car tous égaux à 0.030).
    # Si max_k = 5, on ne doit en garder que 5.
    filtered = filter_passages_by_rrf_score(
        passages,
        enabled=True,
        min_k=1,
        max_k=5,
        factor=0.70,
    )

    assert len(filtered) == 5


def test_filter_passages_by_rrf_score_disabled():
    """Vérifie que si enabled=False, le filtrage dynamique ne fait rien."""
    passages = [
        {"passage": "doc1", "raw_rrf_score": 0.032, "score": 0.9},
        {"passage": "doc2", "raw_rrf_score": 0.016, "score": 0.5},
    ]

    filtered = filter_passages_by_rrf_score(
        passages,
        enabled=False,
    )

    assert len(filtered) == 2


def test_filter_passages_by_rrf_score_fallback_to_normalized_score():
    """Vérifie le repli sur le score normalisé ('score') si le score brut est absent."""
    passages = [
        {"passage": "doc1", "score": 0.9},  # Pas de raw_rrf_score
        {"passage": "doc2", "score": 0.8},
        {"passage": "doc3", "score": 0.4},
    ]

    # Seuil = 0.9 * 0.70 = 0.63. doc1 et doc2 sont >= 0.63.
    filtered = filter_passages_by_rrf_score(
        passages,
        enabled=True,
        min_k=1,
        max_k=10,
        factor=0.70,
    )

    assert len(filtered) == 2
    assert [p["passage"] for p in filtered] == ["doc1", "doc2"]
