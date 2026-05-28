"""
Tests unitaires pour le filtrage RRF dynamique (Option A).
"""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.space_search_service import (
    reciprocal_rank_fusion,
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
