"""Tests unitaires pipeline retrieval lean (sans DB)."""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from app.services.retrieval_pipeline import (
    REF_EXACT_BOOST,
    RetrievalCandidate,
    adaptive_gate,
    adaptive_top_n,
    analyze_query,
    compute_composite_score,
    merge_vector_and_exact,
    smart_parent_or_leaf,
)


def test_analyze_query_extracts_refs():
    qa = analyze_query("cale TGA3817 et référence T141021")
    assert "TGA3817" in qa.product_refs
    assert "T141021" in qa.product_refs
    assert qa.intent == "factual"


def test_analyze_query_dimensions():
    qa = analyze_query("profil 29-32 mm pour menuiserie 55 mm")
    assert len(qa.dimensions) >= 1


def test_analyze_query_comparative_needs_one_hop_with_pivots():
    qa = analyze_query(
        "Différence entre Perform 70 et Perform 76",
        pivot_entities=["perform 70", "perform 76"],
    )
    assert qa.intent == "comparative"
    assert qa.needs_one_hop is True


def test_analyze_query_comparative_without_two_pivots_no_hop():
    qa = analyze_query("Comparaison entre A et B", pivot_entities=["alpha"])
    assert qa.intent == "comparative"
    assert qa.needs_one_hop is False


def test_adaptive_top_n_returns_one_when_clear_winner():
    nodes = [
        NodeWithScore(node=TextNode(id_="a", text="a"), score=5.0),
        NodeWithScore(node=TextNode(id_="b", text="b"), score=-2.0),
        NodeWithScore(node=TextNode(id_="c", text="c"), score=-3.0),
    ]
    kept = adaptive_top_n(nodes, k_max=6, gap_margin=0.15)
    assert len(kept) == 1


def test_adaptive_top_n_returns_multiple_on_close_scores():
    nodes = [
        NodeWithScore(node=TextNode(id_="a", text="a"), score=2.0),
        NodeWithScore(node=TextNode(id_="b", text="b"), score=1.95),
        NodeWithScore(node=TextNode(id_="c", text="c"), score=1.90),
    ]
    kept = adaptive_top_n(nodes, k_max=6, gap_margin=0.15)
    assert len(kept) >= 2


def test_smart_parent_keeps_leaf_when_long():
    long_text = "x" * 250
    leaf = TextNode(
        id_="chunk-1",
        text=long_text,
        metadata={"content_type": "paragraph", "parent_node_id": "parent-1"},
    )
    parent = TextNode(id_="parent-1", text="section entière", metadata={})
    out = smart_parent_or_leaf(leaf, {"parent-1": parent})
    assert out is leaf


def test_smart_parent_promotes_table_row():
    leaf = TextNode(
        id_="chunk-2",
        text="cellule courte",
        metadata={"content_type": "table_row", "parent_node_id": "parent-2"},
    )
    parent = TextNode(id_="parent-2", text="tableau complet", metadata={})
    out = smart_parent_or_leaf(leaf, {"parent-2": parent})
    assert out is parent


def test_smart_parent_promotes_short_leaf():
    leaf = TextNode(
        id_="chunk-3",
        text="court",
        metadata={"content_type": "paragraph", "parent_node_id": "parent-3"},
    )
    parent = TextNode(id_="parent-3", text="section", metadata={})
    out = smart_parent_or_leaf(leaf, {"parent-3": parent})
    assert out is parent


def test_exact_ref_match_boost_outranks_vector_only():
    qa = analyze_query("TGA3817")
    vector_only = [
        NodeWithScore(
            node=TextNode(id_="chunk-1", text="autre contenu", metadata={"document_title": "Doc"}),
            score=0.72,
        ),
    ]
    exact = [
        NodeWithScore(
            node=TextNode(
                id_="chunk-2",
                text="Référence cale TGA3817 obligatoire",
                metadata={"document_title": "Doc"},
            ),
            score=0.85,
        ),
    ]
    merged = merge_vector_and_exact(vector_only, exact, qa)
    assert len(merged) == 2
    assert merged[0].chunk_id == 2
    assert merged[0].ref_exact_match is True
    assert merged[0].composite_score >= merged[1].composite_score


def test_compute_composite_score_ref_boost():
    qa = analyze_query("TGA3817")
    base = compute_composite_score(0.5, qa)
    boosted = compute_composite_score(0.5, qa, ref_exact=True)
    assert boosted - base == pytest.approx(REF_EXACT_BOOST)


def test_adaptive_gate_keeps_near_top():
    cands = [
        RetrievalCandidate(
            node=TextNode(id_="chunk-1", text="a"),
            chunk_id=1,
            vector_similarity=0.8,
            composite_score=0.8,
        ),
        RetrievalCandidate(
            node=TextNode(id_="chunk-2", text="b"),
            chunk_id=2,
            vector_similarity=0.6,
            composite_score=0.6,
        ),
        RetrievalCandidate(
            node=TextNode(id_="chunk-3", text="c"),
            chunk_id=3,
            vector_similarity=0.2,
            composite_score=0.2,
        ),
    ]
    gated = adaptive_gate(cands, tau=0.7)
    ids = {c.chunk_id for c in gated}
    assert 1 in ids
    assert 2 in ids
    assert 3 not in ids
