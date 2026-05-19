"""Tests qualité retrieval : classification requête, filtre hybride strict, métriques eval."""

from pathlib import Path

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "retrieval_eval_dataset.json"


def test_classify_query_type_factual():
    from app.services.space_search_service import classify_query_type, is_factual_query

    assert classify_query_type("Quelle est la hauteur maximale ?") == "factual"
    assert is_factual_query("Quel est le coefficient Uw ?") is True


def test_classify_query_with_product_ref_as_factual():
    from app.services.space_search_service import is_factual_query

    q = (
        "Dans quels cas précis est-il nécessaire de rajouter systématiquement "
        "la cale TGA3817 au droit de chaque roulette ?"
    )
    assert is_factual_query(q) is True


def test_classify_query_type_comparative():
    from app.services.space_search_service import classify_query_type

    assert classify_query_type("Comparaison entre A et B") == "comparative"
    assert classify_query_type("Différence entre Perform 70 et 76") == "comparative"


def test_classify_query_type_exploratory_long():
    from app.services.space_search_service import classify_query_type

    long_q = " ".join(["mot"] * 20)
    assert classify_query_type(long_q) == "exploratory"


def test_filter_hybrid_rejects_lexical_only_without_vector():
    from app.services.space_search_service import _filter_hybrid_candidates

    lexical_only = NodeWithScore(
        node=TextNode(
            id_="chunk-1",
            text="cale roulette installation générale",
            metadata={
                "lexical_norm": 0.15,
                "kag_norm": 0.0,
                "vector_rrf": 0.0,
            },
        ),
        score=0.05,
    )
    assert _filter_hybrid_candidates([lexical_only]) == []


def test_filter_hybrid_rejects_lexical_only_noise():
    from app.services.space_search_service import (
        MIN_VECTOR_SIMILARITY_THRESHOLD,
        RRF_MIN_SCORE,
        _filter_hybrid_candidates,
    )

    weak = NodeWithScore(
        node=TextNode(
            id_="chunk-1",
            text="texte générique installation",
            metadata={
                "vector_similarity": MIN_VECTOR_SIMILARITY_THRESHOLD - 0.1,
                "lexical_norm": 0.2,
                "kag_norm": 0.0,
                "vector_rrf": 0.01,
                "hybrid_score": RRF_MIN_SCORE - 0.01,
            },
        ),
        score=RRF_MIN_SCORE - 0.01,
    )
    assert _filter_hybrid_candidates([weak]) == []


def test_safe_fallback_prefers_vector_not_full_fusion():
    from app.services.space_search_service import (
        MIN_VECTOR_SQL_PREFILTER,
        _safe_fallback_after_empty_hybrid_filter,
    )

    fused = [
        NodeWithScore(
            node=TextNode(id_="chunk-a", text="bruit lexical", metadata={"lexical_norm": 0.2}),
            score=0.2,
        ),
        NodeWithScore(
            node=TextNode(
                id_="chunk-b",
                text="TGA3817 cale",
                metadata={"vector_similarity": MIN_VECTOR_SQL_PREFILTER + 0.05},
            ),
            score=0.05,
        ),
    ]
    vector_only = [
        NodeWithScore(
            node=TextNode(id_="chunk-b", text="TGA3817"),
            score=MIN_VECTOR_SQL_PREFILTER + 0.1,
        ),
    ]
    out = _safe_fallback_after_empty_hybrid_filter(fused, vector_only, k=6)
    assert len(out) <= 15
    assert any(getattr(n.node, "id_", "") == "chunk-b" for n in out)


def test_filter_hybrid_accepts_strong_vector_and_hybrid():
    from app.services.space_search_service import (
        MIN_VECTOR_SIMILARITY_THRESHOLD,
        RRF_MIN_SCORE,
        _filter_hybrid_candidates,
    )

    good = NodeWithScore(
        node=TextNode(
            id_="chunk-2",
            text="Hauteur max 2400 mm",
            metadata={
                "vector_similarity": MIN_VECTOR_SIMILARITY_THRESHOLD + 0.05,
                "vector_rrf": 0.12,
                "hybrid_score": RRF_MIN_SCORE + 0.02,
            },
        ),
        score=RRF_MIN_SCORE + 0.02,
    )
    out = _filter_hybrid_candidates([good])
    assert len(out) == 1


def test_hybrid_fuse_lexical_weight_halved():
    from app.services.space_search_service import RRF_LEXICAL_LIST_WEIGHT, _hybrid_fuse_candidates

    assert RRF_LEXICAL_LIST_WEIGHT == 0.5
    l_only = [
        NodeWithScore(node=TextNode(id_="chunk-9", text="x"), score=1.0),
    ]
    merged = _hybrid_fuse_candidates([], l_only, [])
    assert len(merged) == 1
    meta = merged[0].node.metadata or {}
    assert meta.get("lexical_norm", 0) < 0.02


def test_precision_at_k_and_mrr_helpers():
    from app.services.retrieval_eval_utils import mean_reciprocal_rank, precision_at_k

    retrieved = [
        {"chunk_id": 1, "passage_raw": "hauteur max"},
        {"chunk_id": 2, "passage_raw": "garantie"},
        {"chunk_id": 3, "passage_raw": "pose"},
    ]
    relevant_ids = {1}

    assert precision_at_k(retrieved, relevant_ids, k=3) == pytest.approx(1 / 3)
    assert mean_reciprocal_rank(retrieved, relevant_ids) == pytest.approx(1.0)

    no_hit = [{"chunk_id": 99, "passage_raw": "autre"}]
    assert mean_reciprocal_rank(no_hit, relevant_ids) == 0.0


def test_eval_dataset_fixture_loads():
    from app.services.retrieval_eval_utils import load_eval_dataset

    cases = load_eval_dataset(FIXTURE_PATH)
    assert len(cases) >= 2
    assert cases[0]["query_type"] in ("factual", "comparative", "exploratory")
