"""Tests qualité retrieval : classification requête et métriques eval."""

from pathlib import Path

import pytest

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


def test_space_rerank_min_score_constant():
    from app.services.space_search_service import RERANK_MIN_SCORE

    assert RERANK_MIN_SCORE == 0.30


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
