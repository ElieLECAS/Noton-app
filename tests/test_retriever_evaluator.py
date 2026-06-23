import pytest
from app.services.retriever_evaluator import (
    match_page,
    compute_context_precision,
    compute_context_recall,
    compute_mrr,
    build_question_eval_result,
)

def test_match_page():
    # Match exact
    assert match_page("Notice Perform 70", 12, [{"document_title": "Notice Perform 70", "pages": [12, 13]}]) is True
    # Substring match (casse insensible)
    assert match_page("Notice de pose Perform 70 - Rev3", 12, [{"document_title": "Notice Perform 70", "pages": [12, 13]}]) is True
    assert match_page("Notice Perform 70", 12, [{"document_title": "Notice de pose Perform 70 - Rev3", "pages": [12, 13]}]) is True
    # Page ne correspondant pas
    assert match_page("Notice Perform 70", 14, [{"document_title": "Notice Perform 70", "pages": [12, 13]}]) is False
    # Document ne correspondant pas
    assert match_page("Notice Perform 76", 12, [{"document_title": "Notice Perform 70", "pages": [12, 13]}]) is False

def test_compute_context_precision():
    expected = [{"document_title": "Notice Perform 70", "pages": [12, 13]}]
    
    # Précision parfaite (les pages utiles remontent en premier)
    retrieved = [("Notice Perform 70", 12), ("Notice Perform 70", 13)]
    assert compute_context_precision(retrieved, expected) == 1.0

    # 1ère correcte, 2ème bruit
    retrieved = [("Notice Perform 70", 12), ("Notice Perform 76", 1)]
    assert compute_context_precision(retrieved, expected) == 1.0

    # 1ère bruit, 2ème correcte
    retrieved = [("Notice Perform 76", 1), ("Notice Perform 70", 12)]
    # Hits: [0, 1]. Precision @ 1 est 0, Precision @ 2 est 1/2 = 0.5. Total retrieved = 1.
    assert compute_context_precision(retrieved, expected) == 0.5

    # Aucun résultat utile
    retrieved = [("Notice Perform 76", 1)]
    assert compute_context_precision(retrieved, expected) == 0.0

def test_compute_context_recall():
    expected = [{"document_title": "Notice Perform 70", "pages": [12, 13]}]
    
    # Tous récupérés
    retrieved = [("Notice Perform 70", 12), ("Notice Perform 70", 13)]
    assert compute_context_recall(retrieved, expected) == 1.0

    # Partiellement récupéré
    retrieved = [("Notice Perform 70", 12), ("Notice Perform 76", 1)]
    assert compute_context_recall(retrieved, expected) == 0.5

    # Aucun récupéré
    retrieved = [("Notice Perform 76", 1)]
    assert compute_context_recall(retrieved, expected) == 0.0

def test_compute_mrr():
    expected = [{"document_title": "Notice Perform 70", "pages": [12]}]
    
    # Rang 1
    retrieved = [("Notice Perform 70", 12)]
    assert compute_mrr(retrieved, expected) == 1.0

    # Rang 2
    retrieved = [("Notice Perform 76", 1), ("Notice Perform 70", 12)]
    assert compute_mrr(retrieved, expected) == 0.5

    # Non trouvé
    retrieved = [("Notice Perform 76", 1)]
    assert compute_mrr(retrieved, expected) == 0.0


def test_build_question_eval_result_post_only():
    expected = [{"document_title": "Notice Perform 70", "pages": [12]}]
    passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.85},
    ]

    result = build_question_eval_result(
        question="Quelle est la profondeur d'installation ?",
        q_type="mono-document",
        expected_pages=expected,
        passages=passages,
    )

    assert result["metrics"]["context_precision"] == 1.0
    assert result["metrics"]["context_recall"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
    assert len(result["analysis"]["hits"]) == 1
    assert len(result["analysis"]["misses"]) == 0
    assert "metrics_colpali" not in result
    assert "rerank_delta" not in result


def test_build_question_eval_result_with_colpali_and_delta():
    expected = [{"document_title": "Notice Perform 70", "pages": [12]}]
    colpali_passages = [
        {"document_title": "Notice Perform 76", "page_no": 1, "score": 0.92},
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.81},
    ]
    post_passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.81, "rerank_score": 4.5},
    ]

    result = build_question_eval_result(
        question="Quelle est la profondeur d'installation ?",
        q_type="mono-document",
        expected_pages=expected,
        passages=post_passages,
        colpali_passages=colpali_passages,
        vision_rerank_enabled=True,
    )

    assert result["metrics"]["context_precision"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
    assert result["metrics_colpali"]["context_precision"] == 0.5
    assert result["metrics_colpali"]["mrr"] == 0.5
    assert result["rerank_delta"]["precision"] == 0.5
    assert result["rerank_delta"]["mrr"] == 0.5
    assert result["vision_rerank_enabled"] is True
    assert len(result["analysis_colpali"]["noise"]) == 1
    assert len(result["analysis"]["noise"]) == 0


def test_filter_colpali_pages_dynamic():
    from app.services.page_retrieval_service import UnifiedPageHit, filter_colpali_pages_dynamic

    hits = [
        UnifiedPageHit(document_id=1, page_no=1, colpali_score=0.85, retrieval_sources=["colpali"]),
        UnifiedPageHit(document_id=1, page_no=2, colpali_score=0.78, retrieval_sources=["colpali"]),
        UnifiedPageHit(document_id=1, page_no=3, colpali_score=0.72, retrieval_sources=["colpali"]),
        UnifiedPageHit(document_id=1, page_no=4, colpali_score=0.25, retrieval_sources=["colpali"]),
    ]
    filtered = filter_colpali_pages_dynamic(hits, min_threshold=0.30, relative_margin=0.10)
    pages = {h.page_no for h in filtered}
    assert pages == {1, 2}
    assert 4 not in pages


def test_build_question_eval_result_with_post_rrf():
    expected = [{"document_title": "Notice Perform 70", "pages": [12]}]
    colpali_passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.9},
    ]
    post_rrf_passages = [
        {"document_title": "Notice Perform 76", "page_no": 1, "score": 0.8},
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.7},
    ]
    final_passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.7, "rerank_score": 5.0},
    ]

    result = build_question_eval_result(
        question="test",
        q_type="mono",
        expected_pages=expected,
        passages=final_passages,
        colpali_passages=colpali_passages,
        post_rrf_passages=post_rrf_passages,
        minilm_rerank_enabled=True,
    )

    assert result["metrics_colpali"]["mrr"] == 1.0
    assert result["metrics_post_rrf"]["mrr"] == 0.5
    assert result["metrics"]["mrr"] == 1.0
    assert result["minilm_delta"]["mrr"] == 0.5
    assert result["rrf_delta"]["mrr"] == -0.5


def test_build_question_eval_result_with_kag_delta():
    expected = [{"document_title": "Notice Perform 70", "pages": [12, 13]}]
    pre_kag_passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.8},
    ]
    post_rrf_passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.8},
        {"document_title": "Notice Perform 70", "page_no": 13, "score": 0.7},
    ]
    final_passages = post_rrf_passages

    result = build_question_eval_result(
        question="test",
        q_type="mono",
        expected_pages=expected,
        passages=final_passages,
        pre_kag_passages=pre_kag_passages,
        post_rrf_passages=post_rrf_passages,
        kag_only_passages=[{"document_title": "Notice Perform 70", "page_no": 13, "score": 0.6}],
        kag_enabled=True,
    )

    assert result["metrics_pre_kag"]["context_recall"] == 0.5
    assert result["metrics_post_rrf"]["context_recall"] == 1.0
    assert result["kag_delta"]["recall"] == 0.5
    assert result["metrics_kag_only"]["context_recall"] == 0.5
    assert result["kag_enabled"] is True


def test_build_question_eval_result_with_pgvector_and_lexical_only():
    expected = [{"document_title": "Notice Perform 70", "pages": [12, 13]}]
    pgvector_passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.88},
    ]
    lexical_passages = [
        {"document_title": "Notice Perform 70", "page_no": 13, "score": 0.75},
    ]
    final_passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.9},
        {"document_title": "Notice Perform 70", "page_no": 13, "score": 0.8},
    ]

    result = build_question_eval_result(
        question="test",
        q_type="mono",
        expected_pages=expected,
        passages=final_passages,
        pgvector_only_passages=pgvector_passages,
        lexical_only_passages=lexical_passages,
    )

    assert result["metrics_pgvector_only"]["context_recall"] == 0.5
    assert result["metrics_pgvector_only"]["mrr"] == 1.0
    assert result["metrics_lexical_only"]["context_recall"] == 0.5
    assert result["metrics_lexical_only"]["mrr"] == 1.0
    assert len(result["analysis_pgvector_only"]["hits"]) == 1
    assert len(result["analysis_lexical_only"]["hits"]) == 1


def test_rerank_delta_is_post_minus_colpali():
    expected = [{"document_title": "Notice Perform 70", "pages": [12, 13]}]
    colpali_passages = [
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.9},
        {"document_title": "Notice Perform 70", "page_no": 13, "score": 0.85},
        {"document_title": "Notice Perform 76", "page_no": 1, "score": 0.8},
    ]
    post_passages = [
        {"document_title": "Notice Perform 70", "page_no": 13, "score": 0.85, "rerank_score": 5.0},
        {"document_title": "Notice Perform 70", "page_no": 12, "score": 0.9, "rerank_score": 4.0},
    ]

    result = build_question_eval_result(
        question="test",
        q_type="mono-document",
        expected_pages=expected,
        passages=post_passages,
        colpali_passages=colpali_passages,
    )

    post = result["metrics"]
    colpali = result["metrics_colpali"]
    delta = result["rerank_delta"]

    assert delta["precision"] == round(post["context_precision"] - colpali["context_precision"], 4)
    assert delta["recall"] == round(post["context_recall"] - colpali["context_recall"], 4)
    assert delta["mrr"] == round(post["mrr"] - colpali["mrr"], 4)
