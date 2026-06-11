import pytest
from app.services.retriever_evaluator import (
    match_page,
    compute_context_precision,
    compute_context_recall,
    compute_mrr
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
