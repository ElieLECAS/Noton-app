"""Tests classement voisins graphe multi-hop (anti fan-out)."""
from unittest.mock import MagicMock

from app.services.kag_graph_service import (
    _edge_relation_boost,
    _query_lexical_sim,
    rank_neighbor_entities_for_query,
)


def test_edge_relation_boost_typed_vs_co_occurs():
    assert _edge_relation_boost("appartient_a", 1.25) == 1.25
    assert _edge_relation_boost("co_occurs", 1.25) == 0.85
    assert _edge_relation_boost("unknown", 1.25) == 1.0


def test_query_lexical_sim_overlap():
    assert _query_lexical_sim("Gamme Alpha procédure", "Gamme Alpha") > 0.5
    assert _query_lexical_sim("fenêtre pvc", "Gamme Alpha") < 0.5


def test_rank_neighbor_entities_empty_seeds():
    session = MagicMock()
    assert rank_neighbor_entities_for_query(session, 1, set()) == []


def _mock_exec_rows(rows_list):
    results = []
    for rows in rows_list:
        m = MagicMock()
        m.all.return_value = rows
        results.append(m)
    return results


def test_rank_neighbor_entities_filters_low_weight():
    session = MagicMock()
    session.exec.side_effect = _mock_exec_rows([
        [(1, 10, 0.05, "co_occurs", 1.0, "Hub", 5)],
        [],
    ])
    ranked = rank_neighbor_entities_for_query(
        session,
        space_id=1,
        entity_ids={1},
        query_text="Hub",
        min_weight=0.15,
    )
    assert ranked == []


def test_rank_neighbor_entities_top_k_per_seed():
    session = MagicMock()
    rows_a = [
        (1, 10, 0.9, "appartient_a", 1.0, "Alpha", 2),
        (1, 11, 0.8, "compatible_avec", 1.0, "Beta", 2),
        (1, 12, 0.7, "reference", 1.0, "Gamma", 2),
    ]
    session.exec.side_effect = _mock_exec_rows([rows_a, []])
    ranked = rank_neighbor_entities_for_query(
        session,
        space_id=1,
        entity_ids={1},
        query_text="Alpha Beta",
        top_k_per_entity=2,
        limit=10,
        min_weight=0.15,
    )
    ids = [nid for nid, _ in ranked]
    assert 10 in ids
    assert 11 in ids
    assert 12 not in ids
