"""Tests non-régression KAG bibliothèque (espace / DocumentChunk)."""
from unittest.mock import MagicMock, patch

import pytest

from app.services.kag_graph_service import (
    _canonicalize_entities,
    _try_cloning_kag_from_another_space,
    save_typed_relations_for_chunk,
)


def test_canonicalize_entities_dedupes_by_name_only():
    entities = [
        {"name": "Profil RC2", "type": "profil_code", "importance": 0.5},
        {"name": "Profil RC2", "type": "gamme_systeme", "importance": 0.9},
    ]
    out = _canonicalize_entities(entities)
    assert len(out) == 1
    assert out[0]["type"] == "gamme_systeme"
    assert out[0]["importance"] == 0.9


def test_save_typed_relations_returns_saved_count():
    session = MagicMock()
    session.exec.return_value.first.return_value = None

    with patch(
        "app.services.kag_graph_service._resolve_entity_id_for_space",
        side_effect=[1, 2],
    ):
        n = save_typed_relations_for_chunk(
            session,
            space_id=1,
            chunk_id=10,
            relations=[
                {
                    "entity_a": "A",
                    "entity_b": "B",
                    "relation_type": "appartient_a",
                    "confidence": 0.8,
                }
            ],
        )
    assert n == 1
    assert session.add.called


def test_save_typed_relations_logs_unresolved():
    session = MagicMock()
    session.exec.return_value.first.return_value = None

    with patch(
        "app.services.kag_graph_service._resolve_entity_id_for_space",
        side_effect=[1, None],
    ):
        n = save_typed_relations_for_chunk(
            session,
            space_id=1,
            chunk_id=10,
            relations=[
                {
                    "entity_a": "A",
                    "entity_b": "Inconnu",
                    "relation_type": "reference",
                    "confidence": 0.9,
                }
            ],
        )
    assert n == 0


def test_clone_returns_stats_dict():
    session = MagicMock()
    session.exec.return_value.first.side_effect = [99, None]
    assert _try_cloning_kag_from_another_space(session, document_id=1, target_space_id=2) is None


def test_clone_success_returns_entity_counts():
    session = MagicMock()
    # source space id
    session.exec.return_value.first.return_value = 5

    mock_rel = MagicMock()
    mock_rel.entity_id = 10
    mock_rel.chunk_id = 100
    mock_rel.relevance_score = 0.8

    mock_ent = MagicMock()
    mock_ent.id = 10
    mock_ent.name = "Entité A"
    mock_ent.entity_type = "concept_technique"

    session.exec.return_value.all.side_effect = [
        [mock_rel],
        [mock_ent],
        [],
        [],
    ]

    with patch(
        "app.services.kag_graph_service._get_or_create_entity_for_space",
    ) as mock_create:
        target_ent = MagicMock()
        target_ent.id = 20
        mock_create.return_value = target_ent
        result = _try_cloning_kag_from_another_space(session, 1, 2)

    assert result is not None
    assert result["cloned"] == 1
    assert result["entities"] >= 1
    assert result["relations"] == 1
