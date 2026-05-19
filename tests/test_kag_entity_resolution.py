"""Tests résolution d'entités et relations typées KAG."""
from unittest.mock import patch

import numpy as np

from app.services.kag_extraction_service import (
    is_probably_coreference_mention,
    normalize_entity_core,
    normalize_entity_name,
    normalize_relation_type,
    resolve_entities_coreference_in_chunk,
)


def test_normalize_entity_core_strips_articles():
    assert normalize_entity_core("la Gamme Alpha") == normalize_entity_name("Gamme Alpha")
    assert normalize_entity_core("cette série") == normalize_entity_name("série")


def test_is_probably_coreference_mention():
    assert is_probably_coreference_mention("la gamme") is True
    assert is_probably_coreference_mention("cette série") is True
    assert is_probably_coreference_mention("Gamme Alpha") is False


def test_normalize_relation_type_strict():
    assert normalize_relation_type("depend_de") == "appartient_a"
    assert normalize_relation_type("compatible_avec") == "compatible_avec"
    assert normalize_relation_type("remplace") == "remplace"
    assert normalize_relation_type("est lié à") is None
    assert normalize_relation_type("co_occurs") is None


def test_resolve_entities_canonical_name():
    entities = [
        {"name": "la gamme", "canonical_name": "Gamme Alpha", "type": "gamme_systeme", "importance": 0.7},
        {"name": "Gamme Alpha", "type": "gamme_systeme", "importance": 0.9},
    ]
    out = resolve_entities_coreference_in_chunk(entities)
    assert len(out) == 1
    assert out[0]["name"] == "Gamme Alpha"


@patch("app.services.kag_extraction_service.settings")
@patch("app.services.embedding_service.generate_embeddings_batch")
def test_resolve_entities_embedding_mention(mock_embed, mock_settings):
    mock_settings.KAG_COREFERENCE_ENABLED = True
    mock_settings.KAG_ENTITY_MERGE_SIMILARITY = 0.5
    v_anchor = np.array([1.0, 0.0], dtype=np.float32)
    v_mention = np.array([0.98, 0.02], dtype=np.float32)
    mock_embed.return_value = [v_anchor.tolist(), v_mention.tolist()]

    entities = [
        {"name": "Gamme Alpha", "type": "gamme_systeme", "importance": 0.8},
        {"name": "la gamme", "type": "gamme_systeme", "importance": 0.6},
    ]
    out = resolve_entities_coreference_in_chunk(entities)
    assert len(out) == 1
    assert out[0]["name"] == "Gamme Alpha"
    assert "la gamme" in out[0].get("aliases", []) or any(
        "gamme" in str(a).lower() for a in out[0].get("aliases", [])
    )
