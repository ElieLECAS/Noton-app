import pytest
import json
from app.services.kag_extraction_service import _repair_truncated_json_array

def test_repair_complete_json():
    # Un JSON complet ne doit pas être modifié s'il se termine par "]"
    content = '[{"entity_a": "X", "entity_b": "Y", "relation_type": "depends_on"}]'
    repaired = _repair_truncated_json_array(content)
    assert repaired == content
    assert json.loads(repaired) == [{"entity_a": "X", "entity_b": "Y", "relation_type": "depends_on"}]

def test_repair_truncated_at_comma():
    # Cas où le JSON est coupé juste après une virgule séparant deux éléments
    content = '[{"entity_a": "X", "entity_b": "Y", "relation_type": "depends_on"}, '
    repaired = _repair_truncated_json_array(content)
    assert repaired.endswith("]")
    parsed = json.loads(repaired)
    assert len(parsed) == 1
    assert parsed[0]["entity_a"] == "X"

def test_repair_truncated_in_middle_of_object():
    # Cas où le JSON est coupé en plein milieu du deuxième objet
    content = '[{"entity_a": "X", "entity_b": "Y", "relation_type": "depends_on"}, {"entity_a": "Z", "entity_b": "A"'
    repaired = _repair_truncated_json_array(content)
    assert repaired.endswith("]")
    parsed = json.loads(repaired)
    assert len(parsed) == 1
    assert parsed[0]["entity_a"] == "X"

def test_repair_extremely_truncated():
    # Cas où le JSON n'a aucun objet complet
    content = '[{"entity_a": "X"'
    repaired = _repair_truncated_json_array(content)
    assert repaired == content  # Pas d'accolade fermante, retourne le texte d'origine sans masquer l'erreur
    with pytest.raises(json.JSONDecodeError):
        json.loads(repaired)

def test_repair_non_array():
    # Cas où le texte ne commence pas par un tableau
    content = '{"entity_a": "X"}'
    repaired = _repair_truncated_json_array(content)
    assert repaired == content
