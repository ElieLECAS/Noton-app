"""Tests schémas signaux query understanding."""
from app.services.query_signals_schemas import (
    ExtractedEntity,
    parse_and_validate_signals,
    to_lightweight_signals,
)


def test_parse_and_validate_signals_filters_invalid_categories():
    raw = {
        "intent": "installation",
        "entities": [{"text": "coulisses"}, "MONOBLOC"],
        "inferred_categories": ["mounting", "couleur_inventee", "hardware_adjustment"],
        "confidence": 0.9,
    }
    signals = parse_and_validate_signals(raw, session=None)
    assert signals.intent == "installation"
    assert len(signals.entities) == 2
    assert signals.entities[0].text == "coulisses"
    assert "mounting" in signals.inferred_categories
    assert "hardware_adjustment" in signals.inferred_categories
    assert "couleur_inventee" not in signals.inferred_categories


def test_parse_coerces_string_entities():
    raw = {"entities": ["Perform", {"text": "ALU", "role": "material"}]}
    signals = parse_and_validate_signals(raw, session=None)
    texts = [e.text for e in signals.entities]
    assert "Perform" in texts
    assert "ALU" in texts


def test_material_hint_normalizes_alu():
    raw = {"material_hint": "alu"}
    signals = parse_and_validate_signals(raw, session=None)
    assert signals.material_hint == "aluminium"


def test_to_lightweight_signals_dedupes_entity_texts():
    extraction = parse_and_validate_signals(
        {
            "entities": [{"text": "Perform"}, {"text": "perform"}],
            "detected_references": ["Perform 70"],
        },
        session=None,
    )
    lw = to_lightweight_signals(extraction)
    assert lw.entity_texts == ["Perform", "Perform 70"]


def test_extracted_entity_strips_text():
    entity = ExtractedEntity(text="  MONOBLOC  ")
    assert entity.text == "MONOBLOC"
