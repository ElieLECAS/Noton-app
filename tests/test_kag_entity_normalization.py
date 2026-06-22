"""Tests normalisation entités KAG."""
from app.services.kag_extraction_service import normalize_and_expand_entity, normalize_entity_name


def test_normalize_alu_to_aluminium():
    canonical, aliases = normalize_and_expand_entity("ALU")
    assert canonical == "aluminium"
    assert "ALU" in aliases
    assert "alu" in aliases


def test_normalize_perform_gamme():
    canonical, aliases = normalize_and_expand_entity("Perform")
    assert canonical == "Gamme Perform"
    assert "Perform" in aliases


def test_normalize_entity_name_uses_rules():
    assert normalize_entity_name("ALU") == "aluminium"
    assert normalize_entity_name("perform") == "gamme perform"


def test_unknown_entity_unchanged():
    canonical, aliases = normalize_and_expand_entity("MONOBLOC")
    assert canonical == "MONOBLOC"
    assert aliases == []
