"""Tests métadonnées document et filtres classification."""
from app.services.slot_catalog import (
    build_classification_filters,
    compute_classification_status,
    detect_optional_skip_from_message,
    is_unknown_optional_response,
    normalize_document_classification,
    validate_document_classification,
)


def test_validate_document_classification_complete():
    ok, errors = validate_document_classification(
        product_types=["fenetres"],
        materials=["pvc"],
        source="kommerling",
        proferm_gammes=["perform"],
    )
    assert ok is True
    assert errors == []


def test_validate_document_classification_incomplete():
    ok, errors = validate_document_classification(
        product_types=[],
        materials=["pvc"],
        source="kommerling",
        proferm_gammes=["perform"],
    )
    assert ok is False
    assert any("famille" in e.lower() for e in errors)


def test_compute_classification_status():
    assert compute_classification_status(
        product_types=["fenetres"],
        materials=["pvc"],
        source="technal",
        proferm_gammes=["lumine"],
    ) == "complete"
    assert compute_classification_status(
        product_types=[],
        materials=["pvc"],
        source="technal",
        proferm_gammes=["lumine"],
    ) == "incomplete"


def test_normalize_document_classification_maps_supplier():
    normalized = normalize_document_classification(
        product_types=["fenetres"],
        materials=["aluminium"],
        source="kommerling",
        proferm_gammes=["perform"],
    )
    assert normalized["source"] == "Kommerling"


def test_build_classification_filters_required_only():
    filters = build_classification_filters(
        {
            "intent": "specification",
            "product_family": "fenetres",
            "material": "pvc",
            "product_range": None,
            "supplier": None,
        },
        skipped_optional=["product_range", "supplier"],
    )
    assert filters.product_family == "fenetres"
    assert filters.material == "pvc"
    assert filters.product_range is None
    assert filters.supplier_source is None


def test_build_classification_filters_with_supplier():
    filters = build_classification_filters(
        {
            "product_family": "portes",
            "material": "hybride",
            "product_range": "textural",
            "supplier": "soprofen",
        },
        skipped_optional=[],
    )
    assert filters.product_range == "textural"
    assert filters.supplier_source == "Soprofen"


def test_unknown_optional_phrases():
    assert is_unknown_optional_response("Je ne sais pas") is True
    assert is_unknown_optional_response("PVC") is False
    assert detect_optional_skip_from_message("je sais pas", "supplier") == "supplier"
    assert detect_optional_skip_from_message("je sais pas", "material") is None
