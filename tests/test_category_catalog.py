"""Tests catalogue catégories de contenu."""
from app.services.category_catalog import (
    CONTENT_CATEGORY_SLUGS,
    INTENT_TO_CATEGORIES,
    suggested_categories_for_intent,
)
from app.services.slot_catalog import (
    build_classification_filters,
    parse_content_categories,
)


def test_content_category_slugs_count():
    assert len(CONTENT_CATEGORY_SLUGS) == 16


def test_suggested_categories_for_installation():
    cats = suggested_categories_for_intent("installation")
    assert "mounting" in cats
    assert cats == INTENT_TO_CATEGORIES["installation"]


def test_parse_content_categories_csv():
    assert parse_content_categories("mounting,warranty") == ["mounting", "warranty"]


def test_build_classification_filters_with_categories():
    filters = build_classification_filters(
        {
            "product_family": "fenetres",
            "material": "pvc",
            "content_categories": "warranty,regulatory",
        },
        skipped_optional=["product_range", "supplier"],
    )
    assert filters.content_categories == ["warranty", "regulatory"]
