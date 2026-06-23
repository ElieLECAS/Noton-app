"""Tests du service catégories par espace."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.space_category_service import (
    _build_navigation,
    get_space_categories,
)


def test_get_space_categories_disabled():
    session = MagicMock()
    with patch("app.services.space_category_service.settings") as mock_settings:
        mock_settings.KAG_ENABLED = False
        payload = get_space_categories(session, 1)
    assert payload["status"] == "disabled"
    assert payload["categories"] == []
    session.execute.assert_not_called()


def test_build_navigation_prev_next():
    pages = [
        {"document_id": 1, "page_no": 1, "document_title": "A"},
        {"document_id": 1, "page_no": 3, "document_title": "A"},
        {"document_id": 2, "page_no": 2, "document_title": "B"},
    ]
    nav = _build_navigation(pages, document_id=1, page_no=3)
    assert nav["current_index"] == 1
    assert nav["total"] == 3
    assert nav["prev"]["page_no"] == 1
    assert nav["next"]["page_no"] == 2
