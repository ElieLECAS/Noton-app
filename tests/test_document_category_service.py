"""Tests du service catégories par document."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.services.document_category_service import (
    _build_navigation,
    get_document_category_pages,
)


def test_get_document_category_pages_disabled():
    session = MagicMock()
    with patch("app.services.document_category_service.settings") as mock_settings:
        mock_settings.KAG_ENABLED = False
        payload = get_document_category_pages(session, 1, 2)
    assert payload is None
    session.execute.assert_not_called()


def test_build_navigation_prev_next():
    pages = [
        {"page_no": 1, "chunk_count": 2},
        {"page_no": 3, "chunk_count": 1},
        {"page_no": 7, "chunk_count": 4},
    ]
    nav = _build_navigation(pages, page_no=3)
    assert nav["current_index"] == 1
    assert nav["total"] == 3
    assert nav["prev"]["page_no"] == 1
    assert nav["next"]["page_no"] == 7
