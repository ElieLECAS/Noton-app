"""Tests unitaires extraction KAG (limites modèle compact, validation)."""
from __future__ import annotations

from unittest import mock

import pytest

from app.services.kag_extraction_service import (
    _coerce_kag_page_response,
    _effective_kag_limits,
    _is_compact_extraction_model,
    build_kag_batches,
    extract_page_kag_response,
)


class TestKagLimits:
    def test_compact_model_detected(self):
        assert _is_compact_extraction_model("ministral-3b-latest") is True
        assert _is_compact_extraction_model("mistral-small-latest") is False

    @mock.patch("app.services.kag_extraction_service._kag_extraction_model", return_value="ministral-3b-latest")
    def test_effective_limits_lower_for_3b(self, _mock_model):
        ent, rel = _effective_kag_limits()
        assert ent <= 8
        assert rel <= 6


class TestCoerceKagResponse:
    def test_valid_response(self):
        raw = {
            "page_no": 5,
            "entities": [{"name": "Gâche OB", "type": "product"}],
            "relations": [],
        }
        resp = _coerce_kag_page_response(raw, page_no=5)
        assert resp.entities[0].name == "Gâche OB"

    def test_missing_page_no_defaulted(self):
        raw = {"entities": [{"name": "Vis", "type": "tool"}], "relations": []}
        resp = _coerce_kag_page_response(raw, page_no=2)
        assert resp.page_no == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Aucune entité"):
            _coerce_kag_page_response({"entities": [], "relations": []}, page_no=1)

    def test_categories_only_valid(self):
        resp = _coerce_kag_page_response(
            {"entities": [], "relations": [], "categories": ["warranty"]},
            page_no=1,
            valid_category_slugs=frozenset({"warranty"}),
        )
        assert resp.categories == ["warranty"]


class TestBuildKagBatches:
    def test_sliding_window_overlap(self):
        batches = build_kag_batches(list(range(1, 8)), batch_size=3, overlap=1)
        assert batches == [[1, 2, 3], [3, 4, 5], [5, 6, 7]]


class TestExtractPageKagRetry:
    @mock.patch("app.services.kag_extraction_service.render_page_png_cached", return_value=b"png")
    @mock.patch("app.services.kag_extraction_service._call_kag_vision_api")
    @mock.patch("app.services.kag_extraction_service._coerce_kag_page_response")
    def test_retries_once_on_validation_error(self, mock_coerce, mock_api, _mock_png):
        from app.services.kag_extraction_service import KagPageResponse

        mock_coerce.side_effect = [
            ValueError("JSON invalide"),
            KagPageResponse(
                page_no=24,
                entities=[{"name": "Réf ABC", "type": "reference"}],
                relations=[],
            ),
        ]
        mock_api.return_value = {"page_no": 24, "entities": [], "relations": []}

        result = extract_page_kag_response("/fake.pdf", 24, "Doc", ["chunk texte"])

        assert mock_api.call_count == 2
        assert mock_coerce.call_count == 2
        assert result is not None
        assert result.entities[0].name == "Réf ABC"
