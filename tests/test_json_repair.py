"""Tests de réparation JSON pour sorties LLM tronquées ou mal formées."""
from __future__ import annotations

import json

import pytest

from app.services.multimodal_page_service import (
    _close_truncated_json,
    _parse_json_with_repair,
    _repair_common_json_issues,
)


class TestJsonRepair:
    def test_trailing_comma_removed(self):
        raw = '{"page_no": 1, "entities": [],}'
        parsed = _parse_json_with_repair(raw)
        assert parsed["page_no"] == 1

    def test_truncated_kag_like_json_is_salvaged(self):
        """Simule une réponse KAG coupée par max_tokens (erreur virgule manquante)."""
        entities = [
            {
                "name": f"Produit {i}",
                "type": "product",
                "aliases": [],
                "description": "desc",
                "confidence": 0.9,
            }
            for i in range(12)
        ]
        full = {"page_no": 24, "entities": entities, "relations": []}
        raw = json.dumps(full)[:-40]
        parsed = _parse_json_with_repair(raw)
        assert parsed["page_no"] == 24
        assert len(parsed["entities"]) >= 1

    def test_truncated_mid_object_closed(self):
        raw = (
            '{"page_no": 28, "chunks": [{"heading": "A", "section_type": "step", '
            '"content": "Texte incomplet'
        )
        closed = _close_truncated_json(_repair_common_json_issues(raw))
        parsed = json.loads(closed)
        assert parsed["page_no"] == 28
        assert isinstance(parsed["chunks"], list)

    def test_markdown_wrapper_stripped(self):
        raw = '```json\n{"page_no": 3, "chunks": []}\n```'
        parsed = _parse_json_with_repair(raw)
        assert parsed["page_no"] == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_json_with_repair("")
