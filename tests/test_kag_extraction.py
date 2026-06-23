"""Tests unitaires extraction KAG (limites modèle compact, validation)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from types import SimpleNamespace

from app.services.kag_extraction_service import (
    ChunkCategoryItem,
    KagExtractedEntity,
    _annotate_chunks_with_entities,
    _coerce_kag_page_response,
    _effective_kag_limits,
    _format_chunks_for_kag_prompt,
    _is_compact_extraction_model,
    build_kag_batches,
    extract_page_kag_response,
)


class TestKagLimits:
    def test_compact_model_detected(self):
        assert _is_compact_extraction_model("ministral-3b-latest") is True
        assert _is_compact_extraction_model("mistral-small-latest") is False

    @patch("app.services.kag_extraction_service._kag_extraction_model", return_value="ministral-3b-latest")
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

    def test_chunk_categories_only_valid(self):
        resp = _coerce_kag_page_response(
            {
                "entities": [],
                "relations": [],
                "chunk_categories": [
                    {"chunk_index": 0, "categories": ["mounting", "invalid"]},
                    {"chunk_index": 2, "categories": ["warranty"]},
                ],
            },
            page_no=1,
            valid_category_slugs=frozenset({"mounting", "warranty"}),
        )
        assert resp.chunk_categories == [
            ChunkCategoryItem(chunk_index=0, categories=["mounting"]),
            ChunkCategoryItem(chunk_index=2, categories=["warranty"]),
        ]


class TestBuildKagBatches:
    def test_sliding_window_overlap(self):
        batches = build_kag_batches(list(range(1, 8)), batch_size=3, overlap=1)
        assert batches == [[1, 2, 3], [3, 4, 5], [5, 6, 7]]


class TestKagChunkPrompt:
    def test_format_chunks_includes_chunk_index(self):
        chunk = MagicMock()
        chunk.content = "Texte pose profil"
        chunk.metadata_json = {"heading": "Pose", "section_type": "step"}
        text = _format_chunks_for_kag_prompt({3: [chunk]})
        assert "[chunk_index=0]" in text
        assert "PAGE 3" in text
        assert "Texte pose profil" in text


class TestExtractPageKagRetry:
    @patch("app.services.multimodal_page_service.render_page_png_cached", return_value=b"png")
    @patch("app.services.kag_extraction_service._call_kag_vision_api")
    @patch("app.services.kag_extraction_service._coerce_kag_page_response")
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


class TestAnnotateChunksWithEntities:
    """Les entités de la page sont écrites dans les métadonnées des chunks L1
    (Phase 2 : alimente le texte d'embedding via `_build_embed_text`)."""

    def test_writes_canonical_entity_names_to_metadata(self):
        session = MagicMock()
        chunk = SimpleNamespace(metadata_json={"page_no": 2}, metadata_=None)
        entities = [
            KagExtractedEntity(name="Kömmerling 76", type="product"),
            KagExtractedEntity(name="pvc", type="material"),  # normalisé en PVC
        ]

        _annotate_chunks_with_entities(session, [chunk], entities)

        assert "entities" in chunk.metadata_json
        assert "Kömmerling 76" in chunk.metadata_json["entities"]
        assert "PVC" in chunk.metadata_json["entities"]  # ENTITY_NORMALIZATION_RULES
        assert chunk.metadata_ == chunk.metadata_json
        session.add.assert_called()

    def test_dedup_and_preserves_existing(self):
        session = MagicMock()
        chunk = SimpleNamespace(metadata_json={"entities": ["Existant"]}, metadata_=None)
        entities = [
            KagExtractedEntity(name="Existant", type="other"),
            KagExtractedEntity(name="Nouveau", type="product"),
        ]

        _annotate_chunks_with_entities(session, [chunk], entities)

        ents = chunk.metadata_json["entities"]
        assert ents.count("Existant") == 1
        assert "Nouveau" in ents

    def test_caps_entity_count(self):
        session = MagicMock()
        chunk = SimpleNamespace(metadata_json={}, metadata_=None)
        entities = [KagExtractedEntity(name=f"Ent{i}", type="other") for i in range(30)]

        _annotate_chunks_with_entities(session, [chunk], entities, max_entities=5)

        assert len(chunk.metadata_json["entities"]) == 5

    def test_noop_without_entities(self):
        session = MagicMock()
        chunk = SimpleNamespace(metadata_json={}, metadata_=None)

        _annotate_chunks_with_entities(session, [chunk], [])

        assert "entities" not in chunk.metadata_json
        session.add.assert_not_called()
