"""Tests unitaires du service d'extraction vision Ministral 3B.

Couvre :
- Validation JSON valide multi-chunks → specs correctes
- Chunk LLM > 480 tokens → split serveur
- JSON invalide → fallback pymupdf4llm
- Merge inter-pages : flags LLM, heuristiques, re-split si > 480 tokens
- Validation metadata (chunking_version, step_number, token_count, cross_page_merge)
"""
from __future__ import annotations

from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vision_response(page_no: int = 1, chunks=None):
    """Construit une réponse JSON vision valide."""
    if chunks is None:
        chunks = [
            {
                "heading": "Étape 1",
                "step_number": 1,
                "section_type": "step",
                "content": "Démonter l'équerre de compas OF.",
                "continues_on_next_page": False,
                "continues_from_previous_page": False,
            }
        ]
    return {"page_no": page_no, "chunks": chunks}


def _make_spec(
    page_no: int = 1,
    content: str = "Texte chunk.",
    section_type: str = "step",
    step_number: int | None = None,
    continues_on_next_page: bool = False,
    continues_from_previous_page: bool = False,
    cross_page_merge: bool = False,
    node_id: str | None = None,
):
    import uuid

    meta = {
        "page_no": page_no,
        "page_start": page_no,
        "page_end": page_no,
        "section_type": section_type,
        "chunking_version": "vision_page_v1",
        "extraction_provider": "mistral_vision",
        "content_type": "semantic_leaf",
        "is_leaf": True,
        "token_count": len(content) // 4,
        "continues_on_next_page": continues_on_next_page,
        "continues_from_previous_page": continues_from_previous_page,
    }
    if step_number is not None:
        meta["step_number"] = step_number
    if cross_page_merge:
        meta["cross_page_merge"] = True
    return {
        "content": content,
        "text": content,
        "start_char": 0,
        "end_char": len(content),
        "node_id": node_id or str(uuid.uuid4()),
        "parent_node_id": None,
        "is_leaf": True,
        "hierarchy_level": 1,
        "chunk_index": 0,
        "metadata_json": meta,
    }


# ---------------------------------------------------------------------------
# Tests _validate_and_normalize
# ---------------------------------------------------------------------------


class TestValidateAndNormalize:
    def test_valid_single_chunk(self):
        from app.services.vision_page_extraction_service import _validate_and_normalize

        raw = _make_vision_response(
            page_no=1,
            chunks=[
                {
                    "heading": "En-tête",
                    "step_number": None,
                    "section_type": "document_header",
                    "content": "Date : 01/01/2026  Réf : PRO-PVC-OFOB-01",
                    "continues_on_next_page": False,
                    "continues_from_previous_page": False,
                }
            ],
        )
        specs = _validate_and_normalize(raw, page_no=1, metadata_base={})
        assert len(specs) == 1
        spec = specs[0]
        assert spec["content"] == "Date : 01/01/2026  Réf : PRO-PVC-OFOB-01"
        meta = spec["metadata_json"]
        assert meta["chunking_version"] == "vision_page_v1"
        assert meta["extraction_provider"] == "mistral_vision"
        assert meta["section_type"] == "document_header"
        assert meta["page_no"] == 1
        assert "token_count" in meta

    def test_valid_multi_chunk_step_numbers(self):
        from app.services.vision_page_extraction_service import _validate_and_normalize

        raw = _make_vision_response(
            page_no=2,
            chunks=[
                {"heading": "Étape 1", "step_number": 1, "section_type": "step",
                 "content": "Démonter l'équerre.", "continues_on_next_page": False,
                 "continues_from_previous_page": False},
                {"heading": "Étape 2", "step_number": 2, "section_type": "step",
                 "content": "Installer la têtière.", "continues_on_next_page": False,
                 "continues_from_previous_page": False},
            ],
        )
        specs = _validate_and_normalize(raw, page_no=2, metadata_base={})
        assert len(specs) == 2
        assert specs[0]["metadata_json"]["step_number"] == 1
        assert specs[1]["metadata_json"]["step_number"] == 2

    def test_chunk_over_480_tokens_is_split(self):
        """Un chunk > 480 tokens doit être découpé côté serveur."""
        from app.services.vision_page_extraction_service import _validate_and_normalize

        long_content = "Mot " * 600  # ~600 tokens
        raw = _make_vision_response(
            page_no=1,
            chunks=[
                {
                    "heading": "Section longue",
                    "step_number": None,
                    "section_type": "section",
                    "content": long_content,
                    "continues_on_next_page": False,
                    "continues_from_previous_page": False,
                }
            ],
        )
        specs = _validate_and_normalize(raw, page_no=1, metadata_base={})
        assert len(specs) >= 2, "Le chunk > 480 tokens doit être splité en plusieurs specs"
        for spec in specs:
            assert spec["metadata_json"]["token_count"] <= 500  # marge raisonnable

    def test_empty_chunks_raises(self):
        from app.services.vision_page_extraction_service import _validate_and_normalize

        raw = {"page_no": 1, "chunks": []}
        with pytest.raises(ValueError, match="Aucun chunk valide"):
            _validate_and_normalize(raw, page_no=1, metadata_base={})

    def test_whitespace_only_content_filtered(self):
        from app.services.vision_page_extraction_service import _validate_and_normalize

        raw = _make_vision_response(
            page_no=1,
            chunks=[
                {"heading": None, "step_number": None, "section_type": "section",
                 "content": "   \n  ", "continues_on_next_page": False,
                 "continues_from_previous_page": False},
            ],
        )
        with pytest.raises(ValueError, match="Aucun chunk valide"):
            _validate_and_normalize(raw, page_no=1, metadata_base={})

    def test_invalid_json_raises(self):
        from app.services.vision_page_extraction_service import _validate_and_normalize

        with pytest.raises(ValueError):
            _validate_and_normalize({"page_no": 1, "wrong_key": []}, page_no=1, metadata_base={})

    def test_continues_flags_preserved(self):
        from app.services.vision_page_extraction_service import _validate_and_normalize

        raw = _make_vision_response(
            page_no=1,
            chunks=[
                {"heading": "Étape 4", "step_number": 4, "section_type": "step",
                 "content": "Retirer l'obturateur de manœuvre sur la quincaillerie",
                 "continues_on_next_page": True,
                 "continues_from_previous_page": False},
            ],
        )
        specs = _validate_and_normalize(raw, page_no=1, metadata_base={})
        assert specs[-1]["metadata_json"]["continues_on_next_page"] is True


# ---------------------------------------------------------------------------
# Tests extract_page_chunk_specs (avec mock API)
# ---------------------------------------------------------------------------


class TestExtractPageChunkSpecs:
    def test_success_returns_specs(self, tmp_path):
        from app.services.vision_page_extraction_service import extract_page_chunk_specs

        fake_response = _make_vision_response(page_no=1)
        fake_png = b"\x89PNG\r\n"

        with mock.patch(
            "app.services.vision_page_extraction_service.render_page_png_cached",
            return_value=fake_png,
        ), mock.patch(
            "app.services.vision_page_extraction_service._call_vision_api",
            return_value=fake_response,
        ):
            specs = extract_page_chunk_specs(
                pdf_path="/fake/doc.pdf",
                page_no=1,
                document_title="Doc test",
                metadata_base={"document_id": 1},
            )

        assert len(specs) == 1
        assert specs[0]["metadata_json"]["extraction_provider"] == "mistral_vision"
        assert specs[0]["metadata_json"]["chunking_version"] == "vision_page_v1"

    def test_api_failure_triggers_fallback(self, tmp_path):
        """En cas d'échec API, le fallback pymupdf4llm doit être appelé."""
        from app.services.vision_page_extraction_service import extract_page_chunk_specs

        fake_png = b"\x89PNG\r\n"
        fallback_spec = _make_spec(page_no=1, content="Fallback content.")
        fallback_spec["metadata_json"]["extraction_provider"] = "pymupdf4llm_fallback"

        with mock.patch(
            "app.services.vision_page_extraction_service.render_page_png_cached",
            return_value=fake_png,
        ), mock.patch(
            "app.services.vision_page_extraction_service._call_vision_api",
            side_effect=TimeoutError("timeout"),
        ), mock.patch(
            "app.services.vision_page_extraction_service._fallback_pymupdf4llm",
            return_value=[fallback_spec],
        ) as mock_fallback:
            specs = extract_page_chunk_specs(
                pdf_path="/fake/doc.pdf",
                page_no=1,
                document_title="Doc test",
                metadata_base={},
            )

        mock_fallback.assert_called_once()
        assert specs[0]["metadata_json"]["extraction_provider"] == "pymupdf4llm_fallback"

    def test_png_render_failure_triggers_fallback(self):
        """Si le rendu PNG échoue, le fallback est appelé sans tentative API."""
        from app.services.vision_page_extraction_service import extract_page_chunk_specs

        fallback_spec = _make_spec(page_no=1, content="Fallback content.")
        fallback_spec["metadata_json"]["extraction_provider"] = "pymupdf4llm_fallback"

        with mock.patch(
            "app.services.vision_page_extraction_service.render_page_png_cached",
            side_effect=RuntimeError("render failed"),
        ), mock.patch(
            "app.services.vision_page_extraction_service._fallback_pymupdf4llm",
            return_value=[fallback_spec],
        ) as mock_fallback, mock.patch(
            "app.services.vision_page_extraction_service._call_vision_api",
        ) as mock_api:
            specs = extract_page_chunk_specs(
                pdf_path="/fake/doc.pdf",
                page_no=1,
                document_title="Doc test",
                metadata_base={},
            )

        mock_fallback.assert_called_once()
        mock_api.assert_not_called()
        assert specs[0]["metadata_json"]["extraction_provider"] == "pymupdf4llm_fallback"


# ---------------------------------------------------------------------------
# Tests merge_cross_page_chunks
# ---------------------------------------------------------------------------


class TestMergeCrossPageChunks:
    def test_no_merge_needed(self):
        """Deux chunks indépendants ne doivent pas être fusionnés."""
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        s1 = _make_spec(page_no=1, content="Étape 1 terminée.", step_number=1)
        s2 = _make_spec(page_no=2, content="Étape 2 terminée.", step_number=2)
        result = merge_cross_page_chunks([s1, s2])
        assert len(result) == 2

    def test_merge_on_llm_flag_continues_on_next_page(self):
        """Flag continues_on_next_page=True déclenche la fusion."""
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        s1 = _make_spec(
            page_no=1,
            content="Retirer l'obturateur de manœuvre sur la quincaillerie",
            step_number=4,
            continues_on_next_page=True,
        )
        s2 = _make_spec(
            page_no=2,
            content="pour libérer la manœuvre OB.",
            step_number=4,
            continues_from_previous_page=True,
        )
        result = merge_cross_page_chunks([s1, s2])
        assert len(result) == 1
        merged = result[0]
        assert "obturateur" in merged["content"]
        assert "libérer" in merged["content"]
        assert merged["metadata_json"]["cross_page_merge"] is True
        assert merged["metadata_json"]["page_start"] == 1
        assert merged["metadata_json"]["page_end"] == 2

    def test_merge_on_incomplete_sentence(self):
        """Phrase sans ponctuation finale en bas de page → fusion heuristique."""
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        s1 = _make_spec(
            page_no=1,
            content="Installer et visser la gâche OB (Droite ou GAUCHE) en traverse basse du dormant",
            step_number=5,
        )
        s2 = _make_spec(
            page_no=2,
            content="en tenant compte de la position de l'ouvrant.",
            step_number=5,
        )
        result = merge_cross_page_chunks([s1, s2])
        assert len(result) == 1
        assert result[0]["metadata_json"]["cross_page_merge"] is True

    def test_no_merge_document_header(self):
        """Un document_header ne doit jamais être fusionné."""
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        s1 = _make_spec(page_no=1, content="En-tête du document.", section_type="document_header")
        s2 = _make_spec(page_no=2, content="Étape 1 commence ici.", section_type="step")
        result = merge_cross_page_chunks([s1, s2])
        assert len(result) == 2

    def test_no_merge_different_steps_no_flag(self):
        """Étapes différentes sans flag LLM → pas de fusion."""
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        s1 = _make_spec(page_no=1, content="Étape 3 terminée.", step_number=3)
        s2 = _make_spec(page_no=2, content="Étape 4 commence.", step_number=4)
        result = merge_cross_page_chunks([s1, s2])
        assert len(result) == 2

    def test_merge_over_480_tokens_resplit(self):
        """Un merge > 480 tokens doit être re-splité en 2 chunks."""
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        long_part1 = "Contenu A. " * 150
        long_part2 = "Contenu B. " * 150
        s1 = _make_spec(
            page_no=1,
            content=long_part1.rstrip(),
            step_number=1,
            continues_on_next_page=True,
        )
        s2 = _make_spec(
            page_no=2,
            content=long_part2.rstrip(),
            step_number=1,
            continues_from_previous_page=True,
        )
        result = merge_cross_page_chunks([s1, s2])
        assert len(result) >= 2, "Un merge trop grand doit être re-splité"
        for spec in result:
            assert spec["metadata_json"]["token_count"] <= 520  # marge

    def test_merge_continuity_connector(self):
        """Chunk débutant par un connecteur ('et ', 'puis ') → fusion."""
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        s1 = _make_spec(page_no=1, content="Visser la gâche OB.")
        s2 = _make_spec(page_no=2, content="puis contrôler l'alignement.")
        result = merge_cross_page_chunks([s1, s2])
        assert len(result) == 1

    def test_non_adjacent_pages_not_merged(self):
        """Chunks de pages non adjacentes ne doivent pas être fusionnés."""
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        s1 = _make_spec(page_no=1, content="Page 1.", continues_on_next_page=True)
        s3 = _make_spec(page_no=3, content="Page 3.", continues_from_previous_page=True)
        result = merge_cross_page_chunks([s1, s3])
        assert len(result) == 2

    def test_empty_list(self):
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        assert merge_cross_page_chunks([]) == []

    def test_single_spec(self):
        from app.services.vision_page_extraction_service import merge_cross_page_chunks

        s = _make_spec(page_no=1, content="Seul chunk.")
        result = merge_cross_page_chunks([s])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test intégration : _extract_and_persist_chunks avec mock vision
# ---------------------------------------------------------------------------


class TestExtractAndPersistChunksVision:
    """Vérifie que _extract_and_persist_chunks appelle bien extract_page_chunk_specs
    et merge_cross_page_chunks (sans base de données réelle)."""

    def test_calls_vision_pipeline(self, tmp_path):
        from app.services.document_indexing_service import _extract_and_persist_chunks

        doc = mock.MagicMock()
        doc.id = 42
        doc.title = "Test doc"
        doc.source = "test"
        doc.library_id = 1
        doc.user_id = 99

        fake_spec = _make_spec(page_no=1, content="Chunk vision p1.", step_number=1)
        fake_spec["metadata_json"]["heading"] = "Étape 1"
        fake_spec["metadata_json"]["extraction_provider"] = "mistral_vision"

        with mock.patch(
            "app.services.document_indexing_service._get_pdf_page_count",
            return_value=1,
        ), mock.patch(
            "app.services.document_indexing_service.extract_page_chunk_specs",
            return_value=[fake_spec],
        ) as mock_extract, mock.patch(
            "app.services.document_indexing_service.merge_cross_page_chunks",
            side_effect=lambda specs: specs,
        ) as mock_merge, mock.patch(
            "app.services.document_indexing_service.Session"
        ) as mock_sess_cls:
            sess = mock_sess_cls.return_value.__enter__.return_value
            sess.exec.return_value.all.return_value = []
            sess.get.return_value = doc

            count = _extract_and_persist_chunks(
                session=sess,
                document=doc,
                pdf_path="/fake/doc.pdf",
            )

        mock_extract.assert_called_once_with(
            "/fake/doc.pdf", 1, "Test doc", mock.ANY
        )
        mock_merge.assert_called_once()
        # 1 L0 anchor + 1 L1 chunk = 2
        assert count == 2

    def test_chunking_version_is_vision_page_v1(self, tmp_path):
        """Les chunks persistés doivent avoir chunking_version=vision_page_v1."""
        from app.services.document_indexing_service import _extract_and_persist_chunks, CHUNKING_VERSION

        assert CHUNKING_VERSION == "vision_page_v1"
