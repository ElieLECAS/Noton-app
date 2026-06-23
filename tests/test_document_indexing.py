"""Tests unitaires du nouveau pipeline d'indexation documentaire.

Couvre :
- Les 3 modes (full, text_only, colpali_only) via le service orchestrateur
- L'endpoint API POST /reindex avec body { mode }
- L'endpoint API POST /reindex-all avec body { mode }
- La validation des prérequis (MISTRAL_API_KEY, COLPALI_ENABLED)
- L'upload via _process_document_for_id → mode full
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock, patch
from typing import List

import pytest

from app.services.document_indexing_service import IndexingMode, process_document_indexing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_document(doc_id: int = 1, title: str = "Doc test"):
    doc = mock.MagicMock()
    doc.id = doc_id
    doc.title = title
    doc.source = "TestSource"
    doc.library_id = 10
    doc.user_id = 99
    return doc


def _patch_deps(
    page_texts=None,
    chunk_specs=None,
    embeddings=None,
    colpali_enabled=True,
    mistral_key="sk-test",
):
    """Retourne un context manager qui mock toutes les dépendances lourdes."""
    if page_texts is None:
        page_texts = [(1, "Texte page 1"), (2, "Texte page 2")]
    if chunk_specs is None:
        chunk_specs = [
            {
                "content": "Contenu chunk A",
                "text": "Contenu chunk A",
                "start_char": 0,
                "end_char": 15,
                "node_id": "node-A",
                "parent_node_id": None,
                "is_leaf": True,
                "hierarchy_level": 1,
                "metadata_json": {"page_no": 1},
            }
        ]
    if embeddings is None:
        embeddings = [[0.1] * 1024]

    patches = [
        mock.patch(
            "app.services.document_indexing_service._delete_all_chunks",
        ),
        mock.patch(
            "app.services.document_indexing_service._delete_text_chunks",
        ),
        mock.patch(
            "app.services.document_indexing_service._set_progress",
        ),
        mock.patch(
            "app.services.document_indexing_service._finalize_document",
        ),
        mock.patch(
            "app.services.document_indexing_service._mark_failed",
        ),
        mock.patch(
            "app.services.document_indexing_service._sync_colpali_for_pages",
            return_value=2,
        ),
        mock.patch(
            "app.services.document_indexing_service._extract_and_persist_chunks",
            return_value=3,
        ),
        mock.patch(
            "app.services.document_indexing_service._embed_text_chunks",
            return_value=2,
        ),
        mock.patch(
            "app.services.document_run.is_processing_run_current",
            return_value=True,
        ),
    ]
    return patches


# ---------------------------------------------------------------------------
# Tests des modes
# ---------------------------------------------------------------------------


class TestIndexingModeEnum:
    def test_valid_values(self):
        assert IndexingMode.FULL == "full"
        assert IndexingMode.TEXT_ONLY == "text_only"
        assert IndexingMode.COLPALI_ONLY == "colpali_only"

    def test_from_string(self):
        assert IndexingMode("full") is IndexingMode.FULL
        assert IndexingMode("text_only") is IndexingMode.TEXT_ONLY
        assert IndexingMode("colpali_only") is IndexingMode.COLPALI_ONLY

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            IndexingMode("unknown")


class TestProcessDocumentIndexingFull:
    """Mode full : extraction texte + embeddings + ColPali."""

    def test_full_mode_calls_all_stages(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4 test")

        doc = _make_document()
        patches = _patch_deps()

        with mock.patch("app.services.document_indexing_service.Session") as mock_session_cls, \
             mock.patch("app.services.document_indexing_service._delete_all_chunks") as del_all, \
             mock.patch("app.services.document_indexing_service._delete_text_chunks") as del_text, \
             mock.patch("app.services.document_indexing_service._set_progress") as set_prog, \
             mock.patch("app.services.document_indexing_service._finalize_document") as finalize, \
             mock.patch("app.services.document_indexing_service._mark_failed") as mark_fail, \
             mock.patch("app.services.document_indexing_service._sync_colpali_for_pages", return_value=2) as colpali_sync, \
             mock.patch("app.services.document_indexing_service._extract_and_persist_chunks", return_value=3) as extract, \
             mock.patch("app.services.document_indexing_service._embed_text_chunks", return_value=2) as embed, \
             mock.patch("app.services.document_run.is_processing_run_current", return_value=True), \
             mock.patch("app.services.file_conversion.ensure_pdf_for_ocr", return_value=pdf_path):

            sess_instance = mock_session_cls.return_value.__enter__.return_value
            sess_instance.get.return_value = doc

            result = process_document_indexing(
                document_id=1,
                file_path=pdf_path,
                user_id=99,
                mode=IndexingMode.FULL,
            )

        assert result["status"] == "completed"
        del_all.assert_called_once()
        del_text.assert_not_called()
        extract.assert_called_once()
        embed.assert_called_once()
        colpali_sync.assert_called_once()
        finalize.assert_called_once()
        mark_fail.assert_not_called()


class TestProcessDocumentIndexingTextOnly:
    """Mode text_only : extraction texte + embeddings, ColPali ignoré."""

    def test_text_only_skips_colpali(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4 test")

        doc = _make_document()

        with mock.patch("app.services.document_indexing_service.Session") as mock_session_cls, \
             mock.patch("app.services.document_indexing_service._delete_all_chunks") as del_all, \
             mock.patch("app.services.document_indexing_service._delete_text_chunks") as del_text, \
             mock.patch("app.services.document_indexing_service._set_progress"), \
             mock.patch("app.services.document_indexing_service._finalize_document") as finalize, \
             mock.patch("app.services.document_indexing_service._mark_failed") as mark_fail, \
             mock.patch("app.services.document_indexing_service._sync_colpali_for_pages") as colpali_sync, \
             mock.patch("app.services.document_indexing_service._extract_and_persist_chunks", return_value=3), \
             mock.patch("app.services.document_indexing_service._embed_text_chunks", return_value=2), \
             mock.patch("app.services.document_run.is_processing_run_current", return_value=True), \
             mock.patch("app.services.file_conversion.ensure_pdf_for_ocr", return_value=pdf_path):

            sess_instance = mock_session_cls.return_value.__enter__.return_value
            sess_instance.get.return_value = doc

            result = process_document_indexing(
                document_id=1,
                file_path=pdf_path,
                user_id=99,
                mode=IndexingMode.TEXT_ONLY,
            )

        assert result["status"] == "completed"
        del_text.assert_called_once()
        del_all.assert_not_called()
        colpali_sync.assert_not_called()
        finalize.assert_called_once()
        mark_fail.assert_not_called()


class TestProcessDocumentIndexingColpaliOnly:
    """Mode colpali_only : ColPali uniquement, pas de texte."""

    def test_colpali_only_skips_text_stages(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4 test")

        doc = _make_document()

        with mock.patch("app.services.document_indexing_service.Session") as mock_session_cls, \
             mock.patch("app.services.document_indexing_service._delete_all_chunks") as del_all, \
             mock.patch("app.services.document_indexing_service._delete_text_chunks") as del_text, \
             mock.patch("app.services.document_indexing_service._set_progress"), \
             mock.patch("app.services.document_indexing_service._finalize_document") as finalize, \
             mock.patch("app.services.document_indexing_service._mark_failed") as mark_fail, \
             mock.patch("app.services.document_indexing_service._sync_colpali_for_pages", return_value=2) as colpali_sync, \
             mock.patch("app.services.document_indexing_service._extract_and_persist_chunks") as extract, \
             mock.patch("app.services.document_indexing_service._embed_text_chunks") as embed, \
             mock.patch("app.services.document_run.is_processing_run_current", return_value=True), \
             mock.patch("app.services.file_conversion.ensure_pdf_for_ocr", return_value=pdf_path):

            sess_instance = mock_session_cls.return_value.__enter__.return_value
            sess_instance.get.return_value = doc

            result = process_document_indexing(
                document_id=1,
                file_path=pdf_path,
                user_id=99,
                mode=IndexingMode.COLPALI_ONLY,
            )

        assert result["status"] == "completed"
        del_all.assert_not_called()
        del_text.assert_not_called()
        extract.assert_not_called()
        embed.assert_not_called()
        colpali_sync.assert_called_once()
        finalize.assert_called_once()
        mark_fail.assert_not_called()


class TestProcessDocumentIndexingAbort:
    """Le pipeline s'arrête proprement si run_id devient obsolète."""

    def test_aborted_on_stale_run(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4 test")

        doc = _make_document()

        with mock.patch("app.services.document_indexing_service.Session") as mock_session_cls, \
             mock.patch("app.services.document_indexing_service._delete_all_chunks"), \
             mock.patch("app.services.document_indexing_service._set_progress"), \
             mock.patch("app.services.document_indexing_service._finalize_document") as finalize, \
             mock.patch("app.services.document_indexing_service._mark_failed") as mark_fail, \
             mock.patch("app.services.document_indexing_service._extract_and_persist_chunks") as extract, \
             mock.patch("app.services.document_run.is_processing_run_current", return_value=False), \
             mock.patch("app.services.file_conversion.ensure_pdf_for_ocr", return_value=pdf_path):

            sess_instance = mock_session_cls.return_value.__enter__.return_value
            sess_instance.get.return_value = doc

            result = process_document_indexing(
                document_id=1,
                file_path=pdf_path,
                user_id=99,
                mode=IndexingMode.FULL,
                run_id="stale-run-id",
            )

        assert result["status"] == "aborted"
        extract.assert_not_called()
        finalize.assert_not_called()
        mark_fail.assert_not_called()


# ---------------------------------------------------------------------------
# Tests de l'endpoint API
# ---------------------------------------------------------------------------


class TestReindexEndpointMode:
    """Endpoint POST /api/library/documents/{id}/reindex avec body { mode }."""

    def test_default_mode_is_full(self, client, admin_headers, responsable_headers):
        """Sans body, le mode par défaut doit être full."""
        with mock.patch("app.routers.library.process_document_async"), \
             mock.patch("app.routers.library.save_uploaded_file", return_value="media/documents/ep_test.pdf"):
            r = client.post(
                "/api/library/upload",
                headers=responsable_headers,
                files=[("files", ("ep_test.pdf", b"%PDF-1.4", "application/pdf"))],
                data={"space_ids": "[]", "is_paid": "false"},
            )
        assert r.status_code == 201
        doc_id = r.json()[0]["id"]

        mock_result = mock.MagicMock()
        mock_result.id = "task-test-full"
        with mock.patch("app.tasks.documents.reindex_library_document_task.apply_async", return_value=mock_result), \
             mock.patch("app.config.settings.MISTRAL_API_KEY", "sk-test"), \
             mock.patch("app.config.settings.COLPALI_ENABLED", True):
            r2 = client.post(
                f"/api/library/documents/{doc_id}/reindex",
                headers=admin_headers,
            )
        assert r2.status_code == 200
        assert r2.json()["mode"] == "full"

    def test_text_only_mode_accepted(self, client, admin_headers, responsable_headers):
        with mock.patch("app.routers.library.process_document_async"), \
             mock.patch("app.routers.library.save_uploaded_file", return_value="media/documents/ep_txt.pdf"):
            r = client.post(
                "/api/library/upload",
                headers=responsable_headers,
                files=[("files", ("ep_txt.pdf", b"%PDF-1.4", "application/pdf"))],
                data={"space_ids": "[]", "is_paid": "false"},
            )
        assert r.status_code == 201
        doc_id = r.json()[0]["id"]

        mock_result = mock.MagicMock()
        mock_result.id = "task-test-text"
        with mock.patch("app.tasks.documents.reindex_library_document_task.apply_async", return_value=mock_result), \
             mock.patch("app.config.settings.MISTRAL_API_KEY", "sk-test"), \
             mock.patch("app.config.settings.COLPALI_ENABLED", True):
            r2 = client.post(
                f"/api/library/documents/{doc_id}/reindex",
                headers=admin_headers,
                json={"mode": "text_only"},
            )
        assert r2.status_code == 200
        assert r2.json()["mode"] == "text_only"

    def test_invalid_mode_returns_400(self, client, admin_headers, responsable_headers):
        with mock.patch("app.routers.library.process_document_async"), \
             mock.patch("app.routers.library.save_uploaded_file", return_value="media/documents/ep_inv.pdf"):
            r = client.post(
                "/api/library/upload",
                headers=responsable_headers,
                files=[("files", ("ep_inv.pdf", b"%PDF-1.4", "application/pdf"))],
                data={"space_ids": "[]", "is_paid": "false"},
            )
        assert r.status_code == 201
        doc_id = r.json()[0]["id"]

        r2 = client.post(
            f"/api/library/documents/{doc_id}/reindex",
            headers=admin_headers,
            json={"mode": "invalid_mode"},
        )
        assert r2.status_code == 400
        assert "invalide" in r2.json()["detail"].lower()

    def test_text_mode_requires_mistral_key(self, client, admin_headers, responsable_headers):
        with mock.patch("app.routers.library.process_document_async"), \
             mock.patch("app.routers.library.save_uploaded_file", return_value="media/documents/ep_nokey.pdf"):
            r = client.post(
                "/api/library/upload",
                headers=responsable_headers,
                files=[("files", ("ep_nokey.pdf", b"%PDF-1.4", "application/pdf"))],
                data={"space_ids": "[]", "is_paid": "false"},
            )
        assert r.status_code == 201
        doc_id = r.json()[0]["id"]

        with mock.patch("app.config.settings.MISTRAL_API_KEY", None):
            r2 = client.post(
                f"/api/library/documents/{doc_id}/reindex",
                headers=admin_headers,
                json={"mode": "text_only"},
            )
        assert r2.status_code == 400
        assert "MISTRAL_API_KEY" in r2.json()["detail"]

    def test_colpali_mode_requires_colpali_enabled(self, client, admin_headers, responsable_headers):
        with mock.patch("app.routers.library.process_document_async"), \
             mock.patch("app.routers.library.save_uploaded_file", return_value="media/documents/ep_nocol.pdf"):
            r = client.post(
                "/api/library/upload",
                headers=responsable_headers,
                files=[("files", ("ep_nocol.pdf", b"%PDF-1.4", "application/pdf"))],
                data={"space_ids": "[]", "is_paid": "false"},
            )
        assert r.status_code == 201
        doc_id = r.json()[0]["id"]

        with mock.patch("app.config.settings.COLPALI_ENABLED", False), \
             mock.patch("app.config.settings.MISTRAL_API_KEY", "sk-test"):
            r2 = client.post(
                f"/api/library/documents/{doc_id}/reindex",
                headers=admin_headers,
                json={"mode": "colpali_only"},
            )
        assert r2.status_code == 400
        assert "ColPali" in r2.json()["detail"]


class TestReindexAllEndpointMode:
    """Endpoint POST /api/library/reindex-all avec body { mode }."""

    def test_default_mode_is_full(self, client, admin_headers):
        mock_result = mock.MagicMock()
        mock_result.id = "task-all-full"
        with mock.patch("app.tasks.documents.reindex_all_library_documents_task.apply_async", return_value=mock_result), \
             mock.patch("app.config.settings.MISTRAL_API_KEY", "sk-test"), \
             mock.patch("app.config.settings.COLPALI_ENABLED", True):
            r = client.post("/api/library/reindex-all", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["mode"] == "full"

    def test_text_only_mode_propagated(self, client, admin_headers):
        mock_result = mock.MagicMock()
        mock_result.id = "task-all-text"
        with mock.patch("app.tasks.documents.reindex_all_library_documents_task.apply_async", return_value=mock_result), \
             mock.patch("app.config.settings.MISTRAL_API_KEY", "sk-test"), \
             mock.patch("app.config.settings.COLPALI_ENABLED", True):
            r = client.post("/api/library/reindex-all", headers=admin_headers, json={"mode": "text_only"})
        assert r.status_code == 200
        assert r.json()["mode"] == "text_only"

    def test_invalid_mode_returns_400(self, client, admin_headers):
        r = client.post("/api/library/reindex-all", headers=admin_headers, json={"mode": "bad"})
        assert r.status_code == 400


class TestBuildEmbedText:
    """Phase 4 : préfixe contextuel déterministe (titre + section + catégories +
    entités + matériau/source) injecté dans le texte embeddé, pas dans content."""

    def test_l1_includes_categories_entities_material_source(self):
        from app.services.document_indexing_service import _build_embed_text

        chunk = SimpleNamespace(
            content="Poser le profil seuil.",
            metadata_json={
                "document_title": "Notice Profine 76",
                "heading": "Pose du seuil",
                "page_no": 3,
                "categories": ["mounting"],
                "entities": ["Profine 76", "seuil PMR"],
            },
        )

        text = _build_embed_text(chunk, doc_source="Profine", doc_materials=["pvc"])

        assert "Notice Profine 76" in text
        assert "Pose du seuil" in text
        assert "Pose / montage" in text  # label de catégorie, pas le slug
        assert "Profine 76" in text  # entité
        assert "Source : Profine" in text
        assert "pvc" in text
        assert "Poser le profil seuil." in text  # content conservé
        # le préfixe précède le content
        assert text.index("Notice Profine 76") < text.index("Poser le profil seuil.")

    def test_l2_enrichment_uses_theme_and_category_slug(self):
        from app.services.document_indexing_service import _build_embed_text

        chunk = SimpleNamespace(
            content="Synthèse pose seuil.",
            metadata_json={
                "document_title": "Notice X",
                "content_type": "contextual_enrichment",
                "theme": "Pose du seuil Profine",
                "category_slug": "mounting",
                "page_no": 2,
            },
        )

        text = _build_embed_text(chunk)

        assert "Pose du seuil Profine" in text  # theme (à défaut de heading)
        assert "Pose / montage" in text  # label dérivé de category_slug
        assert "Synthèse pose seuil." in text

    def test_no_metadata_returns_content_only(self):
        from app.services.document_indexing_service import _build_embed_text

        chunk = SimpleNamespace(content="Juste du contenu.", metadata_json={})
        assert _build_embed_text(chunk) == "Juste du contenu."


class TestEmbedTextChunksL1AndL2:
    """Phase 1+4 : l'embedding (en dernier) couvre L1 semantic_leaf ET L2
    contextual_enrichment, mais pas les ancres L0."""

    def test_embeds_l1_and_l2_not_anchor(self):
        from app.services import document_indexing_service as svc

        leaf = SimpleNamespace(
            content="L1 texte",
            metadata_json={"content_type": "semantic_leaf"},
            metadata_=None,
            embedding=None,
        )
        enrich = SimpleNamespace(
            content="L2 synthèse",
            metadata_json={"content_type": "contextual_enrichment"},
            metadata_=None,
            embedding=None,
        )
        anchor = SimpleNamespace(
            content="ancre",
            metadata_json={"content_type": "page_anchor"},
            metadata_=None,
            embedding=None,
        )

        doc = SimpleNamespace(source="Profine", materials=["pvc"])

        sess = MagicMock()
        sess.get.return_value = doc
        sess.exec.return_value.all.return_value = [leaf, enrich, anchor]
        ctx = MagicMock()
        ctx.__enter__.return_value = sess
        ctx.__exit__.return_value = False

        with patch.object(svc, "Session", return_value=ctx), patch(
            "app.services.embedding_service.generate_embeddings_batch",
            return_value=[[0.1] * 1024, [0.2] * 1024],
        ) as gen:
            count = svc._embed_text_chunks(123)

        assert count == 2
        # 2 textes embeddés (L1 + L2), l'ancre L0 est exclue
        assert len(gen.call_args[0][0]) == 2
        assert leaf.embedding is not None
        assert enrich.embedding is not None
        assert anchor.embedding is None


class TestDeleteTextChunksKagCleanup:
    """Les relations KAG doivent être supprimées avant les chunks (FK)."""

    def test_deletes_kag_relations_before_chunks(self):
        from app.services.document_indexing_service import _delete_text_chunks

        session = mock.MagicMock()
        select_result = mock.MagicMock()
        select_result.all.return_value = [(49302,)]
        session.execute.side_effect = [select_result, mock.MagicMock(), mock.MagicMock()]

        with mock.patch("app.services.document_indexing_service.settings.KAG_ENABLED", True), \
             mock.patch(
                 "app.services.document_indexing_service._delete_chunk_foreign_relations",
             ) as delete_relations:
            _delete_text_chunks(session, 425)

        delete_relations.assert_called_once_with(session, [49302])
        assert session.execute.call_count == 3
        session.commit.assert_called_once()
