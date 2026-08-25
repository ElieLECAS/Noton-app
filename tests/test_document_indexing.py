"""Tests unitaires du nouveau pipeline d'indexation documentaire.

Couvre :
- Les 3 modes (full, text_only, colpali_only) via le service orchestrateur
- L'endpoint API POST /reindex avec body { mode }
- L'endpoint API POST /reindex-all avec body { mode }
- La validation des prérequis (MISTRAL_API_KEY, COLPALI_ENABLED)
- L'upload via _process_document_for_id → mode full
"""
from __future__ import annotations

from unittest import mock

import pytest

from app.services.document_indexing_service import (
    IndexingMode,
    _strip_db_unsafe_chars,
    process_document_indexing,
)


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
    """Mode full : extraction texte + ColPali."""

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
        colpali_sync.assert_called_once()
        finalize.assert_called_once()
        mark_fail.assert_not_called()


class TestProcessDocumentIndexingTextOnly:
    """Mode text_only : extraction texte seule, ColPali ignoré."""

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


class TestSeparationCouchesSemantiques:
    """Répartition rapide/lent : text_only n'exécute QUE l'extraction,
    enrichment_only porte les chunks contextuels (passage de nuit)."""

    def _run(self, tmp_path, mode):
        pdf_path = str(tmp_path / "test.pdf")
        (tmp_path / "test.pdf").write_bytes(b"%PDF-1.4 test")
        doc = _make_document()

        with mock.patch("app.services.document_indexing_service.Session") as mock_session_cls, \
             mock.patch("app.services.document_indexing_service._delete_all_chunks"), \
             mock.patch("app.services.document_indexing_service._delete_text_chunks"), \
             mock.patch("app.services.document_indexing_service._set_progress"), \
             mock.patch("app.services.document_indexing_service._finalize_document"), \
             mock.patch("app.services.document_indexing_service._mark_failed"), \
             mock.patch("app.services.document_indexing_service._sync_colpali_for_pages"), \
             mock.patch("app.services.document_indexing_service._extract_and_persist_chunks", return_value=3), \
             mock.patch("app.config.settings.KAG_ENABLED", True), \
             mock.patch("app.config.settings.CONTEXTUAL_ENRICHMENT_ENABLED", True), \
             mock.patch("app.services.kag_extraction_service.extract_kag_for_document",
                        return_value={"entities": 1, "relations": 1, "status": "ok"}) as kag, \
             mock.patch("app.services.kag_extraction_service.embed_kag_entities_for_document"), \
             mock.patch("app.services.kag_extraction_service.cleanup_kag_for_document"), \
             mock.patch("app.services.contextual_enrichment_service.run_contextual_enrichment_for_document",
                        return_value={"chunks": 4, "status": "ok"}) as enrich, \
             mock.patch("app.services.document_run.is_processing_run_current", return_value=True), \
             mock.patch("app.services.file_conversion.ensure_pdf_for_ocr", return_value=pdf_path):

            mock_session_cls.return_value.__enter__.return_value.get.return_value = doc
            result = process_document_indexing(
                document_id=1, file_path=pdf_path, user_id=99, mode=mode
            )
        return result, enrich

    def test_text_only_ne_lance_pas_les_chunks_contextuels(self, tmp_path):
        result, enrich = self._run(tmp_path, IndexingMode.TEXT_ONLY)

        assert result["status"] == "completed"
        enrich.assert_not_called()
        assert "enrichment" not in result

    def test_enrichment_only_lance_les_chunks_contextuels(self, tmp_path):
        result, enrich = self._run(tmp_path, IndexingMode.ENRICHMENT_ONLY)

        enrich.assert_called_once()
        assert result["enrichment"]["status"] == "ok"

    def test_full_lance_tout(self, tmp_path):
        result, enrich = self._run(tmp_path, IndexingMode.FULL)

        enrich.assert_called_once()


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


class TestStripDbUnsafeChars:
    """Régression production (28/07) : un document de 400 pages a fait échouer TOUT
    son commit de chunks — y compris les 400+ pages déjà extraites avec succès —
    parce qu'une seule page contenait un octet NUL (0x00) dans sa couche texte
    native (police corrompue / table CID cassée, artefact connu de certains PDF
    convertis). Postgres/psycopg rejette le NUL dans une colonne texte, côté client,
    avant même d'atteindre le serveur."""

    def test_retire_le_nul(self):
        assert _strip_db_unsafe_chars("avant\x00milieu\x00fin") == "avantmilieufin"

    def test_retire_les_autres_caracteres_de_controle(self):
        # \x01 (SOH), \x1f (US) : non imprimables, jamais légitimes dans du texte.
        assert _strip_db_unsafe_chars("a\x01b\x1fc") == "abc"

    def test_conserve_les_espaces_blancs_legitimes(self):
        assert _strip_db_unsafe_chars("ligne1\nligne2\ttab\rretour") == (
            "ligne1\nligne2\ttab\rretour"
        )

    def test_texte_sans_caractere_de_controle_inchange(self):
        texte = "Profil 76180, largeur 70 mm."
        assert _strip_db_unsafe_chars(texte) == texte

    def test_chaine_vide_et_none(self):
        assert _strip_db_unsafe_chars("") == ""
        assert _strip_db_unsafe_chars(None) is None

    def test_resultat_toujours_insertable_par_postgres(self, db_session):
        """Contrôle bout en bout : le texte nettoyé passe réellement un INSERT,
        là où le texte brut aurait levé ValueError côté psycopg."""
        from sqlalchemy import text as sql_text

        sale = "Notice page 12\x00 suite du texte"
        propre = _strip_db_unsafe_chars(sale)

        with pytest.raises(ValueError, match="NUL"):
            db_session.execute(
                sql_text("SELECT :v AS v"), {"v": sale}
            )

        result = db_session.execute(sql_text("SELECT :v AS v"), {"v": propre}).first()
        assert result.v == "Notice page 12 suite du texte"


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
