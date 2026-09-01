"""Topologie ColPali unique : 1 page = 1 jeu de patches, rattaché à l'anchor de page.

Couvre l'écart n°1 de l'audit ColPali (2026-09-01) :
  - ensure_page_anchors : création/mise à jour idempotentes des L0 page_anchor ;
  - repair_colpali_topology : remap in-place des patches hérités du pipeline
    feuille (copies par chunk texte) vers les anchors, SANS ré-embedding ;
  - ensure_colpali_page_sync : un retraitement multimodal ne ré-embedde pas un
    document dont les patches sont déjà sains ;
  - delete_multimodal_chunks_for_document : ne touche plus à LanceDB.
"""
from __future__ import annotations

from unittest import mock

import pytest
from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.library import Library
from app.services.document_indexing_service import (
    CONTENT_TYPE_PAGE_ANCHOR,
    ensure_page_anchors,
    repair_colpali_topology,
)


@pytest.fixture(autouse=True)
def _colpali_on(monkeypatch):
    monkeypatch.setattr(settings, "COLPALI_ENABLED", True)


def _make_document(db_session: Session, *, source_file_path: str | None = None) -> Document:
    from tests.conftest import create_test_user

    user = create_test_user(db_session, "admin")
    lib = Library(user_id=user.id, name="Lib topo", is_global=True)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)

    doc = Document(
        library_id=lib.id,
        user_id=user.id,
        title="Doc topo.pdf",
        document_type="document",
        source_file_path=source_file_path,
        processing_status="completed",
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


def _add_leaf(db_session: Session, doc: Document, page_no: int, content_type: str) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=100 + page_no,
        content=f"feuille p{page_no}",
        text=f"feuille p{page_no}",
        node_id=f"leaf-{doc.id}-{page_no}-{content_type}",
        is_leaf=True,
        hierarchy_level=1,
        metadata_json={"content_type": content_type, "page_no": page_no},
        metadata_={"content_type": content_type, "page_no": page_no},
    )
    db_session.add(chunk)
    db_session.commit()
    db_session.refresh(chunk)
    return chunk


def _anchors_of(db_session: Session, doc: Document) -> dict[int, DocumentChunk]:
    rows = db_session.exec(
        select(DocumentChunk).where(
            DocumentChunk.document_id == doc.id,
            DocumentChunk.is_leaf == False,  # noqa: E712
        )
    ).all()
    return {
        int((c.metadata_json or {}).get("page_no")): c
        for c in rows
        if (c.metadata_json or {}).get("content_type") == CONTENT_TYPE_PAGE_ANCHOR
    }


class TestEnsurePageAnchors:
    def test_cree_un_anchor_par_page(self, db_session: Session):
        doc = _make_document(db_session)
        anchors, created, updated = ensure_page_anchors(db_session, doc, 3)
        db_session.commit()

        assert created == 3
        assert updated == 0
        assert sorted(anchors) == [1, 2, 3]
        stored = _anchors_of(db_session, doc)
        assert sorted(stored) == [1, 2, 3]
        assert stored[2].node_id == f"page-anchor-{doc.id}-2"
        assert stored[2].metadata_json["page_start"] == 2

    def test_idempotent_et_preserve_les_ids(self, db_session: Session):
        doc = _make_document(db_session)
        first, created1, _ = ensure_page_anchors(db_session, doc, 2)
        db_session.commit()
        ids_before = {p: c.id for p, c in first.items()}

        second, created2, updated2 = ensure_page_anchors(db_session, doc, 2)
        db_session.commit()

        assert created1 == 2
        assert created2 == 0
        assert updated2 == 0  # update_existing=False par défaut
        assert {p: c.id for p, c in second.items()} == ids_before

    def test_update_existing_reecrit_le_contenu(self, db_session: Session):
        doc = _make_document(db_session)
        ensure_page_anchors(db_session, doc, 1)
        db_session.commit()

        anchors, created, updated = ensure_page_anchors(
            db_session,
            doc,
            1,
            headings_by_page={1: "Chapitre pose"},
            update_existing=True,
        )
        db_session.commit()

        assert created == 0
        assert updated == 1
        assert anchors[1].content == "Chapitre pose"


class TestRepairColpaliTopology:
    def test_remap_feuilles_vers_anchors_sans_reembedding(self, db_session: Session):
        """Cas pollué type : patches sur les feuilles multimodales → remap sur anchors."""
        doc = _make_document(db_session)
        anchors, _, _ = ensure_page_anchors(db_session, doc, 2)
        db_session.commit()
        anchor_ids = {p: c.id for p, c in anchors.items()}

        leaf_p1a = _add_leaf(db_session, doc, 1, "page_raw_enriched")
        leaf_p1b = _add_leaf(db_session, doc, 1, "page_window_report")
        leaf_p2 = _add_leaf(db_session, doc, 2, "page_raw_enriched")

        lancedb_ids = {leaf_p1a.id, leaf_p1b.id, leaf_p2.id}
        patches_p1 = [[0.1] * 128, [0.2] * 128]
        patches_p2 = [[0.3] * 128]
        source_p1 = min(leaf_p1a.id, leaf_p1b.id)

        inserted: dict = {}

        def _capture_insert(document_id, chunk_patches_list):
            inserted["document_id"] = document_id
            inserted["batch"] = chunk_patches_list

        with (
            mock.patch(
                "app.services.lancedb_service.get_colpali_chunk_ids_by_document",
                return_value={doc.id: set(lancedb_ids)},
            ),
            mock.patch(
                "app.services.lancedb_service.fetch_colpali_patch_vectors_for_chunks",
                return_value={source_p1: patches_p1, leaf_p2.id: patches_p2},
            ) as fetch_mock,
            mock.patch(
                "app.services.lancedb_service.insert_colpali_patches_batch_lancedb",
                side_effect=_capture_insert,
            ),
            mock.patch(
                "app.services.colpali_service.embed_pdf_pages_colpali"
            ) as embed_mock,
        ):
            result = repair_colpali_topology(doc.id)

        assert result["status"] == "repaired"
        assert result["pages_with_patches"] == 2
        # Le remap est du pur I/O : jamais de passage modèle.
        embed_mock.assert_not_called()
        # UN chunk source par page (le plus petit id, déterministe).
        fetch_mock.assert_called_once_with(sorted({source_p1, leaf_p2.id}))
        # Réinsertion sous les anchors, patches intacts.
        batch = dict(inserted["batch"])
        assert set(batch) == {anchor_ids[1], anchor_ids[2]}
        assert batch[anchor_ids[1]] == patches_p1
        assert batch[anchor_ids[2]] == patches_p2

    def test_topologie_saine_ne_reecrit_rien(self, db_session: Session):
        doc = _make_document(db_session)
        anchors, _, _ = ensure_page_anchors(db_session, doc, 2)
        db_session.commit()
        anchor_id_set = {c.id for c in anchors.values()}

        with (
            mock.patch(
                "app.services.lancedb_service.get_colpali_chunk_ids_by_document",
                return_value={doc.id: set(anchor_id_set)},
            ),
            mock.patch(
                "app.services.lancedb_service.insert_colpali_patches_batch_lancedb"
            ) as insert_mock,
        ):
            result = repair_colpali_topology(doc.id)

        assert result["status"] == "ok"
        insert_mock.assert_not_called()

    def test_orphelins_seuls_purge_et_incomplete(self, db_session: Session):
        """Uniquement des patches orphelins (chunks supprimés) : purge, pas de remap."""
        doc = _make_document(db_session)
        ensure_page_anchors(db_session, doc, 1)
        db_session.commit()

        with (
            mock.patch(
                "app.services.lancedb_service.get_colpali_chunk_ids_by_document",
                return_value={doc.id: {999_991, 999_992}},
            ),
            mock.patch(
                "app.services.lancedb_service.fetch_colpali_patch_vectors_for_chunks",
                return_value={},
            ),
            mock.patch(
                "app.services.lancedb_service.delete_colpali_patches_for_document"
            ) as purge_mock,
            mock.patch(
                "app.services.lancedb_service.insert_colpali_patches_batch_lancedb"
            ) as insert_mock,
        ):
            result = repair_colpali_topology(doc.id)

        assert result["status"] == "incomplete"
        assert result["orphan_targets"] == 2
        purge_mock.assert_called_once_with(doc.id)
        insert_mock.assert_not_called()


class TestEnsureColpaliPageSync:
    def test_document_sain_pas_de_reembedding(self, db_session: Session, tmp_path):
        """Retraitement multimodal d'un doc sain : zéro forward ColQwen2."""
        pdf = tmp_path / "sain.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal")
        doc = _make_document(db_session, source_file_path=str(pdf))
        anchors, _, _ = ensure_page_anchors(db_session, doc, 2)
        db_session.commit()
        anchor_id_set = {c.id for c in anchors.values()}

        from app.services.document_indexing_service import ensure_colpali_page_sync

        with (
            mock.patch(
                "app.services.lancedb_service.get_colpali_chunk_ids_by_document",
                return_value={doc.id: set(anchor_id_set)},
            ),
            mock.patch(
                "app.services.document_indexing_service._get_pdf_page_count",
                return_value=2,
            ),
            mock.patch(
                "app.services.document_indexing_service.sync_colpali_page_anchors"
            ) as sync_mock,
            mock.patch(
                "app.services.colpali_service.embed_pdf_pages_colpali"
            ) as embed_mock,
        ):
            result = ensure_colpali_page_sync(doc.id, str(pdf))

        assert result["status"] == "ok"
        assert result["pages"] == 2
        sync_mock.assert_not_called()
        embed_mock.assert_not_called()

    def test_aucun_patch_declenche_le_sync_complet(self, db_session: Session, tmp_path):
        pdf = tmp_path / "vide.pdf"
        pdf.write_bytes(b"%PDF-1.4 minimal")
        doc = _make_document(db_session, source_file_path=str(pdf))

        from app.services.document_indexing_service import ensure_colpali_page_sync

        with (
            mock.patch(
                "app.services.lancedb_service.get_colpali_chunk_ids_by_document",
                return_value={doc.id: set()},
            ),
            mock.patch(
                "app.services.document_indexing_service._get_pdf_page_count",
                return_value=2,
            ),
            mock.patch(
                "app.services.document_indexing_service.sync_colpali_page_anchors",
                return_value=2,
            ) as sync_mock,
        ):
            result = ensure_colpali_page_sync(doc.id, str(pdf))

        assert result["status"] == "resynced"
        assert result["pages"] == 2
        sync_mock.assert_called_once_with(doc.id, str(pdf))


class TestColpaliRepairDocumentEndpoint:
    def test_repare_un_document(self, client, db_session: Session):
        from tests.conftest import bearer_headers, create_test_user
        from app.services.library_service import get_or_create_user_library

        admin = create_test_user(db_session, "admin")
        headers = bearer_headers(admin.id)
        library = get_or_create_user_library(db_session, admin.id)
        doc = Document(
            library_id=library.id,
            user_id=admin.id,
            title="A réparer.pdf",
            document_type="document",
            processing_status="completed",
        )
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        with mock.patch(
            "app.services.document_indexing_service.repair_colpali_topology",
            return_value={
                "document_id": doc.id,
                "status": "repaired",
                "pages_with_patches": 3,
                "expected_pages": 3,
            },
        ) as repair_mock:
            r = client.post(
                f"/api/library/documents/{doc.id}/colpali-repair", headers=headers
            )

        assert r.status_code == 200
        assert r.json()["status"] == "repaired"
        repair_mock.assert_called_once_with(doc.id)

    def test_refuse_non_admin(self, client, db_session: Session):
        from tests.conftest import bearer_headers, create_test_user

        user = create_test_user(db_session, "lecteur")
        r = client.post(
            "/api/library/documents/999999/colpali-repair",
            headers=bearer_headers(user.id),
        )
        assert r.status_code == 403


class TestMultimodalNeTouchePlusLanceDB:
    def test_delete_multimodal_chunks_preserve_les_patches(self, db_session: Session):
        doc = _make_document(db_session)
        _add_leaf(db_session, doc, 1, "page_raw_enriched")

        from app.services.multimodal_page_service import (
            delete_multimodal_chunks_for_document,
        )

        with mock.patch(
            "app.services.lancedb_service.delete_chunks_lancedb"
        ) as lancedb_delete:
            deleted = delete_multimodal_chunks_for_document(db_session, doc.id)

        assert deleted == 1
        lancedb_delete.assert_not_called()
