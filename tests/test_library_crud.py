"""Bibliothèque : upload, dossiers, documents, espaces (mocks traitement)."""
from __future__ import annotations

import json
from unittest import mock

COMPLETE_CLASSIFICATION = {
    "product_types": json.dumps(["fenetres"]),
    "materials": json.dumps(["pvc"]),
    "proferm_gammes": json.dumps(["perform"]),
    "source": "kommerling",
}


def test_upload_is_paid_true_persisted(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_paid.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("paid.txt", b"x", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "true"},
        )
    assert r.status_code == 201
    doc_id = r.json()[0]["id"]
    gr = client.get(f"/api/library/documents/{doc_id}", headers=responsable_headers)
    assert gr.status_code == 200
    assert gr.json()["is_paid"] is True


def test_upload_is_paid_false(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_free.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("free.txt", b"y", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    assert r.json()[0]["is_paid"] is False


def test_upload_invalid_space_ids_json(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/x.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("a.txt", b"z", "text/plain"))],
            data={"space_ids": "not-json", "is_paid": "false"},
        )
    assert r.status_code == 400


def test_upload_calls_process_once_per_file(client, responsable_headers):
    calls: list[tuple[int, str]] = []

    def _track(doc_id: int, path: str) -> None:
        calls.append((doc_id, path))

    with (
        mock.patch(
            "app.routers.library.process_document_async",
            side_effect=_track,
        ),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_multi.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[
                ("files", ("one.txt", b"1", "text/plain")),
                ("files", ("two.txt", b"2", "text/plain")),
            ],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    assert len(r.json()) == 2
    assert len(calls) == 2
    assert calls[0][0] != calls[1][0]


def test_folder_crud_and_move(client, responsable_headers):
    r = client.post(
        "/api/library/folders",
        headers=responsable_headers,
        json={"name": "Parent pytest", "parent_folder_id": None},
    )
    assert r.status_code == 201
    parent_id = r.json()["id"]

    r = client.post(
        "/api/library/folders",
        headers=responsable_headers,
        json={"name": "Enfant pytest", "parent_folder_id": parent_id},
    )
    assert r.status_code == 201
    child_id = r.json()["id"]

    r = client.put(
        f"/api/library/folders/{child_id}",
        headers=responsable_headers,
        json={"name": "Enfant renommé"},
    )
    assert r.status_code == 200
    assert r.json()["name"] == "Enfant renommé"

    r = client.post(
        f"/api/library/folders/{child_id}/move",
        headers=responsable_headers,
    )
    assert r.status_code == 200
    assert r.json()["parent_folder_id"] is None

    r = client.delete(f"/api/library/folders/{child_id}", headers=responsable_headers)
    assert r.status_code == 204

    r = client.delete(f"/api/library/folders/{parent_id}", headers=responsable_headers)
    assert r.status_code == 204


def test_document_move_to_folder(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/move_me.pdf",
        ),
    ):
        up = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("moveme.txt", b"doc", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert up.status_code == 201
    doc_id = up.json()[0]["id"]

    fr = client.post(
        "/api/library/folders",
        headers=responsable_headers,
        json={"name": "Dossier cible", "parent_folder_id": None},
    )
    folder_id = fr.json()["id"]

    r = client.post(
        f"/api/library/documents/{doc_id}/move",
        headers=responsable_headers,
        params={"new_folder_id": folder_id},
    )
    assert r.status_code == 200
    assert r.json()["folder_id"] == folder_id

    client.delete(f"/api/library/documents/{doc_id}", headers=responsable_headers)
    client.delete(f"/api/library/folders/{folder_id}", headers=responsable_headers)


@mock.patch("app.routers.library.process_document_async")
@mock.patch(
    "app.routers.library.save_uploaded_file",
    return_value="media/documents/spaces_doc.pdf",
)
def test_document_add_and_remove_space(mock_save, mock_proc, client, responsable_headers):
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace doc link"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]

    up = client.post(
        "/api/library/upload",
        headers=responsable_headers,
        files=[("files", ("linked.txt", b"t", "text/plain"))],
        data={"space_ids": "[]", "is_paid": "false"},
    )
    assert up.status_code == 201
    doc_id = up.json()[0]["id"]

    with mock.patch(
        "app.routers.library.dispatch_document_spaces_update",
        return_value="celery-task-doc-spaces-1",
    ) as dispatch_mock:
        r = client.post(
            f"/api/library/documents/{doc_id}/spaces",
            headers=responsable_headers,
            json={"add_space_ids": [space_id], "remove_space_ids": []},
        )
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "queued"
    assert payload["task_id"] == "celery-task-doc-spaces-1"
    dispatch_mock.assert_called_once_with(
        document_id=doc_id,
        add_space_ids=[space_id],
        remove_space_ids=[],
        user_id=mock.ANY,
    )

    with mock.patch(
        "app.routers.library.dispatch_document_spaces_update",
        return_value="celery-task-doc-spaces-2",
    ) as dispatch_mock:
        r = client.post(
            f"/api/library/documents/{doc_id}/spaces",
            headers=responsable_headers,
            json={"add_space_ids": [], "remove_space_ids": [space_id]},
        )
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "queued"
    assert payload["task_id"] == "celery-task-doc-spaces-2"
    dispatch_mock.assert_called_once_with(
        document_id=doc_id,
        add_space_ids=[],
        remove_space_ids=[space_id],
        user_id=mock.ANY,
    )

    client.delete(f"/api/library/documents/{doc_id}", headers=responsable_headers)
    client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)


@mock.patch("app.routers.library.process_document_async")
@mock.patch(
    "app.routers.library.save_uploaded_file",
    return_value="media/documents/spaces_noop_doc.pdf",
)
def test_document_spaces_noop_does_not_dispatch(
    mock_save, mock_proc, client, responsable_headers
):
    up = client.post(
        "/api/library/upload",
        headers=responsable_headers,
        files=[("files", ("linked.txt", b"t", "text/plain"))],
        data={"space_ids": "[]", "is_paid": "false"},
    )
    assert up.status_code == 201
    doc_id = up.json()[0]["id"]

    with mock.patch("app.routers.library.dispatch_document_spaces_update") as dispatch_mock:
        r = client.post(
            f"/api/library/documents/{doc_id}/spaces",
            headers=responsable_headers,
            json={"add_space_ids": [], "remove_space_ids": []},
        )

    assert r.status_code == 200
    assert r.json()["status"] == "noop"
    dispatch_mock.assert_not_called()

    client.delete(f"/api/library/documents/{doc_id}", headers=responsable_headers)


def test_document_export_and_import(client, responsable_headers):
    import io
    import zipfile
    import json
    import os

    # Ensure media directory exists and write dummy file
    os.makedirs("media/documents", exist_ok=True)
    pdf_path = "media/documents/export_test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"pdfcontent")

    # 1. Create a dummy document
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value=pdf_path,
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("export_test.pdf", b"pdfcontent", "application/pdf"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
        assert r.status_code == 201
        doc_id = r.json()[0]["id"]

        # Create a dummy chunk in the SQL database for this document
        from app.models.document_chunk import DocumentChunk
        from sqlmodel import Session, select
        from app.database import engine
        from app.models.document import Document

        with Session(engine) as sess:
            db_doc = sess.get(Document, doc_id)
            db_doc.source = "Proferm"
            sess.add(db_doc)
            sess.commit()

            chunk = DocumentChunk(
                document_id=doc_id,
                chunk_index=0,
                content="dummy chunk text",
                text="dummy chunk text",
                is_leaf=True,
                hierarchy_level=0,
                node_id="node_1",
                start_char=0,
                end_char=16,
                metadata_json={"page_no": 1},
                source="Proferm"
            )
            sess.add(chunk)
            sess.commit()
            sess.refresh(chunk)
            old_chunk_id = chunk.id

        # Mock LanceDB table search results for export
        mock_table = mock.Mock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"chunk_id": old_chunk_id, "patch_index": 0, "vector": [0.5] * 128}
        ]

        with mock.patch("app.services.lancedb_service.get_colpali_table", return_value=mock_table):
            # Request export
            expr = client.get(f"/api/library/documents/{doc_id}/export", headers=responsable_headers)
            assert expr.status_code == 200
            assert expr.headers["content-type"] == "application/zip"

            # Check ZIP contents
            zip_bytes = expr.content
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                namelist = z.namelist()
                assert "metadata.json" in namelist
                assert "colpali_patches.json" in namelist
                assert "export_test.pdf" in namelist

                # Read metadata
                meta = json.loads(z.read("metadata.json").decode("utf-8"))
                assert meta["document"]["title"] == "export_test"
                assert meta["document"]["source"] == "Proferm"
                assert len(meta["chunks"]) == 1
                assert meta["chunks"][0]["source"] == "Proferm"

            # 2. Test Import using the exported ZIP
            with (
                mock.patch("app.routers.library.save_uploaded_file", return_value="media/documents/import_test.pdf"),
                mock.patch("app.services.lancedb_service.insert_colpali_patches_batch_lancedb") as mock_insert,
            ):
                impr = client.post(
                    "/api/library/documents/import",
                    headers=responsable_headers,
                    files={"file": ("export_test.zip", zip_bytes, "application/zip")},
                )
                assert impr.status_code == 201
                imported_doc = impr.json()
                assert imported_doc["title"] == "export_test"

                # Verify it calls batch insert to LanceDB
                mock_insert.assert_called_once()

                # Verify it persisted source in Postgres
                with Session(engine) as sess:
                    imp_doc = sess.get(Document, imported_doc["id"])
                    assert imp_doc.source == "Proferm"
                    
                    imp_chunks = list(sess.exec(select(DocumentChunk).where(DocumentChunk.document_id == imported_doc["id"])).all())
                    assert len(imp_chunks) == 1
                    assert imp_chunks[0].source == "Proferm"

        # Cleanup
        client.delete(f"/api/library/documents/{doc_id}", headers=responsable_headers)
        client.delete(f"/api/library/documents/{imported_doc['id']}", headers=responsable_headers)
        
        # Delete dummy file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists("media/documents/import_test.pdf"):
                os.remove("media/documents/import_test.pdf")
        except Exception:
            pass


def test_library_export_all_and_import(client, responsable_headers):
    import io
    import zipfile
    import json
    import os

    # Ensure media directory exists and write dummy file
    os.makedirs("media/documents", exist_ok=True)
    pdf_path = "media/documents/export_all_test.pdf"
    with open(pdf_path, "wb") as f:
        f.write(b"pdfcontentall")

    # 1. Create a dummy document
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value=pdf_path,
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("export_all_test.pdf", b"pdfcontentall", "application/pdf"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
        assert r.status_code == 201
        doc_id = r.json()[0]["id"]

        # Create a dummy chunk in the SQL database for this document
        from app.models.document_chunk import DocumentChunk
        from sqlmodel import Session, select
        from app.database import engine
        from app.models.document import Document

        with Session(engine) as sess:
            db_doc = sess.get(Document, doc_id)
            db_doc.source = "Technal"
            sess.add(db_doc)
            sess.commit()

            chunk = DocumentChunk(
                document_id=doc_id,
                chunk_index=0,
                content="dummy chunk text for all",
                text="dummy chunk text for all",
                is_leaf=True,
                hierarchy_level=0,
                node_id="node_all_1",
                start_char=0,
                end_char=24,
                metadata_json={"page_no": 1},
                source="Technal"
            )
            sess.add(chunk)
            sess.commit()
            sess.refresh(chunk)
            old_chunk_id = chunk.id

        # Mock LanceDB table search results for export
        mock_table = mock.Mock()
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"chunk_id": old_chunk_id, "patch_index": 0, "vector": [0.7] * 128}
        ]

        with mock.patch("app.services.lancedb_service.get_colpali_table", return_value=mock_table):
            # Request export-all
            expr = client.get("/api/library/export-all", headers=responsable_headers)
            assert expr.status_code == 200
            assert expr.headers["content-type"] == "application/zip"

            # Check ZIP contents
            zip_bytes = expr.content
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                namelist = z.namelist()
                metadata_file = next((f for f in namelist if f.endswith("metadata.json")), None)
                assert metadata_file is not None
                assert metadata_file.startswith("doc_")
                
                patches_file = next((f for f in namelist if f.endswith("colpali_patches.json")), None)
                assert patches_file is not None

                # Read metadata
                meta = json.loads(z.read(metadata_file).decode("utf-8"))
                assert meta["document"]["title"] == "export_all_test"
                assert meta["document"]["source"] == "Technal"
                assert meta["chunks"][0]["source"] == "Technal"

            # 2. Test Import using the exported ZIP (bulk mode)
            with (
                mock.patch("app.routers.library.save_uploaded_file", return_value="media/documents/import_bulk_test.pdf"),
                mock.patch("app.services.lancedb_service.insert_colpali_patches_batch_lancedb") as mock_insert,
            ):
                impr = client.post(
                    "/api/library/documents/import",
                    headers=responsable_headers,
                    files={"file": ("export_library_all.zip", zip_bytes, "application/zip")},
                )
                assert impr.status_code == 201
                imported_doc = impr.json()
                assert imported_doc["title"] == "export_all_test"

                # Verify it calls batch insert to LanceDB
                mock_insert.assert_called_once()

                # Verify it persisted source in Postgres
                with Session(engine) as sess:
                    imp_doc = sess.get(Document, imported_doc["id"])
                    assert imp_doc.source == "Technal"
                    
                    imp_chunks = list(sess.exec(select(DocumentChunk).where(DocumentChunk.document_id == imported_doc["id"])).all())
                    assert len(imp_chunks) == 1
                    assert imp_chunks[0].source == "Technal"

        # Cleanup
        client.delete(f"/api/library/documents/{doc_id}", headers=responsable_headers)
        client.delete(f"/api/library/documents/{imported_doc['id']}", headers=responsable_headers)
        
        # Delete dummy file
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists("media/documents/import_bulk_test.pdf"):
                os.remove("media/documents/import_bulk_test.pdf")
        except Exception:
            pass


def test_classification_options_endpoint(client, responsable_headers):
    r = client.get("/api/library/classification-options", headers=responsable_headers)
    assert r.status_code == 200
    data = r.json()
    assert "product_types" in data
    assert "suppliers" in data
    assert any(s["value"] == "kommerling" for s in data["suppliers"])


def test_upload_single_with_classification_complete(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_classified.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("classified.txt", b"x", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false", **COMPLETE_CLASSIFICATION},
        )
    assert r.status_code == 201
    doc = r.json()[0]
    assert doc["classification_status"] == "complete"
    assert doc["product_types"] == ["fenetres"]
    assert doc["materials"] == ["pvc"]
    assert doc["proferm_gammes"] == ["perform"]
    assert doc["source"] == "Kommerling"


def test_upload_bulk_without_classification_incomplete(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_bulk.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[
                ("files", ("a.txt", b"a", "text/plain")),
                ("files", ("b.txt", b"b", "text/plain")),
            ],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    for doc in r.json():
        assert doc["classification_status"] == "incomplete"


def test_update_document_classification(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_update_class.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("update.txt", b"x", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    doc_id = r.json()[0]["id"]
    assert r.json()[0]["classification_status"] == "incomplete"

    upd = client.put(
        f"/api/library/documents/{doc_id}",
        headers=responsable_headers,
        json={
            "product_types": ["portes"],
            "materials": ["aluminium"],
            "proferm_gammes": ["lumine"],
            "source": "technal",
        },
    )
    assert upd.status_code == 200
    body = upd.json()
    assert body["classification_status"] == "complete"
    assert body["source"] == "Technal"
