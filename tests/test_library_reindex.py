"""Réindexation bibliothèque : endpoint POST /reindex (Celery uniquement)."""
from __future__ import annotations

import os
import tempfile
from unittest import mock

from app.services.document_service_new import (
    DOCUMENT_STATUS_REINDEX_QUEUED,
    mark_all_eligible_documents_reindex_queued,
    reindex_all_library_documents,
)


def test_reindex_endpoint_returns_queued(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_reindex.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("reindex.txt", b"hello", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    doc_id = r.json()[0]["id"]

    mock_result = mock.MagicMock()
    mock_result.id = "task-reindex-xyz"
    with mock.patch(
        "app.tasks.documents.reindex_library_document_task.apply_async",
        return_value=mock_result,
    ):
        r2 = client.post(
            f"/api/library/documents/{doc_id}/reindex",
            headers=responsable_headers,
        )

    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "queued"
    assert body["celery_task_id"] == "task-reindex-xyz"
    assert body["document_id"] == doc_id


def test_reindex_forbidden_without_library_write(client, lecteur_headers):
    r = client.post(
        "/api/library/documents/1/reindex",
        headers=lecteur_headers,
    )
    assert r.status_code == 403


def test_reindex_all_endpoint_returns_queued(client, responsable_headers):
    mock_result = mock.MagicMock()
    mock_result.id = "task-reindex-all-abc"
    with mock.patch(
        "app.tasks.documents.reindex_all_library_documents_task.apply_async",
        return_value=mock_result,
    ):
        r = client.post("/api/library/reindex-all", headers=responsable_headers)

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["celery_task_id"] == "task-reindex-all-abc"


def test_reindex_all_forbidden_lecteur(client, lecteur_headers):
    r = client.post("/api/library/reindex-all", headers=lecteur_headers)
    assert r.status_code == 403


def test_reindex_service_unavailable_returns_503(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_reindex_503.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("a.txt", b"x", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    doc_id = r.json()[0]["id"]

    with mock.patch(
        "app.tasks.documents.reindex_library_document_task.apply_async",
        side_effect=ConnectionError("broker down"),
    ):
        r2 = client.post(
            f"/api/library/documents/{doc_id}/reindex",
            headers=responsable_headers,
        )

    assert r2.status_code == 503
    assert "Celery" in (r2.json().get("detail") or "")


def test_post_reindex_sets_reindex_queued_status(client, responsable_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_reindex_status.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("status.txt", b"hello", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    doc_id = r.json()[0]["id"]

    mock_result = mock.MagicMock()
    mock_result.id = "task-reindex-status"
    with mock.patch(
        "app.tasks.documents.reindex_library_document_task.apply_async",
        return_value=mock_result,
    ):
        r2 = client.post(
            f"/api/library/documents/{doc_id}/reindex",
            headers=responsable_headers,
        )
    assert r2.status_code == 200

    gr = client.get(f"/api/library/documents/{doc_id}", headers=responsable_headers)
    assert gr.status_code == 200
    assert gr.json()["processing_status"] == DOCUMENT_STATUS_REINDEX_QUEUED


def test_reindex_document_not_found_returns_404(client, responsable_headers):
    r = client.post(
        "/api/library/documents/999999999/reindex",
        headers=responsable_headers,
    )
    assert r.status_code == 404


def test_mark_all_eligible_sets_reindex_queued_when_file_on_disk(client, responsable_headers):
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(b"hello pytest")
        with (
            mock.patch("app.routers.library.process_document_async"),
            mock.patch("app.routers.library.save_uploaded_file", return_value=path),
        ):
            r = client.post(
                "/api/library/upload",
                headers=responsable_headers,
                files=[("files", ("disk.txt", b"x", "text/plain"))],
                data={"space_ids": "[]", "is_paid": "false"},
            )
        assert r.status_code == 201
        doc_id = r.json()[0]["id"]

        me = client.get("/api/auth/me", headers=responsable_headers)
        assert me.status_code == 200
        user_id = me.json()["id"]

        n = mark_all_eligible_documents_reindex_queued(user_id)
        assert n >= 1

        gr = client.get(f"/api/library/documents/{doc_id}", headers=responsable_headers)
        assert gr.json()["processing_status"] == DOCUMENT_STATUS_REINDEX_QUEUED
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_reindex_all_worker_marks_queued_and_invokes_reindex_per_doc(client, responsable_headers, tmp_path):
    fake_file = tmp_path / "reindex_all.txt"
    fake_file.write_bytes(b"content")
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value=str(fake_file),
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("a.txt", b"z", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201

    me = client.get("/api/auth/me", headers=responsable_headers)
    user_id = me.json()["id"]

    calls: list[tuple[int, int]] = []

    def _fake_reindex(document_id: int, uid: int):
        calls.append((document_id, uid))
        return {"document_id": document_id, "chunks": 1, "status": "completed"}

    with mock.patch(
        "app.services.document_service_new.reindex_library_document",
        side_effect=_fake_reindex,
    ):
        out = reindex_all_library_documents(user_id)

    assert out.get("marked_queued", 0) >= 1
    assert out["ok"] >= 1
    assert len(calls) >= 1
    assert all(uid == user_id for _, uid in calls)
