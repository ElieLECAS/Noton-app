"""Réindexation bibliothèque : endpoint POST /reindex (Celery uniquement)."""
from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest
from sqlmodel import Session, select

from app.models.document import Document
from app.models.library import Library
from app.services.document_service_new import (
    DOCUMENT_STATUS_REINDEX_QUEUED,
    mark_all_eligible_documents_reindex_queued,
    mark_folder_documents_reindex_queued,
    reindex_all_library_documents,
    reindex_library_document,
)
from tests.conftest import create_test_user


def test_reindex_endpoint_returns_queued(client, responsable_headers, admin_headers):
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
            headers=admin_headers,
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


def test_reindex_all_endpoint_returns_queued(client, admin_headers):
    mock_result = mock.MagicMock()
    mock_result.id = "task-reindex-all-abc"
    with mock.patch(
        "app.tasks.documents.reindex_all_library_documents_task.apply_async",
        return_value=mock_result,
    ):
        r = client.post("/api/library/reindex-all", headers=admin_headers)

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["celery_task_id"] == "task-reindex-all-abc"


def test_reindex_all_forbidden_lecteur(client, lecteur_headers):
    r = client.post("/api/library/reindex-all", headers=lecteur_headers)
    assert r.status_code == 403


def test_reindex_all_forbidden_non_admin_responsable(client, responsable_headers):
    """Réindexation globale : réservée au rôle admin (pas seulement library.write)."""
    r = client.post("/api/library/reindex-all", headers=responsable_headers)
    assert r.status_code == 403
    assert "admin" in (r.json().get("detail") or "").lower()


def test_reindex_service_unavailable_returns_503(
    client, responsable_headers, admin_headers
):
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

    with (
        mock.patch("app.services.task_dispatch.get_task_backend_mode", return_value="celery"),
        mock.patch(
            "app.tasks.documents.reindex_library_document_task.apply_async",
            side_effect=ConnectionError("broker down"),
        ),
    ):
        r2 = client.post(
            f"/api/library/documents/{doc_id}/reindex",
            headers=admin_headers,
        )

    assert r2.status_code == 503
    assert "Celery" in (r2.json().get("detail") or "")


def test_post_reindex_sets_reindex_queued_status(client, responsable_headers, admin_headers):
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
            headers=admin_headers,
        )
    assert r2.status_code == 200

    gr = client.get(f"/api/library/documents/{doc_id}", headers=responsable_headers)
    assert gr.status_code == 200
    assert gr.json()["processing_status"] == DOCUMENT_STATUS_REINDEX_QUEUED


def test_reindex_document_not_found_returns_404(client, admin_headers):
    r = client.post(
        "/api/library/documents/999999999/reindex",
        headers=admin_headers,
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


# --------------------------------------------------------------------------- #
# Retraitement par dossier                                                     #
# --------------------------------------------------------------------------- #


def _upload_doc_with_real_file(client, headers, tmp_path, name: str) -> int:
    """Upload un document dont le source_file_path existe sur disque (éligible retraitement)."""
    fake_file = tmp_path / f"{name}.txt"
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
            headers=headers,
            files=[("files", (f"{name}.txt", b"z", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    return r.json()[0]["id"]


def _create_folder(client, headers, name: str, parent_folder_id=None) -> int:
    r = client.post(
        "/api/library/folders",
        headers=headers,
        json={"name": name, "parent_folder_id": parent_folder_id},
    )
    assert r.status_code == 201
    return r.json()["id"]


def test_reindex_folder_endpoint_returns_queued(client, responsable_headers, admin_headers, tmp_path):
    folder_id = _create_folder(client, responsable_headers, "Dossier retraitement")
    _upload_doc_with_real_file(client, responsable_headers, tmp_path, "in_folder")

    mock_result = mock.MagicMock()
    mock_result.id = "task-reindex-folder-xyz"
    with mock.patch(
        "app.tasks.documents.reindex_folder_library_documents_task.apply_async",
        return_value=mock_result,
    ):
        r = client.post(
            f"/api/library/folders/{folder_id}/reindex",
            headers=admin_headers,
        )

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert body["celery_task_id"] == "task-reindex-folder-xyz"
    assert body["folder_id"] == folder_id


def test_reindex_folder_not_found_returns_404(client, admin_headers):
    r = client.post("/api/library/folders/999999999/reindex", headers=admin_headers)
    assert r.status_code == 404


def test_reindex_folder_forbidden_lecteur(client, lecteur_headers):
    r = client.post("/api/library/folders/1/reindex", headers=lecteur_headers)
    assert r.status_code == 403


def test_reindex_folder_forbidden_non_admin_responsable(client, responsable_headers):
    """Retraitement par dossier : réservé au rôle admin, comme le retraitement global."""
    r = client.post("/api/library/folders/1/reindex", headers=responsable_headers)
    assert r.status_code == 403
    assert "admin" in (r.json().get("detail") or "").lower()


def test_reindex_folder_service_unavailable_returns_503(client, responsable_headers, admin_headers):
    folder_id = _create_folder(client, responsable_headers, "Dossier 503")
    with (
        mock.patch("app.services.task_dispatch.get_task_backend_mode", return_value="celery"),
        mock.patch(
            "app.tasks.documents.reindex_folder_library_documents_task.apply_async",
            side_effect=ConnectionError("broker down"),
        ),
    ):
        r = client.post(
            f"/api/library/folders/{folder_id}/reindex",
            headers=admin_headers,
        )
    assert r.status_code == 503
    assert "Celery" in (r.json().get("detail") or "")


def test_mark_folder_documents_scopes_to_folder_and_subfolders(
    client, responsable_headers, tmp_path
):
    """
    Cœur de la fonctionnalité : le marquage descend dans les sous-dossiers
    MAIS ne touche pas les documents hors du dossier ciblé.
    """
    parent_id = _create_folder(client, responsable_headers, "Parent retraitement")
    child_id = _create_folder(client, responsable_headers, "Enfant retraitement", parent_id)
    other_id = _create_folder(client, responsable_headers, "Dossier voisin")

    doc_parent = _upload_doc_with_real_file(client, responsable_headers, tmp_path, "doc_parent")
    doc_child = _upload_doc_with_real_file(client, responsable_headers, tmp_path, "doc_child")
    doc_other = _upload_doc_with_real_file(client, responsable_headers, tmp_path, "doc_other")
    doc_root = _upload_doc_with_real_file(client, responsable_headers, tmp_path, "doc_root")

    for doc_id, folder_id in (
        (doc_parent, parent_id),
        (doc_child, child_id),
        (doc_other, other_id),
    ):
        mv = client.post(
            f"/api/library/documents/{doc_id}/move?new_folder_id={folder_id}",
            headers=responsable_headers,
        )
        assert mv.status_code in (200, 204), mv.text

    me = client.get("/api/auth/me", headers=responsable_headers)
    user_id = me.json()["id"]

    marked = mark_folder_documents_reindex_queued(user_id, parent_id)
    assert marked == 2, f"attendu 2 documents marqués (parent + enfant), obtenu {marked}"

    def _status(doc_id: int) -> str:
        gr = client.get(f"/api/library/documents/{doc_id}", headers=responsable_headers)
        assert gr.status_code == 200
        return gr.json()["processing_status"]

    assert _status(doc_parent) == DOCUMENT_STATUS_REINDEX_QUEUED
    assert _status(doc_child) == DOCUMENT_STATUS_REINDEX_QUEUED
    assert _status(doc_other) != DOCUMENT_STATUS_REINDEX_QUEUED
    assert _status(doc_root) != DOCUMENT_STATUS_REINDEX_QUEUED


def test_reindex_folder_worker_dispatches_only_folder_docs(client, responsable_headers, tmp_path):
    """La tâche worker n'enfile que les documents du dossier ciblé."""
    from app.tasks.documents import reindex_folder_library_documents_task

    folder_id = _create_folder(client, responsable_headers, "Dossier worker")
    doc_in = _upload_doc_with_real_file(client, responsable_headers, tmp_path, "worker_in")
    doc_out = _upload_doc_with_real_file(client, responsable_headers, tmp_path, "worker_out")

    mv = client.post(
        f"/api/library/documents/{doc_in}/move?new_folder_id={folder_id}",
        headers=responsable_headers,
    )
    assert mv.status_code in (200, 204), mv.text

    me = client.get("/api/auth/me", headers=responsable_headers)
    user_id = me.json()["id"]

    dispatched: list[int] = []

    def _fake_dispatch(document_id: int, uid: int, mode: str = "full", extractor: str = "vision"):
        dispatched.append(document_id)
        return f"task-{document_id}"

    with mock.patch(
        "app.services.task_dispatch.dispatch_reindex_library",
        side_effect=_fake_dispatch,
    ):
        out = reindex_folder_library_documents_task.run(user_id=user_id, folder_id=folder_id)

    assert out["marked_queued"] == 1
    assert out["ok"] == 1
    assert dispatched == [doc_in]
    assert doc_out not in dispatched


def _ensure_global_library(session: Session) -> Library:
    lib = session.exec(select(Library).where(Library.is_global == True)).first()
    if lib is None:
        lib = Library(name="Bibliothèque commune test", user_id=None, is_global=True)
        session.add(lib)
        session.commit()
        session.refresh(lib)
    return lib


def test_reindex_denies_when_private_library_neither_owner_nor_uploader(db_session: Session):
    """Bibliothèque non globale : un tiers ne peut pas réindexer."""
    owner = create_test_user(db_session, "responsable")
    uploader = create_test_user(db_session, "responsable")
    intruder = create_test_user(db_session, "responsable")
    priv = Library(name="Bib privée pytest", user_id=owner.id, is_global=False)
    db_session.add(priv)
    db_session.commit()
    db_session.refresh(priv)

    doc = Document(
        title="doc privé",
        document_type="document",
        library_id=priv.id,
        user_id=uploader.id,
        source_file_path=None,
        processing_status="completed",
        processing_progress=100,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    with pytest.raises(ValueError, match="Document introuvable ou accès refusé"):
        reindex_library_document(doc.id, intruder.id)


@mock.patch(
    "app.services.document_indexing_service.process_document_indexing",
    return_value={"chunks": 1, "status": "completed"},
)
def test_reindex_allows_global_library_when_reindexer_differs_from_uploader(
    _process_indexing,
    db_session: Session,
    tmp_path,
):
    """Bibliothèque globale : le user_id du document est l’uploader, pas le demandeur Celery."""
    uploader = create_test_user(db_session, "responsable")
    reindexer = create_test_user(db_session, "responsable")
    lib = _ensure_global_library(db_session)

    f = tmp_path / "src.txt"
    f.write_bytes(b"hello")

    doc = Document(
        title="doc global",
        document_type="document",
        library_id=lib.id,
        user_id=uploader.id,
        source_file_path=str(f),
        processing_status="completed",
        processing_progress=100,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    out = reindex_library_document(doc.id, reindexer.id)
    assert out["status"] == "completed"
    assert out["document_id"] == doc.id


def test_reindex_all_counts_ok_for_global_docs_from_other_uploaders(
    db_session: Session,
    tmp_path,
    client,
    responsable_headers,
):
    """reindex_all avec user Celery = demandeur : doit traiter les docs dont user_id ≠ demandeur (bib. globale)."""
    fake_file = tmp_path / "reindex_all_multi.txt"
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
            files=[("files", ("uploaded_by_A.txt", b"z", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    doc_id = r.json()[0]["id"]

    other = create_test_user(db_session, "responsable")

    def _fake_reindex(document_id: int, uid: int):
        assert uid == other.id
        return {"document_id": document_id, "chunks": 1, "status": "completed"}

    with mock.patch(
        "app.services.document_service_new.reindex_library_document",
        side_effect=_fake_reindex,
    ):
        out = reindex_all_library_documents(other.id)

    assert out["ok"] >= 1
    failed_for_our_doc = [f for f in out.get("failed", []) if f.get("document_id") == doc_id]
    assert failed_for_our_doc == []
