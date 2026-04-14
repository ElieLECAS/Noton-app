"""File d'attente bibliothèque : stop / skip (admin) et refus non-admin."""
from __future__ import annotations

from unittest import mock

from sqlmodel import Session

from app.models.document import Document
from app.services.document_service_new import (
    DOCUMENT_STATUS_CANCELLED,
    DOCUMENT_STATUS_SKIPPED,
    _should_abort_processing,
    skip_all_library_documents_processing,
    stop_all_library_documents_processing,
)
from app.services.library_service import get_or_create_user_library


def _upload_one_doc(client, headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_queue.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=headers,
            files=[("files", ("q.txt", b"hello", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    return r.json()[0]["id"]


def test_stop_document_forbidden_non_admin(client, responsable_headers, admin_headers):
    doc_id = _upload_one_doc(client, admin_headers)
    r = client.post(
        f"/api/library/documents/{doc_id}/stop",
        headers=responsable_headers,
    )
    assert r.status_code == 403


def test_skip_document_forbidden_non_admin(client, responsable_headers, admin_headers):
    doc_id = _upload_one_doc(client, admin_headers)
    r = client.post(
        f"/api/library/documents/{doc_id}/skip",
        headers=responsable_headers,
    )
    assert r.status_code == 403


def test_stop_all_forbidden_non_admin(client, responsable_headers):
    r = client.post("/api/library/documents/stop-all", headers=responsable_headers)
    assert r.status_code == 403


def test_skip_all_forbidden_non_admin(client, responsable_headers):
    r = client.post("/api/library/documents/skip-all", headers=responsable_headers)
    assert r.status_code == 403


def test_stop_document_admin_ok(client, admin_headers):
    doc_id = _upload_one_doc(client, admin_headers)
    r = client.post(
        f"/api/library/documents/{doc_id}/stop",
        headers=admin_headers,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == doc_id
    assert data["processing_status"] == DOCUMENT_STATUS_CANCELLED


def test_skip_document_admin_pending_ok(client, admin_headers):
    doc_id = _upload_one_doc(client, admin_headers)
    r = client.post(
        f"/api/library/documents/{doc_id}/skip",
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert r.json()["processing_status"] == DOCUMENT_STATUS_SKIPPED


def test_skip_document_while_processing_is_cancelled_like_stop(
    client, admin_headers, db_session: Session
):
    """Skip sur un document « processing » : même effet qu’un stop (cancelled)."""
    doc_id = _upload_one_doc(client, admin_headers)
    doc = db_session.get(Document, doc_id)
    assert doc is not None
    doc.processing_status = "processing"
    db_session.add(doc)
    db_session.commit()

    r = client.post(
        f"/api/library/documents/{doc_id}/skip",
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert r.json()["processing_status"] == DOCUMENT_STATUS_CANCELLED


def test_should_abort_processing_true_after_document_cancelled_via_api(
    client, admin_headers
):
    """Le pipeline doit voir un arrêt dès que le statut en base est cancelled."""
    doc_id = _upload_one_doc(client, admin_headers)
    r = client.post(
        f"/api/library/documents/{doc_id}/stop",
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert _should_abort_processing(doc_id) is True


def test_should_abort_processing_true_after_document_skipped_via_api(
    client, admin_headers
):
    """Idem pour skipped : le worker ne doit pas reprendre ce document."""
    doc_id = _upload_one_doc(client, admin_headers)
    r = client.post(
        f"/api/library/documents/{doc_id}/skip",
        headers=admin_headers,
    )
    assert r.status_code == 200
    assert _should_abort_processing(doc_id) is True


def test_stop_document_404_unknown_id(client, admin_headers):
    r = client.post(
        "/api/library/documents/999999999/stop",
        headers=admin_headers,
    )
    assert r.status_code == 404


def test_stop_document_400_when_not_in_queue(client, admin_headers):
    doc_id = _upload_one_doc(client, admin_headers)
    r = client.post(
        f"/api/library/documents/{doc_id}/stop",
        headers=admin_headers,
    )
    assert r.status_code == 200
    r2 = client.post(
        f"/api/library/documents/{doc_id}/stop",
        headers=admin_headers,
    )
    assert r2.status_code == 400


def test_stop_all_admin_marks_uploaded_docs_cancelled(client, admin_headers):
    """stop-all annule toute la file active : le nombre total dépend des autres tests (bib. partagée)."""
    id_a = _upload_one_doc(client, admin_headers)
    id_b = _upload_one_doc(client, admin_headers)
    r = client.post("/api/library/documents/stop-all", headers=admin_headers)
    assert r.status_code == 200
    cancelled = r.json().get("cancelled")
    assert isinstance(cancelled, int)
    assert cancelled >= 2
    for doc_id in (id_a, id_b):
        gr = client.get(f"/api/library/documents/{doc_id}", headers=admin_headers)
        assert gr.status_code == 200
        assert gr.json()["processing_status"] == DOCUMENT_STATUS_CANCELLED


def test_stop_all_response_has_cancelled_count(client, admin_headers):
    r = client.post("/api/library/documents/stop-all", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "cancelled" in body
    assert isinstance(body["cancelled"], int)


def test_skip_all_response_has_counts(client, admin_headers):
    r = client.post("/api/library/documents/skip-all", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert "skipped" in body and "cancelled_running" in body
    assert isinstance(body["skipped"], int)
    assert isinstance(body["cancelled_running"], int)


def test_skip_all_admin_mixed(db_session: Session, client, admin_headers):
    d1 = _upload_one_doc(client, admin_headers)
    d2 = _upload_one_doc(client, admin_headers)
    doc1 = db_session.get(Document, d1)
    assert doc1 is not None
    doc1.processing_status = "processing"
    db_session.add(doc1)
    db_session.commit()

    r = client.post("/api/library/documents/skip-all", headers=admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body.get("cancelled_running") == 1
    assert body.get("skipped") == 1


def test_service_stop_all_and_skip_all(db_session: Session):
    from tests.conftest import create_test_user

    user = create_test_user(db_session, "admin")
    library = get_or_create_user_library(db_session, user.id)
    a = Document(
        title="a",
        content="x",
        document_type="document",
        processing_status="pending",
        processing_progress=0,
        library_id=library.id,
        user_id=user.id,
    )
    b = Document(
        title="b",
        content="x",
        document_type="document",
        processing_status="processing",
        processing_progress=50,
        library_id=library.id,
        user_id=user.id,
    )
    db_session.add(a)
    db_session.add(b)
    db_session.commit()
    db_session.refresh(a)
    db_session.refresh(b)

    out = skip_all_library_documents_processing(db_session, user.id)
    assert out["skipped"] == 1
    assert out["cancelled_running"] == 1

    c = Document(
        title="c",
        content="x",
        document_type="document",
        processing_status="pending",
        processing_progress=0,
        library_id=library.id,
        user_id=user.id,
    )
    db_session.add(c)
    db_session.commit()
    db_session.refresh(c)

    out2 = stop_all_library_documents_processing(db_session, user.id)
    assert out2["cancelled"] == 1
    db_session.refresh(c)
    assert c.processing_status == DOCUMENT_STATUS_CANCELLED
