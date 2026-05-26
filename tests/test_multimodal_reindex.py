"""Retraitement multimodal bibliothèque : endpoint et service."""
from __future__ import annotations

from unittest import mock

import pytest
from sqlmodel import Session, select

from app.config import settings
from app.models.document_chunk import DocumentChunk
from app.services.document_service_new import (
    DOCUMENT_STATUS_MULTIMODAL_QUEUED,
    multimodal_reindex_library_document,
)
from app.services.multimodal_page_service import PAGE_MULTIMODAL_CONTENT_TYPE


def test_multimodal_reindex_endpoint_returns_queued(
    client, responsable_headers, admin_headers, monkeypatch
):
    monkeypatch.setattr(settings, "MULTIMODAL_ENABLED", True)
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_multimodal.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=responsable_headers,
            files=[("files", ("mm.txt", b"hello", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 201
    doc_id = r.json()[0]["id"]

    mock_result = mock.MagicMock()
    mock_result.id = "task-multimodal-xyz"
    with mock.patch(
        "app.tasks.documents.multimodal_reindex_library_document_task.apply_async",
        return_value=mock_result,
    ):
        r2 = client.post(
            f"/api/library/documents/{doc_id}/multimodal-reindex",
            headers=admin_headers,
        )

    assert r2.status_code == 200
    body = r2.json()
    assert body["status"] == "queued"
    assert body["celery_task_id"] == "task-multimodal-xyz"
    assert body["document_id"] == doc_id


def test_multimodal_reindex_disabled_returns_400(client, admin_headers, monkeypatch):
    monkeypatch.setattr(settings, "MULTIMODAL_ENABLED", False)
    r = client.post(
        "/api/library/documents/1/multimodal-reindex",
        headers=admin_headers,
    )
    assert r.status_code == 400
    assert "MULTIMODAL" in (r.json().get("detail") or "")


def test_multimodal_reindex_forbidden_lecteur(client, lecteur_headers, monkeypatch):
    monkeypatch.setattr(settings, "MULTIMODAL_ENABLED", True)
    r = client.post(
        "/api/library/documents/1/multimodal-reindex",
        headers=lecteur_headers,
    )
    assert r.status_code == 403


def test_multimodal_service_replaces_only_multimodal_chunks(
    session: Session, monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "MULTIMODAL_ENABLED", True)
    from app.models.document import Document
    from app.models.library import Library
    from tests.conftest import create_test_user

    user = create_test_user(session, "admin")
    lib = Library(user_id=user.id, name="Lib MM", is_global=True)
    session.add(lib)
    session.commit()
    session.refresh(lib)

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 minimal")

    doc = Document(
        library_id=lib.id,
        user_id=user.id,
        title="Test MM",
        document_type="document",
        source_file_path=str(pdf),
        processing_status="completed",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    ocr_chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="chunk ocr existant",
        is_leaf=True,
        metadata_json={"content_type": "text_full"},
    )
    old_mm = DocumentChunk(
        document_id=doc.id,
        chunk_index=1,
        content="ancien multimodal",
        is_leaf=True,
        metadata_json={"content_type": PAGE_MULTIMODAL_CONTENT_TYPE, "page_no": 1},
    )
    session.add(ocr_chunk)
    session.add(old_mm)
    session.commit()

    fake_pages = [(1, "# Page 1\n\nSynthèse test")]

    with (
        mock.patch(
            "app.services.file_conversion.ensure_pdf_for_ocr",
            return_value=str(pdf),
        ),
        mock.patch(
            "app.services.multimodal_page_service.build_multimodal_pages_for_pdf",
            return_value=fake_pages,
        ),
        mock.patch(
            "app.services.multimodal_page_service.embed_new_multimodal_chunks",
            return_value=1,
        ),
    ):
        result = multimodal_reindex_library_document(doc.id, user.id)

    assert result["status"] == "completed"
    assert result["chunks"] == 1

    chunks = list(
        session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).all()
    )
    assert len(chunks) == 2
    types = {c.metadata_json.get("content_type") for c in chunks}
    assert PAGE_MULTIMODAL_CONTENT_TYPE in types
    assert "text_full" in types
    mm = [c for c in chunks if c.metadata_json.get("content_type") == PAGE_MULTIMODAL_CONTENT_TYPE]
    assert len(mm) == 1
    assert "Synthèse test" in mm[0].content
    assert "ancien multimodal" not in [c.content for c in chunks]
