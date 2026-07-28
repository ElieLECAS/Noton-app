"""Retraitement multimodal bibliothèque v4 (endpoint /reindex unifié)."""
from __future__ import annotations

from unittest import mock

import pytest
from sqlmodel import Session, select

from app.config import settings
from app.models.document_chunk import DocumentChunk
from app.services.document_service_new import multimodal_reindex_library_document
from app.services.multimodal_page_service import (
    MultimodalChunkSpec,
    PAGE_RAW_ENRICHED_CONTENT_TYPE,
    PAGE_WINDOW_REPORT_CONTENT_TYPE,
    build_raw_chunk_specs_from_page,
    parse_multimodal_page_response,
)


def test_parse_multimodal_page_response_v4():
    data = {
        "page_no": 2,
        "raw_text": "Texte source. [Image: schéma cote 50mm]",
    }
    parsed = parse_multimodal_page_response(data, 2, "raw pymupdf", "Doc test")
    assert "[Image:" in parsed["raw_text"]
    assert parsed.get("pro_reports") is None


def test_build_raw_chunk_specs_from_page():
    parsed = parse_multimodal_page_response(
        {"raw_text": "Corps source technique.", "page_no": 1},
        1,
        "",
        "Mon doc",
    )
    specs = build_raw_chunk_specs_from_page(42, 1, parsed, "Mon doc")
    assert len(specs) >= 1
    assert specs[0].content_type == PAGE_RAW_ENRICHED_CONTENT_TYPE
    assert "Corps source" in specs[0].content
    assert "multimodal-page-42-1-raw" in specs[0].node_id


def test_multimodal_reindex_endpoint_returns_queued(
    client, responsable_headers, admin_headers, monkeypatch
):
    monkeypatch.setattr(settings, "MULTIMODAL_ENABLED", True)
    monkeypatch.setattr(settings, "MISTRAL_API_KEY", "test-key")
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
    assert body["celery_task_id"] == "task-multimodal-xyz"


# NB : l'endpoint /reindex ne gate plus sur MULTIMODAL_ENABLED (seuls MISTRAL_API_KEY
# et COLPALI_ENABLED sont vérifiés désormais, cf. app/routers/library.py) — les 400
# correspondants sont couverts dans tests/test_document_indexing.py. L'ancien test
# postait en plus sur un document_id inexistant, ce qui masquait le vrai problème
# derrière un 404.


def test_multimodal_service_v4_chunks(
    db_session: Session, monkeypatch, tmp_path
):
    monkeypatch.setattr(settings, "MULTIMODAL_ENABLED", True)
    from app.models.document import Document
    from app.models.library import Library
    from tests.conftest import create_test_user

    user = create_test_user(db_session, "admin")
    lib = Library(user_id=user.id, name="Lib MM", is_global=True)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)

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
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    ocr_chunk = DocumentChunk(
        document_id=doc.id,
        chunk_index=0,
        content="chunk ocr existant",
        is_leaf=True,
        metadata_json={"content_type": "text_full"},
    )
    db_session.add(ocr_chunk)
    db_session.commit()

    fake_specs = [
        MultimodalChunkSpec(
            page_no=1,
            content="Texte brut page 1",
            content_type=PAGE_RAW_ENRICHED_CONTENT_TYPE,
            node_id=f"multimodal-page-{doc.id}-1-raw",
            metadata={
                "content_type": PAGE_RAW_ENRICHED_CONTENT_TYPE,
                "page_no": 1,
                "window_id": "win-1-1-2",
                "chunking_version": "multimodal_page_v4",
                "is_leaf": True,
            },
        ),
        MultimodalChunkSpec(
            page_no=1,
            content="Rapport fenêtre pages 1-2 | Perform 70",
            content_type=PAGE_WINDOW_REPORT_CONTENT_TYPE,
            node_id=f"multimodal-{doc.id}-win-1-2-r1",
            metadata={
                "content_type": PAGE_WINDOW_REPORT_CONTENT_TYPE,
                "page_no": 1,
                "window_id": "win-1-1-2",
                "theme": "Pose",
                "chunking_version": "multimodal_page_v4",
                "is_leaf": True,
            },
        ),
    ]

    with (
        mock.patch(
            "app.services.file_conversion.ensure_pdf_for_ocr",
            return_value=str(pdf),
        ),
        mock.patch(
            "app.services.multimodal_page_service.build_multimodal_pages_for_pdf",
            return_value=fake_specs,
        ),
        mock.patch(
            "app.services.multimodal_page_service.embed_new_multimodal_chunks",
            return_value=2,
        ),
    ):
        result = multimodal_reindex_library_document(doc.id, user.id)

    assert result["status"] == "completed"
    assert result["chunks"] == 2

    chunks = list(
        db_session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).all()
    )
    assert len(chunks) == 3
    types = [c.metadata_json.get("content_type") for c in chunks]
    assert types.count(PAGE_RAW_ENRICHED_CONTENT_TYPE) == 1
    assert types.count(PAGE_WINDOW_REPORT_CONTENT_TYPE) == 1
