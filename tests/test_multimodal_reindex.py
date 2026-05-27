"""Retraitement multimodal bibliothèque v3 : raw enrichi + rapport pro par section."""
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
    PAGE_SECTION_REPORT_CONTENT_TYPE,
    PAGE_MULTIMODAL_SUMMARY_CONTENT_TYPE,
    build_chunk_specs_from_page,
    parse_multimodal_page_response,
)


def test_parse_multimodal_page_response_multi_sections():
    data = {
        "page_no": 2,
        "sections": [
            {
                "section_index": 1,
                "heading": "2.1 Généralités",
                "section_kind": "text",
                "raw_text": "Texte source. [Image 1: schéma avec cote 50mm]",
                "pro_report": "Rapport : tolérance 50mm selon NF EN 14351-1.",
                "references": ["NF EN 14351-1"],
                "keywords": ["baie"],
                "norms": ["NF EN 14351-1"],
                "constraints": ["Charge max 120 kg"],
                "dependencies": [],
                "linked_figures": [],
            },
            {
                "section_index": 2,
                "heading": "Tableau 1",
                "section_kind": "table",
                "raw_text": "| A | B |\n|---|---|",
                "pro_report": "Rapport tableau : colonnes A et B avec valeurs Uw.",
                "references": [],
                "keywords": ["Uw"],
                "norms": [],
                "constraints": [],
                "dependencies": [],
                "linked_figures": [],
            },
        ],
    }
    parsed = parse_multimodal_page_response(data, 2, "raw pymupdf", "Doc test")
    assert len(parsed["sections"]) == 2
    assert "[Image 1:" in parsed["sections"][0]["raw_text"]
    assert parsed["sections"][0]["pro_report"].startswith("Rapport")


def test_parse_multimodal_page_response_legacy_content_fallback():
    """Compatibilité : ancien champ content → raw_text."""
    data = {
        "sections": [
            {
                "section_index": 1,
                "heading": "Intro",
                "content": "Ancien format contenu",
            }
        ]
    }
    parsed = parse_multimodal_page_response(data, 1, "fallback pymupdf", "Doc")
    assert parsed["sections"][0]["raw_text"] == "Ancien format contenu"
    assert parsed["sections"][0]["pro_report"]


def test_build_chunk_specs_from_page():
    parsed = parse_multimodal_page_response(
        {
            "sections": [
                {
                    "section_index": 1,
                    "heading": "Intro",
                    "section_kind": "text",
                    "raw_text": "Corps source",
                    "pro_report": "Rapport technique intro.",
                }
            ],
        },
        1,
        "",
        "Mon doc",
    )
    specs = build_chunk_specs_from_page(42, 1, parsed, "Mon doc")
    assert len(specs) == 2
    assert specs[0].content_type == PAGE_RAW_ENRICHED_CONTENT_TYPE
    assert specs[1].content_type == PAGE_SECTION_REPORT_CONTENT_TYPE
    assert "Corps source" in specs[0].content
    assert "Rapport technique" in specs[1].content
    assert specs[0].node_id == "multimodal-page-42-1-s1-raw"
    assert specs[1].node_id == "multimodal-page-42-1-s1-report"


def test_build_chunk_specs_two_sections_four_chunks():
    parsed = parse_multimodal_page_response(
        {
            "sections": [
                {
                    "section_index": 1,
                    "heading": "A",
                    "raw_text": "Texte A",
                    "pro_report": "Rapport A",
                },
                {
                    "section_index": 2,
                    "heading": "B",
                    "raw_text": "Texte B",
                    "pro_report": "Rapport B",
                },
            ],
        },
        1,
        "",
        "Doc",
    )
    specs = build_chunk_specs_from_page(1, 1, parsed, "Doc")
    assert len(specs) == 4
    raw_count = sum(1 for s in specs if s.content_type == PAGE_RAW_ENRICHED_CONTENT_TYPE)
    report_count = sum(1 for s in specs if s.content_type == PAGE_SECTION_REPORT_CONTENT_TYPE)
    assert raw_count == 2
    assert report_count == 2


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


def test_multimodal_reindex_disabled_returns_400(client, admin_headers, monkeypatch):
    monkeypatch.setattr(settings, "MULTIMODAL_ENABLED", False)
    r = client.post(
        "/api/library/documents/1/multimodal-reindex",
        headers=admin_headers,
    )
    assert r.status_code == 400


def test_multimodal_service_multi_chunks_replaces_v1_and_v2(
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
    old_v1 = DocumentChunk(
        document_id=doc.id,
        chunk_index=1,
        content="ancien v1 monolithique",
        is_leaf=True,
        metadata_json={
            "content_type": PAGE_MULTIMODAL_SUMMARY_CONTENT_TYPE,
            "chunking_version": "multimodal_page_v1",
            "page_no": 1,
        },
    )
    session.add(ocr_chunk)
    session.add(old_v1)
    session.commit()

    fake_specs = [
        MultimodalChunkSpec(
            page_no=1,
            content="Texte brut section A",
            content_type=PAGE_RAW_ENRICHED_CONTENT_TYPE,
            node_id=f"multimodal-page-{doc.id}-1-s1-raw",
            metadata={
                "content_type": PAGE_RAW_ENRICHED_CONTENT_TYPE,
                "page_no": 1,
                "section_index": 1,
                "references": ["NF EN 1"],
            },
        ),
        MultimodalChunkSpec(
            page_no=1,
            content="Rapport pro section A",
            content_type=PAGE_SECTION_REPORT_CONTENT_TYPE,
            node_id=f"multimodal-page-{doc.id}-1-s1-report",
            metadata={
                "content_type": PAGE_SECTION_REPORT_CONTENT_TYPE,
                "page_no": 1,
                "section_index": 1,
                "norms": ["NF EN 1"],
            },
        ),
        MultimodalChunkSpec(
            page_no=1,
            content="Texte brut section B",
            content_type=PAGE_RAW_ENRICHED_CONTENT_TYPE,
            node_id=f"multimodal-page-{doc.id}-1-s2-raw",
            metadata={
                "content_type": PAGE_RAW_ENRICHED_CONTENT_TYPE,
                "page_no": 1,
                "section_index": 2,
            },
        ),
        MultimodalChunkSpec(
            page_no=1,
            content="Rapport pro section B",
            content_type=PAGE_SECTION_REPORT_CONTENT_TYPE,
            node_id=f"multimodal-page-{doc.id}-1-s2-report",
            metadata={
                "content_type": PAGE_SECTION_REPORT_CONTENT_TYPE,
                "page_no": 1,
                "section_index": 2,
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
            return_value=4,
        ),
    ):
        result = multimodal_reindex_library_document(doc.id, user.id)

    assert result["status"] == "completed"
    assert result["chunks"] == 4

    chunks = list(
        session.exec(
            select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
        ).all()
    )
    assert len(chunks) == 5  # 1 OCR + 4 multimodal
    types = [c.metadata_json.get("content_type") for c in chunks]
    assert types.count(PAGE_RAW_ENRICHED_CONTENT_TYPE) == 2
    assert types.count(PAGE_SECTION_REPORT_CONTENT_TYPE) == 2
    assert "ancien v1 monolithique" not in [c.content for c in chunks]
    raw_chunks = [
        c
        for c in chunks
        if c.metadata_json.get("content_type") == PAGE_RAW_ENRICHED_CONTENT_TYPE
    ]
    assert raw_chunks[0].metadata_json.get("references") == ["NF EN 1"]
