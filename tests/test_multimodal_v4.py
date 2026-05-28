"""Tests pipeline multimodal v4."""
from __future__ import annotations

from unittest import mock

import pytest

from app.config import settings
from app.services.multimodal_page_service import (
    CHUNKING_VERSION_MULTIMODAL,
    PAGE_RAW_ENRICHED_CONTENT_TYPE,
    PAGE_WINDOW_REPORT_CONTENT_TYPE,
    build_dynamic_windows,
    build_raw_chunk_specs_from_page,
    build_window_report_chunk_specs,
    merge_cut_sections_across_pages,
    parse_multimodal_page_response,
    resolve_extraction_mode,
    split_text_rag_friendly,
    validate_pro_reports,
    validate_raw_page,
    count_image_blocks,
)


def test_resolve_extraction_mode_native():
    assert resolve_extraction_mode("x" * 150) == "native"
    assert resolve_extraction_mode("short") == "scanned"


def test_parse_multimodal_page_response_raw_only():
    data = {"page_no": 3, "raw_text": "Texte page.\n[Image: schéma cote 50 mm]"}
    parsed = parse_multimodal_page_response(
        data, 3, "pymupdf text", "Doc", extraction_mode="native"
    )
    assert parsed["raw_text"].startswith("Texte page")
    assert "pro_reports" not in parsed
    assert parsed["image_block_count"] == 1
    assert parsed["extraction_mode"] == "native"


def test_split_text_rag_friendly_preserves_image_block():
    img = "[Image: plan technique avec cote 1,5 m et référence Perform 70]"
    text = f"Intro.\n\n{img}\n\nFin."
    parts = split_text_rag_friendly(text, max_tokens=50, overlap_tokens=0)
    assert any(img in p for p in parts)
    assert not any(p.rstrip().endswith("[Image:") for p in parts)


def test_merge_cut_sections_raw_only():
    pages = [
        {"page_no": 1, "raw_text": "Phrase coupée sans fin", "page_start": 1, "page_end": 1},
        {"page_no": 2, "raw_text": "suite de la phrase.", "page_start": 2, "page_end": 2},
    ]
    merged = merge_cut_sections_across_pages(pages)
    assert len(merged) == 1
    assert "suite" in merged[0]["raw_text"]
    assert merged[0]["page_end"] == 2


def test_build_dynamic_windows_respects_max_pages():
    pages = [
        {"page_no": i, "raw_text": f"Contenu page {i}.\n\n" + ("mot " * 200)}
        for i in range(1, 8)
    ]
    windows = build_dynamic_windows(99, pages)
    assert len(windows) >= 1
    for w in windows:
        assert w["window_end"] - w["window_start"] + 1 <= getattr(
            settings, "WINDOW_MAX_PAGES", 12
        )
        assert w["window_id"].startswith("win-99-")


def test_build_raw_chunk_specs_metadata():
    parsed = {
        "page_no": 1,
        "page_start": 1,
        "page_end": 1,
        "raw_text": "Court.",
        "extraction_mode": "native",
        "pymupdf_char_count": 10,
        "image_block_count": 0,
        "raw_validation_status": "ok",
    }
    specs = build_raw_chunk_specs_from_page(1, 1, parsed, "Doc")
    assert len(specs) == 1
    assert specs[0].content_type == PAGE_RAW_ENRICHED_CONTENT_TYPE
    assert specs[0].metadata["chunking_version"] == CHUNKING_VERSION_MULTIMODAL
    assert specs[0].metadata["is_leaf"] is True


def test_build_window_report_chunk_specs():
    window = {
        "window_id": "win-1-1-3",
        "window_start": 1,
        "window_end": 3,
        "raw_concat": "texte",
    }
    reports = [
        {
            "theme": "Pose Perform 70",
            "report": "La pose du profil Perform 70 exige un jeu de 5 mm.",
            "references": ["Perform 70 CW"],
            "norms": ["DTU 36.5"],
            "constraints": [],
            "dependencies": [],
            "keywords": ["pose"],
        }
    ]
    specs = build_window_report_chunk_specs(1, window, reports, "Doc")
    assert len(specs) >= 1
    assert specs[0].content_type == PAGE_WINDOW_REPORT_CONTENT_TYPE
    assert specs[0].metadata["window_id"] == "win-1-1-3"
    assert "Perform 70" in specs[0].content


def test_validate_pro_reports_deictic():
    reports = [{"theme": "T", "report": "Ce profil fait 1 m.", "references": []}]
    _, needs = validate_pro_reports(reports)
    assert needs is True


def test_validate_raw_page_retry_schemas(monkeypatch):
    monkeypatch.setattr(
        "app.services.multimodal_page_service.page_has_significant_visuals",
        lambda *a, **k: True,
    )
    parsed = {"page_no": 1, "raw_text": "texte sans image", "extraction_mode": "native"}
    _, status = validate_raw_page(parsed, "long pymupdf " * 20, pdf_path="/x.pdf", page_index=0)
    assert status == "retry_schemas"


def test_count_image_blocks():
    assert count_image_blocks("a [Image: x] b [Image: y]") == 2


def test_tail_sentences_by_tokens():
    from app.services.multimodal_page_service import _tail_sentences_by_tokens
    text = "Première phrase. Deuxième phrase. Troisième phrase."
    
    # Si target_tokens est petit, on doit renvoyer la dernière phrase complète
    res = _tail_sentences_by_tokens(text, 10)
    assert res == "Troisième phrase."
    
    # Si target_tokens est suffisant, on prend plusieurs phrases complètes
    res = _tail_sentences_by_tokens(text, 25)
    assert res == "Deuxième phrase. Troisième phrase."

