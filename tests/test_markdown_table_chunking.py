"""Tests chunking markdown OCR + expansion tableaux."""

_TABLE_3COL = (
    "| Produit | Quantité | Prix |\n"
    "|---------|----------|------|\n"
    "| Vis M4  | 100      | 0.05 |\n"
    "| Écrou   | 50       | 0.10 |\n"
)


def test_split_markdown_into_segments_text_and_table():
    from app.services.chunking_service import _split_markdown_into_segments

    md = f"## Mesures\n\nIntro texte.\n\n{_TABLE_3COL}\n\nSuite."
    segments = _split_markdown_into_segments(md)
    kinds = [s.kind for s in segments]
    assert "text" in kinds
    assert "table" in kinds
    table_seg = next(s for s in segments if s.kind == "table")
    assert "Produit" in table_seg.content
    assert table_seg.parent_heading == "Mesures"


def test_chunk_markdown_hierarchical_with_tables_expands_rows():
    from app.services.chunking_service import chunk_markdown_hierarchical_with_tables

    md = f"## Drainage\n\n{_TABLE_3COL}"
    specs = chunk_markdown_hierarchical_with_tables(md, {"document_id": 1})
    content_types = {s["metadata_json"].get("content_type") for s in specs}

    assert "table_full" in content_types
    assert "table_row" in content_types
    row_specs = [s for s in specs if s["metadata_json"].get("content_type") == "table_row"]
    assert len(row_specs) == 2
    assert all(s["is_leaf"] for s in row_specs)
    assert "col1:Produit=Vis M4" in row_specs[0]["content"]


def test_table_not_split_mid_row():
    from app.services.chunking_service import chunk_markdown_hierarchical_with_tables

    md = _TABLE_3COL
    specs = chunk_markdown_hierarchical_with_tables(md, {"document_id": 2})
    row_specs = [s for s in specs if s["metadata_json"].get("content_type") == "table_row"]
    assert len(row_specs) == 2
    for spec in specs:
        if spec["metadata_json"].get("content_type") != "table_row":
            continue
        assert "| Vis M4 |" not in spec["content"] or "col1:" in spec["content"]


def test_suspicious_row_flagged():
    from app.services.chunking_service import chunk_markdown_hierarchical_with_tables

    md = (
        "| A | B | C |\n"
        "|---|---|---|\n"
        "| v1 | v2 | v3 |\n"
        "| x1 | x2 |\n"
    )
    specs = chunk_markdown_hierarchical_with_tables(md, {"document_id": 3})
    row_specs = [s for s in specs if s["metadata_json"].get("content_type") == "table_row"]
    assert any("[décalage probable]" in s["content"] for s in row_specs)
    full = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    assert full["metadata_json"].get("suspicious_rows") == [1]


def test_page_no_from_char_offset_mid_page():
    from app.services.chunking_service import (
        _build_page_marker_index,
        _page_no_from_char_offset,
        chunk_markdown_hierarchical_with_tables,
    )

    md = "<!-- page:1 -->\n\nPage one.\n\n<!-- page:2 -->\n\nPage two with table.\n\n" + _TABLE_3COL
    page_index = _build_page_marker_index(md)
    table_pos = md.index("| Produit")
    assert _page_no_from_char_offset(table_pos, page_index) == 2

    specs = chunk_markdown_hierarchical_with_tables(md, {"document_id": 4})
    row_specs = [s for s in specs if s["metadata_json"].get("content_type") == "table_row"]
    assert row_specs
    assert row_specs[0]["metadata_json"].get("page_no") == 2


def test_normalize_markdown_tables_canonical():
    from app.services.chunking_service import normalize_markdown_tables, _parse_markdown_table_robust

    messy = "|A|B|\n|-|-|\n|1|2|"
    normalized = normalize_markdown_tables(messy)
    parsed = _parse_markdown_table_robust(normalized)
    assert parsed is not None
    assert parsed.headers == ["A", "B"]


def test_mistral_ocr_response_injects_page_markers():
    from app.services.mistral_ocr_service import _markdown_from_ocr_response

    data = {
        "pages": [
            {"markdown": "Contenu page 1"},
            {"markdown": "Contenu page 2", "index": 2},
        ]
    }
    md = _markdown_from_ocr_response(data)
    assert "<!-- page:1 -->" in md
    assert "<!-- page:2 -->" in md


def test_propagate_page_to_parent():
    from app.services.chunking_service import chunk_markdown_hierarchical_with_tables

    md = "<!-- page:3 -->\n\n## Section\n\n" + _TABLE_3COL
    specs = chunk_markdown_hierarchical_with_tables(md, {"document_id": 5})
    parents = [s for s in specs if not s["is_leaf"]]
    assert any((s.get("metadata_json") or {}).get("page_no") == 3 for s in parents)
