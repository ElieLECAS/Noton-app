"""Tests légers : parsing tableaux Docling et résolution parents (sans DB)."""

from unittest.mock import MagicMock

from llama_index.core.schema import TextNode


def test_parse_markdown_table_simple():
    from app.services.chunking_service import _parse_markdown_table

    t = "| A | B |\n| :--- | :--- |\n| 1 | 2 |\n"
    r = _parse_markdown_table(t)
    assert r is not None
    headers, rows = r
    assert headers == ["A", "B"]
    assert rows == [["1", "2"]]


def test_docling_specs_table_expands_to_rows():
    from app.services.chunking_service import _build_docling_hierarchical_specs

    class _FakeLeaf:
        def __init__(self, node_id, content, headings, label=None):
            self.node_id = node_id
            self._content = content
            self.metadata = {"headings": headings, "page_no": 22, "label": label}

        def get_content(self):
            return self._content

    md = "| Col1 | Col2 |\n| --- | --- |\n| x | y |\n| a | b |\n"
    leaves = [
        _FakeLeaf("t1", md, ["1 Drainage"], label="table"),
    ]
    specs = _build_docling_hierarchical_specs({"document_id": 1}, leaves)
    assert len(specs) == 5
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    summary_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_summary")
    row_specs = [s for s in specs if s["metadata_json"].get("content_type") == "table_row"]
    assert len(row_specs) == 2
    assert row_specs[0]["metadata_json"].get("column_headers") == ["Col1", "Col2"]
    assert row_specs[0]["parent_node_id"] == full_spec["node_id"]
    assert summary_spec["parent_node_id"] == full_spec["node_id"]


def test_resolve_space_parent_multihop_no_document_id():
    from app.services.space_search_service import _resolve_space_parent_with_multihop

    session = MagicMock()
    assert (
        _resolve_space_parent_with_multihop(session, 1, 1, None, "some-uuid", {})
        is None
    )


def test_resolve_space_parent_multihop_delegates_when_parent_in_dict():
    from app.services.space_search_service import _resolve_space_parent_with_multihop

    fake = TextNode(id_="p1", text="section", metadata={})
    assert (
        _resolve_space_parent_with_multihop(None, 1, 1, 42, "p1", {"p1": fake})
        is None
    )


def test_chunk_markdown_hierarchical_node_parent_ids():
    """Vérifie node_id / parent_node_id pour la résolution parent en recherche."""
    from app.services.chunking_service import chunk_markdown_hierarchical

    md = "# Section A\n\n" + ("paragraphe court. " * 80) + "\n\n## Sous-section\n\n" + ("détail. " * 120)
    specs = chunk_markdown_hierarchical(md, {"document_id": 1})
    assert specs
    leaves = [s for s in specs if s["is_leaf"]]
    parents = [s for s in specs if not s["is_leaf"]]
    assert leaves
    parent_ids = {p["node_id"] for p in parents}
    for leaf in leaves:
        assert leaf["node_id"]
        pid = leaf.get("parent_node_id")
        if pid:
            assert pid in parent_ids or any(p["node_id"] == pid for p in specs)
