"""Tests légers pour parsing tableaux et fusion hybride RAG (sans DB)."""

from llama_index.core.schema import NodeWithScore, TextNode


def test_parse_markdown_table_simple():
    from app.services.chunking_service import _parse_markdown_table

    t = "| A | B |\n| :--- | :--- |\n| 1 | 2 |\n"
    r = _parse_markdown_table(t)
    assert r is not None
    headers, rows = r
    assert headers == ["A", "B"]
    assert rows == [["1", "2"]]


def test_hybrid_fuse_candidates_vector_and_lexical():
    from app.services.space_search_service import _hybrid_fuse_candidates

    v = [NodeWithScore(node=TextNode(id_="chunk-1", text="a"), score=0.8)]
    l = [NodeWithScore(node=TextNode(id_="chunk-1", text="a"), score=0.5)]
    merged = _hybrid_fuse_candidates(v, l, [])
    assert len(merged) == 1
    assert merged[0].score > 0
    meta = merged[0].node.metadata or {}
    assert "lexical_norm" in meta
    assert "hybrid_score" in meta


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
    # 1 parent + 2 lignes de données
    assert len(specs) == 3
    row_specs = [s for s in specs if s["is_leaf"]]
    assert len(row_specs) == 2
    assert row_specs[0]["metadata_json"].get("content_type") == "table_row"
    assert row_specs[0]["metadata_json"].get("column_headers") == ["Col1", "Col2"]
