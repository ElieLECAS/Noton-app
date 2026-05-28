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


def test_extract_alphanumeric_codes():
    from app.services.space_search_service import _extract_alphanumeric_codes
    query = "Quelles sont les spécifications pour la gamme Perform-70 et la norme DTU 36.5 chez Soleal ?"
    codes = _extract_alphanumeric_codes(query)
    
    # Doit contenir les codes avec chiffres, acronymes majuscules et noms capitalisés
    assert "perform-70" in codes
    assert "36.5" in codes
    assert "dtu" in codes
    assert "soleal" in codes
    
    # Dédoublonnement
    assert len(codes) == len(set(codes))


def test_reciprocal_rank_fusion_three_channels():
    from app.services.space_search_service import reciprocal_rank_fusion
    from llama_index.core.schema import TextNode, NodeWithScore
    
    n1 = NodeWithScore(node=TextNode(id_="chunk-1", text="text 1"), score=0.9)
    n2 = NodeWithScore(node=TextNode(id_="chunk-2", text="text 2"), score=0.8)
    n3 = NodeWithScore(node=TextNode(id_="chunk-3", text="text 3"), score=0.7)
    
    vector = [n1, n2]
    lexical = [n2, n3]
    alphanumeric = [n3, n1]
    
    res = reciprocal_rank_fusion(vector, lexical, alphanumeric_results=alphanumeric, top_n=3)
    assert len(res) <= 3
    # Tous les chunks doivent être présents
    node_ids = {r.node.id_ for r in res}
    assert "chunk-1" in node_ids
    assert "chunk-2" in node_ids
    assert "chunk-3" in node_ids
    
    # Les scores doivent être normalisés dans [0.1, 0.9]
    for r in res:
        assert 0.1 <= r.score <= 0.9

