"""
Tests unitaires légers pour le chunking documents / Docling (sans Docling runtime).
"""


def test_markdown_h2_sections_no_h2_returns_none():
    from app.services.chunk_service import _try_markdown_h2_sections

    assert _try_markdown_h2_sections("") is None
    assert _try_markdown_h2_sections("pas de section h2 ici") is None


def test_markdown_h2_sections_multiple_splits():
    from app.services.chunk_service import _try_markdown_h2_sections

    text = "# Titre\n\nIntro court.\n## Section A\n\nContenu A.\n## Section B\n\nContenu B."
    chunks = _try_markdown_h2_sections(text)
    assert chunks is not None
    assert len(chunks) >= 2
    assert "##" in chunks[1]["content"]


def test_docling_hierarchical_specs_empty_leaf_nodes():
    from app.services.chunking_service import _build_docling_hierarchical_specs

    specs = _build_docling_hierarchical_specs({"document_id": 1}, [])
    assert specs == []


def test_docling_hierarchical_specs_one_heading_two_leaves():
    """Sans runtime Docling : feuilles factices (API get_content + metadata)."""
    from app.services.chunking_service import (
        CHUNKING_VERSION_DOCLING_HIERARCHICAL,
        _build_docling_hierarchical_specs,
    )

    class _FakeLeaf:
        def __init__(self, node_id, content, headings):
            self.node_id = node_id
            self._content = content
            self.metadata = {"headings": headings, "page_no": 1}

        def get_content(self):
            return self._content

    leaves = [
        _FakeLeaf("leaf-a", "Premier paragraphe.", ["1 Introduction"]),
        _FakeLeaf("leaf-b", "Deuxième paragraphe.", ["1 Introduction"]),
    ]
    specs = _build_docling_hierarchical_specs({"document_id": 42}, leaves)
    assert len(specs) == 3
    assert specs[0]["is_leaf"] is False
    assert specs[1]["is_leaf"] is True
    assert specs[2]["is_leaf"] is True
    assert specs[0]["metadata_json"].get("chunking_version") == CHUNKING_VERSION_DOCLING_HIERARCHICAL
    assert specs[1]["parent_node_id"] == specs[0]["node_id"]
    assert specs[2]["parent_node_id"] == specs[0]["node_id"]
