"""
Tests unitaires légers pour le chunking documents / Docling (sans Docling runtime).
"""
import unittest


class TestMarkdownH2Sections(unittest.TestCase):
    def test_no_h2_returns_none(self):
        from app.services.chunk_service import _try_markdown_h2_sections

        self.assertIsNone(_try_markdown_h2_sections(""))
        self.assertIsNone(_try_markdown_h2_sections("pas de section h2 ici"))

    def test_multiple_h2_splits(self):
        from app.services.chunk_service import _try_markdown_h2_sections

        text = "# Titre\n\nIntro court.\n## Section A\n\nContenu A.\n## Section B\n\nContenu B."
        chunks = _try_markdown_h2_sections(text)
        self.assertIsNotNone(chunks)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("##", chunks[1]["content"])


class TestDoclingHierarchicalSpecs(unittest.TestCase):
    def test_empty_leaf_nodes(self):
        from app.services.chunking_service import _build_docling_hierarchical_specs

        specs = _build_docling_hierarchical_specs({"document_id": 1}, [])
        self.assertEqual(specs, [])

    def test_one_heading_two_leaves_parent_child_structure(self):
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
        self.assertEqual(len(specs), 3)
        self.assertFalse(specs[0]["is_leaf"])
        self.assertTrue(specs[1]["is_leaf"])
        self.assertTrue(specs[2]["is_leaf"])
        self.assertEqual(
            specs[0]["metadata_json"].get("chunking_version"),
            CHUNKING_VERSION_DOCLING_HIERARCHICAL,
        )
        self.assertEqual(specs[1]["parent_node_id"], specs[0]["node_id"])
        self.assertEqual(specs[2]["parent_node_id"], specs[0]["node_id"])


if __name__ == "__main__":
    unittest.main()
