"""
Tests unitaires légers pour le chunking documents / Docling (sans Docling runtime).
"""
import pytest


# ---------------------------------------------------------------------------
# Helpers communs
# ---------------------------------------------------------------------------

class _FakeLeaf:
    """Feuille Docling factice compatible avec l'API get_content + metadata."""
    def __init__(self, node_id, content, headings, label=None, page_no=1):
        self.node_id = node_id
        self._content = content
        self.metadata = {"headings": headings, "page_no": page_no}
        if label:
            self.metadata["label"] = label

    def get_content(self):
        return self._content


_TABLE_3COL = (
    "| Produit | Quantité | Prix |\n"
    "|---------|----------|------|\n"
    "| Vis M4  | 100      | 0.05 |\n"
    "| Écrou   | 50       | 0.10 |\n"
)

_TABLE_2COL = (
    "| Paramètre | Valeur |\n"
    "|-----------|--------|\n"
    "| Tension   | 230 V  |\n"
    "| Courant   | 16 A   |\n"
)

_TABLE_5COL = (
    "| A | B | C | D | E |\n"
    "|---|---|---|---|---|\n"
    "| 1 | 2 | 3 | 4 | 5 |\n"
)


# ---------------------------------------------------------------------------
# Groupe A — _parse_markdown_table_robust
# ---------------------------------------------------------------------------

def test_robust_parser_basic():
    from app.services.chunking_service import _parse_markdown_table_robust

    result = _parse_markdown_table_robust(_TABLE_3COL)
    assert result is not None
    assert result.headers == ["Produit", "Quantité", "Prix"]
    assert len(result.data_rows) == 2
    assert result.data_rows[0] == ["Vis M4", "100", "0.05"]
    assert result.data_rows[1] == ["Écrou", "50", "0.10"]


def test_robust_parser_separator_skipped():
    from app.services.chunking_service import _parse_markdown_table_robust

    result = _parse_markdown_table_robust(_TABLE_3COL)
    assert result is not None
    # Aucune ligne de données ne doit être la ligne séparatrice
    for row in result.data_rows:
        assert not all(set(c) <= {"-", ":", ""} for c in row)


def test_robust_parser_missing_cell_flagged():
    from app.services.chunking_service import _parse_markdown_table_robust

    # Ligne 1 : 3 colonnes ; Ligne 2 : 2 colonnes seulement (colonne manquante)
    text = (
        "| A | B | C |\n"
        "|---|---|---|\n"
        "| v1 | v2 | v3 |\n"
        "| x1 | x2 |\n"
    )
    result = _parse_markdown_table_robust(text)
    assert result is not None
    assert 1 in result.suspicious_row_indices
    # La ligne courte doit avoir été paddée
    assert len(result.data_rows[1]) == 3
    assert result.data_rows[1][2] == ""


def test_robust_parser_empty_cell_middle():
    from app.services.chunking_service import _parse_markdown_table_robust

    text = (
        "| A | B | C |\n"
        "|---|---|---|\n"
        "| v1 |  | v3 |\n"
    )
    result = _parse_markdown_table_robust(text)
    assert result is not None
    assert 0 in result.empty_cell_map
    assert 1 in result.empty_cell_map[0]


def test_robust_parser_single_row():
    from app.services.chunking_service import _parse_markdown_table_robust

    text = "| Col1 | Col2 |\n|------|------|\n| val  | 42   |\n"
    result = _parse_markdown_table_robust(text)
    assert result is not None
    assert len(result.data_rows) == 1
    assert result.data_rows[0] == ["val", "42"]


def test_robust_parser_no_table_returns_none():
    from app.services.chunking_service import _parse_markdown_table_robust

    assert _parse_markdown_table_robust("") is None
    assert _parse_markdown_table_robust("Texte sans tableau") is None
    assert _parse_markdown_table_robust("- item\n- item2") is None


def test_robust_parser_unicode_normalization():
    from app.services.chunking_service import _parse_markdown_table_robust

    # Tiret long U+2013 et espace insécable U+00A0
    text = "| Clé\u00a0| Valeur |\n|---|---|\n| a\u2013b | 42 |\n"
    result = _parse_markdown_table_robust(text)
    assert result is not None
    # L'espace insécable dans l'en-tête doit être normalisé
    assert "\u00a0" not in result.headers[0]
    # Le tiret long dans la valeur doit être converti en tiret ASCII
    assert "\u2013" not in result.data_rows[0][0]
    assert "-" in result.data_rows[0][0]


# ---------------------------------------------------------------------------
# Groupe B — _table_row_chunk_text enrichi
# ---------------------------------------------------------------------------

def test_row_chunk_text_basic():
    from app.services.chunking_service import _table_row_chunk_text

    text = _table_row_chunk_text(
        headers=["Produit", "Prix"],
        cells=["Vis M4", "0.05"],
        parent_heading="2 Références",
        caption=None,
        page_no=None,
        table_id="tbl-001",
        row_index=0,
        total_rows=3,
    )
    assert "col1:Produit=Vis M4" in text
    assert "col2:Prix=0.05" in text
    assert "Ligne 1/3" in text


def test_row_chunk_text_empty_cell_marker():
    from app.services.chunking_service import _table_row_chunk_text

    text = _table_row_chunk_text(
        headers=["A", "B", "C"],
        cells=["v1", "", "v3"],
        parent_heading="",
        caption=None,
        page_no=None,
        table_id="tbl-x",
        row_index=0,
    )
    assert "[vide]" in text


def test_row_chunk_text_suspicious_flag():
    from app.services.chunking_service import _table_row_chunk_text

    text = _table_row_chunk_text(
        headers=["A", "B", "C"],
        cells=["v1", "v2", ""],
        parent_heading="",
        caption=None,
        page_no=None,
        table_id="tbl-x",
        row_index=0,
        suspicious=True,
    )
    assert "[décalage probable]" in text


def test_row_chunk_text_with_caption_and_page():
    from app.services.chunking_service import _table_row_chunk_text

    text = _table_row_chunk_text(
        headers=["X"],
        cells=["42"],
        parent_heading="3 Mesures",
        caption="Tableau des mesures",
        page_no=7,
        table_id="tbl-y",
        row_index=0,
    )
    assert "Tableau des mesures" in text
    assert "p.7" in text
    assert "3 Mesures" in text


def test_row_chunk_includes_total_row_count():
    from app.services.chunking_service import _table_row_chunk_text

    text = _table_row_chunk_text(
        headers=["X"],
        cells=["42"],
        parent_heading="",
        caption=None,
        page_no=None,
        table_id="tbl-z",
        row_index=1,
        total_rows=5,
    )
    assert "Ligne 2/5" in text


# ---------------------------------------------------------------------------
# Groupe C — chunk table_full
# ---------------------------------------------------------------------------

def _make_table_specs(table_text: str, heading: str = "1 Section"):
    from app.services.chunking_service import _build_docling_hierarchical_specs

    leaf = _FakeLeaf("leaf-tbl", table_text, [heading], label="table")
    return _build_docling_hierarchical_specs({"document_id": 99}, [leaf])


def test_table_full_chunk_generated():
    specs = _make_table_specs(_TABLE_3COL)
    content_types = [s["metadata_json"].get("content_type") for s in specs]
    assert "table_full" in content_types


def test_table_full_markdown_canonical():
    specs = _make_table_specs(_TABLE_3COL)
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    # Le texte doit contenir la grille Markdown avec les en-têtes
    assert "| Produit" in full_spec["content"]
    assert "| Quantité" in full_spec["content"]
    assert "| Prix" in full_spec["content"]
    # La ligne séparatrice doit être présente
    assert "|---" in full_spec["content"] or "----" in full_spec["content"]


def test_table_full_contains_json_in_metadata():
    specs = _make_table_specs(_TABLE_3COL)
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    tj = full_spec["metadata_json"].get("table_json")
    assert isinstance(tj, dict)
    assert "headers" in tj
    assert "rows" in tj
    assert tj["nb_rows"] == 2
    assert tj["nb_cols"] == 3


def test_table_full_suspicious_rows_in_metadata():
    from app.services.chunking_service import _build_docling_hierarchical_specs

    # Tableau avec une ligne courte (va déclencher une ligne suspecte)
    table_text = (
        "| A | B | C |\n"
        "|---|---|---|\n"
        "| v1 | v2 | v3 |\n"
        "| x1 | x2 |\n"
    )
    leaf = _FakeLeaf("leaf-sus", table_text, ["1 Test"], label="table")
    specs = _build_docling_hierarchical_specs({"document_id": 1}, [leaf])
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    assert full_spec["metadata_json"]["suspicious_rows"] != []


# ---------------------------------------------------------------------------
# Groupe D — hiérarchie table_row → table_full
# ---------------------------------------------------------------------------

def test_table_rows_parent_is_table_full():
    specs = _make_table_specs(_TABLE_3COL)
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    table_full_node_id = full_spec["node_id"]
    row_specs = [s for s in specs if s["metadata_json"].get("content_type") == "table_row"]
    assert len(row_specs) > 0
    for row_spec in row_specs:
        assert row_spec["parent_node_id"] == table_full_node_id, (
            f"table_row pointe vers {row_spec['parent_node_id']} "
            f"au lieu de {table_full_node_id}"
        )


def test_table_hierarchy_levels():
    specs = _make_table_specs(_TABLE_3COL)
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    row_specs = [s for s in specs if s["metadata_json"].get("content_type") == "table_row"]
    assert full_spec["hierarchy_level"] == 1
    for row_spec in row_specs:
        assert row_spec["hierarchy_level"] == 2


# ---------------------------------------------------------------------------
# Groupe E — chunk table_summary (clé-valeur)
# ---------------------------------------------------------------------------

def test_table_summary_generated_for_2col():
    specs = _make_table_specs(_TABLE_2COL)
    content_types = [s["metadata_json"].get("content_type") for s in specs]
    assert "table_summary" in content_types


def test_table_summary_not_generated_for_wide_table():
    specs = _make_table_specs(_TABLE_5COL)
    content_types = [s["metadata_json"].get("content_type") for s in specs]
    assert "table_summary" not in content_types


def test_table_summary_parent_is_table_full():
    specs = _make_table_specs(_TABLE_2COL)
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    summary_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_summary")
    assert summary_spec["parent_node_id"] == full_spec["node_id"]


# ---------------------------------------------------------------------------
# Groupe F — non-régression
# ---------------------------------------------------------------------------

def test_non_table_leaf_unaffected():
    from app.services.chunking_service import _build_docling_hierarchical_specs

    leaf = _FakeLeaf("leaf-txt", "Un paragraphe quelconque.", ["2 Corps"])
    specs = _build_docling_hierarchical_specs({"document_id": 1}, [leaf])
    leaf_specs = [s for s in specs if s["is_leaf"] is True]
    assert len(leaf_specs) == 1
    assert leaf_specs[0]["metadata_json"].get("content_type") == "text_full"
    assert leaf_specs[0]["metadata_json"].get("heading_path") == ["2 Corps"]
    # Corps seul pour l’embedding ; le contexte est dans metadata_json.
    assert leaf_specs[0]["content"] == "Un paragraphe quelconque."


def test_existing_specs_one_heading_two_leaves():
    """Non-régression : 2 feuilles texte ordinaires → 3 specs (1 parent + 2 feuilles text_full)."""
    from app.services.chunking_service import (
        CHUNKING_VERSION_DOCLING_HIERARCHICAL_V2,
        _build_docling_hierarchical_specs,
    )

    leaves = [
        _FakeLeaf("leaf-a", "Premier paragraphe.", ["1 Introduction"]),
        _FakeLeaf("leaf-b", "Deuxième paragraphe.", ["1 Introduction"]),
    ]
    specs = _build_docling_hierarchical_specs({"document_id": 42}, leaves)
    assert len(specs) == 3
    assert specs[0]["is_leaf"] is False
    assert specs[1]["is_leaf"] is True
    assert specs[2]["is_leaf"] is True
    assert specs[0]["metadata_json"].get("chunking_version") == CHUNKING_VERSION_DOCLING_HIERARCHICAL_V2
    assert specs[1]["metadata_json"].get("content_type") == "text_full"
    assert specs[2]["metadata_json"].get("content_type") == "text_full"
    assert specs[1]["parent_node_id"] == specs[0]["node_id"]
    assert specs[2]["parent_node_id"] == specs[0]["node_id"]


def test_format_text_full_chunk_text():
    from app.services.chunking_service import _format_text_full_chunk_text

    t = _format_text_full_chunk_text(
        "Corps du texte.",
        ["Chapitre 1", "Section A"],
        "Chapitre 1 Section A",
        3,
    )
    assert t == "Corps du texte."


def test_text_windows_when_threshold_enabled(monkeypatch):
    from app.config import settings
    from app.services.chunking_service import _build_docling_hierarchical_specs

    monkeypatch.setattr(settings, "DOCLING_TEXT_WINDOW_CHAR_THRESHOLD", 80)
    monkeypatch.setattr(settings, "DOCLING_TEXT_WINDOW_OVERLAP", 10)

    long_body = "x" * 200
    leaf = _FakeLeaf("leaf-long", long_body, ["1 Intro"])
    specs = _build_docling_hierarchical_specs({"document_id": 1}, [leaf])
    types = [s["metadata_json"].get("content_type") for s in specs if s["is_leaf"]]
    assert "text_full" in types
    assert "text_window" in types
    win_specs = [s for s in specs if s["metadata_json"].get("content_type") == "text_window"]
    assert len(win_specs) >= 2
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "text_full")
    for ws in win_specs:
        assert ws["parent_node_id"] == full_spec["node_id"]
        assert ws["hierarchy_level"] == 2


# ---------------------------------------------------------------------------
# Tests anciens (inchangés)
# ---------------------------------------------------------------------------

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
