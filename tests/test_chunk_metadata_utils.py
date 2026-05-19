"""Tests utilitaires métadonnées chunks."""
from app.services.chunk_metadata_utils import (
    build_embedding_input_text,
    content_type_score_multiplier,
    merged_chunk_metadata,
    meta_is_leaf,
    mmr_diversity_key,
    parent_llm_context_block,
    table_citation_hint,
)


def test_merged_chunk_metadata_primary_wins():
    legacy = {"heading": "Ancien", "page_no": 1}
    primary = {"heading": "Nouveau"}
    m = merged_chunk_metadata(primary, legacy)
    assert m["heading"] == "Nouveau"
    assert m["page_no"] == 1


def test_meta_is_leaf_string_and_bool():
    assert meta_is_leaf({"is_leaf": "true"}) is True
    assert meta_is_leaf({"is_leaf": "false"}) is False
    assert meta_is_leaf({"is_leaf": True}) is True
    assert meta_is_leaf({}, column_value=False) is False


def test_mmr_diversity_key_table_before_parent():
    assert mmr_diversity_key({"table_id": "t1", "parent_node_id": "p1"}) == "table:t1"
    assert mmr_diversity_key({"parent_node_id": "p1"}) == "parent:p1"


def test_content_type_boost_and_suspicious_penalty():
    assert content_type_score_multiplier({"content_type": "table_summary"}) > 1.0
    base = content_type_score_multiplier({"content_type": "text_full"})
    assert content_type_score_multiplier({"content_type": "text_full", "suspicious": True}) < base


def test_build_embedding_input_includes_heading_and_figure():
    text = build_embedding_input_text(
        "corps",
        {
            "parent_heading": "Chapitre 1",
            "figure_title": "Figure 2 — Schéma",
        },
    )
    assert "[Chapitre 1]" in text
    assert "[Figure: Figure 2 — Schéma]" in text
    assert "corps" in text


def test_parent_llm_context_block():
    block = parent_llm_context_block(
        {
            "summary": "Section sur les profils.",
            "generated_questions": ["Quel profil ?", "Quelle norme ?"],
        }
    )
    assert "Résumé de section" in block
    assert "Questions clés" in block


def test_table_citation_hint_from_table_json():
    hint = table_citation_hint(
        {
            "content_type": "table_row",
            "table_id": "tbl-1",
            "row_index": 2,
            "table_json": {"headers": ["A", "B"], "nb_rows": 5, "nb_cols": 2},
        }
    )
    assert hint["table_id"] == "tbl-1"
    assert hint["row_index"] == 2
    assert hint["headers"] == ["A", "B"]
