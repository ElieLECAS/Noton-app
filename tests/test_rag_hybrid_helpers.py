"""Tests légers pour parsing tableaux, fusion hybride RAG et multi-hop (sans DB)."""

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


# ---------------------------------------------------------------------------
# Tests multi-hop
# ---------------------------------------------------------------------------

def test_needs_multi_hop_trigger_keyword():
    from app.services.space_search_service import _needs_multi_hop

    assert _needs_multi_hop("Quel est l'impact de X sur Y ?", []) is True
    assert _needs_multi_hop("Comparaison entre A et B", []) is True
    assert _needs_multi_hop("Quelle est la cause du problème ?", []) is True
    assert _needs_multi_hop("Si X alors que se passe-t-il pour Y ?", []) is True


def test_needs_multi_hop_two_pivots():
    from app.services.space_search_service import _needs_multi_hop

    assert _needs_multi_hop("Donne moi les caractéristiques", ["entite_a", "entite_b"]) is True


def test_needs_multi_hop_false_simple_query():
    from app.services.space_search_service import _needs_multi_hop

    assert _needs_multi_hop("Qu'est-ce que le produit X ?", []) is False
    assert _needs_multi_hop("Prix du produit Y", ["entite_unique"]) is False


def test_needs_multi_hop_empty_query():
    from app.services.space_search_service import _needs_multi_hop

    assert _needs_multi_hop("", []) is False
    assert _needs_multi_hop("", ["a", "b"]) is False


def test_multihop_state_initializes_empty():
    from app.services.space_search_service import _MultiHopState

    state = _MultiHopState()
    assert state.seen_chunk_ids == set()
    assert state.seen_entity_ids == set()
    assert state.seen_entity_names == set()
    assert state.chunk_signals == {}
    assert state.new_chunks_count_by_hop == {}
    assert state.hop_traces == {}


def test_extract_top_entity_names_from_candidates():
    from app.services.space_search_service import _extract_top_entity_names_from_candidates

    n1 = TextNode(id_="chunk-1", text="a", metadata={"kag_matched_entity": "Produit A"})
    n2 = TextNode(id_="chunk-2", text="b", metadata={"kag_matched_entity": "Produit A"})
    n3 = TextNode(id_="chunk-3", text="c", metadata={"kag_matched_entity": "Matériau B"})
    candidates = [
        NodeWithScore(node=n1, score=0.9),
        NodeWithScore(node=n2, score=0.8),
        NodeWithScore(node=n3, score=0.7),
    ]
    entities = _extract_top_entity_names_from_candidates(candidates, top_n=5)
    assert "produit a" in entities
    assert entities[0] == "produit a"  # le plus fréquent en premier
    assert "matériau b" in entities


def test_extract_top_entity_names_no_kag_match():
    from app.services.space_search_service import _extract_top_entity_names_from_candidates

    candidates = [
        NodeWithScore(node=TextNode(id_="chunk-1", text="x", metadata={}), score=0.5)
    ]
    assert _extract_top_entity_names_from_candidates(candidates) == []


def test_apply_multihop_depth_scoring_penalizes_deep_hops():
    from app.services.space_search_service import (
        _MultiHopState,
        _apply_multihop_depth_scoring,
    )

    state = _MultiHopState()
    state.chunk_signals = {
        1: {"vector": 0.8, "lexical": 0.5, "kag": 0.9, "evidence": 1.0, "hop": 0, "path": "hop0"},
        2: {"vector": 0.8, "lexical": 0.5, "kag": 0.9, "evidence": 1.0, "hop": 2, "path": "hop2"},
    }
    n1 = TextNode(id_="chunk-1", text="a", metadata={})
    n2 = TextNode(id_="chunk-2", text="b", metadata={})
    all_nodes = {
        1: NodeWithScore(node=n1, score=0.8),
        2: NodeWithScore(node=n2, score=0.8),
    }
    scored = _apply_multihop_depth_scoring(state, all_nodes)
    assert len(scored) == 2
    scores_by_chunk = {
        int(nws.node.id_.split("-")[1]): float(nws.score) for nws in scored
    }
    # hop0 doit scorer plus haut que hop2 (pénalité 0.00 vs 0.10)
    assert scores_by_chunk[1] > scores_by_chunk[2]


def test_apply_multihop_depth_scoring_metadata_populated():
    from app.services.space_search_service import (
        _MultiHopState,
        _apply_multihop_depth_scoring,
    )

    state = _MultiHopState()
    state.chunk_signals = {
        5: {"vector": 0.5, "lexical": 0.3, "kag": 0.7, "evidence": 0.0, "hop": 1, "path": "hop1:ent"},
    }
    node = TextNode(id_="chunk-5", text="x", metadata={})
    all_nodes = {5: NodeWithScore(node=node, score=0.5)}
    scored = _apply_multihop_depth_scoring(state, all_nodes)
    assert len(scored) == 1
    meta = scored[0].node.metadata or {}
    assert meta["retrieval_hop"] == 1
    assert meta["hop_penalty"] == 0.05
    assert "retrieval_path" in meta


def test_apply_multihop_depth_scoring_score_non_negative():
    from app.services.space_search_service import (
        _MultiHopState,
        _apply_multihop_depth_scoring,
    )

    state = _MultiHopState()
    state.chunk_signals = {
        9: {"vector": 0.0, "lexical": 0.0, "kag": 0.0, "evidence": 0.0, "hop": 3, "path": "hop3"},
    }
    node = TextNode(id_="chunk-9", text="z", metadata={})
    all_nodes = {9: NodeWithScore(node=node, score=0.0)}
    scored = _apply_multihop_depth_scoring(state, all_nodes)
    assert scored[0].score >= 0.0


def test_multihop_constants_values():
    from app.services.space_search_service import (
        MULTI_HOP_ENABLED,
        MULTI_HOP_MAX_HOPS,
        MULTI_HOP_CANDIDATE_BUDGET,
        MULTI_HOP_PER_HOP_LIMIT,
        MULTI_HOP_PATIENCE,
        MH_HOP_PENALTIES,
    )

    assert MULTI_HOP_ENABLED is True
    assert MULTI_HOP_MAX_HOPS == 3
    assert MULTI_HOP_CANDIDATE_BUDGET >= MULTI_HOP_PER_HOP_LIMIT
    assert MULTI_HOP_PATIENCE >= 1
    assert MH_HOP_PENALTIES[0] == 0.00
    assert MH_HOP_PENALTIES[1] < MH_HOP_PENALTIES[2]
    assert MH_HOP_PENALTIES[2] < MH_HOP_PENALTIES[3]


def test_hybrid_fuse_dedup_after_multihop():
    """Non-régression : la fusion hybride déduplique correctement (base du pipeline)."""
    from app.services.space_search_service import _hybrid_fuse_candidates

    v = [NodeWithScore(node=TextNode(id_="chunk-42", text="a"), score=0.7)]
    l = [NodeWithScore(node=TextNode(id_="chunk-42", text="a"), score=0.4)]
    k = [NodeWithScore(node=TextNode(id_="chunk-42", text="a"), score=0.9)]
    merged = _hybrid_fuse_candidates(v, l, k)
    assert len(merged) == 1  # dédup strict sur chunk-id


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
    # 1 parent section + table_full + table_summary (2 col) + 2 lignes table_row
    assert len(specs) == 5
    full_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_full")
    summary_spec = next(s for s in specs if s["metadata_json"].get("content_type") == "table_summary")
    row_specs = [s for s in specs if s["metadata_json"].get("content_type") == "table_row"]
    assert len(row_specs) == 2
    assert row_specs[0]["metadata_json"].get("column_headers") == ["Col1", "Col2"]
    assert row_specs[0]["parent_node_id"] == full_spec["node_id"]
    assert summary_spec["parent_node_id"] == full_spec["node_id"]


def test_resolve_space_parent_multihop_no_document_id():
    """Smoke : sans document_id la chaîne multi-saut ne peut pas charger les chunks."""
    from unittest.mock import MagicMock

    from app.services.space_search_service import _resolve_space_parent_with_multihop

    session = MagicMock()
    assert (
        _resolve_space_parent_with_multihop(session, 1, 1, None, "some-uuid", {})
        is None
    )


def test_resolve_space_parent_multihop_delegates_when_parent_in_dict():
    """Si le parent section est déjà dans le dict, le multi-saut ne s'applique pas (appelant)."""
    from app.services.space_search_service import _resolve_space_parent_with_multihop

    fake = TextNode(id_="p1", text="section", metadata={})
    assert (
        _resolve_space_parent_with_multihop(
            None, 1, 1, 42, "p1", {"p1": fake}
        )
        is None
    )


def test_resolve_note_parent_multihop_no_note_id():
    from unittest.mock import MagicMock

    from app.services.semantic_search_service import _resolve_note_parent_with_multihop

    session = MagicMock()
    assert (
        _resolve_note_parent_with_multihop(session, 1, 1, None, "some-uuid", {})
        is None
    )


def test_resolve_note_parent_multihop_skips_when_parent_in_dict():
    from app.services.semantic_search_service import _resolve_note_parent_with_multihop

    fake = TextNode(id_="p1", text="section", metadata={})
    assert _resolve_note_parent_with_multihop(None, 1, 1, 1, "p1", {"p1": fake}) is None
