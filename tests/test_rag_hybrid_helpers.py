"""Tests légers : parsing tableaux Docling et résolution parents (sans DB)."""

import pytest
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

# Note: Les tests de résolution parent multihop ont été supprimés car la logique associée
# a été retirée du retriever.



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
    
    res = reciprocal_rank_fusion(vector, lexical, alphanumeric_results=alphanumeric, top_n=3, normalize=True)
    assert len(res) <= 3
    # Tous les chunks doivent être présents
    node_ids = {r.node.id_ for r in res}
    assert "chunk-1" in node_ids
    assert "chunk-2" in node_ids
    assert "chunk-3" in node_ids
    
    # Les scores doivent être normalisés dans [0.1, 0.9]
    for r in res:
        assert 0.1 <= r.score <= 0.9


@pytest.mark.asyncio
async def test_space_search_window_aggregation_and_deduplication():
    from unittest import mock
    from llama_index.core.schema import TextNode, NodeWithScore
    from app.services import space_search_service

    session = mock.MagicMock()

    with mock.patch("app.services.space_search_service.get_space_by_id") as mock_get_space, \
         mock.patch("app.services.space_search_service.generate_embedding") as mock_emb, \
         mock.patch("app.services.space_search_service._retrieve_leaves_sql") as mock_leaves, \
         mock.patch("app.services.space_search_service._retrieve_leaves_bm25_sql") as mock_bm25, \
         mock.patch("app.services.space_search_service._retrieve_leaves_alphanumeric_sql") as mock_alpha, \
         mock.patch("app.services.space_search_service.settings") as mock_settings:

        mock_settings.RERANKER_ENABLED = False
        mock_get_space.return_value = mock.MagicMock()
        mock_emb.return_value = [0.1] * 384

        n1 = NodeWithScore(
            node=TextNode(
                id_="chunk-1",
                text="Contenu de la page brute.",
                metadata={"content_type": "page_raw_enriched", "window_id": "win-test-1", "document_title": "Doc1", "document_id": 123}
            ),
            score=0.9
        )
        n2 = NodeWithScore(
            node=TextNode(
                id_="chunk-2",
                text="Rapport de la fenêtre.",
                metadata={"content_type": "page_window_report", "window_id": "win-test-1", "document_title": "Doc1", "document_id": 123}
            ),
            score=0.8
        )

        mock_leaves.return_value = [n1, n2]
        mock_bm25.return_value = []
        mock_alpha.return_value = []

        mock_raw_chunk = mock.MagicMock()
        mock_raw_chunk.metadata_json = {"window_id": "win-test-1", "content_type": "page_raw_enriched"}
        mock_raw_chunk.metadata_ = None
        mock_raw_chunk.content = "Contenu brut récupéré de la DB."
        mock_raw_chunk.text = None
        mock_raw_chunk.chunk_index = 0
        mock_raw_chunk.id = 1

        mock_report_chunk = mock.MagicMock()
        mock_report_chunk.metadata_json = {"window_id": "win-test-1", "content_type": "page_window_report"}
        mock_report_chunk.metadata_ = None
        mock_report_chunk.content = "Rapport de la fenêtre récupéré de la DB."
        mock_report_chunk.text = None
        mock_report_chunk.chunk_index = 0
        mock_report_chunk.id = 2

        session.execute.return_value.scalars.return_value.all.return_value = [mock_raw_chunk, mock_report_chunk]

        result = await space_search_service.search_relevant_passages(
            session=session,
            space_id=1,
            query_text="test window aggregation",
            user_id=1,
            k=15
        )

        passages = result.get("passages", [])
        assert len(passages) == 1
        p = passages[0]
        assert p["chunk_id"] == 1
        assert "Rapport de la fenêtre récupéré de la DB." in p["passage"]


def test_build_space_context_from_passages_includes_page_info():
    from app.routers.chat import build_space_context_from_passages

    passages = [
        {
            "passage": "Contenu du passage.",
            "document_title": "Notice Technique",
            "score": 0.85,
            "page_no": 8,
        },
        {
            "passage": "Autre contenu.",
            "document_title": "Guide de Montage",
            "score": 0.75,
            "page_start": 10,
            "page_end": 12,
        }
    ]

    system_msg = build_space_context_from_passages(passages)
    content = system_msg["content"]

    assert "Notice Technique, page 8" in content
    assert "Guide de Montage, pages 10-12" in content


@pytest.mark.asyncio
async def test_search_relevant_passages_query_expansion_fallback():
    import math
    from unittest import mock
    from llama_index.core.schema import TextNode, NodeWithScore
    from app.services import space_search_service
    from app.services.query_reasoning_service import QueryIntent

    session = mock.MagicMock()

    with mock.patch("app.services.space_search_service.get_space_by_id") as mock_get_space, \
         mock.patch("app.services.space_search_service.generate_embedding") as mock_emb, \
         mock.patch("app.services.space_search_service._retrieve_leaves_sql") as mock_leaves, \
         mock.patch("app.services.space_search_service._retrieve_leaves_bm25_sql") as mock_bm25, \
         mock.patch("app.services.space_search_service._retrieve_leaves_alphanumeric_sql") as mock_alpha, \
         mock.patch("app.services.space_search_service.settings") as mock_settings, \
         mock.patch("app.services.query_reasoning_service.reason_query_intent") as mock_reason, \
         mock.patch("app.services.reranker_service.rerank_nodes") as mock_rerank:

        mock_settings.RERANKER_ENABLED = True
        mock_settings.RAG_MIN_PERTINENCE = 0.75
        mock_settings.RERANK_POOL = 100
        mock_settings.MIN_DYNAMIC_K = 1
        mock_settings.MAX_DYNAMIC_K = 5
        mock_settings.SOFTMAX_CUM_THRESHOLD = 0.8
        
        mock_get_space.return_value = mock.MagicMock()
        mock_emb.return_value = [0.1] * 384
        
        # Premier essai avec un nœud non pertinent
        n_low = NodeWithScore(node=TextNode(id_="chunk-111", text="non-pertinent", metadata={"document_title": "Doc1", "document_id": 123}), score=0.1)
        mock_leaves.return_value = [n_low]
        mock_bm25.return_value = []
        mock_alpha.return_value = []
        
        # Premier rerank : retourne score bas (-2.0 -> sigmoid(-2.0) = 0.12 < 0.75)
        # Second rerank (loop) : retourne score élevé (2.0 -> sigmoid(2.0) = 0.88 >= 0.75)
        mock_rerank.side_effect = [
            [(n_low, -2.0)],
            [(NodeWithScore(node=TextNode(id_="chunk-999", text="pertinent", metadata={"document_title": "Doc1", "document_id": 123}), score=0.9), 2.0)]
        ]

        # Query reasoning retourne des termes d'expansion
        mock_reason.return_value = QueryIntent(
            intent="generic",
            reasoning="test",
            confidence=0.9,
            search_terms=["terme_expansion"],
            detected_references=["REF123"]
        )

        n_high = NodeWithScore(node=TextNode(id_="chunk-999", text="pertinent", metadata={"document_title": "Doc1", "document_id": 123}), score=0.9)
        
        # Mock de retrieve pour la boucle d'expansion
        mock_bm25.side_effect = [
            [], # premier essai
            [n_high], # loop term REF123
            [] # loop term terme_expansion
        ]
        
        mock_alpha.side_effect = [
            [], # premier essai
            [], # loop term REF123
            [] # loop term terme_expansion
        ]

        result = await space_search_service.search_relevant_passages(
            session=session,
            space_id=1,
            query_text="requete initiale",
            user_id=1,
            k=15
        )

        # La boucle élargie a fonctionné et a récupéré le passage de la boucle
        passages = result.get("passages", [])
        assert len(passages) == 1
        assert passages[0]["chunk_id"] == 999
        expected_score = 1.0 / (1.0 + math.exp(-(2.0 + 1.5)))
        assert math.isclose(passages[0]["score"], expected_score)



