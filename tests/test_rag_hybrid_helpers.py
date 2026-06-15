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


def test_is_vision_model():
    from app.services.rag_generation_service import is_vision_model

    assert is_vision_model("mistral-large-latest") is True
    assert is_vision_model("pixtral-12b-2409") is True
    assert is_vision_model("mistral-small-latest") is False


def test_collect_unique_page_keys_respects_max():
    from app.services.rag_generation_service import collect_unique_page_keys

    passages = [
        {"document_id": 1, "page_no": 4},
        {"document_id": 1, "page_no": 4},
        {"document_id": 1, "page_no": 2},
        {"document_id": 2, "page_start": 7},
    ]
    keys = collect_unique_page_keys(passages, max_pages=2)
    assert keys == [(1, 4), (1, 2)]


def test_enrich_colpali_passages_with_pymupdf():
    from unittest.mock import MagicMock, patch
    from app.services.rag_generation_service import enrich_colpali_passages_with_pymupdf

    session = MagicMock()
    doc = MagicMock()
    doc.source_file_path = "/tmp/doc.pdf"
    session.get.return_value = doc

    passages = [
        {
            "passage": "**Doc**\n[ColPali Indexed Page 3]",
            "passage_raw": "[ColPali Indexed Page 3]",
            "document_title": "Doc",
            "document_id": 10,
            "page_no": 3,
        }
    ]

    with patch("app.services.rag_generation_service.os.path.exists", return_value=True), \
         patch(
             "app.services.rag_generation_service.extract_page_text_from_pdf",
             return_value="Charge max 24 V / 40 mA",
         ):
        enriched = enrich_colpali_passages_with_pymupdf(session, passages)

    assert "24 V / 40 mA" in enriched[0]["passage_raw"]
    assert "[Page 3]" in enriched[0]["passage"]


@pytest.mark.asyncio
async def test_build_rag_generation_messages_attaches_images_for_vision_model():
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.services.rag_generation_service import build_rag_generation_messages

    session = MagicMock()
    passages = [
        {
            "passage": "[ColPali Indexed Page 1]",
            "passage_raw": "[ColPali Indexed Page 1]",
            "document_title": "Doc",
            "document_id": 1,
            "page_no": 1,
            "score": 0.9,
        }
    ]

    with patch(
        "app.services.rag_generation_service.enrich_colpali_passages_with_pymupdf",
        side_effect=lambda _s, p: p,
    ), patch(
        "app.routers.chat.build_space_context_from_passages",
        return_value={"role": "system", "content": "PASSAGES"},
    ), patch(
        "app.services.rag_generation_service.render_page_images_for_passages_async",
        new=AsyncMock(return_value=["base64img"]),
    ):
        messages = await build_rag_generation_messages(
            session,
            passages,
            "Question test?",
            model="mistral-large-latest",
        )

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Question test?"
    assert messages[1]["images"] == ["base64img"]


@pytest.mark.asyncio
async def test_build_rag_generation_messages_no_images_for_small_model():
    from unittest.mock import AsyncMock, MagicMock, patch
    from app.services.rag_generation_service import build_rag_generation_messages

    session = MagicMock()
    passages = [{"passage": "texte", "document_title": "Doc", "score": 0.5}]

    render_mock = AsyncMock(return_value=["base64img"])
    with patch(
        "app.services.rag_generation_service.enrich_colpali_passages_with_pymupdf",
        side_effect=lambda _s, p: p,
    ), patch(
        "app.routers.chat.build_space_context_from_passages",
        return_value={"role": "system", "content": "PASSAGES"},
    ), patch(
        "app.services.rag_generation_service.render_page_images_for_passages_async",
        new=render_mock,
    ):
        messages = await build_rag_generation_messages(
            session,
            passages,
            "Question?",
            model="mistral-small-latest",
        )

    render_mock.assert_not_called()
    assert "images" not in messages[1]





