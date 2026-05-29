"""
Tests unitaires ColPali-only pour space_search_service.
"""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from unittest import mock

from app.services.space_search_service import _format_colpali_pages


def test_format_colpali_pages_deduplicates_by_document_and_page():
    nodes = [
        NodeWithScore(
            node=TextNode(
                id_="chunk-1",
                text="",
                metadata={
                    "document_id": 10,
                    "document_title": "Notice A",
                    "page_no": 3,
                    "chunk_index": 2,
                },
            ),
            score=0.92,
        ),
        NodeWithScore(
            node=TextNode(
                id_="chunk-2",
                text="",
                metadata={
                    "document_id": 10,
                    "document_title": "Notice A",
                    "page_no": 3,
                    "chunk_index": 2,
                },
            ),
            score=0.88,
        ),
        NodeWithScore(
            node=TextNode(
                id_="chunk-3",
                text="",
                metadata={
                    "document_id": 11,
                    "document_title": "Notice B",
                    "page_start": 5,
                    "chunk_index": 4,
                },
            ),
            score=0.75,
        ),
    ]

    passages = _format_colpali_pages(nodes, k=5)
    assert len(passages) == 2
    assert passages[0]["document_id"] == 10
    assert passages[0]["page_no"] == 3
    assert passages[0]["content_type"] == "colpali_page"
    assert passages[1]["document_id"] == 11
    assert passages[1]["page_no"] == 5


@pytest.mark.asyncio
async def test_search_relevant_passages_colpali_only():
    from app.services import space_search_service

    session = mock.MagicMock()

    with mock.patch("app.services.space_search_service.get_space_by_id") as mock_get_space, \
         mock.patch("app.services.space_search_service._retrieve_colpali_pages") as mock_retrieve:

        mock_get_space.return_value = mock.MagicMock()
        mock_retrieve.return_value = [
            NodeWithScore(
                node=TextNode(
                    id_="chunk-42",
                    text="",
                    metadata={
                        "document_id": 7,
                        "document_title": "Doc test",
                        "page_no": 2,
                        "chunk_index": 1,
                    },
                ),
                score=0.81,
            )
        ]

        result = await space_search_service.search_relevant_passages(
            session=session,
            space_id=1,
            query_text="cote vitrage",
            user_id=1,
            k=3,
        )

        assert result["status"] == "ok"
        passages = result["passages"]
        assert len(passages) == 1
        assert passages[0]["chunk_id"] == 42
        assert passages[0]["page_no"] == 2
        assert "passage" not in passages[0]
