import pytest
import numpy as np
from unittest import mock
from app.services.lancedb_service import search_colpali_lancedb


class MockQueryBuilder:
    def __init__(self, to_list_return=None):
        self.to_list_return = to_list_return if to_list_return is not None else []
        self.calls = []

    def where(self, clause):
        self.calls.append(("where", clause))
        return self

    def select(self, cols):
        self.calls.append(("select", cols))
        return self

    def limit(self, val):
        self.calls.append(("limit", val))
        return self

    def metric(self, val):
        self.calls.append(("metric", val))
        return self

    def to_list(self):
        self.calls.append("to_list")
        return self.to_list_return

    def to_arrow(self):
        self.calls.append("to_arrow")
        import pyarrow as pa
        return pa.Table.from_pylist(self.to_list_return)


def test_search_colpali_lancedb_small_scale():
    # Setup test vectors: 2 query tokens of dim 128
    query_token_embeddings = [
        [1.0] + [0.0] * 127,  # Token 1: points along x-axis
        [0.0, 1.0] + [0.0] * 126,  # Token 2: points along y-axis
    ]

    # Two chunks (pages). Chunk 1 has perfect matches. Chunk 2 has poor matches.
    # Each chunk has 2 patches.
    chunk_1_patches = [
        {"chunk_id": 1, "document_id": 10, "vector": [1.0] + [0.0] * 127},      # Perfect match for Token 1
        {"chunk_id": 1, "document_id": 10, "vector": [0.0, 1.0] + [0.0] * 126}, # Perfect match for Token 2
    ]
    chunk_2_patches = [
        {"chunk_id": 2, "document_id": 10, "vector": [-1.0] + [0.0] * 127},     # Anti-match for Token 1
        {"chunk_id": 2, "document_id": 10, "vector": [0.0, -1.0] + [0.0] * 126},# Anti-match for Token 2
    ]
    all_patches = chunk_1_patches + chunk_2_patches

    mock_table = mock.Mock()
    
    # We mock three successive table.search() builder chains:
    # 1. Quick count check (selects only chunk_id): returns 4 patches (<= 150000)
    # 2. Main exact MaxSim fetch (selects chunk_id, document_id, vector): returns all_patches
    quick_builder = MockQueryBuilder(to_list_return=[{"chunk_id": 1}, {"chunk_id": 1}, {"chunk_id": 2}, {"chunk_id": 2}])
    fetch_builder = MockQueryBuilder(to_list_return=all_patches)

    mock_table.search.side_effect = [quick_builder, fetch_builder]

    with mock.patch("app.services.lancedb_service.get_colpali_table", return_value=mock_table):
        results = search_colpali_lancedb(
            query_token_embeddings=query_token_embeddings,
            document_ids=[10],
            limit=5
        )

        assert len(results) == 2
        
        # Chunk 1 must be ranked first (distance closest to 0.0)
        assert results[0]["id"] == 1
        # MaxSim sum for Chunk 1:
        # Token 1 matches patch 1 with sim 1.0
        # Token 2 matches patch 2 with sim 1.0
        # Sum = 2.0. Average = 2.0 / 2 = 1.0. Distance = 1.0 - 1.0 = 0.0
        assert pytest.approx(results[0]["_distance"], abs=1e-4) == 0.0
        assert pytest.approx(results[0]["maxsim_score"], abs=1e-4) == 2.0

        # Chunk 2 must be ranked second
        # Token 1 matches patch 1 with sim -1.0 and patch 2 with sim 0.0 -> max is 0.0.
        # Token 2 matches patch 1 with sim 0.0 and patch 2 with sim -1.0 -> max is 0.0.
        # Sum = 0.0. Average = 0.0. Distance = 1.0 - 0.0 = 1.0
        assert results[1]["id"] == 2
        assert pytest.approx(results[1]["_distance"], abs=1e-4) == 1.0
        assert pytest.approx(results[1]["maxsim_score"], abs=1e-4) == 0.0

        # Verify calls on mock builders
        assert quick_builder.calls[0] == ("where", "document_id in (10)")
        assert quick_builder.calls[1] == ("select", ["chunk_id"])
        
        assert fetch_builder.calls[0] == ("where", "document_id in (10)")
        assert fetch_builder.calls[1] == ("select", ["chunk_id", "document_id", "vector"])


def test_search_colpali_lancedb_large_scale():
    # Setup test vectors: 1 query token
    query_token_embeddings = [[1.0] + [0.0] * 127]

    mock_table = mock.Mock()

    # 1. Quick count check returns > 150000 records (e.g. 200,000)
    quick_builder = MockQueryBuilder(to_list_return=[{"chunk_id": i} for i in range(200000)])
    
    # 2. Token-level search (limit 250) returns chunk 1
    token_builder = MockQueryBuilder(to_list_return=[{"chunk_id": 1}])
    
    # 3. Main exact fetch for the candidate chunk 1
    chunk_patches = [{"chunk_id": 1, "document_id": 10, "vector": [1.0] + [0.0] * 127}]
    fetch_builder = MockQueryBuilder(to_list_return=chunk_patches)

    mock_table.search.side_effect = [
        quick_builder,  # Quick check
        token_builder,  # Token search (1st token)
        fetch_builder,  # Fetch for candidate chunks
    ]

    with mock.patch("app.services.lancedb_service.get_colpali_table", return_value=mock_table):
        results = search_colpali_lancedb(
            query_token_embeddings=query_token_embeddings,
            document_ids=[10],
            limit=5
        )

        assert len(results) == 1
        assert results[0]["id"] == 1
        assert pytest.approx(results[0]["_distance"], abs=1e-4) == 0.0

        # Verify correct transition to large-scale pruning
        # Check token builder was limited to 250
        assert token_builder.calls[0] == ("metric", "cosine")
        assert token_builder.calls[1] == ("where", "document_id in (10)")
        assert token_builder.calls[2] == ("select", ["chunk_id", "_distance"])
        assert token_builder.calls[3] == ("limit", 250)

        # Check final fetch filtered specifically by candidate chunks
        assert fetch_builder.calls[0] == ("where", "chunk_id in (1)")
        assert fetch_builder.calls[1] == ("select", ["chunk_id", "document_id", "vector"])
