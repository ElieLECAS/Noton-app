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
    # Le volume du périmètre est compté par count_rows (côté moteur) ; seule reste la
    # récupération exacte des patches (chunk_id, document_id, vector).
    mock_table.count_rows.return_value = 4
    fetch_builder = MockQueryBuilder(to_list_return=all_patches)

    mock_table.search.side_effect = [fetch_builder]

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

        # Le pré-comptage passe par count_rows, jamais par une matérialisation de lignes.
        mock_table.count_rows.assert_called_once_with(filter="document_id in (10)")
        assert fetch_builder.calls[0] == ("where", "document_id in (10)")
        assert fetch_builder.calls[1] == ("select", ["chunk_id", "document_id", "vector"])


def test_search_colpali_lancedb_large_scale():
    # Setup test vectors: 1 query token
    query_token_embeddings = [[1.0] + [0.0] * 127]

    mock_table = mock.Mock()

    # 1. Volume du périmètre > seuil exact -> bascule sur la présélection ANN
    mock_table.count_rows.return_value = 200000

    # 2. Token-level search (limit 250) returns chunk 1
    token_builder = MockQueryBuilder(to_list_return=[{"chunk_id": 1}])

    # 3. Main exact fetch for the candidate chunk 1
    chunk_patches = [{"chunk_id": 1, "document_id": 10, "vector": [1.0] + [0.0] * 127}]
    fetch_builder = MockQueryBuilder(to_list_return=chunk_patches)

    mock_table.search.side_effect = [
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
        assert token_builder.calls[2] == ("select", ["chunk_id"])
        assert token_builder.calls[3] == ("limit", 250)
        # Les identifiants candidats sont lus en Arrow, pas matérialisés en dicts Python.
        assert "to_arrow" in token_builder.calls

        # Check final fetch filtered specifically by candidate chunks
        assert fetch_builder.calls[0] == ("where", "chunk_id in (1)")
        assert fetch_builder.calls[1] == ("select", ["chunk_id", "document_id", "vector"])


def _unit(v):
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).tolist()


class TestNormalisationInvariant:
    """Les vecteurs sont normalisés À L'ÉCRITURE ; la recherche ne renormalise plus."""

    def test_patches_normalises_sautent_la_renormalisation(self):
        from app.services.lancedb_service import _patches_are_normalized

        vecs = np.array([_unit([1.0] + [0.0] * 127), _unit([0.0, 3.0] + [0.0] * 126)], dtype=np.float32)
        assert _patches_are_normalized(vecs) is True

    def test_patches_non_normalises_detectes(self):
        from app.services.lancedb_service import _patches_are_normalized

        vecs = np.array([[2.0] + [0.0] * 127, [0.0, 5.0] + [0.0] * 126], dtype=np.float32)
        assert _patches_are_normalized(vecs) is False

    def test_repli_donne_le_meme_score_que_la_normalisation(self):
        """Un document mal écrit doit être rattrapé : mêmes scores qu'avec des vecteurs unitaires."""
        query = [[1.0] + [0.0] * 127, [0.0, 1.0] + [0.0] * 126]
        # Patches volontairement NON normalisés (norme 4) : sans repli, les scores
        # exploseraient au-delà de 1 et l'ordre comme la distance seraient faux.
        patches = [
            {"chunk_id": 1, "document_id": 10, "vector": [4.0] + [0.0] * 127},
            {"chunk_id": 1, "document_id": 10, "vector": [0.0, 4.0] + [0.0] * 126},
        ]
        mock_table = mock.Mock()
        mock_table.count_rows.return_value = 2
        mock_table.search.side_effect = [MockQueryBuilder(to_list_return=patches)]

        with mock.patch("app.services.lancedb_service.get_colpali_table", return_value=mock_table):
            results = search_colpali_lancedb(query, [10], limit=5)

        assert len(results) == 1
        assert pytest.approx(results[0]["maxsim_score"], abs=1e-4) == 2.0
        assert pytest.approx(results[0]["_distance"], abs=1e-4) == 0.0

    def test_insert_normalise_les_vecteurs(self):
        """L'invariant est posé à l'écriture, pas supposé à la lecture."""
        from app.services.lancedb_service import insert_colpali_patches_batch_lancedb

        mock_table = mock.Mock()
        mock_table.list_indices.return_value = [mock.Mock()]
        with mock.patch("app.services.lancedb_service.get_colpali_table", return_value=mock_table):
            insert_colpali_patches_batch_lancedb(
                document_id=7,
                chunk_patches_list=[(1, [[3.0] + [0.0] * 127, [0.0, 4.0] + [0.0] * 126])],
            )

        rows = mock_table.add.call_args[0][0]
        assert len(rows) == 2
        for row in rows:
            assert pytest.approx(float(np.linalg.norm(row["vector"])), abs=1e-5) == 1.0
