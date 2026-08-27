"""Parallélisation des retrievers (2 canaux depuis le retrait de KAG et pgvector).

Le volet « plafond du boost catégorie » a été supprimé avec le boost lui-même
(2026-07-28) : voir tests/test_kag_removal.py pour les invariants qui le remplacent.
"""
from __future__ import annotations

from unittest import mock

import pytest

from app.services import space_search_service as sss


def _patch_retrievers():
    import app.services.page_retrieval_service as prs

    # **kwargs : retrieve_colpali_pages accepte precomputed_query_embeddings (réutilisation
    # de l'encodage de la requête entre la phase A et la phase B du retriever).
    return (
        mock.patch.object(prs, "retrieve_colpali_pages", lambda s, d, q, p, **kw: ["colpali"]),
        mock.patch.object(prs, "filter_colpali_pages_dynamic", lambda hits: hits),
        mock.patch.object(prs, "retrieve_bm25_pages", lambda s, d, q, p, **kw: ["bm25"]),
    )


@pytest.mark.asyncio
async def test_run_retrievers_parallel_matches_sequential(db_session):
    """Parallèle et séquentiel doivent rendre le MÊME résultat : le mode parallèle est
    une optimisation (threads + sessions dédiées), pas un changement de sémantique."""
    patches = _patch_retrievers()
    with patches[0], patches[1], patches[2]:
        with mock.patch("app.config.settings.RETRIEVAL_PARALLEL_ENABLED", True):
            par = await sss._run_retrievers(
                db_session, 1, [1, 2], "cq", "lq", 10
            )
        with mock.patch("app.config.settings.RETRIEVAL_PARALLEL_ENABLED", False):
            seq = await sss._run_retrievers(
                db_session, 1, [1, 2], "cq", "lq", 10
            )

    assert par == (["colpali"], ["bm25"])
    assert seq == par


@pytest.mark.asyncio
async def test_run_retrievers_respecte_le_gate_colpali(db_session):
    """use_colpali=False → ni encode ColQwen2 ni MaxSim (le poste le plus lourd)."""
    patches = _patch_retrievers()
    with patches[0], patches[1], patches[2], mock.patch(
        "app.config.settings.RETRIEVAL_PARALLEL_ENABLED", True
    ):
        result = await sss._run_retrievers(
            db_session, 1, [1], "cq", "lq", 10, use_colpali=False
        )

    assert result == ([], ["bm25"])
