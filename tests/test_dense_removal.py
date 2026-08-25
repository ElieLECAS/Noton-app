"""Retrait de la voie dense texte (pgvector) du retriever (2026-08-25).

Vérifie que le retrieval se réduit bien à ColPali + BM25, et que plus rien ne
dépend du canal pgvector. Miroir de test_kag_removal.py (retrait du canal KAG).
"""
import pytest

from app.services.context_packer_service import _CHANNEL_FAMILIES
from app.services.page_retrieval_service import UnifiedPageHit, fuse_multimodal_hits


class TestFusionSansPgvector:
    def test_fuse_n_accepte_plus_de_canal_pgvector(self):
        """Le paramètre pgvector_hits a disparu de la signature de fusion."""
        with pytest.raises(TypeError):
            fuse_multimodal_hits([], [], pgvector_hits=[], rrf_k=60, top_k=10)

    def test_hit_n_a_plus_de_pgvector_score(self):
        hit = UnifiedPageHit(document_id=1, page_no=2)
        assert not hasattr(hit, "pgvector_score")

    def test_fusion_deux_canaux(self):
        colpali = [UnifiedPageHit(document_id=1, page_no=1, colpali_score=0.9)]
        bm25 = [UnifiedPageHit(document_id=2, page_no=5, bm25_score=0.4)]

        fused = fuse_multimodal_hits(colpali, bm25, rrf_k=60, top_k=10)

        assert len(fused) == 2
        top = fused[0]
        assert (top.document_id, top.page_no) == (1, 1)
        assert set(top.retrieval_sources) == {"colpali"}
        assert top.rrf_score == pytest.approx(1 / 61, rel=1e-6)


class TestElectionSansFamillePgvector:
    def test_famille_pgvector_retiree(self):
        assert "pgvector" not in _CHANNEL_FAMILIES
        assert set(_CHANNEL_FAMILIES.values()) == {"texte", "visuel"}

    def test_bm25_seul_porte_la_famille_texte(self):
        assert _CHANNEL_FAMILIES["bm25"] == "texte"
        assert _CHANNEL_FAMILIES["colpali"] == "visuel"


class TestRetrievePgvectorSupprime:
    def test_fonctions_retrieval_pgvector_absentes(self):
        import app.services.page_retrieval_service as prs

        for name in (
            "retrieve_pgvector_pages",
            "retrieve_pgvector_page_hits",
            "PageRetrievalHit",
            "fuse_page_hits_rrf",
            "build_weak_hit_pool",
            "expand_neighbor_pages",
            "compute_needs_page_image",
            "format_hybrid_passages",
            "_retrieve_leaves_bm25_sql",
        ):
            assert not hasattr(prs, name), f"{name} devrait avoir été supprimé"

    def test_espace_search_service_sans_pgvector(self):
        import app.services.space_search_service as sss

        assert not hasattr(sss, "_retrieve_leaves_sql")
