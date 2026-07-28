"""
Retrait du KAG et du boost catégorie (2026-07-28).

Vérifie que le retrieval se réduit bien à ColPali + pgvector + BM25, que rien ne
dépend plus du graphe, et que les fonctions déménagées avant la suppression sont
toujours joignables depuis leur nouveau module.
"""
import re

import pytest

from app.services.context_packer_service import _CHANNEL_FAMILIES, _election_score
from app.services.page_retrieval_service import UnifiedPageHit, fuse_multimodal_hits


class TestFusionSansKAG:
    def test_fuse_n_accepte_plus_de_canal_kag(self):
        """Le paramètre kag_hits a disparu de la signature de fusion."""
        with pytest.raises(TypeError):
            fuse_multimodal_hits([], [], [], kag_hits=[], rrf_k=60, top_k=10)

    def test_hit_n_a_plus_de_kag_score(self):
        hit = UnifiedPageHit(document_id=1, page_no=2)
        assert not hasattr(hit, "kag_score")

    def test_fusion_trois_canaux(self):
        colpali = [UnifiedPageHit(document_id=1, page_no=1, colpali_score=0.9)]
        pgvector = [UnifiedPageHit(document_id=1, page_no=1, pgvector_score=0.8)]
        bm25 = [UnifiedPageHit(document_id=2, page_no=5, bm25_score=0.4)]

        fused = fuse_multimodal_hits(colpali, pgvector, bm25, rrf_k=60, top_k=10)

        assert len(fused) == 2
        top = fused[0]
        assert (top.document_id, top.page_no) == (1, 1)
        # Deux canaux sur la même page → deux contributions RRF.
        assert set(top.retrieval_sources) == {"colpali", "pgvector"}
        assert top.rrf_score == pytest.approx(2 / 61, rel=1e-6)


class TestElectionSansFamilleGraphe:
    def test_famille_graphe_retiree(self):
        assert "kag" not in _CHANNEL_FAMILIES
        assert set(_CHANNEL_FAMILIES.values()) == {"texte", "visuel"}

    def test_pgvector_et_bm25_restent_une_seule_famille(self):
        """Le garde-fou d'origine subsiste : le double vote lexical ne compte qu'une fois."""
        assert _CHANNEL_FAMILIES["pgvector"] == _CHANNEL_FAMILIES["bm25"]
        assert _CHANNEL_FAMILIES["colpali"] != _CHANNEL_FAMILIES["pgvector"]

    def test_bonus_familles_borne_a_deux(self):
        """Bonus max = +0,10 (2 familles) au lieu de +0,20 quand le graphe existait."""
        deux = _election_score(
            {"score_max": 1.0, "families": {"texte", "visuel"}, "matched_pages": {1: 1.0}}
        )
        une = _election_score(
            {"score_max": 1.0, "families": {"texte"}, "matched_pages": {1: 1.0}}
        )
        assert une == pytest.approx(1.0)
        assert deux == pytest.approx(1.10)

    def test_meilleur_passage_reste_dominant(self):
        """Un meilleur score_max ne peut pas être renversé par les bonus."""
        fort_une_famille = _election_score(
            {"score_max": 0.90, "families": {"texte"}, "matched_pages": {1: 0.9}}
        )
        faible_deux_familles = _election_score(
            {
                "score_max": 0.60,
                "families": {"texte", "visuel"},
                "matched_pages": {1: 0.6, 2: 0.5, 3: 0.4},
            }
        )
        assert fort_une_famille > faible_deux_familles


class TestBoostsRestants:
    def test_boost_categorie_supprime(self):
        import app.services.retrieval_boost_service as rbs

        assert not hasattr(rbs, "apply_category_boost_to_fused_hits")
        assert not hasattr(rbs, "_bulk_get_page_categories")

    def test_boosts_conserves(self):
        import app.services.retrieval_boost_service as rbs

        assert hasattr(rbs, "apply_soft_boosts_to_passages")
        assert hasattr(rbs, "apply_anchor_boost_to_fused_hits")
        assert hasattr(rbs, "compute_anchor_documents")


class TestDemenagements:
    """Symboles déplacés AVANT la suppression du module KAG : sans eux, le bloc
    COUVERTURE et les fenêtres d'enrichissement casseraient."""

    def test_ref_code_re_disponible(self):
        from app.services.reference_codes import REF_CODE_RE

        assert REF_CODE_RE.search("profil 76180")

    def test_coverage_service_utilise_le_nouveau_module(self):
        from app.services import coverage_service
        from app.services.reference_codes import REF_CODE_RE

        assert coverage_service.REF_CODE_RE is REF_CODE_RE

    def test_fenetres_de_pages_dans_le_service_enrichissement(self):
        from app.services.contextual_enrichment_service import build_page_batches

        # 3 pages, overlap 1 → 1-2-3, 3-4-5, 5-6-7
        assert build_page_batches([1, 2, 3, 4, 5, 6, 7]) == [[1, 2, 3], [3, 4, 5], [5, 6, 7]]

    def test_fenetres_vides_et_page_unique(self):
        from app.services.contextual_enrichment_service import build_page_batches

        assert build_page_batches([]) == []
        assert build_page_batches([4]) == [[4]]


class TestPlusDeDependanceKAG:
    """Aucun module du chemin vivant ne doit plus importer les services KAG."""

    def test_modules_vivants_sans_import_kag(self):
        import pathlib

        root = pathlib.Path("app")
        assert root.is_dir(), "test à lancer depuis la racine du projet"

        offenders = []
        pattern = re.compile(r"from app\.services\.kag_\w+ import|import app\.services\.kag_")
        for path in root.rglob("*.py"):
            if path.name.startswith("kag_"):
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path))

        assert offenders == [], f"imports KAG résiduels : {offenders}"
