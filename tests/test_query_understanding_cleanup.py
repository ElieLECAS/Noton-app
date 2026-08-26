"""Nettoyage du query understanding (2026-08-26) — P0.

Trois implémentations coexistaient, une seule s'exécutait. Ces tests verrouillent la
suppression des deux autres, de la chaîne multi-groupe (D1) et du contrôle de vagueness
pré-retrieval (D2). Miroir de test_kag_removal.py / test_dense_removal.py.
"""
import importlib
import pathlib
import re

import pytest


class TestModulesSupprimes:
    @pytest.mark.parametrize(
        "module",
        [
            "app.services.query_understanding_graph",
            "app.services.query_reasoning_service",
        ],
    )
    def test_module_absent(self, module):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)

    def test_aucun_import_residuel_dans_app(self):
        root = pathlib.Path("app")
        assert root.is_dir(), "test à lancer depuis la racine du projet"

        # Cible les IMPORTS, pas les simples mentions : les docstrings peuvent citer
        # ces modules pour expliquer d'où viennent les symboles déménagés.
        pattern = re.compile(
            r"^\s*(?:from|import)\s+app\.services\.(?:query_understanding_graph|query_reasoning_service)\b",
            re.MULTILINE,
        )
        offenders = [
            str(path)
            for path in root.rglob("*.py")
            if pattern.search(path.read_text(encoding="utf-8"))
        ]
        assert offenders == [], f"imports résiduels : {offenders}"


class TestUnSeulChemin:
    def test_plus_de_graphe_legacy_ni_de_selection(self):
        from app.services import lightweight_query_understanding as lqu

        for name in ("_build_graph_legacy", "_build_graph_fused", "_GRAPH_LEGACY"):
            assert not hasattr(lqu, name), f"{name} devrait avoir été supprimé"
        # Un seul graphe, construit sans condition.
        assert hasattr(lqu, "_build_graph")

    def test_noeuds_morts_supprimes(self):
        from app.services import lightweight_query_understanding as lqu

        for name in (
            "_node_route_decision",
            "_node_extract_signals",
            "_node_assess_vagueness",
            "_node_condense_question",
            "_node_merge_context",
            "_node_plan_multi_query",
            "_node_generate_queries",
        ):
            assert not hasattr(lqu, name), f"{name} devrait avoir été supprimé"

    def test_noeuds_vivants_presents(self):
        from app.services import lightweight_query_understanding as lqu

        assert hasattr(lqu, "_node_fused_understand")
        assert hasattr(lqu, "_node_build_queries_fast")


class TestMultiGroupeSupprime:
    def test_modeles_de_groupe_absents(self):
        from app.services import query_schemas

        assert not hasattr(query_schemas, "QueryGroup")

    def test_fusion_cross_groupes_absente(self):
        from app.services import page_retrieval_service as prs

        assert not hasattr(prs, "fuse_multi_query_groups")

    def test_hit_sans_champ_de_groupe(self):
        from app.services.page_retrieval_service import UnifiedPageHit

        hit = UnifiedPageHit(document_id=1, page_no=1)
        assert not hasattr(hit, "query_group_index")
        assert not hasattr(hit, "query_group_label")

    def test_recherche_sans_parametre_query_groups(self):
        import inspect

        from app.services.space_search_service import (
            search_multimodal_passages,
            search_relevant_passages,
            search_technical_passages,
        )

        for fn in (
            search_multimodal_passages,
            search_relevant_passages,
            search_technical_passages,
        ):
            assert "query_groups" not in inspect.signature(fn).parameters

    def test_retrieval_dun_groupe_absent(self):
        from app.services import space_search_service as sss

        assert not hasattr(sss, "_retrieve_one_group_hits")


class TestVaguenessSupprimee:
    def test_modele_de_clarification_absent(self):
        from app.services import query_schemas

        assert not hasattr(query_schemas, "ClarificationResult")

    def test_resultat_sans_champ_clarification(self):
        from app.services.lightweight_query_understanding import LightweightQueryResult

        assert "clarification" not in LightweightQueryResult.model_fields

    def test_prompt_sans_vagueness(self):
        from app.services.lightweight_query_understanding import (
            FUSED_UNDERSTANDING_EXTRA_PROMPT,
        )

        assert "too_vague" not in FUSED_UNDERSTANDING_EXTRA_PROMPT
        assert "clarification_question" not in FUSED_UNDERSTANDING_EXTRA_PROMPT


class TestSettingsSupprimes:
    @pytest.mark.parametrize(
        "flag",
        [
            "QUERY_MULTI_GROUP_ENABLED",
            "QUERY_VAGUENESS_CHECK_ENABLED",
            "QUERY_CONDENSE_ENABLED",
            "QUERY_FUSED_UNDERSTANDING_ENABLED",
            "QUERY_GENERATE_QUERIES_LLM",
        ],
    )
    def test_flag_absent(self, flag):
        from app.config import settings

        assert not hasattr(settings, flag), f"{flag} devrait avoir été supprimé"

    def test_coupe_circuit_et_timeout_conserves(self):
        from app.config import settings

        assert hasattr(settings, "QUERY_UNDERSTANDING_ENABLED")
        assert hasattr(settings, "QUERY_UNDERSTANDING_TIMEOUT_S")
