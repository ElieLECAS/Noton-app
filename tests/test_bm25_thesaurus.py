"""Thésaurus métier BM25 (P1, 2026-08-26).

BM25 étant le seul canal texte depuis le retrait du dense, le thésaurus est la seule
mitigation du décalage de vocabulaire. Ces tests verrouillent les deux garde-fous qui
l'empêchent de détruire la précision : le plafond après expansion, et la non-expansion
des références produit — plus l'invariant central : jamais d'expansion sur le AND strict.
"""
from unittest import mock

import pytest

from app.services.bm25_thesaurus import (
    expand_term,
    expand_terms,
    is_discriminant,
)


class TestExpansionUnitaire:
    def test_synonyme_metier_connu(self):
        assert "vantail" in expand_term("ouvrant")
        assert "ouvrant" in expand_term("vantail")

    def test_terme_dorigine_exclu(self):
        """Le terme est déjà dans la requête : on ne renvoie que ce qu'il ajoute."""
        assert "ouvrant" not in expand_term("ouvrant")

    def test_insensible_aux_accents_et_a_la_casse(self):
        """Les notices écrivent « béquille », les utilisateurs tapent « bequille »."""
        assert expand_term("bequille") == expand_term("Béquille")
        assert "poignée" in expand_term("BEQUILLE") or "poignee" in expand_term("BEQUILLE")

    def test_sigles_metier(self):
        """OF / OB sont omniprésents dans les notices et jamais écrits en toutes lettres."""
        assert any("française" in s or "francaise" in s for s in expand_term("of"))
        assert any("oscillo" in s for s in expand_term("ob"))

    def test_terme_inconnu_ne_produit_rien(self):
        assert expand_term("zorglub") == []
        assert expand_term("") == []


class TestReferencesProduit:
    def test_reference_est_discriminante(self):
        assert is_discriminant("TGY3702")
        assert is_discriminant("76100")

    def test_mot_metier_nest_pas_discriminant(self):
        assert not is_discriminant("ouvrant")

    def test_reference_jamais_etendue(self):
        """« TGY3702 » n'a pas de synonyme : le diluer ferait perdre au canal lexical sa
        seule vraie force, la correspondance exacte sur les codes."""
        out = expand_terms(["TGY3702"])
        assert out == ["TGY3702"]


class TestPlafond:
    def test_plafond_porte_sur_le_total_apres_expansion(self):
        termes = ["ouvrant", "dormant", "joint", "pose", "ferrure", "reglage"]
        out = expand_terms(termes, max_total=8)
        assert len(out) == 8

    def test_termes_dorigine_jamais_sacrifies(self):
        """La troncature coupe dans les synonymes, jamais dans ce que l'utilisateur a écrit."""
        termes = ["ouvrant", "dormant", "joint", "pose", "ferrure"]
        out = expand_terms(termes, max_total=5)
        assert out == termes

    def test_pas_de_doublon(self):
        out = expand_terms(["ouvrant", "vantail"])
        normalises = [t.lower() for t in out]
        assert len(normalises) == len(set(normalises))


class TestElision:
    """L'élision est la règle en français métier (« réglage de l'ouvrant »)."""

    def test_article_elide_ignore_a_la_correspondance(self):
        assert "vantail" in expand_term("l'ouvrant")
        assert "vantail" in expand_term("d'ouvrant")

    def test_apostrophe_typographique(self):
        assert "vantail" in expand_term("l’ouvrant")

    def test_extraction_retire_larticle_elide(self):
        """Sinon `l'ouvrant` finit nettoyé en `louvrant`, lexème inexistant : le palier
        de dernier recours perdait silencieusement tous les mots élidés."""
        from app.services.page_retrieval_service import (
            _build_bm25_or_tsquery,
            _extract_bm25_fallback_query,
        )

        out = _extract_bm25_fallback_query("réglage de l'ouvrant")
        assert "ouvrant" in out
        terms = [t.strip() for t in out.replace(" OR ", "|").split("|") if t.strip()]
        assert "louvrant" not in _build_bm25_or_tsquery(terms)


class TestPaliersDeRepli:
    """Invariant central : le AND strict n'est JAMAIS étendu."""

    def test_and_strict_non_etendu_par_defaut(self):
        from app.services.page_retrieval_service import _extract_bm25_fallback_query

        sans = _extract_bm25_fallback_query("réglage de l'ouvrant")
        avec = _extract_bm25_fallback_query("réglage de l'ouvrant", expand=True)
        assert "vantail" not in sans
        assert "vantail" in avec

    def test_or_tsquery_etendu_sur_demande(self):
        from app.services.page_retrieval_service import _build_bm25_or_tsquery

        sans = _build_bm25_or_tsquery(["ouvrant"])
        avec = _build_bm25_or_tsquery(["ouvrant"], expand=True)
        assert sans == "ouvrant"
        assert "vantail" in avec

    def test_expansion_respecte_le_setting_de_plafond(self):
        from app.services import page_retrieval_service as prs

        with mock.patch.object(prs.settings, "BM25_EXPANSION_MAX_TERMS", 3):
            out = prs._expand_bm25_terms(["ouvrant", "dormant", "joint"])
        assert len(out) == 3
