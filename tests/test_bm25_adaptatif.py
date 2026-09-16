"""BM25 adaptatif : ancre rare et pondération par rareté.

Deux propriétés du classement lexical de PostgreSQL motivent cette couche, mesurées le
2026-09-16 sur l'espace ROTO :

  * le palier AND strict exige TOUS les termes sur une même page — sur une question en
    langue naturelle il ne matchait 0 fois sur 5 ;
  * ``ts_rank_cd`` ne pondère pas par la rareté : « à quoi correspond la référence
    820255 » sortait la bonne page en 4e position, derrière trois pages qui ne matchaient
    que le mot « correspond ».

Après la couche : 4 pages attendues sur 5 en 1re position au lieu d'une.
"""
import math

import pytest
from sqlmodel import Session

from app.database import engine
from app.services.page_retrieval_service import (
    _bm25_document_frequency,
    _bm25_idf,
    _bm25_page_universe,
    _bm25_rare_terms,
)


@pytest.fixture()
def session():
    with Session(engine) as s:
        yield s


class TestFrequenceDocumentaire:
    def test_un_terme_absent_vaut_zero_et_non_un(self, session):
        """Le piège du LEFT JOIN : ``count(DISTINCT (a, b))`` compte le tuple (NULL, NULL).

        Sans le FILTER, un terme absent du corpus obtient df = 1, passe donc pour le plus
        rare de tous, et devient l'ancre de recherche — l'inverse de ce qu'on veut.
        """
        dfs = _bm25_document_frequency(session, [-424242], ["motQuiNExistePasXYZ"])
        assert dfs.get("motQuiNExistePasXYZ") == 0

    def test_perimetre_vide_ne_leve_pas(self, session):
        assert _bm25_document_frequency(session, [-424242], []) == {}
        assert _bm25_page_universe(session, [-424242]) == 0


class TestSelectionDeLAncre:
    def test_un_terme_absent_n_est_jamais_une_ancre(self):
        assert _bm25_rare_terms({"absent": 0, "rare": 2}, 500) == ["rare"]

    def test_les_termes_sont_ordonnes_du_plus_rare_au_moins_rare(self):
        assert _bm25_rare_terms({"a": 7, "b": 1, "c": 4}, 500) == ["b", "c", "a"]

    def test_le_plafond_est_absolu_et_non_proportionnel(self):
        """Le plafond vaut 8 pages au maximum, même sur un très grand périmètre : une
        ancre doit être rare en valeur absolue, pas seulement en proportion."""
        assert _bm25_rare_terms({"neuf_pages": 9}, 5000) == []
        assert _bm25_rare_terms({"huit_pages": 8}, 5000) == ["huit_pages"]

    def test_sur_un_petit_perimetre_le_plafond_se_resserre(self):
        """30 pages : 10 % = 3. Un terme sur 5 pages n'y discrimine plus assez."""
        assert _bm25_rare_terms({"trois": 3, "cinq": 5}, 30) == ["trois"]

    def test_un_terme_trop_courant_n_est_pas_une_ancre(self):
        """Au-delà du plafond, le terme ne discrimine plus rien."""
        assert _bm25_rare_terms({"courant": 400}, 500) == []

    def test_sans_frequence_aucune_ancre(self):
        assert _bm25_rare_terms({}, 500) == []


class TestPonderationParRarete:
    def test_un_terme_unique_pese_plus_qu_un_terme_courant(self):
        assert _bm25_idf(1, 600) > _bm25_idf(300, 600)

    def test_un_terme_absent_ne_pese_rien(self):
        assert _bm25_idf(0, 600) == 0.0

    def test_perimetre_inconnu_ne_pese_rien(self):
        assert _bm25_idf(3, 0) == 0.0

    def test_valeur_conforme_a_la_formule(self):
        assert _bm25_idf(2, 100) == pytest.approx(math.log(51.0))


class TestCorpusLexical:
    """Le markdown augmenté entre dans le corpus BM25, à côté du texte extrait."""

    def test_le_filtre_couvre_les_deux_natures(self):
        from app.services.page_retrieval_service import _retrievable_text_leaf_filter

        f = _retrievable_text_leaf_filter("dc")
        assert "semantic_leaf" in f and "page_markdown" in f

    def test_aucune_reecriture_de_llm_dans_le_corpus(self):
        """Les chunks contextuels ont été supprimés le 2026-09-16 : ils ne doivent pas
        revenir par cette porte."""
        from app.services.page_retrieval_service import _retrievable_text_leaf_filter

        assert "contextual_enrichment" not in _retrievable_text_leaf_filter("dc")

    def test_une_ligne_par_page_et_non_par_fragment(self):
        """La granularité du corpus lexical suit celle de la matière lue."""
        from app.services.page_markdown_service import (
            CHUNKING_VERSION_PAGE_MARKDOWN,
            CONTENT_TYPE_PAGE_MARKDOWN,
        )

        assert CONTENT_TYPE_PAGE_MARKDOWN == "page_markdown"
        assert CHUNKING_VERSION_PAGE_MARKDOWN == "page_markdown_v1"

    def test_un_retraitement_resynchronise_le_corpus(self):
        """Les purges de chunks emportent les lignes de markdown, qui viennent du FICHIER :
        l'indexation doit les reconstruire, sinon le document sort du corpus en silence."""
        import inspect

        from app.services import document_indexing_service as dis

        source = inspect.getsource(dis.process_document_indexing)
        assert "sync_chunks" in source and "has_markdown" in source

    def test_le_detachement_retire_les_lignes(self):
        import inspect

        from app.routers import library

        assert "delete_chunks" in inspect.getsource(library.delete_page_markdown)

    def test_l_import_indexe_les_pages(self):
        import inspect

        assert "sync_chunks" in inspect.getsource(
            __import__("app.routers.library", fromlist=["x"]).import_page_markdown
        )
