"""Les trois requêtes SQL de BM25 doivent être VALIDES, pas seulement sans exception.

``retrieve_bm25_pages`` attrape toute exception et retourne une liste vide en journalisant
un avertissement. Une requête syntaxiquement fausse rend donc « aucun résultat » au lieu de
planter : le retrieval tombe silencieusement sur ColPali seul, et rien ne le signale.

C'est exactement ce qui s'est produit le 2026-09-16 — le retrait de la colonne
``enrichment_source_pages_text`` a laissé une virgule orpheline avant ``FROM`` dans les trois
requêtes, et BM25 a rendu zéro page sur tout le corpus pendant plusieurs heures.

Ces tests appellent les helpers INTERNES, qui eux propagent : ils touchent le vrai Postgres
avec un document inexistant, donc ils valident la syntaxe sans dépendre d'aucune donnée.
"""
import pytest
from sqlmodel import Session

from app.database import engine
from app.services.page_retrieval_service import (
    _run_bm25_or_pages_query,
    _run_bm25_pages_query,
    _run_bm25_websearch_or_pages_query,
    retrieve_bm25_pages,
)

DOC_INEXISTANT = [-424242]


@pytest.fixture()
def session():
    with Session(engine) as s:
        yield s


class TestSyntaxeDesRequetes:
    def test_requete_and_stricte(self, session):
        assert _run_bm25_pages_query(session, DOC_INEXISTANT, "vitrage", 5) == []

    def test_requete_or_tsquery(self, session):
        assert _run_bm25_or_pages_query(session, DOC_INEXISTANT, "vitrage | parclose", 5) == []

    def test_requete_or_websearch(self, session):
        assert (
            _run_bm25_websearch_or_pages_query(session, DOC_INEXISTANT, "vitrage OR parclose", 5)
            == []
        )


class TestPointDEntree:
    def test_aucune_erreur_journalisee_sur_une_recherche_normale(self, session, caplog):
        """Un avertissement « recherche échouée » signale une requête invalide, pas une
        absence de résultat : il ne doit jamais apparaître sur un appel bien formé."""
        with caplog.at_level("WARNING", logger="app.services.page_retrieval_service"):
            retrieve_bm25_pages(session, DOC_INEXISTANT, "vitrage parclose", 5)
        assert "recherche échouée" not in caplog.text
