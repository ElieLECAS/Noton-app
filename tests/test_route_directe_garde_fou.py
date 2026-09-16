"""La route « direct » ne doit jamais avaler une vraie question technique.

Incident du 2026-09-16 : « Sur le passage de câble, quelle couleur de fil arrive sur la
broche 4 et à quoi sert-elle ? » a été classée ``route=direct`` par la compréhension légère.
Le tour n'a fait aucun retrieval et le modèle a répondu de mémoire — fil bleu, neutre,
moteur de volet roulant — en concluant par une ligne « Source : schémas de câblage des
notices Roto/Askey » entièrement fabriquée.

La même question, reposée, est repartie en ``rag``. La classification est donc NON
DÉTERMINISTE : il faut un garde-fou déterministe, pas un meilleur prompt.

Ces tests portent sur la LOGIQUE du garde-fou, jamais sur le contenu de la base : la suite
tourne contre une base de test vide, et un test qui dépendrait des documents de dev
passerait ici par accident et échouerait en intégration continue.
"""
from unittest import mock

import pytest
from sqlmodel import Session

from app.database import engine
from app.routers.chat import _question_documentaire

ESPACE = 24


@pytest.fixture()
def session():
    with Session(engine) as s:
        yield s


class TestSignauxSansBase:
    """Référence et mesure se lisent dans le message : aucun document requis."""

    @pytest.mark.parametrize(
        "message",
        ["À quoi correspond la référence 820255 ?", "et le 817028 ?", "le profil 76180 convient ?"],
    )
    def test_une_reference_renvoie_aux_documents(self, session, message):
        assert "référence" in (_question_documentaire(session, ESPACE, message) or "")

    @pytest.mark.parametrize(
        "message",
        ["Quelle tige de 2,5 mm utiliser ?", "Il faut du 24 V ou du 12 V ?", "une charge de 40 mA"],
    )
    def test_une_mesure_renvoie_aux_documents(self, session, message):
        assert _question_documentaire(session, ESPACE, message) == "mesure chiffrée"

    def test_la_reference_tranche_avant_toute_requete(self, session):
        """Le signal le moins cher doit trancher sans toucher la base."""
        with mock.patch(
            "app.services.page_retrieval_service.get_space_document_ids",
            side_effect=AssertionError("la base ne doit pas être interrogée"),
        ):
            assert _question_documentaire(session, ESPACE, "et le 820255 ?")


class TestSignalDuCorpus:
    """Le troisième signal : les mots de la question vivent dans les documents de l'espace."""

    def _avec_rares(self, rares):
        return (
            mock.patch(
                "app.services.page_retrieval_service.get_space_document_ids", return_value=[1, 2]
            ),
            mock.patch(
                "app.services.page_retrieval_service._bm25_document_frequency", return_value={}
            ),
            mock.patch("app.services.page_retrieval_service._bm25_page_universe", return_value=100),
            mock.patch("app.services.page_retrieval_service._bm25_rare_terms", return_value=rares),
        )

    def test_deux_termes_rares_suffisent(self, session):
        a, b, c, d = self._avec_rares(["broche", "couleur"])
        with a, b, c, d:
            r = _question_documentaire(session, ESPACE, "quelle couleur de fil sur la broche 4")
        assert r and "termes documentaires" in r

    def test_un_seul_terme_rare_ne_suffit_pas(self, session):
        """Un mot rare isolé peut être un hasard de vocabulaire : on n'aiguille pas dessus."""
        a, b, c, d = self._avec_rares(["broche"])
        with a, b, c, d:
            assert _question_documentaire(session, ESPACE, "parle-moi de la broche") is None

    def test_un_espace_sans_document_ne_declenche_rien(self, session):
        with mock.patch(
            "app.services.page_retrieval_service.get_space_document_ids", return_value=[]
        ):
            assert _question_documentaire(session, ESPACE, "une question quelconque ici") is None


class TestBavardage:
    @pytest.mark.parametrize(
        "message",
        ["Bonjour !", "Merci beaucoup, bonne journée", "Qui es-tu et que sais-tu faire ?", "", "   "],
    )
    def test_reste_en_direct(self, session, message):
        assert _question_documentaire(session, ESPACE, message) is None


class TestRobustesse:
    def test_une_panne_du_controle_ne_casse_pas_le_tour(self, session):
        """Le garde-fou est un filet : s'il tombe, le tour continue sans lui."""
        with mock.patch(
            "app.services.page_retrieval_service.get_space_document_ids",
            side_effect=RuntimeError("base indisponible"),
        ):
            assert _question_documentaire(session, ESPACE, "une question sans code ni mesure") is None


class TestPromptSansDocument:
    def test_la_branche_directe_interdit_les_faits_techniques(self):
        import inspect

        from app.routers import chat

        source = inspect.getsource(chat.stream_space_chat_message)
        assert "AUCUN DOCUMENT POUR CE MESSAGE" in source
        # Ce sont les deux interdits qui ont manqué le 16/09.
        assert "référence" in source and "Source : …" in source

    def test_le_garde_fou_precede_la_branche_directe(self):
        import inspect

        from app.routers import chat

        source = inspect.getsource(chat.stream_space_chat_message)
        assert source.index("_question_documentaire(") < source.index("direct_context = list(")
