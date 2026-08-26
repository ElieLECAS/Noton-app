"""Mode expérimental 100 % PNG (CAG_IMAGE_ONLY, 2026-08-26).

Objectif du mode : mesurer ce que vaut la génération quand elle ne reçoit QUE les pages
en images, sans le texte extrait — qui, sur les planches CAO, est un sac de nombres
désancré et concurrence l'image que le modèle devrait croire.

Ces tests verrouillent les trois propriétés qui rendent le test valide :
  1. le texte des documents disparaît vraiment du contexte ;
  2. un manifeste subsiste (sinon le modèle reçoit des images anonymes et ne peut plus
     ni rattacher une cote à une gamme, ni produire son bloc <sources>) ;
  3. les images couvrent aussi le voisinage packé — c'est le budget libéré par le texte
     qui le paie, sans quoi on mesurerait « moins d'info » et non « image vs texte ».
"""
from unittest import mock

import pytest

from app.services import context_packer_service as cps


def _cag_documents():
    return [
        {
            "index": 1,
            "document_id": 10,
            "document_title": "Notice Perform 76",
            "pages": [4, 5, 6, 7],
            "matched_pages": [5],
            "seed_pages": [5],
            "full_document": False,
            "score": 0.9,
            "election_score": 0.9,
            "has_source_file": True,
        }
    ]


class TestManifeste:
    def test_manifeste_nomme_document_et_pages(self):
        out = cps._build_image_only_manifest(_cag_documents())
        assert "Notice Perform 76" in out
        assert "[Document 1]" in out
        assert "4, 5, 6, 7" in out

    def test_manifeste_signale_les_pages_retrouvees(self):
        """Le modèle doit savoir quelles pages la recherche a réellement matchées."""
        out = cps._build_image_only_manifest(_cag_documents())
        assert "retrouvées par la recherche" in out
        assert "5" in out

    def test_manifeste_dit_que_les_images_font_foi(self):
        out = cps._build_image_only_manifest(_cag_documents())
        assert "IMAGES" in out
        assert "font foi" in out

    def test_manifeste_garde_le_garde_fou_inter_gammes(self):
        """Le risque n°1 du CAG (attribuer une cote à la mauvaise gamme) ne disparaît pas
        parce qu'on passe en images."""
        out = cps._build_image_only_manifest(_cag_documents())
        assert "Perform 70" in out and "Perform 76" in out


class TestSelectionDesImages:
    def _run(self, image_only: bool):
        session = mock.MagicMock()
        doc = mock.MagicMock()
        doc.source_file_path = "/tmp/doc.pdf"
        session.get.return_value = doc

        passages = [
            {"document_id": 10, "page_no": 5, "score": 0.9, "needs_page_image": True},
        ]

        with mock.patch.object(cps.settings, "CAG_IMAGE_ONLY", image_only), mock.patch.object(
            cps.settings, "CAG_MAX_IMAGES", 10
        ), mock.patch(
            "app.services.multimodal_page_service.render_page_png_cached",
            return_value=b"PNG",
        ), mock.patch("os.path.exists", return_value=True):
            return cps.select_cag_images(session, _cag_documents(), passages)

    def test_mode_normal_nenvoie_que_les_pages_matchees(self):
        """Le texte du voisinage est déjà dans le contexte : son PNG n'apporterait rien."""
        _, captions = self._run(image_only=False)
        assert [c["page_no"] for c in captions] == [5]

    def test_mode_image_only_couvre_le_voisinage_packe(self):
        """Sans texte, les images ne complètent plus rien : elles SONT le contenu."""
        _, captions = self._run(image_only=True)
        pages = [c["page_no"] for c in captions]
        assert pages[0] == 5, "la page matchée reste prioritaire"
        assert set(pages) == {4, 5, 6, 7}

    def test_le_plafond_reste_respecte(self):
        session = mock.MagicMock()
        doc = mock.MagicMock()
        doc.source_file_path = "/tmp/doc.pdf"
        session.get.return_value = doc

        with mock.patch.object(cps.settings, "CAG_IMAGE_ONLY", True), mock.patch.object(
            cps.settings, "CAG_MAX_IMAGES", 2
        ), mock.patch(
            "app.services.multimodal_page_service.render_page_png_cached",
            return_value=b"PNG",
        ), mock.patch("os.path.exists", return_value=True):
            images, _ = cps.select_cag_images(session, _cag_documents(), [])

        assert len(images) == 2


class TestLimiteDureApi:
    """Mistral rejette toute requête de plus de 8 images (400, code 3051).

    Régression réelle du 26/08 : avec CAG_MAX_IMAGES=20, la requête a été rejetée, le
    repli de secours a répondu SANS document, et le modèle a inventé une liste de
    couleurs plausible. Le plafond doit donc être imposé par le code, pas par la config.
    """

    def test_plafond_api_impose_meme_si_le_setting_est_plus_haut(self):
        session = mock.MagicMock()
        doc = mock.MagicMock()
        doc.source_file_path = "/tmp/doc.pdf"
        session.get.return_value = doc

        docs = [
            {
                "index": 1,
                "document_id": 10,
                "document_title": "Notice",
                "pages": list(range(1, 21)),
                "matched_pages": [1],
                "seed_pages": [1],
                "full_document": False,
                "score": 0.9,
                "election_score": 0.9,
                "has_source_file": True,
            }
        ]

        with mock.patch.object(cps.settings, "CAG_IMAGE_ONLY", True), mock.patch.object(
            cps.settings, "CAG_MAX_IMAGES", 20
        ), mock.patch(
            "app.services.multimodal_page_service.render_page_png_cached",
            return_value=b"PNG",
        ), mock.patch("os.path.exists", return_value=True):
            images, _ = cps.select_cag_images(session, docs, [])

        assert len(images) == cps.MISTRAL_MAX_IMAGES_PER_REQUEST == 8


class TestReplíEco:
    def test_le_repli_retablit_le_texte(self):
        """Le repli de secours ne peut pas porter d'images : s'il gardait le mode
        image-only, il resterait un manifeste vide et le modèle inventerait."""
        from app.routers.chat import _forced_text_context_if_image_only

        with mock.patch.object(cps.settings, "CAG_IMAGE_ONLY", True):
            with _forced_text_context_if_image_only():
                assert cps.settings.CAG_IMAGE_ONLY is False
            # …et le mode est restauré à la sortie.
            assert cps.settings.CAG_IMAGE_ONLY is True

    def test_sans_image_only_le_repli_ne_touche_a_rien(self):
        from app.routers.chat import _forced_text_context_if_image_only

        with mock.patch.object(cps.settings, "CAG_IMAGE_ONLY", False):
            with _forced_text_context_if_image_only():
                assert cps.settings.CAG_IMAGE_ONLY is False


class TestDetectionModeleVision:
    """Régression réelle du 26/08 : `mistral-medium-latest` était absent de la liste
    blanche → aucune image envoyée, et en mode image-only aucun texte non plus. Le modèle
    a inventé une liste de couleurs entière en disant lui-même, dans son raisonnement,
    « je n'ai pas accès aux images réelles ». Le même piège avait déjà eu lieu avec
    « small » le 20/07 : une liste blanche pourrit et échoue SILENCIEUSEMENT."""

    @pytest.mark.parametrize(
        "model",
        [
            "mistral-medium-latest",
            "mistral-small-latest",
            "mistral-large-latest",
            "pixtral-12b-2409",
        ],
    )
    def test_modeles_multimodaux(self, model):
        from app.services.rag_generation_service import is_vision_model

        assert is_vision_model(model) is True

    @pytest.mark.parametrize(
        "model", ["open-mistral-7b", "open-mixtral-8x7b", "mistral-embed", ""]
    )
    def test_modeles_texte_seul(self, model):
        from app.services.rag_generation_service import is_vision_model

        assert is_vision_model(model) is False


class TestReasoningStructure:
    """Régression du 26/08 : `reasoning_effort=high` était envoyé à TOUS les modèles.
    mistral-medium ne sépare pas sa réflexion du texte → son monologue interne s'est
    affiché dans la réponse, collé au tableau final, sans séparateur récupérable."""

    @pytest.mark.parametrize("model", ["mistral-small-latest", "magistral-medium-latest"])
    def test_modeles_a_reflexion_separee(self, model):
        from app.services.rag_generation_service import supports_structured_reasoning

        assert supports_structured_reasoning(model) is True

    @pytest.mark.parametrize(
        "model", ["mistral-medium-latest", "mistral-large-latest", "open-mistral-7b", ""]
    )
    def test_modeles_sans_reflexion_separee(self, model):
        from app.services.rag_generation_service import supports_structured_reasoning

        assert supports_structured_reasoning(model) is False


class TestDpiConfigurable:
    def test_dpi_vient_du_setting(self):
        """Le DPI était codé en dur à 150 alors que l'extraction lit à 300 : le modèle qui
        répond voyait la page moins bien que celui qui l'a transcrite."""
        session = mock.MagicMock()
        doc = mock.MagicMock()
        doc.source_file_path = "/tmp/doc.pdf"
        session.get.return_value = doc

        with mock.patch.object(cps.settings, "CAG_IMAGE_DPI", 220), mock.patch.object(
            cps.settings, "CAG_IMAGE_ONLY", False
        ), mock.patch(
            "app.services.multimodal_page_service.render_page_png_cached",
            return_value=b"PNG",
        ) as render, mock.patch("os.path.exists", return_value=True):
            cps.select_cag_images(
                session,
                _cag_documents(),
                [{"document_id": 10, "page_no": 5, "score": 0.9}],
            )

        assert render.call_args.kwargs["dpi"] == 220


class TestContexteSansTexte:
    @pytest.mark.asyncio
    async def test_le_texte_des_documents_disparait(self, db_session):
        """Garde-fou central : en mode image-only, aucun bloc de texte documentaire ne
        doit subsister dans le message système."""
        passages = [
            {
                "document_id": 999,
                "document_title": "Doc absent",
                "page_no": 1,
                "score": 0.8,
            }
        ]
        with mock.patch.object(cps.settings, "CAG_IMAGE_ONLY", True):
            out = cps.build_cag_context(
                db_session, passages, system_prompt="PROMPT", emit_sources_tag=False
            )
        assert "DOCUMENTS (contexte complet)" not in out["content"]
