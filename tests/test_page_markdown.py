"""Markdown augmenté : parsing déterministe, validation bloquante, substitution au packer.

Protocole : ``docs/protocole_markdown_augmente_2026-09-16.md``.

La règle que ces tests protègent avant tout : **un markdown dont le nombre de sections ne
correspond pas au PDF est refusé**. Un décalage d'une page fausse l'appariement page ↔ image
pour toutes les pages suivantes, et il est silencieux.
"""
from unittest import mock

import pytest

from app.services.page_markdown_service import (
    ParsedMarkdown,
    conventions_block,
    load_page_records,
    parse_markdown,
    validate,
)


def _markdown(pages, *, conventions=True, frontmatter=True, printed=True):
    parts = []
    if frontmatter:
        parts.append("---\ndocument_id: 1\ntitre: Doc\nsource_sha256: abc123\n---\n")
    if conventions:
        parts.append("## Conventions du document\n\n- Cotation en bleu = épaisseur vitrage.\n")
    for n in pages:
        sep = (
            f'<!-- PageBreak page_pdf="{n}" page_imprimee="{n}" -->'
            if printed
            else f'<!-- PageBreak page_pdf="{n}" -->'
        )
        parts.append(f"{sep}\n## Page {n} — Titre {n}\n\nContenu de la page {n}.\n")
    return "\n".join(parts)


class TestParsing:
    def test_pages_et_titres(self):
        p = parse_markdown(_markdown([1, 2, 3]))
        assert p.page_numbers == [1, 2, 3]
        assert p.pages[0].title == "Page 1 — Titre 1"
        assert "Contenu de la page 1." in p.pages[0].body

    def test_page_imprimee_facultative(self):
        p = parse_markdown(_markdown([1, 2], printed=False))
        assert [x.page_imprimee for x in p.pages] == [None, None]

    def test_page_imprimee_lue_quand_presente(self):
        md = '<!-- PageBreak page_pdf="8" page_imprimee="5" -->\n## Page 5\n\ntexte\n'
        p = parse_markdown(md)
        assert p.pages[0].page_pdf == 8
        assert p.pages[0].page_imprimee == "5"

    def test_frontmatter_et_conventions(self):
        p = parse_markdown(_markdown([1]))
        assert p.frontmatter["source_sha256"] == "abc123"
        assert "épaisseur vitrage" in p.conventions

    def test_frontmatter_illisible_non_bloquant(self):
        md = "---\n: : : pas du yaml : [\n---\n" + _markdown([1], frontmatter=False)
        p = parse_markdown(md)
        assert p.page_numbers == [1]

    def test_sans_frontmatter_ni_conventions(self):
        p = parse_markdown(_markdown([1, 2], conventions=False, frontmatter=False))
        assert p.page_numbers == [1, 2]
        assert p.conventions == ""

    def test_document_vide(self):
        p = parse_markdown("")
        assert p.pages == []

    def test_les_conventions_ne_sont_pas_une_page(self):
        """La section Conventions précède le premier séparateur : jamais comptée comme page."""
        p = parse_markdown(_markdown([1]))
        assert len(p.pages) == 1


class TestValidation:
    def _validate(self, md, expected, sha=None):
        return validate(md, parse_markdown(md), expected_pages=expected, source_sha256=sha)

    def test_markdown_aligne_accepte(self):
        r = self._validate(_markdown([1, 2, 3]), 3)
        assert r.ok is True
        assert r.errors == []

    def test_compte_different_refuse(self):
        r = self._validate(_markdown([1, 2, 3]), 26)
        assert r.ok is False
        assert any("26 page(s)" in e for e in r.errors)

    def test_page_manquante_refusee(self):
        r = self._validate(_markdown([1, 3]), 3)
        assert r.ok is False
        assert any("absente" in e for e in r.errors)

    def test_doublon_refuse(self):
        r = self._validate(_markdown([1, 2, 2]), 3)
        assert r.ok is False
        assert any("double" in e for e in r.errors)

    def test_page_hors_bornes_refusee(self):
        r = self._validate(_markdown([1, 2, 99]), 3)
        assert r.ok is False
        assert any("hors du PDF" in e for e in r.errors)

    def test_separateur_nu_refuse(self):
        """« <!-- PageBreak --> » ne survit ni à une troncature ni à une concaténation."""
        md = "<!-- PageBreak -->\n## Page\n\ntexte\n"
        r = self._validate(md, 1)
        assert r.ok is False
        assert any("sans numéro" in e for e in r.errors)

    def test_desordre_avertit_sans_bloquer(self):
        r = self._validate(_markdown([2, 1, 3]), 3)
        assert r.ok is True
        assert any("ordre croissant" in w for w in r.warnings)

    def test_page_vide_avertit_sans_bloquer(self):
        md = _markdown([1]) + '\n<!-- PageBreak page_pdf="2" -->\n'
        r = self._validate(md, 2)
        assert r.ok is True
        assert r.pages_vides == [2]

    def test_conventions_absentes_avertissent(self):
        r = self._validate(_markdown([1], conventions=False), 1)
        assert r.ok is True
        assert any("Conventions" in w for w in r.warnings)

    def test_hachage_different_avertit_sans_bloquer(self):
        """Le PDF a changé depuis la rédaction : signalé, mais l'import reste possible."""
        r = self._validate(_markdown([1]), 1, sha="autre_hash")
        assert r.ok is True
        assert r.hash_source_ok is False
        assert any("AUTRE version" in w for w in r.warnings)

    def test_hachage_identique_valide(self):
        r = self._validate(_markdown([1]), 1, sha="abc123")
        assert r.hash_source_ok is True

    def test_sans_pdf_le_comptage_ne_bloque_pas(self):
        """PDF introuvable : on ne peut pas compter, on n'invente pas une erreur."""
        r = self._validate(_markdown([1, 2]), None)
        assert r.ok is True

    def test_apercu_liste_les_pages(self):
        r = self._validate(_markdown([1, 2]), 2)
        assert [a["page_pdf"] for a in r.apercu] == [1, 2]
        assert all(a["caracteres"] > 0 for a in r.apercu)


class TestSubstitutionAuPacker:
    """Le packer lit le markdown quand il existe, les fragments sinon."""

    def test_une_page_un_enregistrement(self, tmp_path):
        parsed = parse_markdown(_markdown([1, 2, 3]))
        with mock.patch(
            "app.services.page_markdown_service.load_parsed", return_value=parsed
        ):
            records = load_page_records(1)
        assert [r[0] for r in records] == [1, 2, 3]
        assert all(r[1] == 0 for r in records)

    def test_pages_vides_ecartees(self):
        parsed = parse_markdown(_markdown([1]) + '\n<!-- PageBreak page_pdf="2" -->\n')
        with mock.patch(
            "app.services.page_markdown_service.load_parsed", return_value=parsed
        ):
            records = load_page_records(1)
        assert [r[0] for r in records] == [1]

    def test_absence_de_markdown_rend_none(self):
        with mock.patch("app.services.page_markdown_service.load_parsed", return_value=None):
            assert load_page_records(1) is None

    def test_markdown_sans_page_rend_none(self):
        with mock.patch(
            "app.services.page_markdown_service.load_parsed", return_value=ParsedMarkdown()
        ):
            assert load_page_records(1) is None

    def test_conventions_servies_au_bloc(self):
        parsed = parse_markdown(_markdown([1]))
        with mock.patch(
            "app.services.page_markdown_service.load_parsed", return_value=parsed
        ):
            block = conventions_block(1)
        assert "épaisseur vitrage" in block

    def test_le_packer_prefere_le_markdown(self):
        from app.services.context_packer_service import _load_page_records

        with mock.patch(
            "app.services.page_markdown_service.load_page_records",
            return_value=[(1, 0, "page markdown")],
        ), mock.patch(
            "app.services.context_packer_service._load_leaf_records",
            return_value=[(1, 0, "fragment")],
        ) as leaves:
            records = _load_page_records(mock.MagicMock(), 1)
        assert records == [(1, 0, "page markdown")]
        leaves.assert_not_called()

    def test_le_packer_retombe_sur_les_fragments(self):
        from app.services.context_packer_service import _load_page_records

        with mock.patch(
            "app.services.page_markdown_service.load_page_records", return_value=None
        ), mock.patch(
            "app.services.context_packer_service._load_leaf_records",
            return_value=[(1, 0, "fragment")],
        ):
            records = _load_page_records(mock.MagicMock(), 1)
        assert records == [(1, 0, "fragment")]

    def test_markdown_illisible_ne_casse_pas_le_tour(self):
        """Une exception à la lecture doit dégrader vers les fragments, pas lever."""
        from app.services.context_packer_service import _load_page_records

        with mock.patch(
            "app.services.page_markdown_service.load_page_records",
            side_effect=RuntimeError("fichier corrompu"),
        ), mock.patch(
            "app.services.context_packer_service._load_leaf_records",
            return_value=[(1, 0, "fragment")],
        ):
            records = _load_page_records(mock.MagicMock(), 1)
        assert records == [(1, 0, "fragment")]


class TestProfilDocumentInchange:
    def test_profile_document_juge_le_texte_reellement_extrait(self):
        """``profile_document`` doit continuer de lire les FRAGMENTS : sa détection
        « document muet » porte sur ce que le PDF contient, pas sur une transcription
        écrite à la main."""
        import inspect

        from app.services import context_packer_service as cps

        source = inspect.getsource(cps.profile_document)
        assert "_load_leaf_records(session, document_id)" in source
        assert "_load_page_records" not in source


class TestCycleDeVie:
    """Le markdown est nommé par l'ID du document : il doit disparaître avec lui."""

    def test_suppression_du_document_supprime_le_markdown(self):
        import inspect

        from app.services import document_service_new as dsn

        source = inspect.getsource(dsn.delete_document)
        assert "delete_markdown" in source, (
            "Un markdown laissé derrière un document supprimé se rattacherait "
            "silencieusement à un futur document de même identifiant."
        )

    def test_le_retraitement_ne_touche_pas_au_FICHIER_markdown(self):
        """Retraiter le PDF régénère les fragments ; la transcription écrite à la main
        survit et garde la priorité — sinon chaque retraitement effacerait un travail humain.

        Ses LIGNES BM25, elles, sont emportées par la purge de chunks : l'indexation les
        reconstruit depuis le fichier (cf. TestCorpusLexical)."""
        import inspect

        from app.services import document_indexing_service as dis

        source = inspect.getsource(dis)
        for destructeur in ("write_markdown", "delete_markdown", "PAGE_MARKDOWN_DIR"):
            assert destructeur not in source, (
                f"l'indexation ne doit jamais toucher au fichier markdown ({destructeur})"
            )


class TestPostRetrieverTexteEtImages:
    """Le tour de chat envoie le TEXTE des pages (markdown prioritaire) ET leurs PNG.

    Régression du 2026-09-16 : la voie « image only » du 14/09 packait un manifeste sans
    aucun texte de page, ce qui rendait le markdown augmenté inerte.
    """

    def test_le_routeur_packe_avec_build_cag_context(self):
        import inspect

        from app.routers import chat

        source = inspect.getsource(chat.stream_space_chat_message)
        assert "build_cag_context(" in source
        assert "build_image_only_context" not in source

    def test_le_packer_image_only_a_ete_supprime(self):
        from app.services import context_packer_service as cps

        assert not hasattr(cps, "build_image_only_context")

    def test_le_prompt_annonce_le_texte_et_les_images(self):
        from app.routers.chat import PAGES_SYSTEM_PROMPT as p

        assert "le TEXTE de ses pages" in p
        assert "IMAGES" in p
        assert "L'IMAGE FAIT FOI" in p
        # Garde-fous mesurés : lecture de planche, ligne de tableau, nomenclature.
        assert "légende" in p and "LIGNE qui encadre" in p
        assert "ne totalise JAMAIS" in p

    def test_le_prompt_interdit_les_numeros_de_page_dans_le_corps(self):
        """Mesuré le 14/09 : page citée 43,5 → 83,9 %, page décalée 53 → 9,7 %."""
        from app.routers.chat import PAGES_SYSTEM_PROMPT as p

        assert "N'écris JAMAIS" in p and "un numéro de page" in p

    def test_l_ancien_prompt_image_only_a_disparu(self):
        from app.routers import chat

        assert not hasattr(chat, "IMAGE_ONLY_SYSTEM_PROMPT")


class TestAffichageDansLesVues:
    """Modales de consultation : la page retranscrite s'affiche à la place des fragments."""

    def test_page_section_rend_le_corps_de_la_page(self):
        parsed = parse_markdown(_markdown([1, 2, 3]))
        with mock.patch(
            "app.services.page_markdown_service.load_parsed", return_value=parsed
        ):
            from app.services.page_markdown_service import page_section

            assert "Contenu de la page 2." in page_section(1, 2)
            assert page_section(1, 99) is None

    def test_sans_markdown_la_vue_retombe_sur_les_fragments(self):
        from app.services.lexical_search_service import _page_display_text

        with mock.patch(
            "app.services.page_markdown_service.page_section", return_value=None
        ), mock.patch(
            "app.services.page_retrieval_service.build_consolidated_page_text",
            return_value="fragments indexés",
        ):
            texte, augmente = _page_display_text(1, 1, [])
        assert texte == "fragments indexés"
        assert augmente is False

    def test_avec_markdown_la_vue_sert_la_page(self):
        from app.services.lexical_search_service import _page_display_text

        with mock.patch(
            "app.services.page_markdown_service.page_section", return_value="## Page 1\n\ntexte"
        ), mock.patch(
            "app.services.page_retrieval_service.build_consolidated_page_text",
            return_value="fragments indexés",
        ):
            texte, augmente = _page_display_text(1, 1, [])
        assert texte.startswith("## Page 1")
        assert augmente is True

    def test_une_lecture_en_echec_ne_casse_pas_l_affichage(self):
        from app.services.lexical_search_service import _page_display_text

        with mock.patch(
            "app.services.page_markdown_service.page_section",
            side_effect=RuntimeError("fichier corrompu"),
        ), mock.patch(
            "app.services.page_retrieval_service.build_consolidated_page_text",
            return_value="fragments indexés",
        ):
            texte, augmente = _page_display_text(1, 1, [])
        assert texte == "fragments indexés" and augmente is False

    def test_les_quatre_schemas_exposent_le_drapeau(self):
        from app.routers.library import (
            DocumentCategoryPageDetailResponse,
            DocumentSearchPageDetailResponse,
        )
        from app.routers.spaces import (
            SpaceCategoryPageDetailResponse,
            SpaceSearchPageDetailResponse,
        )

        for modele in (
            DocumentCategoryPageDetailResponse,
            DocumentSearchPageDetailResponse,
            SpaceCategoryPageDetailResponse,
            SpaceSearchPageDetailResponse,
        ):
            assert "markdown_augmente" in modele.model_fields


class TestCitationDeLaPage:
    """Le bloc <sources> doit nommer les pages : c'est lui qui ouvre le PDF au bon endroit.

    Régression du 2026-09-16 : le prompt interdit d'écrire un numéro de page dans le corps
    de la réponse, et la consigne du bloc machine offrait une sortie (« liste vide si
    aucun »). Le modèle résolvait le conflit en n'émettant plus aucune page — 6 réponses
    sur 10 affichaient « (complet) » au lieu de la page.
    """

    def test_le_prompt_distingue_le_corps_de_la_ligne_machine(self):
        from app.routers.chat import PAGES_SYSTEM_PROMPT as p

        assert "porte sur le CORPS de la réponse" in p
        assert "DOIT nommer les pages" in p

    def test_la_consigne_sources_n_offre_plus_de_sortie_facile(self):
        from unittest.mock import MagicMock

        from app.services.context_packer_service import build_cag_context

        session = MagicMock()
        session.exec.return_value.all.return_value = []
        ctx = build_cag_context(session, [{"document_id": 1, "page_no": 1, "score": 1.0}],
                                system_prompt="X")
        contenu = ctx["content"]
        assert "liste vide si aucun" not in contenu
        assert "ouvre le PDF à la bonne page" in contenu

    def test_l_etiquette_suit_la_page_d_atterrissage(self):
        """L'étiquette de source doit nommer la page où le bouton MÈNE, jamais « complet »
        quand une page d'atterrissage existe."""
        import io

        gabarit = io.open("app/templates/space_detail.html", encoding="utf-8").read()
        i = gabarit.index("Source par DOCUMENT (mode CAG)")
        bloc = gabarit[i : i + 900]
        # la page d'atterrissage est testée AVANT le repli « complet »
        assert bloc.index("source.page_no") < bloc.index('" (complet)"')
