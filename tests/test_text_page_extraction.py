"""
Voie d'extraction « texte natif » (pymupdf4llm) — text_page_extraction_service.

Couvre ce qui distingue cette voie de la voie vision :
  - un tableau markdown devient des chunks-lignes autosuffisants (en-têtes réinjectés),
    jamais une coupe aveugle au token ;
  - les références et décimales traversent la chaîne intactes ;
  - une page sans couche texte ne produit rien (l'appelant bascule en vision).
"""
from unittest.mock import patch

import pytest

from app.services.text_page_extraction_service import (
    CHUNKING_VERSION,
    EXTRACTION_PROVIDER_TEXT,
    _split_table_prose_segments,
    _table_br_to_space,
    extract_page_chunk_specs_text,
)

META = {"document_id": 1, "document_title": "Catalogue test"}

TABLE_MD = """## 2.10. Annexe

Texte d'introduction du tableau, suffisamment long pour passer le seuil minimal de page.

|REFERENCE Matiere|CODE CSTB|Coloris|
|---|---|---|
|4038-699 / 4038-099|4003|Blanc ou Gris|
|4091/A-4092/A-4093/A/147|300(variante A)|Gris|
|9718.3|441|Blanc|
"""


def _types(specs):
    return [s["metadata_json"]["section_type"] for s in specs]


class TestSegmentation:
    def test_separe_tableau_et_prose(self):
        segments = _split_table_prose_segments(TABLE_MD)
        kinds = [k for k, _ in segments]
        assert "table" in kinds
        assert "prose" in kinds
        assert kinds.index("prose") < kinds.index("table")

    def test_page_sans_tableau_reste_prose(self):
        segments = _split_table_prose_segments("## Titre\n\nUn paragraphe simple.")
        assert [k for k, _ in segments] == ["prose"]

    def test_br_remplace_par_espace_pas_par_saut_de_ligne(self):
        """<br> → newline casserait la ligne de tableau en deux."""
        assert _table_br_to_space("|a<br>b|c|") == "|a b|c|"
        assert "\n" not in _table_br_to_space("|Coloris<br>plaque|Oui|")


class TestChunksLignesDeTableau:
    def test_une_ligne_par_chunk_avec_entetes(self):
        specs = extract_page_chunk_specs_text(TABLE_MD, 11, META)
        rows = [s for s in specs if s["metadata_json"]["section_type"] == "table_row"]
        assert len(rows) == 3

        first = rows[0]["content"]
        # L'en-tête de colonne est réinjecté dans la ligne : le chunk est autosuffisant.
        assert "CODE CSTB" in first
        assert "4038-699" in first
        assert "p.11" in first

    def test_references_et_decimales_intactes(self):
        specs = extract_page_chunk_specs_text(TABLE_MD, 11, META)
        blob = " ".join(s["content"] for s in specs)
        # La référence décimale ne doit PAS être corrompue (9718.3 -> 97183).
        assert "9718.3" in blob
        assert "4091/A-4092/A-4093/A/147" in blob
        assert "300(variante A)" in blob

    def test_metadonnees_de_tableau(self):
        specs = extract_page_chunk_specs_text(TABLE_MD, 11, META)
        rows = [s for s in specs if s["metadata_json"]["section_type"] == "table_row"]
        meta = rows[0]["metadata_json"]
        assert meta["column_headers"] == ["REFERENCE Matiere", "CODE CSTB", "Coloris"]
        assert meta["row_index"] == 0
        assert meta["table_id"] == rows[1]["metadata_json"]["table_id"]
        assert meta["page_no"] == 11
        assert meta["content_type"] == "semantic_leaf"
        assert meta["extraction_provider"] == EXTRACTION_PROVIDER_TEXT
        assert meta["chunking_version"] == CHUNKING_VERSION
        # is_leaf doit être un booléen (la voie fallback historique posait la chaîne "true").
        assert meta["is_leaf"] is True

    def test_chunk_tableau_complet_ajoute(self):
        specs = extract_page_chunk_specs_text(TABLE_MD, 11, META)
        full = [s for s in specs if s["metadata_json"]["section_type"] == "table"]
        assert len(full) == 1
        assert "table_json" in full[0]["metadata_json"]
        tj = full[0]["metadata_json"]["table_json"]
        assert tj["nb_rows"] == 3
        assert tj["nb_cols"] == 3

    def test_mode_atomic_produit_un_seul_chunk(self):
        with patch("app.config.settings.TEXT_EXTRACTION_TABLE_MODE", "atomic"):
            specs = extract_page_chunk_specs_text(TABLE_MD, 11, META)
        table_specs = [
            s for s in specs if s["metadata_json"]["section_type"].startswith("table")
        ]
        assert len(table_specs) == 1
        assert "4038-699" in table_specs[0]["content"]
        assert "9718.3" in table_specs[0]["content"]

    def test_tableau_jamais_coupe_au_token(self):
        """Même avec un cap ridicule, une ligne de tableau reste entière."""
        with patch("app.config.settings.TEXT_EXTRACTION_MAX_CHUNK_TOKENS", 5):
            specs = extract_page_chunk_specs_text(TABLE_MD, 11, META)
        rows = [s for s in specs if s["metadata_json"]["section_type"] == "table_row"]
        assert len(rows) == 3
        assert "4038-699 / 4038-099" in rows[0]["content"]


class TestProse:
    def test_prose_produit_des_chunks(self):
        md = (
            "## Monter le report de charge\n\n"
            "1. Mettre et visser le report de charge en butee sur le pivot d'angle.\n"
            "2. Retirer la vis haute du palier d'angle.\n"
        )
        specs = extract_page_chunk_specs_text(md, 3, META)
        assert specs
        assert all(s["metadata_json"]["page_no"] == 3 for s in specs)
        assert all(s["metadata_json"]["is_leaf"] is True for s in specs)
        blob = " ".join(s["content"] for s in specs)
        assert "pivot d'angle" in blob


class TestSections:
    """Une section = un chunk. Régression du DTA titré en ###### rendu en un seul
    chunk « document_header » de 826 tokens sans titre."""

    DTA = (
        "###### **2.2.3. Eléments**\n\n"
        "Les meneaux ou traverses sont assembles soit mecaniquement soit par thermosoudure.\n\n"
        "###### 2.2.3.1. Cadre dormant\n\n"
        "Ce systeme de fenetres ne presente pas de particularite par rapport aux fenetres classiques.\n"
    )

    def test_titres_de_tous_niveaux_detectes(self):
        specs = extract_page_chunk_specs_text(self.DTA, 4, META)
        headings = [s["metadata_json"].get("heading") for s in specs]
        assert "2.2.3. Eléments" in headings
        assert "2.2.3.1. Cadre dormant" in headings
        # Plus aucun chunk fourre-tout « document_header » sans titre.
        assert all(s["metadata_json"]["section_type"] == "section" for s in specs)

    def test_une_section_par_chunk(self):
        specs = extract_page_chunk_specs_text(self.DTA, 4, META)
        assert len(specs) == 2
        assert "thermosoudure" in specs[0]["content"]
        assert "thermosoudure" not in specs[1]["content"]

    def test_section_courte_reste_entiere(self):
        specs = extract_page_chunk_specs_text(self.DTA, 4, META)
        first = specs[0]
        assert first["content"].startswith("2.2.3. Eléments")
        assert "section_part" not in first["metadata_json"]

    def test_section_trop_longue_scindee_avec_titre_repete(self):
        body = " ".join(f"phrase numero {i} du corps de section." for i in range(400))
        md = f"## Section volumineuse\n\n{body}\n"
        with patch("app.config.settings.TEXT_EXTRACTION_MAX_CHUNK_TOKENS", 120):
            specs = extract_page_chunk_specs_text(md, 7, META)
        assert len(specs) > 1
        # Aucun fragment orphelin : chaque part porte le titre de sa section.
        assert all(s["content"].startswith("Section volumineuse") for s in specs)
        assert all(s["metadata_json"]["heading"] == "Section volumineuse" for s in specs)
        assert specs[0]["metadata_json"]["section_parts_total"] == len(specs)


class TestRecollageInterPages:
    """Un saut de page ne coïncide pas avec une frontière de section."""

    def _pages(self):
        page1 = (
            "## Quincaillerie\n\n"
            "Quincaillerie PSK200 Portal de Siegenia pour l'oscillo, gaches en zamack.\n"
            "Le vantail repose sur un chariot regle en usine et verifie au montage.\n"
        )
        # La page 2 commence SANS titre : c'est la suite directe de la page 1.
        page2 = (
            "En aluminium ou acier protege contre la corrosion (grade 3 selon EN 1670).\n\n"
            "## Elements\n\nLes meneaux sont assembles mecaniquement ou par thermosoudure.\n"
        )
        return {
            1: extract_page_chunk_specs_text(page1, 1, META),
            2: extract_page_chunk_specs_text(page2, 2, META),
        }

    def test_section_orpheline_marquee(self):
        pages = self._pages()
        assert pages[2][0]["metadata_json"].get("orphan_section_start") is True

    def test_recollage_effectue(self):
        from app.services.text_page_extraction_service import _merge_sections_across_pages

        pages = self._pages()
        merged = _merge_sections_across_pages(pages)
        assert merged == 1

        tail = pages[1][-1]
        assert "Siegenia" in tail["content"]
        assert "EN 1670" in tail["content"]
        assert tail["metadata_json"]["cross_page_merge"] is True
        assert tail["metadata_json"]["page_end"] == 2
        assert tail["metadata_json"]["merged_pages"] == [1, 2]
        # Le fragment orphelin a bien été retiré de la page 2.
        assert all("EN 1670" not in s["content"] for s in pages[2])

    def test_recollage_refuse_si_trop_gros_mais_signale(self):
        from app.services.text_page_extraction_service import _merge_sections_across_pages

        pages = self._pages()
        with patch("app.config.settings.TEXT_EXTRACTION_MAX_CHUNK_TOKENS", 5):
            merged = _merge_sections_across_pages(pages)
        assert merged == 0
        assert pages[1][-1]["metadata_json"]["continues_on_next_page"] is True
        assert pages[2][0]["metadata_json"]["continues_from_previous_page"] is True

    def test_ligne_de_tableau_jamais_recollee(self):
        """Les lignes de tableau sont des unités atomiques."""
        from app.services.text_page_extraction_service import _merge_sections_across_pages

        pages = {
            1: extract_page_chunk_specs_text(TABLE_MD, 1, META),
            2: extract_page_chunk_specs_text(
                "Suite de texte sans titre en haut de la page suivante, assez longue.\n", 2, META
            ),
        }
        merged = _merge_sections_across_pages(pages)
        assert merged == 0


class TestBlocsImageEtPuces:
    """Régression du 28/07 : le motif « picture text » n'acceptait que l'ancienne
    forme à tirets (`--- Start of picture text ---`) alors que pymupdf4llm émet
    désormais un commentaire HTML (`<!-- ... -->`, deux tirets seulement). Le
    nettoyage ne se déclenchait plus : puces et sections finissaient concaténées
    dans un seul chunk, marqueurs bruts inclus."""

    PAGE = (
        "## Les avantages\n\n"
        "☐ Grandes dimensions (jusqu'a L 4,50 m)\n\n"
        "<!-- Start of picture text --> Masse reduite (-35 %)"
        "<br>Performances thermiques : Uw = 1.2\n"
        "<br>UN DESIGN EXCLUSIF<br>Le principe ouvrant cache reduit les masses."
        "<br>LES OUVERTURES<br>2 vantaux - 2 rails <!-- End of picture text -->\n\n"
        ".4\n"
    )

    def test_marqueurs_bruts_absents(self):
        specs = extract_page_chunk_specs_text(self.PAGE, 2, META)
        blob = " ".join(s["content"] for s in specs)
        assert "picture text" not in blob

    def test_titres_en_capitales_promus_en_sections(self):
        """« UN DESIGN EXCLUSIF » et « LES OUVERTURES » sont des sections, pas des puces."""
        specs = extract_page_chunk_specs_text(self.PAGE, 2, META)
        headings = [s["metadata_json"].get("heading") for s in specs]
        assert "UN DESIGN EXCLUSIF" in headings
        assert "LES OUVERTURES" in headings
        assert len(specs) >= 3

    def test_puces_glyphes_converties(self):
        specs = extract_page_chunk_specs_text(self.PAGE, 2, META)
        avantages = next(
            s for s in specs if s["metadata_json"].get("heading") == "Les avantages"
        )
        assert "- Grandes dimensions" in avantages["content"]
        assert "- Masse reduite" in avantages["content"]
        assert "☐" not in avantages["content"]

    def test_puces_non_collees_entre_elles(self):
        """Le `\\s*` du motif consommait les sauts de ligne : la première puce du bloc
        se collait à la ligne précédente."""
        specs = extract_page_chunk_specs_text(self.PAGE, 2, META)
        blob = "\n".join(s["content"] for s in specs)
        assert ")- Masse" not in blob

    def test_folio_supprime_si_egal_au_numero_de_page(self):
        specs = extract_page_chunk_specs_text(self.PAGE, 4, META)
        blob = "\n".join(s["content"] for s in specs)
        assert "\n.4" not in blob
        assert not blob.rstrip().endswith(".4")

    def test_nombre_nu_conserve_si_different_du_folio(self):
        """Sur la page 2, « .4 » n'est PAS un folio : ce pourrait être un code."""
        specs = extract_page_chunk_specs_text(self.PAGE, 2, META)
        blob = "\n".join(s["content"] for s in specs)
        assert ".4" in blob


class TestRecuperationTextePerdu:
    """pymupdf4llm SUPPRIME le texte posé sur un visuel quand les images ne sont pas
    écrites. Mesuré le 28/07 sur une plaquette : 12 % des lignes natives perdues —
    libellés de nuanciers, codes RAL, épaisseurs. Inacceptable pour un extracteur dont
    l'argument est l'exactitude : on récupère depuis page.get_text()."""

    def test_lignes_absentes_du_markdown_sont_detectees(self):
        from app.services.text_page_extraction_service import recover_lost_lines

        markdown = "## Choix des coloris\n\nLes laqués :\n"
        raw = "Choix des coloris\nLes teintés dans la masse :\nBlanc 9016\nLes laqués :\n"

        lost = recover_lost_lines(markdown, raw)

        assert "Les teintés dans la masse :" in lost
        assert "Blanc 9016" in lost
        assert "Les laqués :" not in lost, "déjà présent, ne doit pas être dupliqué"

    def test_pas_de_doublon_dans_les_lignes_recuperees(self):
        from app.services.text_page_extraction_service import recover_lost_lines

        lost = recover_lost_lines("", "Blanc 9016\nBlanc 9016\nIvoire 9001\n")
        assert lost == ["Blanc 9016", "Ivoire 9001"]

    def test_sans_texte_natif_aucune_recuperation(self):
        from app.services.text_page_extraction_service import recover_lost_lines

        assert recover_lost_lines("## Titre", "") == []

    def test_balisage_markdown_ne_cree_pas_de_faux_positif(self):
        """`<sup>` laissait les lettres « sup » dans le texte normalisé : un
        `TEXTURAL®` du PDF ne correspondait plus au `TEXTURAL**<sup>®</sup>` du
        markdown, donc des paragraphes DÉJÀ présents étaient réinjectés en vrac."""
        from app.services.text_page_extraction_service import recover_lost_lines

        markdown = "PROFERM propose **PVC, ALU, HYBRIDE & TEXTURAL**<sup>®</sup> ."
        raw = "PROFERM propose PVC, ALU, HYBRIDE & TEXTURAL®."

        assert recover_lost_lines(markdown, raw) == []

    def test_fragments_de_phrase_recolles(self):
        """Les libellés d'encart sont des boîtes distinctes : une phrase courte y est
        coupée en morceaux qui deviendraient autant de puces illisibles."""
        from app.services.text_page_extraction_service import recover_lost_lines

        raw = "Pivot pouvant\nsupporter le poids\nd'une fenêtre jusqu'à\n130kg.\n"
        lost = recover_lost_lines("## Robustesse", raw)

        assert lost == ["Pivot pouvant supporter le poids d'une fenêtre jusqu'à 130kg."]

    def test_deux_paragraphes_ne_fusionnent_pas(self):
        """Le recollage ne doit mordre que sur des fragments COURTS."""
        from app.services.text_page_extraction_service import _join_wrapped_fragments

        long_a = "a" * 130
        long_b = "b" * 130
        assert _join_wrapped_fragments([long_a, long_b]) == [long_a, long_b]

    def test_integration_le_texte_perdu_arrive_dans_les_chunks(self):
        markdown = (
            "## CHOIX DES COLORIS\n\n"
            "Au-dela des hautes performances, vous pouvez personnaliser vos menuiseries "
            "selon vos envies et choisir parmi un large choix de couleurs.\n"
        )
        raw = markdown + "\nLes teintes dans la masse :\nBlanc 9016\n"

        specs = extract_page_chunk_specs_text(markdown, 8, META, raw_text=raw)
        blob = " ".join(s["content"] for s in specs)

        assert "Les teintes dans la masse" in blob
        assert "Blanc 9016" in blob


class TestMobilierDePage:
    def test_folio_egal_au_numero_de_page_supprime(self):
        from app.services.pdf_extraction_service import clean_pymupdf4llm_markdown

        cleaned = clean_pymupdf4llm_markdown("Contenu utile\n.8\n", page_no=8)
        assert ".8" not in cleaned
        assert "Contenu utile" in cleaned

    def test_code_ral_conserve(self):
        """« 9016 » a la même forme qu'un folio : il ne doit PAS être supprimé.
        Le filtre initial retirait tout nombre nu, ce qui aurait mangé les RAL,
        les codes CSTB (300) et les cotes (1500)."""
        from app.services.pdf_extraction_service import clean_pymupdf4llm_markdown

        cleaned = clean_pymupdf4llm_markdown("Blanc\n9016\n300\n1500\n", page_no=8)
        assert "9016" in cleaned
        assert "300" in cleaned
        assert "1500" in cleaned

    def test_forme_page_n_sur_m_supprimee(self):
        from app.services.pdf_extraction_service import clean_pymupdf4llm_markdown

        cleaned = clean_pymupdf4llm_markdown("Contenu\nPage 3 sur 53\n", page_no=3)
        assert "sur 53" not in cleaned


class TestTitresGrasNonApparie:
    def test_asterisques_orphelines_retirees(self):
        """pymupdf4llm produit des titres à gras ouvert non fermé quand la mise en
        gras déborde du titre."""
        md = (
            "## **INTÉRIEUR ET EXTÉRIEUR PVC\n\n"
            "Texte de la section suffisamment long pour passer le seuil de page.\n"
        )
        specs = extract_page_chunk_specs_text(md, 8, META)
        headings = [s["metadata_json"].get("heading") for s in specs]

        assert "INTÉRIEUR ET EXTÉRIEUR PVC" in headings
        assert not any((h or "").startswith("*") for h in headings)


class TestDeduplication:
    """pymupdf4llm émet un bloc par image : deux images qui se recouvrent produisent
    deux fois le même texte."""

    def test_chunk_contenu_dans_un_autre_est_ecarte(self):
        from app.services.text_page_extraction_service import _drop_duplicate_specs

        specs = [
            {"content": "Le principe ouvrant cache reduit les masses vues d'aluminium."},
            {"content": "Le principe ouvrant cache reduit les masses."},
            {"content": "Section totalement differente sur les ouvertures."},
        ]
        kept = _drop_duplicate_specs(specs)
        contents = [s["content"] for s in kept]

        assert len(kept) == 2
        assert "Le principe ouvrant cache reduit les masses vues d'aluminium." in contents
        assert "Section totalement differente sur les ouvertures." in contents

    def test_ordre_de_lecture_preserve(self):
        from app.services.text_page_extraction_service import _drop_duplicate_specs

        specs = [{"content": "AAA premier bloc"}, {"content": "BBB second bloc plus long"}]
        kept = _drop_duplicate_specs(specs)
        assert [s["content"] for s in kept] == [
            "AAA premier bloc",
            "BBB second bloc plus long",
        ]

    def test_aucun_doublon_rien_ne_change(self):
        from app.services.text_page_extraction_service import _drop_duplicate_specs

        specs = [{"content": "Alpha"}, {"content": "Beta"}]
        assert len(_drop_duplicate_specs(specs)) == 2


class TestPagesSansTexte:
    def test_page_vide_ne_produit_rien(self):
        assert extract_page_chunk_specs_text("", 1, META) == []

    def test_page_trop_courte_ne_produit_rien(self):
        """Une page scannée renvoie quelques caractères parasites, pas du contenu."""
        assert extract_page_chunk_specs_text("Page 12", 12, META) == []


class TestPlafondChunks:
    def test_depassement_du_plafond_est_journalise(self):
        """La voie vision tronquait en silence ; ici tout dépassement est signalé."""
        rows = "\n".join(f"|ref-{i}|code-{i}|val-{i}|" for i in range(40))
        md = f"## Grand tableau\n\nIntroduction suffisamment longue pour la page.\n\n|A|B|C|\n|---|---|---|\n{rows}\n"
        with patch("app.config.settings.TEXT_EXTRACTION_MAX_CHUNKS_PER_PAGE", 10):
            with patch(
                "app.services.text_page_extraction_service.logger"
            ) as mock_logger:
                specs = extract_page_chunk_specs_text(md, 5, META)

        assert len(specs) == 10
        assert mock_logger.warning.called
        assert "PERDU" in mock_logger.warning.call_args[0][0]
