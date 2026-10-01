"""Les sections : découpe à l'indexation, classement par sections, livraison page ou sections."""
from __future__ import annotations

import pytest

from app.services.wiki_index import (
    PAGE_ENTIERE_MAX,
    Livraison,
    WikiIndex,
    decouper_sections,
    titre_section,
)
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def index() -> WikiIndex:
    return load_snapshot(wiki_root()).index


PAGE = """---
title: Essai
type: Profilé
---
Phrase d'ouverture de la page.

# Titre

## Vide

## Tableau

| Réf. | Cote |
| --- | --- |
""" + "".join(f"| R{i:03d} | {i} mm |\n" for i in range(400)) + """
## HTML

<table>
<thead>
<tr><th rowspan="2">Parclose</th><th colspan="2">Joint</th></tr>
<tr><th>B = 8 mm</th><th>B = 7 mm</th></tr>
</thead>
<tbody>
""" + "".join(f"<tr><td>T{i:04d}</td><td>{i}</td><td>{i + 1}</td></tr>\n" for i in range(300)) + """</tbody>
</table>

## Fin

Texte final.
"""


def test_decoupe_lignes_et_titres():
    sections = decouper_sections("/x/essai.md", PAGE, limite=2000)
    lignes = PAGE.splitlines()
    # La frontmatter n'est pas une section ; l'ouverture en est une, sans titre.
    assert sections[0]["titres"] == [] and sections[0]["debut"] == 5
    assert "Phrase d'ouverture" in sections[0]["texte"]
    # Les lignes citées sont celles du fichier.
    for s in sections:
        assert lignes[s["fin"] - 1] in s["texte"] or not lignes[s["fin"] - 1].strip()
    # Un titre suivi aussitôt d'un autre titre rejoint la section suivante.
    assert not any(s["titres"][-1:] == ["Vide"] and s["texte"].strip() == "## Vide" for s in sections)
    assert [s["numero"] for s in sections] == list(range(1, len(sections) + 1))


def test_un_tableau_markdown_coupe_garde_son_en_tete():
    sections = [s for s in decouper_sections("/x/essai.md", PAGE, limite=2000) if s["titres"][-1:] == ["Tableau"]]
    assert len(sections) > 3
    for s in sections[1:]:
        assert s["suite"] and s["texte"].startswith("| Réf. | Cote |\n| --- | --- |")
    assert all(len(s["texte"]) < 2600 for s in sections)


def test_un_tableau_html_coupe_garde_tout_son_en_tete():
    sections = [s for s in decouper_sections("/x/essai.md", PAGE, limite=2000) if s["titres"][-1:] == ["HTML"]]
    assert len(sections) > 3
    for s in sections:
        assert "B = 8 mm" in s["texte"] and "rowspan" in s["texte"]
    for s in sections[:-1]:
        assert s["coupe_tableau"] and s["texte"].rstrip().endswith("</table>")


def test_la_matrice_soleal_fy_garde_ses_trois_lignes_d_en_tete(index):
    """En-tête à trois niveaux (parclose / couleur du joint / B) : la confusion C / A du 23/09."""
    ids = index.sections_par_page["/profiles/soleal-fy-parcloses-et-vitrage.md"]
    matrice = [index.sections[i] for i in ids if "T591005" in index.sections[i]["texte"] and "<td" in index.sections[i]["texte"]]
    assert matrice
    assert all("Rouge TAS0016" in s["texte"] and "B = 6 mm" in s["texte"] for s in matrice)


def test_la_section_du_dta_remonte_sur_une_question_de_limites(index):
    classement = index.classer("PERFORM76 oscillo-battant dimensions maximales")
    assert classement[0]["entree"]["chemin"] in ("/certifications/dta-6-16-2334.md", "/gammes/perform.md")
    dta = next(c for c in classement if c["entree"]["chemin"] == "/certifications/dta-6-16-2334.md")
    meilleure = index.sections[dta["sections"][0][0]]
    assert "Dimensions maximales" in titre_section(meilleure)


def test_page_courte_entiere_page_longue_en_sections(index):
    livraison = Livraison()
    texte, livrees = index.livrer(index.classer("PERFORM76 oscillo-battant dimensions maximales"), livraison)
    modes = {l["chemin"]: l["mode"] for l in livrees}
    for chemin, mode in modes.items():
        taille = len(index.par_chemin[chemin]["corps"].strip())
        assert mode == ("page" if taille <= PAGE_ENTIERE_MAX else "sections")
    assert "sections" in modes.values()
    # Une page en sections arrive avec sa fiche et son sommaire.
    assert "Fiche : " in texte and "Sommaire — lire_page(chemin, section)" in texte and "← livrée" in texte
    assert len(livrees) <= 6
    assert livraison.caracteres <= 60_000 + 10_000  # budget des sections, plus les fiches et sommaires


def test_le_budget_et_le_dedoublonnage(index):
    livraison = Livraison()
    classement = index.classer("crémone oscillo-battant Roto NX")
    _, premieres = index.livrer(classement, livraison, budget=15_000)
    _, suivantes = index.livrer(classement, livraison, budget=15_000)
    vues = {(l["chemin"], s) for l in premieres for s in l["sections"]} | {l["chemin"] for l in premieres if l["mode"] == "page"}
    for l in suivantes:
        if l["mode"] == "page":
            assert l["chemin"] not in vues
        else:
            assert not any((l["chemin"], s) in vues for s in l["sections"])


def test_livraison_serialisable():
    livraison = Livraison({"/a.md#3"}, {"/b.md"}, 1234)
    assert Livraison.depuis_dict(livraison.vers_dict()) == livraison
    assert Livraison.depuis_dict(None) == Livraison()
