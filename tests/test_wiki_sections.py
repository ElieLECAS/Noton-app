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
def snapshot():
    return load_snapshot(wiki_root())


@pytest.fixture(scope="module")
def index(snapshot) -> WikiIndex:
    return snapshot.index


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


def test_la_page_dominante_est_livree_entiere_une_fois(snapshot):
    index = snapshot.index
    livraison = Livraison()
    chemin = "/quincaillerie/soleal-gy-roulements-et-fermetures.md"
    assert len(index.par_chemin[chemin]["corps"].strip()) <= PAGE_ENTIERE_MAX
    texte, livree = index.page_entiere(chemin, livraison, entete="PAGE DOMINANTE")
    assert texte.startswith(f"===== PAGE DOMINANTE : {chemin} — page entière =====")
    assert livree == {"chemin": chemin, "mode": "page", "sections": []}
    assert chemin in livraison.pages and livraison.caracteres == len(index.par_chemin[chemin]["corps"].strip())
    # Une page livrée entière n'est pas renvoyée par lire, et n'est plus dominante.
    assert "déjà été livrée entière" in index.lire(snapshot.pages[chemin], None, livraison, 10**6)[0]
    assert index.page_dominante(index.classer_multi(["TGY3731 clé pompier"]), livraison) is None


def test_une_lecture_partielle_montre_ce_qui_n_a_pas_ete_lu(snapshot):
    """Q26 (02/10) : GLM lisait les tableaux de déductions sans « Ce que donnent ces tableaux ».

    Toute lecture partielle rend le sommaire de la page : les sections lues y sont marquées, les
    autres — dont la légende du tableau — apparaissent sans marque.
    """
    index = snapshot.index
    chemin = "/profiles/systeme-76-cotes-de-debit.md"
    texte, livree = index.lire(snapshot.pages[chemin], "§3", Livraison(), 10**6)
    assert livree["mode"] == "sections" and livree["sections"][0].startswith("§3")
    sommaire = texte.split("Sommaire de la page")[1].split("--- §3")[0]
    lignes = {l.strip().split(" ")[0]: l for l in sommaire.splitlines() if l.strip().startswith("§")}
    assert "← livrée" in lignes["§3"], "la section lue est marquée"
    assert "Ce que donnent ces tableaux" in lignes["§1"] and "livrée" not in lignes["§1"], "la légende n'est pas lue : elle se voit"
    assert "L'exemple du manuel" in lignes["§2"]
    # Lue ensuite, elle change de marque.
    livraison = Livraison()
    index.lire(snapshot.pages[chemin], "§3", livraison, 10**6)
    suite, _ = index.lire(snapshot.pages[chemin], "§1", livraison, 10**6)
    marques = {l.strip().split(" ")[0]: l for l in suite.split("Sommaire de la page")[1].splitlines() if l.strip().startswith("§")}
    assert "← livrée" in marques["§1"] and "déjà livrée" in marques["§3"]


def test_une_section_lue_n_est_jamais_redonnee(snapshot):
    """Rien n'est livré deux fois dans un tour, et le budget se compte en caractères livrés."""
    index = snapshot.index
    livraison = Livraison()
    page = snapshot.pages["/certifications/dta-6-16-2334.md"]
    _, livree = index.lire(page, "dimensions maximales", livraison, 10**6)
    assert livree["mode"] == "sections" and livraison.caracteres > 0
    lu = livraison.caracteres
    seconde, _ = index.lire(page, "dimensions maximales", livraison, 10**6)
    assert "déjà fourni plus haut" in seconde and livraison.caracteres == lu


def test_livraison_serialisable():
    livraison = Livraison({"/a.md#3"}, {"/b.md"}, 1234)
    assert Livraison.depuis_dict(livraison.vers_dict()) == livraison
    assert Livraison.depuis_dict(None) == Livraison()
