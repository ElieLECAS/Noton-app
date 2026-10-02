"""La carte : ce que lit un modèle qui navigue — fiche de la question, fusion des formulations,
page dominante, lignes qui répondent avec l'en-tête de leur tableau, marques « lue »."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.wiki_index import (
    PAGE_ENTIERE_MAX,
    Livraison,
    WikiIndex,
    cotes_de,
)
from app.services.wiki_service import load_snapshot, wiki_root


def page(chemin, titre, corps, **kw):
    return SimpleNamespace(
        id=chemin, reserved=False, missing=False, title=titre, description=kw.get("description", ""),
        tags=kw.get("tags", []), gamme=kw.get("gamme", []), systeme=kw.get("systeme", []), famille=[],
        status="stable", type=kw.get("type", "Profilé"), body=corps, raw_text=corps,
    )


PARCLOSES = """# Parcloses ESSAI76

Les parcloses de la gamme, par épaisseur de vitrage.

## Cotes des parcloses

| Parclose | Épaisseur de vitrage (mm) | Coupe |
| --- | --- | --- |
""" + "".join(f"| {2000 + i} | {16 + i} | coupe-{2000 + i}.png |\n" for i in range(30)) + """| 76507 | 44 | coupe-76507.png |

## Joints

<table>
<thead>
<tr><th>Joint</th><th>Usage</th></tr>
</thead>
<tbody>
<tr><td>J079</td><td>joint de frappe</td></tr>
<tr><td>J080</td><td>joint de vitrage</td></tr>
</tbody>
</table>
"""

FERMETURES = """# Fermetures du coulissant ESSAI

## Fermeture pompier

- Référence : **TGY3731**, avec ses vis de fixation.
- Permet l'ouverture d'urgence depuis l'extérieur à l'aide d'une clé spéciale.

## Cylindres

Les cylindres européens se commandent séparément.
"""

GAMME = """# Gamme ESSAI

La gamme ESSAI76 se décline en fenêtre et porte-fenêtre. Dimensions maximales par ouvrant.
""" + "Texte de remplissage sur la gamme. " * 800


@pytest.fixture(scope="module")
def mini() -> WikiIndex:
    return WikiIndex([
        page("/profiles/essai-parcloses.md", "Parcloses ESSAI76", PARCLOSES, gamme=["ESSAI"], systeme=["76"]),
        page("/quincaillerie/essai-fermetures.md", "Fermetures du coulissant ESSAI", FERMETURES, gamme=["ESSAI"]),
        page("/gammes/essai.md", "Gamme ESSAI", GAMME, type="Gamme", gamme=["ESSAI"]),
    ])


def _sid(index, chemin, mot):
    return next(s["id"] for s in index.sections if s["chemin"] == chemin and mot in s["texte"])


def test_une_ligne_de_tableau_arrive_avec_son_en_tete(mini):
    sid = _sid(mini, "/profiles/essai-parcloses.md", "76507")
    lignes = mini.lignes_qui_repondent(sid, ["76507"])
    assert lignes[0].startswith("| Parclose |") and lignes[1].startswith("| --- ")
    assert any("76507" in l for l in lignes)
    assert not any("2010" in l for l in lignes), "une ligne qui ne répond pas ne vient pas"


def test_une_ligne_de_tableau_html_arrive_avec_son_en_tete(mini):
    sid = _sid(mini, "/profiles/essai-parcloses.md", "J079")
    lignes = mini.lignes_qui_repondent(sid, ["j079"])
    assert any("<th>Joint</th>" in l for l in lignes)
    assert any("J079" in l and "<td>" in l for l in lignes)
    assert not any("J080" in l for l in lignes)


def test_la_fiche_dit_ou_est_la_reference_rare_et_ce_qui_n_existe_pas(mini):
    fiche = mini.fiche_question("Quelle clé pour la fermeture TGY3731 ou la TGY9999 en ESSAI76, 1800 de large ?")
    assert fiche["rares"] == ["tgy3731"]
    assert fiche["absentes"] == ["tgy9999"]
    assert "1800" in fiche["cotes"] and "1800" not in fiche["rares"]
    assert fiche["produits"] == ["ESSAI76"]
    assert "/quincaillerie/essai-fermetures.md" in fiche["texte"] and "**TGY3731**" in fiche["texte"]
    assert "Absent de tout le wiki : TGY9999" in fiche["texte"]
    # Un produit n'est pas une référence : ESSAI76 ne déclenche aucune recherche exacte.
    assert "essai76" not in fiche["rares"]


def test_la_fiche_est_vide_quand_il_n_y_a_rien_a_dire(mini):
    assert mini.fiche_question("Comment choisir une parclose ?")["texte"] == ""


def test_une_formulation_seule_garde_l_ordre_de_classer(mini):
    seule = mini.classer_multi(["parclose vitrage 44"])
    simple = mini.classer("parclose vitrage 44")
    assert [p["entree"]["chemin"] for p in seule] == [p["entree"]["chemin"] for p in simple]
    assert seule[0]["relatif"] == pytest.approx(1.0)


def test_la_fusion_reunit_les_pages_des_formulations_et_respecte_leur_priorite(mini):
    fusion = mini.classer_multi(["fermeture pompier", "parclose 76507"])
    chemins = [p["entree"]["chemin"] for p in fusion]
    assert chemins[0] in ("/quincaillerie/essai-fermetures.md", "/profiles/essai-parcloses.md")
    assert {"/quincaillerie/essai-fermetures.md", "/profiles/essai-parcloses.md"} <= set(chemins[:2])
    # Le score d'une section est celui de la première formulation qui la classe.
    premiere = {sid: s for c in mini.classer("fermeture pompier") for sid, s in c["sections"]}
    for page_ in fusion:
        for sid, note in page_["sections"]:
            if sid in premiere:
                assert note == pytest.approx(premiere[sid])


def test_une_page_domine_si_la_suivante_ne_vaut_pas_la_moitie(mini):
    livraison = Livraison()
    dominante = mini.page_dominante(mini.classer_multi(["TGY3731 clé pompier"]), livraison)
    assert dominante is not None and dominante["entree"]["chemin"] == "/quincaillerie/essai-fermetures.md"
    # Déjà livrée : plus dominante, la carte suffit.
    livraison.pages.add("/quincaillerie/essai-fermetures.md")
    assert mini.page_dominante(mini.classer_multi(["TGY3731 clé pompier"]), livraison) is None


def test_une_page_trop_grande_n_est_jamais_dominante(mini):
    classement = mini.classer_multi(["fenêtre porte-fenêtre dimensions maximales par ouvrant"])
    assert classement[0]["entree"]["chemin"] == "/gammes/essai.md"
    assert len(mini.par_chemin["/gammes/essai.md"]["corps"].strip()) > PAGE_ENTIERE_MAX
    assert mini.page_dominante(classement, Livraison()) is None


def test_la_carte_liste_les_pages_les_sections_et_les_lignes(mini):
    requetes = ["parclose vitrage 44", "76507"]
    classement = mini.classer_multi(requetes)
    texte, listees = mini.carte(classement, mini.jetons_requete(requetes), Livraison())
    assert listees[0] == "/profiles/essai-parcloses.md"
    assert texte.startswith("===== CARTE — ")
    assert "§" in texte and "| 76507 | 44 |" in texte and "| Parclose |" in texte


def test_la_carte_marque_ce_qui_a_ete_lu_et_ne_le_redonne_pas(mini):
    requetes = ["parclose vitrage 44", "76507"]
    classement = mini.classer_multi(requetes)
    livraison = Livraison()
    livraison.pages.add("/profiles/essai-parcloses.md")
    texte, _ = mini.carte(classement, mini.jetons_requete(requetes), livraison)
    assert "page déjà lue entière" in texte and "(lue)" in texte
    assert "| 76507 | 44 |" not in texte, "une section lue n'est pas redonnée"


def test_le_budget_de_la_carte_garde_au_moins_trois_pages():
    pages = [
        page(f"/profiles/page-{i:02d}.md", f"Parclose page {i}", "# Parclose\n\n" + f"parclose vitrage ligne {i}. " * 40)
        for i in range(20)
    ]
    index = WikiIndex(pages)
    requetes = ["parclose vitrage"]
    classement = index.classer_multi(requetes)
    texte, listees = index.carte(classement, index.jetons_requete(requetes), Livraison(), budget=1)
    assert len(listees) == 3 and "autres pages classées plus bas" in texte
    _, toutes = index.carte(classement, index.jetons_requete(requetes), Livraison(), pages=12, budget=10**9)
    assert len(toutes) == 12


def test_une_recherche_sans_resultat_le_dit(mini):
    texte, listees = mini.carte([], [], Livraison())
    assert listees == [] and "Aucune page ne correspond" in texte


# --- le vrai wiki : quelques cas solides, pas de chiffres fins ------------------------------------


@pytest.fixture(scope="module")
def index() -> WikiIndex:
    return load_snapshot(wiki_root()).index


def test_tgy3731_sur_le_vrai_wiki(index):
    question = "Quelle référence de clé pompier dois-je commander avec la fermeture TGY3731 ?"
    fiche = index.fiche_question(question)
    assert fiche["rares"] == ["tgy3731"]
    assert "/quincaillerie/soleal-gy-roulements-et-fermetures.md" in fiche["texte"]
    requetes = [question, "clé pompier TGY3731"]
    classement = index.classer_multi(requetes, cotes=cotes_de(question))
    dominante = index.page_dominante(classement, Livraison())
    assert dominante and dominante["entree"]["chemin"] == "/quincaillerie/soleal-gy-roulements-et-fermetures.md"
    texte, listees = index.carte(classement, index.jetons_requete(requetes), Livraison())
    assert listees[0] == "/quincaillerie/soleal-gy-roulements-et-fermetures.md"
    assert "Fermeture pompier" in texte and len(texte) < 20_000


def test_un_produit_ou_une_annee_n_est_pas_une_reference_rare(index):
    fiche = index.fiche_question("Le catalogue 2024 de la PERFORM76 en 7016 : quelle parclose ?")
    assert "perform76" not in fiche["rares"] and "2024" not in fiche["rares"]
    assert "PERFORM76" in fiche["produits"]


def test_une_reference_absente_du_wiki_est_annoncee(index):
    fiche = index.fiche_question("Quelle est la parclose 3702 ou la TGY9999 ?")
    assert "tgy9999" in fiche["absentes"]
    assert "Absent de tout le wiki" in fiche["texte"]
