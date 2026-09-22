"""L'index de navigation : recherche lexicale, facettes, tri des résultats, anomalies."""
from __future__ import annotations

import pytest

from app.services.wiki_index import (
    WikiIndex,
    formate_resultats,
    normalise,
    trier_resultats,
)
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def index() -> WikiIndex:
    return load_snapshot(wiki_root()).index


def test_normalise_accepte_chaine_entier_et_liste():
    assert normalise("PERFORM") == ["PERFORM"]
    assert normalise(55) == ["55"]
    assert normalise([55, 65]) == ["55", "65"]
    assert normalise(None) == [] and normalise("  ") == []


def test_index_exclut_les_pages_reservees(index):
    chemins = {e["chemin"] for e in index.entries}
    assert "/index.md" not in chemins and "/log.md" not in chemins
    assert "/profiles/perform76-parcloses.md" in chemins


def test_recherche_par_reference(index):
    """Une référence ne figure dans aucun tag : elle se trouve par le texte intégral."""
    resultats = index.search(mots_cles="76507", limite=5)
    assert resultats, "aucune page pour la référence 76507"
    assert resultats[0]["chemin"] == "/profiles/perform76-parcloses.md"


def test_une_facette_remonte_mais_n_exclut_pas(index):
    """Une facette mal choisie ne doit pas cacher la bonne page."""
    sans = index.search(mots_cles="garantie structure LUMINE65", limite=10)
    avec = index.search(mots_cles="garantie structure LUMINE65", tags="coulissant", limite=10)
    garanties = "/garanties/garanties-par-composant.md"
    assert any(p["chemin"] == garanties for p in sans)
    assert any(p["chemin"] == garanties for p in avec)


def test_facette_seule_sert_de_filtre(index):
    """Sans mots-clés il n'y a rien à classer : la facette redevient un filtre."""
    resultats = index.search(mots_cles="", type="Gamme", limite=50)
    assert resultats and all(p["type"] == "Gamme" for p in resultats)


def test_trier_resultats_ecarte_les_anomalies_et_repousse_les_sources(index):
    pages = [
        {"chemin": "/anomalies/contradictions-entre-sources.md"},
        {"chemin": "/sources/technal-dta-soleal-fy.md"},
        {"chemin": "/profiles/soleal-fy-cotes-de-debit.md"},
    ]
    entieres, reste = trier_resultats(pages, completes=3)
    chemins = [p["chemin"] for p in entieres]
    assert "/anomalies/contradictions-entre-sources.md" not in chemins
    # La page concept passe devant la source, quel que soit l'ordre du classement.
    assert chemins == [
        "/profiles/soleal-fy-cotes-de-debit.md",
        "/sources/technal-dta-soleal-fy.md",
    ]
    assert reste == []


def test_formate_resultats_livre_les_premieres_pages_entieres(index):
    pages = index.search(mots_cles="parclose vitrage perform76", limite=8)
    texte = formate_resultats(index, pages, "parclose vitrage perform76", completes=2)
    assert texte.count("===== PAGE ") == 2
    assert "===== AUTRES RÉSULTATS =====" in texte
    # Les pages entières portent leur corps, pas un extrait.
    assert len(texte) > 2000


def test_formate_resultats_sans_resultat(index):
    texte = formate_resultats(index, [], "zzzz")
    assert "Aucune page ne correspond" in texte


def test_index_des_anomalies_et_lecture_d_une_entree(index):
    assert "CTR-09" in index.anomalies
    ligne = index.anomalie("ctr-09")
    assert "/anomalies/contradictions-entre-sources.md" in ligne
    assert "Technal" in ligne or "TECHNAL" in ligne
    assert "Aucune entrée" in index.anomalie("CTR-999")


def test_match_anomalies_rapproche_par_page_lue(index):
    """Une entrée qui pointe une page chargée doit remonter, même sans mot commun."""
    page = next(e for e in index.entries if e["chemin"] == "/garanties/garanties-par-composant.md")
    trouvees = index.match_anomalies("Quelle garantie sur la ferrure Technal ?", [page])
    assert any(e["id"] == "CTR-09" for e in trouvees)


def test_vocabulaire_porte_types_gammes_et_systemes(index):
    vocabulaire = index.vocabulaire()
    assert "TYPES" in vocabulaire and "TAGS" in vocabulaire
    assert "GAMMES" in vocabulaire and "SYSTÈMES" in vocabulaire
    assert "LUMINE" in vocabulaire
