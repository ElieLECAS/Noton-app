"""L'index de navigation : recherche lexicale, facettes, tri des résultats, anomalies."""
from __future__ import annotations

import pytest

from app.services.wiki_index import (
    WikiIndex,
    cotes_de,
    formate_resultats,
    normalise,
    references_de,
    tokenise,
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
    """Une référence ne figure dans aucun tag : elle se trouve par le texte intégral.

    76507 est citée par cinq pages ; ce qui compte est que sa page de famille arrive parmi les
    pages livrées entières, pas qu'elle devance le tableau de vitrage qui la cite six fois.
    """
    resultats = index.search(mots_cles="76507", limite=5)
    assert resultats, "aucune page pour la référence 76507"
    livrees, _ = trier_resultats(resultats, 3)
    assert "/profiles/perform76-parcloses.md" in [p["chemin"] for p in livrees]


def test_le_type_ne_change_pas_le_classement(index):
    """Le modèle devine le type : « Profilé » pour une limite que fixe une page de gamme."""
    requete = "PERFORM76 1 vantail à la française dimension maximale"
    sans = [p["chemin"] for p in index.search(mots_cles=requete, limite=10)]
    avec = [p["chemin"] for p in index.search(mots_cles=requete, type="Profilé", limite=10)]
    assert sans == avec
    assert "/gammes/perform.md" in sans[:3]


def test_seule_une_reference_de_piece_pese_triple(index):
    assert index._reference("tgy3702") and index._reference("76507")
    assert not index._reference("1800", cotes={"1800"})  # une cote de la question
    assert not index._reference("vitrage24")  # collé par la recherche, écrit par aucune page
    assert not index._reference("lumine55") and not index._reference("perform76")  # des produits


def test_la_gamme_ne_masque_pas_la_piece(index):
    """LUMINE55 compté triple faisait passer la gamme, le nuancier et l'argumentaire devant."""
    q = "Sur un châssis alu LUMINE55 en ouvrant apparent, j'ai un vitrage de 24 mm : quelle parclose ?"
    resultats = index.search(mots_cles="LUMINE55 parclose joint vitrage 24", limite=10, cotes=cotes_de(q))
    livrees, _ = trier_resultats(resultats, 3)
    assert "/profiles/soleal-fy-parcloses-et-vitrage.md" in [p["chemin"] for p in livrees]


def test_une_graphie_pour_deux_ecritures():
    assert tokenise("manœuvre") == tokenise("manoeuvre")
    assert "487206" in tokenise("limiteur 487 206")
    assert "lumine65" in tokenise("LUMINE 65") and "lumine65" in tokenise("LUMINE65")
    assert tokenise("parcloses") == tokenise("parclose")
    assert tokenise("2,15 1,00") == ["2", "15", "1", "00"]  # des décimales ne se recollent pas


def test_une_cote_de_la_question_n_est_pas_une_reference():
    q = "Oscillo-battant PERFORM76 un vantail en 1 200 de large sur 1 600 de haut, ça passe ?"
    assert cotes_de(q) == {"1200", "1600"}
    assert references_de(q) == ["perform76"]
    assert cotes_de("SoftOpen sur un INNOSLIDE de 1 800 mm") == {"1800"}
    assert cotes_de("fenêtre 1300 x 2400") == {"1300", "2400"}
    assert references_de("parclose pour un vitrage de 44 mm") == []
    assert references_de("crémone 3 points TGY3702, il me faut la 4 points") == ["tgy3702"]


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
