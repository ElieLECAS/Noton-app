"""L'index de navigation : recherche lexicale, facettes, classement par sections, anomalies."""
from __future__ import annotations

import pytest

from app.services.wiki_index import (
    WikiIndex,
    cotes_de,
    normalise,
    references_de,
    tokenise,
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


def _premieres(classement, n=3):
    return [c["entree"]["chemin"] for c in classement[:n]]


def test_recherche_par_reference(index):
    """Une référence ne figure dans aucun tag : elle se trouve par le texte intégral.

    76507 est citée par cinq pages, et en tête de ligne dans neuf sections ; ce qui compte est
    que sa page de famille soit classée parmi les premières, pas qu'elle devance le tableau de
    vitrage qui la cite six fois.
    """
    assert index.classer("76507"), "aucune page pour la référence 76507"
    assert "/profiles/perform76-parcloses.md" in _premieres(index.classer("76507"), 6)


def test_une_page_de_gamme_remonte_sur_une_limite_dimensionnelle(index):
    """Le modèle ne choisit plus de type : la page qui fixe la limite se trouve par ses mots."""
    classement = index.classer("PERFORM76 1 vantail à la française dimension maximale")
    assert "/gammes/perform.md" in _premieres(classement)


def test_seule_une_reference_de_piece_pese_triple(index):
    assert index._reference("tgy3702") and index._reference("76507")
    assert not index._reference("1800", cotes={"1800"})  # une cote de la question
    assert not index._reference("vitrage24")  # collé par la recherche, écrit par aucune page
    assert not index._reference("lumine55") and not index._reference("perform76")  # des produits


def test_la_gamme_ne_masque_pas_la_piece(index):
    """LUMINE55 compté triple faisait passer la gamme, le nuancier et l'argumentaire devant."""
    q = "Sur un châssis alu LUMINE55 en ouvrant apparent, j'ai un vitrage de 24 mm : quelle parclose ?"
    classement = index.classer("LUMINE55 parclose joint vitrage 24", cotes=cotes_de(q))
    assert "/profiles/soleal-fy-parcloses-et-vitrage.md" in _premieres(classement)


def test_une_graphie_pour_deux_ecritures():
    assert tokenise("manœuvre") == tokenise("manoeuvre")
    assert "487206" in tokenise("limiteur 487 206")
    assert "lumine65" in tokenise("LUMINE 65") and "lumine65" in tokenise("LUMINE65")
    assert tokenise("parcloses") == tokenise("parclose")
    assert tokenise("2,15 1,00") == ["2", "15", "1", "00"]  # des décimales ne se recollent pas
    # PERFORM+ est une gamme à part, pas la PERFORM.
    assert tokenise("PERFORM+") == ["performplus"] and tokenise("PERFORM") == ["perform"]


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
    garanties = "/garanties/garanties-par-composant.md"
    sans = index.classer("garantie structure LUMINE65")
    avec = index.classer("garantie structure LUMINE65", gamme="PERFORM", systeme="76")
    assert garanties in [c["entree"]["chemin"] for c in sans]
    assert garanties in [c["entree"]["chemin"] for c in avec], "la facette a caché la page"


def test_une_facette_bien_choisie_fait_remonter_la_page_du_produit(index):
    requete = "dimensions maximales vantail oscillo-battant"
    sans = [c["entree"]["chemin"] for c in index.classer(requete)]
    avec = [c["entree"]["chemin"] for c in index.classer(requete, gamme="PERFORM")]
    perform = [i for i, p in enumerate(avec) if index.par_chemin[p]["gamme"] == ["PERFORM"]]
    assert perform and perform[0] <= next((i for i, p in enumerate(sans) if index.par_chemin[p]["gamme"] == ["PERFORM"]), 99)


def test_les_registres_d_anomalies_ne_sont_jamais_classes(index):
    """Ils se lisent entrée par entrée ; les entrées utiles sont injectées par le serveur."""
    classement = index.classer("contradiction garantie ferrure Technal CTR-09")
    assert classement
    assert not any(c["entree"]["chemin"].startswith("/anomalies/") for c in classement)


def test_une_source_pese_moins_qu_une_page_concept(index):
    """Une page sources/ résume un document ; la page concept porte la valeur."""
    classement = index.classer("SOLEAL FY parclose joint intérieur")
    concepts = [c for c in classement if not c["entree"]["chemin"].startswith("/sources/")]
    assert concepts and _premieres(classement, 1)[0] == concepts[0]["entree"]["chemin"]


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


def test_vocabulaire_porte_gammes_systemes_et_tous_les_tags(index):
    vocabulaire = index.vocabulaire()
    assert "GAMMES" in vocabulaire and "SYSTÈMES" in vocabulaire and "TAGS" in vocabulaire
    assert "TYPES" not in vocabulaire, "le type ne se demande plus : la facette a disparu"
    assert "LUMINE" in vocabulaire
    tags = {t for e in index.entries for t in e["tags"]}
    assert tags and all(t in vocabulaire for t in tags), "tous les tags, pas les soixante premiers"
