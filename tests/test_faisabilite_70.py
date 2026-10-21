"""Le calculateur du système 70 : évaluateur de formules, tableaux du wiki (deux générations), débit, abaques,
inertie des meneaux, parcloses, limites, fiche de débit.

Les attendus sont ceux des exemples écrits dans le wiki ou recalculés à la main depuis ses tableaux : si une valeur ou
un en-tête change dans le wiki, un test le dit — c'est voulu, un outil qui calcule ne doit pas dériver en silence."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from app.services import faisabilite_70 as f70
from app.services import faisabilite_systemes as reg
from app.services.faisabilite import DonneeIntrouvable, ETUDE, HORS, INFO, OK
from app.services.faisabilite_70 import FormuleIllisible, calcul_debit, donnees, evaluer
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _ctl(r, id_):
    return next(c for c in r["controles"] if c["id"] == id_)


def _ids(r):
    return [c["id"] for c in r["controles"]]


def S(**kw):
    """Une saisie : la fenêtre à deux vantaux sur meneau de l'exemple du manuel, sauf ce que l'on change."""
    base = {"configuration": "2v_meneau", "largeur_mm": 2000, "hauteur_mm": 1200, "dormant": "6101", "ouvrant": "6112",
            "meneau": "6127", "classe_vent": "va2"}
    base.update(kw)
    return base


def V(snap, **kw):
    return f70.verifier(snap, S(**kw))


# ---- l'évaluateur de formules : une grammaire, jamais eval ------------------------------------


@pytest.mark.parametrize("formule, variables, attendu", [
    ("X − 22", {"X": 1000}, 978),
    ("DEO − 70 mm", {"DEO": 1128}, 1058),
    ("DHT − 148", {"DHT": 1000}, 852),
    ("DHT + 13,5", {"DHT": 1000}, 1013.5),
    ("« Débit battement (0141) = DEO – 72 mm »", {"DEO": 1000}, 928),   # seul ce qui suit « = » se calcule
    ("= 952 – 111", {}, 841),
    ("2 x (56)", {}, 112),
    ("(DEO − 6) / 2", {"DEO": 100}, 47),
    ("−DEO", {"DEO": 5}, -5),
])
def test_l_evaluateur_lit_les_formules_du_wiki(formule, variables, attendu):
    assert evaluer(formule, variables) == pytest.approx(attendu)


@pytest.mark.parametrize("formule", ["DHT + 2 × aile de recouvrement", "DEO −", "(DEO", "__import__('os')", "", "DEO ; 3"])
def test_une_formule_illisible_n_est_pas_calculee(formule):
    with pytest.raises(FormuleIllisible):
        evaluer(formule, {"DEO": 1, "DHT": 1})


# ---- la lecture des tableaux : le contrat avec le wiki ----------------------------------------


def test_les_tableaux_se_lisent_pour_les_deux_generations(snap):
    d = donnees(snap)
    g23, g08 = d.generations["2023"], d.generations["2008"]
    # 2023 : débit
    assert g23.debit.dormants["6101"]["deo"] == 36 and g23.debit.dormants["6101"]["vitrage"] == 47
    assert g23.debit.dormants["6101"]["renfort_meneau"] == {"set": 102, "equerre": 47}
    assert g23.debit.meneaux["6127"]["deo"] == 12 and g23.debit.meneaux["6126"]["deo"] is None   # 6126 : « — » en DEO
    assert g23.debit.ouvrants["6112"]["vitrage"] == 56 and g23.debit.ouvrants["2416"]["vitrage"] == 101
    assert "6102" not in g23.debit.dormants                                                       # VER-85
    assert g23.debit.battement_dormant[("6101", "6130,6128,1578,6132")]["deo"] == "X − 22"
    assert "DEO − 12 mm" in g23.debit.battements["6130"]["interieur"]
    # 2008 : les mêmes lignes, d'autres valeurs, jamais mêlées
    assert g08.debit.dormants["6101"]["vitrage"] == 46.5 and g08.debit.ouvrants["6112"]["vitrage"] == 55.5
    assert "6102" in g08.debit.dormants and "2469" in g08.debit.meneaux and "6126" not in g08.debit.meneaux
    assert "DEO – 6 mm" in g08.debit.battements["6130"]["interieur"]
    assert g08.debit.planches is None and g23.debit.planches
    # limites de baie, fiches, verrouillage, porte, seuil
    assert ("Française 1 vantail", 2.15, 1.0) in d.dtd and d.fiches[0] == (0.9, 2) and d.verrouillage_mm == 900
    assert d.vitrage_dtd_mm == 42
    assert d.porte["Porte 1 vantail"]["Profilés plaxés (mm)"] == (1100, 2200)
    assert d.seuil["6101"]["largeur"] == "DHT − 148" and d.seuils == ["9F67", "9F68"]
    # règles des abaques : 35 kg/m² (14 mm) en 2023, 30 (12 mm) en 2008 (CTR-80)
    assert (g23.abaques.verre_kg_m2, g23.abaques.verre_mm) == (35, 14) and (g08.abaques.verre_kg_m2, g08.abaques.verre_mm) == (30, 12)
    assert g23.abaques.rapport_max == 1.25 and g23.abaques.precision_mm == 10
    # vitrage : plages écrites, par feuillure
    assert g23.vitrage.plage == (5, 42) and g23.vitrage.plage_elargisseur == (21, 58)
    assert g08.vitrage.plage == (5, 39) and g08.vitrage.plage_elargisseur == (21, 55)
    # renforts de meneau : l'inertie Iw
    assert g23.meneaux_renfort["2425"]["iw"] == 9.1 and g23.meneaux_renfort["6127"]["renfort"] == "V603"
    assert list(d.classes)[:3] == ["va1", "va2", "va3"] and "800 Pa" in d.classes["va2"]["libelle"]


def test_dix_huit_planches_en_2023_et_quatre_variantes(snap):
    pl = donnees(snap).generations["2023"].abaques.planches
    principales = [p for p in pl if p.variante == "principal"]
    assert len(principales) == 18 and len(pl) - len(principales) == 4
    # une planche titrée « OF/OB » est en réalité une planche OF (INC-138) : le type vient de la bande de paumelles
    p144 = next(p for p in principales if p.pdf == 144)
    assert (p144.classe, p144.genre) == ("VA3", "of2") and "INC-138" in p144.anomalies
    assert next(p for p in principales if p.pdf == 142).genre == "ofob2"
    assert "INC-135" in next(p for p in principales if p.pdf == 145).anomalies
    assert all(p.genre == "ob1" and "INC-137" in p.anomalies for p in principales if p.pdf in (140, 141))
    # les limites du tableau de l'« autre possibilité » bornent aussi l'abaque principal de la même planche
    assert p144.limites["realisation"]


def test_un_en_tete_change_dans_le_wiki_casse_le_calcul(snap):
    page = f70.P + "cotes-de-debit.md"
    modifiee = dataclasses.replace(snap.pages[page], body=snap.pages[page].body.replace("① DEO (mm)", "① DEO modifié (mm)"))
    faux = SimpleNamespace(pages={**snap.pages, page: modifiee}, loaded_at=snap.loaded_at)
    with pytest.raises(DonneeIntrouvable, match="DEO"):
        f70._charge_debit(faux, "2023")


# ---- le débit : l'exemple du manuel ----------------------------------------------------------


def test_exemple_du_manuel_2023_vitrage_840_et_non_841(snap):
    """DHT 2 000 × 1 200, dormant 6101 (a = 36), meneau 6127 (b = 12), ouvrant 6112 : DEO 952 ; le tableau retranche 56
    et non 55,5, donc 840 (l'exemple imprimé écrit 841 : INC-125)."""
    d = donnees(snap)
    deb = calcul_debit(d.generations["2023"], d, {**S(), "generation": "2023", "bas": "dormant"})
    assert deb["deo"] == (952, 1128) and deb["vitrage"] == (840, 1016) and deb["dfo"] == (912, 1088)
    r = V(snap)
    c = _ctl(r, "debit")
    assert c["statut"] == INFO and "INC-125" in c["anomalies"] and "952 × 1128" in c["detail"] and "840 × 1016" in c["detail"]
    assert [e["resultat"] for e in c["valeurs"]["etapes"]][:2] == ["1000", "952"]


def test_exemple_du_manuel_2008_vitrage_841(snap):
    """Le classeur de 2008 retranche 55,5 : l'exemple imprimé (841) est juste ; l'écart avec 2023 est affiché."""
    r = V(snap, generation="2008")
    c = _ctl(r, "debit")
    assert c["valeurs"]["deo"][0] == 952 and c["valeurs"]["vitrage"] == (841, 1017) and "CTR-78" in c["anomalies"]
    g = _ctl(r, "generation")
    assert "55,5 (2008) contre 56 (2023)" in g["detail"] and "CTR-78" in g["anomalies"]
    assert not [c for c in V(snap)["controles"] if c["id"] == "generation"]      # 2023 : pas d'écart à dire


def test_deux_vantaux_a_battement(snap):
    """X = 1 000 : DEO 978 (X − 22 pour le 6101 et le 6130) ; le battement se débite à partir de la hauteur d'ouvrant
    (1 128) : extérieur DEO − 70, intérieur DEO − 12 en 2023 et DEO − 6 en 2008 (CTR-79)."""
    r = V(snap, configuration="2v_battement", battement="6130")
    c = _ctl(r, "debit")
    assert c["valeurs"]["deo"] == (978, 1128) and c["valeurs"]["vitrage"] == (866, 1016)
    assert c["valeurs"]["battement"] == {"exterieur": 1058, "interieur": 1116}
    assert {"CTR-79", "INC-153"} <= set(c["anomalies"])
    assert "ouvrant réduit" in c["detail"]
    r8 = V(snap, configuration="2v_battement", battement="6130", generation="2008")
    assert _ctl(r8, "debit")["valeurs"]["battement"] == {"exterieur": 1058, "interieur": 1122}
    # libellés du tableau et du dessin qui divergent, valeurs identiques : inscrits au registre
    assert {"INC-126"} <= set(_ctl(V(snap, configuration="2v_battement", battement="6128"), "debit")["anomalies"])


def test_le_debit_n_invente_rien(snap):
    # 6102 : aucune cote de dormant dans le manuel de 2023 (VER-85), elle est dans le classeur de 2008
    c = _ctl(V(snap, configuration="1v_of", dormant="6102"), "debit")
    assert c["statut"] == ETUDE and "VER-85" in c["anomalies"]
    assert _ctl(V(snap, configuration="1v_of", dormant="6102", generation="2008"), "debit")["statut"] == INFO
    # 6126 : « — » en ① DEO, la planche ne donne que le vitrage fixe
    c = _ctl(V(snap, meneau="6126"), "debit")
    assert c["statut"] == ETUDE and "6126" in c["detail"]
    # un battement absent du classeur de 2008
    c = _ctl(V(snap, configuration="2v_battement", battement="1578", generation="2008"), "debit")
    assert c["statut"] == ETUDE
    # dormant 6155 sur la planche du battement 6128 avec l'ouvrant 6115 : absent (VER-85)
    c = _ctl(V(snap, configuration="2v_battement", dormant="6155", ouvrant="6115", battement="6128"), "debit")
    assert c["statut"] == ETUDE and "VER-85" in c["anomalies"]


def test_chassis_fixe_et_seuil(snap):
    r = V(snap, configuration="fixe", largeur_mm=1000, hauteur_mm=1000)
    assert _ctl(r, "debit")["valeurs"]["vitrage"] == (906, 906) and _ctl(r, "dtd")["statut"] == INFO
    # 6101, DHT 1 000, seuil 9F67 : seuil 852 (DHT − 148), montants 993,5 (DHT − 6,5)
    r = V(snap, configuration="1v_of", largeur_mm=1000, hauteur_mm=1000, bas="seuil", seuil="9F67")
    s = _ctl(r, "seuil")
    assert s["valeurs"] == {"seuil": 852, "montant": 993.5, "embout": "9F65"}
    # l'ouvrant sur seuil n'a aucune cote de débit : le débit et l'abaque restent sur étude
    assert _ctl(r, "debit")["statut"] == ETUDE and _ctl(r, "ouvrant")["statut"] == ETUDE and r["verdict"] == ETUDE
    # le tableau des seuils est celui de 2023
    assert _ctl(V(snap, configuration="1v_of", bas="seuil", seuil="9F67", generation="2008"), "seuil")["statut"] == ETUDE


# ---- VER-28 : sur tout verdict ---------------------------------------------------------------


def test_ver28_est_affiche_en_tete_de_tout_verdict(snap):
    for kw in (S(), S(generation="2008"), S(configuration="fixe"), S(configuration="porte_1v", ouvrant="2416"),
               S(configuration="soufflet", largeur_mm=1200, hauteur_mm=700)):
        r = f70.verifier(snap, kw)
        premier = r["controles"][0]
        assert premier["id"] == "ver28" and premier["statut"] == INFO and premier["anomalies"] == ["VER-28"]
        assert "aucun document" in premier["detail"].lower() and "PERFORM" in premier["detail"]
        assert any("VER-28" in n for n in r["notes"])


# ---- les limites de baie et la porte -----------------------------------------------------------


def test_dtd(snap):
    ok = V(snap, configuration="1v_of", largeur_mm=722, hauteur_mm=1322)
    assert _ctl(ok, "dtd")["statut"] == OK
    hors = V(snap, configuration="1v_of", largeur_mm=1200, hauteur_mm=2300)
    assert _ctl(hors, "dtd")["statut"] == HORS and hors["verdict"] == HORS and hors["contrainte"]["id"] == "dtd"
    # l'oscillo-battant porte deux couples : 1,40 × 1,30 passe le second
    assert _ctl(V(snap, configuration="1v_ob", largeur_mm=1300, hauteur_mm=1400), "dtd")["statut"] == OK
    # deux vantaux sur meneau : le tableau n'a pas de ligne qui les nomme, 2,00 m passe l'une des deux lignes à deux vantaux
    c = _ctl(V(snap), "dtd")
    assert c["statut"] == ETUDE and "VER-30" in c["anomalies"]
    # à battement : la ligne « oscillo battante 2 vantaux » (VER-30), 1,60 m maxi
    assert _ctl(V(snap, configuration="2v_battement", battement="6130"), "dtd")["statut"] == HORS
    assert _ctl(V(snap, configuration="soufflet", largeur_mm=1200, hauteur_mm=700), "dtd")["statut"] == OK
    assert _ctl(V(snap, configuration="soufflet", largeur_mm=1400, hauteur_mm=700), "dtd")["statut"] == HORS
    assert _ctl(V(snap, configuration="coulissant", largeur_mm=3000, hauteur_mm=2000), "dtd")["statut"] == OK


def test_porte_d_entree(snap):
    r = V(snap, configuration="porte_1v", dormant="6101", ouvrant="2416", largeur_mm=1100, hauteur_mm=2200)
    c = _ctl(r, "porte")
    assert c["statut"] == OK and "CTR-99" in c["anomalies"] and c["valeurs"]["limite"] == (1100, 2300)
    assert _ctl(V(snap, configuration="porte_1v", ouvrant="2416", largeur_mm=1400, hauteur_mm=2200), "porte")["statut"] == HORS
    # plaxé : 1 100 × 2 200 ; la hauteur d'ouvrant sur seuil n'étant pas connue, 2 400 de baie reste « sur étude »
    c = _ctl(V(snap, configuration="porte_1v", ouvrant="2416", largeur_mm=1100, hauteur_mm=2400, finition="filme"), "porte")
    assert c["statut"] == ETUDE
    assert _ctl(V(snap, configuration="porte_1v", ouvrant="2416", generation="2008"), "porte")["statut"] == ETUDE
    # aucune cote de débit : la fiche le dit
    assert "indisponible" in f70.fiche_debit(snap, S(configuration="porte_1v", ouvrant="2416"))


# ---- l'abaque d'ouvrant : les trois exemples du tableau VA2 OF 1 vantail, ouvrant 6112 --------


@pytest.mark.parametrize("deo_l, deo_h, repere, paumelles", [
    (650, 1250, "A3", 3),
    (1050, 1650, "D6", 4),
])
def test_abaque_va2_of_1_vantail(snap, deo_l, deo_h, repere, paumelles):
    """DEO = DHT − 2 × 36 (dormant 6101) : la case et la bande de paumelles viennent du tableau de la planche."""
    r = V(snap, configuration="1v_of", largeur_mm=deo_l + 72, hauteur_mm=deo_h + 72)
    c = _ctl(r, "ouvrant")
    assert c["valeurs"]["repere"] == repere and c["valeurs"]["paumelles"] == paumelles
    assert c["valeurs"]["renforts"] == ["V158", "V258"] and "hypothèse" in c["detail"]
    assert c["sources"][0]["page"].endswith("systeme-70-abaques-va2.md")


def test_abaque_case_sans_repere(snap):
    """950 × 2 250 : la case n'a pas de repère (« - ») et la limite de réalisation (1 880 à 950) est dépassée de loin."""
    c = _ctl(V(snap, configuration="1v_of", largeur_mm=950 + 72, hauteur_mm=2250 + 72), "ouvrant")
    assert c["valeurs"]["repere"] is None and "n'a pas de repère" in c["detail"]
    assert c["statut"] == HORS and "limite de réalisation" in c["detail"]
    # une case sans repère que ni la limite de réalisation ni l'oblique n'expliquent reste « sur étude » : 1 300 de large
    # est hors de la grille de l'abaque
    c = _ctl(V(snap, configuration="1v_of", largeur_mm=1300 + 72, hauteur_mm=1400 + 72), "ouvrant")
    assert c["statut"] == ETUDE and "hors de la grille" in c["detail"]


def test_abaque_jamais_d_interpolation_et_limites_a_la_precision_de_lecture(snap):
    # sur une graduation (700), la bande supérieure : la plus exigeante des deux cases voisines
    d = donnees(snap)
    pl = next(p for p in d.generations["2023"].abaques.planches if p.classe == "VA2" and p.genre == "of1" and "6112" in p.refs)
    assert f70._lire(pl, 700, 1250)["colonne"] == (700, 750) and f70._lire(pl, 699, 1250)["colonne"] == (600, 700)
    assert f70._lire(pl, 650, 1300)["bande"] == (1300, 1400)
    # limite de réalisation : relevée à 900 (2 300) et 1 000 (1 880) ; à 950 la plus basse des deux
    assert f70._limite(pl.limites["realisation"], 950, True)[0] == 1880
    # 1 100 × 1 880 mm : à ±10 mm de la limite → sur étude ; au-delà → hors
    base = dict(configuration="1v_of", largeur_mm=1000 + 72)
    assert _ctl(V(snap, hauteur_mm=1885 + 72, **base), "ouvrant")["statut"] == ETUDE
    assert _ctl(V(snap, hauteur_mm=1950 + 72, **base), "ouvrant")["statut"] == HORS
    assert _ctl(V(snap, hauteur_mm=1700 + 72, **base), "ouvrant")["statut"] == OK
    # la largeur ne dépasse pas 125 % de la hauteur
    c = _ctl(V(snap, configuration="1v_of", largeur_mm=1050 + 72, hauteur_mm=800 + 72), "ouvrant")
    assert c["statut"] == HORS and "25" in c["detail"]


def test_abaque_va3_par_defaut_et_classes_sans_abaque(snap):
    # V*A3 par défaut (catalogue) ; V*A2 sur demande : même ouvrant, autre planche
    d = {**S(configuration="1v_of", largeur_mm=722, hauteur_mm=1322)}
    d.pop("classe_vent")
    assert _ctl(f70.verifier(snap, d), "ouvrant")["sources"][0]["page"].endswith("abaques-va3.md")
    # pas d'abaque pour V*A4 : sur étude, avec la raison
    c = _ctl(V(snap, configuration="1v_of", classe_vent="va4"), "ouvrant")
    assert c["statut"] == ETUDE and "V*A4" in c["detail"]


def test_abaque_ob_deux_vantaux_et_hors_abaques(snap):
    c = _ctl(V(snap, configuration="1v_ob", largeur_mm=722, hauteur_mm=1322), "ouvrant")
    assert c["statut"] == OK and c["valeurs"]["genre"] == "ob1"
    # deux vantaux OF/OB à battement : les ouvrants appariés (6112-6113)
    c = _ctl(V(snap, configuration="2v_battement_ob", battement="6130", largeur_mm=1800, hauteur_mm=1400), "ouvrant")
    assert c["valeurs"]["genre"] == "ofob2"
    # battement 0140 : la planche des ouvrants « 6112-0140 »
    c = _ctl(V(snap, configuration="2v_battement", battement="0140", largeur_mm=1800, hauteur_mm=1400), "ouvrant")
    assert "0140" in c["valeurs"]["planche"]
    # 2418 et 2415 / 2416 ne sont sur aucun abaque
    for ouvrant in ("2418", "2416"):
        c = _ctl(V(snap, configuration="1v_of", ouvrant=ouvrant), "ouvrant")
        assert c["statut"] == ETUDE and "hors abaques" in c["detail"]


def test_abaque_planches_a_anomalies(snap):
    # INC-137 : les deux planches « OF 1 vantail » titrées VA3 aux PDF 140-141 sont en réalité OB
    c = _ctl(V(snap, configuration="1v_ob", classe_vent="va3", largeur_mm=722, hauteur_mm=1322), "ouvrant")
    assert "INC-137" in c["anomalies"]


def test_renfort_ce_que_dit_le_repere(snap):
    r = V(snap, configuration="1v_of", largeur_mm=722, hauteur_mm=1322)
    c = _ctl(r, "renfort")
    assert c["statut"] == INFO and c["valeurs"]["repere"] == "A3" and c["valeurs"]["pieces"] == (0, 0)
    assert "aucun trait épais" in c["detail"] and "800 mm" in c["detail"]
    # D6 sur un vantail à la française : traverses et un montant
    r = V(snap, configuration="1v_of", largeur_mm=1122, hauteur_mm=1722)
    assert _ctl(r, "renfort")["valeurs"]["pieces"] == (2, 1)


def test_couleur_poids_et_precision(snap):
    # un film ou une laque peut avoir un L* < 82 : renfort systématique, V258 exclu — le repère ne suffit pas
    c = _ctl(V(snap, configuration="1v_of", largeur_mm=722, hauteur_mm=1322, finition="filme"), "ouvrant")
    assert c["statut"] == ETUDE and "L*" in c["detail"] and "82" in c["detail"]
    # 12 mm de verre, sous le plafond des abaques ; 16 mm : au-delà, sur étude
    ok = _ctl(V(snap, configuration="1v_of", vitrage="4-12-4-12-4"), "poids")
    assert ok["statut"] == INFO and "CTR-80" in ok["anomalies"]
    lourd = _ctl(V(snap, configuration="1v_of", vitrage="8-16-8"), "poids")
    assert lourd["statut"] == ETUDE and "renforcement systématique" in lourd["detail"]
    # 14 mm de verre : permis en 2023 (35 kg/m²), sur étude en 2008 (30 kg/m², 12 mm : CTR-80)
    assert _ctl(V(snap, configuration="1v_of", vitrage="6-16-8", generation="2023"), "poids")["statut"] == INFO
    assert _ctl(V(snap, configuration="1v_of", vitrage="6-16-8", generation="2008"), "poids")["statut"] == ETUDE


# ---- l'inertie d'un meneau ---------------------------------------------------------------------


def test_meneau_exemples_du_manuel(snap):
    """L 200, a 70, b 50 (V*A2) : 3,4 + 2,7 = 6,1 (imprimé 6,08, une décimale par case) ; le 2425 + 9132 (9,1) suffit, le
    6127 + V603 (3) non. L 140, a 40, b 60 : 0,7 + 0,9 = 1,6."""
    base = dict(portee_cm=200, a_cm=70, b_cm=50)
    c = _ctl(V(snap, meneau="2425", **base), "meneau")
    assert c["statut"] == OK and c["valeurs"]["requis_cm4"] == 6.1 and c["valeurs"]["iw_cm4"] == 9.1
    assert {"INC-141"} <= set(c["anomalies"]) and "800 Pa" in c["detail"]
    c = _ctl(V(snap, meneau="6127", **base), "meneau")
    assert c["statut"] == HORS and "INC-144" in c["anomalies"] and "ne suffit pas" in c["detail"]
    c = _ctl(V(snap, meneau="6127", portee_cm=140, a_cm=40, b_cm=60), "meneau")
    assert c["statut"] == OK and c["valeurs"]["requis_cm4"] == 1.6


def test_meneau_lecture_sans_interpolation_et_cases_vides(snap):
    # a = 65 cm se lit à 70 (la graduation supérieure) : 3,4, pas 3,25
    c = _ctl(V(snap, meneau="2425", portee_cm=200, a_cm=65, b_cm=50), "meneau")
    assert c["valeurs"]["requis_cm4"] == 6.1
    # portée de 205 cm : lue à 210
    assert _ctl(V(snap, meneau="2425", portee_cm=205, a_cm=70, b_cm=50), "meneau")["valeurs"]["portee_lue_cm"] == 210
    # a dépasse la moitié de la portée : la case est vide
    c = _ctl(V(snap, meneau="2425", portee_cm=100, a_cm=70, b_cm=50), "meneau")
    assert c["statut"] == ETUDE and "case vide" in c["detail"]
    # a et b se lisent sur les plans de charge, des images : non saisis, sur étude
    c = _ctl(V(snap, meneau="2425"), "meneau")
    assert c["statut"] == ETUDE and "plans de charge" in c["detail"]
    # V*A3 (1 200 Pa) : 5,1 + 4,0 = 9,1, soit exactement l'inertie du 9132 (9,1) : à moins de la marge d'une case, sur étude
    c = _ctl(V(snap, meneau="2425", portee_cm=200, a_cm=70, b_cm=50, classe_vent="va3"), "meneau")
    assert c["valeurs"]["requis_cm4"] == 9.1 and c["statut"] == ETUDE


# ---- les parcloses ---------------------------------------------------------------------------


def test_parclose_a_l_epaisseur_du_vitrage(snap):
    # 4-16-4 = 24 mm : 2454 (23,5-25) en 54 mm ; 0136 et 6147 écrites sous la 2454
    c = _ctl(V(snap, configuration="1v_of"), "parclose")
    assert c["statut"] == OK and c["valeurs"]["parcloses"][0]["refs"] == ["2454"] and "0136, 6147" in c["detail"]
    assert "CTR-92" in c["anomalies"] and c["valeurs"]["feuillure"] == 54
    # 2008 : 2435 / 0136 (nominal 24) ; la plage de la 2454 (22,5-24) couvre aussi 24 mm
    c = _ctl(V(snap, configuration="1v_of", generation="2008"), "parclose")
    assert c["valeurs"]["parcloses"][0]["refs"] == ["2435", "0136"] and "2454" in c["detail"] and "CTR-91" in c["anomalies"]
    # 4-12-4-12-4 = 36 mm : 76507
    c = _ctl(V(snap, configuration="1v_of", vitrage="4-12-4-12-4"), "parclose")
    assert c["valeurs"]["parcloses"][0]["refs"] == ["76507"] and c["statut"] == OK
    # une parclose sans plage écrite ne se lit qu'à son épaisseur nominale : 4-14-4-14-4 = 40 mm → 76509
    c = _ctl(V(snap, configuration="1v_of", vitrage="4-14-4-14-4"), "parclose")
    assert c["valeurs"]["parcloses"][0]["refs"] == ["76509"] and "sans plage écrite" in c["detail"]


def test_parclose_trous_du_tableau_et_elargisseur(snap):
    # 4-6-4 = 14 mm : le tableau n'a pas de ligne (13 puis 15,5) — VER-86, sur étude, avec les voisines
    c = _ctl(V(snap, configuration="1v_of", vitrage="4-6-4"), "parclose")
    assert c["statut"] == ETUDE and "VER-86" in c["anomalies"] and "2453" in c["detail"] and "76501" in c["detail"]
    # 44 mm : au-delà des 42 mm sans élargisseur ; le manuel l'admet avec l'élargisseur 93025 mais le DTD s'arrête à 42 mm
    c = _ctl(V(snap, configuration="1v_of", vitrage="4-16-4-16-4"), "parclose")
    assert c["statut"] == ETUDE and c["valeurs"]["feuillure"] == 70 and "DTD" in c["detail"]
    assert c["valeurs"]["parcloses"][0]["refs"] == ["76503"]
    # au-delà de 58 mm : hors
    assert _ctl(V(snap, configuration="1v_of", vitrage="10-20-10-20-10"), "parclose")["statut"] == HORS
    # un châssis fixe se lit sur le même tableau (« ouvrant à déport, dormant, meneau »)
    assert _ctl(V(snap, configuration="fixe"), "parclose")["valeurs"]["parcloses"][0]["refs"] == ["2454"]


# ---- la forme du résultat, les brouillons, le registre ------------------------------------------


def test_forme_du_resultat(snap):
    r = V(snap, meneau="2425", portee_cm=200, a_cm=70, b_cm=50)
    assert set(r) >= {"verdict", "libelle", "contrainte", "controles", "brouillons", "notes", "debit", "vitrage", "generation"}
    for c in r["controles"]:
        assert set(c) == {"id", "titre", "statut", "detail", "sources", "anomalies", "valeurs"}
        assert c["statut"] in (OK, INFO, ETUDE, HORS) and c["detail"]
        for s in c["sources"]:
            assert s["page"] in snap.pages
    assert r["libelle"] in ("Réalisable", "Sur étude", "Hors domaine d'emploi")
    # le 2425 (b = 17) au lieu du 6127 (b = 12) : DEO = 1 000 − (36 + 17)
    assert r["debit"]["deo"] == (947, 1128)


def test_les_brouillons_sont_signales(snap):
    """Les pages en brouillon que le calcul lit sont celles que ses contrôles citent, pas toutes celles du wiki."""
    r = V(snap, configuration="1v_of")
    draft = {p for p, pg in snap.pages.items() if pg.status == "draft"}
    cites = {s["page"] for c in r["controles"] for s in c["sources"]}
    assert r["brouillons"] and set(r["brouillons"]) == cites & draft
    assert f70.P + "cotes-de-debit.md" in r["brouillons"] and f70.PAGE_DTD in r["brouillons"]
    # 2008 : les pages du classeur, pas celles de 2023
    r8 = V(snap, configuration="1v_of", generation="2008")
    assert f70.P + "cotes-de-debit-ouvrants-battements-evo2008.md" in r8["brouillons"]
    assert f70.P + "cotes-de-debit.md" not in r8["brouillons"]


def test_les_sources_pointent_la_partie_de_page_qui_porte_la_section(snap):
    r = V(snap, configuration="2v_battement", battement="6130")
    src = _ctl(r, "debit")["sources"]
    pages = {s["page"] for s in src}
    assert f70.P + "cotes-de-debit.md" in pages and f70.P + "cotes-de-debit-battements.md" in pages


def test_le_registre_propose_le_systeme_70(snap):
    s = next(x for x in reg.systemes()["systemes"] if x["id"] == "systeme70")
    assert s["libelle"] == "Système 70 (PERFORM, HYBRIDE, TEXTURAL en 70)" and s["gammes"] == ["PERFORM", "HYBRIDE", "TEXTURAL"]
    assert "VER-28" in s["portee"] and s["legacy"] is False
    o = reg.options(snap, "systeme70")
    assert {c["id"] for c in o["champs"]} >= {"generation", "configuration", "largeur_mm", "hauteur_mm", "dormant", "ouvrant", "meneau",
                                              "battement", "bas", "finition", "vitrage", "classe_vent"}
    assert o["champs"][0]["id"] == "generation" and o["champs"][0]["defaut"] == "2023"
    assert set(o["configurations"]) >= {"fixe", "1v_of", "2v_meneau", "2v_battement", "porte_1v"}
    assert o["brouillons"] and any("VER-28" in a for a in o["avertissements"])
    # le formulaire dérivé des champs : une saisie vide prend les défauts, 2023 et un vantail à la française
    saisie = f70.coercer(o["champs"], {})
    assert saisie["generation"] == "2023" and saisie["configuration"] == "1v_of" and saisie["meneau"] is None
    r = reg.verifier(snap, "systeme70", {"configuration": "1v_of", "largeur_mm": 722, "hauteur_mm": 1322})
    assert r["controles"][0]["id"] == "ver28" and r["verdict"] in (OK, ETUDE)
    fiche = reg.fiche_debit(snap, "systeme70", {"configuration": "1v_of", "largeur_mm": 722, "hauteur_mm": 1322})
    assert fiche["lignes"] and fiche["nomenclature"]
    with pytest.raises(ValueError, match="Configuration"):
        reg.verifier(snap, "systeme70", {"configuration": "coulissant-2"})
    with pytest.raises(ValueError):
        reg.verifier(snap, "systeme70", {"configuration": "1v_of", "vitrage": "4-16"})


def test_api(client, admin_headers):
    s = client.get("/api/faisabilite/systemes", headers=admin_headers).json()["systemes"]
    assert "systeme70" in [x["id"] for x in s]
    o = client.get("/api/faisabilite/s/systeme70/options", headers=admin_headers)
    assert o.status_code == 200 and o.json()["champs"]
    # le navigateur envoie null pour les champs qu'il ne montre pas
    saisie = {"generation": "2023", "configuration": "1v_of", "largeur_mm": 722, "hauteur_mm": 1322, "dormant": "6101",
              "ouvrant": "6112", "meneau": None, "battement": None, "bas": "dormant", "seuil": None, "finition": "blanc",
              "vitrage": "4-16-4", "classe_vent": "va2", "renfort_2008": None, "assemblage": None, "portee_cm": None,
              "a_cm": None, "b_cm": None}
    r = client.post("/api/faisabilite/s/systeme70", headers=admin_headers, json=saisie)
    assert r.status_code == 200 and r.json()["controles"][0]["id"] == "ver28"
    d = client.post("/api/faisabilite/s/systeme70/debit", headers=admin_headers, json=saisie)
    assert d.status_code == 200 and d.json()["lignes"]
    assert client.post("/api/faisabilite/s/systeme70", headers=admin_headers, json={**saisie, "vitrage": "4-16"}).status_code == 422


# ---- la fiche de débit -----------------------------------------------------------------------


def _ligne(f, famille, piece):
    return next(l for l in f["lignes"] if l["famille"] == famille and l["piece"] == piece)


def test_fiche_de_debit_deux_vantaux_sur_meneau(snap):
    f = f70.fiche_debit(snap, S(a_cm=70, b_cm=50, portee_cm=200))
    assert f["lignes"] and f["verdict"] and set(f) >= {"lignes", "nomenclature", "accessoires", "notes", "brouillons", "verdict", "libelle"}
    # le dormant en cote finie, ses renforts : DHT − 2 × 41 (6101)
    assert (_ligne(f, "Dormant", "Traverse haute et basse")["longueur"], _ligne(f, "Dormant", "Traverse haute et basse")["qte"]) == (2000, 2)
    assert _ligne(f, "Dormant", "Montant")["longueur"] == 1200
    assert _ligne(f, "Renfort", "Renfort de dormant, traverses")["longueur"] == 1918
    assert _ligne(f, "Renfort", "Renfort de dormant, montants")["longueur"] == 1118
    # le meneau : DHT − 2 × ④ (42) ; son renfort : set d'assemblage mécanique 102, équerre 47
    assert _ligne(f, "Meneau", "Meneau")["longueur"] == 1116
    assert _ligne(f, "Renfort", "Renfort de meneau")["longueur"] == 1200 - 2 * 102
    eq = f70.fiche_debit(snap, S(assemblage="equerre"))
    assert _ligne(eq, "Renfort", "Renfort de meneau")["longueur"] == 1200 - 2 * 47
    # les ouvrants : DEO 952 × 1128 ; repère B4 (traverses renforcées) : DEO − 2 × 50
    assert _ligne(f, "Ouvrant", "Traverses")["longueur"] == 952 and _ligne(f, "Ouvrant", "Traverses")["qte"] == 4
    assert _ligne(f, "Ouvrant", "Montants")["longueur"] == 1128
    rt = _ligne(f, "Renfort", "Renfort d'ouvrant, traverses")
    assert rt["longueur"] == 852 and rt["qte"] == 4 and rt["ref"] == "V158 / V258"
    assert not [l for l in f["lignes"] if l["piece"] == "Renfort d'ouvrant, montants"]
    # le vitrage : 840 × 1016 ; les parcloses sont listées sans longueur (le wiki ne la donne pas)
    assert "840 × 1016" in _ligne(f, "Vitrage", "Vitrage d'ouvrant")["calcul"]
    p = _ligne(f, "Parclose", "Parclose d'ouvrant, horizontale")
    assert p["ref"] == "2454" and p["longueur"] is None and p["qte"] == 4
    # la nomenclature regroupe par référence ; l'accessoire n'est jamais un renfort
    assert next(n for n in f["nomenclature"] if n["ref"] == "6112")["qte"] == 8
    assert all(not a["designation"].startswith("Renfort") for a in f["accessoires"])
    assert any("cote finie" in n for n in f["notes"]) and any("dimension intérieure X" in n for n in f["notes"])
    for l in f["lignes"]:
        assert set(l) >= {"rep", "famille", "piece", "ref", "qte", "longueur", "calcul", "coupe", "source", "note", "image"}


def test_fiche_de_debit_battement_et_seuil(snap):
    f = f70.fiche_debit(snap, S(configuration="2v_battement", battement="6130", meneau=None))
    assert _ligne(f, "Battement", "Battement extérieur")["longueur"] == 1058
    assert _ligne(f, "Battement", "Battement intérieur")["longueur"] == 1116
    assert _ligne(f, "Battement", "Battement intérieur")["ref"] == "6133"
    assert any("ouvrant réduit" in n for n in f["notes"])
    f = f70.fiche_debit(snap, S(configuration="1v_of", largeur_mm=1000, hauteur_mm=1000, bas="seuil", seuil="9F67"))
    assert _ligne(f, "Seuil", "Seuil 9F67")["longueur"] == 852 and _ligne(f, "Dormant", "Montant sur seuil")["longueur"] == 993.5
    assert any("Ouvrant sur seuil" in n for n in f["notes"])
    # 2008 : 6102 a des cotes, pas en 2023
    assert "indisponible" in f70.fiche_debit(snap, S(configuration="1v_of", dormant="6102"))
    assert "lignes" in f70.fiche_debit(snap, S(configuration="1v_of", dormant="6102", generation="2008"))
