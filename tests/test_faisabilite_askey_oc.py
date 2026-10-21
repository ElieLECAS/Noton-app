"""Le calculateur ASKEY Frappe 65 Ouvrant Caché : débit des applications, zones de la ferrure Ferco, poids.

Les attendus sont les calculs d'exemple écrits dans le wiki (cotes de débit, exemples de l'abaque Ferco) et ses
tableaux : si une valeur ou un en-tête change dans le wiki, le test le dit — c'est voulu."""
from __future__ import annotations

import pytest

from app.services import faisabilite_askey_oc as oc
from app.services import faisabilite_systemes as fs
from app.services.faisabilite import DonneeIntrouvable
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _s(app, L, H, **k):
    return dict(configuration=str(app), largeur_mm=L, hauteur_mm=H, **k)


def _ctl(r, id_):
    return next(c for c in r["controles"] if c["id"] == id_)


def _lignes(f, ref):
    return [(l["qte"], l["longueur"]) for l in f["lignes"] if l["ref"] == ref]


# ---- l'évaluateur de formules ---------------------------------------------------------------

@pytest.mark.parametrize("formule, valeurs, attendu", [
    ("L-17", {"L": 1000}, 983),
    ("H -131", {"H": 1200}, 1069),
    ("H-30,5", {"H": 1200}, 1169.5),
    ("H-30.5", {"H": 1200}, 1169.5),
    ("L/2-95", {"L": 1400}, 605),
    ("L/2 -95", {"L": 1400}, 605),
    ("LCO/2-59", {"LCO": 688}, 285),
    ("(H1+H2)-21", {"H1": 1000, "H2": 300}, 1279),
    ("2H + 2 L2", {"H": 1200, "L2": 300}, 3000),
    ("L-(H-1000)", {"L": 1000, "H": 1200}, 800),
    ("−5+H", {"H": 10}, 5),
])
def test_evaluateur(formule, valeurs, attendu):
    assert oc.evaluer(formule, valeurs)[0] == pytest.approx(attendu)


def test_evaluateur_ecrit_le_calcul():
    assert oc.evaluer("LCO-59", {"LCO": 952}) == (893, "952 − 59")
    assert oc.evaluer("L/2-12", {"L": 1400}) == (688, "1400 ÷ 2 − 12")


@pytest.mark.parametrize("formule", ["__import__('os')", "L**2", "1/0", "L+", "(L", "L)", "", "H 5", "3 +* 4"])
def test_evaluateur_refuse_ce_qu_il_ne_comprend_pas(formule):
    with pytest.raises(oc.FormuleIllisible):
        oc.evaluer(formule, {"L": 1, "H": 1})


def test_evaluateur_variable_inconnue():
    with pytest.raises(oc.VariableInconnue) as e:
        oc.evaluer("X -69.5", {"L": 1, "H": 1})
    assert e.value.nom == "X"
    with pytest.raises(oc.VariableInconnue):
        oc.evaluer("L+2x", {"L": 1})        # « x » n'est pas « X » : le wiki ne définit ni l'une ni l'autre


# ---- la lecture des tableaux : le contrat avec le wiki ------------------------------------------------

def test_les_tableaux_se_lisent(snap):
    d = oc.donnees(snap)
    assert sorted(d.applications) == list(range(1, 17))
    a1 = d.applications[1]
    assert a1.titre == "Application 1 : Frappe OC 1 vantail C45 PV 28mm" and a1.page == oc.PAGES_DEBIT[0]
    assert (len(a1.profils), len(a1.joints), len(a1.accessoires), len(a1.remplissages)) == (9, 0, 11, 1)
    l = a1.profils[2]
    assert (l.ref, l.qte, l.debit, l.coupe) == ("W1010224", 2, "LCO=L-48", "45")
    assert a1.remplissages[0] == {"libelle": "Double vitrage 28mm", "qte": 1, "dimensions": "H -131 ; L -131"}
    assert a1.localisation == "schéma: raw/askey3.pdf, p. 153"      # imprimée une fois, après le dernier tableau
    # les apps 15-16 : coupes au pluriel, joints sans quantité, remplissage inversé
    a15 = d.applications[15]
    assert a15.joints[1].debit == "2H + 2 L2" and a15.joints[1].qte is None
    assert a15.remplissages[0]["dimensions"] == "L2-103 ; H-100,5"
    assert a15.profils[2].coupe == "45-90" and a15.profils[3].coupe == "90-45"
    # le Ferco
    f = d.ferco
    assert f.poids_max == {"C": 65, "C130": 105, "D": 80} and f.gache_kg == 65
    assert [z.total for z in f.zones] == [4, 6, 6, 8, 8, 10]
    assert [(z.gauche, z.droite, z.haute, z.basse) for z in f.zones][1] == (2, 2, 1, 1)
    assert f.zones[1].lff == (900, True, 1250, False) and f.zones[1].hff == (750, True, 1000, False)
    assert f.diagonale == ((500, 300), (1400, 1250)) and f.precision_mm == 50
    assert d.kg_par_m2_mm == 2.5
    assert d.images["1010132"].endswith("dormant-w1010132.png")


def test_le_total_de_points_est_la_somme_des_points(snap):
    for z in oc.donnees(snap).ferco.zones:
        assert z.total == z.gauche + z.droite + z.haute + z.basse, z.numero


def test_les_en_tetes_changes_cassent_le_calcul(snap):
    for page, avant, apres in [
        (oc.PAGES_DEBIT[0], "| Référence | Désignation | Qté | Débit (mm) | Coupe (°) |", "| Référence | Désignation | Qté | Longueur (mm) | Coupe (°) |"),
        (oc.PAGES_DEBIT[0], "| Remplissage | Qté | Dimensions (mm) |", "| Remplissage | Quantité | Dimensions (mm) |"),
        (oc.PAGE_FERCO, "Points sur le montant de gauche", "Points à gauche"),
        (oc.PAGE_FERCO, "| Exécution Unijet | Poids maxi vantail (kg) |", "| Unijet | Poids maxi vantail (kg) |"),
        (oc.PAGE_FERCO, "500 ≤ Lff < 900", "de 500 à 900"),
        (oc.PAGE_FERCO, "précision de lecture ±50 mm", "précision de lecture 50 mm"),
    ]:
        p = snap.pages[page]
        original = p.body
        assert avant in original, avant
        p.body = original.replace(avant, apres)
        try:
            with pytest.raises(DonneeIntrouvable):
                oc._charge(snap)
        finally:
            p.body = original


# ---- le débit : les oracles du wiki ------------------------------------------------------------------

def test_application_1_l1000_h1200(snap):
    f = fs.fiche_debit(snap, "askey-oc", _s(1, 1000, 1200))
    assert _lignes(f, "W1010132") == [(2, 1000), (2, 1200)]
    assert _lignes(f, "W1010224") == [(2, 952), (2, 1152)]
    assert _lignes(f, "W4033002") == [(2, 893), (2, 1093)]
    assert _lignes(f, "W4000169") == [(1, 983)]
    assert _lignes(f, "W4000166") == [(1, 987), (2, 1225)]
    vitrage = next(l for l in f["lignes"] if l["famille"] == "Vitrage")
    assert vitrage["note"] == "hauteur × largeur : 1069 × 869 mm" and vitrage["qte"] == 1
    r = fs.verifier(snap, "askey-oc", _s(1, 1000, 1200))
    assert r["debit"]["variables"] == {"LCO": 952, "HCO": 1152}
    assert r["debit"]["remplissages"] == [{"libelle": "Double vitrage 28mm", "qte": 1, "hauteur": 1069, "largeur": 869}]


def test_application_3_l1400_h1200(snap):
    f = fs.fiche_debit(snap, "askey-oc", _s(3, 1400, 1200))
    r = fs.verifier(snap, "askey-oc", _s(3, 1400, 1200))
    assert r["debit"]["variables"] == {"LCO": 688, "HCO": 1152}
    assert _lignes(f, "W1010303") == [(4, 688), (3, 1152)]
    assert _lignes(f, "W1010417") == [(1, 1150)]
    assert _lignes(f, "W4033002") == [(4, 629), (4, 1093)]
    assert _lignes(f, "Y3710012") == [(1, 1038)]
    assert r["debit"]["remplissages"] == [{"libelle": "Double vitrage 28mm", "qte": 2, "hauteur": 1069, "largeur": 605}]


def test_application_2_fixe(snap):
    f = fs.fiche_debit(snap, "askey-oc", _s(2, 1000, 1200))
    assert _lignes(f, "W3090174") == [(2, 940), (2, 1058)]       # L-60, H-142
    assert _lignes(f, "W3090161") == [(1, 986)]                  # L-14
    assert _lignes(f, "W8020003") == [(1, 900)]                  # L-100
    assert _lignes(f, "W4010003") == [(2, 1000), (2, 1200)]      # joints : L et H, sans retrait
    assert next(l for l in f["lignes"] if l["famille"] == "Vitrage")["note"] == "hauteur × largeur : 1086 × 886 mm"
    r = fs.verifier(snap, "askey-oc", _s(2, 1000, 1200))
    assert _ctl(r, "ferco")["statut"] == "info" and not any(c["id"] == "poids" for c in r["controles"])


def test_applications_12_a_14(snap):
    f = fs.fiche_debit(snap, "askey-oc", _s(12, 1000, 1200))
    assert _lignes(f, "W1715147") == [(2, 1050), (2, 1250)]      # L+50, H+50
    assert _lignes(f, "W4030328") == [(2, 893), (2, 1093)]
    assert _lignes(f, "W4010473") == [(2, 1152), (2, 952)]       # joints HCO, LCO
    f = fs.fiche_debit(snap, "askey-oc", _s(13, 1400, 1200))
    assert _lignes(f, "W1010417") == [(1, 1152)]                 # H-48 (H-50 à l'application 3)
    assert _lignes(f, "W3090223") == [(1, 1038)]
    assert _lignes(f, "W4030328") == [(4, 629), (4, 1093)]
    f = fs.fiche_debit(snap, "askey-oc", _s(14, 1000, 1200))
    assert _lignes(f, "W3090173") == [(2, 940), (2, 1058)]
    assert next(l for l in f["lignes"] if l["famille"] == "Vitrage")["note"] == "hauteur × largeur : 1086 × 886 mm"


def test_un_cote_negatif_n_est_pas_calcule(snap):
    """Un châssis trop petit pour une formule : la ligne n'est pas calculée, et le verdict le dit."""
    r = fs.verifier(snap, "askey-oc", _s(1, 300, 300))
    assert r["verdict"] in ("hors", "etude")
    calc = oc.calcul_application(oc.donnees(snap).applications[1], 50, 50)
    assert any("négative" in (x["erreur"] or "") for x in calc["profils"])


def test_nomenclature_regroupee_par_reference(snap):
    f = fs.fiche_debit(snap, "askey-oc", _s(1, 1000, 1200))
    n = next(x for x in f["nomenclature"] if x["ref"] == "W1010132")
    assert n["qte"] == 4 and n["ml"] == 4.4 and n["famille"] == "Profilés" and n["image"]
    assert any(x["designation"] == "Double vitrage 28mm" and x["qte"] == 1 for x in f["nomenclature"])
    assert any(a["ref"] == "W4080313" and a["qte"] == 6 and a["profil"] == "qté 6" for a in f["accessoires"])
    assert any(a["ref"] == "W0198133" and a["qte"] is None and a["profil"] == "qté 3/ml" for a in f["accessoires"])


# ---- les applications laissées sur étude -----------------------------------------------------------------

@pytest.mark.parametrize("app, morceau", [
    (4, "X, Y"), (5, "x"), (7, "H1, H2, H3"), (10, "L2"), (15, "LCO"), (16, "L1"),     # variables que le wiki ne définit pas
    (6, "INC-790, VER-827"), (8, "INC-791"), (11, "VER-828, VER-827"),                  # cotes contestées par une anomalie
])
def test_applications_sur_etude_avec_la_raison(snap, app, morceau):
    r = fs.verifier(snap, "askey-oc", _s(app, 1000, 1200))
    assert r["verdict"] == "etude" and [c["id"] for c in r["controles"]] == ["debit"]
    assert morceau in _ctl(r, "debit")["detail"]
    assert "Calculées : 1, 2, 3, 12, 13, 14" in _ctl(r, "debit")["detail"]
    f = fs.fiche_debit(snap, "askey-oc", _s(app, 1000, 1200))
    assert list(f) == ["indisponible"] and morceau in f["indisponible"]


def test_les_applications_calculees_sont_exactement_celles_du_contrat(snap):
    d = oc.donnees(snap)
    assert [n for n, a in d.applications.items() if oc.raison_sur_etude(a) is None] == [1, 2, 3, 12, 13, 14]
    # et ces six-là n'ont aucune variable que le wiki ne définisse pas : le calcul est complet
    for n in oc.APPLICATIONS_CALCULABLES:
        assert oc.variables_manquantes(d.applications[n]) == []
        calc = oc.calcul_application(d.applications[n], 1200, 1300)
        assert oc._lignes_en_erreur(calc) == [], n


def test_les_options_disent_ce_qui_est_sur_etude(snap):
    o = fs.options(snap, "askey-oc")
    assert len(o["configurations"]) == 16
    assert o["configurations"]["1"] == "Application 1 : Frappe OC 1 vantail C45 PV 28mm"
    assert o["configurations"]["4"].endswith(" — sur étude")


# ---- la ferrure Ferco : les zones par les inéquations de la planche -----------------------------------------

@pytest.mark.parametrize("lff, hff, statut, zone, points", [
    (800, 1100, "ok", 3, 6),        # l'exemple imprimé : « 800 x 1100 : Conforme »
    (1380, 800, "hors", None, None),   # l'exemple imprimé : « 1380 x 800 : Non conforme »
    (900, 1000, "ok", 4, 8),        # dérivé : la borne basse de Hff 1000 est incluse, la borne haute de zone 2 exclue
    (1200, 1800, "hors", None, None),
    (500, 500, "ok", 1, 4),         # coin bas-gauche de la zone 1 : bornes basses incluses
    (600, 700, "ok", 1, 4),
    (1000, 1700, "hors", None, None),   # 1 000 ≤ Lff n'est dans aucune zone haute (strict)
    (999, 1700, "ok", 6, 10),
    (700, 1700, "ok", 5, 8),
    (1399, 1500, "ok", 4, 8),
    (1400, 1100, "hors", None, None),   # Lff < 1 400 : strict
    (400, 1100, "hors", None, None),
    (800, 2200, "hors", None, None),    # Hff < 2 200 : strict
])
def test_zones_ferco(snap, lff, hff, statut, zone, points):
    z = oc.zone_ferco(snap, lff, hff)
    assert (z["statut"], z["zone"], z["points"]) == (statut, zone, points), z["detail"]


def test_zone_2_au_bord_litigieux_sur_etude(snap):
    """1 100 × 900 : zone 2 par les inéquations, mais la diagonale du graphique passe à 933 mm de Hff à cette
    largeur : le point est à 33 mm dessous, dans la précision de lecture (±50 mm) — VER-817."""
    z = oc.zone_ferco(snap, 1100, 900)
    assert z["statut"] == "etude" and z["zone"] == 2 and z["points"] == 6
    assert z["ecart_diagonale_mm"] == pytest.approx(-33.3, abs=0.1)
    assert "la planche se contredit" in z["detail"]
    # au-dessus de la diagonale de plus de la précision de lecture : la zone 2 est confirmée
    z = oc.zone_ferco(snap, 1100, 990)
    assert z["zone"] == 2 and z["statut"] == "ok" and z["ecart_diagonale_mm"] == pytest.approx(56.7, abs=0.1)
    z = oc.zone_ferco(snap, 1000, 999)
    assert z["zone"] == 2 and z["statut"] == "ok" and z["ecart_diagonale_mm"] > 50


def test_diagonale_contredit_les_inequations_loin_du_bord(snap):
    """(1 200, 800) : zone 2 par les inéquations, 239 mm sous la diagonale : le graphique la contredit, sur étude."""
    z = oc.zone_ferco(snap, 1200, 800)
    assert z["zone"] == 2 and z["statut"] == "etude" and z["ecart_diagonale_mm"] < -200


def test_zone_ferco_dans_le_verdict(snap):
    # application 1, L 1 000 × H 1 200 : LCO 952, HCO 1 152 → zone 4, 8 points
    r = fs.verifier(snap, "askey-oc", _s(1, 1000, 1200))
    c = _ctl(r, "ferco")
    assert c["statut"] == "ok" and c["valeurs"]["zone"] == 4 and c["valeurs"]["points"] == 8
    assert (c["valeurs"]["lff"], c["valeurs"]["hff"]) == (952, 1152)
    assert "hypothèse : Lff / Hff = LCO / HCO" in c["detail"] and c["valeurs"]["hypothese"].startswith("hypothèse")
    # application 3, L 1 400 × H 1 200 : LCO 688, HCO 1 152 → zone 3, 6 points
    c = _ctl(fs.verifier(snap, "askey-oc", _s(3, 1400, 1200)), "ferco")
    assert c["valeurs"]["zone"] == 3 and c["valeurs"]["points"] == 6
    # hors zone : le verdict est « hors domaine »
    r = fs.verifier(snap, "askey-oc", _s(1, 2000, 1200))      # LCO 1 952 : plus large que toute zone
    assert _ctl(r, "ferco")["statut"] == "hors" and r["verdict"] == "hors" and r["contrainte"]["id"] == "ferco"


def test_lff_hff_mesures_remplacent_l_hypothese(snap):
    r = fs.verifier(snap, "askey-oc", _s(1, 1000, 1200, lff_mm=800, hff_mm=1100))
    c = _ctl(r, "ferco")
    assert c["valeurs"]["zone"] == 3 and "hypothèse" not in c["detail"] and "hypothese" not in c["valeurs"]
    # un seul mesuré : l'autre reste une hypothèse, dite
    c = _ctl(fs.verifier(snap, "askey-oc", _s(1, 1000, 1200, lff_mm=800)), "ferco")
    assert (c["valeurs"]["lff"], c["valeurs"]["hff"]) == (800, 1152) and "Hff non mesuré" in c["detail"]


def test_executions_c130_et_d_sur_etude(snap):
    """L'abaque est imprimé pour l'exécution C seulement (VER-818) : aucune zone pour C130 ni D."""
    for ex in ("C130", "D"):
        r = fs.verifier(snap, "askey-oc", _s(1, 1000, 1200, execution=ex))
        c = _ctl(r, "ferco")
        assert c["statut"] == "etude" and "VER-818" in c["anomalies"] and r["verdict"] == "etude"


# ---- le poids de vantail ----------------------------------------------------------------------------------

def test_poids_sans_verre_informe(snap):
    c = _ctl(fs.verifier(snap, "askey-oc", _s(1, 1000, 1200)), "poids")
    assert c["statut"] == "info" and c["valeurs"]["poids_max_kg"] == 65 and "gâche releveur" in c["detail"]


def test_poids_du_verre(snap):
    # 8 mm × 2,5 kg/m²/mm × 1,069 × 0,869 m = 18,6 kg pour 65 kg maxi
    c = _ctl(fs.verifier(snap, "askey-oc", _s(1, 1000, 1200, verre_mm=8)), "poids")
    assert c["statut"] == "ok" and c["valeurs"]["poids_kg"] == 18.6 and c["valeurs"]["poids_max_kg"] == 65
    assert "hypothèse" in c["detail"] and {"CTR-286", "VER-818"} <= set(c["anomalies"])
    # 30 mm : 69,7 kg, au-delà de 65 kg en exécution C (hors) ; C130 (105 kg) le tient, avec la gâche releveur
    r = fs.verifier(snap, "askey-oc", _s(1, 1000, 1200, verre_mm=30))
    c = _ctl(r, "poids")
    assert c["statut"] == "hors" and c["valeurs"]["poids_kg"] == 69.7 and "gâche releveur" in c["detail"]
    assert r["verdict"] == "hors"
    c = _ctl(fs.verifier(snap, "askey-oc", _s(1, 1000, 1200, verre_mm=30, execution="C130")), "poids")
    assert c["statut"] == "ok" and "gâche releveur" in c["detail"]
    # deux vantaux : le poids est celui d'un vantail (1 069 × 605)
    c = _ctl(fs.verifier(snap, "askey-oc", _s(3, 1400, 1200, verre_mm=8)), "poids")
    assert c["valeurs"]["poids_kg"] == 12.9


# ---- la forme du résultat, les anomalies, les brouillons ---------------------------------------------------

def test_forme_du_resultat(snap):
    r = fs.verifier(snap, "askey-oc", _s(1, 1000, 1200, verre_mm=8))
    assert {"verdict", "libelle", "contrainte", "controles", "brouillons", "notes", "debit"} <= set(r)
    assert r["verdict"] == "ok" and r["libelle"] == "Réalisable" and r["contrainte"] is None
    assert [c["id"] for c in r["controles"]] == ["debit", "ferco", "poids"]
    for c in r["controles"]:
        assert set(c) == {"id", "titre", "statut", "detail", "sources", "anomalies", "valeurs"}
        assert c["statut"] in ("ok", "hors", "etude", "info") and c["sources"]
        for s in c["sources"]:
            assert s["page"] in snap.pages
    # toutes les pages ASKEY sont en brouillon : le verdict le dit
    assert set(r["brouillons"]) == {oc.PAGES_DEBIT[0], oc.PAGE_FERCO, "/quincaillerie/roto-nx-champs-application.md"}
    assert any("Parcloses" in n for n in r["notes"])
    # l'application 13 vit sur une autre page de débit
    r = fs.verifier(snap, "askey-oc", _s(13, 1400, 1200))
    assert oc.PAGES_DEBIT[4] in r["brouillons"] and oc.PAGES_DEBIT[0] not in r["brouillons"]
    assert _ctl(r, "debit")["sources"][0]["page"] == oc.PAGES_DEBIT[4]


def test_anomalies_de_l_application_affichees(snap):
    assert _ctl(fs.verifier(snap, "askey-oc", _s(1, 1000, 1200)), "debit")["anomalies"] == ["INC-800", "VER-834"]
    assert _ctl(fs.verifier(snap, "askey-oc", _s(3, 1400, 1200)), "debit")["anomalies"] == ["VER-830", "VER-834"]
    assert set(_ctl(fs.verifier(snap, "askey-oc", _s(13, 1400, 1200)), "debit")["anomalies"]) == {"INC-794", "VER-828", "VER-834"}
    assert set(_ctl(fs.verifier(snap, "askey-oc", _s(14, 1000, 1200)), "debit")["anomalies"]) == {"INC-795", "VER-831", "VER-837", "VER-834"}
    z = _ctl(fs.verifier(snap, "askey-oc", _s(1, 1000, 1200)), "ferco")
    assert set(z["anomalies"]) == {"VER-817", "VER-818", "CTR-287"}


def test_fiche_de_debit_forme(snap):
    f = fs.fiche_debit(snap, "askey-oc", _s(1, 1000, 1200))
    assert {"lignes", "nomenclature", "accessoires", "notes", "brouillons", "verdict", "libelle"} <= set(f)
    assert f["verdict"] == "ok" and f["brouillons"]
    for l in f["lignes"]:
        assert {"rep", "famille", "piece", "ref", "qte", "longueur", "calcul", "coupe", "source", "note", "image"} <= set(l)
        assert l["source"]["page"] in snap.pages and l["source"]["section"]
    assert [l["rep"] for l in f["lignes"]] == list(range(1, len(f["lignes"]) + 1))
    w = next(l for l in f["lignes"] if l["ref"] == "W4033002" and l["longueur"] == 893)
    assert w["calcul"] == "LCO-59 : 952 − 59" and w["coupe"] == "45°" and w["source"]["section"] == "Profilés de l'application 1"
    assert next(l for l in f["lignes"] if l["ref"] == "W1010224" and l["longueur"] == 952)["calcul"] == "LCO = 1000 − 48"
    assert any("surlongueur" in n for n in f["notes"]) and any("INC-800" in n for n in f["notes"])


# ---- la porte commune des calculateurs ----------------------------------------------------------------------

def test_par_le_registre(snap):
    liste = fs.systemes()
    s = next(x for x in liste["systemes"] if x["id"] == "askey-oc")
    assert s["libelle"] == "ASKEY Frappe 65 Ouvrant Caché" and s["gammes"] == ["Frappe 65 Ouvrant Caché"] and s["legacy"] is False
    assert s["portee"]
    assert any(x["id"] == "askey-ov" for x in liste["indisponibles"])      # l'ouvrant visible : aucun calcul
    o = fs.options(snap, "askey-oc")
    assert {"champs", "configurations", "pages", "brouillons", "avertissements"} <= set(o)
    ids = [c["id"] for c in o["champs"]]
    assert {"configuration", "largeur_mm", "hauteur_mm", "execution", "verre_mm"} <= set(ids) and len(ids) == len(set(ids))
    brut = {c["id"]: c.get("defaut") for c in o["champs"]}
    r = fs.verifier(snap, "askey-oc", brut)
    assert r["verdict"] == "ok" and r["controles"]
    assert "lignes" in fs.fiche_debit(snap, "askey-oc", brut)


def test_saisie_invalide(snap):
    with pytest.raises(ValueError):
        fs.verifier(snap, "askey-oc", _s(17, 1000, 1200))
    with pytest.raises(ValueError):
        fs.verifier(snap, "askey-oc", _s(1, "large", 1200))
    with pytest.raises(ValueError):
        fs.verifier(snap, "askey-oc", _s(1, 1000, 1200, execution="E"))
