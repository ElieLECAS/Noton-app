"""Le calculateur du LUMEAL GA : évaluateur de formules, tableaux du wiki, DTA, courbes, débit.

Les attendus sont ceux des exemples écrits dans le wiki ou recalculés à la main depuis ses tableaux : si une
valeur ou un en-tête change dans le wiki, un test le dit — c'est voulu."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services import faisabilite_ga as ga
from app.services import faisabilite_systemes as reg
from app.services.faisabilite import DonneeIntrouvable, ETUDE, HORS, INFO, OK, _corps
from app.services.faisabilite_ga import FormuleIllisible, evaluer, lire_sur_courbe, paire_de
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _ctl(r, id_):
    return next(c for c in r["controles"] if c["id"] == id_)


def _fiche_ligne(f, ref, longueur=None):
    return [l for l in f["lignes"] if l["ref"] == ref and (longueur is None or l["longueur"] == longueur)]


FENETRE = {"configuration": "fenetre2", "largeur_mm": 1800, "hauteur_mm": 1250}


# ---- l'évaluateur de formules : une grammaire, jamais eval -----------------------------------


@pytest.mark.parametrize("formule, variables, attendu", [
    ("H", {"H": 1250}, 1250),
    ("L/2-65", {"L": 1800}, 835),
    ("L - 42", {"L": 1800}, 1758),
    ("L/3 - 177", {"L": 2400}, 623),
    ("L-Lf-241", {"L": 2400, "Lf": 800}, 1359),
    ("2L/3-370", {"L": 2400}, 1230),
    ("6H+2L", {"H": 1250, "L": 1800}, 11100),
    ("4H+1L", {"H": 1250, "L": 1800}, 6800),
    ("6H+4L/3", {"H": 1250, "L": 1800}, 9900),
    ("(L-Lf)/300", {"L": 2400, "Lf": 900}, 5),
    ("(2+L/330)", {"L": 1650}, 7),
    ("3+Lf/300", {"Lf": 900}, 6),
    ("2L/900", {"L": 1800}, 4),
    ("(L1-6)+72", {"L1": 1000}, 1066),
    ("(2L1-116)+12,5", {"L1": 1000}, 1896.5),
    ("3L/4-99.5", {"L": 2000}, 1400.5),
    ("(6H+3L1+3L2)/330", {"H": 1100, "L1": 330, "L2": 330}, 26),
    ("2 x (H - 19)", {"H": 1000}, 1962),
    ("−L/2", {"L": 100}, -50),                      # le signe moins typographique
])
def test_l_evaluateur_lit_les_formules_du_wiki(formule, variables, attendu):
    assert evaluer(formule, variables).valeur == pytest.approx(attendu)


def test_la_lettre_de_coupe_est_separee_du_calcul():
    e = evaluer("L/2-55(AB)", {"L": 1800})
    assert (e.valeur, e.suffixe) == (845, "AB")       # VER-459 : sans légende, rendue telle quelle, jamais calculée
    e = evaluer("L1/2-43.5(A'1)", {"L1": 1000})
    assert (e.valeur, e.suffixe) == (456.5, "A'1")
    # même quand la lettre porte le nom d'une variable : « 3L/4-179.5(H) » (GY) est une lettre de coupe, pas un produit
    e = evaluer("3L/4-179.5(H)", {"H": 10, "L": 1000})
    assert (e.valeur, e.suffixe) == (570.5, "H")


@pytest.mark.parametrize("formule", [
    "4H+2l",                                        # INC-466 : « l » minuscule, variable inconnue
    "Voir environnement pose",                      # du texte, pas une formule
    "__import__('os').system('x')",                 # jamais eval
    "H+",                                           # incomplète
    "(H-2",                                         # parenthèse non fermée
    "H/0",                                          # division par zéro
    "",
])
def test_une_formule_incomprise_est_refusee(formule):
    with pytest.raises(FormuleIllisible):
        evaluer(formule, {"H": 1000, "L": 1000})


# ---- la lecture des tableaux : le contrat avec le wiki -----------------------------------------


def test_les_tableaux_se_lisent(snap):
    d = ga.donnees(snap)
    # DTA : une ligne par paire de montants et par recouvrement, en mm
    l = next(x for x in d.dta["mini"] if x.paire == ("141037", "141037"))
    assert (l.libelle, l.h_mm, l.l_mm) == ("2 x 141037", 1550, 1900)
    l = next(x for x in d.dta["maxi"] if x.paire == ("141037", "141037"))
    assert (l.h_mm, l.l_mm) == (1450, 1800)
    assert next(x for x in d.dta["mini"] if x.paire == ("141024", "141024")).l_mm == 2900
    assert len(d.dta["mini"]) == len(d.dta["maxi"]) == 12
    assert d.dta_vitrages == [24, 26, 28, 30, 32] and d.dta_validite == "27/05/2024 au 31/07/2029"
    # courbes : six paires par abaque, la septième (percussion centrale) n'est pas une paire
    assert set(d.courbes) == {"A2", "A3"} and len(d.courbes["A2"]["colonnes"]) == 6
    assert d.courbes["A2"]["colonnes"][("141037", "141037")][1500] == 2185
    assert d.courbes["A3"]["colonnes"][("141024", "141024")][2500] == 2910  # « 141024 » sans T : INC-444, même profilé
    assert d.courbes["A2"]["colonnes"][("141024", "141037")][1000] is None
    assert d.limites == {"kg": 300, "demi_perimetre": 5000, "precision": 50, "cadre": 3000}
    assert d.densite_verre == 2.5
    # nomenclatures : huit applications, quatre tableaux et un remplissage chacune
    assert set(d.applications) == set(ga.APPLICATIONS)
    fen = d.applications["fenetre2"]
    assert [l.ref for l in fen.profils][:3] == ["T141001", "T141015", "T141017"]
    assert fen.remplissages == [{"libelle": "", "hauteur": "H-143", "largeur": "L/2-66"}]
    assert fen.anomalies == ["INC-456", "VER-236"]
    assert [r["libelle"] for r in d.applications["pf1_fixe"].remplissages] == ["le vantail", "la partie fixe"]
    assert d.plages_traverses["T141015"] == (24, 28) and d.plages_traverses["T141021"] == (29, 32)
    assert d.joints_volume[24] == {"ref": "T411005", "couleur": "rouge"} and d.joints_volume[32]["ref"] == "T411009"
    assert d.roulettes["T401011"]["poids"] == 100 and d.roulettes["TGA3608"]["poids"] == 300
    assert d.images["T141015"].endswith("ouvrant-t141015.png")


def test_les_en_tetes_de_courbes_nomment_leur_paire():
    assert paire_de("H hors tout maximal (mm), vent 1200 Pa, courbe n / 6 (T141024 + 141024)") == ("141024", "141024")
    assert paire_de("H hors tout maximal (mm), vent 800 Pa, courbe n / 2 (T141037 + T141038)") == ("141037", "141038")
    assert paire_de("Courbe 4 : TGY1207 / TGY1209") == ("TGY1207", "TGY1209")
    assert paire_de("H hors tout maximal (mm), vent 800 Pa, courbe n / 7 (percussion centrale)") is None


def test_un_en_tete_change_dans_le_wiki_casse_la_lecture(snap):
    corps = _corps(snap, ga.PAGE_DEBIT)
    ok = ga.lire_nomenclature(corps, "Fenêtre 2 vantaux", ga.PAGE_DEBIT)
    assert len(ok.profils) == 12
    casse = corps.replace("| Référence | Désignation | Quantité | Débit (mm) |",
                          "| Référence | Désignation | Quantité | Longueur (mm) |")
    with pytest.raises(DonneeIntrouvable):
        ga.lire_nomenclature(casse, "Fenêtre 2 vantaux", ga.PAGE_DEBIT)
    with pytest.raises(DonneeIntrouvable):
        ga.lire_nomenclature(corps.replace("L/2-66", "?"), "Fenêtre 2 vantaux", ga.PAGE_DEBIT)


# ---- la courbe, sans interpolation ---------------------------------------------------------------


def test_courbe_a_la_plus_restrictive_des_deux_graduations(snap):
    """Courbe 1 A2, L = 1800 : graduations 1500 (2185) et 2000 (2020) → 2020, précision ±50."""
    c = ga.donnees(snap).courbes["A2"]["colonnes"][("141037", "141037")]
    r = lire_sur_courbe(c, 1800, 1250, 3000, 50)
    assert (r.statut, r.h_max) == (OK, 2020)
    assert lire_sur_courbe(c, 1800, 1970, 3000, 50).statut == OK          # 2020 - 50 : la limite basse de la zone grise
    assert lire_sur_courbe(c, 1800, 2020, 3000, 50).statut == ETUDE       # sur la limite : sur étude
    assert lire_sur_courbe(c, 1800, 2069, 3000, 50).statut == ETUDE
    assert lire_sur_courbe(c, 1800, 2071, 3000, 50).statut == HORS
    assert lire_sur_courbe(c, 2000, 2000, 3000, 50).h_max == 2020         # sur une graduation : la valeur lue


def test_courbe_au_dessus_du_cadre_ou_arretee(snap):
    d = ga.donnees(snap)
    c3 = d.courbes["A2"]["colonnes"][("141024", "141037")]       # courbe 3 : « - » à L = 1000 et 1500
    assert lire_sur_courbe(c3, 1000, 2900, 3000, 50).statut == OK       # au-dessus du cadre : elle ne limite pas
    assert lire_sur_courbe(c3, 1000, 2960, 3000, 50).statut == ETUDE    # au bord du cadre : précision de lecture
    assert lire_sur_courbe(c3, 1250, 2900, 3000, 50).statut == OK       # deux graduations au-dessus du cadre
    mixte = lire_sur_courbe(c3, 1750, 2800, 3000, 50)                   # 1500 au-dessus du cadre, 2000 : 2935
    assert mixte.h_max == 2935 and mixte.statut == OK
    c4 = d.courbes["A2"]["colonnes"][("141038", "141038")]       # courbe 4 : arrêtée avant 4500
    assert lire_sur_courbe(c4, 4000, 2000, 3000, 50).statut == OK
    arretee = lire_sur_courbe(c4, 4500, 2000, 3000, 50)
    assert arretee.statut == ETUDE and arretee.h_max is None and "arrêtée" in arretee.detail
    assert lire_sur_courbe(c4, 4250, 2000, 3000, 50).statut == ETUDE
    # hors des graduations de l'abaque : jamais extrapolé
    assert lire_sur_courbe(c4, 900, 2000, 3000, 50).statut == ETUDE
    assert lire_sur_courbe(c4, 5000, 2000, 3000, 50).statut == ETUDE


# ---- les verdicts ---------------------------------------------------------------------------------


def test_fenetre_courante_realisable(snap):
    r = ga.verifier(snap, FENETRE)
    assert r["verdict"] == OK, [(c["id"], c["statut"], c["detail"]) for c in r["controles"]]
    assert [c["id"] for c in r["controles"]] == ["dta", "courbe", "poids", "demi_perimetre", "vitrage", "roulettes", "cale"]
    assert r["vitrage"] == {"epaisseur_mm": 24, "verre_mm": 8.0, "joint": "T411005"}
    assert r["remplissages"] == [{"libelle": "", "hauteur": 1107, "largeur": 834}]   # H-143 × L/2-66
    assert r["poids_verre_kg"] == pytest.approx(18.5, abs=0.1)


def test_la_forme_du_resultat(snap):
    r = ga.verifier(snap, FENETRE)
    assert {"verdict", "libelle", "contrainte", "controles", "brouillons"} <= set(r)
    for c in r["controles"]:
        assert set(c) == {"id", "titre", "statut", "detail", "sources", "anomalies", "valeurs"}
        assert c["statut"] in (OK, HORS, ETUDE, INFO) and c["detail"]
        for s in c["sources"]:
            assert set(s) == {"page", "section", "localisation"} and s["page"] in snap.pages
    assert r["contrainte"] is None and r["libelle"] == "Réalisable"


def test_dta_oracle_recouvrement_mini_et_maxi(snap):
    """2 x 141037 : recouvrement mini 1,55 × 1,90 m, maxi 1,45 × 1,80 m (H × L)."""
    def dta(h, l, rec):
        return _ctl(ga.verifier(snap, dict(FENETRE, hauteur_mm=h, largeur_mm=l, recouvrement=rec)), "dta")["statut"]
    assert dta(1500, 1850, "mini") == OK
    assert dta(1500, 2000, "mini") == HORS
    assert dta(1550, 1900, "mini") == OK          # la limite exacte est réalisable (≤)
    assert dta(1560, 1900, "mini") == HORS
    assert dta(1450, 1800, "maxi") == OK          # L = 1,80 m : limite exacte du DTA
    assert dta(1450, 1810, "maxi") == HORS
    # recouvrement non précisé : dans les deux, entre les deux, au-delà des deux
    assert dta(1400, 1800, "inconnu") == OK
    assert dta(1500, 1850, "inconnu") == ETUDE
    assert dta(1600, 1850, "inconnu") == HORS


def test_dta_affiche_sa_validite_son_hypothese_et_ses_anomalies(snap):
    c = _ctl(ga.verifier(snap, FENETRE), "dta")
    assert "27/05/2024 au 31/07/2029" in c["detail"] and ga.HYPOTHESE_BAIE in c["detail"]
    assert c["anomalies"] == ["VER-169", "CTR-135"] and c["sources"][0]["page"] == ga.PAGE_DTA
    assert c["valeurs"]["limites"]["mini"] == {"h": 1550, "l": 1900}


def test_montants_choisis_et_paire_inconnue(snap):
    c = _ctl(ga.verifier(snap, dict(FENETRE, montants="141037+141038", hauteur_mm=2100, largeur_mm=2200,
                                    recouvrement="mini")), "dta")
    assert c["statut"] == OK and c["valeurs"]["limites"]["mini"] == {"h": 2150, "l": 2300}
    # 141037 + 141041 : une paire qui n'est dans aucun tableau n'est pas proposée
    with pytest.raises(ValueError, match="Montants centraux"):
        ga.verifier(snap, dict(FENETRE, montants="141037+141041"))
    # la paire de la famille 30-32 mm est au DTA, pas sur les courbes
    r = ga.verifier(snap, dict(FENETRE, montants="141041+141043", hauteur_mm=2100, largeur_mm=2200, recouvrement="mini"))
    assert _ctl(r, "dta")["statut"] == OK and _ctl(r, "courbe")["statut"] == ETUDE


def test_courbe_a2_et_a3_et_son_anomalie(snap):
    """Courbe 1, L = 1800 : 2020 mm en A2 ; en A3 les graduations donnent 1920 (1500) et 1780 (2000) → 1780."""
    a2 = _ctl(ga.verifier(snap, FENETRE), "courbe")
    assert a2["statut"] == OK and a2["valeurs"]["h_max"] == 2020 and a2["anomalies"] == []
    a3 = _ctl(ga.verifier(snap, dict(FENETRE, vent="A3", hauteur_mm=1750)), "courbe")
    assert a3["valeurs"]["h_max"] == 1780 and a3["statut"] == ETUDE      # 1750 est à moins de 50 mm de 1780
    assert _ctl(ga.verifier(snap, dict(FENETRE, vent="A3", hauteur_mm=1800)), "courbe")["statut"] == ETUDE
    assert _ctl(ga.verifier(snap, dict(FENETRE, vent="A3", hauteur_mm=1900)), "courbe")["statut"] == HORS
    # la courbe 6 écrit « 141024 » sans T : INC-444 s'affiche
    c6 = _ctl(ga.verifier(snap, dict(FENETRE, montants="141024+141024", largeur_mm=3500, hauteur_mm=2000)), "courbe")
    assert c6["anomalies"] == ["INC-444"] and c6["statut"] == OK


def test_vitrage_et_joint(snap):
    def vit(**k):
        return _ctl(ga.verifier(snap, dict(FENETRE, **k)), "vitrage")
    assert vit(epaisseur_vitrage_mm="28")["statut"] == OK and vit(epaisseur_vitrage_mm="28")["valeurs"]["joint"] == "T411007"
    c = vit(epaisseur_vitrage_mm="25")                      # 25 mm : au catalogue seulement, pas au DTA (CTR-134)
    assert c["statut"] == ETUDE and "CTR-134" in c["anomalies"] and "TGY5009" in c["detail"]
    c = vit(epaisseur_vitrage_mm="30")                      # T141037 est le montant central de 24 à 28 mm
    assert c["statut"] == ETUDE and "T141037 (24 à 28 mm)" in c["detail"]
    c = vit(epaisseur_vitrage_mm="30", montants="141041+141043")   # la famille 30-32 mm : la fenêtre est faisable
    assert c["statut"] == OK and "T141015" in c["detail"] and "liste de coupe n'est pas établie" in c["detail"]
    assert vit(epaisseur_vitrage_mm="24", verre_mm=30)["statut"] == ETUDE
    # la porte-fenêtre RC3 porte la traverse 29 à 32 mm
    rc3 = ga.verifier(snap, dict(FENETRE, configuration="pf2_rc3", epaisseur_vitrage_mm="32"))
    assert _ctl(rc3, "vitrage")["statut"] == OK and _ctl(rc3, "vitrage")["valeurs"]["joint"] == "T411009"
    assert _ctl(ga.verifier(snap, dict(FENETRE, configuration="pf2_rc3")), "vitrage")["statut"] == ETUDE


def test_charge_et_demi_perimetre_ne_disent_jamais_oui(snap):
    r = ga.verifier(snap, FENETRE)
    assert _ctl(r, "poids")["statut"] == INFO and _ctl(r, "demi_perimetre")["statut"] == INFO
    assert "ne prouve pas" in _ctl(r, "poids")["detail"] and "borne basse" in _ctl(r, "demi_perimetre")["detail"]
    # 20 mm de verre sur 3,0 × 3,0 m : 416 kg de verre, plus que les 300 kg par ouvrant
    lourd = ga.verifier(snap, dict(FENETRE, largeur_mm=6000, hauteur_mm=3000, verre_mm=20))
    assert _ctl(lourd, "poids")["statut"] == HORS and lourd["verdict"] == HORS
    # demi-périmètre : traverse 3935 + montant 2895 = 6830 mm, plus que 5 m
    large = ga.verifier(snap, dict(FENETRE, largeur_mm=8000, hauteur_mm=3000))
    c = _ctl(large, "demi_perimetre")
    assert c["statut"] == HORS and c["valeurs"]["borne_basse_mm"] == 6830 and c["valeurs"]["limite_mm"] == 5000


def test_roulettes_unite_non_imprimee_et_cale(snap):
    r = ga.verifier(snap, FENETRE)
    c = _ctl(r, "roulettes")
    assert c["statut"] == INFO and ga.HYPOTHESE_UNITE in c["detail"] and "T401011" in c["detail"]
    assert set(c["anomalies"]) == {"VER-239", "CTR-197", "VER-332"}
    cale = _ctl(r, "cale")
    assert cale["statut"] == INFO and set(cale["anomalies"]) == {"CTR-198", "CTR-196"} and "TGA3817" in cale["titre"]
    # 20 mm de verre sur 1,53 × 2,36 m : 181 kg ; la roulette simple (100) ne suffit pas, la double (200) oui
    lourd = _ctl(ga.verifier(snap, dict(FENETRE, largeur_mm=3200, hauteur_mm=2500, verre_mm=20)), "roulettes")
    assert lourd["statut"] == ETUDE and "T401012" in lourd["valeurs"]["roulettes_possibles"]
    assert "TGA3608" in lourd["valeurs"]["roulettes_possibles"]


def test_porte_fenetre_dta_sur_etude_courbes_calculees(snap):
    r = ga.verifier(snap, dict(FENETRE, configuration="pf2", hauteur_mm=2150, largeur_mm=2400))
    assert _ctl(r, "dta")["statut"] == ETUDE and "porte-fenêtre n'y figure pas" in _ctl(r, "dta")["detail"]
    assert _ctl(r, "courbe")["statut"] == OK        # 141037 + 141038, courbe 2 : 2550 à L = 2000, 2440 à 2500 → 2440
    assert r["verdict"] == ETUDE and r["contrainte"]["id"] == "dta"


def test_applications_sans_courbe_ni_nomenclature(snap):
    r = ga.verifier(snap, dict(FENETRE, configuration="pf4"))
    assert _ctl(r, "courbe")["statut"] == ETUDE and r["verdict"] == ETUDE
    r = ga.verifier(snap, dict(FENETRE, configuration="autre"))
    assert r["verdict"] == ETUDE and [c["id"] for c in r["controles"]] == ["application"]
    assert r["controles"][0]["anomalies"] == ["VER-242"]
    assert ga.fiche_debit(snap, dict(FENETRE, configuration="autre"))["indisponible"].startswith("Aucune nomenclature")


def test_lf_de_la_porte_fenetre_un_vantail_et_fixe(snap):
    r = ga.verifier(snap, dict(FENETRE, configuration="pf1_fixe", largeur_mm=2400, hauteur_mm=2150, lf_mm=800))
    assert _ctl(r, "lf")["anomalies"] == ["VER-167"]
    assert [x["libelle"] for x in r["remplissages"]] == ["le vantail", "la partie fixe"]
    assert r["remplissages"][0] == {"libelle": "le vantail", "hauteur": 2007, "largeur": 1534}   # H-143, L-Lf-66
    with pytest.raises(ValueError, match="Lf"):
        ga.verifier(snap, dict(FENETRE, configuration="pf1_fixe", lf_mm=5000))


def test_brouillons_et_anomalies_existent(snap):
    r = ga.verifier(snap, FENETRE)
    assert ga.PAGE_DTA in r["brouillons"] and ga.PAGE_DEBIT in r["brouillons"]
    assert ga.PAGE_COURBES not in r["brouillons"]            # la page des courbes est stable
    registres = "".join(p.read_text(encoding="utf-8") for p in (Path(snap.root) / "wiki" / "anomalies").glob("*.md"))
    for c in r["controles"]:
        for a in c["anomalies"]:
            assert re.search(rf"\|\s*{a}\s*\|", registres), f"{a} n'existe pas dans les registres"


def test_les_donnees_changent_avec_l_instantane(snap):
    assert ga.donnees(snap) is ga.donnees(snap)


# ---- la fiche de débit ------------------------------------------------------------------------------


def test_fiche_oracle_fenetre_2_vantaux(snap):
    """H = 1250, L = 1800 : rail T141001 1758 ; T141015 2×835 ; T141031 2×1250 ; T141037 2×1145 ; T341000 2×1702 ;
    T821004 2×805 ; vitrage 1107 × 834 ; T411002 6H+2L = 11 100."""
    f = ga.fiche_debit(snap, FENETRE)
    assert {"verdict", "libelle", "lignes", "nomenclature", "accessoires", "notes", "brouillons"} <= set(f)
    attendu = {"T141001": (1, 1758), "T141015": (2, 835), "T141031": (2, 1250), "T141037": (2, 1145),
               "T341000": (2, 1702), "T821004": (2, 805), "T411002": (1, 11100)}
    for ref, (q, l) in attendu.items():
        (ligne,) = _fiche_ligne(f, ref)
        assert (ligne["qte"], ligne["longueur"]) == (q, l), ref
    (vit,) = [l for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert vit["calcul"] == "H-143 × L/2-66 = 1107 × 834 mm" and vit["qte"] == 2
    assert f["verdict"] == OK and "INC-456" in " ".join(f["notes"])
    # chaque ligne a la forme attendue par la page
    for l in f["lignes"]:
        assert {"rep", "famille", "piece", "ref", "qte", "longueur", "calcul", "coupe", "source", "note", "image"} <= set(l)
        assert l["source"]["page"] == ga.PAGE_DEBIT and l["source"]["section"] == "Fenêtre 2 vantaux"
    assert [l["rep"] for l in f["lignes"]] == list(range(1, len(f["lignes"]) + 1))
    assert _fiche_ligne(f, "T141015")[0]["image"].endswith("ouvrant-t141015.png")
    assert {n["famille"] for n in f["nomenclature"]} == {"Profilés", "Joints", "Accessoires", "Visserie", "Remplissage"}
    n = next(n for n in f["nomenclature"] if n["ref"] == "T141015")
    assert n["qte"] == 2 and n["ml"] == pytest.approx(1.67)


def test_fiche_joint_de_la_prise_de_volume(snap):
    """La nomenclature écrit le joint de 24 mm ; à 28 mm la ligne prend le joint T411007 et le dit."""
    f = ga.fiche_debit(snap, dict(FENETRE, epaisseur_vitrage_mm="28"))
    assert _fiche_ligne(f, "T411005") == []
    (j,) = _fiche_ligne(f, "T411007")
    assert j["longueur"] == 8600 and "hypothèse" in j["note"]
    assert "hypothèse" not in _fiche_ligne(ga.fiche_debit(snap, FENETRE), "T411005")[0]["note"]


def test_fiche_indisponible_hors_de_l_epaisseur_de_la_nomenclature(snap):
    """La nomenclature de la fenêtre porte la traverse T141015 (24 à 28 mm) : pas de liste de coupe à 30 mm."""
    f = ga.fiche_debit(snap, dict(FENETRE, epaisseur_vitrage_mm="30", montants="141041+141043"))
    assert "T141015" in f["indisponible"] and "30 mm" in f["indisponible"]
    # la porte-fenêtre RC3 porte la traverse T141021 (29 à 32 mm)
    assert "T141021" in ga.fiche_debit(snap, dict(FENETRE, configuration="pf2_rc3"))["indisponible"]
    assert ga.fiche_debit(snap, dict(FENETRE, configuration="pf2_rc3", epaisseur_vitrage_mm="32"))["lignes"]


def test_fiche_formules_a_variable_lf_et_a_quantite_en_formule(snap):
    f = ga.fiche_debit(snap, {"configuration": "pf1_fixe", "largeur_mm": 2400, "hauteur_mm": 2150, "lf_mm": 900})
    assert _fiche_ligne(f, "T141015")[0]["longueur"] == 1435       # L-Lf-65
    assert _fiche_ligne(f, "T341025", 1404) and _fiche_ligne(f, "T341025", 1402)    # L-Lf-96 et L-Lf-98 : deux lignes
    (clip,) = _fiche_ligne(f, "TGA3612")
    assert clip["qte"] == 5 and "arrondi non écrit" not in clip["note"]            # (L-Lf)/300 = 5 pile
    (vis,) = _fiche_ligne(f, "TVE017")
    assert vis["qte"] == 6                                                          # 3+Lf/300
    f = ga.fiche_debit(snap, {"configuration": "pf1_fixe", "largeur_mm": 2400, "hauteur_mm": 2150, "lf_mm": 800})
    assert _fiche_ligne(f, "TGA3612")[0]["qte"] == pytest.approx(5.33) and "arrondi non écrit" in _fiche_ligne(f, "TGA3612")[0]["note"]
    vitres = [l for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert [(l["qte"], l["calcul"]) for l in vitres] == [(1, "H-143 × L-Lf-66 = 2007 × 1534 mm"),
                                                         (1, "H-75 × Lf-66 = 2075 × 734 mm")]
    assert "la planche ne dit pas laquelle" not in vitres[0]["note"]    # ici la planche nomme le vantail et la partie fixe


def test_fiche_toutes_les_applications_calculent_leurs_lignes(snap):
    """Seules deux lignes du wiki ne se calculent pas : le joint de la RC3 (« 4H+2l », INC-466) et le seuil
    extérieur PMR (« Voir environnement pose »)."""
    for k in ga.APPLICATIONS:
        f = ga.fiche_debit(snap, {"configuration": k, "largeur_mm": 2400, "hauteur_mm": 2150, "lf_mm": 800,
                                  "epaisseur_vitrage_mm": "32" if k == "pf2_rc3" else "24"})
        non = [n for n in f["notes"] if n.startswith("Lignes non calculées")]
        if k == "pf2_rc3":
            assert "T411009" in non[0] and "INC-466" in " ".join(f["notes"])
        elif k == "pf2_pmr":
            assert "T401028" in non[0] and "du texte" in non[0]
        else:
            assert non == [], (k, non)
        assert f["lignes"] and f["nomenclature"]
        for l in f["lignes"]:
            assert l["qte"] is not None and l["qte"] > 0
            if l["famille"] == "Profilés":
                assert l["longueur"] is not None and l["longueur"] > 0


def test_fiche_formules_avec_espaces_et_deux_longueurs_pour_une_reference(snap):
    f = ga.fiche_debit(snap, {"configuration": "pf3", "largeur_mm": 2400, "hauteur_mm": 2150})
    assert [(l["qte"], l["longueur"]) for l in _fiche_ligne(f, "T341000")] == [(1, 2246), (2, 2302)]   # L - 154 ; L - 98
    f = ga.fiche_debit(snap, {"configuration": "pf4", "largeur_mm": 2400, "hauteur_mm": 2150})
    assert [(l["qte"], l["longueur"]) for l in _fiche_ligne(f, "T341000")] == [(1, 2246), (1, 2358)]
    assert _fiche_ligne(f, "T141015")[0]["qte"] == 4 and _fiche_ligne(f, "T141015")[0]["longueur"] == 554   # L/4-46


# ---- par le registre ------------------------------------------------------------------------------


def test_le_registre_propose_le_lumeal_ga(snap):
    s = next(x for x in reg.systemes()["systemes"] if x["id"] == "lumeal-ga")
    assert s["libelle"] == "LUMEAL GA (coulissant LUMINE)" and s["gammes"] == ["LUMINE"] and s["legacy"] is False
    assert s["portee"] == ga.PORTEE
    o = reg.options(snap, "lumeal-ga")
    assert {"champs", "configurations", "pages", "brouillons", "avertissements"} <= set(o)
    ids = [c["id"] for c in o["champs"]]
    assert ids[:3] == ["configuration", "largeur_mm", "hauteur_mm"]
    assert set(o["configurations"]) == set(ga.APPLICATIONS) | {"autre"}
    assert all(p in snap.pages for p in o["pages"]) and ga.PAGE_DEBIT in o["brouillons"]
    assert all(a for a in o["avertissements"])
    r = reg.verifier(snap, "lumeal-ga", FENETRE)
    assert r["verdict"] == OK and r["application"] == "Fenêtre 2 vantaux"
    f = reg.fiche_debit(snap, "lumeal-ga", FENETRE)
    assert f["lignes"] and f["verdict"] == OK


def test_la_saisie_est_verifiee_contre_les_champs(snap):
    with pytest.raises(ValueError, match="Application"):
        ga.verifier(snap, dict(FENETRE, configuration="coulissant"))
    with pytest.raises(ValueError, match="Largeur hors tout L : au moins"):
        ga.verifier(snap, dict(FENETRE, largeur_mm=10))
    with pytest.raises(ValueError, match="Épaisseur du double vitrage"):
        ga.verifier(snap, dict(FENETRE, epaisseur_vitrage_mm="40"))
    # un champ non montré (montants pour une application sans courbe) n'est pas vérifié
    assert ga.verifier(snap, dict(FENETRE, configuration="pf4", montants="n'importe quoi"))["verdict"] == ETUDE


def test_les_courbes_sont_donnees_a_la_page_avec_le_point_saisi(snap):
    """La page trace les courbes relevées : celle de la paire saisie en avant, le point colorié par son verdict."""
    r = reg.verifier(snap, "lumeal-ga", dict(FENETRE))
    g = r["graphe_courbes"]
    assert g["point"] == [1800, 1250] and g["statut"] == _ctl(r, "courbe")["statut"]
    choisies = [c for c in g["courbes"] if c["choisie"]]
    assert len(choisies) == 1 and len(g["courbes"]) > 1
    # Les graduations du wiki, telles quelles (courbe 1 : 2185 mm à L = 1500, 2020 mm à L = 2000) ; rien n'est
    # interpolé, y compris au point saisi (L = 1800 lit la plus restrictive des deux : 2020).
    assert [1500, 2185.0] in choisies[0]["points"] and [2000, 2020.0] in choisies[0]["points"]
    assert all(l % 500 == 0 for l, _ in choisies[0]["points"])
    # Pas de courbe lue (application sans courbe) : pas de tracé.
    sans = reg.verifier(snap, "lumeal-ga", {"configuration": "pf2_fixe_central", "largeur_mm": 3000, "hauteur_mm": 2200})
    assert sans["graphe_courbes"] is None
