"""Le calculateur du SOLEAL GY : DTA par épaisseur de vitrage, courbes des montants centraux et latéraux, débit.

Les attendus sont ceux des exemples écrits dans le wiki ou recalculés à la main depuis ses tableaux : si une
valeur ou un en-tête change dans le wiki, un test le dit — c'est voulu."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services import faisabilite_ga as ga
from app.services import faisabilite_gy as gy
from app.services import faisabilite_systemes as reg
from app.services.faisabilite import DonneeIntrouvable, ETUDE, HORS, INFO, OK, _corps
from app.services.faisabilite_ga import FormuleIllisible, evaluer, lire_lignes, classer_table
from app.services.wiki_service import load_snapshot, wiki_root
from app.services.wiki_tables import tables_toutes


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _ctl(r, id_):
    return next(c for c in r["controles"] if c["id"] == id_)


def _lignes(f, ref):
    return [(l["qte"], l["longueur"]) for l in f["lignes"] if l["ref"] == ref]


PF = {"configuration": "pf2", "largeur_mm": 2400, "hauteur_mm": 2150}
FENETRE = {"configuration": "fenetre2", "largeur_mm": 1800, "hauteur_mm": 1300}


# ---- la grammaire du GY, sur toutes les planches d'application -------------------------------------


@pytest.mark.parametrize("formule, variables, attendu", [
    ("L/2-91", {"L": 2400}, 1109),
    ("L/3-132(B)", {"L": 2400}, 668),
    ("3L/4-99.5(BE)", {"L": 2000}, 1400.5),
    ("3L/4-179.5(H)", {"H": 1000, "L": 2000}, 1320.5),     # « (H) » est une lettre de coupe, pas un produit par H
    ("(L1-6)+72", {"L1": 1000}, 1066),
    ("(2L1-116)+12,5", {"L1": 1000}, 1896.5),
    ("L1/2-43.5(A'1)", {"L1": 1000}, 456.5),
    ("L/2-28(B\")", {"L": 1000}, 472),
    ("H+32", {"H": 1000}, 1032),
    ("6H+4L", {"H": 2150, "L": 2400}, 22500),
    ("2+H/300", {"H": 1200}, 6),
    ("(6H+3L1+3L2)/330", {"H": 1100, "L1": 330, "L2": 330}, 26),
    ("H-77", {"H": 2150}, 2073),
])
def test_la_grammaire_lit_les_formules_du_gy(formule, variables, attendu):
    assert evaluer(formule, variables).valeur == pytest.approx(attendu)


def test_toutes_les_planches_d_application_se_lisent(snap):
    """Chaque cellule « Débit » et « Qté » des 18 pages d'application (1 433 cellules) est lue par la grammaire :
    une formule nouvelle que la grammaire refuse fait échouer le test, avec la page, la référence et la raison."""
    pages = sorted(p for p in snap.pages if p.startswith("/profiles/soleal-gy-application-"))
    assert len(pages) == 18
    vues, refusees = 0, []
    for page in pages:
        for t in tables_toutes(_corps(snap, page)):
            genre = classer_table(t)
            if genre is None:
                continue
            for l in lire_lignes(t, genre):
                for texte in (l.debit, l.qte):
                    if not texte:
                        continue
                    noms = set(re.findall(r"[A-Za-z][A-Za-z0-9]*", re.sub(r"\([A-Za-z][A-Za-z0-9'\"]*\)$", "", texte)))
                    try:
                        evaluer(texte, {n: 1000.0 for n in noms})
                        vues += 1
                    except FormuleIllisible as exc:
                        refusees.append((page.split("/")[-1], l.ref, texte, str(exc)))
    assert refusees == []
    assert vues > 1400


# ---- la lecture des tableaux : le contrat avec le wiki -----------------------------------------------


def test_les_tableaux_se_lisent(snap):
    d = gy.donnees(snap)
    fen, pf, pmr = (d.dta[k] for k in ("Fenêtre 2 vantaux", "Porte-fenêtre 2 vantaux", "Porte-fenêtre 2 vantaux + Seuil PMR"))
    assert (fen.h_mm, fen.l_mm, pf.h_mm, pf.l_mm) == (1650, 1850, 2500, 3000)
    assert fen.paires == {(24, 28): ("TGY1207", "TGY1207"), (29, 32): ("TGY1213", "TGY1213")}
    assert pf.paires == {(24, 28): ("TGY1207", "TGY1209"), (29, 32): ("TGY1213", "TGY1215")}
    # INC-382 : « TGY12013 » n'existe pas, la paire du PMR à 29-32 mm est illisible
    assert pmr.paires[(29, 32)] is None and pmr.imprime[(29, 32)] == "TGY12013 + TGY1215"
    assert d.dta_validite == "25/06/2024 au 31/07/2029" and d.dta_vitrage == (24, 32)
    assert d.lateraux_porte_fenetre == ["TGY1303", "TGY1304"]
    assert d.limites == {"kg": 200, "demi_perimetre": 5000, "precision": 25, "cadre": 3000}
    c = d.courbes["A2"]["centraux"]
    assert c["pression"] == 800 and len(c["colonnes"]) == 6
    assert c["colonnes"][("TGY1207", "TGY1209")][3000] == 2700 and c["colonnes"][("TGY1207", "TGY1207")][1000] == 2340
    assert d.courbes["A3"]["centraux"]["pression"] == 1200
    lat = d.courbes["A2"]["lateraux"]["colonnes"]
    assert lat[("TGY1202", "TGY1202")][3000] == 1970 and lat[("TGY1303", "TGY1303")][2000] == 2900
    assert lat[("TGY1200", "TGY1200")][1000] is None
    app = d.application
    assert [(l.ref, l.qte, l.debit) for l in app.profils][:3] == [("T141015", "4", "L/2-91"), ("T341000", "2", "L-95"),
                                                                  ("T431025", "2", "L/2-55(AB)")]
    assert app.profils[0].coupe == "||" and next(l for l in app.profils if l.ref == "TGY1104").coupe == "/\\"
    assert app.remplissages == [{"libelle": "", "hauteur": "H-179", "largeur": "L/2-91.5"}]
    assert {"VER-458", "VER-459", "VER-460", "VER-461"} <= set(app.anomalies)
    assert gy.paire_application(app) == ("TGY1207", "TGY1208") and gy.lateral_application(app) == "TGY1202"
    assert d.plages == {"T141015": (24, 28), "T141021": (29, 32)}
    assert d.joints_volume[24]["ref"] == "T411005" and d.joints_volume[29]["ouvrant"] == "29 à 32 mm"
    assert d.roulettes["T401012"]["poids"] == 200 and d.roulettes["T401011"]["poids"] == 100
    assert len(d.cale) == 3 and d.densite_verre == 2.5
    assert d.images["TGY1207"].endswith("montant-central-tgy1207.png")


def test_un_en_tete_change_dans_le_wiki_casse_la_lecture(snap):
    corps = _corps(snap, gy.PAGE_APPLICATION)
    assert len(ga.lire_nomenclature(corps, None, gy.PAGE_APPLICATION, "Remplissage").profils) == 13
    with pytest.raises(DonneeIntrouvable):
        ga.lire_nomenclature(corps.replace("| Référence | Désignation | Qté | Coupe | Débit (mm) |",
                                           "| Référence | Désignation | Qté | Coupe | Longueur (mm) |"),
                             None, gy.PAGE_APPLICATION, "Remplissage")
    dta = _corps(snap, gy.PAGE_DTA)
    assert "| H (m) | L (m) |" in dta        # l'en-tête que ``_charge`` exige du tableau du DTA


# ---- le DTA, les montants et les courbes -------------------------------------------------------------


def test_dta_oracle(snap):
    """Porte-fenêtre 2 vantaux : 2,50 × 3,00 dans le DTA, 2,60 × 3,00 hors ; fenêtre : 1,65 × 1,85."""
    def dta(**k):
        return _ctl(gy.verifier(snap, dict(PF, **k)), "dta")["statut"]
    assert dta(hauteur_mm=2500, largeur_mm=3000) == OK
    assert dta(hauteur_mm=2600, largeur_mm=3000) == HORS
    assert dta(hauteur_mm=2500, largeur_mm=3010) == HORS
    f = lambda h, l: _ctl(gy.verifier(snap, dict(FENETRE, hauteur_mm=h, largeur_mm=l)), "dta")["statut"]
    assert f(1650, 1850) == OK and f(1700, 1850) == HORS and f(1650, 1900) == HORS


def test_dta_montants_selon_l_epaisseur_du_vitrage(snap):
    r = gy.verifier(snap, PF)
    c = _ctl(r, "dta")
    assert c["statut"] == OK and c["valeurs"]["paire"] == ["TGY1207", "TGY1209"] and c["valeurs"]["lateral"] == "TGY1303"
    assert gy.verifier(snap, PF)["montants"] == {"centraux": ["TGY1207", "TGY1209"], "lateral": "TGY1303"}
    assert c["anomalies"] == ["CTR-139", "VER-178"] and "25/06/2024 au 31/07/2029" in c["detail"]
    assert gy.HYPOTHESE_BAIE in c["detail"]
    # 29 à 32 mm : TGY1213 + TGY1215, qui n'ont pas de courbe
    r = gy.verifier(snap, dict(PF, epaisseur_vitrage_mm="30"))
    assert _ctl(r, "dta")["valeurs"]["paire"] == ["TGY1213", "TGY1215"] and _ctl(r, "dta")["statut"] == OK
    assert _ctl(r, "courbe")["statut"] == ETUDE and "pas de courbe" in _ctl(r, "courbe")["detail"]
    # la fenêtre : TGY1207 + TGY1207 à 24 mm, TGY1213 + TGY1213 à 29 mm
    assert _ctl(gy.verifier(snap, FENETRE), "dta")["valeurs"]["paire"] == ["TGY1207", "TGY1207"]
    assert _ctl(gy.verifier(snap, dict(FENETRE, epaisseur_vitrage_mm="29")), "dta")["valeurs"]["paire"] == ["TGY1213", "TGY1213"]


def test_la_planche_pose_une_paire_que_le_dta_ne_nomme_pas(snap):
    """TGY1207 + TGY1208 : la paire de la planche d'application, absente du tableau du DTA."""
    r = gy.verifier(snap, dict(PF, montants="TGY1207+TGY1208"))
    c = _ctl(r, "dta")
    assert c["statut"] == ETUDE and "absente du tableau du DTA" in c["detail"] and "TGY1207 + TGY1209" in c["detail"]
    courbe = _ctl(r, "courbe")                     # la courbe 2 existe : 2360 à L = 2500, 2460 à 2000 → 2360
    assert courbe["statut"] == OK and courbe["valeurs"]["h_max"] == 2360
    assert r["verdict"] == ETUDE and r["contrainte"]["id"] == "dta"
    o = reg.options(snap, "soleal-gy")
    assert "TGY1207 + TGY1208 (planche d'application)" in o["champs"][5]["options"].values()
    assert any("TGY1207 + TGY1208" in a and "absente du tableau du DTA" in a for a in o["avertissements"])


def test_portes_fenetres_et_montants_lateraux_renforces(snap):
    """TGY1202 (la planche) : le DTA veut TGY1303 ou TGY1304, et la courbe du TGY1202 est dépassée ; TGY1304 : pas de courbe."""
    r = gy.verifier(snap, dict(PF, lateral="TGY1202"))
    assert _ctl(r, "dta")["statut"] == ETUDE and "TGY1303 ou TGY1304" in _ctl(r, "dta")["detail"]
    lat = _ctl(r, "courbe_lateraux")                # TGY1202 : 2180 à L = 2000, 2040 à 2500 → 2040 ; H = 2150
    assert lat["statut"] == HORS and lat["valeurs"]["h_max"] == 2040 and r["verdict"] == HORS
    r = gy.verifier(snap, dict(PF, lateral="TGY1304"))
    assert _ctl(r, "dta")["statut"] == OK and _ctl(r, "courbe_lateraux")["statut"] == ETUDE
    # une fenêtre : le DTA ne dit rien des latéraux, c'est celui de la planche
    r = gy.verifier(snap, FENETRE)
    assert _ctl(r, "dta")["valeurs"]["lateral"] == "TGY1202"
    assert "latéraux renforcés" not in _ctl(r, "dta")["detail"]


def test_courbes_oracles_a2_et_a3(snap):
    """TGY1207 / TGY1209 (courbe 4), L = 3000 : 2700 en A2, H = 2500 est réalisable ; en A3 la courbe vaut 2500 : sur étude."""
    base = dict(PF, largeur_mm=3000, hauteur_mm=2500)
    a2 = _ctl(gy.verifier(snap, base), "courbe")
    assert a2["statut"] == OK and a2["valeurs"]["h_max"] == 2700 and a2["anomalies"] == ["VER-455"]
    assert gy.HYPOTHESE_COURBE in a2["detail"] and "Courbe 4 : TGY1207 / TGY1209" in a2["detail"]
    a3 = _ctl(gy.verifier(snap, dict(base, vent="A3")), "courbe")
    assert a3["statut"] == ETUDE and a3["valeurs"]["h_max"] == 2500 and a3["valeurs"]["classe"] == "A3"
    assert _ctl(gy.verifier(snap, dict(base, vent="A3", hauteur_mm=2400)), "courbe")["statut"] == OK
    assert _ctl(gy.verifier(snap, dict(base, vent="A3", hauteur_mm=2600)), "courbe")["statut"] == HORS


def test_courbe_entre_deux_graduations_et_inerties_equivalentes(snap):
    """TGY1207 + TGY1207 (courbe 1), L = 2400 : graduations 2000 (1920) et 2500 (1840) → 1840 ; ±25 mm."""
    c = lambda h, **k: _ctl(gy.verifier(snap, dict(FENETRE, largeur_mm=1800, hauteur_mm=h, **k)), "courbe")
    assert c(1300)["valeurs"]["h_max"] == 1920            # entre 1500 (2070) et 2000 (1920) : la plus restrictive
    r = gy.verifier(snap, dict(PF, montants="TGY1208+TGY1209"))
    assert "VER-457" in _ctl(r, "courbe")["anomalies"]    # TGY1208 = TGY1211 = TGY1210 : inerties pourtant différentes


def test_vitrage_charge_et_demi_perimetre(snap):
    def vit(**k):
        return _ctl(gy.verifier(snap, dict(PF, **k)), "vitrage")
    assert vit(epaisseur_vitrage_mm="28")["valeurs"]["joint"] == "T411007" and vit(epaisseur_vitrage_mm="28")["statut"] == OK
    c = vit(epaisseur_vitrage_mm="30")
    assert c["statut"] == OK and c["valeurs"]["joint"] == "T411008" and "liste de coupe n'est pas établie" in c["detail"]
    assert vit(epaisseur_vitrage_mm="24", verre_mm=30)["statut"] == ETUDE
    r = gy.verifier(snap, PF)
    assert _ctl(r, "poids")["statut"] == INFO and _ctl(r, "demi_perimetre")["statut"] == INFO
    assert r["poids_verre_kg"] == pytest.approx(43.7, abs=0.1)
    assert r["remplissages"] == [{"libelle": "", "hauteur": 1971, "largeur": 1108.5}]    # H-179 × L/2-91.5
    # 20 mm de verre sur 2,8 × 2,9 m : 410 kg, plus que les 200 kg par ouvrant
    lourd = gy.verifier(snap, dict(PF, largeur_mm=6000, hauteur_mm=3000, verre_mm=20))
    assert _ctl(lourd, "poids")["statut"] == HORS and _ctl(lourd, "poids")["valeurs"]["limite_kg"] == 200
    large = _ctl(gy.verifier(snap, dict(PF, largeur_mm=10000, hauteur_mm=3000)), "demi_perimetre")
    assert large["statut"] == HORS and large["valeurs"]["limite_mm"] == 5000


def test_roulettes_et_cale(snap):
    r = gy.verifier(snap, PF)
    c = _ctl(r, "roulettes")
    assert c["statut"] == INFO and "T401012" in c["detail"] and gy.HYPOTHESE_UNITE in c["detail"] and c["anomalies"] == ["VER-203"]
    cale = _ctl(r, "cale")
    assert cale["statut"] == INFO and cale["anomalies"] == ["VER-204", "VER-480"] and "sans connecteur" in cale["detail"]
    assert "T141015" in cale["detail"] and "1,5 m" in cale["detail"] and "120 kg" in cale["detail"]
    lourd = _ctl(gy.verifier(snap, dict(PF, largeur_mm=4000, hauteur_mm=3000, verre_mm=20)), "roulettes")
    assert lourd["statut"] == ETUDE and "toutes les roulettes" in lourd["detail"]


def test_la_forme_du_resultat_et_les_anomalies(snap):
    r = gy.verifier(snap, PF)
    assert r["verdict"] == OK, [(c["id"], c["statut"], c["detail"]) for c in r["controles"]]
    assert [c["id"] for c in r["controles"]] == ["dta", "courbe", "courbe_lateraux", "poids", "demi_perimetre", "vitrage",
                                                 "roulettes", "cale"]
    assert {"verdict", "libelle", "contrainte", "controles", "brouillons"} <= set(r)
    registres = "".join(p.read_text(encoding="utf-8") for p in (Path(snap.root) / "wiki" / "anomalies").glob("*.md"))
    for c in r["controles"]:
        assert set(c) == {"id", "titre", "statut", "detail", "sources", "anomalies", "valeurs"} and c["detail"]
        assert c["statut"] in (OK, HORS, ETUDE, INFO)
        for s in c["sources"]:
            assert set(s) == {"page", "section", "localisation"} and s["page"] in snap.pages
        for a in c["anomalies"]:
            assert re.search(rf"\|\s*{a}\s*\|", registres), f"{a} n'existe pas dans les registres"
    pmr = gy.verifier(snap, dict(PF, configuration="pf2_pmr"))
    assert "INC-382" in _ctl(pmr, "dta")["anomalies"] and re.search(r"\|\s*INC-382\s*\|", registres)
    # pages en brouillon lues : la planche, les courbes, les joints, les roulettes ; le DTA est stable
    assert gy.PAGE_APPLICATION in r["brouillons"] and gy.PAGE_COURBES in r["brouillons"]
    assert gy.PAGE_DTA not in r["brouillons"]


def test_seuil_pmr_a_29_32_mm_la_paire_du_dta_est_illisible(snap):
    r = gy.verifier(snap, dict(PF, configuration="pf2_pmr", epaisseur_vitrage_mm="30"))
    c = _ctl(r, "dta")
    assert c["statut"] == ETUDE and "TGY12013 + TGY1215" in c["detail"] and "n'existe pas (INC-382)" in c["detail"]
    ok24 = gy.verifier(snap, dict(PF, configuration="pf2_pmr"))
    assert _ctl(ok24, "dta")["statut"] == OK


def test_typologie_hors_2_vantaux(snap):
    r = gy.verifier(snap, dict(PF, configuration="autre"))
    assert r["verdict"] == ETUDE and [c["id"] for c in r["controles"]] == ["typologie"]
    assert gy.fiche_debit(snap, dict(PF, configuration="autre"))["indisponible"].startswith("Aucune nomenclature")


# ---- la fiche de débit ------------------------------------------------------------------------------------


def test_fiche_oracle_porte_fenetre_2_vantaux(snap):
    """H = 2150, L = 2400 : T141015 4×1109 ; rail T341000 2×2305 ; TGY1202 2×2073 ; TGY1207 et TGY1208 2073 ; dormant
    TGY1104 2×2150 + 2400 ; TGY1120 2400 ; vitrage 1971 × 1108,5 ; T411005 4H+2L = 13 400 ; TGY5018 6H+4L = 22 500."""
    f = gy.fiche_debit(snap, dict(PF, montants="TGY1207+TGY1208", lateral="TGY1202"))
    assert _lignes(f, "T141015") == [(4, 1109)] and _lignes(f, "T341000") == [(2, 2305)]
    assert _lignes(f, "TGY1202") == [(2, 2073)] and _lignes(f, "TGY1207") == [(1, 2073)] and _lignes(f, "TGY1208") == [(1, 2073)]
    assert _lignes(f, "TGY1104") == [(2, 2150), (1, 2400)] and _lignes(f, "TGY1120") == [(1, 2400)]
    assert _lignes(f, "T411005") == [(1, 13400)] and _lignes(f, "TGY5018") == [(1, 22500)]
    assert _lignes(f, "T431025") == [(2, 1145)] and _lignes(f, "TGY4006") == [(2, 2103), (4, 1132.5)]
    (vit,) = [l for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert vit["calcul"] == "H-179 × L/2-91.5 = 1971 × 1108,5 mm"
    assert vit["qte"] == 2 and "un par vantail" in vit["note"]
    # la planche pose TGY1207 + TGY1208 (absente du DTA) et TGY1202, dont la courbe est dépassée à 2150 × 2400
    assert f["verdict"] == HORS


def test_fiche_forme_pictogrammes_et_sources(snap):
    f = gy.fiche_debit(snap, PF)
    assert {"verdict", "libelle", "lignes", "nomenclature", "accessoires", "notes", "brouillons"} <= set(f)
    for l in f["lignes"]:
        assert {"rep", "famille", "piece", "ref", "qte", "longueur", "calcul", "coupe", "source", "note", "image"} <= set(l)
        assert l["source"]["page"] == gy.PAGE_APPLICATION
    assert [l["rep"] for l in f["lignes"]] == list(range(1, len(f["lignes"]) + 1))
    rail = next(l for l in f["lignes"] if l["ref"] == "T341000")
    assert rail["coupe"] == "pictogramme « || » (non légendé)" and rail["source"]["section"] == "Profilés"
    dormant = next(l for l in f["lignes"] if l["ref"] == "TGY1104")
    assert dormant["coupe"] == "pictogramme « /\\ » (non légendé)"
    assert {l["source"]["section"] for l in f["lignes"]} == {"Profilés", "Joints", "Accessoires", "Vis", "Remplissage"}
    assert next(l for l in f["lignes"] if l["ref"] == "T141015")["image"].endswith("ouvrant-t141015.png")
    n = next(n for n in f["nomenclature"] if n["ref"] == "T141015")
    assert n["qte"] == 4 and n["ml"] == pytest.approx(4.44)
    assert "VER-459" in " ".join(f["notes"])        # les anomalies que la page cite


def test_fiche_les_suffixes_de_coupe_ne_sont_pas_des_calculs(snap):
    """« L/2-55(AB) » et « L/2-67.5(AB) » : la lettre de coupe n'est définie nulle part (VER-459), le calcul l'ignore."""
    f = gy.fiche_debit(snap, PF)
    assert next(l for l in f["lignes"] if l["ref"] == "T431025")["calcul"] == "L/2-55(AB)"
    assert next(l for l in f["lignes"] if l["ref"] == "T431025")["longueur"] == 1145
    assert not [n for n in f["notes"] if n.startswith("Lignes non calculées")]


def test_fiche_montants_choisis(snap):
    """TGY1207 + TGY1209 (DTA) et TGY1303 (DTA) : la nomenclature les pose, avec le débit H-77 des règles de débit."""
    f = gy.fiche_debit(snap, PF)
    assert _lignes(f, "TGY1207") == [(1, 2073)] and _lignes(f, "TGY1209") == [(1, 2073)] and _lignes(f, "TGY1208") == []
    assert _lignes(f, "TGY1303") == [(2, 2073)] and _lignes(f, "TGY1202") == []
    notes = " ".join(f["notes"])
    assert "Montants centraux TGY1207 + TGY1209" in notes and "Montant latéral TGY1303" in notes
    assert "sans planche d'application qui le confirme" in notes
    assert f["verdict"] == OK
    # une paire de la même référence : une ligne de deux pièces
    f = gy.fiche_debit(snap, dict(PF, montants="TGY1207+TGY1207"))
    assert _lignes(f, "TGY1207") == [(2, 2073)]
    # la planche telle quelle ne produit aucune note de remplacement
    f = gy.fiche_debit(snap, dict(PF, montants="TGY1207+TGY1208", lateral="TGY1202"))
    assert not [n for n in f["notes"] if n.startswith("Montant")]


def test_fiche_joint_de_la_prise_de_volume_et_indisponibilites(snap):
    f = gy.fiche_debit(snap, dict(PF, epaisseur_vitrage_mm="28"))
    assert _lignes(f, "T411005") == [] and _lignes(f, "T411007") == [(1, 13400)]
    assert "hypothèse" in next(l for l in f["lignes"] if l["ref"] == "T411007")["note"]
    assert "T141015" in gy.fiche_debit(snap, dict(PF, epaisseur_vitrage_mm="30"))["indisponible"]
    assert "pas de nomenclature de la fenêtre" in gy.fiche_debit(snap, FENETRE)["indisponible"]
    assert "seuil PMR" in gy.fiche_debit(snap, dict(PF, configuration="pf2_pmr"))["indisponible"]


# ---- par le registre ----------------------------------------------------------------------------------------


def test_le_registre_propose_le_soleal_gy(snap):
    s = next(x for x in reg.systemes()["systemes"] if x["id"] == "soleal-gy")
    assert s["libelle"] == "SOLEAL GY (coulissant LUMINE)" and s["gammes"] == ["LUMINE"] and s["legacy"] is False
    assert s["portee"] == gy.PORTEE
    o = reg.options(snap, "soleal-gy")
    assert {"champs", "configurations", "pages", "brouillons", "avertissements"} <= set(o)
    assert [c["id"] for c in o["champs"]] == ["configuration", "largeur_mm", "hauteur_mm", "epaisseur_vitrage_mm", "verre_mm",
                                              "montants", "lateral", "vent"]
    assert set(o["configurations"]) == {"fenetre2", "pf2", "pf2_pmr", "autre"}
    assert all(p in snap.pages for p in o["pages"]) and gy.PAGE_APPLICATION in o["brouillons"]
    assert all(a for a in o["avertissements"])
    r = reg.verifier(snap, "soleal-gy", PF)
    assert r["verdict"] == OK and r["application"] == "Porte-fenêtre 2 vantaux"
    f = reg.fiche_debit(snap, "soleal-gy", PF)
    assert f["lignes"] and f["verdict"] == OK


def test_la_saisie_est_verifiee_contre_les_champs(snap):
    with pytest.raises(ValueError, match="Typologie"):
        gy.verifier(snap, dict(PF, configuration="coulissant"))
    with pytest.raises(ValueError, match="Largeur hors tout L : au moins"):
        gy.verifier(snap, dict(PF, largeur_mm=10))
    with pytest.raises(ValueError, match="Montants centraux d'ouvrant"):
        gy.verifier(snap, dict(PF, montants="TGY1207+TGY9999"))
    with pytest.raises(ValueError, match="Classe de vent"):
        gy.verifier(snap, dict(PF, vent="A9"))
    # un champ non montré (typologie « autre ») n'est pas vérifié
    assert gy.verifier(snap, dict(PF, configuration="autre", vent="A9"))["verdict"] == ETUDE


def test_les_courbes_des_montants_centraux_sont_donnees_a_la_page(snap):
    from app.services import faisabilite_systemes as reg
    r = reg.verifier(snap, "soleal-gy", {"configuration": "pf2", "largeur_mm": 3000, "hauteur_mm": 2500})
    g = r["graphe_courbes"]
    assert g["point"] == [3000, 2500] and sum(1 for c in g["courbes"] if c["choisie"]) == 1
    assert all(p[1] > 0 for c in g["courbes"] for p in c["points"])
