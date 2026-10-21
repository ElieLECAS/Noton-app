"""Le calculateur du SOLEAL FY : DTA et sa validité, inertie du châssis 2 vantaux, quincaillerie cachée et visible,
vitrage, deux générations de nomenclature, liste de coupe.

Les attendus sont ceux des exemples écrits dans le wiki ou recalculés à la main depuis ses tableaux : si une
valeur ou un en-tête change dans le wiki, un test le dit — c'est voulu. La date du jour est fixée : l'échéance du
DTA (31/10/2026) ne doit pas faire dépendre les tests de la date où on les lance."""
from __future__ import annotations

import datetime
import json
import re
from dataclasses import replace

import pytest

from app.services import faisabilite_fy as fy
from app.services import faisabilite_systemes as reg
from app.services.faisabilite import DonneeIntrouvable, ETUDE, HORS, INFO, OK, _corps
from app.services.faisabilite_ga import FormuleIllisible, evaluer
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


@pytest.fixture(autouse=True)
def date_fixe(monkeypatch):
    """Le 1er juin 2026 : le DTA est valide, à plus de 90 jours de son échéance."""
    monkeypatch.setattr(fy, "_aujourdhui", lambda: datetime.date(2026, 6, 1))


def _ctl(r, id_):
    return next(c for c in r["controles"] if c["id"] == id_)


def _ids(r):
    return [c["id"] for c in r["controles"]]


def _lignes(f, ref):
    return [(l["qte"], l["longueur"]) for l in f["lignes"] if l["ref"] == ref]


V1 = {"configuration": "v1", "largeur_mm": 1000, "hauteur_mm": 1350}          # catalogue 6319, Apparent, FY 55
V2 = {"configuration": "v2", "largeur_mm": 1200, "hauteur_mm": 1350}
V2_6057 = dict(V2, generation="6057")


# ---- toutes les formules des planches se lisent -----------------------------------------------------------


def test_toutes_les_formules_des_50_applications_se_lisent(snap):
    """Chaque cellule « Débit », « Qté » et chaque cote de remplissage des 50 applications (28 en 6319, 22 en 6057) est lue
    par l'évaluateur, hors les cas que le calcul écarte lui-même et dont il parle : quantité renvoyée à un abaque
    (« *** », « voir abaque », « - »), quantité par mètre linéaire (« 3/ml/H ») et cote de vitrage préfixée (VER-357)."""
    d = fy.donnees(snap)
    assert len(d.apps) == 50
    refusees, vues = set(), 0
    for app in d.apps.values():
        formules = ([l.debit for l in app.profils + app.joints] + [l.qte for l in app.profils + app.accessoires + app.vis]
                    + [c for r in app.remplissages for c in (r["hauteur"], r["largeur"])])
        for f in filter(None, formules):
            f = fy._PAR_METRE.sub(r"(\1*\2/1000)", f)          # « 3/ml/H » : 3 par mètre linéaire de H
            noms = set(re.findall(r"[A-Za-z][A-Za-z0-9]*", f))
            try:
                evaluer(f, {n: 1000.0 for n in noms})
                vues += 1
            except FormuleIllisible:
                refusees.add(f)
    assert vues > 1000
    assert {f for f in refusees if "préfixe non défini" not in f} == {"***", "voir abaque", "-"}
    assert {f for f in refusees if "préfixe non défini" in f} == {
        "2H3-98 [préfixe non défini]", "2L5/2-83 [préfixe non défini]", "2H2-79 [préfixe non défini]"}


# ---- la lecture des tableaux : le contrat avec le wiki ------------------------------------------------------


def test_les_tableaux_se_lisent(snap):
    d = fy.donnees(snap)
    # le DTA : 10 lignes, une par type de fenêtre et ouvrant
    assert len(d.dta) == 10
    assert d.dta[("2 vantaux OF", "a")] == (2225, 1550) and d.dta[("2 vantaux OF", "m")] == (2200, 1550)
    assert d.dta[("1 vantail OB", "a")] == (1575, 1150) and d.dta[("Soufflet", "m")] == (550, 1250)
    assert d.dta[("2 vantaux OF + fixe latéral", "a")] == (2225, 2350)
    assert d.dta_validite == (datetime.date(2023, 5, 25), datetime.date(2026, 10, 31))
    assert d.dta_vitrage_max == 32 and d.dta_sans_pmr is True
    # les courbes d'inertie : précision de ±0,03 m, « - » = courbe non tracée
    assert d.inertie_precision == 30
    c = d.inertie[("Apparent", "FY55")]
    assert c["courbe"] == 1 and c["montants"] == "T215180 + T215186"
    assert c["valeurs"] == {0: None, 500: None, 1000: None, 1500: 2950, 2000: 2760, 2500: 2630, 2800: 2570, 3500: None}
    assert d.inertie[("Minimal", "FY55")]["valeurs"][1000] == 2850
    assert d.inertie[("Apparent", "FY65")]["valeurs"][2500] == 2880 and d.inertie[("Apparent", "FY65")]["courbe"] == 2
    # les abaques de quincaillerie cachée
    assert sorted(d.abaques["F"]) == sorted(d.abaques["OB"]) == [80, 110, 160]
    ligne = d.abaques["F"][80].lignes[900]
    assert ligne[0] == 3000 and [(c.kg_m2, c.lo, c.hi) for c in ligne[1]][-1] == (80.0, 600.0, 1100.0)
    assert d.abaques["F"][80].lignes[800][1][0].dagger is True            # « 2500 à 3000 † »
    assert any(c.kg_m2 is None for _, champs in d.abaques["OB"][80].lignes.values() for c in champs)   # « bleu vif, valeur non imprimée »
    assert [z.lettre for z in d.zones["F"]] == list("ABCDEFGH") and [z.lettre for z in d.zones["OB"]] == list("ABCDEFGHI")
    assert [z.lettre for z in d.zones["S"]] == list("ABCD")
    zd = d.zones["S"][3]
    assert zd.wv == (1601, 2250) and zd.wv_abaque == 2400 and zd.hv == (510, 800, 1075)      # INC-619
    assert d.hv_mini["F"] == {"1 vantail": 520, "2 vantaux": 780} and d.hv_mini["visible"] == d.hv_mini["OB"]
    assert d.soufflet_kg == 110 and d.pmr_minimal == (24, 26)
    assert (d.precision_abaque, d.precision_dagger) == (50, 100)      # lues sur la page des abaques
    assert [r[0] for r in d.articulations] == ["≤ 80", "≤ 110", "≤ 160"]
    assert d.cale_6319 == 1200 and d.cale_6057 == (1200, 4)
    # la quincaillerie visible
    assert list(d.paumelles) == ["T940025", "T940087", "T940026", "T940035", "T940023"]
    assert (d.paumelles["T940025"]["poids"], d.paumelles["T940025"]["lv"]) == (70, 1400)
    assert d.paumelles["T940023"]["poids"] == 130 and d.paumelles["T940035"]["pivot"] is True
    assert d.cas_paumelles["T940026"] == [("1", "T940026", "2"), ("2", "T940026", "2"), ("2", "T940024", "1"),
                                          ("3", "T940026", "3")]
    assert d.pivot_sans_pmr is True
    # les cotes indicées des ensembles menuisés
    assert sorted(d.cotes) == ["H1", "H2", "H3", "H4", "L1", "L2", "L3", "L4", "L5"]
    assert d.cotes["H3"] == ["equilibre"] and d.cotes["L1"] == ["ensemble", "equilibre"]
    assert d.densite_verre == 2.5
    assert d.images["T215180"].endswith(".png")


def test_une_page_changee_dans_le_wiki_casse_la_lecture(snap, monkeypatch):
    """L'en-tête d'un tableau, la phrase de validité ou une section qui change fait échouer le chargement."""
    corps = _corps(snap, "/profiles/soleal-fy-qc-55-applications-ouvrant-apparent.md")
    sec = "Châssis 1 vantail, FY 55, ouvrant Apparent"
    assert len(fy.lire_application(corps, "p", sec, "x", "x").profils) == 6
    with pytest.raises(DonneeIntrouvable):
        fy.lire_application(corps.replace("| Profilé | Désignation | Qté | Coupe | Débit (mm) |",
                                          "| Profilé | Désignation | Qté | Coupe | Longueur (mm) |"), "p", sec, "x", "x")
    with pytest.raises(DonneeIntrouvable):
        fy.lire_application(corps, "p", "Châssis 3 vantaux, FY 55, ouvrant Apparent", "x", "x")
    page = fy.PAGE_DTA
    body = snap.pages[page].body
    assert "| Type de fenêtre | Ouvrants | Hauteur de baie H (m) | Largeur de baie L (m) |" in body
    monkeypatch.setitem(snap.pages, page, replace(snap.pages[page], body=body.replace("| Hauteur de baie H (m) |", "| Hauteur H (m) |")))
    with pytest.raises(DonneeIntrouvable):
        fy._charge(snap)
    # la validité est lue dans le texte de la page : sans elle, pas de calcul
    monkeypatch.setitem(snap.pages, page, replace(snap.pages[page], body=body,
                                                   raw_text=snap.pages[page].raw_text.replace("valide du", "valable du")))
    with pytest.raises(DonneeIntrouvable, match="validité"):
        fy._charge(snap)


# ---- le DTA : limites, ce qu'il ne couvre pas, validité -----------------------------------------------------------


def test_dta_oracle_2_vantaux_fya(snap):
    """2 vantaux OF FYa : 2,20 × 1,50 m dans le DTA (limite 2,225 × 1,55), 2,30 × 1,50 hors."""
    dedans = _ctl(fy.verifier(snap, dict(V2, hauteur_mm=2200, largeur_mm=1500)), "dta")
    assert dedans["statut"] == OK and dedans["valeurs"]["dans"] is True and dedans["valeurs"]["limite"] == {"h": 2225, "l": 1550}
    assert "hypothèse" in dedans["detail"] and "Soleal 55 FYa" in dedans["detail"]
    dehors = fy.verifier(snap, dict(V2, hauteur_mm=2300, largeur_mm=1500))
    assert _ctl(dehors, "dta")["statut"] == HORS and dehors["verdict"] == HORS and dehors["contrainte"]["id"] == "dta"
    assert "Certificat de Qualification" in _ctl(dehors, "dta")["detail"]


def test_dta_ouvrant_minimal_et_autres_types(snap):
    mini = _ctl(fy.verifier(snap, dict(V2, ouvrant="Minimal", hauteur_mm=2210, largeur_mm=1500)), "dta")
    assert mini["statut"] == HORS and mini["valeurs"]["limite"] == {"h": 2200, "l": 1550} and "INC-403" in mini["anomalies"]
    ob = _ctl(fy.verifier(snap, {"configuration": "v1_ob", "largeur_mm": 1100, "hauteur_mm": 1500}), "dta")
    assert ob["statut"] == OK and ob["valeurs"]["limite"] == {"h": 1575, "l": 1150}
    souf = _ctl(fy.verifier(snap, {"configuration": "soufflet", "largeur_mm": 1200, "hauteur_mm": 580}), "dta")
    assert souf["statut"] == HORS and souf["valeurs"]["limite"] == {"h": 575, "l": 1250}
    fixe_lat = _ctl(fy.verifier(snap, {"configuration": "v2_fixe", "largeur_mm": 2300, "hauteur_mm": 2200}), "dta")
    assert fixe_lat["statut"] == OK and fixe_lat["valeurs"]["limite"] == {"h": 2225, "l": 2350}
    # une porte-fenêtre à seuil se lit comme le vantail du tableau (« fenêtres et portes-fenêtres à 1 ou 2 vantaux »)
    pf = _ctl(fy.verifier(snap, {"configuration": "pf2", "largeur_mm": 1500, "hauteur_mm": 2100, "generation": "6057"}), "dta")
    assert pf["statut"] == OK and pf["valeurs"]["type_dta"] == "2 vantaux OF"


def test_dta_ne_couvre_que_le_fy_55_et_pas_le_seuil_pmr(snap):
    r65 = fy.verifier(snap, dict(V2, module="65"))
    d65 = _ctl(r65, "dta")
    assert d65["statut"] == ETUDE and "Aucun DTA pour le SOLEAL FY 65" in d65["detail"] and "VER-392" in d65["anomalies"]
    assert r65["verdict"] == ETUDE and r65["contrainte"]["id"] == "dta"
    pmr = _ctl(fy.verifier(snap, {"configuration": "v2_pmr", "largeur_mm": 1200, "hauteur_mm": 2200}), "dta")
    assert pmr["statut"] == ETUDE and "seuil" in pmr["detail"] and "CTR-141" in pmr["anomalies"]
    # des types que le tableau ne porte pas : pas de limite inventée
    for conf in ("fixe", "v2_ob"):
        c = _ctl(fy.verifier(snap, {"configuration": conf, "largeur_mm": 1000, "hauteur_mm": 1000, "generation": "6057"}), "dta")
        assert c["statut"] == ETUDE and "n'a pas de ligne" in c["detail"], conf


def test_dta_la_validite_est_lue_et_affichee(snap):
    """25/05/2023 → 31/10/2026, lue sur la page ; affichée sur le contrôle, dans les avertissements, dans les valeurs."""
    c = _ctl(fy.verifier(snap, V2), "dta")
    assert "valide du 25/05/2023 au 31/10/2026" in c["detail"] and "DTA 6/12-2016_V5" in c["detail"]
    assert c["valeurs"]["validite"] == {"du": "2023-05-25", "au": "2026-10-31", "jours_restants": 152, "expire": False}
    assert any("25/05/2023 au 31/10/2026" in a for a in fy.options(snap)["avertissements"])
    # la date vient du wiki : le texte de la page en porte la source
    assert "valide du 25/05/2023 au 31/10/2026" in snap.pages[fy.PAGE_DTA].raw_text
    # aucune des deux dates n'est écrite dans le module
    source = open(fy.__file__, encoding="utf-8").read()
    assert "31/10" not in source and "25/05" not in source


def test_dta_a_l_approche_de_l_echeance_puis_echu(snap, monkeypatch):
    monkeypatch.setattr(fy, "_aujourdhui", lambda: datetime.date(2026, 10, 21))
    proche = _ctl(fy.verifier(snap, V2), "dta")
    assert proche["statut"] == OK and "expire dans 10 jour(s)" in proche["detail"]
    monkeypatch.setattr(fy, "_aujourdhui", lambda: datetime.date(2026, 10, 31))
    assert _ctl(fy.verifier(snap, V2), "dta")["statut"] == OK            # valide le jour même de l'échéance
    monkeypatch.setattr(fy, "_aujourdhui", lambda: datetime.date(2026, 11, 1))
    r = fy.verifier(snap, V2)
    echu = _ctl(r, "dta")
    assert echu["statut"] == ETUDE and "échu le 31/10/2026" in echu["detail"] and echu["valeurs"]["validite"]["expire"] is True
    assert r["verdict"] == ETUDE and r["contrainte"]["id"] == "dta"
    # un avis échu n'efface pas un « hors domaine » : la baie reste hors limite
    hors = _ctl(fy.verifier(snap, dict(V2, hauteur_mm=2300, largeur_mm=1500)), "dta")
    assert hors["statut"] == HORS and "échu le 31/10/2026" in hors["detail"]
    assert "échu" in fy.options(snap)["avertissements"][0]


# ---- l'inertie du châssis 2 vantaux -----------------------------------------------------------------------------


def test_inertie_oracles_courbe_1_fy55_apparent(snap):
    """W = 2,0 m : H maxi 2,76 m. H = 2,70 → réalisable ; H = 2,75 → sur étude (à moins de ±0,03 m) ; W = 1,0 → sur étude (« - »)."""
    def inertie(**k):
        return _ctl(fy.verifier(snap, dict(V2, **k)), "inertie")
    ok = inertie(largeur_mm=2000, hauteur_mm=2700)
    assert ok["statut"] == OK and ok["valeurs"]["h_max"] == 2760 and "±30" in ok["detail"]
    assert inertie(largeur_mm=2000, hauteur_mm=2750)["statut"] == ETUDE
    assert inertie(largeur_mm=2000, hauteur_mm=2800)["statut"] == HORS
    nontracee = inertie(largeur_mm=1000, hauteur_mm=2400)
    assert nontracee["statut"] == ETUDE and nontracee["valeurs"]["h_max"] is None and "n'est pas tracée" in nontracee["detail"]
    # entre deux graduations : la plus restrictive, jamais d'interpolation
    entre = inertie(largeur_mm=2200, hauteur_mm=2600)
    assert entre["statut"] == OK and entre["valeurs"]["h_max"] == 2630 and "la plus restrictive" in entre["detail"]
    assert inertie(largeur_mm=2200, hauteur_mm=2640)["statut"] == ETUDE
    # une graduation sans valeur à côté d'une valeur : « - » n'est pas étendue
    assert inertie(largeur_mm=1200, hauteur_mm=2400)["statut"] == ETUDE
    assert inertie(largeur_mm=3600, hauteur_mm=2400)["statut"] == ETUDE


def test_inertie_minimal_et_fy65(snap):
    mini = _ctl(fy.verifier(snap, dict(V2, ouvrant="Minimal", largeur_mm=1000, hauteur_mm=2400)), "inertie")
    assert mini["statut"] == OK and mini["valeurs"]["h_max"] == 2850 and mini["valeurs"]["montants"] == "TFY1338 + T215186"
    ok65 = _ctl(fy.verifier(snap, dict(V2, module="65", largeur_mm=2500, hauteur_mm=2800)), "inertie")
    assert ok65["statut"] == OK and ok65["valeurs"]["h_max"] == 2880 and ok65["valeurs"]["courbe"] == 2
    assert "VER-392" in ok65["anomalies"]
    assert _ctl(fy.verifier(snap, dict(V2, module="65", largeur_mm=2500, hauteur_mm=2870)), "inertie")["statut"] == ETUDE


def test_inertie_ne_decide_que_hors_dta(snap):
    """Dans le DTA (L ≤ 1,55 m), la courbe est donnée à titre d'information : une courbe non tracée ne bloque pas."""
    r = fy.verifier(snap, dict(V2, largeur_mm=1200, hauteur_mm=1350))
    c = _ctl(r, "inertie")
    assert c["statut"] == INFO and "ne décide pas" in c["detail"] and "n'est pas tracée" in c["detail"]
    assert r["verdict"] == OK
    # seulement pour un châssis à 2 vantaux
    assert "inertie" not in _ids(fy.verifier(snap, V1)) and "inertie" in _ids(fy.verifier(snap, V2))
    assert "inertie" in _ids(fy.verifier(snap, {"configuration": "v2_fixe", "largeur_mm": 2000, "hauteur_mm": 2000}))


# ---- la quincaillerie cachée ---------------------------------------------------------------------------------------


def test_qc_champ_zone_et_valeur_lue_sans_comparaison(snap):
    """2 vantaux, H = 1350, W = 1200 : vantail 575 × 1306 ; zone C ; les rangées 500 et 600 de l'abaque 80 kg disent 80 kg/m²."""
    r = fy.verifier(snap, V2)
    champ, zone, art = _ctl(r, "ferrure_champ"), _ctl(r, "ferrure_zone"), _ctl(r, "ferrure")
    assert champ["statut"] == OK and champ["valeurs"] == {"hv": 1306, "wv": 575, "classe": 80}
    assert "hypothèse" in champ["detail"] and "780" in champ["detail"] and "VER-406" in champ["anomalies"]
    assert zone["statut"] == INFO and zone["valeurs"] == {"zones": ["C"], "kg_m2": [80.0]}
    assert "ne dit pas ce que cette valeur mesure" in zone["detail"] and "n'est pas comparée" in zone["detail"]
    assert art["statut"] == INFO and "TFZ60000" in art["detail"] and "180°" in art["detail"]
    assert r["verdict"] == OK


def test_qc_la_valeur_en_kg_par_m2_n_est_jamais_decisive(snap):
    """Quelle que soit la valeur lue (20 à 80 kg/m²), le contrôle de zone reste une information."""
    for h in (900, 1350, 1800, 2300):
        z = _ctl(fy.verifier(snap, dict(V1, hauteur_mm=h)), "ferrure_zone")
        assert z["statut"] == INFO and z["valeurs"]["kg_m2"], h


def test_qc_frontiere_de_deux_zones(snap):
    """Hv = 1200 exactement : bandes « jusqu'à 1200 » et « 1200 à 1600 », zones A et C — la planche ne départage pas."""
    z = _ctl(fy.verifier(snap, dict(V1, hauteur_mm=1244)), "ferrure_zone")
    assert z["valeurs"]["zones"] == ["A", "C"] and "frontière" in z["detail"]


def test_qc_hors_du_champ_de_l_abaque(snap):
    large = _ctl(fy.verifier(snap, dict(V1, largeur_mm=1700)), "ferrure_champ")
    assert large["statut"] == HORS and "hors de l'abaque" in large["detail"]
    haut = _ctl(fy.verifier(snap, dict(V1, largeur_mm=1244, hauteur_mm=3100)), "ferrure_champ")
    assert haut["statut"] == HORS and "3000" in haut["detail"]
    # au-dessous de la diagonale du bas : la zone est blanche
    bas = _ctl(fy.verifier(snap, dict(V1, largeur_mm=1244, hauteur_mm=700)), "ferrure_champ")
    assert bas["statut"] == HORS and "850" in bas["detail"]
    # à moins de la précision de lecture (±50 mm) du haut du champ : sur étude, pas un « oui »
    limite = _ctl(fy.verifier(snap, dict(V1, largeur_mm=1244, hauteur_mm=3040)), "ferrure_champ")     # Hv = 2996, champ à 3000
    assert limite["statut"] == ETUDE
    # la hauteur mini du vantail : 2 vantaux, 780 mm
    mini = _ctl(fy.verifier(snap, dict(V2, hauteur_mm=800)), "ferrure_champ")
    assert mini["statut"] == HORS


def test_qc_classe_de_poids_face_au_verre(snap):
    """Le poids est celui du verre seul (hypothèse affichée) : au-delà de la classe c'est « hors », en deçà une info."""
    lourd = fy.verifier(snap, dict(V1, hauteur_mm=2200, verre_mm=40, epaisseur_vitrage_mm=50))
    poids = _ctl(lourd, "poids")
    assert poids["valeurs"]["poids_verre_kg"] == 176.3 and "hypothèse" in poids["detail"] and "VER-190" in poids["anomalies"]
    f = _ctl(lourd, "ferrure")
    assert f["statut"] == HORS and "aucune" in f["detail"] and lourd["verdict"] == HORS
    moyen = fy.verifier(snap, dict(V1, hauteur_mm=2200, verre_mm=30, epaisseur_vitrage_mm=40))
    assert _ctl(moyen, "ferrure")["statut"] == HORS and "160" in _ctl(moyen, "ferrure")["detail"]
    ok160 = fy.verifier(snap, dict(V1, hauteur_mm=2200, verre_mm=30, epaisseur_vitrage_mm=40, ferrure_kg="160"))
    f160 = _ctl(ok160, "ferrure")
    assert f160["statut"] == INFO and "INC-601" in f160["anomalies"] and "90°" in f160["detail"]
    assert "INC-601" in _ctl(ok160, "ferrure_champ")["anomalies"]


def test_qc_ob_et_soufflet(snap):
    ob = fy.verifier(snap, {"configuration": "v1_ob", "largeur_mm": 1000, "hauteur_mm": 1350, "ferrure_kg": "110"})
    assert _ctl(ob, "ferrure_champ")["statut"] == OK and "OB / BO" in _ctl(ob, "ferrure_champ")["detail"]
    z = _ctl(ob, "ferrure_zone")
    assert z["valeurs"]["zones"] == ["E"] and "« Zones D & E »" in z["detail"]
    assert "TFZ60002" in _ctl(ob, "ferrure")["detail"]
    # le soufflet : une seule classe, 110 kg, lue sur la page
    s = lambda **k: fy.verifier(snap, dict({"configuration": "soufflet", "largeur_mm": 900, "hauteur_mm": 700}, **k))
    ok = s()
    assert _ctl(ok, "ferrure_champ")["statut"] == OK and _ctl(ok, "ferrure_champ")["valeurs"]["zone"] == "B"
    assert "110 kg" in _ctl(ok, "ferrure")["detail"]
    hors = _ctl(s(hauteur_mm=550), "ferrure_champ")                      # Hv = 506 < 510
    assert hors["statut"] == HORS and "INC-619" in hors["anomalies"]
    # la limite haute de la zone C descend en ligne droite de 1200 à 1075 : l'entre-deux n'est pas relevé
    descend = _ctl(s(largeur_mm=1300, hauteur_mm=1144), "ferrure_champ")
    assert descend["statut"] == ETUDE and "n'est pas relevée" in descend["detail"]
    # la zone D : 2250 sur le titre des planches, 2400 sur l'abaque (INC-619)
    d2300 = _ctl(s(largeur_mm=2350, hauteur_mm=700), "ferrure_champ")
    assert d2300["statut"] == ETUDE and "2250" in d2300["detail"]
    assert _ctl(s(largeur_mm=2500, hauteur_mm=700), "ferrure_champ")["statut"] == HORS


def test_qc_fy65_n_est_pas_disponible_en_lumine65(snap):
    f = _ctl(fy.verifier(snap, dict(V2, module="65")), "ferrure")
    assert f["statut"] == ETUDE and "n'est pas disponible en LUMINE65" in f["detail"] and "VER-392" in f["anomalies"]
    assert "ferrure_champ" not in _ids(fy.verifier(snap, dict(V2, module="65")))


# ---- la quincaillerie visible (catalogue 6057) --------------------------------------------------------------------


def test_visible_paumelle_poids_et_largeur(snap):
    r = fy.verifier(snap, V2_6057)
    p = _ctl(r, "paumelle")
    assert p["statut"] == INFO and p["valeurs"]["poids_maxi_kg"] == 70 and p["valeurs"]["lv_maxi_mm"] == 1400
    assert {"VER-190", "VER-370"} <= set(p["anomalies"]) and "hypothèse" in p["detail"]
    assert "ferrure_champ" not in _ids(r)                    # pas d'abaque de quincaillerie cachée en 6057
    # la largeur du vantail au-delà de Lv maxi : hors
    large = _ctl(fy.verifier(snap, dict(V1, largeur_mm=1500, generation="6057")), "paumelle")
    assert large["statut"] == HORS and "Lv" in large["detail"]
    # le poids du verre au-delà du poids maxi de la paumelle : hors ; une paumelle plus forte le reçoit
    lourd = dict(V1, hauteur_mm=2200, verre_mm=28, epaisseur_vitrage_mm=40, generation="6057")
    assert _ctl(fy.verifier(snap, lourd), "paumelle")["statut"] == HORS
    assert _ctl(fy.verifier(snap, dict(lourd, paumelle="T940023")), "paumelle")["statut"] == INFO
    assert "VER-372" in _ctl(fy.verifier(snap, dict(lourd, paumelle="T940023")), "paumelle")["anomalies"]


def test_visible_le_nombre_de_paumelles_n_est_pas_calcule(snap):
    n = _ctl(fy.verifier(snap, V2_6057), "paumelles_nombre")
    assert n["statut"] == INFO and "VER-370" in n["anomalies"] and "n'est pas calculé" in n["detail"]
    assert "cas 3 : 4 T940025 ou T940087" in n["detail"]               # les quantités écrites par cas, pas un choix de cas
    assert n["valeurs"]["cas"][0] == ["1", "T940025 ou T940087", "2"]


def test_visible_le_pivot_n_est_pas_compatible_avec_le_seuil_pmr(snap):
    r = fy.verifier(snap, {"configuration": "v1_pmr", "largeur_mm": 1000, "hauteur_mm": 2000, "generation": "6057",
                           "paumelle": "T940035"})
    p = _ctl(r, "paumelle")
    assert p["statut"] == HORS and "pivot ne sont pas compatibles avec le seuil PMR" in p["detail"]
    assert {"VER-369", "VER-371"} <= set(p["anomalies"])
    ok = _ctl(fy.verifier(snap, {"configuration": "v1_pmr", "largeur_mm": 1000, "hauteur_mm": 2000, "generation": "6057"}),
              "paumelle")
    assert ok["statut"] == INFO


def test_ensemble_menuise_la_paumelle_n_est_pas_controlee(snap):
    """Les vitrages d'un ensemble n'ont pas de repère de position (VER-360) : on ne sait pas lequel est le vantail."""
    saisie = {"configuration": "ensemble", "largeur_mm": 2000, "hauteur_mm": 2000, "generation": "6057", "cote_h1_mm": 800,
              "cote_h2_mm": 1200, "cote_l1_mm": 1000, "cote_l2_mm": 1000}
    r = fy.verifier(snap, saisie)
    assert _ctl(r, "paumelle")["statut"] == ETUDE and "VER-360" in _ctl(r, "paumelle")["anomalies"]
    manque = fy.verifier(snap, {k: v for k, v in saisie.items() if k != "cote_l2_mm"})
    assert _ctl(manque, "cotes")["statut"] == ETUDE and "L2" in _ctl(manque, "cotes")["detail"]


# ---- le vitrage et la cale ------------------------------------------------------------------------------------------


def test_vitrage_parclose_et_joint(snap):
    v = _ctl(fy.verifier(snap, V1), "vitrage")
    assert v["statut"] == OK and "T591005" in v["detail"] and v["valeurs"]["epaisseur"] == 24 and "CTR-142" in v["anomalies"]
    assert _ctl(fy.verifier(snap, dict(V1, epaisseur_vitrage_mm=32)), "vitrage")["statut"] == OK
    m = _ctl(fy.verifier(snap, dict(V1, ouvrant="Minimal")), "vitrage")
    assert m["statut"] == OK and "TFY1338" in m["detail"]
    # au-delà du DTA (32 mm) : seul le catalogue prévoit plus (CTR-142)
    plus = _ctl(fy.verifier(snap, dict(V1, epaisseur_vitrage_mm=40)), "vitrage")
    assert plus["statut"] == ETUDE and "jusqu'à 32 mm" in plus["detail"]
    # le verre seul ne peut pas égaler l'épaisseur du vitrage
    assert _ctl(fy.verifier(snap, dict(V1, verre_mm=24)), "vitrage")["statut"] == ETUDE


def test_vitrage_fy65_et_seuil_pmr_minimal(snap):
    a65 = _ctl(fy.verifier(snap, dict(V1, module="65")), "vitrage")
    assert a65["statut"] == ETUDE and "FY 65" in a65["detail"] and "VER-395" in a65["anomalies"]
    m65 = _ctl(fy.verifier(snap, dict(V1, module="65", ouvrant="Minimal")), "vitrage")
    assert "CTR-218" in m65["anomalies"]
    pmr = {"configuration": "v1_pmr", "largeur_mm": 1000, "hauteur_mm": 2200, "ouvrant": "Minimal"}
    assert _ctl(fy.verifier(snap, dict(pmr, epaisseur_vitrage_mm=26)), "vitrage")["statut"] == OK
    sans = _ctl(fy.verifier(snap, dict(pmr, epaisseur_vitrage_mm=32)), "vitrage")
    assert sans["statut"] == ETUDE and "24 et 26" in sans["detail"]


def test_cale_au_dela_de_1200_mm_selon_le_catalogue(snap):
    c = _ctl(fy.verifier(snap, V1), "cale")
    assert c["statut"] == INFO and "CTR-219" in c["anomalies"] and "6319" in c["detail"] and "1200" in c["detail"]
    assert "cale" not in _ids(fy.verifier(snap, dict(V1, hauteur_mm=1100)))
    c57 = _ctl(fy.verifier(snap, dict(V1, generation="6057", ouvrant="Minimal")), "cale")
    assert "6057" in c57["detail"] and "4 mm" in c57["detail"] and "Minimale" in c57["detail"]


# ---- la liste de coupe : les oracles du catalogue 6319 ---------------------------------------------------------------


def test_fiche_oracle_1_vantail_6319(snap):
    """1 vantail H = 1350, W = 1000 : ouvrants 2×1306 et 2×956 ; parcloses 2×1175 et 2×869 ; dormant 2×1350 et 2×1000 ;
    vitrage 1207 × 857 ; joints 2H+2W = 4700 mm."""
    f = fy.fiche_debit(snap, V1)
    assert f["verdict"] == OK and f["generation"] == "6319"
    assert _lignes(f, "T215180") == [(2, 1306), (2, 956)] and _lignes(f, "T591005") == [(2, 1175), (2, 869)]
    assert _lignes(f, "TFY1119") == [(2, 1350), (2, 1000)]
    assert [l["longueur"] for l in f["lignes"] if l["famille"] == "Joints"] == [4700] * 4
    rempl = next(l for l in f["lignes"] if l["famille"] == "Remplissage")
    assert "1207 × 857" in rempl["calcul"] and rempl["qte"] == 1
    assert all("T410009" != l["ref"] for l in f["lignes"])             # la ligne du 6057 n'entre jamais dans le 6319


def test_fiche_oracle_2_vantaux_6319(snap):
    """2 vantaux H = 1350, W = 1200 : ouvrant 4×1306, 4×575 ; battement T215186 1234 ; parcloses 4×1175, 4×488 ;
    vitrage 1207 × 476 ; T410010 4H+2W = 7800 ; TFY5024 3H+2W = 6450."""
    f = fy.fiche_debit(snap, V2)
    assert _lignes(f, "T215180") == [(4, 1306), (4, 575)] and _lignes(f, "T215186") == [(1, 1234)]
    assert _lignes(f, "T591005") == [(4, 1175), (4, 488)]
    assert _lignes(f, "T410010") == [(1, 7800)] and _lignes(f, "TFY5024") == [(1, 6450)]
    rempl = next(l for l in f["lignes"] if l["famille"] == "Remplissage")
    assert "1207 × 476" in rempl["calcul"] and rempl["qte"] == 2
    nomen = {n["ref"]: n for n in f["nomenclature"]}
    assert nomen["T215180"]["qte"] == 8 and nomen["T215180"]["ml"] == 7.52


def test_fiche_forme_sources_et_symboles_de_coupe(snap):
    f = fy.fiche_debit(snap, V2)
    assert set(f) >= {"lignes", "nomenclature", "accessoires", "notes", "brouillons", "verdict", "libelle"}
    ligne = next(l for l in f["lignes"] if l["ref"] == "T215180")
    assert set(ligne) >= {"rep", "famille", "piece", "ref", "qte", "longueur", "calcul", "coupe", "source", "note", "image"}
    assert ligne["coupe"] == "pictogramme « /\\ » (non légendé)" and ligne["image"]
    assert ligne["source"]["page"].endswith("soleal-fy-qc-55-applications-ouvrant-apparent.md")
    assert ligne["source"]["section"] == "Châssis 2 vantaux, FY 55, ouvrant Apparent"
    droite = next(l for l in f["lignes"] if l["ref"] == "T591005")
    assert droite["coupe"] == "pictogramme « || » (non légendé)"           # la barre échappée des planches est recollée
    assert any("VER-396" in n for n in f["notes"]) and f["brouillons"]


def test_fiche_quantites_sans_nombre_et_par_metre_lineaire(snap):
    f = fy.fiche_debit(snap, V2)
    # « *** » : jamais un nombre inventé, on le dit
    assert not _lignes(f, "T740012") and any("T740012" in n and "« *** »" in n for n in f["notes"])
    # « 3/ml/H » : 3 par mètre linéaire de H = 4,05 pour H = 1350, arrondi non écrit
    vis = next(l for l in f["lignes"] if l["ref"] == "T770011")
    assert vis["qte"] == pytest.approx(4.05) and "arrondi" in vis["note"]
    f57 = fy.fiche_debit(snap, V2_6057)
    assert any("T940025" in n and "voir abaque" in n and "VER-370" in n for n in f57["notes"])
    assert any("T940031" in n and "voir abaque" in n for n in f57["notes"])
    assert not _lignes(f57, "T940025")
    # une quantité absente (« - ») est dite absente, pas renvoyée à un abaque
    eq = fy.fiche_debit(snap, {"configuration": "equilibre", "largeur_mm": 2000, "hauteur_mm": 2000, "generation": "6057",
                               "ouvrant": "Minimal"})
    assert any("T720059" in n and "quantité non imprimée" in n for n in eq["notes"])


# ---- deux générations, jamais fusionnées ----------------------------------------------------------------------------


def test_fiche_6057_memes_valeurs_autres_references_de_joint(snap):
    f = fy.fiche_debit(snap, V2_6057)
    assert f["generation"] == "6057"
    assert _lignes(f, "T215180") == [(4, 1306), (4, 575)] and _lignes(f, "T591005") == [(4, 1175), (4, 488)]
    assert _lignes(f, "T410009") == [(1, 6450)] and _lignes(f, "TFY5008") == [(1, 7800)]
    assert not _lignes(f, "TFY5024") and not _lignes(f, "TCZ50001") and not _lignes(f, "TFY3868")    # rien du 6319
    rempl = next(l for l in f["lignes"] if l["famille"] == "Remplissage")
    assert "1207 × 476" in rempl["calcul"] and "L/2-124" in rempl["calcul"]
    prof = next(l for l in f["lignes"] if l["ref"] == "T215180")
    assert prof["coupe"].startswith("type de coupe « onglet / onglet »") and "VER-352" in prof["coupe"]
    assert prof["source"]["page"] == "/profiles/soleal-fy-cotes-de-debit.md"


def test_l_ecart_entre_les_deux_generations_est_affiche(snap):
    g = _ctl(fy.verifier(snap, V1), "generation")
    assert g["statut"] == INFO and "Catalogue 6319.003 (QC, courant)" in g["detail"] and "jamais fusionnés" in g["detail"]
    assert "joints seulement en 6057 : T410009, TFY5008" in g["detail"] and "TFY5024" in g["detail"]
    assert "VER-396" in g["anomalies"] and len(g["sources"]) == 2 and g["valeurs"]["ecarts"]
    g57 = _ctl(fy.verifier(snap, dict(V1, generation="6057")), "generation")
    assert "Catalogue 6057.003 (visible, ancien)" in g57["detail"] and "joints seulement en 6319" in g57["detail"]
    assert "VER-352" in g57["anomalies"]
    # le courant par défaut
    assert fy.options(snap)["champs"][3]["defaut"] == "6319"
    # le FY 65 n'a pas de contrepartie en 6057
    assert "n'a pas cette application" in _ctl(fy.verifier(snap, dict(V1, module="65")), "generation")["detail"]


def test_une_application_absente_d_un_catalogue_est_dite_pas_remplacee_par_l_autre(snap):
    r = fy.verifier(snap, {"configuration": "fixe", "largeur_mm": 1000, "hauteur_mm": 1000})
    g = _ctl(r, "generation")
    assert "n'a pas de nomenclature" in g["detail"] and "6057.003" in g["detail"] and "à choisir explicitement" in g["detail"]
    assert _ctl(r, "application")["statut"] == ETUDE and r["verdict"] == ETUDE
    f = fy.fiche_debit(snap, {"configuration": "fixe", "largeur_mm": 1000, "hauteur_mm": 1000})
    assert "n'a pas de nomenclature" in f["indisponible"] and "pas mélangés" in f["indisponible"]
    ok = fy.fiche_debit(snap, {"configuration": "fixe", "largeur_mm": 1000, "hauteur_mm": 1000, "generation": "6057"})
    assert _lignes(ok, "T591005") == [(2, 902), (2, 946)] and _lignes(ok, "TFY1119") == [(2, 1000), (2, 1000)]
    assert "n'a pas de nomenclature" in fy.fiche_debit(snap, {"configuration": "v2_fixe", "largeur_mm": 2000,
                                                                "hauteur_mm": 2000})["indisponible"]


# ---- ensembles menuisés, Equilibre et le seuil PMR -------------------------------------------------------------------------


def test_fiche_ensemble_menuise_avec_ses_cotes(snap):
    """H2 = 1200, L1 = 1000 : ouvrant L1-37 = 963 et H2-37 = 1163 ; quatre vitrages dont H2-136 × L1-136 = 1064 × 864 ;
    TAY0001 : 24 vis."""
    saisie = {"configuration": "ensemble", "largeur_mm": 2000, "hauteur_mm": 2000, "generation": "6057", "cote_h1_mm": 800,
              "cote_h2_mm": 1200, "cote_l1_mm": 1000, "cote_l2_mm": 1000}
    f = fy.fiche_debit(snap, saisie)
    assert _lignes(f, "T215180") == [(2, 963), (2, 1163)] and _lignes(f, "TAY0001") == [(24, None)]
    rempl = [l["calcul"] for l in f["lignes"] if l["famille"] == "Remplissage"]
    assert len(rempl) == 4 and "H2-136 × L1-136 = 1064 × 864" in rempl[0] and "H2-59 × L2-59 = 1141 × 941" in rempl[3]
    assert any("VER-353" in n and "VER-360" in n for n in f["notes"])
    # sans les cotes : les lignes qui en dépendent ne sont pas calculées, et on le dit
    sans = fy.fiche_debit(snap, {k: v for k, v in saisie.items() if not k.startswith("cote_")})
    assert any("Cotes H1, H2, L1, L2 non saisies" in n for n in sans["notes"])
    assert not _lignes(sans, "T215180")


def test_equilibre_la_cote_de_vitrage_prefixee_n_est_pas_lue(snap):
    """« 2H3-98 / 2L5/2-83 » : le préfixe 2 n'est défini nulle part (VER-357) ; ni lu comme 2 × H3 − 98, ni calculé."""
    cotes = {f"cote_{v}_mm": 1000 for v in ("h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4", "l5")}
    f = fy.fiche_debit(snap, dict({"configuration": "equilibre", "largeur_mm": 2000, "hauteur_mm": 2000, "generation": "6057",
                                   "ouvrant": "Minimal"}, **cotes))
    notes = " ".join(f["notes"])
    assert "2H3-98 [préfixe non défini]" in notes and "VER-357" in notes
    assert not any("2H3" in l["calcul"] for l in f["lignes"] if l["famille"] == "Remplissage")
    assert len([l for l in f["lignes"] if l["famille"] == "Remplissage"]) == 8


def test_seuil_pmr_6319_les_formules_propres_au_seuil(snap):
    """PMR Apparent FY 55, H = 2200, W = 1000 : ouvrant H-23,5 = 2176,5 ; seuil TFY1160 W-104 = 896 ; vitrage H-123 × W-143."""
    f = fy.fiche_debit(snap, {"configuration": "v1_pmr", "largeur_mm": 1000, "hauteur_mm": 2200})
    assert (2, 2176.5) in _lignes(f, "T215180") and (1, 896) in _lignes(f, "TFY1160")
    rempl = next(l for l in f["lignes"] if l["famille"] == "Remplissage")
    assert "2077 × 857" in rempl["calcul"]
    assert any("CTR-208" in n and "ancienne rédaction" in n for n in f["notes"])
    g = _ctl(fy.verifier(snap, {"configuration": "v1_pmr", "largeur_mm": 1000, "hauteur_mm": 2200}), "generation")
    assert {"CTR-220", "CTR-208"} <= set(g["anomalies"]) and "ne sont pas lues" in g["detail"]
    gm = _ctl(fy.verifier(snap, {"configuration": "v1_pmr", "largeur_mm": 1000, "hauteur_mm": 2200, "ouvrant": "Minimal"}),
              "generation")
    assert "CTR-209" in gm["anomalies"]


# ---- la forme du résultat, les anomalies, les brouillons ---------------------------------------------------------------


def test_la_forme_du_resultat_et_les_sources(snap):
    r = fy.verifier(snap, V2)
    assert {"verdict", "libelle", "contrainte", "controles", "brouillons", "application", "generation", "module", "ouvrant",
            "vitrage", "vantail", "remplissages", "poids_verre_kg"} <= set(r)
    assert r["verdict"] == OK and r["libelle"] == "Réalisable" and r["contrainte"] is None
    assert r["vantail"] == {"hv": 1306, "wv": 575} and r["remplissages"] == [{"libelle": "", "hauteur": 1207, "largeur": 476}]
    assert r["poids_verre_kg"] == 11.5
    for c in r["controles"]:
        assert set(c) == {"id", "titre", "statut", "detail", "sources", "anomalies", "valeurs"}
        assert c["statut"] in (OK, HORS, ETUDE, INFO) and c["detail"] and c["sources"]
        for s in c["sources"]:
            assert s["page"] in snap.pages, s
    # les pages en brouillon que le calcul lit sont dites
    assert fy.PAGE_DTA in r["brouillons"] and "/profiles/soleal-fy-qc-55-applications-ouvrant-apparent.md" in r["brouillons"]
    assert fy.PAGE_INERTIE not in r["brouillons"]                       # stable
    assert r["brouillons"] == sorted(r["brouillons"])


def test_les_anomalies_citees_existent_dans_le_wiki(snap):
    registre = " ".join(snap.pages[p].body for p in snap.pages if p.startswith("/anomalies/"))
    cites = set()
    for saisie in (V2, dict(V2, module="65"), dict(V2, ouvrant="Minimal"), V2_6057,
                   {"configuration": "v2_pmr", "largeur_mm": 1200, "hauteur_mm": 2200, "generation": "6057", "paumelle": "T940035"},
                   {"configuration": "v2_ob", "largeur_mm": 1200, "hauteur_mm": 1350, "ferrure_kg": "160"},
                   {"configuration": "soufflet", "largeur_mm": 900, "hauteur_mm": 700},
                   {"configuration": "v1_pmr", "largeur_mm": 1000, "hauteur_mm": 2200, "ouvrant": "Minimal", "module": "65"}):
        for c in fy.verifier(snap, saisie)["controles"]:
            cites |= set(c["anomalies"])
    assert {"INC-402", "INC-403", "CTR-141", "CTR-142", "CTR-218", "CTR-219", "CTR-220", "VER-190", "VER-352", "VER-369",
            "VER-370", "VER-392", "VER-395", "VER-396", "VER-406", "INC-601", "INC-619"} <= cites
    assert [a for a in sorted(cites) if not re.search(rf"\| {a} \|", registre)] == []


def test_les_pages_lues_existent_et_les_brouillons_sont_dits(snap):
    o = fy.options(snap)
    assert all(p in snap.pages for p in o["pages"]) and fy.PAGE_DTA in o["brouillons"]
    assert fy.PAGE_INERTIE in o["pages"] and fy.PAGE_INERTIE not in o["brouillons"]
    assert all(a for a in o["avertissements"])
    assert any("VER-406" in a for a in o["avertissements"]) and any("VER-370" in a for a in o["avertissements"])


def test_les_resultats_sont_serialisables_en_json(snap):
    """La page reçoit du JSON : aucun tuple de dataclass, aucune date, aucun objet du module ne doit s'y glisser."""
    saisies = [V1, V2, V2_6057, dict(V2, module="65"), dict(V2, ouvrant="Minimal"),
               {"configuration": "v1_pmr", "largeur_mm": 1000, "hauteur_mm": 2200},
               {"configuration": "soufflet", "largeur_mm": 900, "hauteur_mm": 700},
               {"configuration": "v1_ob", "largeur_mm": 1000, "hauteur_mm": 1350, "ferrure_kg": "110"},
               {"configuration": "fixe", "largeur_mm": 1000, "hauteur_mm": 1000, "generation": "6057"},
               {"configuration": "v2_fixe", "largeur_mm": 2000, "hauteur_mm": 2000},
               {"configuration": "equilibre", "largeur_mm": 2000, "hauteur_mm": 2000, "generation": "6057", "ouvrant": "Minimal"}]
    for saisie in saisies:
        json.dumps(fy.verifier(snap, saisie))
        json.dumps(fy.fiche_debit(snap, saisie))
    json.dumps(fy.options(snap))


# ---- par le registre, et la saisie -----------------------------------------------------------------------------------------


def test_le_registre_propose_le_soleal_fy(snap):
    s = next(x for x in reg.systemes()["systemes"] if x["id"] == "soleal-fy")
    assert s["libelle"] == "SOLEAL FY (fenêtre à frappe LUMINE)" and s["gammes"] == ["LUMINE"] and s["legacy"] is False
    assert s["portee"] == fy.PORTEE and "FY 65" in s["portee"] and "6319.003" in s["portee"]
    o = reg.options(snap, "soleal-fy")
    assert {"champs", "configurations", "pages", "brouillons", "avertissements"} <= set(o)
    ids = [c["id"] for c in o["champs"]]
    assert ids[:10] == ["configuration", "largeur_mm", "hauteur_mm", "generation", "module", "ouvrant", "epaisseur_vitrage_mm",
                        "verre_mm", "ferrure_kg", "paumelle"]
    assert set(ids[10:]) == {f"cote_{v}_mm" for v in ("h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4", "l5")}
    assert set(o["configurations"]) == set(fy.TYPOLOGIES) and o["configurations"]["v2"].startswith("Fenêtre")
    r = reg.verifier(snap, "soleal-fy", V2)
    assert r["verdict"] == OK and r["application"].startswith("Fenêtre ou porte-fenêtre 2 vantaux")
    f = reg.fiche_debit(snap, "soleal-fy", V2)
    assert f["lignes"] and f["verdict"] == OK and f["nomenclature"]


def test_la_saisie_est_verifiee_contre_les_champs(snap):
    with pytest.raises(ValueError, match="Typologie"):
        fy.verifier(snap, dict(V2, configuration="coulissant"))
    with pytest.raises(ValueError, match="Largeur hors tout L \\(W\\) : au moins"):
        fy.verifier(snap, dict(V2, largeur_mm=10))
    with pytest.raises(ValueError, match="Catalogue de débit"):
        fy.verifier(snap, dict(V2, generation="1999"))
    with pytest.raises(ValueError, match="Paumelle"):
        fy.verifier(snap, dict(V2_6057, paumelle="T000000"))
    # le module n'est pas montré avec le catalogue 6057 : il n'est pas vérifié, et vaut 55
    r = fy.verifier(snap, dict(V2_6057, module="99"))
    assert r["module"] == "55" and r["generation"] == "6057"
    # la quincaillerie cachée n'est pas montrée en 6057, ni la paumelle en 6319
    assert fy.verifier(snap, dict(V2_6057, ferrure_kg="999"))["verdict"] == OK
    assert fy.verifier(snap, dict(V2, paumelle="T000000"))["verdict"] == OK
