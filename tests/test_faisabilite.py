"""Le vérificateur de faisabilité PERFORM76 : lecture des tableaux du wiki, débit, verdicts.

Les attendus sont recalculés à la main depuis les pages du wiki : si une valeur change dans le
wiki, le test le dit — c'est voulu, un outil qui calcule ne doit pas dériver en silence."""
from __future__ import annotations

import pytest

from app.services import faisabilite as fa
from app.services.faisabilite import Saisie, calcul_debit, donnees, lire_vitrage, verifier
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _ctl(r, id_):
    return next(c for c in r["controles"] if c["id"] == id_)


# ---- la lecture des tableaux : le contrat avec le wiki --------------------------------------

def test_les_tableaux_se_lisent(snap):
    d = donnees(snap)
    assert d.debit_dormants["76171"]["deo"] == 38 and d.debit_meneaux["76372"]["deo"] == 13
    assert d.debit_ouvrants["76281"] == {"dfo": 20, "vitrage": 52, "section": "Cotes de débit des ouvrants"}
    assert d.debit_battements[("76171", "76471")]["deo"] == 32
    assert ("76185", "76471") not in d.debit_battements  # VER-21 : aucune planche
    assert d.renfort_battement == {"76471": "V316", "76472": "V317"}
    assert ("1 vantail OF", 2.15, 1.0) in d.dta
    # l'abaque d'ouvrant simple des ouvrants PROFERM : limites, zones, courbes de verre
    ab = next(a for a in d.abaques if "76281" in a.ouvrants)
    assert ab.renfort == "V266.Z" and ab.zones == (75, 130)
    assert ab.couleurs["blanc"].hauteur_maxi == 235 and ab.couleurs["standard"].largeur_maxi == 120
    assert ab.verre[100][12] == 193 and ab.verre[50][12] is None
    # seules les références que PROFERM propose sont offertes
    assert d.dormants_proferm == ["76171", "76172", "76177", "76180", "76185"]
    assert set(d.ouvrants_proferm) == {"76281", "76275", "76272", "76279"}
    assert any(p["ref"] == "76501" and p["epaisseur"] == 24 and p["image"] for p in d.parcloses_ouvrant)


def test_vitrage():
    v = lire_vitrage("4-16-4")
    assert (v.verre_mm, v.total_mm) == (8, 24)
    v = lire_vitrage("44.2-16-4")  # feuilleté : deux verres de 4, deux films de 0,38
    assert (v.verre_mm, v.total_mm) == (12, 28.76)
    assert lire_vitrage("4-12-4-12-4").verre_mm == 12  # l'exemple du manuel
    with pytest.raises(ValueError):
        lire_vitrage("4-16")


# ---- le débit ------------------------------------------------------------------------------

def test_exemple_du_manuel(snap):
    """DHT 2 000 × 1 200, dormant 76171, meneau 76372, ouvrant 76271 : DEO 949, vitrage 829."""
    deb = calcul_debit(donnees(snap), Saisie("2v_meneau", 2000, 1200, "76171", "76271", meneau="76372"))
    assert deb["deo"][0] == 949
    assert deb["vitrage"][0] == 829


def test_debit_seuil(snap):
    deb = calcul_debit(donnees(snap), Saisie("1v_of", 900, 2150, "76171", "76281", bas="seuil"))
    assert deb["deo"] == (900 - 2 * 38, 2150 - 38 - 10)


# ---- les verdicts --------------------------------------------------------------------------

def test_fenetre_courante_realisable(snap):
    r = verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76281"))
    assert r["verdict"] == "ok", [(c["titre"], c["statut"], c["detail"]) for c in r["controles"]]
    assert _ctl(r, "parclose")["valeurs"]["parcloses"][0]["ref"] == "76501"
    # vantail 82,4 × 132,4 cm : plus large que 75, plus haut que 130 → zone D
    assert _ctl(r, "renfort")["valeurs"]["zone"] == "D"


def test_dta_depasse(snap):
    r = verifier(snap, Saisie("1v_of", 1200, 2300, "76171", "76281"))
    assert r["verdict"] == "hors"
    assert _ctl(r, "dta")["statut"] == "hors"


def test_ob_admet_l_un_des_deux_couples(snap):
    # 1,40 × 1,30 m (H × L) : hors du 2,15 × 1,00, dans le 1,50 × 1,40
    r = verifier(snap, Saisie("1v_ob", 1300, 1400, "76171", "76281"))
    assert _ctl(r, "dta")["statut"] == "ok"


def test_dta_ramene_au_vantail_sur_l_abaque(snap):
    # 1 vantail OF, 1200 × 1400 : dans la zone bleue, hors du DTA 2,15 × 1,00 m, soit 92,4 × 207,4 cm de vantail
    r = verifier(snap, Saisie("1v_of", 1200, 1400, "76171", "76281"))
    assert _ctl(r, "ouvrant")["statut"] == "ok"
    assert _ctl(r, "dta")["statut"] == "hors"
    assert [(q["largeur"], q["hauteur"]) for q in r["graphe"]["dta"]] == [(pytest.approx(92.4), pytest.approx(207.4))]
    # l'OB porte deux couples : deux rectangles
    r = verifier(snap, Saisie("1v_ob", 1300, 1400, "76171", "76281"))
    assert len(r["graphe"]["dta"]) == 2


def test_courbe_de_verre_sans_interpolation(snap):
    # DEO 950 mm = 95 cm, entre les graduations 90 (242) et 100 (193) : la plus restrictive, 193
    base = dict(configuration="1v_of", largeur_mm=950 + 76, dormant="76171", ouvrant="76281", vitrage="4-12-4-12-4")
    ok = verifier(snap, Saisie(hauteur_mm=1850 + 76, **base))          # 185 cm
    assert _ctl(ok, "verre")["statut"] == "ok"
    proche = verifier(snap, Saisie(hauteur_mm=1910 + 76, **base))      # 191 cm : à 2 cm, sous ±3 cm
    assert _ctl(proche, "verre")["statut"] == "etude"
    hors = verifier(snap, Saisie(hauteur_mm=2000 + 76, **base))        # 200 cm
    assert _ctl(hors, "verre")["statut"] == "hors"
    assert "193" in _ctl(hors, "verre")["detail"]


def test_j079_decale_de_deux_courbes(snap):
    base = dict(configuration="1v_of", largeur_mm=950 + 76, hauteur_mm=2000 + 76, dormant="76171",
                ouvrant="76281", vitrage="4-12-4-12-4")
    assert _ctl(verifier(snap, Saisie(j079=True, **base)), "verre")["statut"] == "ok"


def test_plus_de_12_mm_de_verre_impose_le_renfort_total(snap):
    r = verifier(snap, Saisie("1v_of", 800, 1000, "76171", "76281", vitrage="44.2-16-44.2"))
    assert _ctl(r, "renfort")["detail"].startswith("Renforcement total")


def test_ouvrants_76272_et_76279_ont_leur_abaque_simple(snap):
    """V326.Z et V314.Z : mêmes limites, aucune courbe de verre (la table de verre est facultative)."""
    d = donnees(snap)
    ab = fa.abaque_simple(d, "76272")
    assert ab.renfort == "V326.Z ou V314.Z" and ab.verre == {} and ab.zones == (80, 130)
    assert ab.couleurs["blanc"].hauteur_maxi == 250 and ab.couleurs["standard"].largeur_maxi == 140
    assert [x for x in ab.equivalents] and fa.abaque_simple(d, "76279").renfort == ab.renfort
    # vantail 82,4 × 132,4 cm, blanc : dans la limite ; zone D (> 80 et > 130) ; le DTA décide du reste
    for ouvrant in ("76272", "76279"):
        r = verifier(snap, Saisie("1v_of", 900, 1400, "76171", ouvrant))
        assert _ctl(r, "ouvrant")["statut"] == "ok", _ctl(r, "ouvrant")["detail"]
        assert _ctl(r, "renfort")["valeurs"]["zone"] == "D"
        assert _ctl(r, "renfort")["valeurs"]["renfort"] == "V326.Z ou V314.Z"
        assert _ctl(r, "verre")["statut"] == "ok"
        assert r["verdict"] == "ok", [(c["titre"], c["statut"], c["detail"]) for c in r["controles"]]
    # sans courbe d'épaisseur, 12 mm de verre et plus : l'abaque ne dit rien, sur étude
    r = verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76272", vitrage="4-12-4-12-4"))
    assert _ctl(r, "verre")["statut"] == "etude" and r["verdict"] == "etude"
    # la limite de couleur standard (220 cm de haut) borne encore
    r = verifier(snap, Saisie("1v_of", 900, 2350, "76171", "76272", couleur="standard"))
    assert _ctl(r, "ouvrant")["statut"] == "hors"


def test_ouvrant_aux_abaques_differents_garde_le_premier(snap):
    # 76271 : trois abaques aux limites différentes (V306.Z, V307.Z, V308), comme avant le premier
    assert fa.abaque_simple(donnees(snap), "76271").renfort == "V306.Z"
    assert fa.abaque_simple(donnees(snap), "00000") is None


def test_76185_avec_battement_ver21(snap):
    r = verifier(snap, Saisie("2v_battement", 1400, 1400, "76185", "76281", battement="76471"))
    c = _ctl(r, "debit")
    assert c["statut"] == "etude" and "VER-21" in c["anomalies"]


def test_battement_sans_renfort_non_nomme(snap):
    r = verifier(snap, Saisie("2v_battement", 1400, 1400, "76171", "76281", battement="76473", vent="1,2"))
    assert _ctl(r, "vent")["statut"] == "etude"


def test_battement_courbe_hors_cadre_ne_limite_pas(snap):
    # à 0,8 kN/m², la courbe du V316 n'a pas de tracé dans le cadre de l'abaque
    r = verifier(snap, Saisie("2v_battement", 1400, 1400, "76171", "76281", battement="76471", vent="0,8"))
    assert _ctl(r, "vent")["statut"] == "ok" and "INC-53" in _ctl(r, "vent")["anomalies"]


def test_parcloses(snap):
    assert _ctl(verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76281", vitrage="4-14-4")), "parclose")["statut"] == "hors"  # 22 mm
    r = verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76281", vitrage="44.2-16-4"))  # 28,76 mm
    assert _ctl(r, "parclose")["statut"] == "etude"
    r = verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76281", vitrage="4-20-4-20-4"))  # 52 mm
    assert _ctl(r, "parclose")["statut"] == "hors" and "CTR-17" in _ctl(r, "parclose")["anomalies"]


def test_isolant_et_tapee(snap):
    d = donnees(snap)
    assert d.appuis_par_tapee["76772"] == ["76768"] and d.pivot_kg == 100
    base = dict(configuration="1v_of", largeur_mm=900, hauteur_mm=1400, ouvrant="76281")
    # la même tapée donne 15 mm de plus sur un 76171 que sur un 76180
    ok = _ctl(verifier(snap, Saisie(dormant="76171", isolant_mm=135, **base)), "isolant")
    assert ok["statut"] == "ok" and "tapée 6140" in ok["detail"] and ok["valeurs"]["tapees"][0]["image"]
    assert "tapée 6141" in _ctl(verifier(snap, Saisie(dormant="76180", isolant_mm=140, **base)), "isolant")["detail"]
    # 140 mm n'existe pas sur un 76171 : la page dit « se traite en 135 ou en 155 »
    c = _ctl(verifier(snap, Saisie(dormant="76171", isolant_mm=140, **base)), "isolant")
    assert c["statut"] == "etude" and "135 ou 155" in c["detail"]
    # au-delà de 155 mm sur un 76171, le dormant bas devient un 76180
    assert "76180" in _ctl(verifier(snap, Saisie(dormant="76171", isolant_mm=175, **base)), "isolant")["detail"]
    # pas de colonne pour le 76172
    assert _ctl(verifier(snap, Saisie(dormant="76172", isolant_mm=100, **base)), "isolant")["statut"] == "etude"
    # sans isolant saisi, pas de contrôle
    assert not [c for c in verifier(snap, Saisie(dormant="76171", **base))["controles"] if c["id"] == "isolant"]


def test_roto(snap):
    d = donnees(snap)
    # CTR-18 : entre manuel et catalogue, les bornes les plus basses
    b = d.roto_p[("130", "cdr1n")]
    assert (b["lff_max"], b["hff_max"]) == (1400, 2600) and b["ctr18"]
    assert d.roto_p[("130", "cdr3")]["lff_min"] == 490  # CDR 3 : le catalogue seul
    base = dict(largeur_mm=900, hauteur_mm=1400, dormant="76171", ouvrant="76281")
    ob = _ctl(verifier(snap, Saisie("1v_ob", **base)), "roto")
    assert ob["statut"] == "ok" and ob["valeurs"]["lff"] == 900 - 2 * 38 - 2 * 20 and "hypothèse" in ob["detail"]
    # côté paumelles P, pas de champ pour l'OF ; Designo II en a un
    assert _ctl(verifier(snap, Saisie("1v_of", **base)), "roto")["statut"] == "info"
    assert _ctl(verifier(snap, Saisie("1v_of", paumelles="designo", version="80", **base)), "roto")["statut"] == "ok"
    # le report de charge Designo II n'existe qu'en oscillo-battant
    assert _ctl(verifier(snap, Saisie("1v_of", paumelles="designo", version="report", **base)), "roto")["statut"] == "hors"
    # CDR 2 : HFF mini 510 mm
    bas = _ctl(verifier(snap, Saisie("1v_ob", largeur_mm=900, hauteur_mm=600, dormant="76171", ouvrant="76281",
                                     securite="cdr2")), "roto")
    assert bas["statut"] == "hors" and "HFF" in bas["detail"]


def test_pivot(snap):
    ok = _ctl(verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76281")), "pivot")
    assert ok["statut"] == "ok" and "CTR-01" in ok["anomalies"]
    # 1 400 × 2 150 en 10-16-10-16-10 : vitrage 1 220 × 1 970 mm, 30 mm de verre, ~ 180 kg
    lourd = verifier(snap, Saisie("1v_of", 1400, 2150, "76171", "76281", vitrage="10-16-10-16-10"))
    assert _ctl(lourd, "pivot")["statut"] == "hors"


def test_les_brouillons_sont_signales(snap):
    r = verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76281"))
    statut = snap.pages[fa.PAGE_ABAQUES].status
    assert (fa.PAGE_ABAQUES in r["brouillons"]) == (statut == "draft")


# ---- l'API et la page ----------------------------------------------------------------------

def test_api_requires_auth(client):
    assert client.get("/api/faisabilite/options").status_code == 401
    assert client.post("/api/faisabilite", json={}).status_code == 401


def test_api_admin_only(client, lecteur_headers, responsable_headers):
    for headers in (lecteur_headers, responsable_headers):
        assert client.get("/api/faisabilite/options", headers=headers).status_code == 403
        assert client.post("/api/faisabilite", headers=headers, json={}).status_code == 403


def test_api(client, admin_headers):
    o = client.get("/api/faisabilite/options", headers=admin_headers)
    assert o.status_code == 200 and "76173" not in o.json()["dormants"]
    r = client.post("/api/faisabilite", headers=admin_headers, json={
        "configuration": "1v_of", "largeur_mm": 900, "hauteur_mm": 1400, "dormant": "76171", "ouvrant": "76281"})
    assert r.status_code == 200 and r.json()["verdict"] == "ok"
    bad = client.post("/api/faisabilite", headers=admin_headers, json={
        "configuration": "1v_of", "largeur_mm": 900, "hauteur_mm": 1400, "dormant": "76171", "ouvrant": "76281",
        "vitrage": "4-16"})
    assert bad.status_code == 422


def test_page(client, admin_headers, responsable_headers):
    assert client.get("/faisabilite", follow_redirects=False).headers["location"] == "/login"
    r = client.get("/faisabilite", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    r = client.get("/faisabilite", headers=admin_headers, follow_redirects=False)
    assert r.status_code == 200 and "/api/faisabilite" in r.text and 'id="nav-faisabilite"' in r.text


# ---- compléments : poids selon la ferrure, points de verrouillage ----------------------------

def test_poids_admissible_selon_ferrure_et_acier(snap):
    d = donnees(snap)
    assert d.poids_ferrure[("100", "sans")]["kg"] == 80 and d.poids_ferrure[("100", "long")]["kg"] == 100
    assert d.poids_ferrure[("130", "court")]["kg"] == 110 and d.poids_ferrure[("130", "long")]["kg"] == 130
    base = dict(configuration="1v_of", largeur_mm=900, hauteur_mm=1400, dormant="76171", ouvrant="76281")
    # 4-16-4 : 8 mm de verre, vitrage 744 × 1 244 mm, ~ 18,5 kg
    ok = _ctl(verifier(snap, Saisie(ferrure_poids="100", acier_dormant="sans", **base)), "poids_ferrure")
    assert ok["statut"] == "ok" and ok["valeurs"]["admissible_kg"] == 80 and "verre seul" in ok["detail"]
    assert "VER-25" not in ok["anomalies"]
    # 1 400 × 2 150 en 10-16-10-16-10 : ~ 180 kg de verre, au-delà même de la ferrure de 130 kg sur acier long
    lourd = dict(configuration="1v_of", largeur_mm=1400, hauteur_mm=2150, dormant="76171", ouvrant="76281",
                 vitrage="10-16-10-16-10")
    c = _ctl(verifier(snap, Saisie(ferrure_poids="130", acier_dormant="long", **lourd)), "poids_ferrure")
    assert c["statut"] == "hors" and "VER-25" in c["anomalies"]
    # une ligne qui n'existe pas : sur étude
    assert _ctl(verifier(snap, Saisie(ferrure_poids="150", **base)), "poids_ferrure")["statut"] == "etude"
    # sans ferrure saisie, pas de contrôle ; châssis fixe : pas de vantail
    assert not [c for c in verifier(snap, Saisie(**base))["controles"] if c["id"] == "poids_ferrure"]
    assert not [c for c in verifier(snap, Saisie("fixe", 900, 1400, "76171", ferrure_poids="100"))["controles"]
                if c["id"] == "poids_ferrure"]
    # la limite change avec l'acier : 18,5 kg passe partout, un vitrage de ~ 94 kg dépasse 90 kg (acier court) mais pas 100 kg
    mid = dict(configuration="1v_of", largeur_mm=1150, hauteur_mm=1800, dormant="76171", ouvrant="76281",
               vitrage="8-16-8-16-8")  # 24 mm de verre, vitrage 970 × 1 620 mm : ~ 94 kg
    kg = _ctl(verifier(snap, Saisie(**mid)), "poids")["valeurs"]["poids_kg"]
    assert 90 < kg < 100
    assert _ctl(verifier(snap, Saisie(ferrure_poids="100", acier_dormant="long", **mid)), "poids_ferrure")["statut"] == "ok"
    assert _ctl(verifier(snap, Saisie(ferrure_poids="100", acier_dormant="court", **mid)), "poids_ferrure")["statut"] == "hors"
    assert _ctl(verifier(snap, Saisie(ferrure_poids="130", acier_dormant="sans", **mid)), "poids_ferrure")["statut"] == "ok"


def test_points_de_verrouillage_inc10(snap):
    d = donnees(snap)
    assert d.verrouillage_cm == {"paumelle": 70, "galets": 80}
    base = dict(configuration="1v_of", largeur_mm=900, hauteur_mm=1400, dormant="76171", ouvrant="76281")
    ok = _ctl(verifier(snap, Saisie(distance_verrouillage_cm=65, **base)), "verrouillage")
    assert ok["statut"] == "ok" and ok["anomalies"] == ["INC-10"]
    entre = _ctl(verifier(snap, Saisie(distance_verrouillage_cm=75, **base)), "verrouillage")
    assert entre["statut"] == "etude" and "70" in entre["detail"] and "80" in entre["detail"]
    assert _ctl(verifier(snap, Saisie(distance_verrouillage_cm=90, **base)), "verrouillage")["statut"] == "hors"
    assert _ctl(verifier(snap, Saisie(distance_verrouillage_cm=70, **base)), "verrouillage")["statut"] == "ok"
    assert not [c for c in verifier(snap, Saisie(**base))["controles"] if c["id"] == "verrouillage"]


# ---- Roto NX : les courbes de hauteur maxi par poids de vitrage ------------------------------

def test_les_courbes_roto_se_lisent(snap):
    c = donnees(snap).roto_courbes
    assert set(c) == {"P130", "P150", "D80"}
    p130, p150, d80 = c["P130"], c["P150"], c["D80"]
    assert sorted(p130.maxi) == [40, 50, 60] and sorted(p150.maxi) == [40, 50, 60, 80]
    assert p130.maxi[60][900] == 2450 and p130.maxi[60][1400] is None   # « hors champ »
    assert p130.maxi[50][1000] == 2630 and p130.bas[800] == 530 and p130.bas[300] == 0   # « bord bas »
    assert p130.deuxieme_compas == [1400, 1500, 1600]
    assert p150.maxi[80][1100] == 1760 and p150.maxi[80][1200] is None
    assert d80.maxi[20][1400] == 2400 and d80.maxi[50][1400] is None and d80.deuxieme_compas == [1200, 1300, 1400]


def test_champ_courbes_sans_interpolation_et_avec_marge(snap):
    p130 = donnees(snap).roto_courbes["P130"]
    # 24 mm de verre = 60 kg/m² ; à LFF 1000 la courbe est à 2 210 mm
    assert fa.champ_courbes(p130, 1000, 2000, 60)[0] == "ok"
    assert fa.champ_courbes(p130, 1000, 2200, 60)[0] == "etude"      # à moins de 25 mm de la courbe
    assert fa.champ_courbes(p130, 1000, 2300, 60)[0] == "hors"
    # LFF 950 : entre 900 (2 450) et 1 000 (2 210), la plus restrictive
    statut, phrase, v = fa.champ_courbes(p130, 950, 2300, 60)
    assert statut == "hors" and v["hff_maxi"] == 2210 and "plus restrictive" in phrase
    # au-delà de la verticale de la courbe : « hors champ »
    assert fa.champ_courbes(p130, 1400, 1500, 60)[0] == "hors"
    # sous la droite basse (LFF 800 : 530 mm)
    assert fa.champ_courbes(p130, 800, 400, 60)[0] == "hors"
    # entre deux courbes, la plus lourde : 45 kg/m² se lit sur la courbe des 50
    assert fa.champ_courbes(p130, 1000, 2000, 45)[2]["courbe_kg_m2"] == 50
    # plus léger que la plus légère (40) : dedans ok, dehors on ne sait pas
    assert fa.champ_courbes(p130, 1000, 2000, 20)[0] == "ok"
    leger = fa.champ_courbes(p130, 1500, 2300, 20)
    assert leger[0] == "etude" and leger[2]["mode"] == "leger" and leger[2]["deuxieme_compas"]
    # plus lourd que la plus lourde (60) : dehors hors, dedans on ne sait pas
    assert fa.champ_courbes(p130, 1400, 1500, 70)[0] == "hors"
    assert fa.champ_courbes(p130, 800, 2000, 70)[0] == "etude"


def test_roto_ob_courbes_dans_le_verdict(snap):
    # LFF 1 500, HFF 2 300 (DFO = dimension − 116) en 4-16-4 (20 kg/m²) : la courbe des 40 kg/m² est à 2 180 mm
    r = verifier(snap, Saisie("1v_ob", 1616, 2416, "76171", "76281"))
    roto = _ctl(r, "roto")
    assert roto["statut"] == "etude" and "diagramme d'application" in roto["detail"]
    assert roto["valeurs"]["courbe_kg_m2"] == 40 and roto["valeurs"]["mode"] == "leger"
    assert any("Diagrammes d'application" in x["section"] for x in roto["sources"])
    # la fenêtre courante reste réalisable
    assert _ctl(verifier(snap, Saisie("1v_ob", 900, 1400, "76171", "76281")), "roto")["statut"] == "ok"
    # version 150 kg : même lecture sur son diagramme
    r150 = _ctl(verifier(snap, Saisie("1v_ob", 1616, 2416, "76171", "76281", version="150")), "roto")
    assert r150["statut"] == "etude" and "INC-13" in r150["anomalies"]


def test_roto_designo_80_courbes_et_diagrammes_non_releves(snap):
    # Designo II 80 kg : LFF 1 400, HFF 2 500 > 2 400 sur la courbe des 20 kg/m², sous le bord haut 2 600 de la boîte
    base = dict(configuration="1v_of", dormant="76171", ouvrant="76281", paumelles="designo")
    hors = _ctl(verifier(snap, Saisie(largeur_mm=1516, hauteur_mm=2616, version="80", **base)), "roto")
    assert hors["statut"] == "hors" and "diagramme d'application" in hors["detail"]
    # Designo 100 kg : courbes relevées par points, non lues : la boîte seule, et le détail le dit
    cent = _ctl(verifier(snap, Saisie(largeur_mm=900, hauteur_mm=1400, version="100", **base)), "roto")
    assert cent["statut"] == "ok" and "ne sont pas relevées" in cent["detail"]


# ---- AluClip Pro : ouvrant 76271 capoté A072, sans renfort acier -----------------------------

def _alu(**kw):
    base = dict(configuration="1v_of", largeur_mm=900, hauteur_mm=1400, dormant="76171", ouvrant="76271",
                couleur="capot_alu", aluclip_pro=True)
    base.update(kw)
    return Saisie(**base)


def test_aluclip_pro_se_lit(snap):
    ap = donnees(snap).aluclip_pro
    assert ap["limite"] == fa.LimiteCouleur(250, 100, 150, 150, 150, 120)
    assert ap["limite_2v"] == fa.LimiteCouleur(250, 100, 135, 180, 135, 108)
    assert (ap["iw"], ap["ig"], ap["j079_kg"]) == (0.8, 1.5, 40)
    assert ap["verre"][120][12] == 214 and ap["verre"][80][28] == 229 and ap["verre"][50][12] is None
    # une courbe par battement, lue sur la coupe dessinée (INC-54 : le titre du premier abaque se trompe)
    b = ap["battements"]
    assert set(b) == {"76471", "76472"} and (b["76472"]["renfort"], b["76471"]["renfort"]) == ("V317", "V316")
    assert b["76472"]["courbes"]["1,2"][50] == 250 and b["76471"]["courbes"]["1,2"][50] == 244
    assert b["76472"]["courbes"]["0,8"][80] == 249 and b["76471"]["courbes"]["0,8"][80] == 243


def test_aluclip_pro_un_vantail(snap):
    r = verifier(snap, _alu())
    assert r["verdict"] == "ok", [(c["titre"], c["statut"], c["detail"]) for c in r["controles"]]
    assert _ctl(r, "ouvrant")["statut"] == "ok" and "capot A072" in _ctl(r, "ouvrant")["detail"]
    assert _ctl(r, "renfort")["valeurs"] == {"zone": None, "renfort": None} and "Pas de renfort acier" in _ctl(r, "renfort")["detail"]
    assert _ctl(r, "j079")["statut"] == "info"      # ~ 17 kg de verre : sous 40 kg
    # la limite oblique : 122,4 cm de large, 205 cm de haut admis ; 232 cm demandés
    assert _ctl(verifier(snap, _alu(largeur_mm=1300, hauteur_mm=2400)), "ouvrant")["statut"] == "hors"
    # le même vantail en blanc standard (76281) n'a pas la même limite : l'abaque AluClip Pro est le sien
    assert verifier(snap, _alu(vitrage="4-12-4-12-4", largeur_mm=1300, hauteur_mm=1700))["graphe"]["limite"]["coin_hauteur"] == 150


def test_aluclip_pro_equerre_j079_des_40_kg(snap):
    # 12 mm de verre sur 1 104 × 1 504 mm : ~ 50 kg de verre
    lourd = dict(largeur_mm=1300, hauteur_mm=1700, vitrage="4-12-4-12-4")
    sans = _ctl(verifier(snap, _alu(**lourd)), "j079")
    assert sans["statut"] == "hors" and "impérative" in sans["detail"]
    avec = verifier(snap, _alu(j079=True, **lourd))
    assert _ctl(avec, "j079")["statut"] == "ok"
    # plus de 12 mm de verre : le renforcement total exigé n'existe pas sur ce profilé, sur étude
    r = verifier(snap, _alu(vitrage="6-12-6-12-6"))
    assert _ctl(r, "renfort")["statut"] == "etude" and r["verdict"] == "etude"


def test_aluclip_pro_deux_vantaux_a_battement(snap):
    base = dict(configuration="2v_battement", largeur_mm=1400, hauteur_mm=1400, battement="76471", vent="1,2")
    r = verifier(snap, _alu(**base))
    vent = _ctl(r, "vent")
    assert vent["statut"] == "ok" and "INC-54" in vent["anomalies"] and "V316" in vent["detail"]
    assert "INC-54" in _ctl(r, "ouvrant")["anomalies"] and r["graphe"]["limite"]["coin_largeur"] == 135
    # 100 cm de large : 200 cm sur la courbe 1,2 kN/m² du 76471 ; 232 cm demandés
    haut = verifier(snap, _alu(**dict(base, largeur_mm=2064, hauteur_mm=2400)))
    assert _ctl(haut, "vent")["statut"] == "hors"
    # pas d'abaque AluClip Pro pour le 76473
    assert _ctl(verifier(snap, _alu(**dict(base, battement="76473"))), "vent")["statut"] == "etude"


def test_aluclip_pro_exige_le_76271_en_capot(snap):
    with pytest.raises(ValueError):
        verifier(snap, _alu(ouvrant="76281"))
    with pytest.raises(ValueError):
        verifier(snap, _alu(couleur="blanc"))
    with pytest.raises(ValueError):
        verifier(snap, _alu(configuration="fixe"))
    assert fa.options(snap)["aluclip_pro"] == {"ouvrant": "76271", "couleur": "capot_alu", "battements": ["76471", "76472"]}


# ---- soufflet, fixe latéral, meneau ----------------------------------------------------------

def test_soufflet_se_lit(snap):
    d = donnees(snap)
    assert ("Soufflet (loqueteaux)", 0.8, 1.3) in d.dta and ("2 vantaux OF + fixe latéral", 2.15, 2.4) in d.dta
    assert d.roto_soufflet["oblique"] == (1200, 1200, 800, 2400) and (310, 449) in d.roto_soufflet["cremone"]
    b = d.roto_p[("soufflet", "base")]
    assert (b["lff_min"], b["lff_max"], b["hff_min"], b["hff_max"], b["pv_max"]) == (310, 2400, 290, 1200, 80)
    assert "soufflet" not in fa.options(snap)["versions"]["P"]


def test_soufflet_realisable(snap):
    # 1 200 × 700 mm (L × H) : vantail 1 124 × 624 mm, lu sur l'abaque en inversant hauteur et largeur
    r = verifier(snap, Saisie("soufflet", 1200, 700, "76171", "76281"))
    assert r["verdict"] == "ok", [(c["titre"], c["statut"], c["detail"]) for c in r["controles"]]
    assert r["graphe"]["point"] == [pytest.approx(62.4), pytest.approx(112.4)]
    assert "inversant" in _ctl(r, "ouvrant")["detail"]
    dta = _ctl(r, "dta")
    assert dta["valeurs"] == {"h_maxi": 0.8, "l_maxi": 1.3} and "loqueteaux" in dta["detail"]
    roto = _ctl(r, "roto")
    assert roto["statut"] == "ok" and "soufflet" in roto["detail"]
    ids = {c["id"] for c in r["controles"]}
    assert "pivot" not in ids and "paumelles" not in ids
    # le DTA : 1,50 m de large dépasse 1,30 m
    assert _ctl(verifier(snap, Saisie("soufflet", 1500, 700, "76171", "76281")), "dta")["statut"] == "hors"


def test_soufflet_roto_limite_oblique_et_cremone(snap):
    # LFF 2 000 : la limite oblique est à ~ 933 mm de HFF ; 1 100 demandés
    r = verifier(snap, Saisie("soufflet", 2116, 1216, "76171", "76281"))
    roto = _ctl(r, "roto")
    assert roto["statut"] == "hors" and "Limite oblique" in roto["detail"]
    # LFF 400 : entre 310 et 449 mm, la crémone verrou est exigée (INC-193)
    etroit = _ctl(verifier(snap, Saisie("soufflet", 516, 716, "76171", "76281")), "roto")
    assert "uniquement avec crémone verrou" in etroit["detail"] and "INC-193" in etroit["anomalies"]
    # côté Designo II, le champ du soufflet n'est pas relevé en tableau : sur étude
    assert _ctl(verifier(snap, Saisie("soufflet", 1200, 700, "76171", "76281", paumelles="designo", version="80")),
                "roto")["statut"] == "etude"


def test_deux_vantaux_et_fixe_lateral(snap):
    r = verifier(snap, Saisie("2v_fixe_lateral", 2300, 2100, "76171", "76281"))
    assert _ctl(r, "dta")["statut"] == "ok" and _ctl(r, "dta")["valeurs"] == {"h_maxi": 2.15, "l_maxi": 2.4}
    deb = _ctl(r, "debit")
    assert deb["statut"] == "etude" and "fixe latéral" in deb["detail"] and deb["sources"][0]["page"] == fa.PAGE_FIXE_LATERAL
    assert r["verdict"] == "etude" and r["debit"]["vantaux"] == 0
    assert verifier(snap, Saisie("2v_fixe_lateral", 2500, 2100, "76171", "76281"))["verdict"] == "hors"


def test_meneau_76373_entre_deux_ouvrants_et_vent(snap):
    d = donnees(snap)
    assert d.part_ouvrant == {"76281": 34, "76275": 34, "76272": 74, "76279": 74}
    r = verifier(snap, Saisie("2v_meneau", 2000, 1200, "76171", "76281", meneau="76373"))
    m = _ctl(r, "meneau")
    assert m["statut"] == "info" and m["valeurs"]["composition"] == "34 + 124 + 34" and m["valeurs"]["largeur"] == 192
    assert "49 / 52 / 49, soit 150" in m["detail"]
    haut = _ctl(verifier(snap, Saisie("2v_meneau", 2000, 1200, "76171", "76272", meneau="76373")), "meneau")
    assert haut["valeurs"]["composition"] == "74 + 124 + 74" and haut["valeurs"]["largeur"] == 272
    # le 76372 n'est dessiné qu'avec un seul ouvrant : on le dit
    seul = _ctl(verifier(snap, Saisie("2v_meneau", 2000, 1200, "76171", "76281", meneau="76372")), "meneau")
    assert "ne dessine pas" in seul["detail"] and "34 + 98" in seul["detail"]
    # le vent sur le meneau n'est pas donné par le wiki : une information, pas un verdict
    v = _ctl(r, "vent_meneau")
    assert v["statut"] == "info" and "aucune courbe de vent" in v["detail"]
    assert not [c for c in verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76281"))["controles"] if c["id"] == "vent_meneau"]
