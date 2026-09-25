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


def test_ouvrant_sans_abaque_simple_est_sur_etude(snap):
    r = verifier(snap, Saisie("1v_of", 900, 1400, "76171", "76272"))
    assert _ctl(r, "ouvrant")["statut"] == "etude"
    assert r["verdict"] == "etude"


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


def test_api(client, lecteur_headers):
    o = client.get("/api/faisabilite/options", headers=lecteur_headers)
    assert o.status_code == 200 and "76173" not in o.json()["dormants"]
    r = client.post("/api/faisabilite", headers=lecteur_headers, json={
        "configuration": "1v_of", "largeur_mm": 900, "hauteur_mm": 1400, "dormant": "76171", "ouvrant": "76281"})
    assert r.status_code == 200 and r.json()["verdict"] == "ok"
    bad = client.post("/api/faisabilite", headers=lecteur_headers, json={
        "configuration": "1v_of", "largeur_mm": 900, "hauteur_mm": 1400, "dormant": "76171", "ouvrant": "76281",
        "vitrage": "4-16"})
    assert bad.status_code == 422


def test_page(client, responsable_headers):
    assert client.get("/faisabilite", follow_redirects=False).status_code == 303
    r = client.get("/faisabilite", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 200 and "/api/faisabilite" in r.text and 'id="nav-faisabilite"' in r.text
