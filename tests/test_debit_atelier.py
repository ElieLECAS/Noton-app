"""La fiche de débit PERFORM76 (audit § 9.2) : liste de coupe, nomenclature, accessoires.

Les longueurs attendues sont recalculées à la main depuis `systeme-76-cotes-de-debit.md` : une
cote à déduire par coupe, deux coupes par pièce."""
from __future__ import annotations

import pytest

from app.services.debit_atelier import fiche
from app.services.faisabilite import Saisie
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _lignes(r, piece):
    return [l for l in r["lignes"] if l["piece"] == piece]


def _une(r, piece):
    ls = _lignes(r, piece)
    assert len(ls) == 1, (piece, [l["piece"] for l in r["lignes"]])
    return ls[0]


def test_exemple_du_manuel(snap):
    """DHT 2 000 × 1 200, dormant 76171, meneau 76372, ouvrant 76271 : DEO 949, vitrage 829."""
    r = fiche(snap, Saisie("2v_meneau", 2000, 1200, "76171", "76271", meneau="76372"))
    assert _une(r, "Traverses")["longueur"] == 949 and _une(r, "Traverses")["qte"] == 4
    assert _une(r, "Vitrage d'ouvrant")["calcul"].startswith("829 × ")
    assert _une(r, "Meneau de dormant")["longueur"] == 1200 - 2 * 40
    assert _une(r, "Renfort de meneau")["longueur"] == 1200 - 2 * 71
    assert _une(r, "Renfort de dormant, traverses")["longueur"] == 2000 - 2 * 45


def test_renfort_d_ouvrant_selon_la_zone(snap):
    # vantail 94,9 × 112,4 cm, blanc : plus large que 75, moins haut que 130 → zone B, traverses seules
    r = fiche(snap, Saisie("2v_meneau", 2000, 1200, "76171", "76281", meneau="76372"))
    assert _une(r, "Renfort d'ouvrant, traverses")["ref"] == "V266.Z"
    assert _une(r, "Renfort d'ouvrant, traverses")["longueur"] == 949 - 2 * 47
    assert not _lignes(r, "Renfort d'ouvrant, montants")
    # en couleur, le renforcement est systématique : traverses et montants
    c = fiche(snap, Saisie("2v_meneau", 2000, 1200, "76171", "76281", meneau="76372", couleur="standard"))
    assert _lignes(c, "Renfort d'ouvrant, traverses") and _lignes(c, "Renfort d'ouvrant, montants")


def test_parclose_et_vitrage(snap):
    r = fiche(snap, Saisie("1v_of", 900, 1400, "76171", "76281"))
    p = _une(r, "Parclose d'ouvrant, horizontale")
    assert p["ref"] == "76501" and p["longueur"] == (900 - 76) - 2 * 49 and p["qte"] == 2
    fixe = fiche(snap, Saisie("fixe", 900, 1400, "76171", vitrage="4-20-4-20-4"))  # 52 mm : pas de parclose
    assert _une(fixe, "Parclose de dormant, verticale")["ref"] == "à choisir"
    ok = fiche(snap, Saisie("fixe", 900, 1400, "76171", vitrage="4-20-4"))       # 28 mm : dormant 2634
    assert _une(ok, "Parclose de dormant, verticale")["ref"] == "2634"
    assert _une(ok, "Parclose de dormant, verticale")["longueur"] == 1400 - 2 * 46


def test_renfort_76173_dissymetrique(snap):
    r = fiche(snap, Saisie("fixe", 1000, 1500, "76173"))
    assert _une(r, "Renfort de dormant, montants")["longueur"] == 1500 - 75 - 45


def test_seuil(snap):
    r = fiche(snap, Saisie("1v_of", 900, 2150, "76171", "76281", bas="seuil"))
    assert _une(r, "Montant sur seuil")["longueur"] == 2150 - 20
    assert _une(r, "Seuil aluminium")["ref"] == "A076"
    assert _une(r, "Renfort de dormant, montants sur seuil")["longueur"] == 2150 - 45 - 100
    assert _une(r, "Traverse haute")["qte"] == 1


def test_battement(snap):
    r = fiche(snap, Saisie("2v_battement", 1400, 1400, "76171", "76281", battement="76471"))
    deo_h = 1400 - 2 * 38
    assert _une(r, "Battement")["longueur"] == deo_h - 2 * 47
    assert _une(r, "Renfort de battement")["ref"] == "V316"
    assert _une(r, "Renfort de battement")["longueur"] == deo_h - 2 * 59
    assert not _lignes(fiche(snap, Saisie("2v_battement", 1400, 1400, "76171", "76281", battement="76473")),
                       "Renfort de battement")


def test_surcote_de_soudure(snap):
    sans = fiche(snap, Saisie("1v_of", 900, 1400, "76171", "76281"))
    avec = fiche(snap, Saisie("1v_of", 900, 1400, "76171", "76281"), surcote_soudure_mm=3)
    assert _une(avec, "Montants")["longueur"] == _une(sans, "Montants")["longueur"] + 6
    # une pièce non soudée ne bouge pas
    assert _une(avec, "Parclose d'ouvrant, verticale")["longueur"] == _une(sans, "Parclose d'ouvrant, verticale")["longueur"]
    assert any("cote finie" in n for n in sans["notes"]) and not any("cote finie" in n for n in avec["notes"])


def test_tapee_et_appui(snap):
    r = fiche(snap, Saisie("1v_of", 900, 1400, "76171", "76281", isolant_mm=175))
    assert _une(r, "Tapée de doublage")["ref"] == "6142"
    assert _une(r, "Pièce d'appui")["ref"] == "6137"


def test_nomenclature_et_accessoires(snap):
    r = fiche(snap, Saisie("1v_of", 900, 1400, "76171", "76281"))
    n = {x["ref"]: x for x in r["nomenclature"]}
    # quatre barres d'ouvrant de 824 et 1 324 mm : 4,30 ml
    assert n["76281"]["qte"] == 4 and n["76281"]["ml"] == round(2 * 0.824 + 2 * 1.324, 2)
    # les pièces sans référence gardent leur nom
    assert any(x["designation"] == "Paumelles" for x in r["nomenclature"])
    refs = {a["ref"] for a in r["accessoires"]}
    assert "M137" in refs                      # support de cale de l'ouvrant 76281
    assert "A042" not in refs                  # capot : seulement en capot aluminium
    assert "A042" in {a["ref"] for a in fiche(snap, Saisie("1v_of", 900, 1400, "76171", "76281",
                                                            couleur="capot_alu"))["accessoires"]}
    assert all(l["source"]["page"].startswith("/") for l in r["lignes"])


def test_api_debit(client, lecteur_headers):
    assert client.post("/api/faisabilite/debit", json={}).status_code == 401
    r = client.post("/api/faisabilite/debit", headers=lecteur_headers, json={
        "configuration": "1v_of", "largeur_mm": 900, "hauteur_mm": 1400, "dormant": "76171", "ouvrant": "76281",
        "surcote_soudure_mm": 3})
    assert r.status_code == 200 and r.json()["surcote"] == 3 and r.json()["lignes"]
