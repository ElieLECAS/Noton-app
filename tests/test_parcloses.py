"""Le calculateur de parcloses et de joints : chaque gamme lue dans sa propre forme de tableau."""
from __future__ import annotations

import pytest

from app.services.parcloses import chercher, solutions
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(wiki_root())


def _ctx(r, systeme, contexte):
    g = next((g for g in r["groupes"] if g["systeme"] == systeme), None)
    assert g, (systeme, [g["systeme"] for g in r["groupes"]])
    c = next((c for c in g["contextes"] if c["contexte"] == contexte), None)
    assert c, (contexte, [c["contexte"] for c in g["contextes"]])
    return c["solutions"]


def _ok(sols):
    return {s["ref"] for s in sols if s["statut"] == "ok"}


def test_toutes_les_tables_se_lisent(snap):
    sol = solutions(snap)
    assert not [s for s in sol if "erreur" in s]
    systemes = {s["systeme"] for s in sol}
    assert {"Système 76 Advanced", "PERFORM76 — cahier technique PROFERM", "SOLEAL FY 55", "ASKEY Frappe 65 Ouvrant Caché",
            "ASKEY Coulissant 65 NV", "LUMEAL GA", "SOLEAL GY 55"} <= systemes


def test_76_tolerance_et_deux_familles_de_joint(snap):
    # 44.2-16-4 = 28,76 mm : A = 28 (+1 / −0,5) avec un joint de 4 mm, ou B = 28 (la 76527, A = 26) avec 2 mm
    r = chercher(snap, vitrage="44.2-16-4")
    ouv = _ctx(r, "Système 76 Advanced", "Ouvrant")
    assert _ok(ouv) == {"76512", "76513", "76526", "76527"}
    el = {s["ref"]: {e["role"]: e for e in s["elements"]} for s in ouv}
    assert el["76526"]["Joint de vitrage"]["valeur"] == "PCE, G049.T ou G047"
    assert "4 mm" in el["76526"]["Joint de vitrage"]["detail"] and el["76526"]["Support de cale"]["valeur"] == "M137"
    assert el["76527"]["Joint de vitrage"]["valeur"] == "G048" and "2 mm" in el["76527"]["Joint de vitrage"]["detail"]
    capot = _ctx(r, "Système 76 Advanced", "Ouvrant, capot alu et joint EPDM")
    assert {e["role"]: e["valeur"] for e in next(s for s in capot if s["ref"] == "76526")["elements"]}["Joint de vitrage"] == "G178"
    # hors tolérance : rien de « proche » là où la source écrit une tolérance
    assert all(s["statut"] == "ok" for s in ouv)


def test_76_limites_de_tolerance(snap):
    # 76501 : A = 24, tolérance +1,0 / −0,5 → 23,5 à 25
    assert "76501" in _ok(_ctx(chercher(snap, epaisseur=23.5), "Système 76 Advanced", "Ouvrant"))
    assert "76501" in _ok(_ctx(chercher(snap, epaisseur=25), "Système 76 Advanced", "Ouvrant"))
    # 25,2 mm : entre deux tolérances, aucune parclose d'ouvrant du 76 ne le tient
    r = chercher(snap, epaisseur=25.2)
    g = next((g for g in r["groupes"] if g["systeme"] == "Système 76 Advanced"), {"contextes": []})
    assert not [c for c in g["contextes"] if c["contexte"] == "Ouvrant"]


def test_76_dormant_sans_compensateur_depuis_le_poster(snap):
    sols = _ctx(chercher(snap, epaisseur=24), "Système 76 Advanced", "Dormant et meneau sans compensateur")
    assert _ok(sols) == {"2630"} and "INC-52" in sols[0]["anomalies"]


def test_perform76_cahier(snap):
    sols = _ctx(chercher(snap, vitrage="4-16-4"), "PERFORM76 — cahier technique PROFERM", "Ouvrant")
    assert _ok(sols) == {"76501"} and sols[0]["image"]


def test_soleal_fy_matrice(snap):
    # l'exemple de la page : 24 mm → T591005 (droite) ou TFY2412 (arrondie), joint vert TAS0017
    sols = _ctx(chercher(snap, epaisseur=24), "SOLEAL FY 55", "Ouvrant apparent et fixe")
    exacts = [s for s in sols if s["statut"] == "ok"]
    assert {s["ref"] for s in exacts} == {"T591005", "TFY2412"}
    for x in exacts:
        el = {e["role"]: e for e in x["elements"]}
        assert el["Parclose"]["valeur"] == x["ref"] and "hauteur C = 15 mm" in el["Parclose"]["detail"]
        assert el["Joint intérieur"]["valeur"] == "TAS0017 · vert" and el["Joint intérieur"]["detail"] == "B = 7 mm"
        assert el["Joint extérieur"]["valeur"] == "T410010" and x["recommande"]
    # l'élargisseur ne nomme que la couleur du joint : la référence vient de la matrice
    elarg = _ctx(chercher(snap, epaisseur=52), "SOLEAL FY 55", "Fixe avec élargisseur de feuillure T530031")
    ok52 = next(s for s in elarg if s["statut"] == "ok")
    assert {e["role"]: e["valeur"] for e in ok52["elements"]}["Joint intérieur"] == "TAS0017 · vert"
    # l'élargisseur T530031 ouvre les triples lourds jusqu'à 70 mm, sur châssis fixe
    assert _ok(_ctx(chercher(snap, epaisseur=52), "SOLEAL FY 55", "Fixe avec élargisseur de feuillure T530031")) == {"T591005"}


def test_profils_par_epaisseur(snap):
    r = chercher(snap, epaisseur=28)
    oc = _ctx(r, "ASKEY Frappe 65 Ouvrant Caché", "Profilés ouvrants cachés")
    assert "W1010142" in _ok(oc)
    w = {e["role"]: e["valeur"] for e in next(s for s in oc if s["ref"] == "W1010142")["elements"]}
    assert w["Profilé d'ouvrant"] == "W1010142" and w["Parclose TPE"].startswith("W4033002")
    assert "T141015" in _ok(_ctx(r, "LUMEAL GA", "Profilés ouvrants"))           # 24 à 28 mm
    assert "T141021" not in _ok(_ctx(r, "LUMEAL GA", "Profilés ouvrants"))       # 29 à 32 mm


def test_filtre_par_gamme_et_non_calculables(snap):
    r = chercher(snap, epaisseur=28, gamme="PERFORM76")
    assert {g["systeme"] for g in r["groupes"]} <= {"Système 76 Advanced", "PERFORM76 — cahier technique PROFERM"}
    assert r["groupes"] and not r["non_calculables"]
    lumine = chercher(snap, epaisseur=24, gamme="SOLEAL FY")
    assert {g["systeme"] for g in lumine["groupes"]} == {"SOLEAL FY 55"} and lumine["sans_solution"] == ["SOLEAL FY 65"]
    tout = chercher(snap, epaisseur=28)
    assert {x["systeme"] for x in tout["non_calculables"]} >= {"ASKEY Frappe 65 Ouvrant Visible"}
    assert "LUMEAL GA" in tout["gammes"]
    with pytest.raises(ValueError):
        chercher(snap)
    with pytest.raises(ValueError):
        chercher(snap, epaisseur=28, gamme="inconnue")


def test_api_et_page(client, lecteur_headers, responsable_headers):
    assert client.get("/api/parcloses?epaisseur=28").status_code == 401
    r = client.get("/api/parcloses?vitrage=4-16-4&gamme=LUMEAL GA", headers=lecteur_headers)
    assert r.status_code == 200 and r.json()["epaisseur"] == 24
    assert [g["systeme"] for g in r.json()["groupes"]] == ["LUMEAL GA"]
    assert client.get("/api/parcloses", headers=lecteur_headers).status_code == 422
    assert client.get("/parcloses", follow_redirects=False).status_code == 303
    p = client.get("/parcloses", headers=responsable_headers, follow_redirects=False)
    assert p.status_code == 200 and "/api/parcloses" in p.text and 'id="nav-parcloses"' in p.text
