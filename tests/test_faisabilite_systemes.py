"""Le registre des calculateurs : une porte, un format de résultat, un formulaire dérivé des champs."""
from __future__ import annotations

import sys
import types

import pytest

from app.services import faisabilite_systemes as reg
from app.services.faisabilite_systemes import ETUDE, HORS, INFO, OK, coercer, synthese


def _controle(statut: str, id_: str = "c") -> dict:
    return {"id": id_, "titre": id_, "statut": statut, "detail": "", "sources": [], "anomalies": [], "valeurs": {}}


CHAMPS = [
    {"id": "configuration", "libelle": "Configuration", "type": "select", "options": {"fixe": "Fixe", "1v": "1 vantail"}, "defaut": "1v"},
    {"id": "largeur_mm", "libelle": "Largeur", "type": "nombre", "min": 300, "max": 6000, "defaut": 1200},
    {"id": "hauteur_mm", "libelle": "Hauteur", "type": "nombre", "min": 300, "max": 4000, "defaut": 1400},
    {"id": "ouvrant", "libelle": "Ouvrant", "type": "select", "options": {"a": "A", "b": "B"}, "defaut": "a", "si": {"configuration": ["1v"]}},
    {"id": "j079", "libelle": "Équerres", "type": "case", "defaut": False},
]


def test_synthese_le_pire_statut_decide():
    assert synthese([_controle(OK), _controle(INFO)])["verdict"] == OK
    r = synthese([_controle(OK), _controle(ETUDE, "x"), _controle(INFO)], ["/b.md", "/a.md"], debit={"deo": 1})
    assert r["verdict"] == ETUDE and r["libelle"] == "Sur étude" and r["contrainte"]["id"] == "x"
    assert r["brouillons"] == ["/a.md", "/b.md"] and r["debit"] == {"deo": 1}
    # Hors domaine l'emporte, et c'est lui qu'on montre en tête.
    r = synthese([_controle(ETUDE, "e"), _controle(HORS, "h")])
    assert r["verdict"] == HORS and r["contrainte"]["id"] == "h"
    assert synthese([])["contrainte"] is None


def test_coercer_types_defauts_et_bornes():
    s = coercer(CHAMPS, {"largeur_mm": "1 500,5".replace(" ", ""), "j079": 1})
    assert s == {"configuration": "1v", "largeur_mm": 1500.5, "hauteur_mm": 1400.0, "ouvrant": "a", "j079": True}
    with pytest.raises(ValueError, match="Largeur : au moins 300"):
        coercer(CHAMPS, {"largeur_mm": 100})
    with pytest.raises(ValueError, match="Hauteur : nombre attendu"):
        coercer(CHAMPS, {"hauteur_mm": "grand"})
    with pytest.raises(ValueError, match="Configuration"):
        coercer(CHAMPS, {"configuration": "coulissant"})


def test_coercer_ignore_un_champ_dont_la_condition_n_est_pas_remplie():
    s = coercer(CHAMPS, {"configuration": "fixe", "ouvrant": "zzz"})
    assert s["ouvrant"] is None  # non montré, donc non vérifié


def test_le_registre_propose_perform76_puis_les_modules_ecrits(monkeypatch):
    fake = types.ModuleType("faux_calculateur")
    fake.ID, fake.LIBELLE, fake.GAMMES, fake.PORTEE = "faux", "Faux système", ["Gamme X"], "Rien de réel."
    monkeypatch.setitem(sys.modules, "faux_calculateur", fake)
    monkeypatch.setattr(reg, "MODULES", {"faux": "faux_calculateur", "absent": "app.services.faisabilite_pas_encore"})
    r = reg.systemes()
    ids = [s["id"] for s in r["systemes"]]
    assert ids == ["perform76", "faux"]  # un module pas encore écrit n'est pas proposé
    assert r["systemes"][0]["legacy"] is True and r["systemes"][1]["legacy"] is False
    assert r["systemes"][1]["portee"] == "Rien de réel."
    assert [x["id"] for x in r["indisponibles"]] == ["askey-ov"] and "VER-970" in r["indisponibles"][0]["raison"]


def test_le_registre_delegue_au_module(monkeypatch):
    fake = types.ModuleType("faux_calculateur")
    fake.options = lambda snap: {"champs": CHAMPS}
    fake.verifier = lambda snap, saisie: synthese([_controle(OK)], [], vu=saisie)
    fake.fiche_debit = lambda snap, saisie: {"indisponible": "pas de débit"}
    monkeypatch.setitem(sys.modules, "faux_calculateur", fake)
    monkeypatch.setattr(reg, "MODULES", {"faux": "faux_calculateur"})
    assert reg.options(None, "faux")["champs"] == CHAMPS
    assert reg.verifier(None, "faux", {"largeur_mm": 1})["vu"] == {"largeur_mm": 1}
    assert reg.fiche_debit(None, "faux", {})["indisponible"] == "pas de débit"
    with pytest.raises(KeyError):
        reg.options(None, "inconnu")


# ---- les routes ----------------------------------------------------------------------------


def test_routes_exigent_le_role_admin(client, lecteur_headers, responsable_headers):
    assert client.get("/api/faisabilite/systemes").status_code == 401
    for headers in (lecteur_headers, responsable_headers):
        assert client.get("/api/faisabilite/systemes", headers=headers).status_code == 403
        assert client.get("/api/faisabilite/s/perform76/options", headers=headers).status_code == 403
        assert client.post("/api/faisabilite/s/perform76", headers=headers, json={}).status_code == 403


def test_routes_systeme_inconnu_et_perform76_par_la_porte_generique(client, admin_headers):
    r = client.get("/api/faisabilite/systemes", headers=admin_headers)
    assert r.status_code == 200 and r.json()["systemes"][0]["id"] == "perform76"
    assert client.get("/api/faisabilite/s/inconnu/options", headers=admin_headers).status_code == 404
    assert client.post("/api/faisabilite/s/inconnu", headers=admin_headers, json={}).status_code == 404
    # PERFORM76 répond pareil par la porte générique et par la route d'origine.
    saisie = {"configuration": "1v_of", "largeur_mm": 1200, "hauteur_mm": 1400, "dormant": "76171", "ouvrant": "76281"}
    ancien = client.post("/api/faisabilite", headers=admin_headers, json=saisie).json()
    nouveau = client.post("/api/faisabilite/s/perform76", headers=admin_headers, json=saisie)
    assert nouveau.status_code == 200 and nouveau.json()["verdict"] == ancien["verdict"]
    assert [c["id"] for c in nouveau.json()["controles"]] == [c["id"] for c in ancien["controles"]]
    debit = client.post("/api/faisabilite/s/perform76/debit", headers=admin_headers, json=saisie)
    assert debit.status_code == 200 and debit.json()["lignes"]
