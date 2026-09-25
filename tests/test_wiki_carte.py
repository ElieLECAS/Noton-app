"""La carte mentale : quels nœuds, dans quel ordre, et aucun niveau inutile."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest

from app.services.wiki_carte import PAGES_SANS_TYPE, TYPE_PLURIEL, carte
from app.services.wiki_service import load_snapshot, wiki_root


def _write(path: Path, meta: str, body: str = "Corps.") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{meta}\n---\n\n{body}\n", encoding="utf-8")


def _noeuds(n: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    yield n
    for c in n.get("children", []):
        yield from _noeuds(c)


def _enfant(n: Dict[str, Any], label: str) -> Dict[str, Any]:
    trouves = [c for c in n.get("children", []) if c["label"] == label]
    assert trouves, f"« {label} » absent sous « {n['label']} » : {[c['label'] for c in n.get('children', [])]}"
    return trouves[0]


def _chemin(n: Dict[str, Any], *labels: str) -> Dict[str, Any]:
    for label in labels:
        n = _enfant(n, label)
    return n


def _pages(n: Dict[str, Any]) -> List[str]:
    return [x["page"] for x in _noeuds(n) if x.get("page") and not x.get("children")]


@pytest.fixture
def arbre(tmp_path: Path) -> Dict[str, Any]:
    """Un petit wiki qui exerce chaque règle de la carte."""
    w = tmp_path / "wiki" / "wiki"
    _write(w / "index.md", "type: Réservé\ntitle: Index", "[Alpha](/gammes/alpha.md) [Fantôme](/x/fantome.md)")
    _write(w / "log.md", "type: Réservé\ntitle: Journal")
    # Deux systèmes déclarés : Alpha se range par système.
    _write(w / "gammes/alpha.md", "type: Gamme\ntitle: Alpha\ngamme: Alpha\nsysteme: [70, 76]\n"
                                  "fournisseur: ACME\ntags: [pvc]")
    _write(w / "profiles/a70.md", "type: Profilé\ntitle: Dormant 70\ngamme: Alpha\nsysteme: 70\nusage: atelier\n"
                                    "fournisseur: ACME")
    _write(w / "profiles/a76.md", "type: Profilé\ntitle: Dormant 76\ngamme: Alpha\nsysteme: 76\nfournisseur: ACME")
    # Page commune, sans gamme : n'ouvre dans Alpha que les systèmes qu'Alpha déclare.
    _write(w / "profiles/joints.md", "type: Profilé\ntitle: Joints communs\nsysteme: [70, 99]")
    # Un seul système : le niveau est sauté ; huit pages de trois types : rangées par type.
    _write(w / "gammes/beta.md", "type: Gamme\ntitle: Beta\ngamme: Beta\nsysteme: 80\ntags: [aluminium]")
    for i in range(5):
        _write(w / f"profiles/b{i}.md", f"type: Profilé\ntitle: Profil B{i}\ngamme: Beta\nsysteme: 80")
    for i in range(2):
        _write(w / f"quincaillerie/b{i}.md", f"type: Quincaillerie\ntitle: Ferrure B{i}\ngamme: Beta\nsysteme: 80")
    _write(w / "vitrages/b.md", "type: Vitrage\ntitle: Vitrage B\ngamme: Beta\nsysteme: 80")
    # Trois pages de types différents : listées telles quelles.
    _write(w / "gammes/gamma.md", "type: Gamme\ntitle: Gamma\ngamme: Gamma")
    _write(w / "profiles/g.md", "type: Profilé\ntitle: Profil G\ngamme: Gamma")
    _write(w / "vitrages/g.md", "type: Vitrage\ntitle: Vitrage G\ngamme: Gamma")
    _write(w / "garanties/g.md", "type: Garantie\ntitle: Garantie G\ngamme: Gamma")
    _write(w / "fournisseurs/acme.md", "type: Fournisseur\ntitle: ACME\nfournisseur: ACME")
    _write(w / "fournisseurs/vide.md", "type: Fournisseur\ntitle: Vide & Cie\nfournisseur: Vide & Cie")
    _write(w / "anomalies/inc.md", "type: Anomalie\ntitle: Incohérences\ngamme: Alpha")
    _write(w / "sources/doc.md", "type: Document source\ntitle: Catalogue Alpha\ngamme: Alpha")
    _write(w / "reference/glossaire.md", "type: Référence\ntitle: Glossaire")
    # Un système qu'aucune gamme ne déclare : la page ne doit pas disparaître de la carte.
    _write(w / "procedures/s90.md", "type: Procédure\ntitle: Pose 90\nsysteme: 90")
    return carte(load_snapshot(tmp_path / "wiki"))


def test_racine_et_branches(arbre):
    assert arbre["label"] == "Wiki PROFERM" and arbre["key"] == "r"
    assert [b["label"] for b in arbre["children"]] == [
        "Gammes", "Métiers", "Fournisseurs", "Hors gamme", "Registres d’anomalies", "Documents sources"]
    assert [b for b in arbre["children"] if b.get("cross")] == [_enfant(arbre, "Métiers")]


def test_ni_reservees_ni_fantomes_et_cles_uniques(arbre):
    tous = list(_noeuds(arbre))
    pages = {n.get("page") for n in tous}
    assert not pages & {"/index.md", "/log.md", "/x/fantome.md"}
    cles = [n["key"] for n in tous]
    assert len(cles) == len(set(cles))
    # La clé est le chemin d'indices : celle d'un enfant prolonge celle de son parent.
    for n in tous:
        for i, c in enumerate(n.get("children", [])):
            assert c["key"] == f"{n['key']}.{i}"


def test_aucune_branche_vide(arbre):
    for n in _noeuds(arbre):
        assert n.get("page") or n.get("children"), n["label"]
        assert n.get("children") is None or n["children"], n["label"]


def test_gammes_par_materiau(arbre):
    gammes = _enfant(arbre, "Gammes")
    assert [(m["label"], [g["label"] for g in m["children"]]) for m in gammes["children"]] == [
        ("PVC", ["Alpha"]), ("Aluminium", ["Beta"]), ("Autres gammes", ["Gamma"])]
    alpha = _chemin(gammes, "PVC", "Alpha")
    # Le nœud d'une gamme ouvre sa page et porte son entrée.
    assert alpha["page"] == "/gammes/alpha.md" and alpha["hub"] == ["gamme", "Alpha"]


def test_une_gamme_se_range_par_systeme(arbre):
    alpha = _chemin(arbre, "Gammes", "PVC", "Alpha")
    assert [c["label"] for c in alpha["children"]] == ["Système 70", "Système 76"]
    s70 = _enfant(alpha, "Système 70")
    assert s70["hub"] == ["systeme", "70"]
    assert _pages(s70) == ["/profiles/a70.md", "/profiles/joints.md"]
    assert _pages(_enfant(alpha, "Système 76")) == ["/profiles/a76.md"]


def test_page_commune_nouvre_que_les_systemes_declares(arbre):
    alpha = _chemin(arbre, "Gammes", "PVC", "Alpha")
    assert all("99" not in n["label"] for n in _noeuds(alpha))


def test_niveau_systeme_saute_si_un_seul(arbre):
    beta = _chemin(arbre, "Gammes", "Aluminium", "Beta")
    assert not any(n.get("hub", [""])[0] == "systeme" for n in _noeuds(beta))


def test_par_type_au_dela_de_six_pages(arbre):
    beta = _chemin(arbre, "Gammes", "Aluminium", "Beta")
    assert [c["label"] for c in beta["children"]] == ["Profilés", "Quincaillerie", "Vitrage B"]
    assert len(_enfant(beta, "Profilés")["children"]) == 5
    # Une feuille porte la description de sa page : l'infobulle de la carte.
    assert "description" in _enfant(_enfant(beta, "Profilés"), "Profil B0")
    # Un type d'une seule page ne fait pas un niveau : la page prend sa place.
    assert _enfant(beta, "Vitrage B")["page"] == "/vitrages/b.md"
    gamma = _chemin(arbre, "Gammes", "Autres gammes", "Gamma")
    assert len(gamma["children"]) == 3 <= PAGES_SANS_TYPE
    assert all(c.get("page") and not c.get("children") for c in gamma["children"])


def test_anomalies_et_sources_hors_des_gammes(arbre):
    gammes = _enfant(arbre, "Gammes")
    assert "/anomalies/inc.md" not in _pages(gammes)
    assert "/sources/doc.md" not in _pages(gammes)
    assert _pages(_enfant(arbre, "Registres d’anomalies")) == ["/anomalies/inc.md"]
    assert _pages(_enfant(arbre, "Documents sources")) == ["/sources/doc.md"]


def test_fournisseurs_et_hors_gamme(arbre):
    fournisseurs = _enfant(arbre, "Fournisseurs")
    acme = _enfant(fournisseurs, "ACME")
    assert acme["page"] == "/fournisseurs/acme.md" and acme["hub"] == ["fournisseur", "ACME"]
    # La page de gamme couvre deux systèmes : elle passe devant eux, pas dans chacun.
    assert [c["label"] for c in acme["children"]] == ["Alpha", "Système 70", "Système 76"]
    assert _pages(acme) == ["/gammes/alpha.md", "/profiles/a70.md", "/profiles/a76.md"]
    # Un fournisseur sans autre page reste : son nœud ouvre sa fiche.
    assert _enfant(fournisseurs, "Vide & Cie")["page"] == "/fournisseurs/vide.md"
    assert _pages(_enfant(arbre, "Hors gamme")) == ["/reference/glossaire.md", "/procedures/s90.md"]


def test_metiers_par_produit_sans_entree_vide(arbre):
    metiers = _enfant(arbre, "Métiers")
    # Seul « atelier » porte une page : les trois autres métiers n'ouvriraient qu'une entrée vide.
    assert [m["label"] for m in metiers["children"]] == ["Atelier et fabrication"]
    atelier = _enfant(metiers, "Atelier et fabrication")
    assert atelier["hub"] == ["usage", "atelier"]
    alpha = _enfant(atelier, "Alpha")
    assert alpha["page"] == "/gammes/alpha.md"
    assert _pages(alpha) == ["/profiles/a70.md"]


# ---- le vrai wiki, par l'API ------------------------------------------------------------

def test_carte_requires_auth(client):
    assert client.get("/api/wiki/carte").status_code == 401


def test_carte_du_vrai_wiki(client, lecteur_headers):
    r = client.get("/api/wiki/carte", headers=lecteur_headers)
    assert r.status_code == 200
    arbre = r.json()
    snap = load_snapshot(wiki_root())
    concept = {p.id: p for p in snap.concept_pages}
    tous = list(_noeuds(arbre))
    # Chaque nœud-page désigne une page écrite, jamais index, log ou fantôme.
    assert all(n["page"] in concept for n in tous if n.get("page"))
    # Toute page du wiki est quelque part sur la carte.
    assert set(concept) <= {n.get("page") for n in tous}
    # Toutes les gammes sont sous « Gammes ».
    gammes_carte = {n["hub"][1] for n in _noeuds(_enfant(arbre, "Gammes")) if n.get("hub", [""])[0] == "gamme"}
    assert gammes_carte == {g for p in concept.values() if p.type == "Gamme" for g in p.gamme}
    # Un regroupement par type n'existe qu'avec deux pages au moins.
    for n in tous:
        if n.get("children") and not n.get("hub") and n["label"] in TYPE_PLURIEL.values() \
                and n["key"].count(".") >= 2:
            assert len(n["children"]) >= 2, n["label"]
