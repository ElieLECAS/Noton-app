"""Les sommaires du modèle : à quel produit va chaque page, sur quel chemin, et ce que dit la racine."""
from __future__ import annotations

import re
from typing import List

import pytest

from app.services import wiki_sommaires
from app.services.wiki_service import WikiPage, load_snapshot, wiki_root
from app.services.wiki_sommaires import PROFERM, chemin_du_sommaire, sommaires


def _page(chemin: str, type_: str = "Profilé", **facettes: List[str]) -> WikiPage:
    return WikiPage(id=chemin, title=chemin.rsplit("/", 1)[-1][:-3].replace("-", " ").capitalize(),
                    type=type_, description=f"Ce que porte {chemin}.", **facettes)


@pytest.fixture
def wiki() -> List[WikiPage]:
    return [
        _page("/gammes/alpha.md", "Gamme", gamme=["Alpha"], systeme=["70"], fournisseur=["ACME"]),
        _page("/profiles/dormants-76.md", gamme=["Alpha"], systeme=["76"], fournisseur=["ACME"], famille=["dormants"]),
        _page("/profiles/cotes-76.md", gamme=["Alpha"], systeme=["76"], fournisseur=["ACME"], famille=["cotes-de-debit"]),
        _page("/profiles/divers-76.md", systeme=["76"], fournisseur=["ACME"]),
        _page("/certifications/dta-76.md", "Certification", systeme=["76"], fournisseur=["ACME"]),
        _page("/sources/manuel.md", "Document source", systeme=["70", "76"], fournisseur=["ACME"]),
        _page("/portes/alpha-portes.md", "Porte d'entrée", gamme=["Alpha"]),
        _page("/commercial/alpha.md", "Commercial", gamme=["Alpha"]),
        # Une page de gamme qui ne cite qu'une ferrure : elle ne la donne pas à son fournisseur.
        _page("/gammes/beta.md", "Gamme", gamme=["Beta"], systeme=["Ferrure NX"], fournisseur=["ACME"]),
        _page("/quincaillerie/nx-1.md", "Quincaillerie", systeme=["Ferrure NX"], fournisseur=["FERRO"]),
        _page("/quincaillerie/nx-2.md", "Quincaillerie", systeme=["Ferrure NX"], fournisseur=["FERRO"]),
        _page("/reference/glossaire.md", "Référence"),
        WikiPage(id="/index.md", title="Index", type="Réservé", reserved=True),
        WikiPage(id="/x/fantome.md", title="Fantôme", type="À écrire", missing=True),
    ]


def _lignes(texte: str) -> List[str]:
    return [l for l in texte.splitlines() if l.startswith("* [")]


def test_chaque_page_va_a_son_produit(wiki):
    s = sommaires(wiki)
    assert {c: [p.id for p in x.pages] for c, x in s.par_chemin.items()} == {
        "/index/alpha.md": ["/commercial/alpha.md", "/portes/alpha-portes.md"],
        "/index/ferrure-nx.md": ["/gammes/beta.md", "/quincaillerie/nx-1.md", "/quincaillerie/nx-2.md"],
        "/index/systeme-70.md": ["/gammes/alpha.md", "/sources/manuel.md"],
        "/index/systeme-76.md": ["/certifications/dta-76.md", "/profiles/cotes-76.md",
                                 "/profiles/divers-76.md", "/profiles/dormants-76.md", "/sources/manuel.md"],
        "/index/transverse.md": ["/reference/glossaire.md"],
    }
    # Ni réservée ni fantôme ; une page de deux systèmes est dans les deux.
    assert [x.chemin for x in s.de_page("/sources/manuel.md")] == ["/index/systeme-70.md", "/index/systeme-76.md"]
    assert s.de_page("/index.md") == [] and s.de_page("/x/fantome.md") == []


def test_chemins_des_sommaires():
    assert chemin_du_sommaire("76") == "/index/systeme-76.md"
    assert chemin_du_sommaire("HYBRIDE+") == "/index/hybride-plus.md" != chemin_du_sommaire("HYBRIDE")
    assert chemin_du_sommaire("Chrono PSE²") == "/index/chrono-pse2.md"
    assert chemin_du_sommaire("Frappe 65 Ouvrant Caché") == "/index/frappe-65-ouvrant-cache.md"


def test_une_collision_est_signalee_sans_perdre_de_produit():
    s = sommaires([_page("/a.md", systeme=["Roto NX"]), _page("/b.md", systeme=["Roto-NX"])])
    assert sorted(x.produit for x in s.par_chemin.values()) == ["Roto NX", "Roto-NX"]
    assert len(s.par_chemin) == 2 and any("même chemin" in a for a in s.alertes)


def test_fournisseur_d_un_produit(wiki):
    s = sommaires(wiki)
    assert s.du_produit("Ferrure NX").fournisseur == "FERRO"
    assert s.du_produit("76").fournisseur == "ACME"
    # Aucune page de la gamme ne nomme de fournisseur : c'est une gamme PROFERM.
    assert s.du_produit("Alpha").fournisseur == PROFERM


def test_un_sommaire_se_lit_par_type_puis_par_famille(wiki):
    texte = sommaires(wiki).du_produit("76").texte
    assert re.findall(r"^## (.+)$", texte, re.M) == ["Certifications (1)", "Profilés (3)", "Documents sources (1)"]
    # La famille trie et s'affiche ; une page sans famille vient après.
    profils = _lignes(texte.split("## Profilés")[1].split("## ")[0])
    assert profils == [
        "* [Cotes 76](/profiles/cotes-76.md) `cotes-de-debit` - Ce que porte /profiles/cotes-76.md.",
        "* [Dormants 76](/profiles/dormants-76.md) `dormants` - Ce que porte /profiles/dormants-76.md.",
        "* [Divers 76](/profiles/divers-76.md) - Ce que porte /profiles/divers-76.md.",
    ]
    assert "gammes PROFERM : Alpha" in texte
    assert "Pages communes avec : Système 70 (1) /index/systeme-70.md" in texte


def test_la_racine_mene_a_chaque_sommaire(wiki):
    s = sommaires(wiki)
    for chemin in s.par_chemin:
        assert chemin in s.racine, chemin
    # Une gamme : sa page, ses pages propres, puis ses systèmes, le plus employé d'abord
    # (76 porte deux pages Alpha, 70 seulement la page de gamme).
    alpha = next(l for l in s.racine.splitlines() if l.startswith("- **Alpha**"))
    assert alpha == ("- **Alpha** — [Alpha](/gammes/alpha.md) · PROFERM, 2 pages propres à la gamme : /index/alpha.md"
                     " · systèmes : Système 76 (5 pages) /index/systeme-76.md, Système 70 (2 pages) /index/systeme-70.md")
    # Le sommaire d'une gamme n'est pas répété parmi les systèmes.
    assert s.racine.count("/index/alpha.md") == 1
    assert "### FERRO" in s.racine and "- **Ferrure NX** — 3 pages" in s.racine


def test_un_sommaire_trop_gros_est_signale(wiki, monkeypatch):
    monkeypatch.setattr(wiki_sommaires, "SOMMAIRE_MAX", 300)
    assert any(a.startswith("/index/systeme-76.md") for a in sommaires(wiki).alertes)


# ---- le vrai wiki ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vrai():
    snap = load_snapshot(wiki_root())
    return snap, sommaires(snap.pages.values())


def test_vrai_wiki_aucune_page_ne_manque(vrai):
    snap, s = vrai
    rangees = {p.id for x in s.par_chemin.values() for p in x.pages}
    assert rangees == {p.id for p in snap.concept_pages}
    assert s.alertes == []


def test_vrai_wiki_tailles(vrai):
    snap, s = vrai
    index = snap.pages["/index.md"].raw_text
    # La racine remplace l'index plat du prompt : au moins vingt fois plus courte.
    assert len(s.racine) * 20 < len(index)
    assert max(len(x.texte) for x in s.par_chemin.values()) <= wiki_sommaires.SOMMAIRE_MAX


def test_vrai_wiki_produits_et_fournisseurs(vrai):
    _, s = vrai
    assert s.du_produit("Roto NX").fournisseur == "ROTO"
    assert s.du_produit("76").fournisseur == "KÖMMERLING"
    assert s.du_produit("SOLEAL FY").fournisseur == "TECHNAL"
    assert s.du_produit("Frappe 65 Ouvrant Caché").fournisseur == "ASKEY"
    # La carte des directives profine sert les deux systèmes.
    assert [x.produit for x in s.de_page("/sources/profine-directives-generales.md")] == ["70", "76"]
    lumine = next(l for l in s.racine.splitlines() if l.startswith("- **LUMINE**"))
    for systeme in ("SOLEAL FY", "SOLEAL GY", "SOLEAL PY", "LUMEAL GA"):
        assert systeme in lumine
