"""Une page coupée en plusieurs : la tête et ses suites se lisent d'un seul tenant."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.services import wiki_famille
from app.services.faisabilite import _corps
from app.services.wiki_service import load_snapshot, wiki_root


def _write(path: Path, meta: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{meta}\n---\n\n{body}\n", encoding="utf-8")


@pytest.fixture
def snap(tmp_path: Path):
    w = tmp_path / "wiki" / "wiki"
    _write(w / "index.md", "type: Réservé\ntitle: Index", "")
    _write(w / "log.md", "type: Réservé\ntitle: Journal", "")
    _write(w / "profiles/tete.md", "type: Profilé\ntitle: Tête",
           "# Un\n\nCorps de la tête.\n\n# Voir aussi\n\nAutres parties de ce sujet :\n\n"
           "- [Suite B](/profiles/tete-b.md)\n- [Suite A](/profiles/tete-a.md)")
    for nom, titre in (("a", "Partie A"), ("b", "Partie B")):
        _write(w / f"profiles/tete-{nom}.md", f"type: Profilé\ntitle: Suite {nom.upper()}",
               f"Cette page reprend la suite {nom} ; le contexte est sur [Tête](/profiles/tete.md).\n\n"
               f"# {titre}\n\nCorps de {nom}.\n\n# Voir aussi\n\n{wiki_famille.PHRASE} 3 ; les autres :\n\n- [Tête](/profiles/tete.md)")
    # Un voisin de nom qui n'est PAS une suite.
    _write(w / "profiles/tete-manuel.md", "type: Profilé\ntitle: Manuel", "# Manuel\n\nAutre page.")
    return load_snapshot(tmp_path / "wiki")


def test_les_suites_viennent_des_marques_pas_du_nom(snap):
    # L'ordre est celui du « Voir aussi » de la tête, pas l'ordre alphabétique.
    assert wiki_famille.suites(snap, "/profiles/tete.md") == ["/profiles/tete-b.md", "/profiles/tete-a.md"]
    assert wiki_famille.suites(snap, "/profiles/tete-manuel.md") == []
    assert wiki_famille.suites(snap, "/profiles/tete-a.md") == []


def test_le_corps_de_la_famille_est_d_un_seul_tenant(snap):
    corps = wiki_famille.corps_famille(snap, "/profiles/tete.md")
    assert corps.index("Corps de la tête") < corps.index("Corps de b") < corps.index("Corps de a")
    assert "Autre page" not in corps
    # C'est ce que lisent les calculateurs.
    assert _corps(snap, "/profiles/tete.md") == corps
    assert _corps(snap, "/profiles/tete-manuel.md") == snap.pages["/profiles/tete-manuel.md"].body


def test_une_source_designe_la_partie_qui_porte_la_section(snap):
    assert wiki_famille.partie_de_section(snap, "/profiles/tete.md", "Partie A") == "/profiles/tete-a.md"
    assert wiki_famille.partie_de_section(snap, "/profiles/tete.md", "Un") == "/profiles/tete.md"
    assert wiki_famille.partie_de_section(snap, "/profiles/tete.md", "Inconnue") == "/profiles/tete.md"
    brut = {"controles": [{"sources": [{"page": "/profiles/tete.md", "section": "Partie B", "localisation": ""}]}],
            "debit": {"page": "/profiles/tete-manuel.md", "section": "Manuel"}}
    r = wiki_famille.relocaliser(snap, brut)
    assert r["controles"][0]["sources"][0]["page"] == "/profiles/tete-b.md"
    assert r["debit"]["page"] == "/profiles/tete-manuel.md"


# ---- le vrai wiki ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vrai():
    return load_snapshot(wiki_root())


def test_les_pages_lues_par_les_calculateurs_sont_des_familles(vrai):
    for page, n in (("/certifications/dta-6-16-2334.md", 4), ("/profiles/systeme-76-cotes-de-debit.md", 2),
                    ("/profiles/systeme-76-abaques-dimensionnels.md", 3),
                    ("/profiles/systeme-70-profiles-complementaires.md", 2)):
        assert len(wiki_famille.suites(vrai, page)) == n, page
    # Le « manuel » du système 70 est une autre page, pas une suite des profils complémentaires.
    assert "/profiles/systeme-70-profiles-complementaires-manuel.md" not in wiki_famille.suites(
        vrai, "/profiles/systeme-70-profiles-complementaires.md")


def test_aucune_suite_orpheline(vrai):
    """Toute page qui se dit partie d'un sujet a sa tête, et sa tête la liste."""
    fam = wiki_famille.familles(vrai)
    suites = {c for liste in fam.values() for c in liste}
    for p in vrai.concept_pages:
        if wiki_famille.PHRASE in p.body:
            assert p.id in suites, f"{p.id} se dit partie d'un sujet mais sa tête est introuvable"
