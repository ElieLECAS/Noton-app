"""Le wiki : chargement, prompt système, clé de cache, lint, rechargement."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from app.config import settings
from app.services import wiki_service
from app.services.wiki_service import (
    WikiUnavailable,
    build_system_prompt,
    get_snapshot,
    load_snapshot,
    reset_snapshot,
    wiki_root,
)


@pytest.fixture(scope="module")
def real_snapshot():
    return load_snapshot(wiki_root())


def test_real_wiki_loads(real_snapshot):
    snap = real_snapshot
    assert len(snap.concept_pages) >= 60
    assert snap.pages["/index.md"].reserved and snap.pages["/log.md"].reserved
    parcloses = snap.pages["/profiles/perform76-parcloses.md"]
    assert parcloses.type == "Profilé"
    assert parcloses.title.startswith("Parcloses")
    assert parcloses.out_links and parcloses.in_degree > 0
    # Chaque lien relie deux nœuds connus (page ou fantôme).
    assert all(l["source"] in snap.pages and l["target"] in snap.pages for l in snap.links)
    assert snap.raw_files == sorted(snap.raw_files)


def test_le_prompt_permanent_ne_contient_pas_le_wiki(real_snapshot):
    """Consignes, vocabulaire, index des anomalies — et rien du corps des pages."""
    prompt = real_snapshot.system_prompt
    consignes = wiki_service.CONSIGNES_PATH.read_text(encoding="utf-8").rstrip()
    assert prompt.startswith(consignes)
    assert "===== VOCABULAIRE DU WIKI" in prompt
    assert "===== INDEX DES ANOMALIES" in prompt
    # L'index des anomalies ne porte qu'un identifiant et un sujet par entrée.
    assert "CTR-09 | " in prompt
    # Aucune page n'est recopiée : c'est tout l'objet de la navigation outillée.
    assert "===== PAGE /" not in prompt
    for page in real_snapshot.concept_pages[:20]:
        if len(page.body) > 400:
            assert page.body[:400] not in prompt
    # Il reste petit : il est payé à chaque appel d'un tour d'outils.
    assert real_snapshot.estimated_tokens < wiki_service.TOKEN_WARNING_THRESHOLD


def test_cache_key_covers_whole_prompt(real_snapshot):
    index = real_snapshot.index
    _, k1 = build_system_prompt(index, "consignes A")
    _, k1_again = build_system_prompt(index, "consignes A")
    _, k2 = build_system_prompt(index, "consignes B")
    assert k1 == k1_again
    assert k1 != k2
    assert k1.startswith("lia-wiki-") and len(k1) == len("lia-wiki-") + 32


def test_estimated_tokens_and_stats(real_snapshot):
    stats = real_snapshot.stats()
    assert stats["pages"] == len(real_snapshot.concept_pages)
    # Le prompt permanent se mesure toujours (le journal avertit s'il enfle), mais il ne sort
    # plus dans les statistiques : l'administration ne montre que le poids du wiki.
    assert real_snapshot.estimated_tokens == round(
        real_snapshot.char_count / wiki_service.CHARS_PER_TOKEN
    )
    assert stats["wiki_chars"] == sum(len(p.raw_text) for p in real_snapshot.concept_pages)
    assert stats["wiki_chars"] > real_snapshot.char_count * 10
    assert set(stats["lint"]) == {
        "frontmatter_errors", "not_in_index", "index_dead_links", "unknown_facets"
    }
    assert stats["types"]["Document source"] >= 1
    graph = real_snapshot.graph_payload()
    assert all("body" not in n for n in graph["nodes"])
    assert {"id", "title", "type", "inDegree", "outDegree", "missing", "reserved"} <= set(graph["nodes"][0])


def test_page_payload_and_raw_path(real_snapshot):
    payload = real_snapshot.page_payload("/profiles/perform76-parcloses.md")
    assert payload is not None
    assert "body" in payload and payload["body"]
    assert isinstance(payload["sources"], list) and payload["sources"][0].get("resource")
    assert payload["inLinks"] and payload["outLinks"]
    assert real_snapshot.page_payload("/nexiste/pas.md") is None
    assert real_snapshot.raw_path("../.env") is None
    assert real_snapshot.raw_path("..\\..\\.env") is None
    assert real_snapshot.raw_path("inconnu.pdf") is None
    if real_snapshot.raw_files:
        assert real_snapshot.raw_path(real_snapshot.raw_files[0]) is not None


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def small_wiki(tmp_path: Path) -> Path:
    root = tmp_path / "connaissance"
    wiki = root / "wiki"
    _write(wiki / "index.md", "# Gammes\n\n* [Alpha](/gammes/alpha.md) - La gamme alpha.\n")
    _write(wiki / "log.md", "# Update Log\n\n## 2026-09-18\n\n* **Create**: alpha\n")
    _write(
        wiki / "gammes" / "alpha.md",
        "---\ntype: Gamme\ntitle: Gamme Alpha\ndescription: La gamme alpha.\nstatus: stable\n"
        "tags: [alpha]\nsources:\n  - resource: raw/alpha.pdf\n    title: Brochure alpha\n"
        "stale_after: 2020-01-01\n---\n\n# Alpha\n\nVoir [Beta](/gammes/beta.md) et [Fantôme](/normes/fantome.md)."
        " Encore [Beta](/gammes/beta.md).\n",
    )
    _write(
        wiki / "gammes" / "beta.md",
        "---\ntype: Gamme\ntitle: Gamme Beta\ndescription: La gamme beta.\nstatus: draft\n---\n\n# Beta\n\nSeule.\n",
    )
    _write(wiki / "gammes" / "cassee.md", "---\ntitle: [pas fermé\n---\n\nCorps.\n")
    (root / "raw").mkdir()
    (root / "raw" / "alpha.pdf").write_bytes(b"%PDF-1.4 test")
    return root


def test_small_wiki_graph_and_lint(small_wiki: Path):
    snap = load_snapshot(small_wiki)
    alpha, beta = snap.pages["/gammes/alpha.md"], snap.pages["/gammes/beta.md"]
    ghost = snap.pages["/normes/fantome.md"]
    assert ghost.missing and ghost.type == "À écrire" and ghost.folder == "normes"
    assert alpha.out_degree == 2  # beta (compté une fois, count=2) + fantôme
    beta_link = next(l for l in snap.links if l["source"] == alpha.id and l["target"] == beta.id)
    assert beta_link["count"] == 2
    assert beta.in_degree == 1 and ghost.in_degree == 1
    assert alpha.stale is True and beta.stale is False
    assert alpha.source_titles == ["Brochure alpha"]

    lint = snap.lint
    assert "/gammes/beta.md" in lint["not_in_index"]
    assert "/gammes/cassee.md" in lint["not_in_index"]
    assert lint["ghosts"] == ["/normes/fantome.md"]
    assert lint["stale"] == ["/gammes/alpha.md"]
    assert lint["drafts"] == ["/gammes/beta.md"]
    assert any(e.startswith("/gammes/cassee.md") for e in lint["frontmatter_errors"])
    # Alpha n'est citée par aucune page concept (seulement par l'index) : orpheline.
    assert "/gammes/alpha.md" in lint["orphans"]
    assert "/gammes/beta.md" not in lint["orphans"]

    assert snap.raw_files == ["alpha.pdf"]
    assert snap.raw_path("alpha.pdf") == small_wiki / "raw" / "alpha.pdf"
    assert "/log.md =====" not in snap.system_prompt
    assert snap.page_payload("/normes/fantome.md") is None
    node = snap.page_payload("/gammes/alpha.md")
    assert node["outLinks"][1]["missing"] is True
    assert node["sources"][0]["resource"] == "raw/alpha.pdf"


def test_real_wiki_facettes_de_navigation(real_snapshot):
    """Chaque gamme et chaque fournisseur cités ont leur page : aucune entrée fantôme dans la
    navigation, et les facettes sortent dans le graphe que lit l'interface."""
    assert real_snapshot.lint["unknown_facets"] == []
    dormants = real_snapshot.pages["/profiles/perform76-dormants.md"]
    assert dormants.gamme == ["PERFORM"] and dormants.systeme == ["76"]
    assert dormants.fournisseur == ["KÖMMERLING"] and dormants.usage == ["atelier"]
    node = next(n for n in real_snapshot.graph_payload()["nodes"] if n["id"] == dormants.id)
    assert node["gamme"] == ["PERFORM"] and node["usage"] == ["atelier"]
    # Le système nomme le système du fournisseur, jamais une profondeur de profilé.
    systemes = {s for p in real_snapshot.concept_pages for s in p.systeme}
    assert not systemes & {"55", "65", "100"}


def test_facettes_inconnues_au_lint(small_wiki: Path):
    wiki = small_wiki / "wiki"
    _write(
        wiki / "profiles" / "gamma.md",
        "---\ntype: Profilé\ntitle: Gamma\ngamme: Alpah\nfournisseur: ACME\nusage: [atelier, vente]\n"
        "---\n\n# Gamma\n",
    )
    _write(wiki / "gammes" / "alpha.md", "---\ntype: Gamme\ntitle: Alpha\ngamme: Alpha\n---\n\n# Alpha\n")
    fautes = load_snapshot(small_wiki).lint["unknown_facets"]
    assert fautes == [
        "/profiles/gamma.md — fournisseur « ACME » sans page fournisseur",
        "/profiles/gamma.md — gamme « Alpah » sans page de gamme",
        "/profiles/gamma.md — usage « vente » inconnu",
    ]


def test_get_snapshot_reloads_when_a_file_changes(small_wiki: Path, monkeypatch):
    monkeypatch.setattr(settings, "WIKI_DIR", str(small_wiki))
    reset_snapshot()
    first = get_snapshot()
    assert get_snapshot() is first
    key_before = first.cache_key

    page = small_wiki / "wiki" / "gammes" / "beta.md"
    page.write_text(page.read_text(encoding="utf-8") + "\nUne ligne de plus.\n", encoding="utf-8")
    future = time.time() + 5
    os.utime(page, (future, future))

    second = get_snapshot()
    assert second is not first
    # Le corps d'une page ne figure plus dans le prompt permanent : la clé de cache ne bouge
    # donc pas, et c'est bien ce qu'on veut — le préfixe mis en cache reste valide. Ce qui doit
    # changer, c'est le contenu servi par l'index.
    assert second.cache_key == key_before
    assert "Une ligne de plus." in second.pages["/gammes/beta.md"].body
    entree = next(e for e in second.index.entries if e["chemin"] == "/gammes/beta.md")
    assert "Une ligne de plus." in entree["corps"]
    reset_snapshot()


def test_le_vocabulaire_change_invalide_le_cache(small_wiki: Path, monkeypatch):
    """Un tag nouveau entre dans le vocabulaire, donc dans le prompt : la clé doit bouger."""
    monkeypatch.setattr(settings, "WIKI_DIR", str(small_wiki))
    reset_snapshot()
    avant = get_snapshot().cache_key

    page = small_wiki / "wiki" / "gammes" / "alpha.md"
    texte = page.read_text(encoding="utf-8").replace("tags: [alpha]", "tags: [alpha, tag-tout-neuf]", 1)
    page.write_text(texte, encoding="utf-8")
    future = time.time() + 5
    os.utime(page, (future, future))

    apres = get_snapshot()
    assert "tag-tout-neuf" in apres.system_prompt
    assert apres.cache_key != avant
    reset_snapshot()


def test_missing_wiki_dir_raises(tmp_path: Path):
    with pytest.raises(WikiUnavailable):
        load_snapshot(tmp_path / "nulle-part")


def test_search_reaches_the_body(small_wiki: Path):
    snap = load_snapshot(small_wiki)
    # « Seule. » n'est ni dans un titre ni dans une description : seul le corps le porte.
    hits = snap.search("seule")
    assert [h["id"] for h in hits] == ["/gammes/beta.md"]
    assert hits[0]["excerpt"] == "Seule."
    # Tous les mots doivent tomber dans la MÊME page.
    assert [h["id"] for h in snap.search("alpha fantôme")] == ["/gammes/alpha.md"]
    assert snap.search("seule fantôme") == []
    # Réservées (index.md, log.md) et fantômes restent hors recherche, requête vide aussi.
    assert snap.search("update log") == []
    assert snap.search("   ") == []
    # Le chemin compte aussi : beta se trouve par son identifiant (aucun extrait, le mot
    # n'est pas dans son corps), alpha par le lien qui la cite.
    by_path = {h["id"]: h["excerpt"] for h in snap.search("gammes/beta.md")}
    assert by_path["/gammes/beta.md"] == ""
    assert "Beta" in by_path["/gammes/alpha.md"]
