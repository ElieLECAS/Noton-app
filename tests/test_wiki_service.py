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


def test_prompt_order_and_exclusions(real_snapshot):
    prompt = real_snapshot.system_prompt
    consignes = wiki_service.CONSIGNES_PATH.read_text(encoding="utf-8").rstrip()
    assert prompt.startswith(consignes)
    assert "/log.md =====" not in prompt
    i_index = prompt.index("===== TABLE DES MATIÈRES /index.md =====")
    i_anomalies = prompt.index("===== PAGE /anomalies/")
    i_profiles = prompt.index("===== PAGE /profiles/")
    assert i_index < i_anomalies < i_profiles
    assert prompt.endswith("===== FIN DU WIKI =====")
    # Toutes les pages concept y sont, une fois.
    for page in real_snapshot.concept_pages:
        assert prompt.count(f"===== PAGE {page.id} =====") == 1


def test_cache_key_covers_whole_prompt(real_snapshot):
    pages = real_snapshot.pages
    _, k1 = build_system_prompt(pages, "consignes A")
    _, k1_again = build_system_prompt(pages, "consignes A")
    _, k2 = build_system_prompt(pages, "consignes B")
    assert k1 == k1_again
    assert k1 != k2
    assert k1.startswith("lia-wiki-") and len(k1) == len("lia-wiki-") + 32


def test_estimated_tokens_and_stats(real_snapshot):
    stats = real_snapshot.stats()
    assert stats["pages"] == len(real_snapshot.concept_pages)
    assert stats["estimated_tokens"] == round(real_snapshot.char_count / wiki_service.CHARS_PER_TOKEN)
    assert set(stats["lint"]) == {"frontmatter_errors", "not_in_index", "index_dead_links"}
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
    assert second.cache_key != key_before
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
