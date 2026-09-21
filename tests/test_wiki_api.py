"""L'API du wiki : graphe, pages, PDF sources, statistiques."""
from __future__ import annotations

from app.services.wiki_service import load_snapshot, wiki_root


def test_graph_requires_auth(client):
    assert client.get("/api/wiki/graph").status_code == 401
    assert client.get("/api/wiki/pages/index.md").status_code == 401


def test_graph_nodes_without_bodies(client, lecteur_headers):
    r = client.get("/api/wiki/graph", headers=lecteur_headers)
    assert r.status_code == 200
    data = r.json()
    ids = {n["id"] for n in data["nodes"]}
    assert "/index.md" in ids and "/profiles/perform76-parcloses.md" in ids
    assert all("body" not in n for n in data["nodes"])
    assert data["links"] and {"source", "target", "count"} <= set(data["links"][0])


def test_page_with_and_without_leading_slash(client, lecteur_headers):
    r1 = client.get("/api/wiki/pages/profiles/perform76-parcloses.md", headers=lecteur_headers)
    r2 = client.get("/api/wiki/pages//profiles/perform76-parcloses.md", headers=lecteur_headers)
    assert r1.status_code == 200
    page = r1.json()
    assert page["id"] == "/profiles/perform76-parcloses.md"
    assert page["type"] == "Profilé" and page["body"]
    assert page["inLinks"] and page["outLinks"]
    assert r2.status_code in (200, 404)  # la double barre dépend du routeur, jamais d'un autre fichier


def test_unknown_page_and_traversal_are_404(client, lecteur_headers):
    assert client.get("/api/wiki/pages/nexiste/pas.md", headers=lecteur_headers).status_code == 404
    assert client.get("/api/wiki/pages/..%2F..%2Fapp%2Fconfig.py", headers=lecteur_headers).status_code == 404
    assert client.get("/api/wiki/pages/../../.env", headers=lecteur_headers).status_code == 404


def test_raw_pdf(client, lecteur_headers):
    assert client.get("/api/wiki/raw/inconnu.pdf", headers=lecteur_headers).status_code == 404
    assert client.get("/api/wiki/raw/..%2F.env", headers=lecteur_headers).status_code == 404
    snap = load_snapshot(wiki_root())
    if snap.raw_files:
        r = client.get(f"/api/wiki/raw/{snap.raw_files[0]}", headers=lecteur_headers)
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")


def test_stats_admin_only(client, lecteur_headers, admin_headers):
    assert client.get("/api/wiki/stats", headers=lecteur_headers).status_code == 403
    r = client.get("/api/wiki/stats", headers=admin_headers)
    assert r.status_code == 200
    stats = r.json()
    assert stats["pages"] >= 60
    assert {"links", "ghosts", "orphans", "stale", "drafts", "types", "chars",
            "estimated_tokens", "token_warning", "cache_key", "loaded_at", "lint"} <= set(stats)
    assert "last_call" in stats


def test_search_requires_auth_and_reads_the_body(client, lecteur_headers):
    assert client.get("/api/wiki/search?q=parclose").status_code == 401
    r = client.get("/api/wiki/search?q=parclose", headers=lecteur_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["q"] == "parclose"
    assert "/profiles/perform76-parcloses.md" in {p["id"] for p in data["pages"]}
    assert all({"id", "excerpt"} == set(p) for p in data["pages"])
    assert client.get("/api/wiki/search", headers=lecteur_headers).json()["pages"] == []
