"""Redirections des pages HTML et endpoint /api/auth/me."""
from __future__ import annotations


def test_root_redirects_unauthenticated(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in (r.headers.get("location") or "")


def test_wiki_redirects_unauthenticated(client):
    r = client.get("/wiki", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in (r.headers.get("location") or "")


def test_admin_page_redirects_unauthenticated(client):
    r = client.get("/admin", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in (r.headers.get("location") or "")


def test_root_ok_authenticated(client, responsable_headers):
    r = client.get("/", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 200
    assert "/api/chat/stream" in r.text


def test_wiki_page_ok_authenticated(client, responsable_headers):
    r = client.get("/wiki", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 200
    assert "/api/wiki/graph" in r.text  # la liste des pages, pas une vue graphe
    # Plus de vue graphe ni de raccourci vers la carte : elle a son entrée dans la navigation.
    for absent in ('id="view-graph"', 'id="view-map"', 'id="btn-graph"', 'id="btn-map"', 'id="btn-locate"', "d3.min.js"):
        assert absent not in r.text, absent


def test_carte_redirects_unauthenticated(client):
    r = client.get("/carte", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in (r.headers.get("location") or "")


def test_carte_page_ok_authenticated(client, responsable_headers):
    r = client.get("/carte", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 200
    assert "/api/wiki/carte" in r.text
    # Un lien à part entière dans la barre de navigation, présent sur toutes les pages.
    assert 'id="nav-carte"' in r.text and 'href="/carte"' in r.text
    wiki = client.get("/wiki", headers=responsable_headers, follow_redirects=False)
    assert 'id="nav-carte"' in wiki.text


def test_login_redirects_when_already_authenticated(client, responsable_headers):
    token = responsable_headers["Authorization"].split(" ", 1)[1].strip()
    client.cookies.set("authToken", token)
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers.get("location") or ""
    assert loc == "/" or loc.endswith("/") and "login" not in loc


def test_api_auth_me_unauthorized(client):
    assert client.get("/api/auth/me").status_code == 401


def test_api_auth_me_ok_bearer(client, admin_headers):
    r = client.get("/api/auth/me", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert "config.manage_users" in data["permissions"]


def test_api_auth_me_ok_cookie(client, lecteur_headers):
    token = lecteur_headers["Authorization"].split(" ", 1)[1].strip()
    client.cookies.set("authToken", token)
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    assert r.json()["permissions"] == []
