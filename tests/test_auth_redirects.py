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


def test_legacy_addresses_redirect_to_chat(client, responsable_headers):
    for url in ("/spaces/1", "/library", "/admin/sav-trees"):
        r = client.get(url, headers=responsable_headers, follow_redirects=False)
        assert r.status_code == 303
        assert r.headers.get("location") == "/"


def test_root_ok_authenticated(client, responsable_headers):
    r = client.get("/", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 200
    assert "/api/chat/stream" in r.text


def test_wiki_page_ok_authenticated(client, responsable_headers):
    r = client.get("/wiki", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 200
    assert "/api/wiki/graph" in r.text


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
