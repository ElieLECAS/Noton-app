"""Redirections pages HTML et endpoint /api/auth/me."""
from __future__ import annotations


def test_root_redirects_unauthenticated(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in (r.headers.get("location") or "")


def test_library_redirects_unauthenticated(client):
    r = client.get("/library", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in (r.headers.get("location") or "")


def test_space_page_redirects_unauthenticated(client):
    r = client.get("/spaces/1", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in (r.headers.get("location") or "")


def test_admin_page_redirects_unauthenticated(client):
    r = client.get("/admin", follow_redirects=False)
    assert r.status_code == 303
    assert "/login" in (r.headers.get("location") or "")


def test_root_ok_authenticated(client, responsable_headers):
    r = client.get("/", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 200


def test_login_redirects_when_already_authenticated(client, responsable_headers):
    token = responsable_headers["Authorization"].split(" ", 1)[1].strip()
    client.cookies.set("authToken", token)
    r = client.get("/login", follow_redirects=False)
    assert r.status_code == 303
    loc = r.headers.get("location") or ""
    assert loc == "/" or loc.endswith("/") and "login" not in loc


def test_api_auth_me_unauthorized(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401


def test_api_auth_me_ok_bearer(client, admin_headers):
    r = client.get("/api/auth/me", headers=admin_headers)
    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert "permissions" in data
    assert "config.manage_users" in data["permissions"]


def test_api_auth_me_ok_cookie(client, lecteur_headers):
    token = lecteur_headers["Authorization"].split(" ", 1)[1].strip()
    client.cookies.set("authToken", token)
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    body = r.json()
    assert "lecteur" in body.get("roles", [])
