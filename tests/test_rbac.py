"""Permissions RBAC : seule l'administration est protégée ; chat et wiki sont ouverts."""
from __future__ import annotations

from sqlmodel import select

from app.models.permission import Permission
from app.models.role import Role
from app.models.role_permission import RolePermission


def test_admin_users_forbidden_for_lecteur_and_responsable(client, lecteur_headers, responsable_headers):
    assert client.get("/api/admin/users", headers=lecteur_headers).status_code == 403
    assert client.get("/api/admin/users", headers=responsable_headers).status_code == 403
    assert client.get("/api/admin/roles", headers=responsable_headers).status_code == 403


def test_admin_endpoints_ok_for_admin(client, admin_headers):
    assert client.get("/api/admin/users", headers=admin_headers).status_code == 200
    assert client.get("/api/admin/roles", headers=admin_headers).status_code == 200
    r = client.get("/api/admin/feedbacks/stats", headers=admin_headers)
    assert r.status_code == 200
    assert {"total", "positive", "negative", "ratio", "classifications"} == set(r.json())
    r = client.get("/api/admin/conversations", headers=admin_headers)
    assert r.status_code == 200
    assert "total_conversations" in r.json()


def test_chat_and_wiki_open_to_every_role(client, lecteur_headers):
    assert client.get("/api/wiki/graph", headers=lecteur_headers).status_code == 200
    r = client.post("/api/conversations", headers=lecteur_headers, json={"title": "Lecteur"})
    assert r.status_code == 201
    client.delete(f"/api/conversations/{r.json()['id']}", headers=lecteur_headers)


def test_seeded_roles_only_carry_admin_permissions(db_session):
    codes = {p.code for p in db_session.exec(select(Permission)).all()}
    assert {"config.manage_users", "config.manage_roles"} <= codes
    assert not any(c.startswith(("library.", "space.", "feedback.")) for c in codes
                   if c in {"library.read", "library.write", "space.create", "space.read",
                            "space.update", "space.delete", "feedback.auto_faq"}
                   and _role_has(db_session, c))


def _role_has(session, code: str) -> bool:
    perm = session.exec(select(Permission).where(Permission.code == code)).first()
    if perm is None:
        return False
    return session.exec(select(RolePermission).where(RolePermission.permission_id == perm.id)).first() is not None


def test_lecteur_and_responsable_have_no_permissions(db_session):
    for name in ("lecteur", "responsable"):
        role = db_session.exec(select(Role).where(Role.name == name)).first()
        assert role is not None
        links = db_session.exec(select(RolePermission).where(RolePermission.role_id == role.id)).all()
        assert links == []
