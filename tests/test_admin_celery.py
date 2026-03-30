"""Admin (users / rôles) et dispatch Celery bibliothèque."""
from __future__ import annotations

from unittest import mock

from app.config import settings
from app.services import task_dispatch


def test_admin_users_forbidden_lecteur(client, lecteur_headers):
    r = client.get("/api/admin/users", headers=lecteur_headers)
    assert r.status_code == 403


def test_admin_users_ok_admin(client, admin_headers):
    r = client.get("/api/admin/users", headers=admin_headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_admin_create_user_and_assign_role_me_permissions(client, admin_headers):
    from tests.conftest import bearer_headers

    r = client.get("/api/admin/roles", headers=admin_headers)
    assert r.status_code == 200
    roles = {row["name"]: row["id"] for row in r.json()}
    assert "lecteur" in roles

    suffix = __import__("uuid").uuid4().hex[:8]
    create = client.post(
        "/api/admin/users",
        headers=admin_headers,
        json={
            "username": f"admintest_{suffix}",
            "email": f"admintest_{suffix}@pytest.example.com",
            "password": "SecurePass123!",
        },
    )
    assert create.status_code == 201
    new_id = create.json()["id"]

    ar = client.post(
        "/api/admin/users/assign-role",
        headers=admin_headers,
        json={"user_id": new_id, "role_id": roles["lecteur"]},
    )
    assert ar.status_code == 200

    me = client.get("/api/auth/me", headers=bearer_headers(new_id))
    assert me.status_code == 200
    perms = me.json().get("permissions", [])
    assert "library.read" in perms
    assert "library.write" not in perms


def test_dispatch_library_document_enqueues_celery():
    mock_result = mock.MagicMock()
    mock_result.id = "celery-task-pytest"

    with mock.patch.object(settings, "TASK_BACKEND_MODE", "celery"):
        with mock.patch(
            "app.tasks.documents.process_library_document.apply_async",
            return_value=mock_result,
        ) as apply_async:
            task_dispatch.dispatch_library_document(42, "/data/file.pdf")

    apply_async.assert_called_once_with(
        args=[42, "/data/file.pdf"],
        queue="documents",
    )
