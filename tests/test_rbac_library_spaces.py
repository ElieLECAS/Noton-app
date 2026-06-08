"""Permissions RBAC : bibliothèque et espaces."""
from __future__ import annotations

from unittest import mock

from app.models.space import Space
from tests.conftest import create_test_user


def test_get_library_ok_lecteur(client, lecteur_headers):
    r = client.get("/api/library", headers=lecteur_headers)
    assert r.status_code == 200
    assert "id" in r.json()


def test_upload_forbidden_lecteur(client, lecteur_headers):
    with (
        mock.patch("app.routers.library.process_document_async"),
        mock.patch(
            "app.routers.library.save_uploaded_file",
            return_value="media/documents/pytest_fake.pdf",
        ),
    ):
        r = client.post(
            "/api/library/upload",
            headers=lecteur_headers,
            files=[("files", ("a.txt", b"hello", "text/plain"))],
            data={"space_ids": "[]", "is_paid": "false"},
        )
    assert r.status_code == 403
    detail = (r.json().get("detail") or "").lower()
    assert "library.write" in detail or "permission" in detail


@mock.patch("app.routers.library.process_document_async")
@mock.patch(
    "app.routers.library.save_uploaded_file",
    return_value="media/documents/pytest_fake.pdf",
)
def test_upload_ok_responsable(mock_save, mock_proc, client, responsable_headers):
    r = client.post(
        "/api/library/upload",
        headers=responsable_headers,
        files=[("files", ("doc.txt", b"content", "text/plain"))],
        data={"space_ids": "[]", "is_paid": "false"},
    )
    assert r.status_code == 201
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_create_space_forbidden_lecteur(client, lecteur_headers):
    r = client.post(
        "/api/spaces",
        headers=lecteur_headers,
        json={"name": "Espace test", "description": None},
    )
    assert r.status_code == 403


def test_create_space_ok_responsable(client, responsable_headers):
    r = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace pytest", "description": "d"},
    )
    assert r.status_code == 201
    assert r.json()["name"] == "Espace pytest"


def test_delete_space_forbidden_lecteur(client, lecteur_headers):
    r = client.delete("/api/spaces/999999", headers=lecteur_headers)
    assert r.status_code == 403


def test_get_foreign_space_404(client, responsable_headers, db_session):
    other = create_test_user(db_session, "responsable")
    space = Space(name="Privé autre user", user_id=other.id, is_shared=False)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    r = client.get(f"/api/spaces/{space.id}", headers=responsable_headers)
    assert r.status_code == 404


def test_delete_space_with_feedback_cascade(client, responsable_headers, db_session):
    from app.models.user import User
    from app.models.message_feedback import MessageFeedback
    from sqlmodel import select

    # 1. Get the user matching the responsable headers
    user = db_session.exec(
        select(User).where(User.username.like("u_responsable_%"))
    ).first()
    assert user is not None

    # 2. Create a space owned by this user
    space = Space(name="Pytest cascade space", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    # 3. Create a message feedback inside this space
    feedback = MessageFeedback(
        message_id=None,
        user_id=user.id,
        space_id=space.id,
        is_positive=True,
        comment="Test de suppression en cascade",
        query_text="Requête",
        response_text="Réponse",
        chunk_ids=[]
    )
    db_session.add(feedback)
    db_session.commit()
    db_session.refresh(feedback)

    # 4. Call delete API to delete the space
    r = client.delete(f"/api/spaces/{space.id}", headers=responsable_headers)
    assert r.status_code == 204

    # 5. Verify both the space and feedback have been deleted
    db_session.expire_all()
    assert db_session.get(Space, space.id) is None
    assert db_session.get(MessageFeedback, feedback.id) is None

