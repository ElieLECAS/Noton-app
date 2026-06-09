"""Permissions RBAC : bibliothèque et espaces."""
from __future__ import annotations

from unittest import mock

from app.models.space import Space
from tests.conftest import create_test_user


def get_user_from_headers(headers, session):
    from app.services.auth_service import decode_token
    from app.models.user import User
    token = headers["Authorization"].split(" ")[1]
    payload = decode_token(token)
    assert payload is not None, "Failed to decode JWT token"
    return session.get(User, int(payload["sub"]))


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
    # Lecteur is now allowed to create private spaces (is_shared=False is the default).
    # However, they should NOT be allowed to create a shared space.
    r = client.post(
        "/api/spaces",
        headers=lecteur_headers,
        json={"name": "Espace test commun", "description": None, "is_shared": True},
    )
    assert r.status_code == 403

    # But they can create a private space
    r_ok = client.post(
        "/api/spaces",
        headers=lecteur_headers,
        json={"name": "Espace test privé", "description": None, "is_shared": False},
    )
    assert r_ok.status_code == 201
    assert r_ok.json()["is_shared"] is False


def test_create_space_ok_responsable(client, responsable_headers):
    # Responsable can create a private space
    r = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace pytest", "description": "d", "is_shared": False},
    )
    assert r.status_code == 201
    assert r.json()["name"] == "Espace pytest"
    assert r.json()["is_shared"] is False

    # Responsable cannot create a shared space
    r_fail = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace pytest commun", "description": "d", "is_shared": True},
    )
    assert r_fail.status_code == 403


def test_delete_space_restrictions(client, lecteur_headers, responsable_headers, admin_headers, db_session):
    from app.models.user import User
    from sqlmodel import select
    
    # Get user ids
    lecteur_user = get_user_from_headers(lecteur_headers, db_session)
    responsable_user = get_user_from_headers(responsable_headers, db_session)
    
    # 1. Create a shared space
    shared_space = Space(name="Shared Space", is_shared=True)
    db_session.add(shared_space)
    db_session.commit()
    db_session.refresh(shared_space)
    
    # Non-admin cannot delete shared space
    r = client.delete(f"/api/spaces/{shared_space.id}", headers=lecteur_headers)
    assert r.status_code == 403
    
    r = client.delete(f"/api/spaces/{shared_space.id}", headers=responsable_headers)
    assert r.status_code == 403
    
    # 2. Create private spaces for both
    l_space = Space(name="Lecteur Space", user_id=lecteur_user.id, is_shared=False)
    r_space = Space(name="Responsable Space", user_id=responsable_user.id, is_shared=False)
    db_session.add(l_space)
    db_session.add(r_space)
    db_session.commit()
    db_session.refresh(l_space)
    db_session.refresh(r_space)
    
    # Lecteur cannot delete Responsable's space (returns 404 since it's not visible/found to lecteur)
    r = client.delete(f"/api/spaces/{r_space.id}", headers=lecteur_headers)
    assert r.status_code == 404
    
    # Lecteur can delete their own space
    r = client.delete(f"/api/spaces/{l_space.id}", headers=lecteur_headers)
    assert r.status_code == 204
    
    # Responsable can delete their own space
    r = client.delete(f"/api/spaces/{r_space.id}", headers=responsable_headers)
    assert r.status_code == 204
    
    # Admin can delete shared space
    r = client.delete(f"/api/spaces/{shared_space.id}", headers=admin_headers)
    assert r.status_code == 204


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
    user = get_user_from_headers(responsable_headers, db_session)
    assert user is not None

    # 2. Create a space owned by this user
    space = Space(name="Pytest cascade space", user_id=user.id, is_shared=False)
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
    space_id = space.id
    feedback_id = feedback.id
    r = client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)
    assert r.status_code == 204

    # 5. Verify both the space and feedback have been deleted
    db_session.expire_all()
    assert db_session.get(Space, space_id) is None
    assert db_session.get(MessageFeedback, feedback_id) is None


def test_manage_document_spaces_permissions(client, lecteur_headers, responsable_headers, db_session):
    from app.models.user import User
    from app.models.document import Document
    from app.models.space import Space
    from app.models.document_space import DocumentSpace
    from app.services.library_service import get_or_create_user_library
    from sqlmodel import select

    # 1. Fetch user instances
    lecteur_user = get_user_from_headers(lecteur_headers, db_session)
    responsable_user = get_user_from_headers(responsable_headers, db_session)

    # 2. Get the library of the lecteur
    library = get_or_create_user_library(db_session, lecteur_user.id)

    # 3. Create a library document
    doc = Document(
        title="Document test",
        content="Notice technique",
        document_type="document",
        processing_status="completed",
        processing_progress=100,
        library_id=library.id,
        user_id=lecteur_user.id
    )
    db_session.add(doc)
    
    # 4. Create spaces
    # Private space owned by lecteur
    l_private_space = Space(name="Lecteur Private Space", user_id=lecteur_user.id, is_shared=False)
    # Private space owned by responsable
    r_private_space = Space(name="Responsable Private Space", user_id=responsable_user.id, is_shared=False)
    # Shared space (common)
    shared_space = Space(name="Shared Space", is_shared=True)
    
    db_session.add(l_private_space)
    db_session.add(r_private_space)
    db_session.add(shared_space)
    
    db_session.commit()
    db_session.refresh(doc)
    db_session.refresh(l_private_space)
    db_session.refresh(r_private_space)
    db_session.refresh(shared_space)

    # A. Lecteur should be able to link the document to their own private space
    r = client.post(
        f"/api/library/documents/{doc.id}/spaces",
        headers=lecteur_headers,
        json={"add_space_ids": [l_private_space.id], "remove_space_ids": []}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "queued"

    # B. Lecteur should NOT be able to link the document to a common space (since they lack library.write)
    r = client.post(
        f"/api/library/documents/{doc.id}/spaces",
        headers=lecteur_headers,
        json={"add_space_ids": [shared_space.id], "remove_space_ids": []}
    )
    assert r.status_code == 403

    # C. Lecteur should NOT be able to link the document to another user's private space
    r = client.post(
        f"/api/library/documents/{doc.id}/spaces",
        headers=lecteur_headers,
        json={"add_space_ids": [r_private_space.id], "remove_space_ids": []}
    )
    assert r.status_code == 403

    # D. Responsable (who has library.write) should be able to link to both their own private space and the shared space
    r = client.post(
        f"/api/library/documents/{doc.id}/spaces",
        headers=responsable_headers,
        json={"add_space_ids": [r_private_space.id, shared_space.id], "remove_space_ids": []}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "queued"

