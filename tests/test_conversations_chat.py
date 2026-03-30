"""Conversations / messages et chat espace (mocks Mistral / recherche)."""
from __future__ import annotations

import json
from unittest import mock

import pytest
from sqlmodel import select

from app.models.user_role import UserRole


@pytest.fixture
def space_and_conversation(client, responsable_headers):
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace conv pytest"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]
    cr = client.post(
        "/api/conversations",
        headers=responsable_headers,
        json={"title": "Ma conv", "space_id": space_id},
    )
    assert cr.status_code == 201
    conv_id = cr.json()["id"]
    yield space_id, conv_id
    client.delete(f"/api/conversations/{conv_id}", headers=responsable_headers)
    client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)


def test_create_conversation_unknown_space_404(client, responsable_headers):
    r = client.post(
        "/api/conversations",
        headers=responsable_headers,
        json={"title": "X", "space_id": 999_999_999},
    )
    assert r.status_code == 404


def test_conversation_crud_and_messages(client, responsable_headers, space_and_conversation):
    space_id, conv_id = space_and_conversation

    r = client.patch(
        f"/api/conversations/{conv_id}",
        headers=responsable_headers,
        json={"title": "Titre mis à jour"},
    )
    assert r.status_code == 200
    assert r.json()["title"] == "Titre mis à jour"

    r = client.post(
        f"/api/conversations/{conv_id}/messages",
        headers=responsable_headers,
        json={
            "conversation_id": conv_id,
            "role": "user",
            "content": "Bonjour test",
        },
    )
    assert r.status_code == 200
    msg_id = r.json()["id"]

    r = client.get(f"/api/conversations/{conv_id}/messages", headers=responsable_headers)
    assert r.status_code == 200
    assert any(m["id"] == msg_id for m in r.json())


def test_conversation_other_user_404(client, responsable_headers, db_session):
    from app.models.space import Space
    from tests.conftest import bearer_headers, create_test_user

    user_b = create_test_user(db_session, "responsable")
    space = Space(name="Iso B", user_id=user_b.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    from app.models.conversation import Conversation

    conv = Conversation(title="Secrète", user_id=user_b.id, space_id=space.id)
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)

    r = client.get(
        f"/api/conversations/{conv.id}",
        headers=responsable_headers,
    )
    assert r.status_code == 404

    db_session.delete(conv)
    db_session.delete(space)
    for ur in db_session.exec(select(UserRole).where(UserRole.user_id == user_b.id)).all():
        db_session.delete(ur)
    db_session.delete(user_b)
    db_session.commit()


async def _fake_mistral_stream(*args, **kwargs):
    yield json.dumps({"message": {"content": "chunk"}})


def test_space_chat_stream_empty_space_no_error(client, responsable_headers):
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace vide chat"},
    )
    space_id = sp.json()["id"]
    try:
        with mock.patch(
            "app.routers.chat.mistral_chat_stream",
            _fake_mistral_stream,
        ):
            r = client.post(
                f"/api/spaces/{space_id}/chat/stream",
                headers=responsable_headers,
                json={
                    "message": "Question sans doc",
                    "model": "mistral-small-latest",
                    "provider": "mistral",
                    "conversation_id": None,
                },
            )
        assert r.status_code == 200
        assert "done" in r.text.lower()
    finally:
        client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)


def test_space_chat_stream_with_conversation_persists(
    client, responsable_headers, space_and_conversation
):
    space_id, conv_id = space_and_conversation
    with mock.patch(
        "app.routers.chat.mistral_chat_stream",
        _fake_mistral_stream,
    ):
        r = client.post(
            f"/api/spaces/{space_id}/chat/stream",
            headers=responsable_headers,
            json={
                "message": "Hello",
                "model": "mistral-small-latest",
                "provider": "mistral",
                "conversation_id": conv_id,
            },
        )
    assert r.status_code == 200

    msgs = client.get(
        f"/api/conversations/{conv_id}/messages",
        headers=responsable_headers,
    )
    assert msgs.status_code == 200
    roles = [m["role"] for m in msgs.json()]
    assert "user" in roles
