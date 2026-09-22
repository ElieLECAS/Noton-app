"""Conversations, messages, historique et tour de chat (Mistral simulé)."""
from __future__ import annotations

import json
from unittest import mock

import pytest
from sqlmodel import select

from app.config import settings
from app.models.message import Message
from app.models.user_role import UserRole
from tests.conftest import extract_sse_events, extract_sse_message_text


def _seed_message(db_session, conversation_id: int, role: str, content: str) -> int:
    """Insère un message directement : seul le tour de chat en écrit par l'API."""
    msg = Message(conversation_id=conversation_id, role=role, content=content)
    db_session.add(msg)
    db_session.commit()
    db_session.refresh(msg)
    return msg.id


@pytest.fixture
def conversation(client, responsable_headers):
    cr = client.post("/api/conversations", headers=responsable_headers, json={"title": "Ma conv"})
    assert cr.status_code == 201
    conv_id = cr.json()["id"]
    yield conv_id
    client.delete(f"/api/conversations/{conv_id}", headers=responsable_headers)


def test_conversation_crud_and_messages(client, responsable_headers, conversation, db_session):
    r = client.patch(f"/api/conversations/{conversation}", headers=responsable_headers, json={"title": "Titre mis à jour"})
    assert r.status_code == 200 and r.json()["title"] == "Titre mis à jour"

    msg_id = _seed_message(db_session, conversation, "user", "Bonjour test")
    r = client.get(f"/api/conversations/{conversation}/messages", headers=responsable_headers)
    assert r.status_code == 200 and any(m["id"] == msg_id for m in r.json())

    # Seul le tour de chat écrit des messages : pas d'écriture par l'API des conversations.
    r = client.post(
        f"/api/conversations/{conversation}/messages",
        headers=responsable_headers,
        json={"conversation_id": conversation, "role": "assistant", "content": "faux"},
    )
    assert r.status_code == 405

    r = client.get("/api/conversations", headers=responsable_headers)
    assert r.status_code == 200
    mine = next(c for c in r.json() if c["id"] == conversation)
    assert mine["message_count"] == 1


def test_conversation_other_user_404(client, responsable_headers, db_session):
    from app.models.conversation import Conversation
    from tests.conftest import create_test_user

    user_b = create_test_user(db_session, "responsable")
    conv = Conversation(title="Secrète", user_id=user_b.id)
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)
    try:
        assert client.get(f"/api/conversations/{conv.id}", headers=responsable_headers).status_code == 404
        r = client.post(
            "/api/chat/stream", headers=responsable_headers,
            json={"message": "Question", "conversation_id": conv.id},
        )
        assert r.status_code == 404
    finally:
        db_session.delete(conv)
        for ur in db_session.exec(select(UserRole).where(UserRole.user_id == user_b.id)).all():
            db_session.delete(ur)
        db_session.delete(user_b)
        db_session.commit()


def _appel_outil(nom, arguments, identifiant="c1"):
    return {"id": identifiant, "type": "function",
            "function": {"name": nom, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def _fabrique_stream(reponse, contextes=None):
    """Un faux flux qui cherche d'abord, puis répond — le chemin normal d'un tour.

    Sans le premier appel d'outil, le serveur renverrait le modèle lire (aucune page chargée),
    ce qui ajouterait un aller-retour à chaque test.
    """
    appels = {"n": 0}

    async def fake(message, **kwargs):
        appels["n"] += 1
        if contextes is not None:
            contextes.append([dict(m) for m in kwargs["context"]])
        if appels["n"] % 2 == 1:
            yield json.dumps({"tool_calls": [_appel_outil("chercher", {"mots_cles": "parclose 76507"})]})
            return
        for evenement in reponse:
            yield json.dumps(evenement)

    return fake


_fake_stream = _fabrique_stream([
    {"thinking": "je lis"},
    {"message": {"content": "La parclose 76507 "}},
    {"message": {"content": "(/profiles/perform76-parcloses.md)."}},
    {"usage": {"prompt_tokens": 185508, "completion_tokens": 12,
               "prompt_tokens_details": {"cached_tokens": 185472}}},
])


def test_chat_stream_persists_question_and_answer(client, responsable_headers, conversation):
    with mock.patch("app.services.wiki_chat_service.chat_stream", _fake_stream):
        r = client.post(
            "/api/chat/stream", headers=responsable_headers,
            json={"message": "Quelle parclose pour 44 mm ?", "conversation_id": conversation},
        )
    assert r.status_code == 200
    events = extract_sse_events(r.text)
    kinds = [next(iter(e)) for e in events]
    assert kinds == ["etape", "thinking", "message", "message", "sources", "done"]
    assert extract_sse_message_text(r.text) == "La parclose 76507 (/profiles/perform76-parcloses.md)."
    done = events[-1]
    assert done["message_id"] and done["trace"]["cited_pages"] == ["/profiles/perform76-parcloses.md"]
    assert done["trace"]["cached_tokens"] == 185472

    msgs = client.get(f"/api/conversations/{conversation}/messages", headers=responsable_headers).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "Quelle parclose pour 44 mm ?"
    assistant = msgs[1]
    assert assistant["id"] == done["message_id"]
    assert assistant["model"] == settings.MODEL_FAST
    sources = json.loads(assistant["sources"])
    assert sources[0]["path"] == "/profiles/perform76-parcloses.md" and sources[0]["exists"] is True
    assert assistant["metadata_json"]["trace"]["prompt_tokens"] == 185508


def test_chat_stream_sends_history_after_system_prompt(client, responsable_headers, conversation):
    contextes = []
    spy = _fabrique_stream([{"message": {"content": "ok"}}], contextes)

    with mock.patch("app.services.wiki_chat_service.chat_stream", spy):
        client.post("/api/chat/stream", headers=responsable_headers, json={"message": "Première", "conversation_id": conversation})
        client.post("/api/chat/stream", headers=responsable_headers, json={"message": "Seconde", "conversation_id": conversation})
    # Deux appels par tour (recherche puis réponse) : le premier contexte du second tour porte
    # l'historique, et seulement lui — les messages d'outils n'y sont pas persistés.
    ctx = contextes[2]
    assert ctx[0]["role"] == "system"
    assert [(m["role"], m["content"]) for m in ctx[1:]] == [
        ("user", "Première"), ("assistant", "ok"), ("user", "Seconde"),
    ]


def test_chat_stream_error_persists_nothing_but_the_question(client, responsable_headers, conversation):
    async def boom(message, **kwargs):
        raise RuntimeError("Mistral injoignable")
        yield  # pragma: no cover

    with mock.patch("app.services.wiki_chat_service.chat_stream", boom):
        r = client.post("/api/chat/stream", headers=responsable_headers, json={"message": "Hello", "conversation_id": conversation})
    assert r.status_code == 200
    events = extract_sse_events(r.text)
    assert any("error" in e for e in events) and not any("done" in e for e in events)
    msgs = client.get(f"/api/conversations/{conversation}/messages", headers=responsable_headers).json()
    assert [m["role"] for m in msgs] == ["user"]


def test_chat_stream_rejects_empty_message(client, responsable_headers, conversation):
    r = client.post("/api/chat/stream", headers=responsable_headers, json={"message": "   ", "conversation_id": conversation})
    assert r.status_code == 422


def test_load_history_is_bounded(client, responsable_headers, conversation, db_session, monkeypatch):
    from app.services.wiki_chat_service import load_history

    for i in range(6):
        _seed_message(db_session, conversation, "user" if i % 2 == 0 else "assistant", f"message {i}")
    monkeypatch.setattr(settings, "CHAT_HISTORY_MAX_MESSAGES", 3)
    history = load_history(db_session, conversation)
    # 3 derniers = assistant 3, user 4, assistant 5 → l'assistant de tête est retiré.
    assert [(m["role"], m["content"]) for m in history] == [("user", "message 4"), ("assistant", "message 5")]

    monkeypatch.setattr(settings, "CHAT_HISTORY_MAX_MESSAGES", 10)
    monkeypatch.setattr(settings, "CHAT_HISTORY_MAX_CHARS", 20)
    history = load_history(db_session, conversation)
    assert sum(len(m["content"]) for m in history) <= 20
    assert not history or history[0]["role"] == "user"


def test_feedback_on_assistant_message(client, responsable_headers, conversation):
    with mock.patch("app.services.wiki_chat_service.chat_stream", _fake_stream):
        r = client.post("/api/chat/stream", headers=responsable_headers, json={"message": "Q", "conversation_id": conversation})
    message_id = extract_sse_events(r.text)[-1]["message_id"]

    r = client.post(f"/api/conversations/messages/{message_id}/feedback", headers=responsable_headers, json={"is_positive": False})
    assert r.status_code == 422  # commentaire obligatoire pour un retour négatif
    r = client.post(
        f"/api/conversations/messages/{message_id}/feedback", headers=responsable_headers,
        json={"is_positive": False, "comment": "Mauvaise famille", "category": "Hallucination"},
    )
    assert r.status_code == 201
    fb = r.json()
    assert fb["query_text"] == "Q" and fb["category"] == "Hallucination"

    r = client.get("/api/conversations/feedbacks/mine", headers=responsable_headers)
    assert r.status_code == 200
    item = next(i for i in r.json()["items"] if i["message_id"] == message_id)
    assert item["conversation_id"] == conversation
