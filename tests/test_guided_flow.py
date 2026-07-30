"""Arbre SAV — runtime déterministe (refonte 2026-07-30).

Le guidé ne démarre plus JAMAIS depuis une classification LLM d'un message libre :
- un message libre reçoit une réponse RAG (non-régression « softclose ») ;
- un parcours démarre par /sav/start (bouton/picker/chip), traverse le SNAPSHOT publié
  (zéro retrieval par étape), gère ← Précédent / Quitter / feedback de feuille.

Les embeddings de l'index d'entrée sont mockés (aucun appel API en test).
"""
from __future__ import annotations

import json
from unittest import mock

import pytest
from sqlmodel import Session, select

from app.database import engine
from app.models.conversation import Conversation
from app.models.guided_gap import GuidedGap
from app.models.guided_session import GuidedSession
from tests.conftest import extract_sse_events


@pytest.fixture(autouse=True)
def _no_llm_wording():
    """Le TEXTE des tours est rédigé par LIA (compose_step_message / compose_leaf_answer).

    Ces tests valident la STRUCTURE du parcours, pas la formulation : on remplace la
    rédaction par la description du SAV — sinon chaque tour partirait en appel réseau et
    les assertions dépendraient d'un texte non déterministe.
    """
    def _text(session, **kw):
        node = kw.get("node") or {}
        return str(node.get("message") or node.get("title") or "")

    # compose_step_message renvoie {message, labels} (les libellés des boutons sont eux
    # aussi reformulés par LIA) ; on garde les libellés du SAV pour que les assertions
    # portent sur les valeurs du parcours et pas sur une formulation générée.
    def _turn(session, **kw):
        return {"message": _text(session, **kw), "labels": {}}

    with mock.patch("app.services.guided_flow_service.compose_step_message", side_effect=_turn), \
         mock.patch("app.services.guided_flow_service.compose_leaf_answer", side_effect=_text):
        yield


@pytest.fixture
def space_and_conversation(client, responsable_headers):
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace arbre SAV pytest"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]
    cr = client.post(
        "/api/conversations",
        headers=responsable_headers,
        json={"title": "Conv arbre SAV", "space_id": space_id},
    )
    assert cr.status_code == 201
    conv_id = cr.json()["id"]
    yield space_id, conv_id
    client.delete(f"/api/conversations/{conv_id}", headers=responsable_headers)
    client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)


def _me_id(client, headers) -> int:
    r = client.get("/api/auth/me", headers=headers)
    assert r.status_code == 200
    return r.json()["id"]


def _tree_payload():
    """Arbre : root → (Complètement → moteur ? → [Oui → diagnostic | Non → escalade])
    | (Partiellement → diagnostic)."""
    return {
        "meta": {"root_node_key": "root"},
        "nodes": [
            {
                "node_key": "root",
                "step_type": "question",
                "title": "Type de blocage",
                "message": "Le volet est-il bloqué complètement ou partiellement ?",
                "choices": [
                    {"label": "Complètement bloqué", "value": "total", "next_node_key": "moteur"},
                    {"label": "Partiellement", "value": "partiel", "next_node_key": "diag_butee"},
                ],
                "attachments": [],
            },
            {
                "node_key": "moteur",
                "step_type": "question",
                "title": "Bruit moteur",
                "message": "Le moteur fait-il du bruit quand vous commandez le volet ?",
                "choices": [
                    {"label": "Oui, il grogne", "value": "bruit", "next_node_key": "diag_condensateur"},
                    {"label": "Non, aucun bruit", "value": "silence", "next_node_key": "escalade"},
                ],
                "attachments": [],
            },
            {
                "node_key": "diag_condensateur",
                "step_type": "diagnostic",
                "title": "Condensateur",
                "message": "Le condensateur du moteur est probablement HS — remplacement nécessaire.",
                "is_terminal": True,
                "termination_type": "resolution",
                "choices": [],
                "attachments": [],
            },
            {
                "node_key": "diag_butee",
                "step_type": "diagnostic",
                "title": "Butée haute",
                "message": "Régler la butée haute selon la notice.",
                "is_terminal": True,
                "termination_type": "resolution",
                "choices": [],
                "attachments": [],
            },
            {
                "node_key": "escalade",
                "step_type": "escalation",
                "title": "SAV",
                "message": "Moteur muet : intervention SAV nécessaire.",
                "is_terminal": True,
                "termination_type": "escalation",
                "choices": [],
                "attachments": [],
            },
        ],
    }


def _publish_tree(space_id: int, user_id: int) -> str:
    """Crée + publie l'arbre via la couche service (embeddings mockés). Retourne le slug."""
    from app.services.guided_authoring_service import create_tree, publish_tree, save_tree_draft

    with mock.patch(
        "app.services.embedding_service.generate_embeddings_batch",
        side_effect=lambda texts, **kw: [None] * len(texts),
    ):
        with Session(engine) as s:
            tree = create_tree(
                s,
                title="Volet roulant bloqué",
                entry_symptom="blocage_manoeuvre",
                space_id=space_id,
                user_id=user_id,
            )
            save_tree_draft(s, tree.id, _tree_payload(), user_id)
            result = publish_tree(s, tree.id, note="test", user_id=user_id)
            assert result["version"] == 1
            return tree.slug


def _start(client, headers, space_id, conv_id, slug):
    return client.post(
        f"/api/spaces/{space_id}/sav/start",
        headers=headers,
        json={"conversation_id": conv_id, "tree_slug": slug},
    )


def _choice(client, headers, space_id, conv_id, value, label=""):
    return client.post(
        f"/api/spaces/{space_id}/chat/stream",
        headers=headers,
        json={
            "message": label or value,
            "model": "mistral-small-latest",
            "provider": "mistral",
            "conversation_id": conv_id,
            "guided_choice": {"value": value, "label": label or value},
        },
    )


def _step_of(response) -> dict:
    events = extract_sse_events(response.text)
    steps = [e["step"] for e in events if "step" in e]
    assert steps, f"Aucun événement step dans : {response.text[:400]}"
    return steps[-1]


def test_full_walk_resolution_with_feedback(client, responsable_headers, space_and_conversation):
    space_id, conv_id = space_and_conversation
    slug = _publish_tree(space_id, _me_id(client, responsable_headers))

    # Le picker liste l'arbre publié.
    entries = client.get(f"/api/spaces/{space_id}/sav/entries", headers=responsable_headers)
    assert entries.status_code == 200
    assert any(e["tree_slug"] == slug for e in entries.json()["entries"])

    # Démarrage explicite → racine.
    r = _start(client, responsable_headers, space_id, conv_id, slug)
    assert r.status_code == 200
    step = _step_of(r)
    assert "bloqué complètement" in step["message"]
    assert {c["value"] for c in step["choices"]} == {"total", "partiel"}
    assert step["is_terminal"] is False

    # Clic « Complètement » → question moteur.
    step = _step_of(_choice(client, responsable_headers, space_id, conv_id, "total", "Complètement bloqué"))
    assert "moteur" in step["message"].lower()
    assert step["can_go_back"] is True
    assert step["breadcrumb"] == ["Complètement bloqué"]

    # Clic « Oui, il grogne » → feuille diagnostic, feedback attendu (pas terminal).
    step = _step_of(_choice(client, responsable_headers, space_id, conv_id, "bruit", "Oui, il grogne"))
    assert "condensateur" in step["message"].lower()
    assert step["is_terminal"] is False
    assert {c["value"] for c in step["choices"]} == {"__feedback_yes", "__feedback_no"}

    # Feedback « résolu » → terminal, session close, pointeur nettoyé.
    step = _step_of(_choice(client, responsable_headers, space_id, conv_id, "__feedback_yes", "Oui, résolu"))
    assert step["is_terminal"] is True

    with Session(engine) as s:
        gsession = s.exec(
            select(GuidedSession).where(GuidedSession.conversation_id == conv_id)
        ).first()
        assert gsession.status == "resolved"
        assert gsession.resolved_feedback is True
        assert gsession.tree_version == 1
        conv = s.get(Conversation, conv_id)
        assert "guided" not in (conv.query_context or {})


def test_back_button_returns_to_previous_node(client, responsable_headers, space_and_conversation):
    space_id, conv_id = space_and_conversation
    slug = _publish_tree(space_id, _me_id(client, responsable_headers))

    _start(client, responsable_headers, space_id, conv_id, slug)
    step = _step_of(_choice(client, responsable_headers, space_id, conv_id, "total", "Complètement bloqué"))
    assert "moteur" in step["message"].lower()

    step = _step_of(_choice(client, responsable_headers, space_id, conv_id, "__back", "← Précédent"))
    assert "bloqué complètement" in step["message"]
    assert step["node_key"] == "root"


def test_quit_closes_session_and_clears_pointer(client, responsable_headers, space_and_conversation):
    space_id, conv_id = space_and_conversation
    slug = _publish_tree(space_id, _me_id(client, responsable_headers))

    _start(client, responsable_headers, space_id, conv_id, slug)
    step = _step_of(_choice(client, responsable_headers, space_id, conv_id, "__quit", "Quitter"))
    assert step["is_terminal"] is True

    with Session(engine) as s:
        gsession = s.exec(
            select(GuidedSession).where(GuidedSession.conversation_id == conv_id)
        ).first()
        assert gsession.status == "abandoned"
        conv = s.get(Conversation, conv_id)
        assert "guided" not in (conv.query_context or {})


def test_feedback_no_escalates_with_recap(client, responsable_headers, space_and_conversation):
    space_id, conv_id = space_and_conversation
    slug = _publish_tree(space_id, _me_id(client, responsable_headers))

    _start(client, responsable_headers, space_id, conv_id, slug)
    _choice(client, responsable_headers, space_id, conv_id, "partiel", "Partiellement")
    step = _step_of(_choice(client, responsable_headers, space_id, conv_id, "__feedback_no", "Non résolu"))
    assert step["is_terminal"] is True
    assert step["step_type"] == "escalation"
    recap = step["escalation_recap"]
    assert recap["contact"]
    assert any("Partiellement" in str(qa.get("answer", "")) for qa in recap.get("qa_path", []))

    with Session(engine) as s:
        gsession = s.exec(
            select(GuidedSession).where(GuidedSession.conversation_id == conv_id)
        ).first()
        assert gsession.status == "escalated"
        assert gsession.resolved_feedback is False


def test_free_text_mapped_to_choice(client, responsable_headers, space_and_conversation):
    space_id, conv_id = space_and_conversation
    slug = _publish_tree(space_id, _me_id(client, responsable_headers))
    _start(client, responsable_headers, space_id, conv_id, slug)

    with mock.patch(
        "app.services.guided_flow_service.map_free_text_to_choice", return_value="total"
    ):
        r = client.post(
            f"/api/spaces/{space_id}/chat/stream",
            headers=responsable_headers,
            json={
                "message": "il est bloqué à fond, impossible de le descendre",
                "model": "mistral-small-latest",
                "provider": "mistral",
                "conversation_id": conv_id,
            },
        )
    step = _step_of(r)
    assert "moteur" in step["message"].lower()  # a avancé comme le choix « total »


def test_free_text_unmapped_represents_node(client, responsable_headers, space_and_conversation):
    space_id, conv_id = space_and_conversation
    slug = _publish_tree(space_id, _me_id(client, responsable_headers))
    _start(client, responsable_headers, space_id, conv_id, slug)

    with mock.patch(
        "app.services.guided_flow_service.map_free_text_to_choice", return_value=None
    ):
        r = client.post(
            f"/api/spaces/{space_id}/chat/stream",
            headers=responsable_headers,
            json={
                "message": "je ne sais pas trop",
                "model": "mistral-small-latest",
                "provider": "mistral",
                "conversation_id": conv_id,
            },
        )
    step = _step_of(r)
    assert step["node_key"] == "root"  # re-présente le nœud courant
    assert "bloqué complètement" in step["message"]


def test_start_unknown_tree_is_graceful(client, responsable_headers, space_and_conversation):
    space_id, conv_id = space_and_conversation
    r = _start(client, responsable_headers, space_id, conv_id, "arbre_inexistant")
    assert r.status_code == 200
    step = _step_of(r)
    assert step["is_terminal"] is True
    with Session(engine) as s:
        conv = s.get(Conversation, conv_id)
        assert "guided" not in (conv.query_context or {})


async def _fake_mistral_stream(*args, **kwargs):
    yield json.dumps({"message": {"content": "La fonctionnalité est softclose et softopen."}})


def test_free_message_never_hijacked(client, responsable_headers, space_and_conversation):
    """Non-régression « softclose » : un message libre reçoit TOUJOURS la réponse RAG,
    même à connotation produit — plus aucun départ guidé par classification LLM."""
    space_id, conv_id = space_and_conversation
    _publish_tree(space_id, _me_id(client, responsable_headers))  # arbre publié présent

    with mock.patch("app.config.settings.QUERY_UNDERSTANDING_ENABLED", False), mock.patch(
        "app.routers.chat.mistral_chat_stream", _fake_mistral_stream
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=mock.AsyncMock(
            return_value={
                "passages": [
                    {
                        "passage": "softclose", "passage_raw": "softclose",
                        "document_id": 1, "document_title": "Doc", "chunk_id": 1,
                        "chunk_index": 0, "page_no": 1, "page_start": 1, "page_end": 1,
                        "score": 0.9, "section": "",
                    }
                ],
                "status": "ok",
                "reason": None,
            }
        ),
    ):
        r = client.post(
            f"/api/spaces/{space_id}/chat/stream",
            headers=responsable_headers,
            json={
                "message": "je cherche une fonction qui ralentit le vantail de mon coulissant PVC",
                "model": "mistral-small-latest",
                "provider": "mistral",
                "conversation_id": conv_id,
            },
        )
    assert r.status_code == 200
    events = extract_sse_events(r.text)
    assert not any("step" in e for e in events), "le tour a été détourné en mode guidé"
    assert "softclose" in r.text
    with Session(engine) as s:
        gsession = s.exec(
            select(GuidedSession).where(GuidedSession.conversation_id == conv_id)
        ).first()
        assert gsession is None or gsession.status != "active"


def test_gap_recorded_when_no_tree_matches(client, responsable_headers, space_and_conversation):
    """Symptôme détecté sans arbre publié → GuidedGap (backlog), réponse RAG inchangée."""
    space_id, conv_id = space_and_conversation  # AUCUN arbre publié ici

    from app.services.lightweight_query_understanding import LightweightQueryResult
    from app.services.query_signals_schemas import LightweightQuerySignals

    lw = LightweightQueryResult(
        route="rag",
        ready_for_retrieval=True,
        signals=LightweightQuerySignals(
            intent="troubleshooting",
            detected_symptom="infiltration_eau",
        ),
    )
    with mock.patch("app.config.settings.QUERY_UNDERSTANDING_ENABLED", True), mock.patch(
        "app.services.lightweight_query_understanding.run_lightweight_understanding",
        new=mock.AsyncMock(return_value=lw),
    ), mock.patch(
        "app.routers.chat.mistral_chat_stream", _fake_mistral_stream
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=mock.AsyncMock(return_value={"passages": [], "status": "ok", "reason": "no_results"}),
    ), mock.patch(
        "app.services.embedding_service.generate_embedding", return_value=None
    ):
        r = client.post(
            f"/api/spaces/{space_id}/chat/stream",
            headers=responsable_headers,
            json={
                "message": "j'ai de l'eau qui rentre par la fenêtre",
                "model": "mistral-small-latest",
                "provider": "mistral",
                "conversation_id": conv_id,
            },
        )
    assert r.status_code == 200
    with Session(engine) as s:
        gap = s.exec(
            select(GuidedGap).where(
                GuidedGap.space_id == space_id,
                GuidedGap.detected_symptom == "infiltration_eau",
            )
        ).first()
        assert gap is not None
        assert gap.count >= 1
        s.delete(gap)
        s.commit()
