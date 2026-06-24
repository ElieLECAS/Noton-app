"""Moteur de guidage procédural ("aiguillage" SAV / chantier).

Mocks : decide_guided_mode (classification), generate_routing_step (aiguillage LLM),
search_technical_passages (retrieval). On vérifie le contrat SSE {"step": ...},
la création/reprise de GuidedSession, la marche how-to, l'escalade, et la rétro-compat
(GUIDED_FLOW_ENABLED=False ⇒ pipeline one-shot inchangé).
"""
from __future__ import annotations

from unittest import mock

import pytest
from sqlmodel import Session, select

from app.database import engine
from app.models.conversation import Conversation
from app.models.guided_session import GuidedSession
from app.services.procedural_router_service import (
    EscalationRecap,
    RoutingChoice,
    RoutingStep,
)
from app.services.query_reasoning_service import GuidedModeDecision
from tests.conftest import extract_sse_events, extract_sse_message_text


@pytest.fixture
def space_and_conversation(client, responsable_headers):
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace guidage pytest"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]
    cr = client.post(
        "/api/conversations",
        headers=responsable_headers,
        json={"title": "Conv guidage", "space_id": space_id},
    )
    assert cr.status_code == 201
    conv_id = cr.json()["id"]
    yield space_id, conv_id
    client.delete(f"/api/conversations/{conv_id}", headers=responsable_headers)
    client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)


def _passage(step_text: str, page_no: int = 1):
    return {
        "passage": step_text,
        "passage_raw": step_text,
        "document_id": 1,
        "document_title": "Notice de pose LUMEAL GA",
        "chunk_id": 10,
        "chunk_index": 0,
        "page_no": page_no,
        "page_start": page_no,
        "page_end": page_no,
        "score": 0.9,
        "section": "Pose",
    }


def _retrieval_ok(passages=None):
    return mock.AsyncMock(
        return_value={
            "passages": passages if passages is not None else [_passage("Étape 1 : positionner l'embout.")],
            "status": "ok",
            "reason": None,
        }
    )


def _howto_step(message="Étape 1 : positionnez l'embout sur le profil.", terminal=False):
    if terminal:
        return RoutingStep(step_type="resolution", message=message, choices=[], is_terminal=True)
    return RoutingStep(
        step_type="instruction",
        message=message,
        choices=[
            RoutingChoice(label="C'est fait, étape suivante", value="next"),
            RoutingChoice(label="Je suis bloqué", value="stuck"),
        ],
        cited_pages=[{"document_title": "Notice de pose LUMEAL GA", "page_no": 1}],
        is_terminal=False,
    )


def _guided_post(client, headers, space_id, conv_id, message, guided_choice=None):
    body = {
        "message": message,
        "model": "mistral-small-latest",
        "provider": "mistral",
        "conversation_id": conv_id,
    }
    if guided_choice is not None:
        body["guided_choice"] = guided_choice
    return client.post(
        f"/api/spaces/{space_id}/chat/stream",
        headers=headers,
        json=body,
    )


def _get_conversation(conv_id: int) -> Conversation:
    with Session(engine) as s:
        return s.get(Conversation, conv_id)


def _get_session_for_conv(conv_id: int) -> GuidedSession | None:
    with Session(engine) as s:
        return s.exec(
            select(GuidedSession).where(GuidedSession.conversation_id == conv_id)
        ).first()


def test_guided_first_step_creates_session(
    client, responsable_headers, space_and_conversation
):
    space_id, conv_id = space_and_conversation
    with mock.patch("app.config.settings.GUIDED_FLOW_ENABLED", True), mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(
            return_value=GuidedModeDecision(
                is_guided=True, flow_kind="howto", topic="pose embout profil alu"
            )
        ),
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=_retrieval_ok(),
    ), mock.patch(
        "app.services.guided_flow_service.generate_routing_step",
        new=mock.AsyncMock(return_value=_howto_step()),
    ):
        r = _guided_post(
            client, responsable_headers, space_id, conv_id,
            "comment monter l'embout sur le profil alu ?",
        )

    assert r.status_code == 200
    events = extract_sse_events(r.text)
    step_events = [e["step"] for e in events if "step" in e]
    assert len(step_events) == 1
    step = step_events[0]
    assert step["step_type"] == "instruction"
    assert step["is_terminal"] is False
    assert len(step["choices"]) == 2
    assert step["guided_session_id"]
    # Le message de l'étape a bien été streamé en chunks
    assert "positionnez" in extract_sse_message_text(r.text).lower()
    assert any(e.get("done") for e in events)

    # GuidedSession créée + pointeur dans query_context
    gs = _get_session_for_conv(conv_id)
    assert gs is not None
    assert gs.status == "active"
    assert gs.flow_kind == "howto"
    assert len(gs.path) == 1
    conv = _get_conversation(conv_id)
    assert conv.query_context["guided"]["active_session_id"] == gs.id
    assert conv.query_context["guided"]["phase"] == "guided_active"

    # Message assistant persisté avec metadata_json.guided_step
    msgs = client.get(
        f"/api/conversations/{conv_id}/messages", headers=responsable_headers
    ).json()
    assistant = [m for m in msgs if m["role"] == "assistant"][-1]
    assert assistant["metadata_json"]["guided_step"]["step_type"] == "instruction"


def test_guided_resume_advances_step(
    client, responsable_headers, space_and_conversation
):
    space_id, conv_id = space_and_conversation
    gen_mock = mock.AsyncMock(
        side_effect=[
            _howto_step("Étape 1 : positionnez l'embout."),
            _howto_step("Étape 2 : clippez l'embout."),
        ]
    )
    with mock.patch("app.config.settings.GUIDED_FLOW_ENABLED", True), mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(
            return_value=GuidedModeDecision(is_guided=True, flow_kind="howto", topic="pose embout")
        ),
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=_retrieval_ok(),
    ), mock.patch(
        "app.services.guided_flow_service.generate_routing_step",
        new=gen_mock,
    ):
        # Tour 1 : démarrage
        r1 = _guided_post(
            client, responsable_headers, space_id, conv_id, "comment poser l'embout ?"
        )
        assert r1.status_code == 200
        # Tour 2 : l'utilisateur clique « étape suivante » (reprise, pas de reclassement)
        r2 = _guided_post(
            client, responsable_headers, space_id, conv_id, "C'est fait, étape suivante",
            guided_choice={"value": "next", "label": "C'est fait, étape suivante"},
        )
        assert r2.status_code == 200

    step2 = [e["step"] for e in extract_sse_events(r2.text) if "step" in e][0]
    assert step2["step_index"] == 1

    gs = _get_session_for_conv(conv_id)
    assert len(gs.path) == 2
    # La réponse de l'utilisateur a été enregistrée dans la 1re étape
    assert gs.path[0]["user_selection"]["value"] == "next"
    assert "C'est fait, étape suivante" in (gs.path[0]["observations"] or [])


def test_guided_resolution_clears_pointer(
    client, responsable_headers, space_and_conversation
):
    space_id, conv_id = space_and_conversation
    with mock.patch("app.config.settings.GUIDED_FLOW_ENABLED", True), mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(
            return_value=GuidedModeDecision(is_guided=True, flow_kind="howto", topic="pose")
        ),
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=_retrieval_ok(),
    ), mock.patch(
        "app.services.guided_flow_service.generate_routing_step",
        new=mock.AsyncMock(return_value=_howto_step("Pose terminée.", terminal=True)),
    ):
        r = _guided_post(client, responsable_headers, space_id, conv_id, "comment poser ?")

    assert r.status_code == 200
    step = [e["step"] for e in extract_sse_events(r.text) if "step" in e][0]
    assert step["is_terminal"] is True
    assert step["choices"] == []

    gs = _get_session_for_conv(conv_id)
    assert gs.status == "resolved"
    conv = _get_conversation(conv_id)
    assert "guided" not in (conv.query_context or {})


def test_guided_escalation_builds_recap(
    client, responsable_headers, space_and_conversation
):
    space_id, conv_id = space_and_conversation
    escalation = RoutingStep(
        step_type="escalation",
        message="Je transmets votre demande au SAV.",
        choices=[],
        is_terminal=True,
        escalation_recap=EscalationRecap(summary="Volet bloqué non résolu."),
    )
    with mock.patch("app.config.settings.GUIDED_FLOW_ENABLED", True), mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(
            return_value=GuidedModeDecision(
                is_guided=True, flow_kind="diagnostic", topic="volet roulant bloqué"
            )
        ),
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=_retrieval_ok([_passage("Conditions de garantie : 2 ans.")]),
    ), mock.patch(
        "app.services.guided_flow_service.generate_routing_step",
        new=mock.AsyncMock(return_value=escalation),
    ):
        r = _guided_post(
            client, responsable_headers, space_id, conv_id, "mon volet roulant ne fonctionne plus"
        )

    assert r.status_code == 200
    step = [e["step"] for e in extract_sse_events(r.text) if "step" in e][0]
    assert step["step_type"] == "escalation"
    assert step["is_terminal"] is True
    recap = step["escalation_recap"]
    assert recap is not None
    assert "contact" in recap
    assert recap["warranty_excerpt"]  # extrait garantie récupéré

    gs = _get_session_for_conv(conv_id)
    assert gs.status == "escalated"


def test_guided_classifier_non_guided_falls_through(
    client, responsable_headers, space_and_conversation
):
    """GUIDED_FLOW_ENABLED=True mais demande factuelle ⇒ pipeline RAG standard."""
    space_id, conv_id = space_and_conversation
    with mock.patch("app.config.settings.GUIDED_FLOW_ENABLED", True), mock.patch(
        "app.config.settings.QUERY_UNDERSTANDING_ENABLED", False
    ), mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(return_value=GuidedModeDecision(is_guided=False)),
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=mock.AsyncMock(
            return_value={"passages": [], "status": "ok", "reason": "no_results"}
        ),
    ):
        r = _guided_post(
            client, responsable_headers, space_id, conv_id,
            "quelle est la tolérance de pose ?",
        )

    assert r.status_code == 200
    # Aucun aiguillage : pas d'événement step, fallback "aucune source pertinente"
    assert not any("step" in e for e in extract_sse_events(r.text))
    assert "seuil minimum de 75%" in extract_sse_message_text(r.text)
    assert _get_session_for_conv(conv_id) is None


def test_guided_disabled_no_session(
    client, responsable_headers, space_and_conversation
):
    """Rétro-compat : flag désactivé ⇒ aucune classification guidée, pipeline inchangé."""
    space_id, conv_id = space_and_conversation
    guided_mode = mock.AsyncMock(return_value=GuidedModeDecision(is_guided=True))
    with mock.patch(
        "app.config.settings.QUERY_UNDERSTANDING_ENABLED", False
    ), mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode", new=guided_mode
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=mock.AsyncMock(
            return_value={"passages": [], "status": "ok", "reason": "no_results"}
        ),
    ):
        r = _guided_post(
            client, responsable_headers, space_id, conv_id,
            "comment monter l'embout sur le profil alu ?",
        )

    assert r.status_code == 200
    guided_mode.assert_not_called()  # jamais classé en mode guidé
    assert _get_session_for_conv(conv_id) is None
