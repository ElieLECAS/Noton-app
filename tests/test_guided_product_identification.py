"""Étape 0 d'identification produit du mode guidé (refonte anti-dérive INNOSLIDE).

Quand l'utilisateur n'a PAS nommé le produit (product_named=False), le parcours guidé
commence par une question d'identification dont les choix viennent des documents du
retrieval — le SUJET n'est verrouillé qu'après confirmation par l'utilisateur, jamais
déduit du retrieval. Un choix permanent « changer de produit » rouvre l'étape 0.
"""
from __future__ import annotations

from unittest import mock

import pytest
from sqlmodel import Session, select

from app.database import engine
from app.models.document import Document
from app.models.guided_session import GuidedSession
from app.models.library import Library
from app.services.procedural_router_service import RoutingChoice, RoutingStep
from app.services.query_reasoning_service import GuidedModeDecision
from tests.conftest import create_test_user, extract_sse_events, extract_sse_message_text


@pytest.fixture(autouse=True)
def _neutralize_fused_understanding():
    """Décision guidée portée par le fused (P0.4) : on la neutralise (present=False) pour
    que resolve_guided_mode retombe sur decide_guided_mode (mocké) — tests déterministes,
    sans appel LLM réseau."""
    from app.services.lightweight_query_understanding import (
        GuidedDecision,
        LightweightQueryResult,
    )

    with mock.patch(
        "app.services.lightweight_query_understanding.run_lightweight_understanding",
        new=mock.AsyncMock(
            return_value=LightweightQueryResult(
                route="rag", ready_for_retrieval=True, guided=GuidedDecision(present=False)
            )
        ),
    ):
        yield


@pytest.fixture
def space_and_conversation(client, responsable_headers):
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace identification pytest"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]
    cr = client.post(
        "/api/conversations",
        headers=responsable_headers,
        json={"title": "Conv identification", "space_id": space_id},
    )
    assert cr.status_code == 201
    conv_id = cr.json()["id"]
    yield space_id, conv_id
    client.delete(f"/api/conversations/{conv_id}", headers=responsable_headers)
    client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)


@pytest.fixture
def product_documents(db_session: Session):
    """Deux documents de gammes distinctes pour alimenter les choix d'identification."""
    user = create_test_user(db_session, "responsable")
    lib = Library(name="Lib identification", user_id=user.id, is_global=False)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)

    docs = []
    for title, gammes in (
        ("Notice poignée KSR", ["KSR PVC"]),
        ("Notice poignée coulissant", ["INNOSLIDE"]),
    ):
        d = Document(
            title=title,
            document_type="written",
            processing_status="completed",
            library_id=lib.id,
            user_id=user.id,
            proferm_gammes=gammes,
        )
        db_session.add(d)
        db_session.commit()
        db_session.refresh(d)
        docs.append(d)
    return docs


def _retrieval_for(docs):
    passages = [
        {
            "passage": f"Contenu {d.title}",
            "passage_raw": f"Contenu {d.title}",
            "document_id": d.id,
            "document_title": d.title,
            "chunk_id": 100 + i,
            "chunk_index": 0,
            "page_no": 1,
            "page_start": 1,
            "page_end": 1,
            "score": 0.9 - 0.1 * i,
            "section": "Réglage",
        }
        for i, d in enumerate(docs)
    ]
    return mock.AsyncMock(return_value={"passages": passages, "status": "ok", "reason": None})


def _howto_step():
    return RoutingStep(
        step_type="instruction",
        message="Repérez la vis de fixation sous le cache.",
        choices=[
            RoutingChoice(label="C'est fait, étape suivante", value="next"),
            RoutingChoice(label="Je suis bloqué", value="stuck"),
        ],
        is_terminal=False,
    )


def _decision_unnamed():
    return GuidedModeDecision(
        is_guided=True,
        flow_kind="howto",
        topic="réglage hauteur poignée",
        product_named=False,
        needs_intent_clarification=False,
    )


def _post(client, headers, space_id, conv_id, message, guided_choice=None):
    body = {
        "message": message,
        "model": "mistral-small-latest",
        "provider": "mistral",
        "conversation_id": conv_id,
    }
    if guided_choice is not None:
        body["guided_choice"] = guided_choice
    return client.post(f"/api/spaces/{space_id}/chat/stream", headers=headers, json=body)


def _gsession(conv_id) -> GuidedSession | None:
    with Session(engine) as s:
        return s.exec(
            select(GuidedSession).where(GuidedSession.conversation_id == conv_id)
        ).first()


def _patches(docs, gen_mock):
    return (
        mock.patch("app.config.settings.GUIDED_FLOW_ENABLED", True),
        mock.patch(
            "app.services.query_reasoning_service.decide_guided_mode",
            new=mock.AsyncMock(return_value=_decision_unnamed()),
        ),
        mock.patch(
            "app.services.space_search_service.search_technical_passages",
            new=_retrieval_for(docs),
        ),
        mock.patch("app.services.guided_flow_service.generate_routing_step", new=gen_mock),
    )


def test_unnamed_product_triggers_identification_step(
    client, responsable_headers, space_and_conversation, product_documents
):
    space_id, conv_id = space_and_conversation
    gen_mock = mock.AsyncMock(return_value=_howto_step())
    p1, p2, p3, p4 = _patches(product_documents, gen_mock)
    with p1, p2, p3, p4:
        r = _post(
            client, responsable_headers, space_id, conv_id,
            "comment régler la hauteur de poignée ?",
        )

    assert r.status_code == 200
    step = [e["step"] for e in extract_sse_events(r.text) if "step" in e][0]
    assert step["step_type"] == "question"
    labels = [c["label"] for c in step["choices"]]
    # Choix = gammes des documents candidats + échappatoire, PAS un produit imposé.
    assert "KSR PVC" in labels and "INNOSLIDE" in labels
    assert labels[-1] == "Autre / je ne sais pas"
    assert "quel produit ou quelle gamme" in extract_sse_message_text(r.text)
    # L'aiguillage LLM n'a PAS été appelé : étape déterministe.
    gen_mock.assert_not_called()

    gs = _gsession(conv_id)
    assert gs.accumulated_signals["pending_product_identification"] is True
    assert gs.path[0]["node_key"] == "product_identification"


def test_identification_answer_locks_topic_and_adds_restart_choice(
    client, responsable_headers, space_and_conversation, product_documents
):
    space_id, conv_id = space_and_conversation
    gen_mock = mock.AsyncMock(return_value=_howto_step())
    p1, p2, p3, p4 = _patches(product_documents, gen_mock)
    with p1, p2, p3, p4:
        _post(client, responsable_headers, space_id, conv_id, "comment régler la hauteur de poignée ?")
        r2 = _post(
            client, responsable_headers, space_id, conv_id, "KSR PVC",
            guided_choice={"value": "product_1", "label": "KSR PVC"},
        )

    assert r2.status_code == 200
    gs = _gsession(conv_id)
    assert gs.accumulated_signals["pending_product_identification"] is False
    assert gs.accumulated_signals["confirmed_product"] == "KSR PVC"
    # Sujet = tâche + produit CONFIRMÉ par l'utilisateur.
    assert "réglage hauteur poignée" in gs.topic and "KSR PVC" in gs.topic
    # Le produit confirmé ancre aussi le retrieval des étapes suivantes.
    assert "KSR PVC" in gs.accumulated_signals["entity_texts"]

    step2 = [e["step"] for e in extract_sse_events(r2.text) if "step" in e][0]
    assert step2["step_type"] == "instruction"
    values = [c["value"] for c in step2["choices"]]
    assert "restart_product" in values  # choix permanent « changer de produit »


def test_restart_product_reopens_identification(
    client, responsable_headers, space_and_conversation, product_documents
):
    space_id, conv_id = space_and_conversation
    gen_mock = mock.AsyncMock(return_value=_howto_step())
    p1, p2, p3, p4 = _patches(product_documents, gen_mock)
    with p1, p2, p3, p4:
        _post(client, responsable_headers, space_id, conv_id, "comment régler la hauteur de poignée ?")
        _post(
            client, responsable_headers, space_id, conv_id, "KSR PVC",
            guided_choice={"value": "product_1", "label": "KSR PVC"},
        )
        r3 = _post(
            client, responsable_headers, space_id, conv_id,
            "Ce n'est pas mon produit / recommencer",
            guided_choice={"value": "restart_product", "label": "Ce n'est pas mon produit / recommencer"},
        )

    assert r3.status_code == 200
    step3 = [e["step"] for e in extract_sse_events(r3.text) if "step" in e][0]
    assert step3["step_type"] == "question"
    assert any(c["value"] == "unknown" for c in step3["choices"])

    gs = _gsession(conv_id)
    assert gs.accumulated_signals["pending_product_identification"] is True
    # Le sujet retombe sur l'étiquette de tâche, sans produit.
    assert gs.topic == "réglage hauteur poignée"


def test_unknown_product_proceeds_without_locking(
    client, responsable_headers, space_and_conversation, product_documents
):
    space_id, conv_id = space_and_conversation
    gen_mock = mock.AsyncMock(return_value=_howto_step())
    p1, p2, p3, p4 = _patches(product_documents, gen_mock)
    with p1, p2, p3, p4:
        _post(client, responsable_headers, space_id, conv_id, "comment régler la hauteur de poignée ?")
        r2 = _post(
            client, responsable_headers, space_id, conv_id, "Autre / je ne sais pas",
            guided_choice={"value": "unknown", "label": "Autre / je ne sais pas"},
        )

    assert r2.status_code == 200
    gs = _gsession(conv_id)
    assert gs.accumulated_signals["confirmed_product"] == ""
    assert gs.topic == "réglage hauteur poignée"  # aucun produit verrouillé
    # Le routeur reprend la main (questions d'observation) — sans choix « recommencer »
    # puisqu'aucun produit n'est verrouillé.
    step2 = [e["step"] for e in extract_sse_events(r2.text) if "step" in e][0]
    assert step2["step_type"] == "instruction"
    assert all(c["value"] != "restart_product" for c in step2["choices"])


def test_intent_ambiguity_defers_to_oneshot(
    client, responsable_headers, space_and_conversation
):
    """needs_intent_clarification=True ⇒ pas de mode guidé : le one-shot clarifie."""
    space_id, conv_id = space_and_conversation
    decision = GuidedModeDecision(
        is_guided=True,
        flow_kind="howto",
        topic="hauteur poignée",
        product_named=False,
        needs_intent_clarification=True,
    )
    with mock.patch("app.config.settings.GUIDED_FLOW_ENABLED", True), mock.patch(
        "app.config.settings.QUERY_UNDERSTANDING_ENABLED", False
    ), mock.patch(
        "app.services.query_reasoning_service.decide_guided_mode",
        new=mock.AsyncMock(return_value=decision),
    ), mock.patch(
        "app.services.space_search_service.search_technical_passages",
        new=mock.AsyncMock(
            return_value={"passages": [], "status": "ok", "reason": "no_results"}
        ),
    ):
        r = _post(
            client, responsable_headers, space_id, conv_id,
            "comment régler la hauteur de poignée ?",
        )

    assert r.status_code == 200
    # Aucun aiguillage guidé : pas d'événement step, pas de GuidedSession.
    assert not any("step" in e for e in extract_sse_events(r.text))
    assert _gsession(conv_id) is None
