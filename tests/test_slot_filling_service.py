"""Tests unitaires du slot filling menuiserie (Espace)."""
from __future__ import annotations

import json
from unittest import mock

import pytest
from sqlmodel import Session

from app.services.query_reasoning_service import RetrievalDecision
from tests.conftest import extract_sse_message_text
from app.models.conversation import Conversation
from app.models.message import Message
from app.services.slot_filling_service import (
    SlotState,
    build_canonical_query,
    build_clarification_actions,
    build_clarification_message,
    build_retrieval_query,
    build_retrieval_query_from_conversation,
    build_slot_context_for_generation,
    extract_slots_heuristic,
    filter_passages_by_slot_material,
    load_slot_state_from_session,
    merge_slot_states,
    process_slot_filling,
    slot_state_to_sources_payload,
    validate_slots,
)


def test_vague_fenetre_missing_material():
    state = extract_slots_heuristic("J'ai un problème sur ma fenêtre qui ferme pas")
    result = validate_slots(state)
    assert result.ready is False
    assert "material" in result.missing_required_slots
    assert "type" not in result.missing_required_slots
    assert "problem_symptom" not in result.missing_required_slots
    assert any("matériau" in q.lower() for q in result.clarification_questions)


def test_vague_fenetre_without_specific_symptom():
    state = extract_slots_heuristic("J'ai un problème avec ma fenêtre")
    result = validate_slots(state)
    assert result.ready is False
    assert "material" in result.missing_required_slots
    assert "problem_symptom" in result.missing_required_slots
    assert result.slot_state.problem_symptom is None


def test_pvc_followup_still_needs_problem_symptom():
    """Répondre uniquement au matériau ne doit pas déclencher le RAG."""
    turn1 = extract_slots_heuristic("J'ai un problème avec ma fenêtre")
    turn1_validated = validate_slots(turn1)
    assert turn1_validated.ready is False

    turn2_extracted = extract_slots_heuristic(
        "PVC",
        history=[{"role": "user", "content": "J'ai un problème avec ma fenêtre"}],
    )
    merged = merge_slot_states(turn1_validated.slot_state, turn2_extracted)
    merged = merge_slot_states(merged, SlotState(material="pvc"))
    result = validate_slots(merged)
    assert result.ready is False
    assert result.slot_state.material == "pvc"
    assert "problem_symptom" in result.missing_required_slots


def test_coulissant_missing_galandage_and_material():
    state = extract_slots_heuristic("Mon coulissant ferme mal")
    result = validate_slots(state)
    assert result.ready is False
    assert "galandage" in result.missing_required_slots
    assert "material" in result.missing_required_slots


def test_complete_diagnostic_ready():
    state = extract_slots_heuristic(
        "Fenêtre PVC, gamme Perform 70, ferme mal côté poignée"
    )
    result = validate_slots(state)
    assert result.ready is True
    assert result.canonical_query
    assert "pvc" in result.canonical_query.lower()
    assert "perform" in result.canonical_query.lower()


def test_coulissant_galandage_oui_ready():
    state = extract_slots_heuristic("Coulissant à galandage alu, le vantail coince en fin de course")
    result = validate_slots(state)
    assert result.ready is True
    assert "galandage" in result.canonical_query.lower()


def test_recherche_reference_requires_type():
    state = extract_slots_heuristic("Quelle est la cote du dormant Profine Perform 70 ?")
    result = validate_slots(state)
    # Profine + Perform détectés mais type manquant
    assert result.ready is False
    assert "type" in result.missing_required_slots


def test_recherche_reference_with_type_ready():
    state = extract_slots_heuristic(
        "Fenêtre PVC Profine Perform 70, cote du dormant"
    )
    result = validate_slots(state)
    assert result.ready is True


def test_merge_slot_states_preserves_previous():
    previous = SlotState(type="fenetre", material="pvc")
    new = SlotState(problem_symptom="ferme mal côté poignée")
    merged = merge_slot_states(previous, new)
    assert merged.type == "fenetre"
    assert merged.material == "pvc"
    assert merged.problem_symptom == "ferme mal côté poignée"


def test_build_retrieval_query_combines_canonical_and_raw():
    q = build_retrieval_query("fenêtre pvc diagnostic ferme mal", "ma fenêtre ferme pas")
    assert "pvc" in q.lower()
    assert "fenêtre" in q.lower() or "fenetre" in q.lower()


def test_build_retrieval_query_adds_pvc_material_terms():
    state = SlotState(type="fenetre", material="pvc", problem_symptom="poignée bloquée")
    q = build_retrieval_query(
        build_canonical_query(state),
        "pvc",
        slot_state=state,
    )
    assert "menuiserie" in q.lower()
    assert "pvc" in q.lower()


def test_filter_passages_excludes_aluminium_doc_for_pvc():
    passages = [
        {"document_title": "catalogue-aluminium-fenetres-2021", "score": 0.9},
        {"document_title": "catalogue-pvc-fenetres-2021", "score": 0.8},
    ]
    state = SlotState(material="pvc")
    filtered = filter_passages_by_slot_material(passages, state)
    assert len(filtered) == 1
    assert "pvc" in filtered[0]["document_title"].lower()


def test_filter_passages_excludes_pvc_doc_for_alu():
    passages = [
        {"document_title": "catalogue-aluminium-fenetres", "score": 0.9},
        {"document_title": "catalogue-pvc-fenetres", "score": 0.85},
    ]
    state = SlotState(material="alu")
    filtered = filter_passages_by_slot_material(passages, state)
    assert len(filtered) == 1
    assert "aluminium" in filtered[0]["document_title"].lower()


def test_filter_passages_keeps_ambiguous_title_without_material_signal():
    passages = [
        {"document_title": "notice-poignee-fenetre-GA-5156", "score": 0.9},
    ]
    state = SlotState(material="pvc")
    assert filter_passages_by_slot_material(passages, state) == passages


def test_filter_passages_returns_empty_when_only_incompatible():
    passages = [
        {"document_title": "catalogue-aluminium-fenetres-2021", "score": 0.9},
    ]
    state = SlotState(material="pvc")
    assert filter_passages_by_slot_material(passages, state) == []


def test_slot_context_generation_material_constraint():
    state = SlotState(type="fenetre", material="pvc", problem_symptom="poignée ne tourne plus")
    block = build_slot_context_for_generation(state)
    assert "PVC" in block
    assert "aluminium" in block.lower()
    assert "CONTRAINTE MATÉRIAU" in block
    assert "CONTRAINTE COTES" in block


def test_membrane_monomur_requires_type_and_material():
    text = "retombée membrane applique extérieure monomur"
    state = extract_slots_heuristic(text)
    result = validate_slots(state, conversation_text=text)
    assert result.ready is False
    assert result.slot_state.intent_type == "norme_procedure"
    assert "type" in result.missing_required_slots
    assert "material" in result.missing_required_slots


def test_norme_procedure_pose_requires_context_usage():
    state = SlotState(
        intent_type="norme_procedure",
        type="fenetre",
        material="pvc",
        component_part="membrane",
    )
    result = validate_slots(state, conversation_text="retombée membrane applique extérieure")
    assert result.ready is False
    assert "context_usage" in result.missing_required_slots


def test_retrieval_query_aggregates_conversation_history():
    turn1_text = "retombée membrane monomur applique extérieure"
    turn1 = extract_slots_heuristic(turn1_text)
    turn2 = extract_slots_heuristic(
        "fenêtre",
        history=[{"role": "user", "content": turn1_text}],
    )
    merged = merge_slot_states(turn1, turn2)
    merged = merge_slot_states(merged, SlotState(type="fenetre", material="pvc"))
    canonical = build_canonical_query(merged)
    history = [{"role": "user", "content": turn1_text}]
    q = build_retrieval_query_from_conversation(
        canonical,
        history,
        "fenêtre",
        slot_state=merged,
    )
    assert "membrane" in q.lower()
    assert "monomur" in q.lower()
    assert "fenêtre" in q.lower() or "fenetre" in q.lower()


def test_pose_followup_pvc_still_needs_context_when_applique_detected():
    turn1_text = "retombée membrane applique extérieure monomur"
    turn1 = validate_slots(
        extract_slots_heuristic(turn1_text),
        conversation_text=turn1_text,
    )
    merged = merge_slot_states(turn1.slot_state, SlotState(type="fenetre"))
    merged = merge_slot_states(merged, SlotState(material="pvc"))
    result = validate_slots(merged, conversation_text=turn1_text)
    # context_usage déjà détecté (monomur ou applique) → prêt si tout rempli
    if result.slot_state.context_usage:
        assert result.ready is True
    else:
        assert "context_usage" in result.missing_required_slots


def test_build_retrieval_query_adds_pose_terms_for_monomur():
    state = SlotState(
        type="fenetre",
        material="pvc",
        context_usage="monomur",
        component_part="membrane",
    )
    q = build_retrieval_query(
        build_canonical_query(state),
        "retombée membrane monomur",
        slot_state=state,
    )
    assert "monomur" in q.lower()
    assert "schéma" in q.lower() or "schema" in q.lower()


def test_pose_page_score_adjustment_boosts_schema_over_garde():
    from unittest.mock import MagicMock

    from app.services.space_search_service import _adjust_colpali_scores_for_pose

    def make_node(score: float, page_no: int, text: str):
        node = MagicMock()
        node.metadata = {"page_no": page_no}
        node.get_content.return_value = text
        nws = MagicMock()
        nws.node = node
        nws.score = score
        return nws

    garde = make_node(0.9, 4, "Avis domaine d'emploi dossier technique")
    schema = make_node(0.85, 46, "POSE COUPE schéma mise en oeuvre calfeutrement")
    nodes = _adjust_colpali_scores_for_pose(
        [garde, schema],
        "retombée membrane monomur applique extérieure",
    )
    assert schema.score > garde.score


def test_clarification_recap_formats_context_usage_labels():
    state = SlotState(
        intent_type="norme_procedure",
        component_part="membrane",
        context_usage="applique_exterieure",
    )
    from app.services.slot_filling_service import SlotValidationResult, build_clarification_message

    validation = SlotValidationResult(
        ready=False,
        missing_required_slots=["type"],
        clarification_questions=["Quel type de produit concerne votre demande ?"],
        slot_state=state,
        conversation_text=(
            "Quelle retombée de membrane prévoir en applique extérieure sur monomur ?"
        ),
    )
    msg = build_clarification_message(validation)
    assert "applique_exterieure" not in msg
    assert "applique extérieure" in msg.lower()
    assert "monomur" in msg.lower()
    assert "membrane" in msg.lower()


def test_build_clarification_message_format():
    state = extract_slots_heuristic("Mon coulissant ferme mal")
    validation = validate_slots(state)
    msg = build_clarification_message(validation)
    assert "déjà compris" in msg.lower()
    assert "coulissant" in msg.lower()
    assert "matériau" in msg.lower()
    assert "galandage" in validation.missing_required_slots
    assert len(validation.clarification_questions) == 1


def test_progressive_asks_one_question_at_a_time():
    state = extract_slots_heuristic("J'ai un problème avec ma fenêtre")
    result = validate_slots(state)
    assert len(result.clarification_questions) == 1
    assert "matériau" in result.clarification_questions[0].lower()
    assert "problem_symptom" in result.missing_required_slots


def test_clarification_actions_type_buttons():
    state = SlotState(intent_type="diagnostic_probleme", problem_symptom="ferme mal")
    actions = build_clarification_actions(state, ["type"])
    labels = [a["label"] for a in actions]
    assert labels == ["Fenêtre", "Porte", "Coulissant"]
    assert all(a["slot"] == "type" for a in actions)


def test_clarification_actions_material_coulissant_pvc_alu_only():
    state = SlotState(type="coulissant", intent_type="diagnostic_probleme", problem_symptom="coince")
    actions = build_clarification_actions(state, ["material"])
    values = [a["value"] for a in actions]
    assert values == ["pvc", "alu"]


def test_clarification_actions_material_fenetre_all_options():
    state = SlotState(type="fenetre", intent_type="diagnostic_probleme", problem_symptom="ferme mal")
    actions = build_clarification_actions(state, ["material"])
    values = [a["value"] for a in actions]
    assert values == ["pvc", "alu", "mixte", "bois"]


def test_validate_slots_includes_clarification_actions():
    state = extract_slots_heuristic("J'ai un problème sur ma poignée")
    result = validate_slots(state)
    assert result.clarification_actions
    assert result.clarification_actions[0]["slot"] == "type"


def test_clarification_recap_after_material_answer():
    turn1 = validate_slots(extract_slots_heuristic("J'ai un problème avec ma fenêtre"))
    merged = merge_slot_states(turn1.slot_state, SlotState(material="pvc"))
    result = validate_slots(merged)
    msg = build_clarification_message(result)
    assert "PVC" in msg
    assert "symptôme" in msg.lower()
    assert "ferme mal" in msg.lower() or "crémone" in msg.lower()


def test_clarification_optional_precision_hint_on_symptom_step():
    state = SlotState(type="fenetre", material="pvc", intent_type="diagnostic_probleme")
    result = validate_slots(state)
    msg = build_clarification_message(result)
    assert "gamme" in msg.lower()
    assert "💡" in msg


def test_slot_state_persistence_roundtrip(db_session: Session):
    from tests.conftest import create_test_user
    from app.models.space import Space

    user = create_test_user(db_session, "responsable")
    space = Space(name="Slot test", user_id=user.id)
    db_session.add(space)
    db_session.commit()
    db_session.refresh(space)

    conv = Conversation(title="Slot conv", user_id=user.id, space_id=space.id)
    db_session.add(conv)
    db_session.commit()
    db_session.refresh(conv)

    state = extract_slots_heuristic("Mon coulissant ferme mal")
    validation = validate_slots(state)
    sources = slot_state_to_sources_payload(validation)

    msg = Message(
        conversation_id=conv.id,
        role="assistant",
        content=build_clarification_message(validation),
        sources=sources,
    )
    db_session.add(msg)
    db_session.commit()

    loaded = load_slot_state_from_session(db_session, conv.id)
    assert loaded is not None
    assert loaded.type == "coulissant"


@pytest.mark.asyncio
async def test_process_slot_filling_merges_previous():
    previous = SlotState(type="fenetre", material="pvc")
    with mock.patch(
        "app.services.slot_filling_service.extract_slots_llm",
        new=mock.AsyncMock(return_value=None),
    ):
        result = await process_slot_filling(
            "ferme mal côté poignée",
            history=[],
            previous_state=previous,
            use_llm=False,
        )
    assert result.ready is True
    assert result.slot_state.type == "fenetre"
    assert result.slot_state.material == "pvc"


def test_space_chat_stream_need_clarification(client, responsable_headers):
    """Requête vague -> need_clarification SSE, pas de retrieval."""
    sp = client.post(
        "/api/spaces",
        headers=responsable_headers,
        json={"name": "Espace slot filling"},
    )
    assert sp.status_code == 201
    space_id = sp.json()["id"]
    try:
        mock_decision = mock.AsyncMock(
            return_value=RetrievalDecision(decision="rag", reasoning="Question technique")
        )
        with mock.patch(
            "app.services.query_reasoning_service.decide_retrieval_route",
            new=mock_decision,
        ), mock.patch(
            "app.services.space_search_service.search_technical_passages",
            new=mock.AsyncMock(),
        ) as mock_search:
            r = client.post(
                f"/api/spaces/{space_id}/chat/stream",
                headers=responsable_headers,
                json={
                    "message": "J'ai un problème sur ma fenêtre qui ferme pas",
                    "model": "mistral-small-latest",
                    "provider": "mistral",
                    "conversation_id": None,
                },
            )
        assert r.status_code == 200
        assert "need_clarification" in r.text
        assert "clarification_actions" in r.text
        text_content = extract_sse_message_text(r.text)
        assert "matériau" in text_content.lower()
        mock_search.assert_not_called()
    finally:
        client.delete(f"/api/spaces/{space_id}", headers=responsable_headers)

