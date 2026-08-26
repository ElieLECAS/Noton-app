"""Système de continuation de conversation — état persistant (fil de la discussion).

Couvre :
  * conversation_state_service : fusion des entités en focus (récence, dédup, plafond,
    reset sur topic_shift), calcul du sujet courant (LLM > persisté > repli), rendu
    des blocs prompt (compréhension et génération).
  * lightweight_query_understanding : injection du fil persistant dans le prompt
    fusionné, propagation du current_topic LLM, fusion des focus_entities dans le
    query_context retourné, compaction des extraits d'historique (tableaux Markdown).
"""
from __future__ import annotations

import json
from unittest import mock

import pytest

from app.services import lightweight_query_understanding as lqu
from app.services.conversation_state_service import (
    FOCUS_ENTITIES_MAX,
    build_conversation_state,
    format_generation_state_block,
    format_state_facts,
    merge_focus_entities,
)


def _fake_chat(content: dict):
    return {"choices": [{"message": {"content": json.dumps(content)}}]}


# ---------------------------------------------------------------------------
# merge_focus_entities
# ---------------------------------------------------------------------------


def test_merge_focus_entities_recency_and_dedup():
    merged = merge_focus_entities(
        ["6101", "9F68"],
        ["9f68", "A474"],
        topic_shift=False,
    )
    # Nouvelles d'abord, dédup insensible à la casse (la graphie récente gagne).
    assert merged == ["9f68", "A474", "6101"]


def test_merge_focus_entities_reset_on_topic_shift():
    merged = merge_focus_entities(["6101", "9F68"], ["Lumine65"], topic_shift=True)
    assert merged == ["Lumine65"]


def test_merge_focus_entities_capped():
    previous = [f"REF{i}" for i in range(FOCUS_ENTITIES_MAX)]
    merged = merge_focus_entities(previous, ["NOUVELLE"], topic_shift=False)
    assert len(merged) == FOCUS_ENTITIES_MAX
    assert merged[0] == "NOUVELLE"


# ---------------------------------------------------------------------------
# build_conversation_state
# ---------------------------------------------------------------------------


def test_state_llm_topic_evolves_without_shift():
    state = build_conversation_state(
        {"current_topic": "dormant 6101", "focus_entities": ["6101"]},
        topic_shift=False,
        llm_topic="dormant 6101 — dimensions",
        fallback_topic="tu as ses dimensions ?",
        new_entities=[],
    )
    assert state["current_topic"] == "dormant 6101 — dimensions"
    assert state["focus_entities"] == ["6101"]


def test_state_keeps_persisted_topic_when_llm_silent():
    state = build_conversation_state(
        {"current_topic": "dormant 6101"},
        topic_shift=False,
        llm_topic="",
        fallback_topic="et ses dimensions ?",
        new_entities=[],
    )
    assert state["current_topic"] == "dormant 6101"


def test_state_reset_on_topic_shift_uses_fallback():
    state = build_conversation_state(
        {"current_topic": "dormant 6101", "focus_entities": ["6101"]},
        topic_shift=True,
        llm_topic="",
        fallback_topic="pose fenêtre Lumine65",
        new_entities=["Lumine65"],
    )
    assert state["current_topic"] == "pose fenêtre Lumine65"
    assert state["focus_entities"] == ["Lumine65"]


# ---------------------------------------------------------------------------
# Rendu des blocs prompt
# ---------------------------------------------------------------------------


def test_format_state_facts_empty_on_first_turn():
    assert format_state_facts(None) == ""
    assert format_state_facts({}) == ""
    assert format_state_facts({"current_topic": "", "focus_entities": []}) == ""


def test_format_state_facts_renders_topic_and_entities():
    facts = format_state_facts(
        {"current_topic": "dormant 6101", "focus_entities": ["6101", "9F68"]}
    )
    assert "Sujet courant : dormant 6101" in facts
    assert "6101, 9F68" in facts


def test_generation_block_includes_resolved_question_when_different():
    block = format_generation_state_block(
        {"current_topic": "dormant 6101", "focus_entities": ["6101"]},
        standalone_question="Quelles sont les dimensions du dormant 6101 ?",
        original_message="tu as ses dimensions ?",
    )
    assert "FIL DE LA CONVERSATION" in block
    assert "dimensions du dormant 6101" in block
    assert "renvoient à ce sujet courant" in block


def test_generation_block_omits_question_when_identical():
    block = format_generation_state_block(
        {"current_topic": "dormant 6101"},
        standalone_question="Parle-moi du dormant 6101",
        original_message="parle-moi du dormant 6101",
    )
    assert "signifie" not in block


def test_generation_block_empty_without_state():
    assert format_generation_state_block({}, standalone_question="x", original_message="y") == ""


# ---------------------------------------------------------------------------
# Compréhension fusionnée — fil persistant injecté + current_topic propagé
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fused_prompt_carries_persisted_state(db_session):
    state = {
        "user_message": "tu as ses dimensions ?",
        "original_user_message": "tu as ses dimensions ?",
        "session": db_session,
        "history": [
            {"role": "user", "content": "parle moi du dormant 6101"},
            {"role": "assistant", "content": "Fiche technique — 6101 ..."},
        ],
        "persisted_context": {
            "current_topic": "dormant 6101",
            "focus_entities": ["6101", "9F68"],
        },
    }
    payload = {
        "route": "rag",
        "topic_shift": False,
        "standalone_question": "dimensions du dormant 6101",
        "current_topic": "dormant 6101 — dimensions",
        "too_vague": False,
        "entities": [],
        "confidence": 0.9,
    }
    with mock.patch.object(lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(payload))) as mock_chat:
        out = await lqu._node_fused_understand(state)

    # Le fil persistant est fourni au LLM de compréhension.
    user_prompt = mock_chat.await_args.kwargs["context"][1]["content"]
    assert "État de la conversation" in user_prompt
    assert "dormant 6101" in user_prompt
    assert "9F68" in user_prompt
    # Le sujet mis à jour par le LLM est propagé.
    assert out["llm_current_topic"] == "dormant 6101 — dimensions"


@pytest.mark.asyncio
async def test_run_fused_merges_focus_entities_across_turns(db_session):
    understand_payload = {
        "route": "rag",
        "topic_shift": False,
        "standalone_question": "seuil 9F68 du dormant 6101",
        "current_topic": "dormant 6101 — seuil",
        "entities": [{"text": "9F68"}],
        "detected_references": ["9F68"],
        "confidence": 0.9,
    }

    # Un seul appel LLM depuis le nettoyage du 2026-08-26 : plus d'aiguillage à faire.
    with mock.patch.object(
        lqu, "chat", new=mock.AsyncMock(return_value=_fake_chat(understand_payload))
    ):
        result = await lqu.run_lightweight_understanding(
            user_message="et son seuil ?",
            history=[
                {"role": "user", "content": "parle moi du dormant 6101"},
                {"role": "assistant", "content": "Le dormant 6101 ..."},
            ],
            session=db_session,
            persisted_context={
                "current_topic": "dormant 6101",
                "focus_entities": ["6101"],
            },
        )

    qc = result.query_context
    assert qc["current_topic"] == "dormant 6101 — seuil"
    # Nouvelle entité en tête, ancienne conservée (pas de topic_shift).
    assert qc["focus_entities"][0] == "9F68"
    assert "6101" in qc["focus_entities"]


# ---------------------------------------------------------------------------
# Compaction des extraits d'historique (réponses structurées → bruit maîtrisé)
# ---------------------------------------------------------------------------


def test_history_snippet_strips_markdown_tables():
    fiche_markdown = (
        "## Fiche technique — 6101\n"
        "| Référence | Longueur (mm) |\n"
        "|-----------|---------------|\n"
        "| A469 | 31,5 |\n"
        "Dormant standard référencé 6101, seuil 9F68."
    )
    snippet = lqu._history_snippet(
        [{"role": "assistant", "content": fiche_markdown}], turns=6, cap=400
    )
    assert "A469" not in snippet  # rangée de tableau retirée
    assert "6101" in snippet
    assert "seuil 9F68" in snippet
