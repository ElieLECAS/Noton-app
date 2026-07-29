"""B4/B5 (plan boucle agentique 2026-07-29) — juge de suffisance et actions de relance.

Ce qui est verrouillé ici :
  * le verrou anti-complaisance MÉCANIQUE : « sufficient » sans citation retrouvable dans
    le pack → verdict non actionnable (unknown/evidence_not_found) ;
  * le tri-état d'infra : timeout / JSON invalide / verdict hors vocabulaire → unknown ;
  * la résolution de l'élection via les dossiers candidats (un index inconnu est écarté :
    le juge ne peut pas élire un document qu'on ne lui a pas montré) ;
  * les actions de relance : relancer LA MÊME requête = pas une action (le biais de
    densité de référence est déterministe, on aurait les mêmes nomenclatures en tête).
"""
from __future__ import annotations

import json

import pytest

import app.services.mistral_service as mistral_service
from app.config import settings
from app.services.retrieval_judge_service import (
    apply_judge_action,
    build_judge_messages,
    build_judge_note_block,
    evidence_in_pack,
    judge_candidates,
    normalize_verdict,
    parse_judge_json,
    passages_pool_key,
)
from app.services.slot_catalog import expected_content_for_intent

PACK_TEXT = (
    "DOCUMENT 2 : « SOLEAL-GY-55-Catalogue-fabrication »\n"
    "[page 111 — ★ page retrouvée par la recherche]\n"
    "Connecter la tringle de la fermeture 3 points à la rallonge, puis glisser le "
    "support de guidage de la rallonge dans la tringle.\n"
)
CAG_DOCUMENTS = [
    {"index": 1, "document_id": 10, "document_title": "Catalogue conception"},
    {"index": 2, "document_id": 20, "document_title": "Catalogue fabrication"},
]


def _verdict_payload(**overrides):
    payload = {
        "verdict": "sufficient",
        "confidence": 0.9,
        "evidence": "Connecter la tringle de la fermeture 3 points à la rallonge",
        "elected": [{"document_index": 2, "pages": [111], "role": "steps"}],
        "missing": "",
        "next_action": {
            "rewritten_query": "",
            "widen_scope": False,
            "drop_anchor": False,
            "restrict_to_document_index": None,
            "raise_k": False,
        },
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_judge_json_valid():
    assert parse_judge_json(json.dumps(_verdict_payload()))["verdict"] == "sufficient"


def test_parse_judge_json_with_prose_wrapper():
    raw = "Voici mon analyse :\n" + json.dumps(_verdict_payload()) + "\nVoilà."
    assert parse_judge_json(raw) is not None


def test_parse_judge_json_invalid():
    assert parse_judge_json("pas du json") is None
    assert parse_judge_json("") is None


# ---------------------------------------------------------------------------
# Preuve littérale
# ---------------------------------------------------------------------------


def test_evidence_exact_match():
    assert evidence_in_pack(
        "Connecter la tringle de la fermeture 3 points à la rallonge", PACK_TEXT
    )


def test_evidence_tolerates_accents_case_and_spacing():
    assert evidence_in_pack(
        "connecter la tringle de la fermeture 3 points a la rallonge", PACK_TEXT
    )


def test_evidence_with_ellipsis_segments():
    assert evidence_in_pack(
        "Connecter la tringle de la fermeture 3 points ... support de guidage de la rallonge",
        PACK_TEXT,
    )


def test_fabricated_evidence_is_rejected():
    assert not evidence_in_pack("Fixer la crémone sur le haut dormant avec 6 vis", PACK_TEXT)


def test_too_short_evidence_is_rejected():
    assert not evidence_in_pack("tringle", PACK_TEXT)


# ---------------------------------------------------------------------------
# Normalisation du verdict (contrôles mécaniques)
# ---------------------------------------------------------------------------


def test_sufficient_with_verified_evidence_is_actionable():
    verdict = normalize_verdict(_verdict_payload(), CAG_DOCUMENTS, PACK_TEXT)
    assert verdict["status"] == "ok"
    assert verdict["verdict"] == "sufficient"
    assert verdict["evidence_verified"] is True
    assert verdict["elected"][0]["document_id"] == 20
    assert verdict["elected"][0]["pages"] == [111]


def test_sufficient_without_findable_evidence_is_degraded():
    """LE verrou anti-complaisance : la preuve n'existe pas dans le pack → pas d'élection,
    pas de relance — le verdict vaut « pas de juge » et se voit dans la trace."""
    payload = _verdict_payload(evidence="Fixer la crémone 3 points sur le haut dormant")
    verdict = normalize_verdict(payload, CAG_DOCUMENTS, PACK_TEXT)
    assert verdict["status"] == "unknown"
    assert verdict["status_reason"] == "evidence_not_found"


def test_insufficient_is_actionable_without_evidence():
    payload = _verdict_payload(
        verdict="insufficient",
        evidence="",
        missing="les étapes de pose de la rallonge",
        next_action={"rewritten_query": "installation rallonge 4eme point TGY3704"},
    )
    verdict = normalize_verdict(payload, CAG_DOCUMENTS, PACK_TEXT)
    assert verdict["status"] == "ok"
    assert verdict["verdict"] == "insufficient"
    assert verdict["next_action"]["rewritten_query"].startswith("installation")


def test_unknown_document_index_is_dropped():
    payload = _verdict_payload(
        elected=[{"document_index": 9, "pages": [1]}, {"document_index": 2, "pages": [111]}]
    )
    verdict = normalize_verdict(payload, CAG_DOCUMENTS, PACK_TEXT)
    assert [e["document_id"] for e in verdict["elected"]] == [20]


def test_low_confidence_is_unknown():
    payload = _verdict_payload(confidence=0.2)
    verdict = normalize_verdict(payload, CAG_DOCUMENTS, PACK_TEXT, min_confidence=0.5)
    assert verdict["status"] == "unknown"
    assert verdict["status_reason"] == "low_confidence"


def test_invalid_verdict_vocabulary_is_unknown():
    verdict = normalize_verdict(
        _verdict_payload(verdict="maybe"), CAG_DOCUMENTS, PACK_TEXT
    )
    assert verdict["status"] == "unknown"
    assert verdict["status_reason"] == "invalid_verdict"


def test_parse_failure_is_unknown():
    verdict = normalize_verdict(None, CAG_DOCUMENTS, PACK_TEXT)
    assert verdict["status"] == "unknown"
    assert verdict["status_reason"] == "parse_error"


def test_restrict_action_resolves_document_id():
    payload = _verdict_payload(
        verdict="insufficient",
        evidence="",
        next_action={"restrict_to_document_index": 2},
    )
    verdict = normalize_verdict(payload, CAG_DOCUMENTS, PACK_TEXT)
    assert verdict["next_action"]["restrict_to_document_id"] == 20


# ---------------------------------------------------------------------------
# Actions de relance
# ---------------------------------------------------------------------------


def test_same_query_without_levers_is_no_action():
    verdict = {
        "next_action": {
            "rewritten_query": "  Comment installer la rallonge ?  ",
            "widen_scope": False,
            "drop_anchor": False,
            "restrict_to_document_id": None,
            "raise_k": False,
        }
    }
    assert apply_judge_action(verdict, current_query="comment installer la rallonge ?") is None


def test_rewritten_query_is_an_action():
    verdict = {
        "next_action": {
            "rewritten_query": "installation rallonge 4eme point TGY3704 coulissant",
            "widen_scope": False,
            "drop_anchor": False,
            "restrict_to_document_id": None,
            "raise_k": False,
        }
    }
    action = apply_judge_action(verdict, current_query="comment installer la rallonge ?")
    assert action["query_text"].startswith("installation rallonge")
    assert "requête" in action["label"]


def test_flags_alone_are_actions():
    verdict = {
        "next_action": {
            "rewritten_query": "",
            "widen_scope": True,
            "drop_anchor": True,
            "restrict_to_document_id": 20,
            "raise_k": True,
        }
    }
    action = apply_judge_action(verdict, current_query="q")
    assert action["widen_scope"] and action["drop_anchor"] and action["raise_k"]
    assert action["restrict_document_id"] == 20


def test_missing_next_action_is_none():
    assert apply_judge_action({"next_action": None}, current_query="q") is None


# ---------------------------------------------------------------------------
# Aides d'orchestration
# ---------------------------------------------------------------------------


def test_passages_pool_key_detects_no_progress():
    pool_a = [{"document_id": 1, "page_no": 42}, {"document_id": 2, "page_no": 66}]
    pool_b = [{"document_id": 2, "page_no": 66}, {"document_id": 1, "page_no": 42}]
    pool_c = [{"document_id": 2, "page_no": 111}]
    assert passages_pool_key(pool_a) == passages_pool_key(pool_b)
    assert passages_pool_key(pool_a) != passages_pool_key(pool_c)


def test_judge_note_block_lists_election_and_evidence():
    verdict = normalize_verdict(_verdict_payload(), CAG_DOCUMENTS, PACK_TEXT)
    note = build_judge_note_block(verdict)
    assert "NOTE DE RECHERCHE" in note
    assert "Catalogue fabrication" in note
    assert "pages 111" in note
    assert "Connecter la tringle" in note


def test_note_block_without_election_is_none():
    verdict = normalize_verdict(
        _verdict_payload(elected=[]), CAG_DOCUMENTS, PACK_TEXT
    )
    assert build_judge_note_block(verdict) is None


def test_expected_content_mapping():
    assert "nomenclature" in expected_content_for_intent("installation")
    assert "conception" in expected_content_for_intent("product_selection")
    assert expected_content_for_intent(None) == expected_content_for_intent("inconnu")


def test_judge_messages_carry_images_and_coverage():
    messages = build_judge_messages(
        question="q",
        intent="installation",
        expected_content="des étapes",
        judge_context="CTX",
        coverage_line="ABSENTES des dossiers : TGY3704",
        round_index=2,
        previous_missing="les étapes",
        images=["b64=="],
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["images"] == ["b64=="]
    assert "RELANCE n°1" in messages[1]["content"]
    assert "ABSENTES des dossiers" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Appel LLM (mocké)
# ---------------------------------------------------------------------------


def _judge_pack():
    return {"context_text": PACK_TEXT, "cag_documents": CAG_DOCUMENTS, "coverage": {}}


@pytest.mark.asyncio
async def test_judge_candidates_happy_path(monkeypatch):
    async def _fake_chat(*args, **kwargs):
        assert kwargs.get("response_format") == {"type": "json_object"}
        assert kwargs.get("temperature") == 0.0
        return {"choices": [{"message": {"content": json.dumps(_verdict_payload())}}]}

    monkeypatch.setattr(mistral_service, "chat", _fake_chat)
    verdict = await judge_candidates(
        question="comment installer la rallonge ?",
        intent="installation",
        expected_content="des étapes",
        judge_pack=_judge_pack(),
    )
    assert verdict["status"] == "ok"
    assert verdict["verdict"] == "sufficient"
    assert verdict["model"] == settings.MODEL_JUDGE
    assert verdict["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_judge_candidates_api_failure_is_unknown(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("down")

    monkeypatch.setattr(mistral_service, "chat", _boom)
    verdict = await judge_candidates(
        question="q",
        intent=None,
        expected_content="x",
        judge_pack=_judge_pack(),
    )
    assert verdict["status"] == "unknown"
    assert verdict["status_reason"] == "api_error"


@pytest.mark.asyncio
async def test_judge_candidates_garbage_output_is_unknown(monkeypatch):
    async def _fake_chat(*args, **kwargs):
        return {"choices": [{"message": {"content": "je pense que oui"}}]}

    monkeypatch.setattr(mistral_service, "chat", _fake_chat)
    verdict = await judge_candidates(
        question="q",
        intent=None,
        expected_content="x",
        judge_pack=_judge_pack(),
    )
    assert verdict["status"] == "unknown"
    assert verdict["status_reason"] == "parse_error"
