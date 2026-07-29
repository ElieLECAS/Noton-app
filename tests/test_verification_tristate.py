"""B0 (plan boucle agentique 2026-07-29) — tri-état du juge de vérification.

Avant : un JSON invalide ou une erreur API produisait un blanc-seing silencieux
(answers_question=True, grounded=True) — un gate bloquant qui « valide » sur chaque
timeout ne bloque rien. Après : ``judge_status="unknown"`` — le juge n'a pas statué,
seul le contrôle programmatique fait foi, et l'échec est visible dans la trace.
"""
from __future__ import annotations

import pytest

import app.services.mistral_service as mistral_service
import app.services.response_verification_service as rvs
from app.services.response_verification_service import parse_verification_json


# ---------------------------------------------------------------------------
# parse_verification_json
# ---------------------------------------------------------------------------


def test_valid_json_is_ok():
    parsed = parse_verification_json(
        '{"answers_question": true, "grounded": false, "issues": ["x"]}'
    )
    assert parsed["judge_status"] == "ok"
    assert parsed["grounded"] is False
    assert parsed["parse_error"] is False


def test_invalid_json_is_unknown():
    parsed = parse_verification_json("le juge répond en prose")
    assert parsed["judge_status"] == "unknown"
    assert parsed["parse_error"] is True


def test_empty_output_is_unknown():
    assert parse_verification_json("")["judge_status"] == "unknown"


def test_non_dict_json_is_unknown():
    assert parse_verification_json("[1, 2]")["judge_status"] == "unknown"


# ---------------------------------------------------------------------------
# judge_relevance — erreur API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_failure_yields_unknown(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(mistral_service, "chat", _boom)
    result = await rvs.judge_relevance("q", "r", "ctx", model="test")
    assert result["judge_status"] == "unknown"
    assert result["parse_error"] is True


# ---------------------------------------------------------------------------
# verify_response — un juge unknown ne blanchit ni ne condamne
# ---------------------------------------------------------------------------


async def _unknown_judge(question, response_text, context_text, *, model):
    return {
        "answers_question": True,
        "grounded": True,
        "issues": [],
        "parse_error": True,
        "judge_status": "unknown",
    }


@pytest.mark.asyncio
async def test_unknown_judge_with_clean_programmatic_check_is_ok(monkeypatch):
    monkeypatch.setattr(rvs, "judge_relevance", _unknown_judge)
    context = "Le seuil 9F67 mesure 20 mm."
    result = await rvs.verify_response(
        question="q",
        response_text="Le seuil 9F67 fait 20 mm.",
        context_text=context,
        model="test",
        document_blocks=[context],
        cag_documents=[{"index": 1, "document_id": 1, "document_title": "Doc", "pages": [1]}],
    )
    assert result["judge_status"] == "unknown"
    assert result["ok"] is True
    assert result["judge_suspect"] is False  # unknown n'est pas un verdict négatif


@pytest.mark.asyncio
async def test_unknown_judge_does_not_mask_programmatic_failure(monkeypatch):
    monkeypatch.setattr(rvs, "judge_relevance", _unknown_judge)
    context = "Le seuil 9F67 est livré avec ses embouts."
    result = await rvs.verify_response(
        question="q",
        response_text="Le seuil 9F67 fait 125 mm.",
        context_text=context,
        model="test",
        document_blocks=[context],
        cag_documents=[{"index": 1, "document_id": 1, "document_title": "Doc", "pages": [1]}],
    )
    assert result["judge_status"] == "unknown"
    assert result["unsupported_claims"] == ["125 mm"]
    assert result["ok"] is False
