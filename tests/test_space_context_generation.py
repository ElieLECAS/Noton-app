"""Axe 3 — génération : budget contexte 256k + plafond de tokens de réponse.

  * build_space_context_from_passages honore les plafonds (pages entières non tronquées
    sous le cap ; troncature au-delà ; borne globale respectée).
  * chat_stream_wrapper transmet bien max_tokens à l'API (fin des réponses coupées à 1024).
"""
from __future__ import annotations

from unittest import mock

import pytest

from app.routers import chat as chat_module


def _passage(text: str, *, doc="Notice", page=1, score=0.9) -> dict:
    return {"passage": text, "document_title": doc, "page_no": page, "score": score}


def test_full_page_not_truncated_under_cap():
    long_page = "A" * 9000  # > ancien cap 4000, < nouveau cap 12000
    with mock.patch.object(chat_module, "SPACE_CONTEXT_MAX_PASSAGE_CHARS", 12000), mock.patch.object(
        chat_module, "SPACE_CONTEXT_MAX_CHARS", 80000
    ):
        msg = chat_module.build_space_context_from_passages([_passage(long_page)])
    assert "A" * 9000 in msg["content"]
    assert "…" not in msg["content"]


def test_passage_truncated_above_cap():
    long_page = "B" * 9000
    with mock.patch.object(chat_module, "SPACE_CONTEXT_MAX_PASSAGE_CHARS", 4000), mock.patch.object(
        chat_module, "SPACE_CONTEXT_MAX_CHARS", 80000
    ):
        msg = chat_module.build_space_context_from_passages([_passage(long_page)])
    assert "…" in msg["content"]
    # Tronqué autour du cap, pas les 9000 chars.
    assert msg["content"].count("B") < 4100


def test_multiple_full_pages_fit_under_global_cap():
    pages = [_passage("C" * 8000, page=p) for p in range(1, 6)]  # 5 pages ~8000 chars
    with mock.patch.object(chat_module, "SPACE_CONTEXT_MAX_PASSAGE_CHARS", 12000), mock.patch.object(
        chat_module, "SPACE_CONTEXT_MAX_CHARS", 80000
    ):
        msg = chat_module.build_space_context_from_passages(pages)
    # Les 5 pages tiennent (5 * ~8000 = 40000 < 80000).
    assert "(5 passages.)" in msg["content"]


def test_global_cap_limits_passage_count():
    pages = [_passage("D" * 8000, page=p) for p in range(1, 11)]  # 10 pages
    with mock.patch.object(chat_module, "SPACE_CONTEXT_MAX_PASSAGE_CHARS", 12000), mock.patch.object(
        chat_module, "SPACE_CONTEXT_MAX_CHARS", 20000
    ):
        msg = chat_module.build_space_context_from_passages(pages)
    # Cap global 20000 → ~2 pages de 8000 seulement.
    assert "(2 passages.)" in msg["content"]


@pytest.mark.asyncio
async def test_chat_stream_wrapper_passes_max_tokens_default():
    captured: dict = {}

    async def fake_mistral(**kwargs):
        captured.update(kwargs)
        for c in ["Ré", "ponse"]:
            yield c

    with mock.patch("app.config.settings.LLM_PROVIDER", "mistral"), mock.patch(
        "app.config.settings.SPACE_CHAT_MAX_TOKENS", 2048
    ), mock.patch(
        "app.config.settings.GENERATION_REASONING_EFFORT", "none"
    ), mock.patch.object(chat_module, "mistral_chat_stream", new=fake_mistral):
        chunks = [
            c
            async for c in chat_module.chat_stream_wrapper(
                "", "mistral-large-latest", [{"role": "system", "content": "x"}]
            )
        ]

    assert chunks == ["Ré", "ponse"]
    # Sans override ni reasoning, le wrapper transmet SPACE_CHAT_MAX_TOKENS (plus le repli 1024).
    assert captured["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_chat_stream_wrapper_reasoning_bumps_tokens():
    """Reasoning high : le plancher de tokens est relevé (le thinking consomme le budget)."""
    captured: dict = {}

    async def fake_mistral(**kwargs):
        captured.update(kwargs)
        yield "x"

    with mock.patch("app.config.settings.LLM_PROVIDER", "mistral"), mock.patch(
        "app.config.settings.SPACE_CHAT_MAX_TOKENS", 2048
    ), mock.patch(
        "app.config.settings.GENERATION_REASONING_EFFORT", "high"
    ), mock.patch(
        "app.config.settings.GENERATION_REASONING_MAX_TOKENS", 8192
    ), mock.patch.object(chat_module, "mistral_chat_stream", new=fake_mistral):
        _ = [
            c
            async for c in chat_module.chat_stream_wrapper(
                "", "mistral-large-latest", [{"role": "system", "content": "x"}]
            )
        ]

    assert captured["max_tokens"] == 8192
    assert captured.get("reasoning_effort") == "high"


@pytest.mark.asyncio
async def test_chat_stream_wrapper_explicit_max_tokens_override():
    captured: dict = {}

    async def fake_mistral(**kwargs):
        captured.update(kwargs)
        yield "x"

    with mock.patch("app.config.settings.LLM_PROVIDER", "mistral"), mock.patch(
        "app.config.settings.GENERATION_REASONING_EFFORT", "none"
    ), mock.patch.object(
        chat_module, "mistral_chat_stream", new=fake_mistral
    ):
        _ = [
            c
            async for c in chat_module.chat_stream_wrapper(
                "", "mistral-large-latest", [{"role": "system", "content": "x"}], max_tokens=512
            )
        ]

    assert captured["max_tokens"] == 512
