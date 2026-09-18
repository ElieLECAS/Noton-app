"""Le tour de chat : citations vérifiées, anomalies, flux d'événements, trace."""
from __future__ import annotations

import asyncio
import json

import pytest

from app.config import settings
from app.services import wiki_service
from app.services.wiki_chat_service import (
    WikiAnswer,
    extract_anomalies,
    extract_citations,
)
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot(wiki_root())


def test_extract_citations_dedupes_and_flags_unknown(snapshot):
    text = (
        "La parclose 76507 (/profiles/perform76-parcloses.md) tient 44 mm. "
        "Voir aussi /profiles/perform76-parcloses.md et /profiles/page-inventee.md ; "
        "le schéma est dans raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 5."
    )
    cites = extract_citations(text, snapshot)
    assert [c["path"] for c in cites] == [
        "/profiles/perform76-parcloses.md",
        "/profiles/page-inventee.md",
    ]
    assert cites[0]["exists"] is True and cites[0]["type"] == "Profilé"
    assert cites[1]["exists"] is False and cites[1]["title"] == "page inventee"


def test_extract_citations_ignores_urls_and_index(snapshot):
    cites = extract_citations("Rien ici : https://exemple.fr/x/y.md ni (/index.md).", snapshot)
    assert [c["path"] for c in cites] == ["/index.md"]
    assert cites[0]["exists"] is True


def test_extract_anomalies():
    found = extract_anomalies("Attention CTR-17 et INC-02 ; INC-02 encore, puis VER-35.")
    assert [a["id"] for a in found] == ["CTR-17", "INC-02", "VER-35"]
    assert found[0]["path"] == "/anomalies/contradictions-entre-sources.md"
    assert found[1]["path"] == "/anomalies/incoherences-internes.md"
    assert found[2]["path"] == "/anomalies/informations-a-verifier.md"


def _events(chunks):
    return [json.loads(c[6:]) for c in chunks if c.startswith("data: ")]


def test_wiki_answer_streams_and_traces(snapshot, monkeypatch):
    captured = {}

    async def fake_stream(message, *, model, context, max_tokens, temperature, **kwargs):
        captured.update({"model": model, "context": context, "max_tokens": max_tokens,
                         "temperature": temperature, **kwargs})
        yield json.dumps({"thinking": "je relis la page des parcloses"})
        yield json.dumps({"message": {"content": "Parclose **76507** "}})
        yield json.dumps({"message": {"content": "(/profiles/perform76-parcloses.md). Voir CTR-17 et /x/inconnue.md"}})
        yield json.dumps({"usage": {"prompt_tokens": 185508, "completion_tokens": 40,
                                    "prompt_tokens_details": {"cached_tokens": 185472}}})

    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "high")
    answer = WikiAnswer(
        question="Quelle parclose pour 44 mm ?",
        history=[{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Bonjour !"}],
        snapshot=snapshot,
        model="mistral-small-latest",
        stream_fn=fake_stream,
    )

    async def collect():
        return [chunk async for chunk in answer.run()]

    events = _events(asyncio.run(collect()))
    kinds = [next(iter(e)) for e in events]
    assert kinds == ["stage", "thinking", "message", "message", "sources"]
    assert events[0]["stage"]["label"] == "Lecture du wiki"
    assert events[-1]["sources"][0]["path"] == "/profiles/perform76-parcloses.md"
    assert events[-1]["sources"][1] == {
        "path": "/x/inconnue.md", "title": "inconnue", "type": "", "status": "", "exists": False,
    }
    assert events[-1]["anomalies"] == [{"id": "CTR-17", "path": "/anomalies/contradictions-entre-sources.md"}]

    # Le prompt système est le premier message, l'historique suit, la question ferme.
    ctx = captured["context"]
    assert ctx[0]["role"] == "system" and ctx[0]["content"] == snapshot.system_prompt
    assert [m["role"] for m in ctx[1:]] == ["user", "assistant", "user"]
    assert ctx[-1]["content"] == "Quelle parclose pour 44 mm ?"
    assert captured["prompt_cache_key"] == snapshot.cache_key
    assert captured["reasoning_effort"] == "high"
    assert captured["max_tokens"] == settings.CHAT_MAX_TOKENS
    assert captured["temperature"] == settings.CHAT_TEMPERATURE

    assert answer.text.startswith("Parclose **76507**")
    assert answer.thinking == "je relis la page des parcloses"
    assert answer.trace["prompt_tokens"] == 185508
    assert answer.trace["cached_tokens"] == 185472
    assert answer.trace["cited_pages"] == ["/profiles/perform76-parcloses.md"]
    assert answer.trace["unknown_citations"] == ["/x/inconnue.md"]
    assert answer.trace["anomalies"] == ["CTR-17"]
    assert answer.trace["history_messages"] == 2
    assert answer.trace["wiki_hash"] == snapshot.cache_key
    assert wiki_service.last_call()["prompt_tokens"] == 185508


def test_wiki_answer_without_reasoning_effort(snapshot, monkeypatch):
    captured = {}

    async def fake_stream(message, **kwargs):
        captured.update(kwargs)
        yield json.dumps({"message": {"content": "ok"}})

    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "")
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake_stream)

    async def collect():
        return [chunk async for chunk in answer.run()]

    asyncio.run(collect())
    assert "reasoning_effort" not in captured
    assert answer.model == settings.MODEL_FAST
    assert answer.trace["prompt_tokens"] is None and answer.trace["cached_tokens"] is None
