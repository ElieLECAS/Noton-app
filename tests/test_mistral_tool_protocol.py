"""Protocole d'outils côté Mistral : recomposition des tool_calls streamés et nettoyage des
messages sans casser l'appariement appel ↔ résultat.

Avant le lecteur agentique, ``_clean_messages`` supprimait un assistant à ``content=""``
porteur de ``tool_calls`` et fusionnait deux messages ``tool`` consécutifs — l'API répondait
400 au premier round d'outils.
"""
from __future__ import annotations

import json

from app.services.mistral_service import ToolCallAccumulator, _clean_messages


# ---------------------------------------------------------------------------
# ToolCallAccumulator
# ---------------------------------------------------------------------------


def test_accumulator_reassembles_fragmented_arguments():
    acc = ToolCallAccumulator()
    acc.feed([{"index": 0, "id": "abc123def", "function": {"name": "lire_pages", "arguments": '{"document_id": 4'}}])
    acc.feed([{"index": 0, "function": {"arguments": '05, "pages": [111]}'}}])
    calls = acc.finish()
    assert len(calls) == 1
    assert calls[0]["id"] == "abc123def"
    assert calls[0]["function"]["name"] == "lire_pages"
    assert json.loads(calls[0]["function"]["arguments"]) == {"document_id": 405, "pages": [111]}


def test_accumulator_handles_parallel_calls_by_index():
    acc = ToolCallAccumulator()
    acc.feed(
        [
            {"index": 0, "id": "aaaaaaaaa", "function": {"name": "chercher_code", "arguments": '{"code": "TGY3704"}'}},
            {"index": 1, "id": "bbbbbbbbb", "function": {"name": "plan_du_document", "arguments": '{"document_id": 405}'}},
        ]
    )
    calls = acc.finish()
    assert [c["function"]["name"] for c in calls] == ["chercher_code", "plan_du_document"]


def test_accumulator_without_index_uses_id_to_open_new_call():
    acc = ToolCallAccumulator()
    acc.feed([{"id": "aaaaaaaaa", "function": {"name": "rechercher", "arguments": '{"question": "a'}}])
    acc.feed([{"function": {"arguments": 'bc"}'}}])  # prolonge le dernier
    acc.feed([{"id": "bbbbbbbbb", "function": {"name": "lire_pages", "arguments": "{}"}}])
    calls = acc.finish()
    assert len(calls) == 2
    assert json.loads(calls[0]["function"]["arguments"]) == {"question": "abc"}


def test_accumulator_generates_valid_id_and_empty_args():
    acc = ToolCallAccumulator()
    acc.feed([{"index": 0, "function": {"name": "plan_du_document"}}])
    calls = acc.finish()
    assert calls[0]["function"]["arguments"] == "{}"
    # Mistral exige des identifiants de 9 caractères alphanumériques.
    assert len(calls[0]["id"]) == 9 and calls[0]["id"].isalnum()


def test_accumulator_drops_nameless_and_accepts_dict_arguments():
    acc = ToolCallAccumulator()
    acc.feed([{"index": 0, "function": {"arguments": "{}"}}])  # sans nom → ignoré
    acc.feed([{"index": 1, "id": "ccccccccc", "function": {"name": "zoomer", "arguments": {"page": 3}}}])
    calls = acc.finish()
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"page": 3}
    assert acc.has_calls


# ---------------------------------------------------------------------------
# _clean_messages — protocole d'outils
# ---------------------------------------------------------------------------

_TC = [{"id": "abc123def", "type": "function", "function": {"name": "lire_pages", "arguments": "{}"}}]


def test_assistant_with_tool_calls_and_empty_content_is_kept():
    out = _clean_messages(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "", "tool_calls": _TC},
            {"role": "tool", "tool_call_id": "abc123def", "name": "lire_pages", "content": "[page 1] texte"},
        ]
    )
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"] == _TC
    assert out[3]["tool_call_id"] == "abc123def"
    assert out[3]["name"] == "lire_pages"


def test_consecutive_tool_messages_are_not_merged():
    out = _clean_messages(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "plan", "tool_calls": _TC + [{"id": "bbbbbbbbb", "type": "function", "function": {"name": "chercher_code", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "abc123def", "content": "r1"},
            {"role": "tool", "tool_call_id": "bbbbbbbbb", "content": "r2"},
            {"role": "user", "content": "Images des pages demandées", "images": ["QUJD"]},
        ]
    )
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "tool", "tool", "user"]
    assert [m["tool_call_id"] for m in out if m["role"] == "tool"] == ["abc123def", "bbbbbbbbb"]
    # Le message user d'images suit le protocole, avec ses parties image.
    assert isinstance(out[4]["content"], list)
    assert any(p.get("type") == "image_url" for p in out[4]["content"])


def test_assistant_after_tool_round_is_not_merged_with_tool_call_assistant():
    out = _clean_messages(
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "", "tool_calls": _TC},
            {"role": "tool", "tool_call_id": "abc123def", "content": "r1"},
            {"role": "assistant", "content": "Réponse finale."},
        ]
    )
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant"]
    assert out[3]["content"] == "Réponse finale."
    assert "tool_calls" not in out[3]


def test_plain_consecutive_messages_still_merge():
    out = _clean_messages(
        [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "assistant", "content": ""},  # vide sans tool_calls → toujours supprimé
            {"role": "assistant", "content": "C"},
        ]
    )
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[0]["content"] == "A\n\nB"
