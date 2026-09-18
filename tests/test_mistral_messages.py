"""Mise en forme des messages pour l'API Mistral."""
from __future__ import annotations

from app.services.mistral_service import _clean_messages


def test_system_messages_are_merged_first():
    out = _clean_messages([
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "Q"},
    ])
    assert out == [{"role": "system", "content": "A\n\nB"}, {"role": "user", "content": "Q"}]


def test_consecutive_same_role_messages_merge_and_empty_are_dropped():
    out = _clean_messages([
        {"role": "user", "content": "Q1"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "R"},
        {"role": "assistant", "content": [{"type": "text", "text": "R2"}]},
    ])
    assert out == [{"role": "user", "content": "Q1\n\nQ2"}, {"role": "assistant", "content": "R\n\nR2"}]


def test_sequence_starts_with_user_after_system():
    out = _clean_messages([{"role": "system", "content": "S"}, {"role": "assistant", "content": "R"}])
    assert [m["role"] for m in out] == ["system", "user", "assistant"]
    out = _clean_messages([{"role": "assistant", "content": "R"}])
    assert [m["role"] for m in out] == ["user", "assistant"]


def test_empty_input():
    assert _clean_messages([]) == []
