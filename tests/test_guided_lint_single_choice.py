"""Une question à une seule réponse ne trie rien — le builder doit le signaler."""

from __future__ import annotations

from app.services.guided_lint_service import has_blocking_issues, lint_tree


def _draft(nodes):
    return {"meta": {"root_node_key": "root"}, "nodes": nodes}


def _q(key, *children):
    return {
        "node_key": key,
        "title": key,
        "message": "?",
        "is_terminal": False,
        "choices": [{"label": c, "value": c, "next_node_key": c} for c in children],
    }


def _leaf(key, kind="resolution"):
    return {
        "node_key": key,
        "title": key,
        "message": "texte",
        "is_terminal": True,
        "termination_type": kind,
        "choices": [],
    }


def test_etape_a_une_seule_reponse_signalee_sans_bloquer():
    issues = lint_tree(_draft([_q("root", "a"), _q("a", "sav"), _leaf("sav", "escalation")]))
    codes = [i.code for i in issues]
    assert codes.count("single_choice") == 2  # root et « a »
    assert not has_blocking_issues(issues)


def test_deux_reponses_ne_declenchent_rien():
    issues = lint_tree(
        _draft([_q("root", "a", "sav"), _leaf("a"), _leaf("sav", "escalation")])
    )
    assert [i.code for i in issues if i.code == "single_choice"] == []
