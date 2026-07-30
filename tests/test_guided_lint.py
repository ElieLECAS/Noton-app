"""Lint des arbres SAV — chaque code sur des arbres synthétiques (pur, sans DB)."""
from app.services.guided_lint_service import has_blocking_issues, lint_tree


def _draft(nodes, root="root"):
    return {"meta": {"root_node_key": root}, "nodes": nodes}


def _codes(issues):
    return {i.code for i in issues}


def _leaf(key, **kw):
    base = {
        "node_key": key,
        "step_type": "diagnostic",
        "title": key,
        "message": "Remplacer la pièce X.",
        "is_terminal": True,
        "termination_type": "resolution",
        "choices": [],
        "attachments": [],
    }
    base.update(kw)
    return base


def _question(key, choices, **kw):
    base = {
        "node_key": key,
        "step_type": "question",
        "title": key,
        "message": f"Question {key} ?",
        "is_terminal": False,
        "choices": choices,
        "attachments": [],
    }
    base.update(kw)
    return base


def _valid_tree():
    return _draft([
        _question("root", [
            {"label": "Oui", "value": "oui", "next_node_key": "leaf_ok"},
            {"label": "Non", "value": "non", "next_node_key": "leaf_sav"},
        ]),
        _leaf("leaf_ok"),
        _leaf("leaf_sav", step_type="escalation", termination_type="escalation",
              message="Transmission au SAV."),
    ])


def test_valid_tree_has_no_blocking_issue():
    issues = lint_tree(_valid_tree())
    assert not has_blocking_issues(issues)


def test_missing_root():
    issues = lint_tree(_draft([_leaf("feuille")], root="absent"))
    assert _codes(issues) == {"missing_root"}
    assert has_blocking_issues(issues)


def test_dead_end_when_case_has_no_sub_case_and_is_not_a_leaf():
    issues = lint_tree(_draft([_question("root", [], message="")]))
    assert "dead_end" in _codes(issues)
    assert has_blocking_issues(issues)


def test_unnamed_case_blocks():
    """Le SAV ne rédige aucun texte : seul le NOM du cas est obligatoire."""
    issues = lint_tree(_draft([
        _question("root", [{"label": "Oui", "value": "oui", "next_node_key": "sans_nom"}]),
        _leaf("sans_nom", title="", message="", attachments=[]),
    ]))
    assert "unnamed_case" in _codes(issues)
    assert has_blocking_issues(issues)


def test_broken_next():
    issues = lint_tree(_draft([
        _question("root", [{"label": "Oui", "value": "oui", "next_node_key": "nulle_part"}]),
    ]))
    assert "broken_next" in _codes(issues)


def test_orphan_node():
    draft = _valid_tree()
    draft["nodes"].append(_leaf("perdu"))
    issues = lint_tree(draft)
    assert "orphan_node" in _codes(issues)


def test_duplicate_choice_labels_and_values():
    issues = lint_tree(_draft([
        _question("root", [
            {"label": "Oui", "value": "v1", "next_node_key": "leaf_ok"},
            {"label": "oui", "value": "v1", "next_node_key": "leaf_ok"},
        ]),
        _leaf("leaf_ok"),
    ]))
    assert {"duplicate_choice_label", "duplicate_choice_value"} <= _codes(issues)


def test_solution_without_notice_is_only_advice():
    """Une solution nommée mais sans notice reste publiable : LIA donnera l'intitulé."""
    draft = _valid_tree()
    for n in draft["nodes"]:
        if n["node_key"] == "leaf_ok":
            n["message"] = ""
            n["attachments"] = []
    issues = lint_tree(draft)
    assert "leaf_without_notice" in _codes(issues)
    assert not has_blocking_issues(issues)


def test_solution_with_notice_has_no_advice():
    draft = _valid_tree()
    for n in draft["nodes"]:
        if n["node_key"] == "leaf_ok":
            n["message"] = ""
            n["attachments"] = [{"document_id": 1, "page_start": 3}]
    issues = lint_tree(draft)
    assert "leaf_without_notice" not in _codes(issues)


def test_terminal_with_choices():
    draft = _valid_tree()
    for n in draft["nodes"]:
        if n["node_key"] == "leaf_ok":
            n["choices"] = [{"label": "Oups", "value": "x", "next_node_key": "root"}]
    issues = lint_tree(draft)
    assert "terminal_with_choices" in _codes(issues)


def test_no_escalation_branch_is_warning_only():
    issues = lint_tree(_draft([
        _question("root", [{"label": "Oui", "value": "oui", "next_node_key": "leaf_ok"}]),
        _leaf("leaf_ok"),
    ]))
    assert "no_escalation_branch" in _codes(issues)
    assert not has_blocking_issues(issues)
