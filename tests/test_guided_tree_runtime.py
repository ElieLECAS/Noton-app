"""Traversée pure des snapshots d'arbres SAV + compatibilité de périmètre (sans DB)."""
from app.services.authored_tree_service import (
    FEEDBACK_NO_VALUE,
    FEEDBACK_YES_VALUE,
    breadcrumb_from_path,
    get_snapshot_node,
    resolve_next_key,
    step_payload_from_node,
    visible_choices,
)
from app.services.guided_entry_index_service import normalize_text, perimeter_compatible


def _snapshot():
    return {
        "title": "Volet bloqué",
        "flow_kind": "diagnostic",
        "root_node_key": "root",
        "version": 2,
        "nodes": {
            "root": {
                "node_key": "root",
                "step_type": "question",
                "title": "Blocage",
                "message": "Le volet est-il bloqué complètement ?",
                "is_terminal": False,
                "choices": [
                    {"label": "Complètement", "value": "total", "next_node_key": "pvc_branch"},
                    {"label": "Partiellement", "value": "partiel", "next_node_key": "leaf"},
                ],
                "attachments": [],
            },
            "pvc_branch": {
                "node_key": "pvc_branch",
                "step_type": "instruction",
                "title": "Vérif PVC",
                "message": "Vérifier la lame finale PVC.",
                "is_terminal": False,
                "perimeter_condition": {"materials": ["pvc"]},
                "choices": [{"label": "Fait", "value": "fait", "next_node_key": "leaf"}],
                "attachments": [],
            },
            "leaf": {
                "node_key": "leaf",
                "step_type": "diagnostic",
                "title": "Diagnostic",
                "message": "Régler la butée haute.",
                "is_terminal": True,
                "termination_type": "resolution",
                "choices": [],
                "attachments": [{"document_id": 4, "document_title": "Notice", "page_start": 12}],
            },
        },
    }


def test_resolve_next_key_known_choice():
    node = get_snapshot_node(_snapshot(), "root")
    assert resolve_next_key(node, "total") == "pvc_branch"


def test_resolve_next_key_unknown_choice_multi():
    node = get_snapshot_node(_snapshot(), "root")
    assert resolve_next_key(node, "inconnu") is None


def test_resolve_next_key_linear_node_follows_single_transition():
    node = get_snapshot_node(_snapshot(), "pvc_branch")
    assert resolve_next_key(node, "nimporte_quoi") == "leaf"


def test_visible_choices_filters_perimeter_incompatible_branch():
    snap = _snapshot()
    node = get_snapshot_node(snap, "root")
    all_choices = visible_choices(node, snap, perimeter=None)
    assert {c["value"] for c in all_choices} == {"total", "partiel"}
    alu_choices = visible_choices(node, snap, perimeter={"materials": ["alu"]})
    assert {c["value"] for c in alu_choices} == {"partiel"}
    pvc_choices = visible_choices(node, snap, perimeter={"materials": ["pvc"]})
    assert {c["value"] for c in pvc_choices} == {"total", "partiel"}


def test_resolution_leaf_awaits_feedback():
    snap = _snapshot()
    node = get_snapshot_node(snap, "leaf")
    payload = step_payload_from_node(node, snap, awaiting_feedback=True)
    values = {c["value"] for c in payload["choices"]}
    assert values == {FEEDBACK_YES_VALUE, FEEDBACK_NO_VALUE}
    assert payload["is_terminal"] is False  # la session attend le feedback
    assert payload["attachments"][0]["document_id"] == 4


def test_terminal_payload_when_feedback_done():
    snap = _snapshot()
    node = get_snapshot_node(snap, "leaf")
    payload = step_payload_from_node(node, snap, awaiting_feedback=False)
    assert payload["is_terminal"] is True
    assert payload["choices"] == []


def test_breadcrumb_uses_answer_labels():
    path = [
        {"node_key": "root", "user_selection": {"label": "Complètement"}},
        {"node_key": "pvc_branch", "user_selection": {"label": "Fait"}},
        {"node_key": "leaf", "user_selection": None},
    ]
    assert breadcrumb_from_path(path) == ["Complètement", "Fait"]


def test_normalize_text_strips_accents_and_case():
    assert normalize_text("  Ça   FROTTE éà ") == "ca frotte ea"


def test_perimeter_compatible_rules():
    # Pas de filtres ou pas de périmètre → compatible.
    assert perimeter_compatible(None, {"materials": ["pvc"]})
    assert perimeter_compatible({"materials": ["pvc"]}, None)
    # Axe renseigné des deux côtés → intersection requise.
    assert perimeter_compatible({"materials": ["pvc"]}, {"materials": ["PVC"]})
    assert not perimeter_compatible({"materials": ["pvc"]}, {"materials": ["alu"]})
    # Axe absent d'un côté → sans contrainte.
    assert perimeter_compatible({"materials": ["pvc"]}, {"proferm_gammes": ["perform"]})
