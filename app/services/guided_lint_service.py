"""Lint des arbres SAV — bloquant à la publication, informatif dans l'éditeur.

Opère sur la forme « draft » (dict) produite/consommée par guided_authoring_service :
{meta: {root_node_key, ...}, nodes: [{node_key, step_type, title, message, is_terminal,
termination_type, choices: [{label, value, next_node_key}], attachments: [...]}]}.

Codes émis (severity: error = bloque la publication, warning = informatif) :
- dead_end              nœud non terminal sans choix valide
- orphan_node           nœud injoignable depuis root
- missing_root          root_node_key absent des nœuds
- broken_next           choix pointant vers un node_key inexistant
- duplicate_choice_label  deux choix du même nœud portent le même libellé
- duplicate_choice_value  deux choix du même nœud portent la même valeur
- leaf_without_content  feuille sans pièce jointe NI message (rien à montrer au client)
- no_escalation_branch  aucun chemin ne mène à une escalade (warning)
- depth_warning         profondeur > 12 (warning)
- terminal_with_choices feuille qui porte encore des choix
- single_choice         question à une seule réponse : étape inutile (warning)
- empty_message         nœud non terminal sans question/message
"""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

MAX_DEPTH_WARNING = 12


class LintIssue(BaseModel):
    severity: str  # error | warning
    code: str
    node_key: str = ""
    message: str


def _node_map(draft: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(n.get("node_key") or ""): n for n in (draft.get("nodes") or [])}


def _name(node: Dict[str, Any], key: str) -> str:
    """Nom lisible d'une étape pour les messages : titre, sinon début du texte client.

    Jamais l'identifiant technique — ces messages sont lus par le SAV, pas par un dev.
    """
    title = str(node.get("title") or "").strip()
    if title:
        return title
    message = str(node.get("message") or "").strip()
    if message:
        return (message[:50] + "…") if len(message) > 50 else message
    return "étape sans nom"


def lint_tree(draft: Dict[str, Any]) -> List[LintIssue]:
    issues: List[LintIssue] = []
    nodes = _node_map(draft)
    meta = draft.get("meta") or {}
    root_key = str(meta.get("root_node_key") or "root")

    if root_key not in nodes:
        issues.append(
            LintIssue(
                severity="error",
                code="missing_root",
                node_key=root_key,
                message="Ce guide n'a pas de première question.",
            )
        )
        return issues  # tout le reste est inévaluable sans racine

    has_escalation = False
    for key, node in nodes.items():
        choices = [c for c in (node.get("choices") or []) if isinstance(c, dict)]
        is_terminal = bool(node.get("is_terminal"))
        step_type = str(node.get("step_type") or "question")

        if step_type == "escalation" or node.get("termination_type") == "escalation":
            has_escalation = True

        if is_terminal:
            if choices:
                issues.append(
                    LintIssue(
                        severity="error",
                        code="terminal_with_choices",
                        node_key=key,
                        message=f"« {_name(node, key)} » est marquée comme une fin, mais elle propose encore des réponses au client.",
                    )
                )
            # Une solution n'a besoin QUE d'un nom : LIA rédige l'explication à partir
            # de la notice rattachée. Sans notice, elle se contente de l'intitulé → conseil.
            if not (node.get("attachments") or []) and not str(node.get("message") or "").strip():
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="leaf_without_notice",
                        node_key=key,
                        message=f"« {_name(node, key)} » n'a pas de notice rattachée : LIA ne pourra donner que cet intitulé au client.",
                    )
                )
        else:
            valid_choices = [c for c in choices if str(c.get("next_node_key") or "")]
            if not valid_choices:
                issues.append(
                    LintIssue(
                        severity="error",
                        code="dead_end",
                        node_key=key,
                        message=f"À l'étape « {_name(node, key)} », le client n'a aucune réponse possible : il serait bloqué.",
                    )
                )
            elif len(valid_choices) == 1:
                # Une question à une seule réponse ne trie rien : le client lit une étape
                # de plus pour aboutir au même endroit. Les deux cas doivent fusionner.
                issues.append(
                    LintIssue(
                        severity="warning",
                        code="single_choice",
                        node_key=key,
                        message=(
                            f"« {_name(node, key)} » ne propose qu'une seule réponse : cette étape "
                            "ne fait pas avancer le client. Fusionnez-la avec le cas du dessous."
                        ),
                    )
                )
        # Le SAV ne rédige aucun texte : ce qui doit exister, c'est le NOM du cas
        # (c'est lui que le client cliquera, et qui nomme la solution).
        if not str(node.get("title") or "").strip() and not str(node.get("message") or "").strip():
            issues.append(
                LintIssue(
                    severity="error",
                    code="unnamed_case",
                    node_key=key,
                    message="Un cas n'a pas de nom : le client verrait un bouton vide.",
                )
            )

        seen_labels: set = set()
        seen_values: set = set()
        for c in choices:
            label = str(c.get("label") or "").strip().lower()
            value = str(c.get("value") or "").strip().lower()
            nxt = str(c.get("next_node_key") or "")
            if nxt and nxt not in nodes:
                issues.append(
                    LintIssue(
                        severity="error",
                        code="broken_next",
                        node_key=key,
                        message=f"À l'étape « {_name(node, key)} », la réponse « {c.get('label')} » mène à une étape qui a été supprimée.",
                    )
                )
            if label:
                if label in seen_labels:
                    issues.append(
                        LintIssue(
                            severity="error",
                            code="duplicate_choice_label",
                            node_key=key,
                            message=f"L'étape « {_name(node, key)} » propose deux fois la réponse « {c.get('label')} ».",
                        )
                    )
                seen_labels.add(label)
            if value:
                if value in seen_values:
                    issues.append(
                        LintIssue(
                            severity="error",
                            code="duplicate_choice_value",
                            node_key=key,
                            message=f"L'étape « {_name(node, key)} » a deux réponses techniquement identiques : renommez-en une.",
                        )
                    )
                seen_values.add(value)

    # Parcours depuis la racine : orphelins + profondeur
    reachable: set = set()
    stack = [(root_key, 0)]
    max_depth = 0
    while stack:
        key, depth = stack.pop()
        if key in reachable or key not in nodes:
            continue
        reachable.add(key)
        max_depth = max(max_depth, depth)
        for c in nodes[key].get("choices") or []:
            nxt = str(c.get("next_node_key") or "")
            if nxt and nxt in nodes and nxt not in reachable:
                stack.append((nxt, depth + 1))

    for key in nodes:
        if key not in reachable:
            issues.append(
                LintIssue(
                    severity="error",
                    code="orphan_node",
                    node_key=key,
                    message=f"L'étape « {_name(nodes[key], key)} » n'est jamais atteinte : aucune réponse ne mène jusqu'à elle.",
                )
            )

    if max_depth > MAX_DEPTH_WARNING:
        issues.append(
            LintIssue(
                severity="warning",
                code="depth_warning",
                message=f"Le guide peut enchaîner {max_depth} questions d'affilée : c'est long pour le client.",
            )
        )

    if not has_escalation:
        issues.append(
            LintIssue(
                severity="warning",
                code="no_escalation_branch",
                message="Aucun chemin ne permet de passer la main au SAV : prévoyez une sortie pour les cas non résolus.",
            )
        )

    return issues


def has_blocking_issues(issues: List[LintIssue]) -> bool:
    return any(i.severity == "error" for i in issues)
