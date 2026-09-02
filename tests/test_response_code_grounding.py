"""B7a (plan boucle agentique 2026-07-29) — codes produits dans le grounding programmatique.

Cas réel du 29/07 (espace « Coulissant Alu ») : le générateur répond « la crémone 4 points
est TGY3710 » alors que le contexte packé ne contient que TGY3702. Le juge LLM l'avait vu
(« Remarques ») mais le contrôle programmatique, aveugle aux codes produits, n'avait rien
signalé — et le verdict n'était de toute façon pas actionnable. Ces tests verrouillent :
un code inventé devient une affirmation non étayée PROGRAMMATIQUE, qu'aucun juge
complaisant ni aucun judge_suspect ne peut blanchir.
"""
from __future__ import annotations

import pytest

import app.services.response_verification_service as rvs
from app.services.response_verification_service import (
    check_reference_grounding,
    extract_response_reference_codes,
)

CONTEXT_TGY = (
    "DOCUMENT 1 : SOLEAL-GY-55-Catalogue-conception\n"
    "[page 66 — ★ page retrouvée par la recherche]\n"
    "Fermeture 3 points TGY3702, rallonge 4eme point TGY3704, vis TGY3723.\n"
)


# ---------------------------------------------------------------------------
# Extraction des codes côté réponse
# ---------------------------------------------------------------------------


def test_extracts_alphanumeric_codes():
    codes = extract_response_reference_codes(
        "La crémone 4 points est TGY3710, à fixer avec 6 vis TGY3723."
    )
    assert codes == ["TGY3710", "TGY3723"]


def test_pure_numeric_codes_are_ignored():
    """« page 111 », « 6111 », quantités et millésimes sont indiscernables d'un nombre
    quelconque dans un texte GÉNÉRÉ : jamais d'accusation à tort sur du numérique pur."""
    codes = extract_response_reference_codes(
        "Voir page 111 du catalogue 5746, édition 092021, profil 6111."
    )
    assert codes == []


def test_mixed_forms_are_supported():
    codes = extract_response_reference_codes("Le seuil 9F67 en RAL9016.")
    assert "9F67" in codes
    assert "RAL9016" in codes


def test_deduplication_and_case():
    codes = extract_response_reference_codes("tgy3702 puis TGY3702 encore")
    assert codes == ["TGY3702"]


# ---------------------------------------------------------------------------
# Présence littérale dans le contexte
# ---------------------------------------------------------------------------


def test_present_codes_are_supported():
    assert check_reference_grounding("Associer TGY3702 et TGY3704.", CONTEXT_TGY) == []


def test_invented_code_is_flagged():
    assert check_reference_grounding(
        "La crémone 4 points est TGY3710.", CONTEXT_TGY
    ) == ["TGY3710"]


def test_word_boundaries_prevent_prefix_match():
    """« TGY370 » ne doit pas être validé parce que « TGY3702 » contient ce préfixe."""
    assert check_reference_grounding("Utiliser TGY370.", CONTEXT_TGY) == ["TGY370"]


def test_case_insensitive_context_match():
    assert check_reference_grounding("Utiliser tgy3702.", CONTEXT_TGY) == []
