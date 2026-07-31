"""La désignation commerciale doit survivre à la reformulation client des boutons.

Cas réellement observé : deux produits d'un même arbre TECHNAL reformulés en
« Coulissant à deux vantaux » et « Coulissant à un ou deux vantaux » — indistinguables,
parce que LUMEAL GA et SOLEAL GY 55 avaient disparu des libellés.
"""

from __future__ import annotations

import pytest

from app.services.guided_flow_service import designations, keep_designation


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Coulissant LUMEAL GA", "LUMEAL GA"),
        ("Coulissant SOLEAL GY 55", "SOLEAL GY 55"),
        ("Fenêtre ou porte-fenêtre SOLEAL FY 55", "SOLEAL FY 55"),
        ("Serrure motorisée Eneo CC (clavier à code)", "Eneo CC"),
        ("Moteur autonome solaire Oximo 40 WF RTS", "Oximo 40 WF RTS"),
        ("Chargeur de batterie MVOS-ALI", "MVOS-ALI"),
    ],
)
def test_designation_reperee(title, expected):
    assert expected in designations(title)


@pytest.mark.parametrize(
    "title",
    [
        "Le pêne n'engage pas et la porte s'ouvre un peu",
        "Le vantail est dur à faire glisser",
        "Rien de tout ça, ou le coulissant reste dur",
    ],
)
def test_intitule_sans_reference_ne_declenche_rien(title):
    assert designations(title) == []
    assert keep_designation(title, "Le volet ne bouge pas") == "Le volet ne bouge pas"


def test_designation_reinjectee_quand_la_redaction_la_perd():
    assert (
        keep_designation("Coulissant SOLEAL GY 55", "Coulissant à deux vantaux")
        == "SOLEAL GY 55 — Coulissant à deux vantaux"
    )


def test_designation_deja_presente_laisse_le_libelle_intact():
    label = "SOLEAL GY 55 — coulissant à deux vantaux"
    assert keep_designation("Coulissant SOLEAL GY 55", label) == label


def test_comparaison_insensible_a_la_casse_et_aux_accents():
    assert keep_designation("Serrure Eneo CC", "serrure eneo cc à code") == "serrure eneo cc à code"


def test_libelle_reste_borne_a_90_caracteres():
    out = keep_designation("Coulissant SOLEAL GY 55", "x" * 120)
    assert len(out) <= 90


def test_deux_produits_voisins_restent_distinguables():
    """Le vrai critère : après passage du garde-fou, les libellés diffèrent."""
    a = keep_designation("Coulissant LUMEAL GA", "Coulissant à deux vantaux")
    b = keep_designation("Coulissant SOLEAL GY 55", "Coulissant à un ou deux vantaux")
    assert a != b
    assert "LUMEAL GA" in a and "SOLEAL GY 55" in b
