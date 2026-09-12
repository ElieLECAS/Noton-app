"""Lecteur de page délégué : parsing du relevé et désambiguïsation par la couleur.

Toutes les fonctions testées ici sont pures — aucun appel modèle, aucune image. Ce qu'elles
protègent a été mesuré le 2026-09-12 sur la planche des parcloses (doc 438 p.8) : le modèle
relève toujours les bonnes valeurs, mais l'étiquette de couleur bascule EN BLOC d'un run à
l'autre. C'est donc le recoupement déterministe, pas le jugement du modèle, qui tranche.
"""
from __future__ import annotations

from app.services.page_reader_service import (
    detect_convention_color,
    normalize_value,
    parse_page_reading,
    resolve_by_color,
    survey_entry_for,
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_reads_survey_convention_and_answer():
    raw = """{"convention": "Cotation en bleu = épaisseur vitrage",
      "releve": [{"repere": "Parclose 2636", "valeurs": [
          {"valeur": "30", "couleur": "bleu"}, {"valeur": "27", "couleur": "noir"}]}],
      "absent": false, "ambigu": false, "reponse": "30",
      "citations": ["Parclose 2636", "30"]}"""
    r = parse_page_reading(raw, document_id=438, page_no=8)
    assert r.ok and r.answer == "30"
    assert r.convention.startswith("Cotation en bleu")
    assert r.survey[0]["repere"] == "Parclose 2636"
    assert [v["valeur"] for v in r.survey[0]["valeurs"]] == ["30", "27"]


def test_parse_tolerates_code_fence_and_loose_values():
    raw = '```json\n{"releve": [{"ref": "2452", "valeurs": ["16", "41.5"]}], "reponse": "16"}\n```'
    r = parse_page_reading(raw, document_id=1, page_no=1)
    assert r.answer == "16"
    assert r.survey[0]["repere"] == "2452"
    assert [v["valeur"] for v in r.survey[0]["valeurs"]] == ["16", "41.5"]


def test_absent_and_ambiguous_always_drop_the_value():
    """Un modèle qui se contredit (absent + valeur) ne doit JAMAIS voir sa valeur retenue :
    c'est sous cette forme qu'une cote inventée par analogie remonterait en texte propre."""
    for flag in ("absent", "ambigu"):
        r = parse_page_reading(
            '{"%s": true, "reponse": "38 mm"}' % flag, document_id=1, page_no=1
        )
        assert r.answer == "" and not r.ok


def test_unparseable_output_is_an_error_not_an_empty_reading():
    r = parse_page_reading("désolé, je ne peux pas lire cette image", document_id=1, page_no=1)
    assert r.error is not None and not r.ok
    r2 = parse_page_reading("", document_id=1, page_no=1)
    assert r2.error is not None


# ---------------------------------------------------------------------------
# Désambiguïsation par la couleur
# ---------------------------------------------------------------------------


def test_detect_convention_color():
    assert detect_convention_color("Cotation en bleu = épaisseur vitrage") == "bleu"
    assert detect_convention_color("Les cotes en ROUGE sont les entraxes") == "rouge"
    assert detect_convention_color("Cotes en millimètres") is None


def test_normalize_value_tolerates_units_and_decimal_comma():
    assert normalize_value("29,5") == normalize_value("29.5") == "29.5"
    assert normalize_value("30 mm") == "30"
    assert normalize_value("") == ""


def test_resolve_by_color_picks_the_single_coloured_value():
    values = [{"valeur": "30", "couleur": "noir"}, {"valeur": "27", "couleur": "bleu"}]
    # Les étiquettes du modèle sont ici INVERSÉES ; seul l'ensemble coloré fait foi.
    value, status = resolve_by_color(values, ["16", "18", "30", "31"])
    assert (value, status) == ("30", "resolved")


def test_resolve_by_color_refuses_when_both_or_none_match():
    values = [{"valeur": "30", "couleur": ""}, {"valeur": "28", "couleur": ""}]
    assert resolve_by_color(values, ["30", "28"])[1] == "ambiguous"
    assert resolve_by_color(values, ["16", "18"])[1] == "ambiguous"
    assert resolve_by_color(values, [])[1] == "unusable"


def test_survey_entry_lookup_is_loose_on_formatting():
    survey = [{"repere": "Parclose 2636", "valeurs": []}, {"repere": "2634", "valeurs": []}]
    assert survey_entry_for(survey, "2636")["repere"] == "Parclose 2636"
    assert survey_entry_for(survey, "parclose-2634")["repere"] == "2634"
    assert survey_entry_for(survey, "9999") is None
    assert survey_entry_for(survey, "") is None
