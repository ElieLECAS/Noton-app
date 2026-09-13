"""Lecteur de page délégué : parsing du relevé rendu par le modèle vision.

Toutes les fonctions testées ici sont pures — aucun appel modèle, aucune image. Ce qu'elles
protègent : une sortie illisible devient une ERREUR explicite et non « rien sur cette page »,
et un « absent » ou « ambigu » ne laisse jamais passer une valeur. C'est ce contrat, et non
une règle ajoutée, qui empêche une valeur devinée de revenir en texte propre.
"""
from __future__ import annotations

from app.services.page_reader_service import (
    parse_page_reading,
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
# Repérage dans le relevé
# ---------------------------------------------------------------------------


def test_survey_entry_lookup_is_loose_on_formatting():
    survey = [{"repere": "Parclose 2636", "valeurs": []}, {"repere": "2634", "valeurs": []}]
    assert survey_entry_for(survey, "2636")["repere"] == "Parclose 2636"
    assert survey_entry_for(survey, "parclose-2634")["repere"] == "2634"
    assert survey_entry_for(survey, "9999") is None
    assert survey_entry_for(survey, "") is None


# ---------------------------------------------------------------------------
# Non-régressions du 2026-09-12 : faux code « FORM76 » et fuite du raisonnement
# ---------------------------------------------------------------------------


def test_gamme_name_is_not_a_reference_code():
    """« PERFORM76 » produisait le code fantôme « FORM76 » (le motif démarrait au milieu du
    mot), introuvable par construction : le contrôle de sortie déclenchait donc une
    correction à CHAQUE réponse de l'espace Perform."""
    from app.services.reference_codes import REF_CODE_RE

    def codes(text):
        return [m.group(0) for m in REF_CODE_RE.finditer(text)]

    assert codes("la gamme PERFORM76 est en PVC") == []
    assert codes("le PROFORM76") == []
    assert "FORM76" not in codes("Dossier technique gamme Perform76 CCV03")
    # Les vraies références restent détectées.
    assert codes("crémone TGY3702 et rallonge TGY3704") == ["TGY3702", "TGY3704"]
    assert codes("dormant 76177 rénovation") == ["76177"]
    assert codes("réf.SL1600") == ["SL1600"] and codes("RAL9016") == ["RAL9016"]


def test_response_control_no_longer_fires_on_a_gamme_name():
    from app.services.response_verification_service import check_reader_output

    res = check_reader_output(
        response_text="Le dormant rénovation de la gamme Perform76 accepte un délignage.",
        evidence_text="[page 7] Dormant rénovation, délignage de l'aile à effectuer sur chantier.",
        question="délignage maxi dormant rénovation",
    )
    assert res["ok"] is True and res["unsupported_claims"] == []


def test_control_round_reasoning_never_reaches_the_user():
    """Le round de contrôle répondait « Voici la vérification et la correction : … » suivi de
    ses constats, et tout partait à l'utilisateur. Seul le contenu déclaré est montré."""
    from app.services.reader_agent_service import extract_declared_answer

    leaked = (
        "Voici la vérification et la correction:\n"
        "Les documents consultés ne mentionnent pas le terme \"FORM76\"…\n"
        "<reponse_finale>Le délignage maxi de l'aile est de 20 mm, à effectuer sur le "
        "chantier.</reponse_finale>"
    )
    out = extract_declared_answer(leaked)
    assert out == "Le délignage maxi de l'aile est de 20 mm, à effectuer sur le chantier."
    assert "vérification" not in out and "FORM76" not in out


def test_declared_answer_falls_back_to_full_text():
    """Pas de balise, ou balise vide → on montre tout : mieux vaut bavard que muet."""
    from app.services.reader_agent_service import extract_declared_answer

    assert extract_declared_answer("Réponse directe.") == "Réponse directe."
    assert extract_declared_answer("X <reponse_finale>  </reponse_finale>").startswith("X ")
    # Balise ouverte non refermée (réponse tronquée par le plafond de tokens).
    assert extract_declared_answer("meta\n<reponse_finale>Début tronqué") == "Début tronqué"
