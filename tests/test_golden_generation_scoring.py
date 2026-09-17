"""Le scoring du golden de génération — fonctions pures, testées sans réseau ni DB.

Un runner d'évaluation qui se trompe est pire qu'aucun runner : il fait basculer une
décision d'architecture sur du bruit. Ces tests verrouillent les deux pièges du domaine :
  * les frontières numériques (« 30 » ne doit jamais être validé par « 130 » ni « 30.5 »,
    sinon la confusion de cote sur une planche passe inaperçue) ;
  * les séparateurs de milliers (« 3 500 mm » dans le document ≡ « 3500 » dans le golden).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.scripts.eval_golden_generation import (
    aggregate,
    expected_pairs,
    normalize_text,
    pages_from_sources,
    pages_packed,
    pages_seen_as_image,
    regex_any_matches,
    score_answer,
    stability,
    value_present,
)

GOLDEN = Path("tests/fixtures/golden/space29_perform76_generation.json")


# ---------------------------------------------------------------------------
# value_present
# ---------------------------------------------------------------------------


def test_une_cote_ne_matche_pas_un_nombre_plus_long():
    assert value_present("30", "l'épaisseur est de 30 mm")
    assert not value_present("30", "la largeur est de 130 mm")
    assert not value_present("30", "compter 30.5 mm de jeu")
    assert not value_present("16", "le profil 7616 grainé")


def test_virgule_et_point_decimaux_sont_equivalents():
    assert value_present("41.5", "la cote noire vaut 41,5 mm")
    assert value_present("1.5", "renfort de 1.5 mm")
    assert value_present("26.5", "26,5")


def test_separateur_de_milliers_du_document():
    # Le catalogue écrit « 3 500 mm » (espace fine), le golden porte « 3500 ».
    assert value_present("3500", "largeur maxi 3 500 mm")
    assert value_present("4200", "H 2400 x L 4 200")
    assert not value_present("3500", "largeur 13 500 mm")


def test_code_produit_avec_frontieres_alphanumeriques():
    assert value_present("76373", "le meneau dormant 124 mm est le 76373")
    assert not value_present("76373", "référence 763730")
    assert value_present("NT1947", "patte de pose nt1947")
    assert not value_present("NT1947", "patte NT19470")


def test_texte_insensible_aux_accents_et_a_la_casse():
    assert value_present("post-extrudé", "avec un joint POST-EXTRUDE ou équivalent")
    assert value_present("ivoire", "teinte Ivoire proche 9001")


# ---------------------------------------------------------------------------
# score_answer
# ---------------------------------------------------------------------------


def test_valeur_juste_sans_piege():
    attendu = {"type": "valeur", "valeurs": ["16"], "mode": "tous", "interdits": ["41.5"]}
    res = score_answer(attendu, "La parclose 2452 accepte un vitrage de 16 mm.")
    assert res["verdict"] == "juste"
    assert res["trouves"] == ["16"]


def test_valeur_avec_piege_est_ambigue_pas_juste():
    """La bonne valeur ET la valeur piège : signalé pour relecture, jamais compté juste."""
    attendu = {"type": "valeur", "valeurs": ["30"], "mode": "tous", "interdits": ["27"]}
    res = score_answer(attendu, "Parclose 2636 : 30 mm (la cote 27 mm est la hauteur).")
    assert res["verdict"] == "ambigu"
    assert res["interdits_presents"] == ["27"]


def test_valeur_manquante_est_fausse():
    attendu = {"type": "valeur", "valeurs": ["30"], "mode": "tous", "interdits": ["27"]}
    res = score_answer(attendu, "Parclose 2636 : 27 mm.")
    assert res["verdict"] == "faux"
    assert res["manquants"] == ["30"]


def test_mode_un_accepte_une_seule_des_valeurs():
    attendu = {
        "type": "valeur",
        "valeurs": ["220", "240", "263", "283"],
        "mode": "un",
        "interdits": ["513"],
    }
    assert score_answer(attendu, "axe de poignée à 240 mm")["verdict"] == "juste"
    assert score_answer(attendu, "axe de poignée à 513 mm")["verdict"] == "faux"


def test_regex_any_qualifie_la_valeur_en_condition_et():
    """« 4 » seul ne veut rien dire : la regex « 6 pans » qualifie la valeur."""
    attendu = {
        "type": "valeur", "valeurs": ["4", "2"], "mode": "tous",
        "interdits": [], "regex_any": ["6 pans"],
    }
    assert score_answer(attendu, "clé 6 pans de 4 mm (+ ou - 2 mm)")["verdict"] == "juste"
    assert score_answer(attendu, "il faut 4 vis et 2 cales")["verdict"] == "faux"


def test_liste_exige_un_minimum_d_elements():
    attendu = {
        "type": "liste",
        "elements": ["compensation bois", "fond de joint", "silicone", "cale latérale"],
        "min": 3,
    }
    ok = score_answer(attendu, "Compensation bois, fond de joint et silicone.")
    assert ok["verdict"] == "juste"
    ko = score_answer(attendu, "Du silicone et rien d'autre.")
    assert ko["verdict"] == "faux"
    assert "fond de joint" in ko["manquants"]


def test_abstention_ok_quand_le_modele_dit_que_ca_n_existe_pas():
    attendu = {
        "type": "abstention",
        "regex_any": ["n['’]existe", "ne (figure|mentionne|précise|trouve)"],
        "interdits": [],
    }
    assert score_answer(attendu, "La parclose 2637 n'existe pas dans la planche.")["verdict"] == "abstention_ok"
    assert score_answer(attendu, "La parclose 2637 accepte 31 mm.")["verdict"] == "abstention_ko"


def test_texte_passe_par_regex():
    attendu = {"type": "texte", "regex_any": ["fond (de )?feuillure"]}
    assert score_answer(attendu, "FFO = hauteur d'axe au fond de feuillure")["verdict"] == "juste"
    assert score_answer(attendu, "FFO est une abréviation interne.")["verdict"] == "faux"


def test_score_ne_leve_jamais_sur_une_reponse_vide_ou_un_attendu_partiel():
    assert score_answer({}, "")["verdict"] == "faux"
    assert score_answer({"type": "texte"}, "")["verdict"] == "faux"
    assert score_answer({"type": "abstention", "regex_any": ["[["]}, "x")["verdict"] == "abstention_ko"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def test_pages_citees_preferent_used_pages():
    sources = [{"document_id": 438, "pages": [6, 7, 8], "used_pages": [8]}]
    assert pages_from_sources(sources) == [(438, 8)]


def test_pages_citees_retombent_sur_les_pages_packees():
    sources = [{"document_id": 438, "pages": [8, 9]}]
    assert pages_from_sources(sources) == [(438, 8), (438, 9)]


def test_pages_packees_viennent_du_manifeste():
    trace = {"packed_documents": [{"document_id": 438, "pages": [6, 8, 12]}]}
    assert set(pages_packed(trace)) == {(438, 6), (438, 8), (438, 12)}


def test_pages_vues_ne_comptent_que_les_images():
    """Le manifeste liste toutes les pages packées ; seules les IMAGES ont été vues."""
    trace = {
        "packed_documents": [{"document_id": 438, "pages": [1, 2, 3, 4, 5, 6, 8, 12]}],
        "images": [{"document_id": 438, "page_no": 8}],
    }
    assert set(pages_seen_as_image(trace)) == {(438, 8)}
    assert (438, 12) not in set(pages_seen_as_image(trace))


def test_expected_pairs_ignore_les_entrees_malformees():
    assert expected_pairs([{"document_id": 438, "pages": [8, "x"]}]) == [(438, 8)]


# ---------------------------------------------------------------------------
# Agrégation
# ---------------------------------------------------------------------------


def _row(**kw):
    base = {
        "verdict": "juste", "difficulte": "simple", "tags": ["parclose"],
        "page_citee": True, "doc_cite": True, "page_decalee": False,
        "page_vue": True, "page_packee": True, "first_token_s": 1.0, "total_s": 2.0,
        "tool_calls": 0, "control_rounds": 0, "stopped_by": "final", "degraded": None,
        "verification_action": "passed", "tool_names": [], "scope_card": False,
    }
    base.update(kw)
    return base


def test_agregat_compte_la_reussite_abstention_incluse():
    rows = [_row(), _row(verdict="abstention_ok"), _row(verdict="faux"), _row(verdict="ambigu")]
    agg = aggregate(rows)
    assert agg["n"] == 4
    assert agg["reussite_pct"] == 50.0
    assert agg["verdicts"]["ambigu"] == 1
    assert agg["page_citee_pct"] == 100.0


def test_stabilite_signale_un_verdict_qui_change():
    runs = {
        "g1": [_row(answer="30 mm"), _row(verdict="faux", answer="27 mm")],
        "g2": [_row(answer="a"), _row(answer="a")],
    }
    stab = stability(runs)
    assert stab["n_questions_instables"] == 1
    assert stab["details"][0]["id"] == "g1"


# ---------------------------------------------------------------------------
# Le golden lui-même
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden absent")
def test_golden_est_bien_forme():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    ids = [e["id"] for e in data["entries"]]
    assert len(ids) == len(set(ids)), "identifiants dupliqués"
    for entry in data["entries"]:
        assert entry["question"].strip()
        assert entry["preuve"], f"{entry['id']} sans page de preuve"
        assert expected_pairs(entry["preuve"]), f"{entry['id']} : preuve illisible"
        attendu = entry["attendu"]
        assert attendu["type"] in ("valeur", "liste", "texte", "abstention")
        if attendu["type"] == "valeur":
            assert attendu.get("valeurs"), f"{entry['id']} sans valeur attendue"
        if attendu["type"] == "liste":
            assert attendu.get("elements") and attendu.get("min")
        if attendu["type"] in ("texte", "abstention"):
            assert attendu.get("regex_any"), f"{entry['id']} sans regex"
        for pattern in attendu.get("regex_any") or []:
            regex_any_matches([pattern], "test")  # ne doit pas lever


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden absent")
def test_la_verite_du_golden_valide_son_propre_attendu():
    """Le champ ``verite`` (ce qui est lu sur la page) doit passer son propre score.

    C'est le garde-fou contre un attendu mal écrit : si la phrase de vérité ne satisfait
    pas le critère, c'est le critère qui est faux, pas le modèle.
    """
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    faux_negatifs = []
    for entry in data["entries"]:
        attendu = entry["attendu"]
        if attendu["type"] == "abstention":
            continue  # la vérité décrit l'absence, elle n'est pas une réponse modèle
        res = score_answer(attendu, entry["verite"])
        if res["verdict"] not in ("juste", "ambigu"):
            faux_negatifs.append((entry["id"], res["manquants"]))
    assert not faux_negatifs, f"attendus non satisfaits par leur propre vérité : {faux_negatifs}"


def test_normalize_text_est_idempotent():
    once = normalize_text("  Épaisseur  3 500 mm  ")
    assert normalize_text(once) == once


def test_une_valeur_en_fin_de_phrase_est_trouvee():
    """Le point final ne doit pas bloquer le match — bug attrapé par l'auto-validation."""
    assert value_present("76373", "La référence est 76373.")
    assert value_present("6137", "patte NT1947, appui 6137.")
    assert value_present("76281", "Ouvrant droit 76281, ouvrant galbé 76275.")
    # mais une suite décimale invalide toujours
    assert not value_present("30", "30.5 mm")
    assert not value_present("5", "1.5 mm")


def test_un_motif_regex_garde_ses_classes_de_caracteres():
    r"""`[\s\S]` ne doit pas etre mis en minuscules (il deviendrait `[\s\s]`)."""
    assert regex_any_matches([r"compensation bois[\s\S]{0,400}mise à niveau"],
                             "V1 : compensation bois ; V2 : mise à niveau bois.")
    assert regex_any_matches([r"uniquement[\s\S]{0,60}ouvrants?"],
                             "Montage uniquement compatible avec les ouvrants.")


def test_une_page_packee_sans_image_n_est_pas_vue():
    """Le pack de lecture a disparu : une page n'est vue que si son PNG est joint."""
    trace = {
        "packed_documents": [{"document_id": 438, "pages": [6, 8]}],
        "images": [{"document_id": 438, "page_no": 8}],
    }
    vues = set(pages_seen_as_image(trace))
    assert (438, 8) in vues
    assert (438, 6) not in vues, "packée n'est pas vue"
