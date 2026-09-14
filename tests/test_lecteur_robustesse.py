"""Robustesse du lecteur — les trois défauts que la campagne du 14/09 a fait apparaître.

Aucun de ces tests ne porte sur la façon de LIRE une page : le lecteur reçoit le PNG et la
question, sans règle ajoutée. Ils portent sur ce qui entoure la lecture et qui, mesuré,
faisait perdre des réponses pourtant correctes :

  1. **contrat de sortie** — une page qui ne porte pas la réponse le dit (``present``), et
     une sortie mal formée devient une erreur explicite plutôt qu'une page vide ;
  2. **blocs machine égarés** — le modèle émet parfois ``<sources>`` dans le canal des
     appels d'outils au lieu du texte (4 tours sur 62) ; l'orchestrateur répondait « outil
     inconnu », le bloc était perdu et l'interface affichait 20 pages de sources au lieu
     d'une, quand ce n'était pas une réponse vide ;
  3. **citations recalées et round de contrôle** — le modèle cite le numéro IMPRIMÉ sur la
     planche (décalé de −3 sur ce dossier, 51,6 % de citations fausses), et une reprise de
     contrôle pouvait rendre « Voici les corrections nécessaires… » à la place de la
     réponse. Les pages déclarées sont donc recalées sur celles réellement lues, et une
     reprise dégénérée rend la main au brouillon.
"""
from __future__ import annotations

import pytest

from app.services.page_reader_service import (
    PageReading,
    answer_number,
    parse_page_reading,
)
from app.services.reader_agent_service import ReadDocument, ReaderLoop, split_leaked_blocks


# ---------------------------------------------------------------------------
# 1. Contrat de lecture
# ---------------------------------------------------------------------------


def test_le_contrat_localise_le_repere_avant_de_choisir_la_valeur():
    """« repere » et « valeurs_du_repere » rendent l'erreur visible : une valeur juste
    attribuée au mauvais repère ne peut plus passer pour une lecture correcte."""
    reading = parse_page_reading(
        '{"type_page": "planche cotée", "contenu": "…", "convention": "Cotation en bleu = '
        'épaisseur vitrage", "repere": "Parclose 2636", "valeurs_du_repere": '
        '[{"valeur": "30", "couleur": "bleu"}, {"valeur": "27", "couleur": "noir"}], '
        '"present": true, "ambigu": false, "reponse": "30", "citations": ["Parclose 2636", '
        '"30"], "score": 9}',
        document_id=438,
        page_no=8,
    )
    assert reading.error is None
    assert reading.answer == "30"
    assert reading.score == 9
    assert reading.ok


def test_present_false_vaut_absence():
    reading = parse_page_reading(
        '{"present": false, "reponse": "", "contenu": "autre chose", "score": 2}',
        document_id=1, page_no=1,
    )
    assert reading.absent is True
    assert reading.ok is False
    assert reading.usable is True, "la page reste exploitable par son contenu"


def test_l_ancien_contrat_absent_reste_lu():
    """Un modèle qui rend l'ancienne clé ne doit pas faire perdre la page."""
    reading = parse_page_reading('{"absent": true, "reponse": ""}', document_id=1, page_no=1)
    assert reading.absent is True


def test_sans_cle_de_presence_la_page_est_presente():
    reading = parse_page_reading('{"reponse": "30", "contenu": "x"}', document_id=1, page_no=1)
    assert reading.absent is False
    assert reading.answer == "30"


def test_un_score_hors_bornes_est_ramene_dans_l_echelle():
    assert parse_page_reading('{"score": 99}', document_id=1, page_no=1).score == 10
    assert parse_page_reading('{"score": -4}', document_id=1, page_no=1).score == 0
    assert parse_page_reading('{"score": "sept"}', document_id=1, page_no=1).score == 0


def test_answer_number_ne_vote_que_sur_du_chiffre():
    assert answer_number("30 mm") == 30.0
    assert answer_number("41,5 mm") == 41.5
    assert answer_number("clipper la parclose") is None
    assert answer_number("") is None


# ---------------------------------------------------------------------------
# 3. Bloc machine égaré dans le canal des appels d'outils
# ---------------------------------------------------------------------------


def _appel(nom: str, args: str = "{}") -> dict:
    return {"id": "abc123456", "function": {"name": nom, "arguments": args}}


def test_un_bloc_sources_en_nom_d_outil_est_recupere():
    calls, fuite = split_leaked_blocks(
        [_appel('<sources>{"used":[{"doc_id":438,"pages":[5]}]}</sources>')]
    )
    assert calls == []
    assert "doc_id" in fuite and fuite.startswith("<sources>")


def test_un_bloc_evidence_dans_les_arguments_est_recupere():
    calls, fuite = split_leaked_blocks(
        [_appel("repondre", '<evidence>["Parclose 2636", "30"]</evidence>')]
    )
    assert calls == []
    assert "<evidence>" in fuite


def test_les_vrais_appels_traversent_intacts():
    calls, fuite = split_leaked_blocks(
        [_appel("chercher_code", '{"code": "76373"}'), _appel("<sources>{}</sources>")]
    )
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "chercher_code"
    assert "<sources>" in fuite


def test_aucun_appel_aucune_fuite():
    assert split_leaked_blocks([]) == ([], "")


# ---------------------------------------------------------------------------
# 4. Recalage des pages citées
# ---------------------------------------------------------------------------


def _loop_avec_pages(pages: set) -> ReaderLoop:
    loop = ReaderLoop(messages=[], model="m", stream_fn=None, tools=[])
    loop.read_documents = {
        438: ReadDocument(document_id=438, index=1, title="Dossier", pages=set(pages))
    }
    return loop


class _FiltreFactice:
    def __init__(self, used):
        self.used_documents = used


def test_une_page_imprimee_est_recalee_sur_les_pages_lues():
    """Le modèle cite « page 5 » (le cartouche de la planche) ; la page lue est la 8."""
    loop = _loop_avec_pages({6, 8})
    loop.source_filter = _FiltreFactice([{"doc_id": 438, "pages": [5]}])
    assert loop.used_pages_by_index() == {1: [6, 8]}


def test_une_page_declaree_qui_existe_est_conservee():
    loop = _loop_avec_pages({6, 8})
    loop.source_filter = _FiltreFactice([{"doc_id": 438, "pages": [8]}])
    assert loop.used_pages_by_index() == {1: [8]}


def test_le_tri_ne_garde_que_les_pages_reellement_lues():
    loop = _loop_avec_pages({6, 8})
    loop.source_filter = _FiltreFactice([{"doc_id": 438, "pages": [8, 5, 99]}])
    assert loop.used_pages_by_index() == {1: [8]}


def test_sans_page_lue_la_declaration_est_conservee_telle_quelle():
    """Un document rencontré par un outil n'a pas de page lue : rien à recaler."""
    loop = _loop_avec_pages(set())
    loop.source_filter = _FiltreFactice([{"doc_id": 438, "pages": [12]}])
    assert loop.used_pages_by_index() == {1: [12]}


# ---------------------------------------------------------------------------
# 6. Un round de contrôle ne doit jamais détruire la réponse
# ---------------------------------------------------------------------------


from app.services.reader_agent_service import _is_degenerate  # noqa: E402


def test_une_reprise_vide_est_degeneree():
    assert _is_degenerate("", "Une réponse complète de plusieurs phrases." * 3)


def test_une_reprise_qui_commente_le_controle_est_degeneree():
    """Cas mesuré le 14/09 : « Voici les corrections nécessaires pour le bloc `. »"""
    assert _is_degenerate(
        "Voici les corrections nécessaires pour le bloc `.", "x" * 300
    )
    assert _is_degenerate("**Réponse corrigée** :", "x" * 300)


def test_un_brouillon_substantiel_reduit_a_un_moignon_est_degenere():
    assert _is_degenerate("26 mm", "x" * 400)


def test_une_reponse_ponctuelle_courte_reste_legitime():
    """« 30 mm » est une réponse complète : le seuil de longueur ne vaut que face à un
    brouillon substantiel, sinon on jetterait les bonnes réponses brèves."""
    assert not _is_degenerate("30 mm.", "27 mm.")


def test_une_vraie_correction_passe():
    assert not _is_degenerate("y" * 250, "x" * 300)


def test_sans_brouillon_rien_n_est_degenere():
    """Premier tour : il n'y a rien à préserver."""
    assert not _is_degenerate("", "")
