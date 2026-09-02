"""Élection du document (phase A du retriever) — « dans QUEL document est la réponse ? ».

Chaque test reproduit une pathologie mesurée sur le corpus réel :

  * E1 « preuve détruite » — un document qui place beaucoup de pages doit pouvoir
    gagner sur la distribution, signal que la coupe top_k + le quota par document
    effaçaient avant même que l'élection le voie.
  * E2 « catalogue vs notice » — mais le volume ne doit JAMAIS renverser une meilleure
    page nette (parité avec T1 de test_election_packing).
  * E3 « empilement par défaut » — trois documents qui scorent ne donnent plus trois
    documents dans le contexte : sans preuve, le dominant part seul (c'est la cause de
    la réponse qui recollait des références de gammes différentes).
  * E4 « comparaison légitime » — une question comparative admet bien un second document.
  * E5 « complémentarité d'archétypes » — notice texte + planche muette vue par ColPali.
  * E6 « versions concurrentes » — CC01 et CC02 du même cahier technique ne coexistent
    plus (cotes contradictoires, budget gaspillé).
  * P1/P2 « détenteur du meilleur passage » — le score d'élection agrège par document et
    récompense l'affinité thématique plutôt que la preuve ; les deux pannes mesurées le
    01/09 (LUMINE 55 et couleurs Perform) jetaient la page qui portait la réponse alors
    qu'elle était en TÊTE du pool fusionné.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from sqlmodel import Session

from app.models.document import Document
from app.models.library import Library
from app.config import settings
from app.services.document_election_service import (
    DOMINANCE_RATIO,
    ELECTION_PAGE_CAP,
    ElectedDocument,
    dedupe_versions,
    elect_documents,
    election_candidate_passages,
    format_election_log,
    is_comparative_query,
    normalize_title_family,
    score_documents,
)
from app.services.page_retrieval_service import UnifiedPageHit
from tests.conftest import create_test_user


def _hit(
    doc_id: int,
    page: int,
    rrf: float,
    *,
    sources=("bm25",),
    title: str | None = None,
    colpali: float | None = None,
    bm25: float | None = None,
) -> UnifiedPageHit:
    return UnifiedPageHit(
        document_id=doc_id,
        page_no=page,
        rrf_score=rrf,
        retrieval_sources=list(sources),
        document_title=title or f"Doc {doc_id}",
        colpali_score=colpali,
        bm25_score=bm25,
    )


# ---------------------------------------------------------------------------
# P — L'élection suit les PASSAGES : les détenteurs des trois meilleurs sont élus d'office
# ---------------------------------------------------------------------------


def _pool(spec) -> list[UnifiedPageHit]:
    """Construit un pool depuis (doc_id, titre, [(page, rrf, colpali, bm25), …])."""
    hits = []
    for doc_id, title, pages in spec:
        for page, rrf, colpali, bm25 in pages:
            sources = [s for s, v in (("colpali", colpali), ("bm25", bm25)) if v]
            hits.append(
                _hit(doc_id, page, rrf, sources=tuple(sources), title=title,
                     colpali=colpali, bm25=bm25)
            )
    return hits


def test_p1_cas_lumine_le_detenteur_du_meilleur_passage_devient_dominant():
    """Panne mesurée en prod le 01/09 (requête « performance acoustique LUMINE 55 »).

    Lumine55 détient le MEILLEUR passage du pool (rrf 0,0323, ColPali 0,694 — les deux
    retrievers le classent 1er) et porte SEUL la réponse (40 dB, page 1). Le catalogue
    général, dont la meilleure page n'arrive qu'au 4e rang, gagnait par ses 17 pages
    (×1,30 contre ×1,15) et la phase B jetait ensuite tous les non-élus.

    Désormais l'élection suit les passages : Lumine55 est DOMINANT (passage n°1), le
    catalogue reste élu en complément au titre du volume.
    """
    pool = _pool([
        (424, "2024-04_CATALOGUE-GENERAL_WEB",
         [(14, 0.0299, 0.617, 0.400), (4, 0.0296, 0.594, 0.400), (27, 0.0164, None, 0.900)]
         + [(p, 0.0135 - i * 0.0002, None, 0.200) for i, p in enumerate(range(1, 15))]),
        (439, "Lumine55", [(1, 0.0323, 0.694, 0.600), (2, 0.0301, 0.630, 0.400)]),
        (429, "Lumine65", [(3, 0.0300, 0.643, 0.300), (2, 0.0297, 0.627, 0.400)]),
    ])
    result = elect_documents(None, pool, query_text="performance acoustique LUMINE 55 en dB")

    assert result.elected[0].document_id == 439, "le détenteur du meilleur passage doit mener"
    assert result.elected[0].reason == "top_passage:1"
    assert 424 in result.elected_ids, "le volume reste une preuve : le catalogue est complément"
    assert next(d.reason for d in result.elected if d.document_id == 424) == "best_election_score"


def test_p2_cas_perform_le_depliant_qui_porte_les_couleurs_devient_dominant():
    """Panne mesurée le 01/09 (« couleurs de la gamme Perform »), sens INVERSE de P1.

    Le dossier technique Perform76 place 8 pages parce que tout le document parle de
    Perform — mais AUCUNE ne traite des couleurs. Le dépliant #425 détient le meilleur
    passage (rrf 0,0307, ColPali 0,687) et 3 pages utiles ; il perdait 0,0368 à 0,0385.
    """
    pool = _pool([
        (438, "Dossier technique gamme Perform76",
         [(5, 0.0296, 0.651, 0.300), (4, 0.0291, 0.640, 0.300)]
         + [(p, 0.0180 - i * 0.0005, 0.600, None) for i, p in enumerate(range(6, 12))]),
        (425, "2023-06_DEPLIANT-GENERAL",
         [(3, 0.0307, 0.687, 0.400), (4, 0.0250, 0.610, 0.300), (5, 0.0240, 0.600, None)]),
    ])
    result = elect_documents(None, pool, query_text="couleurs disponibles gamme Perform")

    assert result.elected[0].document_id == 425
    assert 438 in result.elected_ids


def test_p3_cas_tgy_le_catalogue_soleal_classe_2e_par_colpali_est_elu():
    """Panne mesurée le 01/09 (« rallonge TGY3704 ») : la référence n'existe QU'EN IMAGE.

    ColPali met deux catalogues SOLEAL en tête (0,741 et 0,740) ; BM25, qui ne trouve que
    des mots communs, remonte une notice Roto de 124 pages et un guide de câblage. À
    égalité RRF parfaite (rang 1 dans chaque canal = 1/61), l'ancienne élection admettait
    la notice Roto en « complément de canal » et laissait le catalogue de fabrication #397
    — celui qui a répondu — perdre un départage sur son numéro d'identifiant.
    """
    pool = _pool([
        (400, "SOLEAL-GY-55 catalogue conception",
         [(89, 0.0164, 0.741, None)]
         + [(p, 0.0150 - i * 0.0003, 0.62, None) for i, p in enumerate(range(90, 100))]),
        (415, "Roto Safe E jonction de câble", [(25, 0.0164, None, 0.50), (26, 0.0130, None, 0.30)]),
        (397, "SOLEAL-GY-55 catalogue fabrication", [(111, 0.0161, 0.740, None), (114, 0.0159, 0.709, None)]),
        (391, "Montage Roto NX", [(p, 0.0161 - i * 0.0002, None, 0.40) for i, p in enumerate(range(60, 74))]),
    ])
    result = elect_documents(
        None, pool, query_text="installer la rallonge TGY3704 sur une cremone TGY3702"
    )

    assert 397 in result.elected_ids, "le 2e passage ColPali (0,740) a encore été jeté"
    assert result.elected[0].document_id == 400, "à RRF égal, le passage vu par ColPali mène"
    assert 391 not in result.elected_ids, (
        "la notice Roto (BM25 0,40 = miette du repli OR) ne doit plus entrer en complément"
    )


def test_p_a_rrf_egal_le_passage_vu_par_colpali_passe_devant():
    """Rang 1 ColPali et rang 1 BM25 valent tous deux 1/61 : le canal fiable départage."""
    pool = _pool([
        (1, "Texte seul", [(5, 0.0164, None, 0.50)]),
        (2, "Planche ColPali", [(9, 0.0164, 0.74, None)]),
    ])
    result = elect_documents(None, pool, query_text="cote")
    assert result.elected[0].document_id == 2


def test_p_le_detenteur_est_elu_meme_dernier_au_score_d_election():
    """Une fiche d'UNE page sans aucun bonus, dernière au score d'élection, mène quand même
    si elle porte le meilleur passage — c'est exactement le cas où la règle sert."""
    pool = _pool([
        (1, "Catalogue", [(p, 0.30, 0.60, 0.50) for p in range(1, 9)]),   # 0.30 × 1.30
        (3, "Intermédiaire", [(p, 0.29, 0.58, 0.45) for p in range(1, 5)]),  # 0.29 × 1.25
        (2, "Fiche isolée", [(3, 0.31, 0.70, None)]),                     # 0.31 × 1.00
    ])
    result = elect_documents(None, pool, query_text="cote du dormant")

    ordre = [c.document_id for c in result.candidates]
    assert ordre.index(2) == 2, "la fiche doit bien être dernière au score d'élection"
    assert result.elected[0].document_id == 2


def test_p_un_passage_decroche_du_meilleur_nest_pas_un_top():
    """Deux pages seulement, 0,90 contre 0,10 : le second n'est pas « top-3 » par défaut."""
    pool = _pool([
        (1, "Notice", [(3, 0.90, 0.70, 0.60)]),
        (2, "Autre", [(9, 0.10, 0.55, 0.40)]),
    ])
    result = elect_documents(None, pool, query_text="hauteur de poignee")
    assert result.decision == "mono_document"
    assert result.elected_ids == [1]


def test_p_un_document_qui_tient_les_trois_meilleurs_passages_part_seul():
    """Trois passages en tête du même document = mono : la règle n'empile pas pour rien."""
    pool = _pool([
        (1, "Notice", [(3, 0.50, 0.70, 0.60), (4, 0.48, 0.65, 0.55), (5, 0.46, 0.62, 0.50)]),
        (2, "Autre", [(9, 0.40, 0.55, 0.40)]),
    ])
    result = elect_documents(None, pool, query_text="hauteur de poignee")
    assert result.decision == "mono_document"
    assert result.elected_ids == [1]


def test_p_max_docs_a_un_garde_le_meilleur_passage():
    """Plafond à 1 : c'est le détenteur du meilleur passage qui reste, pas le volume."""
    pool = _pool([
        (1, "Catalogue", [(p, 0.30, 0.60, 0.50) for p in range(1, 9)]),
        (2, "Fiche isolée", [(3, 0.31, 0.70, None)]),
    ])
    result = elect_documents(None, pool, query_text="cote", max_docs=1)
    assert result.elected_ids == [2]


def test_p_le_volume_reste_une_preuve_de_complement():
    """La fiche tient les TROIS meilleurs passages ; le catalogue, 4e au passage mais premier
    au score d'élection (8 pages, deux canaux), entre en complément au titre du volume."""
    pool = _pool([
        (2, "Fiche", [(3, 0.31, 0.70, 0.50), (4, 0.30, 0.66, 0.45), (5, 0.295, 0.62, 0.40)]),
        (1, "Catalogue", [(p, 0.29, 0.60, 0.50) for p in range(1, 9)]),
    ])
    result = elect_documents(None, pool, query_text="cote")
    trace = result.to_trace()

    assert result.candidates[0].document_id == 1, "le catalogue doit bien gagner au score d'élection"
    assert result.decision == "complement:2"          # passage n°1 + volume
    assert trace["elected"][0]["reason"] == "top_passage:1"
    assert trace["elected"][1]["reason"] == "best_election_score"
    assert "top_passage:1" in format_election_log(result)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_e1_distribution_departage_a_score_max_egal():
    """À meilleure page identique, le document qui en place plusieurs gagne.

    C'est exactement le signal que le quota (8 pages/doc) et la coupe top-20
    supprimaient avant que l'élection ne l'observe.
    """
    hits = [
        _hit(1, 4, 0.50),
        _hit(2, 4, 0.50),
        _hit(2, 5, 0.45),
        _hit(2, 6, 0.40),
        _hit(2, 7, 0.35),
    ]
    ranked = score_documents(hits)
    assert [d.document_id for d in ranked] == [2, 1]
    assert ranked[0].page_count == 4
    assert ranked[0].election_score > ranked[1].election_score


def test_e2_meilleure_page_nette_bat_le_volume():
    """Parité avec T1 : six pages moyennes ne battent pas LA bonne page."""
    catalogue = [_hit(1, p, 0.30) for p in range(1, 7)]
    notice = [_hit(2, 3, 0.90)]
    ranked = score_documents(catalogue + notice)
    assert ranked[0].document_id == 2, "le catalogue volumineux a repris l'avantage"


def test_bonus_familles_recompense_accord_des_canaux():
    """Deux canaux d'accord sur un document valent mieux qu'un seul, à score égal."""
    un_canal = [_hit(1, 2, 0.40, sources=("bm25",))]
    deux_canaux = [_hit(2, 2, 0.40, sources=("bm25", "colpali"))]
    ranked = score_documents(un_canal + deux_canaux)
    assert ranked[0].document_id == 2
    assert ranked[0].families == {"texte", "visuel"}


def test_pool_vide_rend_une_decision_explicite():
    result = elect_documents(None, [])
    assert result.decision == "empty"
    assert result.elected == []
    assert result.elected_ids == []
    assert "aucun document" in format_election_log(result)


def test_score_non_positif_ne_recoit_pas_de_bonus_multiplicatif():
    """Garde-fou : un score ≤ 0 ne doit pas être « amélioré » par le bonus.

    Les scores RRF sont structurellement positifs (1/(k+rang), et le boost d'ancre est
    un facteur > 1), donc ce chemin est défensif : on vérifie surtout qu'un pool
    dégénéré ne produit ni exception ni score gonflé.
    """
    ranked = score_documents([_hit(1, 1, 0.0), _hit(1, 2, -0.4)])
    assert len(ranked) == 1
    assert ranked[0].election_score <= 0.0


# ---------------------------------------------------------------------------
# Règle de décision 1 / 2 / 3 documents
# ---------------------------------------------------------------------------


def test_e3_trois_meilleurs_passages_de_trois_documents_donnent_trois_elus():
    """Trois passages en tête, trois documents : les trois partent au contexte.

    C'est le renversement volontaire de l'ancienne règle « sans preuve, le dominant part
    seul » : la preuve, c'est désormais le classement des PASSAGES. Le retriever place le
    bon document dans le top-3 dans plus de 85 % des cas ; l'élection ne doit plus le jeter.
    Le recollage inter-documents est tenu en aval (blocs séparés, consigne anti-croisement,
    plancher d'images par document).
    """
    hits = [
        _hit(1, 3, 0.50, sources=("bm25",)),
        _hit(2, 3, 0.48, sources=("bm25",)),
        _hit(3, 3, 0.47, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="hauteur de poignee")
    assert result.decision == "top_passages:3"
    assert result.elected_ids == [1, 2, 3]
    assert [d.reason for d in result.elected] == ["top_passage:1", "top_passage:2", "top_passage:3"]
    # La marge compare désormais les deux meilleurs PASSAGES.
    assert result.margin == pytest.approx(0.48 / 0.50, rel=1e-3)


def test_e3bis_au_dela_du_troisieme_passage_il_faut_une_preuve():
    """Le 4e document, même proche, n'entre plus sans preuve : la règle des passages est
    bornée à trois, et la comparaison / complémentarité gardent leur seuil."""
    hits = [
        _hit(1, 3, 0.50, sources=("bm25",)),
        _hit(2, 3, 0.48, sources=("bm25",)),
        _hit(3, 3, 0.47, sources=("bm25",)),
        _hit(4, 3, 0.46, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="hauteur de poignee")
    assert result.elected_ids == [1, 2, 3]
    assert 4 in [c.document_id for c in result.candidates], "il reste candidat, visible du juge"


def test_e4_question_comparative_admet_un_second_document_au_dela_des_passages():
    """Le document comparé n'est PAS dans les trois meilleurs passages (le premier les
    tient tous) : c'est la question comparative qui l'admet, sous seuil de dominance."""
    hits = [
        _hit(1, 3, 0.50, sources=("bm25",)),
        _hit(1, 4, 0.49, sources=("bm25",)),
        _hit(1, 5, 0.48, sources=("bm25",)),
        _hit(2, 3, 0.40, sources=("bm25",)),
    ]
    result = elect_documents(
        None, hits, query_text="quelle difference entre le Perform 76 et le Kommerling 70 ?"
    )
    assert result.decision == "comparative:2"
    assert result.elected_ids == [1, 2]
    assert result.elected[1].role == "complement"
    assert result.elected[1].reason.startswith("comparative_query:")


def test_question_comparative_mais_score_trop_ecarte_reste_mono():
    """La preuve ne suffit pas : le second doit aussi être proche du dominant."""
    hits = [
        _hit(1, 3, 0.90, sources=("bm25",)),
        _hit(2, 3, 0.10, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="comparer les deux gammes")
    assert result.decision == "mono_document"
    assert result.elected_ids == [1]


def test_e5_complementarite_archetypes_notice_plus_planche(monkeypatch):
    """Notice retrouvée par son TEXTE + planche muette que seul ColPali voit."""
    from app.services import document_election_service as svc

    monkeypatch.setattr(svc.settings, "COLPALI_DOMINANCE_MIN_SCORE", 0.55)
    # La notice tient les trois meilleurs passages ; la planche est 4e — seule la
    # complémentarité de canaux peut l'admettre.
    hits = [
        _hit(1, 8, 0.50, sources=("bm25",), bm25=1.4),
        _hit(1, 9, 0.49, sources=("bm25",), bm25=1.2),
        _hit(1, 10, 0.48, sources=("bm25",), bm25=1.1),
        _hit(2, 14, 0.40, sources=("colpali",), colpali=0.80),
    ]
    result = elect_documents(None, hits, query_text="cote du dormant")
    assert result.decision == "complement:2"
    assert result.elected_ids == [1, 2]
    assert result.elected[1].reason == "channel_complement"


def test_complementarite_refusee_si_colpali_trop_faible(monkeypatch):
    from app.services import document_election_service as svc

    monkeypatch.setattr(svc.settings, "COLPALI_DOMINANCE_MIN_SCORE", 0.55)
    hits = [
        _hit(1, 8, 0.50, sources=("bm25",), bm25=1.4),
        _hit(1, 9, 0.49, sources=("bm25",), bm25=1.2),
        _hit(1, 10, 0.48, sources=("bm25",), bm25=1.1),
        _hit(2, 14, 0.40, sources=("colpali",), colpali=0.20),
    ]
    result = elect_documents(None, hits, query_text="cote du dormant")
    assert result.decision == "mono_document"
    assert result.elected_ids == [1]


def test_complement_texte_refuse_sur_une_miette_du_repli_or():
    """Cas TGY3704 : dominant visuel, candidat texte trouvé par le repli OR (ts_rank 0,40).

    BM25 n'y avait trouvé que des mots communs ; la différence de canal n'est pas une
    preuve quand le canal lui-même n'a rien de solide à montrer."""
    hits = [
        _hit(1, 89, 0.50, sources=("colpali",), colpali=0.74),
        _hit(1, 90, 0.49, sources=("colpali",), colpali=0.70),
        _hit(1, 91, 0.48, sources=("colpali",), colpali=0.66),
        _hit(2, 71, 0.40, sources=("bm25",), bm25=0.40),
    ]
    result = elect_documents(None, hits, query_text="installer la rallonge")
    assert result.elected_ids == [1]


def test_complement_texte_admis_sur_un_vrai_match_lexical():
    hits = [
        _hit(1, 89, 0.50, sources=("colpali",), colpali=0.74),
        _hit(1, 90, 0.49, sources=("colpali",), colpali=0.70),
        _hit(1, 91, 0.48, sources=("colpali",), colpali=0.66),
        _hit(2, 71, 0.40, sources=("bm25",), bm25=4.2),
    ]
    result = elect_documents(None, hits, query_text="installer la rallonge")
    assert result.elected_ids == [1, 2]
    assert result.elected[1].reason == "channel_complement"


def test_meme_famille_de_canaux_nest_pas_une_complementarite():
    """Deux documents texte : le second, hors des trois meilleurs passages, reste concurrent."""
    hits = [
        _hit(1, 8, 0.50, sources=("bm25",), bm25=1.4),
        _hit(1, 9, 0.49, sources=("bm25",), bm25=1.2),
        _hit(1, 10, 0.48, sources=("bm25",), bm25=1.1),
        _hit(2, 9, 0.40, sources=("bm25",), bm25=1.3),
    ]
    result = elect_documents(None, hits, query_text="cote du dormant")
    assert result.elected_ids == [1]


def test_max_docs_plafonne_le_nombre_delus():
    hits = [
        _hit(1, 1, 0.50, sources=("bm25",)),
        _hit(2, 1, 0.49, sources=("bm25",)),
        _hit(3, 1, 0.48, sources=("bm25",)),
    ]
    result = elect_documents(
        None, hits, query_text="comparer 1 et 2", max_docs=2
    )
    assert len(result.elected) == 2


def test_seuil_de_dominance_est_bien_relatif():
    """Juste sous le seuil → écarté ; juste au-dessus → admis (question comparative)."""
    juste_sous = DOMINANCE_RATIO * 0.50 - 0.01
    juste_au_dessus = DOMINANCE_RATIO * 0.50 + 0.01
    q = "comparer les deux"

    r1 = elect_documents(
        None, [_hit(1, 1, 0.50, sources=("bm25",)), _hit(2, 1, juste_sous, sources=("bm25",))],
        query_text=q,
    )
    assert r1.elected_ids == [1]

    r2 = elect_documents(
        None, [_hit(1, 1, 0.50, sources=("bm25",)), _hit(2, 1, juste_au_dessus, sources=("bm25",))],
        query_text=q,
    )
    assert r2.elected_ids == [1, 2]


# ---------------------------------------------------------------------------
# Détection de comparaison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "quelle difference entre le 76177 et le 76185 ?",
        "peux-tu comparer les deux gammes",
        "Perform 76 vs Kommerling 70",
        "le 76185 par rapport au 76177",
        "quel est l'equivalent chez Technal ?",
    ],
)
def test_questions_comparatives_detectees(question):
    ok, marker = is_comparative_query(question)
    assert ok, f"non détectée : {question}"
    assert marker


@pytest.mark.parametrize(
    "question",
    [
        # Le piège central : une ÉNUMÉRATION cite beaucoup de références mais veut UN
        # document. La compter comme comparative rouvrirait le recollage inter-documents.
        "pour les dormants standards donne moi pour chaque seuil les embouts compatibles",
        "quelle est la charge maximale par ouvrant sur le pivot bas ?",
        "liste des references de parclose pour les dormants",
        "",
    ],
)
def test_questions_non_comparatives(question):
    ok, _ = is_comparative_query(question)
    assert not ok, f"faussement comparative : {question}"


# ---------------------------------------------------------------------------
# Déduplication de versions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "Dossier technique PERFORM 76 24 08 2026",
            "Dossier technique PERFORM 76 26 08 2026",
        ),
        ("Cahier technique Perform 76 CC01", "Cahier technique Perform76 CC02"),
        ("DTA_6_16-2335_V4.pdf", "DTA_6_16-2335_V5.pdf"),
        ("Notice pose (1).pdf", "Notice pose.pdf"),
        ("Notice 2025-04-24", "Notice 2026-08-26"),
    ],
)
def test_titres_de_meme_famille_convergent(a, b):
    assert normalize_title_family(a) == normalize_title_family(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Deux PRODUITS distincts ne doivent jamais fusionner (perte de rappel).
        ("Cahier technique Perform 76", "Cahier technique Perform 70"),
        ("DTA Kommerling 70", "DTA Trocal 88"),
        ("Notice pose fenetre", "Notice reglage volet"),
    ],
)
def test_titres_de_produits_distincts_ne_fusionnent_pas(a, b):
    assert normalize_title_family(a) != normalize_title_family(b)


def test_e6_deux_versions_ne_sont_pas_toutes_deux_candidates():
    """CC01 + CC02 → une seule survit, l'autre est tracée comme écartée."""
    hits = [
        _hit(1, 3, 0.50, title="Cahier technique Perform 76 CC01"),
        _hit(2, 3, 0.49, title="Cahier technique Perform 76 CC02"),
    ]
    result = elect_documents(None, hits, query_text="hauteur de poignee")
    assert len(result.candidates) == 1, "les deux révisions coexistent encore"
    assert len(result.rejected_versions) == 1
    assert result.rejected_versions[0]["reason"] == "version_superseded"
    assert result.rejected_versions[0]["kept_document_id"] == result.elected_ids[0]


def test_deux_versions_meme_sur_question_comparative_restent_une_seule():
    """Une question comparative ne doit pas ressusciter la version périmée."""
    hits = [
        _hit(1, 3, 0.50, title="Cahier technique Perform 76 CC01"),
        _hit(2, 3, 0.49, title="Cahier technique Perform 76 CC02"),
    ]
    result = elect_documents(None, hits, query_text="quelle difference entre les deux")
    assert len(result.elected) == 1


def test_dedupe_sans_titre_ne_fusionne_pas_par_defaut():
    """Titres vides : chaque document garde sa propre famille (pas de fusion aveugle)."""
    docs = [
        ElectedDocument(document_id=1, title="", election_score=0.5, score_max=0.5),
        ElectedDocument(document_id=2, title="", election_score=0.4, score_max=0.4),
    ]
    kept, rejected = dedupe_versions(None, docs)
    assert [d.document_id for d in kept] == [1, 2]
    assert rejected == []


def test_dedupe_arbitre_sur_updated_at(db_session: Session):
    """Avec la base : c'est la révision la plus récemment mise à jour qui survit."""
    user = create_test_user(db_session, "responsable")
    lib = Library(name="Lib", user_id=user.id, is_global=False)
    db_session.add(lib)
    db_session.commit()
    db_session.refresh(lib)

    titre = "Cahier technique Perform 76"
    vieux = Document(
        title=f"{titre} CC01",
        document_type="written",
        processing_status="completed",
        library_id=lib.id,
        user_id=user.id,
        updated_at=datetime(2026, 8, 24, 10, 0, 0),
    )
    recent = Document(
        title=f"{titre} CC02",
        document_type="written",
        processing_status="completed",
        library_id=lib.id,
        user_id=user.id,
        updated_at=datetime(2026, 8, 26, 16, 0, 0),
    )
    db_session.add_all([vieux, recent])
    db_session.commit()
    db_session.refresh(vieux)
    db_session.refresh(recent)

    # La VIEILLE version score mieux : seule la date doit la faire perdre.
    hits = [
        _hit(vieux.id, 3, 0.60, title=vieux.title),
        _hit(recent.id, 3, 0.55, title=recent.title),
    ]
    result = elect_documents(db_session, hits, query_text="hauteur de poignee")
    assert result.elected_ids == [recent.id]
    assert result.rejected_versions[0]["document_id"] == vieux.id


# ---------------------------------------------------------------------------
# Pack du juge & trace
# ---------------------------------------------------------------------------


def test_candidate_passages_couvre_tous_les_candidats_pas_seulement_les_elus():
    """Le juge doit garder une vue LARGE, sinon il ne peut jamais contredire l'élection."""
    hits = [
        _hit(1, 3, 0.50, sources=("bm25",)),
        _hit(1, 4, 0.45, sources=("bm25",)),
        _hit(2, 9, 0.40, sources=("bm25",)),
        _hit(3, 2, 0.35, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="hauteur de poignee")
    # Passages 0,50 / 0,45 (doc 1) et 0,40 (doc 2) sont les trois meilleurs → deux élus ;
    # le doc 3 (0,35 = 70 % du meilleur, à la limite) reste candidat.
    assert result.elected_ids == [1, 2]

    passages = election_candidate_passages(result)
    doc_ids = {p["document_id"] for p in passages}
    assert doc_ids == {1, 2, 3}, "le juge ne verrait que les documents élus"
    # Passages LÉGERS : build_cag_context recharge le texte depuis la base.
    assert all("passage" not in p for p in passages)
    assert all(p["page_no"] > 0 for p in passages)


def test_candidate_passages_plafonne_les_pages_par_document():
    from app.services.document_election_service import CANDIDATE_PAGES_PER_DOC

    hits = [_hit(1, p, 0.50 - p * 0.01, sources=("bm25",)) for p in range(1, 20)]
    result = elect_documents(None, hits, query_text="test")
    passages = election_candidate_passages(result)
    assert len(passages) == CANDIDATE_PAGES_PER_DOC
    # Les pages retenues sont les mieux notées (ici les premières).
    assert [p["page_no"] for p in passages] == list(range(1, CANDIDATE_PAGES_PER_DOC + 1))


def test_trace_est_json_serialisable():
    """La trace part en colonne JSON et dans le SSE : aucun set/dataclass ne doit fuir."""
    import json

    hits = [
        _hit(1, 3, 0.50, sources=("bm25", "colpali")),
        _hit(2, 3, 0.20, sources=("bm25",)),
    ]
    result = elect_documents(None, hits, query_text="test")
    payload = json.dumps(result.to_trace())
    assert '"decision"' in payload
    assert '"elected"' in payload
    assert '"candidates"' in payload
