"""Formats des résultats d'outils du lecteur (fonctions pures, sans DB) et contrôle de sortie."""
from __future__ import annotations

from app.services.reader_tools import (
    _collapse_sections,
    format_code_occurrences,
    format_pages_text,
    format_plan,
    format_search_results,
)
from app.services.response_verification_service import check_reader_output, evidence_in_pack
from app.services.stream_source_filter import EvidenceTagStreamFilter, SourcesTagStreamFilter


# ---------------------------------------------------------------------------
# rechercher
# ---------------------------------------------------------------------------


def test_search_results_are_compact_and_expose_channels():
    passages = [
        {"document_id": 405, "document_title": "Notice", "page_no": 111, "score": 0.0312,
         "retrieval_sources": ["colpali", "bm25"], "passage_raw": "Engager le tenon " * 40},
        {"document_id": 424, "document_title": "Catalogue", "page_no": 42, "score": 0.028,
         "retrieval_sources": ["colpali"], "passage_raw": "Page 42 — contenu visuel uniquement"},
    ]
    text, evidence, docs = format_search_results(passages)
    assert text.startswith("1. doc 405 « Notice » p.111  0.031  [colpali, bm25]")
    assert "[colpali]" in text and "demande l'image" in text
    assert len(text.splitlines()[1]) <= 320  # extrait borné
    assert "Engager le tenon" in evidence and "contenu visuel" not in evidence
    assert docs == {405: "Notice", 424: "Catalogue"}


def test_search_no_results_says_so_with_scope():
    text, evidence, docs = format_search_results([], document_id=405)
    assert "Aucune page trouvée" in text and "document 405" in text
    assert evidence == "" and docs == {}


# ---------------------------------------------------------------------------
# lire_pages
# ---------------------------------------------------------------------------


def test_pages_text_marks_mute_pages_and_out_of_range():
    out = format_pages_text(
        document_id=405, title="Notice", header="Source : Technal", page_count=124,
        pages=[110, 111, 112, 200], text_by_page={110: "Crémones", 112: "Serrer les 6 vis TGY3723"},
        image_pages=[111],
    )
    assert out.startswith("=== doc 405 « Notice » ===\nSource : Technal\nDocument de 124 page(s).")
    assert "[page 110]\nCrémones" in out
    assert "[page 111] (page muette — aucun texte extrait ; image jointe ci-dessous)" in out
    assert "[page 112]\nSerrer" in out
    assert "[page 200] (hors du document : 124 pages)" in out


def test_pages_text_truncates_over_cap():
    big = "x" * 15000
    out = format_pages_text(document_id=1, title="T", header="", page_count=3, pages=[1, 2, 3],
                            text_by_page={1: big, 2: big, 3: big})
    assert "texte tronqué" in out
    assert out.count("[page ") == 2  # la 3e page n'est pas entamée


# ---------------------------------------------------------------------------
# chercher_code
# ---------------------------------------------------------------------------


def test_code_absent_is_said_explicitly_with_document_count():
    text, evidence, docs, pages = format_code_occurrences("tgy3710", [], n_documents=31)
    assert text.startswith("TGY3710 — aucun chunk ne contient cette référence dans le périmètre (31 document(s))")
    assert "ne la déduis pas" in text
    assert docs == {} and pages == []


def test_code_short_extracts_rank_by_spec_density():
    rows = [
        {"chunk_id": 1, "document_id": 424, "document_title": "Catalogue", "page_no": 42,
         "content": "TGY3704 — rallonge 4e point"},
        {"chunk_id": 2, "document_id": 424, "document_title": "Catalogue", "page_no": 42,
         "content": "TGY3704 — Rallonge 4e point pour crémone TGY3702, longueur 300 mm, 6 vis, 2,5 Nm"},
    ]
    text, evidence, docs, pages = format_code_occurrences("TGY3704", rows, n_documents=3)
    assert text.startswith("TGY3704 — doc 424 « Catalogue », p.42\n« TGY3704 — Rallonge 4e point pour crémone TGY3702, longueur 300 mm")
    assert "300 mm" in evidence
    assert docs == {424: "Catalogue"} and pages[0] == (424, 42)


def test_code_only_in_long_passages_points_to_lire_pages():
    rows = [{"chunk_id": 1, "document_id": 405, "document_title": "Notice", "page_no": 111, "content": "TGY3704 " + "blabla " * 400}]
    text, evidence, docs, pages = format_code_occurrences("TGY3704", rows, n_documents=3)
    assert "passage(s) long(s)" in text and "lire_pages" in text and "doc 405 « Notice » p. 111" in text
    assert pages == [(405, 111)]


# ---------------------------------------------------------------------------
# plan_du_document
# ---------------------------------------------------------------------------


def test_plan_with_and_without_sections():
    sections = _collapse_sections([(1, "Sommaire"), (1, "Sommaire"), (12, "Usinages"), (None, "X"), (108, "Crémones et rallonges")])
    assert sections == [(1, "Sommaire"), (12, "Usinages"), (108, "Crémones et rallonges")]
    text = format_plan(document_id=405, title="Notice", header="", page_count=84, pages_with_text=79,
                       mode="full_text", sections=sections, matched_pages=[111])
    assert "84 page(s), 79 avec texte" in text and "mode conseillé : texte" in text
    assert "Pages retrouvées par la recherche : 111" in text
    assert "Sections : p.1 Sommaire · p.12 Usinages · p.108 Crémones et rallonges" in text
    empty = format_plan(document_id=1, title="T", header="", page_count=3, pages_with_text=0, mode="image_first", sections=[])
    assert "Aucun titre de section indexé" in empty and "demande les images" in empty


# ---------------------------------------------------------------------------
# Filtres de balises : doc_id et <evidence>
# ---------------------------------------------------------------------------


def test_sources_filter_accepts_doc_id_and_doc():
    f = SourcesTagStreamFilter()
    out = f.feed('R. <sources>{"used":[{"doc_id":405,"pages":[1]},{"doc":2,"pages":[3]}]}</sources>')
    out += f.finalize()
    assert out == "R. "
    assert f.used_documents == [{"doc_id": 405, "pages": [1]}, {"doc": 2, "pages": [3]}]


def test_evidence_filter_parses_json_and_text_fallback():
    f = EvidenceTagStreamFilter()
    out = f.feed('Texte <evidence>["a b c (doc 1 p.2)", {"quote": "d e f"}]</evidence>')
    out += f.finalize()
    assert out == "Texte "
    assert f.citations == ["a b c (doc 1 p.2)", "d e f"]
    g = EvidenceTagStreamFilter()
    g.feed("<evidence>« première citation »\n- seconde citation</evidence>")
    assert g.citations == ["première citation", "seconde citation"]


# ---------------------------------------------------------------------------
# Contrôle de sortie
# ---------------------------------------------------------------------------

_EVIDENCE = (
    "[page 42]\nTGY3704 — Rallonge 4e point pour crémone TGY3702, longueur 300 mm\n"
    "[page 111]\nEngager le tenon TGY3704 dans le boîtier. Serrer les 6 vis TGY3723 au couple de 2,5 N·m."
)


def test_invented_code_is_ko_with_actionable_feedback():
    res = check_reader_output(
        response_text="Pour passer en 4 points, montez la rallonge TGY3710.",
        evidence_text=_EVIDENCE, question="rallonge 4 points crémone TGY3702 ?",
    )
    assert res["ok"] is False
    assert res["unsupported_codes"] == ["TGY3710"]
    assert "chercher_code" in res["feedback"] and "TGY3710" in res["feedback"]


def test_grounded_answer_passes_and_citations_are_verified():
    res = check_reader_output(
        response_text="La rallonge TGY3704 se monte en engageant le tenon dans le boîtier ; serrer les 6 vis TGY3723.",
        evidence_text=_EVIDENCE, question="rallonge 4 points ?",
        citations=["Engager le tenon TGY3704 dans le boîtier (doc 405 p.111)", "phrase inventée qui n'existe pas du tout ici"],
    )
    assert res["ok"] is True and res["feedback"] is None
    assert res["citations_total"] == 2
    assert res["citations_unverified"] == ["phrase inventée qui n'existe pas du tout ici"]


def test_question_codes_are_exempt_and_measurements_checked():
    res = check_reader_output(
        response_text="La TGY3702 accepte la rallonge ; couple de serrage 4,5 N·m.",
        evidence_text=_EVIDENCE, question="couple de serrage rallonge TGY3702 ?",
    )
    assert res["unsupported_codes"] == []
    assert any("4,5" in c for c in res["unsupported_claims"])
    assert res["ok"] is False and "lire_pages" in res["feedback"]


def test_measurement_read_on_a_seen_image_is_not_flagged():
    """Cote absente du texte mais page vue en IMAGE : le contrôle est aveugle, il se tait.

    Cas réel du 04/09 (parclose 2452) : l'épaisseur de vitrage n'est qu'une cote de la
    coupe — la signaler « non étayée » faisait réécrire une bonne réponse en « les
    documents ne précisent pas »."""
    res = check_reader_output(
        response_text="La parclose 2452 reçoit un vitrage de 30 mm.",
        evidence_text=_EVIDENCE,
        question="épaisseur de vitrage parclose 2452 ?",
        image_pages_seen=[(438, 8)],
    )
    assert res["ok"] is True and res["feedback"] is None
    assert res["claims_from_image"] == ["30 mm"]


def test_norms_and_ral_stay_checked_even_with_images():
    """L'exemption ne vaut QUE pour les cotes : une norme ou un RAL inventé reste signalé."""
    res = check_reader_output(
        response_text="Conforme à la NF P20-302, teinte RAL 9016.",
        evidence_text=_EVIDENCE,
        question="norme et teinte ?",
        image_pages_seen=[(438, 8)],
    )
    assert res["ok"] is False
    assert any("NF" in c for c in res["unsupported_claims"])
    assert any("RAL" in c.upper() for c in res["unsupported_claims"])


def test_measurement_without_image_is_still_flagged():
    """Sans image vue, une cote absente du texte reste une invention signalée."""
    res = check_reader_output(
        response_text="La parclose 2452 reçoit un vitrage de 30 mm.",
        evidence_text=_EVIDENCE,
        question="épaisseur de vitrage parclose 2452 ?",
    )
    assert res["ok"] is False
    assert any("30 mm" in c for c in res["unsupported_claims"])
    assert res["claims_from_image"] == []


def test_evidence_in_pack_tolerances():
    assert evidence_in_pack("ENGAGER le tenon tgy3704 dans le boitier", _EVIDENCE)
    assert evidence_in_pack("Engager le tenon TGY3704 … couple de 2,5 N·m", _EVIDENCE)
    assert not evidence_in_pack("TGY", _EVIDENCE)
