"""Filtre de stream du bloc final <sources>{...}</sources> (mode CAG).

Le bloc machine ne doit jamais atteindre l'utilisateur, même coupé entre plusieurs
chunks SSE ; le JSON parsé alimente les sources UI ; un faux positif (balise jamais
fermée) est relâché tel quel pour ne rien perdre de la réponse.
"""
from __future__ import annotations

from app.services.stream_source_filter import SourcesTagStreamFilter


def _run(chunks):
    f = SourcesTagStreamFilter()
    out = "".join(f.feed(c) for c in chunks)
    out += f.finalize()
    return out, f


def test_block_in_single_chunk_is_hidden_and_parsed():
    out, f = _run(['Réponse.\n<sources>{"used":[{"doc":1,"pages":[3,4]}]}</sources>'])
    assert out == "Réponse.\n"
    assert f.used_documents == [{"doc": 1, "pages": [3, 4]}]


def test_block_split_across_many_chunks():
    chunks = ["Voici. ", "<sou", 'rces>{"used":[{"doc":2,"pa', 'ges":[5]}]}</sou', "rces>"]
    out, f = _run(chunks)
    assert out == "Voici. "
    assert f.used_documents == [{"doc": 2, "pages": [5]}]


def test_no_block_passes_text_through_unchanged():
    out, f = _run(["Bonjour, ", "la cote est 300 mm."])
    assert out == "Bonjour, la cote est 300 mm."
    assert f.used_documents == []


def test_angle_brackets_in_normal_text_are_kept():
    out, _ = _run(["a < b et x <s", "pan> fin"])
    assert out == "a < b et x <span> fin"


def test_invalid_json_yields_empty_sources():
    out, f = _run(["Texte <sources>pas du json</sources> suite"])
    assert out == "Texte  suite"
    assert f.used_documents == []


def test_unclosed_tag_short_is_swallowed_if_jsonlike():
    # Stream coupé au milieu du bloc : contenu JSON → avalé (jamais affiché).
    out, f = _run(["Réponse. <sources>", '{"used":[{"doc":1,"pages":[2]}]}'])
    assert out == "Réponse. "
    assert f.used_documents == [{"doc": 1, "pages": [2]}]


def test_unclosed_tag_nonjson_is_released():
    out, _ = _run(["Début <sources>et du texte normal jamais fermé"])
    assert out == "Début <sources>et du texte normal jamais fermé"


def test_partial_open_tag_at_end_is_released_on_finalize():
    out, _ = _run(["fin de phrase <sour"])
    assert out == "fin de phrase <sour"


def test_capture_overflow_releases_text():
    # Balise ouverte puis > 4000 chars sans fermeture → tout est relâché.
    big = "x" * 5000
    out, f = _run(["A <sources>" + big + " B"])
    assert out.startswith("A <sources>")
    assert big in out
    assert f.used_documents == []


def test_used_documents_coerces_types():
    _, f = _run(['<sources>{"used":[{"doc":"3","pages":[1,"2"]},{"doc":null}]}</sources>'])
    # doc "3" coerçable → gardé (pages: seuls les nombres) ; doc null → écarté.
    assert f.used_documents == [{"doc": 3, "pages": [1]}]
