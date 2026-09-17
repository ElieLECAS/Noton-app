"""Génération de la réponse : un appel, un flux, un contrôle d'ancrage.

Portage des tests du lecteur agentique qui couvrent du comportement CONSERVÉ après sa
suppression : récupération des blocs machine égarés dans le canal ``tool_calls``, recalage
des pages citées sur les pages réellement packées, et forme de l'appel unique.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from app.services.answer_generation_service import (
    AnswerGeneration,
    ReadDocument,
    split_leaked_blocks,
)


def _ev(*items: Dict[str, Any]) -> List[str]:
    return [json.dumps(i) for i in items]


class FakeStream:
    """Les événements rendus par le modèle ; enregistre contexte et kwargs reçus."""

    def __init__(self, events: List[str]):
        self.events = list(events)
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, message, *, model, context, max_tokens=None, **kwargs):
        self.calls.append({"context": [dict(m) for m in context], "kwargs": dict(kwargs)})
        for ev in self.events:
            yield ev


def _parse(events: List[str]) -> List[Dict[str, Any]]:
    return [json.loads(e[len("data: ") :].strip()) for e in events]


async def _collect(gen: AnswerGeneration) -> List[Dict[str, Any]]:
    return _parse([e async for e in gen.run()])


def _messages() -> List[Dict[str, Any]]:
    return [{"role": "system", "content": "PROMPT"}, {"role": "user", "content": "Question ?"}]


_INITIAL_DOCS = [
    {
        "index": 1,
        "document_id": 405,
        "document_title": "Notice",
        "pages": [110, 111],
        "has_source_file": True,
    }
]


# ---------------------------------------------------------------------------
# 1. L'appel unique
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_un_seul_appel_nu_sans_outils():
    """Le tour est UN appel : aucun `tools` ni `tool_choice` n'est proposé au modèle."""
    stream = FakeStream(_ev({"message": {"content": "Direct."}}))
    gen = AnswerGeneration(messages=_messages(), model="m", stream_fn=stream)
    events = await _collect(gen)
    assert len(stream.calls) == 1
    assert stream.calls[0]["kwargs"] == {}
    assert gen.final_text == "Direct."
    assert "".join(e["message"]["content"] for e in events if "message" in e) == "Direct."


@pytest.mark.asyncio
async def test_le_texte_part_au_fil_de_l_eau_morceau_par_morceau():
    """Un chunk du modèle = un événement SSE. Pas de tampon rejoué en fin de tour."""
    stream = FakeStream(
        _ev(
            {"message": {"content": "La parclose "}},
            {"message": {"content": "2452 accepte "}},
            {"message": {"content": "16 mm."}},
        )
    )
    gen = AnswerGeneration(messages=_messages(), model="m", stream_fn=stream)
    events = await _collect(gen)
    morceaux = [e["message"]["content"] for e in events if "message" in e]
    assert morceaux == ["La parclose ", "2452 accepte ", "16 mm."]
    assert gen.final_text == "La parclose 2452 accepte 16 mm."


@pytest.mark.asyncio
async def test_une_balise_coupee_entre_deux_chunks_ne_fuit_pas():
    """Le filtre retient le préfixe potentiel : `<sou` ne doit jamais atteindre l'écran."""
    stream = FakeStream(
        _ev(
            {"message": {"content": "16 mm.<sou"}},
            {"message": {"content": 'rces>{"used":[{"doc_id":405,"pages":[111]}]}</sour'}},
            {"message": {"content": "ces>"}},
        )
    )
    gen = AnswerGeneration(
        messages=_messages(), model="m", stream_fn=stream, initial_documents=_INITIAL_DOCS
    )
    events = await _collect(gen)
    affiche = "".join(e["message"]["content"] for e in events if "message" in e)
    assert affiche == "16 mm."
    assert "<sou" not in affiche and "sources" not in affiche
    assert gen.used_pages_by_index() == {1: [111]}


@pytest.mark.asyncio
async def test_une_balise_jamais_refermee_est_relachee_en_fin_de_flux():
    """Le modèle ouvre `<sources>` puis le flux coupe : le retenu ne doit pas disparaître."""
    stream = FakeStream(_ev({"message": {"content": "Réponse. <sourc"}}))
    gen = AnswerGeneration(messages=_messages(), model="m", stream_fn=stream)
    events = await _collect(gen)
    affiche = "".join(e["message"]["content"] for e in events if "message" in e)
    assert affiche.startswith("Réponse.")
    assert gen.final_text == "Réponse. <sourc".rstrip()


@pytest.mark.asyncio
async def test_le_premier_token_est_date():
    stream = FakeStream(_ev({"thinking": "…"}, {"message": {"content": "16 mm."}}))
    gen = AnswerGeneration(messages=_messages(), model="m", stream_fn=stream)
    await _collect(gen)
    assert gen.trace["first_token_ms"] is not None
    assert gen.trace["first_token_ms"] <= gen.trace["duration_ms"]


@pytest.mark.asyncio
async def test_le_raisonnement_est_emis_au_fil_de_l_eau():
    stream = FakeStream(
        _ev({"thinking": "Je regarde la planche."}, {"message": {"content": "16 mm."}})
    )
    gen = AnswerGeneration(messages=_messages(), model="m", stream_fn=stream)
    events = await _collect(gen)
    assert events[0] == {"thinking": "Je regarde la planche."}
    assert gen.reasoning_parts == ["Je regarde la planche."]


@pytest.mark.asyncio
async def test_les_balises_machine_ne_partent_pas_a_l_utilisateur():
    stream = FakeStream(
        _ev(
            {
                "message": {
                    "content": 'Réponse. <sources>{"used":[{"doc_id":405,"pages":[111]}]}</sources>'
                    '<evidence>["engager le tenon (doc 405 p.111)"]</evidence>'
                }
            }
        )
    )
    gen = AnswerGeneration(
        messages=_messages(), model="m", stream_fn=stream, initial_documents=_INITIAL_DOCS
    )
    await _collect(gen)
    assert gen.final_text == "Réponse."
    assert gen.used_pages_by_index() == {1: [111]}
    assert gen.evidence_citations == ["engager le tenon (doc 405 p.111)"]


@pytest.mark.asyncio
async def test_le_verdict_du_controle_est_trace_sans_relance():
    """Sans outils il n'y a plus de round de contrôle : le verdict est constaté, pas rejoué."""
    stream = FakeStream(_ev({"message": {"content": "La cote est 30 mm."}}))
    appels: List[str] = []

    def _check(texte, preuve, citations, pages_image):
        appels.append(texte)
        return {"ok": False, "unsupported_claims": ["30 mm"]}

    gen = AnswerGeneration(
        messages=_messages(), model="m", stream_fn=stream, output_check=_check
    )
    await _collect(gen)
    assert len(stream.calls) == 1, "un verdict KO ne doit PAS relancer le modèle"
    assert appels == ["La cote est 30 mm."]
    assert gen.verification["action"] == "flagged"
    assert gen.trace["verification_action"] == "flagged"


@pytest.mark.asyncio
async def test_un_controle_qui_explose_n_empeche_pas_la_reponse():
    stream = FakeStream(_ev({"message": {"content": "Réponse."}}))

    def _check(*_args):
        raise RuntimeError("boum")

    gen = AnswerGeneration(
        messages=_messages(), model="m", stream_fn=stream, output_check=_check
    )
    events = await _collect(gen)
    assert gen.verification is None
    assert "".join(e["message"]["content"] for e in events if "message" in e) == "Réponse."


# ---------------------------------------------------------------------------
# 2. Bloc machine égaré dans le canal des appels d'outils
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


def test_le_nom_nu_sources_est_rehabille_en_balise():
    calls, fuite = split_leaked_blocks([_appel("sources", '{"used":[{"doc_id":438,"pages":[5]}]}')])
    assert calls == []
    assert fuite.startswith("<sources>") and fuite.endswith("</sources>")


def test_aucun_appel_aucune_fuite():
    assert split_leaked_blocks([]) == ([], "")


@pytest.mark.asyncio
async def test_un_bloc_sources_egare_rejoint_la_reponse():
    """Mesuré le 14/09 : sans rattrapage, le tour finissait avec des sources de repli."""
    stream = FakeStream(
        _ev(
            {"message": {"content": "Réponse."}},
            {"tool_calls": [_appel('<sources>{"used":[{"doc_id":405,"pages":[111]}]}</sources>')]},
        )
    )
    gen = AnswerGeneration(
        messages=_messages(), model="m", stream_fn=stream, initial_documents=_INITIAL_DOCS
    )
    await _collect(gen)
    assert gen.final_text == "Réponse."
    assert gen.used_pages_by_index() == {1: [111]}
    assert gen.trace["leaked_blocks"] == 1


# ---------------------------------------------------------------------------
# 3. Recalage des pages citées
# ---------------------------------------------------------------------------


def _generation_avec_pages(pages: set) -> AnswerGeneration:
    gen = AnswerGeneration(messages=[], model="m", stream_fn=None)
    gen.read_documents = {
        438: ReadDocument(document_id=438, index=1, title="Dossier", pages=set(pages))
    }
    return gen


class _FiltreFactice:
    def __init__(self, used):
        self.used_documents = used


def test_une_page_imprimee_est_recalee_sur_les_pages_packees():
    """Le modèle cite « page 5 » (le cartouche de la planche) ; la page fournie est la 8."""
    gen = _generation_avec_pages({6, 8})
    gen.source_filter = _FiltreFactice([{"doc_id": 438, "pages": [5]}])
    assert gen.used_pages_by_index() == {1: [6, 8]}


def test_une_page_declaree_qui_existe_est_conservee():
    gen = _generation_avec_pages({6, 8})
    gen.source_filter = _FiltreFactice([{"doc_id": 438, "pages": [8]}])
    assert gen.used_pages_by_index() == {1: [8]}


def test_le_tri_ne_garde_que_les_pages_reellement_packees():
    gen = _generation_avec_pages({6, 8})
    gen.source_filter = _FiltreFactice([{"doc_id": 438, "pages": [8, 5, 99]}])
    assert gen.used_pages_by_index() == {1: [8]}


def test_sans_page_packee_la_declaration_est_conservee_telle_quelle():
    gen = _generation_avec_pages(set())
    gen.source_filter = _FiltreFactice([{"doc_id": 438, "pages": [12]}])
    assert gen.used_pages_by_index() == {1: [12]}
