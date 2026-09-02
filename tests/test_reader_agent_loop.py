"""Boucle du lecteur agentique : rounds d'outils, budgets, élagage d'images, contrôle de
sortie en retour outil, repli sans outils. Flux Mistral simulé (aucun réseau, aucune DB)."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import httpx
import pytest

from app.services.reader_agent_service import LoopBudget, ReaderLoop, ToolResult, ToolSpec


def _ev(*items: Dict[str, Any]) -> List[str]:
    return [json.dumps(i) for i in items]


def _tc(id_: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class FakeStream:
    """Une liste d'événements par appel du modèle ; enregistre contexte et kwargs reçus."""

    def __init__(self, rounds: List[List[str]], *, raise_first_400: bool = False):
        self.rounds = list(rounds)
        self.calls: List[Dict[str, Any]] = []
        self.raise_first_400 = raise_first_400

    async def __call__(self, message, *, model, context, max_tokens=None, **kwargs):
        self.calls.append({"context": [dict(m) for m in context], "kwargs": dict(kwargs)})
        if self.raise_first_400 and len(self.calls) == 1:
            req = httpx.Request("POST", "http://mistral.test")
            raise httpx.HTTPStatusError(
                "400", request=req, response=httpx.Response(400, text="bad tools", request=req)
            )
        for ev in self.rounds.pop(0):
            yield ev


def _spec(name: str, handler, *, max_images: int = 0) -> ToolSpec:
    return ToolSpec(
        name=name,
        schema={"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}},
        handler=handler,
        max_images=max_images,
        label=lambda a, n=name: f"Outil {n}",
    )


def _parse(events: List[str]) -> List[Dict[str, Any]]:
    return [json.loads(e[len("data: "):].strip()) for e in events]


async def _collect(loop: ReaderLoop) -> List[Dict[str, Any]]:
    return _parse([e async for e in loop.run()])


def _messages() -> List[Dict[str, Any]]:
    return [{"role": "system", "content": "PROMPT"}, {"role": "user", "content": "Question ?"}]


_INITIAL_DOCS = [{"index": 1, "document_id": 405, "document_title": "Notice", "pages": [110], "has_source_file": True}]

_FINAL = (
    'Réponse finale. <sources>{"used":[{"doc_id":405,"pages":[111]}]}</sources>'
    '<evidence>["engager le tenon TGY3704 dans le boîtier (doc 405 p.111)"]</evidence>'
)


# ---------------------------------------------------------------------------
# Round d'outil puis réponse finale
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_round_then_final_answer():
    seen_args: List[Dict[str, Any]] = []

    async def lire_pages(args):
        seen_args.append(args)
        return ToolResult(
            text="[page 111] Engager le tenon TGY3704 dans le boîtier.",
            pages_read=[(405, 111)],
            documents={405: "Notice"},
        )

    stream = FakeStream(
        [
            _ev({"message": {"content": "Je lis la page 111."}}, {"tool_calls": [_tc("abc123def", "lire_pages", {"document_id": 405, "pages": [111]})]}),
            _ev({"message": {"content": _FINAL}}),
        ]
    )
    loop = ReaderLoop(
        messages=_messages(), model="m", stream_fn=stream, tools=[_spec("lire_pages", lire_pages)],
        initial_documents=_INITIAL_DOCS, evidence_seed=["[page 110] Crémones et rallonges"],
    )
    events = await _collect(loop)

    # Le plan est streamé en « réflexion », l'action en « stage », la réponse en « message ».
    assert {"thinking": "Je lis la page 111."} in events
    assert any(e.get("stage", {}).get("label") == "Outil lire_pages" for e in events)
    joined = "".join(e["message"]["content"] for e in events if "message" in e)
    assert joined == "Réponse finale."
    assert "<sources>" not in joined and "<evidence>" not in joined

    # L'outil a reçu ses arguments ; l'historique suit le protocole assistant → tool.
    assert seen_args == [{"document_id": 405, "pages": [111]}]
    roles = [m["role"] for m in stream.calls[1]["context"]]
    assert roles == ["system", "user", "assistant", "tool"]
    tool_msg = stream.calls[1]["context"][3]
    assert tool_msg["tool_call_id"] == "abc123def"
    assert "(appels restants : 5/6 · images restantes : 8/8)" in tool_msg["content"]

    # Sorties : preuves, sources par index, journal.
    assert loop.evidence_citations == ["engager le tenon TGY3704 dans le boîtier (doc 405 p.111)"]
    assert "Engager le tenon" in loop.evidence_text
    assert loop.used_pages_by_index() == {1: [111]}
    assert loop.read_documents[405].pages == {110, 111}
    assert loop.trace["tool_calls"] == 1 and len(loop.trace["rounds"]) == 2
    assert loop.stopped_by == "final"
    assert stream.calls[0]["kwargs"]["tool_choice"] == "auto"
    assert [t["function"]["name"] for t in stream.calls[0]["kwargs"]["tools"]] == ["lire_pages"]


# ---------------------------------------------------------------------------
# Budgets et arrêt sur non-progrès
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exhaustion_forces_answer_with_tool_choice_none():
    calls = 0

    async def h(args):
        nonlocal calls
        calls += 1
        return ToolResult(text=f"r{calls}")

    stream = FakeStream(
        [
            _ev({"tool_calls": [_tc("aaaaaaaaa", "t", {"x": 1}), _tc("bbbbbbbbb", "t", {"x": 2})]}),
            _ev({"message": {"content": "Fini."}}),
        ]
    )
    loop = ReaderLoop(messages=_messages(), model="m", stream_fn=stream, tools=[_spec("t", h)], budget=LoopBudget(max_tool_calls=1))
    events = await _collect(loop)

    assert calls == 1, "le second appel du round dépasse le budget : non exécuté"
    tool_msgs = [m for m in stream.calls[1]["context"] if m["role"] == "tool"]
    assert "Budget d'appels épuisé" in tool_msgs[1]["content"]
    assert stream.calls[1]["kwargs"]["tool_choice"] == "none"
    assert loop.stopped_by == "budget"
    assert "".join(e["message"]["content"] for e in events if "message" in e) == "Fini."


@pytest.mark.asyncio
async def test_repeated_identical_call_stops_tools():
    async def h(args):
        return ToolResult(text="même résultat")

    stream = FakeStream(
        [
            _ev({"tool_calls": [_tc("aaaaaaaaa", "t", {"q": "x"})]}),
            _ev({"tool_calls": [_tc("bbbbbbbbb", "t", {"q": "x"})]}),  # identique
            _ev({"message": {"content": "Fini."}}),
        ]
    )
    loop = ReaderLoop(messages=_messages(), model="m", stream_fn=stream, tools=[_spec("t", h)])
    await _collect(loop)

    second_tool_msg = [m for m in stream.calls[2]["context"] if m["role"] == "tool"][1]
    assert "Appel identique déjà effectué" in second_tool_msg["content"]
    assert stream.calls[2]["kwargs"]["tool_choice"] == "none"
    assert loop.stopped_by == "repeat"
    assert loop.tool_calls_used == 1


@pytest.mark.asyncio
async def test_unknown_tool_is_reported_not_crashed():
    stream = FakeStream(
        [
            _ev({"tool_calls": [_tc("aaaaaaaaa", "inconnu", {})]}),
            _ev({"message": {"content": "Fini."}}),
        ]
    )
    loop = ReaderLoop(messages=_messages(), model="m", stream_fn=stream, tools=[_spec("t", None)])
    await _collect(loop)
    tool_msg = [m for m in stream.calls[1]["context"] if m["role"] == "tool"][0]
    assert "Outil inconnu" in tool_msg["content"]
    assert loop.trace["rounds"][0]["calls"][0]["error"] is True


# ---------------------------------------------------------------------------
# Images : message user après le tool, élagage sous la limite API
# ---------------------------------------------------------------------------


def _img(page: int) -> Dict[str, Any]:
    return {"b64": "QUJD", "document_id": 405, "page_no": page, "document_title": "Notice"}


@pytest.mark.asyncio
async def test_tool_images_go_in_a_user_message_and_are_pruned_over_the_api_cap():
    async def imgs(args):
        pages = args["pages"]
        return ToolResult(text="pages muettes", images=[_img(p) for p in pages], pages_read=[(405, p) for p in pages])

    stream = FakeStream(
        [
            _ev({"tool_calls": [_tc("aaaaaaaaa", "lire_pages", {"pages": [1, 2, 3, 4, 5]})]}),
            _ev({"tool_calls": [_tc("bbbbbbbbb", "lire_pages", {"pages": [6, 7, 8, 9, 10]})]}),
            _ev({"message": {"content": "Fini."}}),
        ]
    )
    loop = ReaderLoop(
        messages=_messages(), model="m", stream_fn=stream,
        tools=[_spec("lire_pages", imgs, max_images=5)], budget=LoopBudget(max_tool_images=10),
    )
    await _collect(loop)

    # Round 2 : 5 images dans un message user après le tool.
    ctx2 = stream.calls[1]["context"]
    assert [m["role"] for m in ctx2] == ["system", "user", "assistant", "tool", "user"]
    assert len(ctx2[4]["images"]) == 5
    assert "Image 1 = document 405 « Notice », page 1" in ctx2[4]["content"]

    # Round 3 : 10 images cumulées > 8 → les plus anciennes sont élaguées, placeholder texte.
    ctx3 = stream.calls[2]["context"]
    total = sum(len(m.get("images") or []) for m in ctx3)
    assert total <= 8
    pruned = [m for m in ctx3 if m["role"] == "user" and "Images déjà vues" in str(m.get("content"))]
    assert pruned and pruned[0]["images"] == []
    assert "doc 405 p.1" in pruned[0]["content"]
    assert loop.tool_images_used == 10


@pytest.mark.asyncio
async def test_per_call_and_cumulative_image_caps():
    async def imgs(args):
        return ToolResult(text="ok", images=[_img(p) for p in range(1, 6)])

    stream = FakeStream(
        [
            _ev({"tool_calls": [_tc("aaaaaaaaa", "lire_pages", {})]}),
            _ev({"message": {"content": "Fini."}}),
        ]
    )
    loop = ReaderLoop(
        messages=_messages(), model="m", stream_fn=stream,
        tools=[_spec("lire_pages", imgs, max_images=2)], budget=LoopBudget(max_tool_images=8),
    )
    await _collect(loop)
    tool_msg = [m for m in stream.calls[1]["context"] if m["role"] == "tool"][0]
    assert "3 image(s) non jointe(s)" in tool_msg["content"]
    assert len(stream.calls[1]["context"][-1]["images"]) == 2


# ---------------------------------------------------------------------------
# Contrôle de sortie → retour outil → réponse réparée
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_output_check_feedback_round_then_repaired():
    async def chercher_code(args):
        return ToolResult(text=f"{args['code']} — aucun chunk ne contient cette référence (31 documents).")

    def check(text, evidence, citations):
        # Mini-modèle du contrôle réel : un code cité est étayé s'il figure dans ce que le
        # lecteur a lu — y compris le résultat d'outil qui constate son ABSENCE (« TGY3710 —
        # aucun chunk… »), ce qui permet de dire honnêtement qu'il n'est pas documenté.
        if "TGY3710" in text and "TGY3710" not in evidence:
            return {"ok": False, "unsupported_claims": ["TGY3710"], "unsupported_codes": ["TGY3710"], "feedback": "CONTRÔLE : TGY3710 absent."}
        return {"ok": True, "unsupported_claims": [], "unsupported_codes": [], "feedback": None}

    stream = FakeStream(
        [
            _ev({"message": {"content": "La rallonge est la TGY3710."}}),
            _ev({"tool_calls": [_tc("aaaaaaaaa", "chercher_code", {"code": "TGY3710"})]}),
            _ev({"message": {"content": "Les documents ne mentionnent pas de TGY3710 ; la rallonge documentée est TGY3704."}}),
        ]
    )
    loop = ReaderLoop(messages=_messages(), model="m", stream_fn=stream, tools=[_spec("chercher_code", chercher_code)], output_check=check)
    events = await _collect(loop)

    joined = "".join(e["message"]["content"] for e in events if "message" in e)
    assert "TGY3704" in joined and "TGY3710 ;" in joined
    # Le brouillon et la consigne de contrôle sont dans l'historique du tour…
    ctx = stream.calls[1]["context"]
    assert ctx[-2] == {"role": "assistant", "content": "La rallonge est la TGY3710."}
    assert ctx[-1]["role"] == "user" and "CONTRÔLE" in ctx[-1]["content"]
    # …et la vérification finale porte l'action « repaired ».
    assert loop.trace["control_rounds"] == 1
    assert loop.verification["ok"] is True
    assert loop.verification["action"] == "repaired"
    assert loop.verification["unsupported_before_repair"] == ["TGY3710"]
    assert any(e.get("stage", {}).get("key") == "control" for e in events)


@pytest.mark.asyncio
async def test_output_check_still_ko_after_feedback_is_flagged_not_looped():
    def check(text, evidence, citations):
        return {"ok": False, "unsupported_claims": ["X1"], "unsupported_codes": ["X1"], "feedback": "CONTRÔLE : X1."}

    stream = FakeStream(
        [
            _ev({"message": {"content": "Brouillon X1."}}),
            _ev({"message": {"content": "Toujours X1."}}),
        ]
    )
    loop = ReaderLoop(messages=_messages(), model="m", stream_fn=stream, tools=[_spec("t", None)], output_check=check)
    events = await _collect(loop)
    assert len(stream.calls) == 2, "un seul round de contrôle"
    assert loop.verification["action"] == "flagged"
    assert "".join(e["message"]["content"] for e in events if "message" in e) == "Toujours X1."


# ---------------------------------------------------------------------------
# Repli : 400 avec outils → un jet sans outils sur le pack initial
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_400_with_tools_degrades_to_no_tools_once():
    stream = FakeStream([_ev({"message": {"content": "Réponse sans outils."}})], raise_first_400=True)
    loop = ReaderLoop(messages=_messages(), model="m", stream_fn=stream, tools=[_spec("t", None)])
    events = await _collect(loop)
    assert loop.degraded is True
    assert str(loop.trace["degraded"]).startswith("tools_unavailable")
    assert "tools" in stream.calls[0]["kwargs"] and "tools" not in stream.calls[1]["kwargs"]
    assert "".join(e["message"]["content"] for e in events if "message" in e) == "Réponse sans outils."


@pytest.mark.asyncio
async def test_no_tools_means_single_plain_call():
    stream = FakeStream([_ev({"message": {"content": "Direct."}})])
    loop = ReaderLoop(messages=_messages(), model="m", stream_fn=stream, tools=[])
    events = await _collect(loop)
    assert stream.calls[0]["kwargs"] == {}
    assert loop.final_text == "Direct."
    assert loop.trace["rounds"][0]["final"] is True
