"""Le tour vocal : ce qui se dit, le découpage en phrases, le biais de vocabulaire, le flux."""
from __future__ import annotations

import asyncio
import json

import pytest
from sqlmodel import select

from app.config import settings
from app.models.message import Message
from app.services import vocal_service, wiki_chat_service, wiki_service
from app.services.vocal_service import Phraseur, TourVocal, biais_vocabulaire, texte_parle
from app.services.wiki_chat_service import WikiAnswer
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot(wiki_root())


@pytest.fixture(autouse=True)
def _attentes_vierges():
    vocal_service._attentes.clear()
    yield
    vocal_service._attentes.clear()


def _events(chunks):
    return [json.loads(c[6:]) for c in chunks if c.startswith("data: ")]


def _outil(nom, arguments, identifiant="call_1"):
    return {
        "id": identifiant,
        "type": "function",
        "function": {"name": nom, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _synthese_factice(journal):
    """Deux fragments par segment ; le journal garde ce qui a été envoyé à la synthèse."""

    async def synthese(texte):
        journal.append(texte)
        yield "QUJD"  # « ABC »
        yield "REVG"  # « DEF »

    return synthese


# ---------------------------------------------------------------------------
# Ce qui se dit
# ---------------------------------------------------------------------------


def test_texte_parle_retire_les_citations_et_lit_les_anomalies():
    texte = (
        "La parclose 76507 (/profiles/perform76-parcloses.md) tient 44 mm en ouvrant "
        "(/profiles/perform76-parcloses.md, /gammes/perform76.md). Attention CTR-03 et INC-07, "
        "voir /anomalies/informations-a-verifier.md."
    )
    dit = texte_parle(texte)
    assert ".md" not in dit and "/" not in dit
    assert "contradiction entre sources 3" in dit and "incohérence interne 7" in dit
    assert "44 millimètres" in dit
    assert dit.startswith("La parclose 76507 tient 44 millimètres en ouvrant.")


def test_texte_parle_deplie_unites_symboles_et_markdown():
    dit = texte_parle("**Uw** de 1,3 W/m².K → soit 12 % de mieux, ⚠️ n°2 sur 5 m² et 20 °C. [Voir](/x.md) `code`.")
    assert dit == (
        "U w de 1,3 watts par mètre carré kelvin, donc soit 12 pour cent de mieux, attention, "
        "numéro 2 sur 5 mètres carrés et 20 degrés. Voir code."
    )


def test_texte_parle_aplatit_listes_et_tableaux():
    texte = "# Titre\n\n- Première étape.\n- Deuxième étape.\n\n| Réf | Vitrage |\n| --- | --- |\n| 76507 | 44 mm |\n\n![Coupe](/assets/x.png)\n"
    dit = texte_parle(texte)
    assert dit == "Titre\nPremière étape.\nDeuxième étape.\nRéf, Vitrage.\n76507, 44 millimètres."
    assert texte_parle("") == "" and texte_parle("![Coupe](/assets/x.png)") == ""


# ---------------------------------------------------------------------------
# Le découpage en phrases
# ---------------------------------------------------------------------------


def test_phraseur_premiere_phrase_seule_puis_groupes():
    p = Phraseur(groupe=60, minimum=10)
    assert p.pousser("La parclose 76507 tient 44") == []
    # La première phrase part dès qu'elle est complète.
    assert p.pousser(" millimètres. Elle se") == ["La parclose 76507 tient 44 millimètres."]
    # Les suivantes attendent d'atteindre la taille du groupe.
    assert p.pousser(" monte en ouvrant. Voir aussi la 76508.") == []
    assert p.pousser(" Elle accepte 48 millimètres, sur dormant seulement. Fin.") == [
        "Elle se monte en ouvrant. Voir aussi la 76508. Elle accepte 48 millimètres, sur dormant seulement."
    ]
    assert p.vider() == ["Fin."]
    assert p.vider() == []


def test_phraseur_minimum_paragraphe_et_decimales():
    p = Phraseur(groupe=200, minimum=40)
    # « Oui. » attend la phrase suivante ; 44.2 n'est pas une fin de phrase.
    assert p.pousser("Oui. ") == []
    assert p.pousser("Le STADIP 44.2/16/4 passe sur PERFORM76 en ouvrant. Le reste") == [
        "Oui. Le STADIP 44.2/16/4 passe sur PERFORM76 en ouvrant."
    ]
    # Un saut de paragraphe force la coupe.
    assert p.pousser(" suit ici\n\nAutre paragraphe") == ["Le reste suit ici"]
    assert p.vider() == ["Autre paragraphe"]


def test_phraseur_coupe_un_long_passage_sans_ponctuation():
    p = Phraseur(groupe=50, minimum=10)
    segments = p.pousser("mot " * 40)
    assert segments and all(len(s) <= 60 for s in segments)
    p.reinitialiser()
    assert p.vider() == []


# ---------------------------------------------------------------------------
# Biais de vocabulaire et prompt vocal
# ---------------------------------------------------------------------------


def test_biais_vocabulaire(snapshot):
    biais = biais_vocabulaire(snapshot.index)
    assert 30 <= len(biais) <= vocal_service.BIAIS_MAXIMUM
    assert biais[:3] == ["PROFERM", "LIA", "parclose"]
    assert "LUMINE" in biais and "PERFORM" in biais
    assert all(" " not in t and "," not in t for t in biais)
    assert len({t.lower() for t in biais}) == len(biais)


def test_le_prompt_vocal_ajoute_la_forme_parlee_devant_les_regles(snapshot):
    vocales = wiki_service.CONSIGNES_VOCALES_PATH.read_text(encoding="utf-8").rstrip()
    generales = wiki_service.CONSIGNES_PATH.read_text(encoding="utf-8").rstrip()
    assert snapshot.vocal_prompt.startswith(vocales)
    assert generales in snapshot.vocal_prompt
    assert "===== VOCABULAIRE DU WIKI" in snapshot.vocal_prompt
    assert "===== INDEX DES ANOMALIES" in snapshot.vocal_prompt
    assert snapshot.vocal_cache_key.startswith("lia-vocal-") and snapshot.vocal_cache_key != snapshot.cache_key


def test_une_consigne_vocale_modifiee_recharge_le_wiki(tmp_path, monkeypatch):
    signature = wiki_service._signature(wiki_root())
    assert len(signature) == 6
    faux = tmp_path / "vocal_consignes.md"
    faux.write_text("Parle.", encoding="utf-8")
    monkeypatch.setattr(wiki_service, "CONSIGNES_VOCALES_PATH", faux)
    assert wiki_service._signature(wiki_root()) != signature
    snap = load_snapshot(wiki_root())
    assert snap.vocal_prompt.startswith("Parle.\n\n")


# ---------------------------------------------------------------------------
# Le tour : texte et audio entrelacés
# ---------------------------------------------------------------------------


def _collect(tour):
    async def run():
        return [chunk async for chunk in tour.run()]

    return _events(asyncio.run(run()))


def test_le_tour_vocal_dit_la_reponse_phrase_par_phrase(snapshot, monkeypatch):
    appels = []

    async def fake_stream(message, *, model, context, max_tokens, temperature, **kwargs):
        appels.append(context)
        if len(appels) == 1:
            assert kwargs["prompt_cache_key"] == snapshot.vocal_cache_key
            assert context[0]["content"] == snapshot.vocal_prompt
            yield json.dumps({"tool_calls": [_outil("chercher", {"mots_cles": "parclose 76507"})]})
            yield json.dumps({"usage": {"prompt_tokens": 5000, "completion_tokens": 30}})
            return
        yield json.dumps({"message": {"content": "La parclose 76507 tient 44 millimètres en ouvrant "}})
        yield json.dumps({"message": {"content": "(/profiles/perform76-parcloses.md). Elle ne se monte pas en dormant, "}})
        yield json.dumps({"message": {"content": "où c'est la 76508 qui convient."}})
        yield json.dumps({"usage": {"prompt_tokens": 9000, "completion_tokens": 60}})

    journal = []
    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "")
    answer = WikiAnswer(
        question="Quelle parclose pour 44 millimètres ?", history=[], snapshot=snapshot, stream_fn=fake_stream,
        system_prompt=snapshot.vocal_prompt, cache_key=snapshot.vocal_cache_key, images=False,
    )
    tour = TourVocal(answer=answer, synthese_fn=_synthese_factice(journal))
    events = _collect(tour)

    types = [next(iter(e)) for e in events]
    assert "etape" in types and "sources" in types
    # La phrase d'attente est dite au premier appel d'outil, avant toute réponse.
    attente = next(e for e in events if e.get("phrase") and e["phrase"]["index"] == -1)
    assert attente["phrase"]["attente"] is True and attente["phrase"]["texte"] in vocal_service.PHRASES_ATTENTE
    assert types.index("etape") < types.index("phrase")
    # Deux segments dits : la première phrase seule, puis le reste ; les chemins ne sont pas lus.
    phrases = [e["phrase"] for e in events if e.get("phrase") and e["phrase"]["index"] >= 0]
    assert [p["index"] for p in phrases] == [0, 1]
    assert phrases[0]["texte"] == "La parclose 76507 tient 44 millimètres en ouvrant (/profiles/perform76-parcloses.md)."
    assert journal[1:] == [
        "La parclose 76507 tient 44 millimètres en ouvrant.",
        "Elle ne se monte pas en dormant, où c'est la 76508 qui convient.",
    ]
    audio = [e["audio"] for e in events if e.get("audio")]
    assert [a["index"] for a in audio] == [-1, -1, 0, 0, 1, 1]
    assert all(a["data"] in ("QUJD", "REVG") for a in audio)
    assert tour.mesures["phrases"] == 2 and tour.mesures["premier_son_ms"] is not None
    assert answer.sources[0]["path"] == "/profiles/perform76-parcloses.md"


def test_le_tour_vocal_ne_dit_ni_le_preambule_ni_la_reponse_hative(snapshot, monkeypatch):
    """Le texte écrit avant un appel d'outil, et celui effacé par la relance, ne se disent pas."""
    appels = []

    async def fake_stream(message, *, model, context, max_tokens, temperature, **kwargs):
        appels.append(context)
        if len(appels) == 1:
            # Réponse hâtive sans la moindre page : le serveur renvoie lire (reset).
            yield json.dumps({"message": {"content": "La parclose est la 76500, sans aucun doute. Voilà tout."}})
            yield json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            return
        if len(appels) == 2:
            # Un préambule, puis un appel d'outil : le préambule est effacé (reset).
            yield json.dumps({"message": {"content": "Je vais chercher dans le wiki tout de suite. "}})
            yield json.dumps({"tool_calls": [_outil("chercher", {"mots_cles": "parclose 44"})]})
            yield json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            return
        yield json.dumps({"message": {"content": "La parclose 76507 convient pour 44 millimètres (/profiles/perform76-parcloses.md)."}})
        yield json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    journal = []
    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "")
    answer = WikiAnswer(question="Quelle parclose pour 44 ?", history=[], snapshot=snapshot, stream_fn=fake_stream, images=False)
    tour = TourVocal(answer=answer, synthese_fn=_synthese_factice(journal), attente=False)
    events = _collect(tour)
    assert sum(1 for e in events if e.get("reset")) == 2
    assert journal == ["La parclose 76507 convient pour 44 millimètres."]
    phrases = [e["phrase"] for e in events if e.get("phrase")]
    # Les segments jetés n'ont jamais été envoyés en synthèse : ils ne consomment pas d'index.
    assert len(phrases) == 1 and phrases[0]["index"] == 0


def test_le_tour_vocal_survit_a_une_synthese_en_panne(snapshot, monkeypatch):
    async def fake_stream(message, *, model, context, max_tokens, temperature, **kwargs):
        yield json.dumps({"tool_calls": [_outil("chercher", {"mots_cles": "parclose 76507"})]})
        yield json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        return
        yield  # pragma: no cover

    async def fake_stream_2(message, *, model, context, max_tokens, temperature, **kwargs):
        yield json.dumps({"message": {"content": "La parclose 76507 tient 44 millimètres (/profiles/perform76-parcloses.md)."}})
        yield json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    flux = [fake_stream, fake_stream_2]

    async def stream(message, **kwargs):
        async for chunk in flux.pop(0)(message, **kwargs):
            yield chunk

    async def synthese_en_panne(texte):
        raise RuntimeError("HTTP 503")
        yield  # pragma: no cover

    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "")
    answer = WikiAnswer(question="Quelle parclose pour 44 ?", history=[], snapshot=snapshot, stream_fn=stream, images=False)
    tour = TourVocal(answer=answer, synthese_fn=synthese_en_panne)
    events = _collect(tour)
    assert not any(e.get("audio") for e in events)
    assert sum(1 for e in events if e.get("avertissement")) == 1
    assert any(e.get("sources") for e in events)
    assert answer.text.startswith("La parclose 76507")


# ---------------------------------------------------------------------------
# La route et la page
# ---------------------------------------------------------------------------


def test_page_vocale_redirige_puis_sert(client, responsable_headers):
    r = client.get("/vocal", follow_redirects=False)
    assert r.status_code == 303 and "/login" in (r.headers.get("location") or "")
    r = client.get("/vocal", headers=responsable_headers, follow_redirects=False)
    assert r.status_code == 200 and "/api/vocal/tour" in r.text and "lia-capteur" in r.text


@pytest.fixture
def conversation_vocale(client, responsable_headers):
    r = client.post("/api/conversations", headers=responsable_headers, json={"title": "Vocale", "mode": "vocal"})
    assert r.status_code == 201 and r.json()["mode"] == "vocal"
    conv_id = r.json()["id"]
    yield conv_id
    client.delete(f"/api/conversations/{conv_id}", headers=responsable_headers)


def _flux_reponse(texte):
    appels = []

    async def fake_stream(message, *, model, context, max_tokens, temperature, **kwargs):
        appels.append(context)
        if len(appels) == 1:
            yield json.dumps({"tool_calls": [_outil("chercher", {"mots_cles": "parclose"})]})
            yield json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})
            return
        yield json.dumps({"message": {"content": texte}})
        yield json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    return fake_stream


def test_les_conversations_vocales_sont_listees_a_part(client, responsable_headers, conversation_vocale):
    r = client.get("/api/conversations", headers=responsable_headers)
    assert all(c["id"] != conversation_vocale for c in r.json())
    r = client.get("/api/conversations", headers=responsable_headers, params={"mode": "vocal"})
    assert any(c["id"] == conversation_vocale for c in r.json())
    assert client.get("/api/conversations", headers=responsable_headers, params={"mode": "sms"}).status_code == 422
    r = client.post("/api/conversations", headers=responsable_headers, json={"title": "X", "mode": "sms"})
    assert r.status_code == 201 and r.json()["mode"] == "chat"
    client.delete(f"/api/conversations/{r.json()['id']}", headers=responsable_headers)


def test_tour_vocal_ecrit_stream_et_persiste(client, responsable_headers, conversation_vocale, db_session, monkeypatch):
    journal = []
    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "")
    monkeypatch.setattr(wiki_chat_service, "chat_stream", _flux_reponse(
        "La parclose 76507 tient 44 millimètres en ouvrant (/profiles/perform76-parcloses.md)."
    ))
    monkeypatch.setattr(vocal_service, "synthese", _synthese_factice(journal))

    r = client.post(
        f"/api/vocal/tour?conversation_id={conversation_vocale}", headers=responsable_headers,
        json={"texte": "Quelle parclose pour 44 millimètres ?"},
    )
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = _events(r.text.splitlines())
    assert events[0] == {"transcription": {"texte": "Quelle parclose pour 44 millimètres ?", "ecrite": True}}
    assert any(e.get("audio", {}).get("index") == 0 for e in events)
    done = events[-1]
    assert done["done"] is True and done["message_id"]
    assert done["trace"]["vocal"]["phrases"] == 1 and done["trace"]["vocal"]["voix"] == settings.VOCAL_VOIX
    assert done["trace"]["vocal"]["transcription_ms"] is None
    assert journal[-1] == "La parclose 76507 tient 44 millimètres en ouvrant."

    messages = db_session.exec(
        select(Message).where(Message.conversation_id == conversation_vocale).order_by(Message.id)
    ).all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "Quelle parclose pour 44 millimètres ?"
    assert messages[1].id == done["message_id"] and messages[1].metadata_json["vocal"] is True
    assert json.loads(messages[1].sources)[0]["path"] == "/profiles/perform76-parcloses.md"


def test_tour_vocal_transcrit_l_audio(client, responsable_headers, conversation_vocale, db_session, monkeypatch):
    recu = {}

    async def fake_transcrire(audio, type_mime, biais):
        recu.update(audio=audio, type_mime=type_mime, biais=list(biais))
        return {"texte": "Quelle parclose pour 44 millimètres ?", "ms": 420, "secondes_audio": 3}

    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "")
    monkeypatch.setattr(vocal_service, "transcrire", fake_transcrire)
    monkeypatch.setattr(wiki_chat_service, "chat_stream", _flux_reponse("La parclose 76507 (/profiles/perform76-parcloses.md)."))
    monkeypatch.setattr(vocal_service, "synthese", _synthese_factice([]))

    r = client.post(
        f"/api/vocal/tour?conversation_id={conversation_vocale}",
        headers={**responsable_headers, "Content-Type": "audio/wav"}, content=b"RIFF\x00\x00\x00\x00WAVEfmt ",
    )
    assert r.status_code == 200
    events = _events(r.text.splitlines())
    assert events[0] == {"transcription": {"texte": "Quelle parclose pour 44 millimètres ?", "ms": 420}}
    assert recu["audio"].startswith(b"RIFF") and recu["type_mime"] == "audio/wav" and "PROFERM" in recu["biais"]
    assert events[-1]["trace"]["vocal"]["transcription_ms"] == 420
    assert events[-1]["trace"]["vocal"]["modele_transcription"] == settings.VOCAL_MODELE_TRANSCRIPTION


def test_tour_vocal_silence_et_refus(client, responsable_headers, conversation_vocale, db_session, monkeypatch):
    async def rien(audio, type_mime, biais):
        return {"texte": "", "ms": 100, "secondes_audio": 0}

    monkeypatch.setattr(vocal_service, "transcrire", rien)
    r = client.post(
        f"/api/vocal/tour?conversation_id={conversation_vocale}",
        headers={**responsable_headers, "Content-Type": "audio/wav"}, content=b"RIFF....",
    )
    events = _events(r.text.splitlines())
    assert events == [{"transcription": {"texte": "", "ms": 100}}, {"done": True, "vide": True}]
    assert not db_session.exec(select(Message).where(Message.conversation_id == conversation_vocale)).all()

    base = f"/api/vocal/tour?conversation_id={conversation_vocale}"
    assert client.post(base, headers={**responsable_headers, "Content-Type": "text/plain"}, content=b"x").status_code == 415
    assert client.post(base, headers=responsable_headers, json={"texte": "  "}).status_code == 422
    assert client.post(base, headers={**responsable_headers, "Content-Type": "audio/wav"}, content=b"").status_code == 422
    assert client.post("/api/vocal/tour?conversation_id=999999", headers=responsable_headers, json={"texte": "x"}).status_code == 404
    assert client.post(base, json={"texte": "x"}).status_code == 401
