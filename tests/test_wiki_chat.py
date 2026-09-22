"""Le tour de chat : boucle d'outils, anomalies injectées, coupes filtrées, citations vérifiées."""
from __future__ import annotations

import asyncio
import copy
import json
import re

import pytest

from app.config import settings
from app.services import wiki_service
from app.services.wiki_chat_service import (
    WikiAnswer,
    coupes_des_pages,
    extract_anomalies,
    extract_citations,
    preparer_images,
    read_wiki_page,
)
from app.services.wiki_service import load_snapshot, wiki_root


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot(wiki_root())


def _events(chunks):
    return [json.loads(c[6:]) for c in chunks if c.startswith("data: ")]


def _collect(answer):
    async def run():
        return [chunk async for chunk in answer.run()]

    return _events(asyncio.run(run()))


def _outil(nom, arguments, identifiant="call_1"):
    return {
        "id": identifiant,
        "type": "function",
        "function": {"name": nom, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


# ---------------------------------------------------------------------------
# Citations et anomalies
# ---------------------------------------------------------------------------


def test_extract_citations_dedupes_and_flags_unknown(snapshot):
    text = (
        "La parclose 76507 (/profiles/perform76-parcloses.md) tient 44 mm. "
        "Voir aussi /profiles/perform76-parcloses.md et /profiles/page-inventee.md ; "
        "le schéma est dans raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 5."
    )
    cites = extract_citations(text, snapshot)
    assert [c["path"] for c in cites] == [
        "/profiles/perform76-parcloses.md",
        "/profiles/page-inventee.md",
    ]
    assert cites[0]["exists"] is True and cites[0]["type"] == "Profilé"
    assert cites[1]["exists"] is False and cites[1]["title"] == "page inventee"


def test_extract_citations_ignores_urls_and_index(snapshot):
    cites = extract_citations("Rien ici : https://exemple.fr/x/y.md ni (/index.md).", snapshot)
    assert [c["path"] for c in cites] == ["/index.md"]
    assert cites[0]["exists"] is True


def test_extract_anomalies():
    found = extract_anomalies("Attention CTR-17 et INC-02 ; INC-02 encore, puis VER-35.")
    assert [a["id"] for a in found] == ["CTR-17", "INC-02", "VER-35"]
    assert found[0]["path"] == "/anomalies/contradictions-entre-sources.md"
    assert found[1]["path"] == "/anomalies/incoherences-internes.md"
    assert found[2]["path"] == "/anomalies/informations-a-verifier.md"


# ---------------------------------------------------------------------------
# Lecture de page
# ---------------------------------------------------------------------------


def test_read_wiki_page_refuse_un_chemin_hors_wiki(snapshot):
    assert "Chemin invalide" in read_wiki_page("../../etc/passwd", snapshot)
    assert "Page introuvable" in read_wiki_page("/profiles/inexistante.md", snapshot)
    assert read_wiki_page("/profiles/perform76-parcloses.md", snapshot).startswith("---")


# ---------------------------------------------------------------------------
# La boucle d'outils
# ---------------------------------------------------------------------------


def test_le_modele_cherche_puis_repond(snapshot, monkeypatch):
    """Un tour normal : un appel d'outil, puis la réponse, avec l'étape et les pages livrées."""
    contextes = []

    async def fake_stream(message, *, model, context, max_tokens, temperature, **kwargs):
        contextes.append(copy.deepcopy(context))
        if len(contextes) == 1:
            assert kwargs["tools"] and kwargs["tool_choice"] == "auto"
            yield json.dumps({"tool_calls": [_outil("chercher", {"mots_cles": "parclose 76507"})]})
            yield json.dumps({"usage": {"prompt_tokens": 5000, "completion_tokens": 30}})
            return
        yield json.dumps({"message": {"content": "La parclose **76507** "}})
        yield json.dumps({"message": {"content": "(/profiles/perform76-parcloses.md) tient 44 mm."}})
        yield json.dumps(
            {"usage": {"prompt_tokens": 18000, "completion_tokens": 60,
                       "prompt_tokens_details": {"cached_tokens": 4600}}}
        )

    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "high")
    answer = WikiAnswer(
        question="Quelle parclose pour 44 mm ?",
        history=[{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Bonjour !"}],
        snapshot=snapshot,
        model="mistral-small-latest",
        stream_fn=fake_stream,
    )
    events = _collect(answer)
    kinds = [next(iter(e)) for e in events]
    assert kinds[0] == "etape" and kinds[-1] == "sources"
    assert "message" in kinds

    etape = events[0]["etape"]
    assert etape["outil"] == "chercher" and etape["libelle"] == "Recherche"
    assert etape["detail"] == "parclose 76507"
    # chercher livre les premières pages entières : elles comptent comme lues.
    assert "/profiles/perform76-parcloses.md" in etape["pages"]

    # Le premier contexte : prompt permanent, historique, question.
    premier = contextes[0]
    assert premier[0]["role"] == "system" and premier[0]["content"] == snapshot.system_prompt
    assert premier[-1]["content"] == "Quelle parclose pour 44 mm ?"

    # Le second : l'appel d'outil et son résultat ont été ajoutés (le bloc d'anomalies, qui
    # est injecté après, peut fermer la liste).
    second = contextes[1]
    resultats = [m for m in second if m["role"] == "tool"]
    assert len(resultats) == 1 and resultats[0]["tool_call_id"] == "call_1"
    assert "===== PAGE 1 :" in resultats[0]["content"]
    assert any(m.get("tool_calls") for m in second)

    assert answer.text.startswith("La parclose **76507**")
    assert answer.trace["appels"] == 2
    assert answer.trace["prompt_tokens"] == 23000
    assert answer.trace["completion_tokens"] == 90
    assert answer.trace["cited_pages"] == ["/profiles/perform76-parcloses.md"]
    assert answer.trace["steps"][0]["outil"] == "chercher"
    assert wiki_service.last_call()["appels"] == 2


def test_le_serveur_injecte_les_anomalies(snapshot):
    """La règle 2 ne dépend pas de la discipline du modèle : le serveur pousse les entrées."""
    contextes = []

    async def fake_stream(message, *, context, **kwargs):
        contextes.append(copy.deepcopy(context))
        if len(contextes) == 1:
            yield json.dumps(
                {"tool_calls": [_outil("chercher", {"mots_cles": "garantie ferrure Technal"})]}
            )
            return
        yield json.dumps({"message": {"content": "10 ans (/fournisseurs/technal.md), voir CTR-09."}})

    answer = WikiAnswer(
        question="La ferrure Technal est garantie combien de temps ?",
        history=[],
        snapshot=snapshot,
        stream_fn=fake_stream,
    )
    _collect(answer)
    injecte = [
        m for m in contextes[-1]
        if m["role"] == "system" and "ENTRÉES D'ANOMALIE" in str(m.get("content"))
    ]
    assert injecte, "aucune entrée d'anomalie injectée"
    # Les entrées poussées sont celles rapprochées de la question et des pages chargées : on
    # vérifie qu'elles portent bien un identifiant de registre, pas laquelle a été retenue.
    assert re.search("(INC|CTR|VER)-[0-9]+", injecte[-1]["content"])
    assert answer.anomalies == [{"id": "CTR-09", "path": "/anomalies/contradictions-entre-sources.md"}]


def test_relance_quand_aucune_page_n_a_ete_chargee(snapshot):
    """Répondre sans avoir rien ouvert est la faute la plus fréquente : on renvoie lire, une fois."""
    contextes = []

    async def fake_stream(message, *, context, **kwargs):
        contextes.append(copy.deepcopy(context))
        if len(contextes) == 1:
            yield json.dumps({"message": {"content": "Le wiki ne couvre pas ce sujet."}})
            return
        yield json.dumps({"tool_calls": [_outil("chercher", {"mots_cles": "régions climatiques"})]})
        if len(contextes) >= 3:
            return

    answer = WikiAnswer(question="Quelle région climatique pour le 59 ?", history=[],
                        snapshot=snapshot, stream_fn=fake_stream)
    events = _collect(answer)

    # Ce qui avait commencé à s'afficher est effacé : c'était un faux refus.
    assert any("reset" in e for e in events)
    relance = [m for m in contextes[1] if m["role"] == "system" and "Tu n'as chargé aucune page" in str(m.get("content"))]
    assert relance, "la relance n'a pas été injectée"
    assert len(contextes) >= 2


def test_boucle_bornee(snapshot):
    """Un modèle qui n'en finit pas d'appeler des outils s'arrête sur un message clair."""

    async def fake_stream(message, *, context, **kwargs):
        yield json.dumps({"tool_calls": [_outil("chercher", {"mots_cles": "x"})]})

    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake_stream)
    events = _collect(answer)
    assert "error" in events[-1]
    assert "allers-retours" in events[-1]["error"]


def test_sans_reasoning_effort(snapshot, monkeypatch):
    captured = {}

    async def fake_stream(message, **kwargs):
        captured.update(kwargs)
        yield json.dumps({"tool_calls": [_outil("lire_page", {"chemin": "/gammes/lumine.md"})]})
        yield json.dumps({"message": {"content": ""}})

    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "")
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake_stream)

    async def run():
        async for _ in answer.run():
            break

    asyncio.run(run())
    assert "reasoning_effort" not in captured
    assert answer.model == settings.MODEL_FAST


def test_lire_anomalie(snapshot):
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot)
    contenu = answer._executer("lire_anomalie", {"identifiant": "CTR-09"}, [])
    assert "CTR-09" in contenu
    assert "Outil inconnu" in answer._executer("inexistant", {}, [])


# ---------------------------------------------------------------------------
# Les coupes
# ---------------------------------------------------------------------------


PAGE_COUPES = {
    "chemin": "/profiles/perform76-parcloses.md",
    "corps": (
        "| Réf. | Vitrage | Coupe |\n"
        "| --- | --- | --- |\n"
        "| 2452 | 16 | ![Parclose 2452](/assets/profiles/perform76/parcloses/parclose-2452.png) |\n"
        "| 2451 | 18 | ![Parclose 2451](/assets/profiles/perform76/parcloses/parclose-2451.png) |\n"
    ),
}


def test_coupes_des_pages_apparie_la_reference_et_son_image():
    coupes = coupes_des_pages([PAGE_COUPES])
    assert coupes["2452"] == "/assets/profiles/perform76/parcloses/parclose-2452.png"
    assert coupes["2451"] == "/assets/profiles/perform76/parcloses/parclose-2451.png"


def test_une_coupe_inventee_est_retiree(snapshot):
    texte = "Voici la coupe.\n\n![Parclose 9999](/assets/profiles/perform76/parcloses/parclose-9999.png)"
    sortie, journal = preparer_images(texte, "coupe de la 9999 ?", [PAGE_COUPES], snapshot.root)
    # L'image est retirée ; le chemin ne subsiste que dans la note qui explique le retrait.
    assert "![Parclose 9999](" not in sortie
    assert journal["inexistantes"] == ["/assets/profiles/perform76/parcloses/parclose-9999.png"]
    assert "absente du wiki" in sortie


def test_l_image_de_la_ligne_voisine_est_remplacee(snapshot):
    """Le couple référence/image de la page fait foi, pas le choix du modèle."""
    texte = "La coupe de la 2452 :\n\n![Parclose 2451](/assets/profiles/perform76/parcloses/parclose-2451.png)"
    sortie, journal = preparer_images(texte, "coupe de la parclose 2452 ?", [PAGE_COUPES], snapshot.root)
    assert "parclose-2451.png" not in sortie
    assert "parclose-2452.png" in sortie
    assert journal["hors_sujet"] == ["/assets/profiles/perform76/parcloses/parclose-2451.png"]
    assert journal["ajoutees"] == ["/assets/profiles/perform76/parcloses/parclose-2452.png"]


def test_hors_demande_de_coupe_les_images_valides_passent(snapshot):
    texte = "![Parclose 2452](/assets/profiles/perform76/parcloses/parclose-2452.png)"
    sortie, journal = preparer_images(texte, "quelle parclose pour 16 mm ?", [PAGE_COUPES], snapshot.root)
    assert "parclose-2452.png" in sortie
    assert journal["hors_sujet"] == []
