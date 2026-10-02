"""Le tour de chat : carte, lecture, raisonnement renvoyé, budget, anomalies, coupes, citations."""
from __future__ import annotations

import asyncio
import copy
import json
import re

import pytest

from app.config import settings
from app.services import wiki_chat_service, wiki_service
from app.services.wiki_chat_service import (
    WikiAnswer,
    coupes_des_pages,
    extract_anomalies,
    extract_citations,
    preparer_images,
    read_wiki_page,
    resume_precedent,
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
        "function": {"name": nom, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)},
    }


def _chercher(*requetes, identifiant="call_1"):
    return _outil("chercher", {"requetes": list(requetes)}, identifiant)


def _lire(chemin, *sections, identifiant="call_2"):
    lecture = {"chemin": chemin}
    if sections:
        lecture["sections"] = list(sections)
    return _outil("lire", {"lectures": [lecture]}, identifiant)


def _scenario(*appels, contextes=None, options=None):
    """Un faux flux : chaque appel du tour reçoit les événements du suivant de la liste.

    Un appel est une liste d'événements du flux (``tool_calls``, ``message``, ``thinking``…) ; le
    dernier est répété si le tour en demande davantage. ``contextes`` et ``options`` recueillent ce
    que le serveur a envoyé à chaque appel.
    """
    vus = {"n": 0}

    async def fake(message, *, context, **kwargs):
        if contextes is not None:
            contextes.append(copy.deepcopy(context))
        if options is not None:
            options.append(dict(kwargs))
        evenements = appels[min(vus["n"], len(appels) - 1)]
        vus["n"] += 1
        for evenement in evenements:
            yield json.dumps(evenement)

    return fake


PARCLOSES = "/profiles/perform76-parcloses.md"
DTA = "/certifications/dta-6-16-2334.md"


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
# La fiche de la question et le fil d'un tour à l'autre
# ---------------------------------------------------------------------------


def test_la_fiche_accompagne_la_question_sans_la_modifier(snapshot):
    question = "Quelle référence de clé pompier dois-je commander avec la fermeture TGY3731 ?"
    answer = WikiAnswer(question=question, history=[], snapshot=snapshot)
    dernier = answer.messages()[-1]
    assert dernier["role"] == "user" and dernier["content"].startswith(question)
    assert "FICHE DE LA QUESTION" in dernier["content"] and "TGY3731" in dernier["content"]
    assert "/quincaillerie/soleal-gy-roulements-et-fermetures.md" in dernier["content"]
    assert answer.question == question, "la question enregistrée reste la question seule"
    # Rien à dire : rien n'est ajouté.
    sobre = WikiAnswer(question="Comment choisir une parclose ?", history=[], snapshot=snapshot)
    assert sobre.messages()[-1]["content"] == "Comment choisir une parclose ?"


def test_le_tour_precedent_est_resume_dans_la_fiche(snapshot):
    trace = {"steps": [
        {"outil": "chercher", "requetes": ["parclose vitrage 44 PERFORM76", "76507"], "livraisons": []},
        {"outil": "lire", "livraisons": [
            {"chemin": PARCLOSES, "mode": "sections", "sections": ["§2 Cotes des parcloses d'ouvrant", "§9 Vitrage de série"]},
            {"chemin": "/gammes/perform.md", "mode": "page", "sections": []},
            {"chemin": DTA, "mode": "sommaire", "sections": []},
        ]},
    ]}
    resume = resume_precedent(trace)
    assert resume.startswith("Tour précédent — requêtes : parclose vitrage 44 PERFORM76 ; 76507.")
    assert f"{PARCLOSES} §2 §9" in resume and "/gammes/perform.md (page entière)" in resume
    assert DTA not in resume, "un simple sommaire n'est pas une lecture"
    assert resume_precedent(None) == "" and resume_precedent({"steps": []}) == ""

    answer = WikiAnswer(question="Et pour 48 mm ?", history=[], snapshot=snapshot, precedent=trace)
    contenu = answer.messages()[-1]["content"]
    assert contenu.startswith("Et pour 48 mm ?") and "FICHE DE LA QUESTION" in contenu and resume in contenu


# ---------------------------------------------------------------------------
# Le tour : carte, lecture, réponse
# ---------------------------------------------------------------------------


def test_le_modele_cherche_lit_puis_repond(snapshot, monkeypatch):
    """Un tour normal : une carte, une lecture, la réponse — trois appels, le raisonnement renvoyé."""
    contextes, options = [], []
    fake = _scenario(
        [{"thinking": "Je cherche la parclose "}, {"thinking": "de 44 mm."},
         {"tool_calls": [_chercher("parclose vitrage 44 PERFORM76", "76507")]},
         {"usage": {"prompt_tokens": 5000, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 4600}}}],
        [{"tool_calls": [_lire(PARCLOSES, "§2")]},
         {"usage": {"prompt_tokens": 9000, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 5000}}}],
        [{"message": {"content": "La parclose **76507** "}},
         {"message": {"content": "(/profiles/perform76-parcloses.md) tient 44 mm."}},
         {"finish_reason": "stop"},
         {"usage": {"prompt_tokens": 18000, "completion_tokens": 60, "prompt_tokens_details": {"cached_tokens": 9000}}}],
        contextes=contextes, options=options,
    )
    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "high")
    answer = WikiAnswer(
        question="Quelle parclose pour 44 mm ?",
        history=[{"role": "user", "content": "Bonjour"}, {"role": "assistant", "content": "Bonjour !"}],
        snapshot=snapshot, model="glm-test", stream_fn=fake,
    )
    events = _collect(answer)
    kinds = [next(iter(e)) for e in events]
    assert kinds.count("etape") == 2 and kinds[-1] == "sources" and "message" in kinds and "thinking" in kinds

    carte, lecture = [e["etape"] for e in events if "etape" in e]
    assert carte["outil"] == "chercher" and carte["libelle"] == "Carte"
    assert carte["detail"] == "parclose vitrage 44 PERFORM76 | 76507"
    assert PARCLOSES in carte["carte"] and carte["requetes"] == ["parclose vitrage 44 PERFORM76", "76507"]
    assert lecture["outil"] == "lire" and lecture["libelle"] == "Lecture"
    assert lecture["pages"] == [PARCLOSES]
    assert lecture["livraisons"][0]["mode"] == "sections" and lecture["livraisons"][0]["sections"][0].startswith("§2")
    assert lecture["budget"]["appel"] == 2 and lecture["budget"]["max"] == wiki_chat_service.MAX_APPELS
    assert lecture["budget"]["lu"] > 0

    # Le premier contexte : prompt permanent, historique, question.
    premier = contextes[0]
    assert premier[0]["role"] == "system" and premier[0]["content"] == snapshot.system_prompt
    assert premier[-1]["content"].startswith("Quelle parclose pour 44 mm ?")
    assert options[0]["tools"] and options[0]["tool_choice"] == "auto" and options[0]["reasoning_effort"] == "high"

    # Le second : l'appel d'outil, SON RAISONNEMENT et le résultat (la carte, avec le budget).
    second = contextes[1]
    appel = next(m for m in second if m.get("tool_calls"))
    assert appel["content"][0] == {"type": "thinking", "thinking": [{"type": "text", "text": "Je cherche la parclose de 44 mm."}]}
    resultats = [m for m in second if m["role"] == "tool"]
    assert len(resultats) == 1 and resultats[0]["tool_call_id"] == "call_1"
    assert "===== CARTE —" in resultats[0]["content"] and "Tour : appel 1/" in resultats[0]["content"]

    assert answer.text.startswith("La parclose **76507**")
    assert answer.trace["appels"] == 3
    assert answer.trace["prompt_tokens"] == 32000 and answer.trace["completion_tokens"] == 110
    # Le cache se cumule comme le reste : chacun des appels a réutilisé le préfixe.
    assert answer.trace["cached_tokens"] == 18600
    assert answer.trace["cited_pages"] == [PARCLOSES] and answer.trace["citees_non_lues"] == []
    assert answer.trace["pages_lues"] == [PARCLOSES] and answer.trace["finish_reason"] == "stop"
    assert [s["outil"] for s in answer.trace["steps"]] == ["chercher", "lire"]
    assert wiki_service.last_call()["appels"] == 3 and wiki_service.last_call()["cached_tokens"] == 18600


def test_le_preambule_est_efface_de_l_ecran_mais_garde_dans_le_contexte(snapshot):
    contextes = []
    fake = _scenario(
        [{"thinking": "réflexion"}, {"message": {"content": "Je cherche."}}, {"tool_calls": [_chercher("parclose 76507")]}],
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"message": {"content": "76507 (/profiles/perform76-parcloses.md)."}}],
        contextes=contextes,
    )
    events = _collect(WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake))
    assert any("reset" in e for e in events)
    appel = next(m for m in contextes[1] if m.get("tool_calls"))
    assert [b["type"] for b in appel["content"]] == ["thinking", "text"] and appel["content"][1]["text"] == "Je cherche."


def test_une_page_dominante_arrive_avec_la_carte(snapshot):
    answer = WikiAnswer(
        question="Quelle référence de clé pompier dois-je commander avec la fermeture TGY3731 ?",
        history=[], snapshot=snapshot,
    )
    pages_lues = []
    resultat = answer._executer("chercher", {"requetes": ["clé pompier TGY3731"]}, pages_lues)
    gy = "/quincaillerie/soleal-gy-roulements-et-fermetures.md"
    assert "===== CARTE —" in resultat and "PAGE DOMINANTE (livrée entière)" in resultat
    assert [p["chemin"] for p in pages_lues] == [gy] and gy in answer.livraison.pages
    # Une carte seule ne lit rien : une recherche sur un sujet large ne livre aucune page.
    large = WikiAnswer(question="Parcloses PERFORM76", history=[], snapshot=snapshot)
    lues = []
    assert "PAGE DOMINANTE" not in large._executer("chercher", {"requetes": ["parclose vitrage 44 PERFORM76"]}, lues)
    assert lues == []


def test_chercher_sans_requete_ou_avec_une_chaine(snapshot):
    answer = WikiAnswer(question="parclose 76507", history=[], snapshot=snapshot)
    assert answer._executer("chercher", {}, []).startswith("Aucune requête")
    assert "===== CARTE —" in answer._executer("chercher", {"requetes": "parclose 76507"}, [])
    # Une formulation qui répète la question n'est pas cherchée deux fois.
    assert answer._executer("chercher", {"requetes": ["parclose 76507", "76507 parclose"]}, [])


def test_lire_en_lot_budget_et_erreurs(snapshot, monkeypatch):
    """Plusieurs lectures en un appel ; une erreur ne fait pas échouer les autres ; rien n'est redonné."""
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot)
    pages_lues = []
    lot = {"lectures": [
        {"chemin": DTA, "sections": ["dimensions maximales"]},
        {"chemin": "/x/inexistante.md"},
        {"chemin": "/gammes/lumine.md"},
        "../../etc/passwd",
    ]}
    resultat = answer._executer("lire", lot, pages_lues)
    assert "section(s) demandée(s)" in resultat and "1 vantail OB" in resultat
    assert "Page introuvable : /x/inexistante.md" in resultat and "Chemin invalide" in resultat
    assert [p["chemin"] for p in pages_lues] == [DTA, "/gammes/lumine.md"]

    complete = answer._executer("lire", {"lectures": [{"chemin": DTA}]}, pages_lues)
    assert "déjà fourni plus haut" in complete
    sommaire = answer._executer("lire", {"lectures": [{"chemin": DTA, "sections": ["sommaire"]}]}, [])
    assert "§1" in sommaire and "(déjà livrée)" in sommaire
    inconnue = answer._executer("lire", {"lectures": [{"chemin": DTA, "sections": ["zzzz"]}]}, [])
    assert "Aucune section « zzzz »" in inconnue
    assert answer._executer("lire", {}, []).startswith("Aucune lecture")
    # Une lecture seule peut être donnée sans liste.
    assert "type: Gamme" in WikiAnswer(question="q", history=[], snapshot=snapshot)._executer(
        "lire", {"lectures": {"chemin": "/gammes/lumine.md"}}, [])

    monkeypatch.setattr(wiki_chat_service, "BUDGET_LECTURE", 1000)
    plein = WikiAnswer(question="q", history=[], snapshot=snapshot)
    plein.livraison.caracteres = 1000
    assert "Budget de lecture du tour atteint" in plein._executer("lire", {"lectures": [{"chemin": "/gammes/lumine.md"}]}, [])


def test_un_sommaire_n_est_pas_une_lecture(snapshot):
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot)
    pages_lues = []
    answer._executer("lire", {"lectures": [{"chemin": DTA, "sections": ["sommaire"]}]}, pages_lues)
    assert pages_lues == []


def test_le_serveur_injecte_les_anomalies_apres_une_lecture(snapshot):
    """La règle 2 ne dépend pas de la discipline du modèle : le serveur pousse les entrées, après une lecture."""
    contextes = []
    fake = _scenario(
        [{"tool_calls": [_chercher("garantie ferrure Technal")]}],
        [{"tool_calls": [_lire("/fournisseurs/technal.md")]}],
        [{"message": {"content": "10 ans (/fournisseurs/technal.md), voir CTR-09."}}],
        contextes=contextes,
    )
    answer = WikiAnswer(question="La ferrure Technal est garantie combien de temps ?", history=[],
                        snapshot=snapshot, stream_fn=fake)
    events = _collect(answer)
    est_injecte = lambda contexte: [  # noqa: E731
        m for m in contexte[1:] if m["role"] == "system" and "===== ENTRÉES D'ANOMALIE" in str(m.get("content"))
    ]
    # Rien avant la première recherche : on pose une question, il cherche dans le wiki.
    assert not est_injecte(contextes[0])
    # Après la carte : seulement si une page a été lue (une page dominante), jamais sur la carte seule.
    carte = next(e["etape"] for e in events if "etape" in e)
    assert bool(est_injecte(contextes[1])) == bool(carte["pages"])
    injecte = est_injecte(contextes[-1])
    assert injecte, "aucune entrée d'anomalie injectée après la lecture"
    assert re.search("(INC|CTR|VER)-[0-9]+", injecte[-1]["content"])
    assert answer.anomalies == [{"id": "CTR-09", "path": "/anomalies/contradictions-entre-sources.md"}]


def test_dernier_appel_sans_outils(snapshot, monkeypatch):
    """Le dernier appel part sans outils : le modèle répond avec ce qu'il a lu, au lieu d'une erreur."""
    monkeypatch.setattr(wiki_chat_service, "MAX_APPELS", 3)
    contextes, options = [], []

    async def fake(message, *, context, **kwargs):
        contextes.append(copy.deepcopy(context))
        options.append(dict(kwargs))
        if "tools" in kwargs:
            n = len(contextes)
            outil = _chercher("parclose 76507") if n == 1 else _lire(PARCLOSES, "§2")
            yield json.dumps({"tool_calls": [outil]})
        else:
            yield json.dumps({"message": {"content": "Avec ce que j'ai lu : 76507 (/profiles/perform76-parcloses.md)."}})

    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake)
    events = _collect(answer)
    assert ["tools" in o for o in options] == [True, True, False]
    assert "tool_choice" not in options[2]
    assert contextes[2][-1]["role"] == "system" and "dernier appel" in contextes[2][-1]["content"]
    avant_dernier = [m for m in contextes[2] if m["role"] == "tool"][-1]["content"]
    assert "Tour : appel 2/3" in avant_dernier and "plus d'outils" in avant_dernier
    assert "error" not in events[-1] and answer.text.startswith("Avec ce que j'ai lu")
    assert answer.trace["appels"] == 3 and answer.trace["citees_non_lues"] == []


def test_relance_quand_aucune_page_n_a_ete_lue(snapshot):
    """Répondre sans avoir rien lu est la faute la plus fréquente : on renvoie lire, une fois."""
    contextes = []
    fake = _scenario(
        [{"message": {"content": "Le wiki ne couvre pas ce sujet."}}],
        [{"tool_calls": [_chercher("régions climatiques")]}],
        [{"message": {"content": "Je ne sais pas."}}],
        contextes=contextes,
    )
    answer = WikiAnswer(question="Quelle région climatique pour le 59 ?", history=[], snapshot=snapshot, stream_fn=fake)
    events = _collect(answer)
    # Ce qui avait commencé à s'afficher est effacé : c'était un faux refus.
    assert any("reset" in e for e in events)
    relance = [m for m in contextes[1] if m["role"] == "system" and "Tu n'as lu aucune page" in str(m.get("content"))]
    assert relance, "la relance n'a pas été injectée"
    # Une seule fois : la deuxième réponse sans lecture est acceptée.
    assert len(contextes) == 3 and answer.text == "Je ne sais pas."


def test_nom_d_outil_normalise_et_arguments_illisibles(snapshot):
    contextes = []
    fake = _scenario(
        [{"tool_calls": [_outil("chercher\n</arg_value>", {"requetes": ["parclose 76507"]})]}],
        [{"tool_calls": [_outil("lire", "{pas du json", "call_2")]}],
        [{"tool_calls": [_outil("inexistant", {}, "call_3")]}],
        [{"message": {"content": "fin"}}],
        contextes=contextes,
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake)
    events = _collect(answer)
    etapes = [e["etape"] for e in events if "etape" in e]
    assert etapes[0]["outil"] == "chercher" and etapes[0]["carte"], "le nom suivi de balises est compris"
    resultats = [m["content"] for m in contextes[-1] if m["role"] == "tool"]
    assert resultats[1].startswith("Arguments illisibles") and "Tour : appel 2/" in resultats[1]
    assert resultats[2].startswith("Outil inconnu : inexistant")
    # Le nom que le modèle a écrit est celui qui lui revient, apparié à son appel.
    assert [m["tool_call_id"] for m in contextes[-1] if m["role"] == "tool"] == ["call_1", "call_2", "call_3"]


def test_une_reponse_coupee_et_une_reponse_vide_sont_relevees(snapshot):
    coupee = _scenario(
        [{"tool_calls": [_chercher("parclose 76507")]}],
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"message": {"content": "La parclose est la "}}, {"finish_reason": "length"}],
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=coupee)
    events = _collect(answer)
    assert answer.trace["finish_reason"] == "length" and "Réponse coupée" in answer.text
    assert any("Réponse coupée" in (e.get("message") or {}).get("content", "") for e in events)

    # Le raisonnement a mangé toute la limite : rien n'a été écrit, et on le dit comme tel.
    epuisee = _scenario(
        [{"tool_calls": [_chercher("parclose 76507")]}],
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"thinking": "une très longue réflexion"}, {"finish_reason": "length"}],
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=epuisee)
    _collect(answer)
    assert "réflexion a épuisé la limite" in answer.text and answer.trace["finish_reason"] == "length"

    vide = _scenario(
        [{"tool_calls": [_chercher("parclose 76507")]}],
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"finish_reason": "stop"}],
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=vide)
    _collect(answer)
    assert answer.text == "" and answer.trace["reponses_vides"] == 1


def test_une_reponse_qui_cite_une_page_non_lue_est_renvoyee_lire_une_fois(snapshot):
    """3 réponses sur 10 (puis 4) citaient une page vue seulement dans une carte : le serveur renvoie lire."""
    contextes = []
    fake = _scenario(
        [{"tool_calls": [_chercher("parclose 76507")]}],
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"message": {"content": "76507 (/profiles/perform76-parcloses.md), voir aussi (/gammes/lumine.md)."}}],
        [{"tool_calls": [_lire("/gammes/lumine.md")]}],
        [{"message": {"content": "76507 (/profiles/perform76-parcloses.md) ; LUMINE (/gammes/lumine.md)."}}],
        contextes=contextes,
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake)
    events = _collect(answer)
    relance = [m for m in contextes[3] if m["role"] == "system" and "Tu cites des pages que tu n'as pas lues" in str(m.get("content"))]
    assert relance and "/gammes/lumine.md" in relance[0]["content"] and "/profiles/perform76-parcloses.md" not in relance[0]["content"]
    assert any("reset" in e for e in events), "la réponse sans source disparaît de l'écran"
    assert answer.trace["appels"] == 5 and answer.trace["citees_non_lues"] == []
    assert answer.trace["relance_citations"] == ["/gammes/lumine.md"]


def test_la_relance_sur_les_citations_n_a_lieu_qu_une_fois_et_pas_faute_d_appels(snapshot, monkeypatch):
    # Une seule fois : la deuxième réponse qui cite encore une page non lue est acceptée, et relevée.
    fake = _scenario(
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"message": {"content": "Voir (/gammes/lumine.md)."}}],
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake)
    _collect(answer)
    assert answer.trace["appels"] == 3 and answer.trace["citees_non_lues"] == ["/gammes/lumine.md"]
    # Pas de relance quand il ne reste pas de quoi lire puis répondre.
    monkeypatch.setattr(wiki_chat_service, "MAX_APPELS", 3)
    fake = _scenario(
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"message": {"content": "Voir (/gammes/lumine.md)."}}],
    )
    court = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake)
    _collect(court)
    assert court.trace["appels"] == 2 and court.trace["relance_citations"] == []


def test_les_registres_d_anomalies_et_l_index_ne_declenchent_pas_la_relance(snapshot):
    fake = _scenario(
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"message": {"content": "76507 (/profiles/perform76-parcloses.md), CTR-03 (/anomalies/contradictions-entre-sources.md), (/index.md)."}}],
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake)
    _collect(answer)
    assert answer.trace["appels"] == 2 and answer.trace["relance_citations"] == []


def test_une_page_citee_sans_avoir_ete_lue_est_relevee(snapshot):
    fake = _scenario(
        [{"tool_calls": [_lire("/gammes/lumine.md")]}],
        [{"message": {"content": "Voir (/gammes/lumine.md) et (/profiles/perform76-parcloses.md)."}}],
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake)
    _collect(answer)
    assert answer.trace["citees_non_lues"] == [PARCLOSES]
    assert set(answer.trace["cited_pages"]) == {"/gammes/lumine.md", PARCLOSES}


def test_la_page_d_une_coupe_servie_rejoint_les_sources(snapshot):
    """Une réponse réduite à une image avait zéro source ; la coupe vient pourtant d'une page."""
    fake = _scenario(
        [{"tool_calls": [_chercher("parclose 76507")]}],
        [{"tool_calls": [_lire(PARCLOSES, "§2")]}],
        [{"message": {"content": "![Parclose 76507](/assets/profiles/perform76/parcloses/parclose-76507.png)"}}],
    )
    answer = WikiAnswer(question="Montre-moi la coupe de la parclose 76507", history=[],
                        snapshot=snapshot, stream_fn=fake)
    _collect(answer)
    # L'image servie est celle que le contrôle des coupes a retenue ; sa page est la source.
    servies = re.findall(r"\]\((/assets/[^)]+)\)", answer.text)
    assert servies and answer.sources
    assert all(any(i in snapshot.pages[s["path"]].body for s in answer.sources) for i in servies)


def test_sans_reasoning_effort(snapshot, monkeypatch):
    captured = {}

    async def fake_stream(message, **kwargs):
        captured.update(kwargs)
        yield json.dumps({"tool_calls": [_lire("/gammes/lumine.md")]})
        yield json.dumps({"message": {"content": ""}})

    monkeypatch.setattr(settings, "GENERATION_REASONING_EFFORT", "")
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake_stream)

    async def run():
        async for _ in answer.run():
            break

    asyncio.run(run())
    assert "reasoning_effort" not in captured
    assert answer.model == settings.MODEL_FAST


def test_le_cout_estime_suit_les_tarifs_du_modele(snapshot, monkeypatch):
    """Le prompt cumulé contient la part en cache, facturée au dixième : l'entrée au plein prix est la différence."""
    monkeypatch.setattr(settings, "MODEL_PRIX_ENTREE", 1.40)
    monkeypatch.setattr(settings, "MODEL_PRIX_CACHE", 0.14)
    monkeypatch.setattr(settings, "MODEL_PRIX_SORTIE", 4.40)
    assert wiki_chat_service.cout_estime(100_000, 60_000, 1_000) == 0.0688
    assert wiki_chat_service.cout_estime(None, None, None) == 0.0
    fake = _scenario(
        [{"tool_calls": [_lire("/gammes/lumine.md")]}, {"usage": {"prompt_tokens": 40_000, "completion_tokens": 100,
                                                                   "prompt_tokens_details": {"cached_tokens": 0}}}],
        [{"message": {"content": "Voir (/gammes/lumine.md)."}},
         {"usage": {"prompt_tokens": 42_000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 40_000}}}],
    )
    answer = WikiAnswer(question="q", history=[], snapshot=snapshot, stream_fn=fake)
    _collect(answer)
    assert answer.trace["cout_estime_usd"] == wiki_chat_service.cout_estime(82_000, 40_000, 300)
    # Sans tarif, le coût n'est pas affiché.
    monkeypatch.setattr(settings, "MODEL_PRIX_ENTREE", 0.0)
    monkeypatch.setattr(settings, "MODEL_PRIX_SORTIE", 0.0)
    assert wiki_chat_service.cout_estime(100_000, 60_000, 1_000) is None


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
