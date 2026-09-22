"""Le tour vocal : une question dite, le même wiki, une réponse lue à voix haute.

Trois étages, et rien de neuf entre eux :

* **la transcription** — Voxtral Mini Transcribe (``POST /v1/audio/transcriptions``), par lots :
  une demi-seconde pour dix secondes d'audio, mesuré le 22/09/2026. Le temps réel (WebSocket)
  n'a pas été retenu : il refuse le biais de vocabulaire, et c'est le vocabulaire qui fait la
  différence sur ce wiki — « part close » redevient « parclose », « Proferme » redevient
  « PROFERM » dès qu'on lui souffle les mots du métier (``biais_vocabulaire``) ;
* **le tour de chat**, inchangé : ``WikiAnswer``, les trois outils, l'injection des anomalies,
  les citations vérifiées. Seul le prompt permanent change — les consignes parlées précèdent
  les consignes générales — et les coupes sont laissées de côté ;
* **la synthèse** — Voxtral TTS (``POST /v1/audio/speech``) en flux, phrase par phrase, dès que
  la première phrase de la réponse est complète, sans attendre la fin de la génération. L'audio
  arrive en float32 24 kHz mono, base64 ; il est relayé tel quel dans le flux SSE, le navigateur
  le décode et l'enchaîne.

Ce qui est dit n'est pas exactement ce qui est écrit : ``texte_parle`` retire les chemins de
pages cités (l'écran les affiche en pastilles), lit les identifiants d'anomalie en clair et
déplie les unités. La question, elle, arrive au modèle telle que transcrite.

Une précaution sur le moment où l'on commence à parler : tant que le modèle n'a chargé aucune
page, ce qu'il écrit peut être effacé par le serveur (la relance de ``WikiAnswer``) — on ne le
dit donc pas encore. Dès qu'une page est chargée, la première phrase complète part en synthèse ;
ce qui attendait part avec elle.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from app.config import settings
from app.services.wiki_chat_service import ANOMALY_RE, CITATION_RE, WikiAnswer, sse
from app.services.wiki_index import WikiIndex

logger = logging.getLogger(__name__)

# L'audio rendu par Voxtral TTS en « pcm » : float32 little-endian, mono, 24 kHz.
FREQUENCE_SYNTHESE = 24_000
# Au-delà, l'enregistrement est refusé avant d'être envoyé ; le navigateur coupe bien avant
# (quarante-cinq secondes de parole font 1,4 Mo en 16 kHz 16 bits).
AUDIO_MAX_OCTETS = 8 * 1024 * 1024


class TranscriptionImpossible(RuntimeError):
    """L'API de transcription n'a pas rendu de texte exploitable."""


class SyntheseImpossible(RuntimeError):
    """L'API de synthèse n'a pas rendu d'audio."""


def _url(chemin: str) -> str:
    return (settings.MISTRAL_BASE_URL or "https://api.mistral.ai").rstrip("/") + chemin


def _autorisation() -> Dict[str, str]:
    if not settings.MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY n'est pas configurée")
    return {"Authorization": f"Bearer {settings.MISTRAL_API_KEY}"}


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

# Les mots du métier soufflés à la transcription avant ceux tirés du wiki. Un terme = un mot,
# sans espace ni virgule (contrainte de l'API) ; cent termes au plus.
BIAIS_METIER = [
    "PROFERM", "LIA", "parclose", "parcloses", "dormant", "ouvrant", "tapée", "meneau",
    "traverse", "vitrage", "quincaillerie", "crémone", "paumelle", "gâche", "compas",
    "oscillo-battant", "coulissant", "galandage", "menuiserie", "PVC", "aluminium", "hybride",
    "plaxé", "Uw", "Ug", "Sw", "AEV", "DTA", "STADIP", "Roto", "profine", "Soprofen", "Technal",
]
BIAIS_MAXIMUM = 100
_BIAIS_INTERDIT_RE = re.compile(r"[\s,]")


def biais_vocabulaire(index: WikiIndex) -> List[str]:
    """Le vocabulaire soufflé à la transcription : les mots du métier, puis les gammes, les
    systèmes et les tags les plus fréquents du wiki, tant qu'il reste de la place.

    C'est le levier qui manque au modèle temps réel. Sans lui, « parclose » est transcrit
    « part close » et n'est plus trouvé par ``chercher`` ; avec lui, mesuré le 22/09/2026,
    « Proferme » redevient « PROFERM » et « part clause » redevient « parclose ».
    """
    vus: set = set()
    biais: List[str] = []

    def ajoute(terme: Any) -> None:
        t = str(terme or "").strip()
        if len(t) < 2 or _BIAIS_INTERDIT_RE.search(t) or t.lower() in vus or len(biais) >= BIAIS_MAXIMUM:
            return
        vus.add(t.lower())
        biais.append(t)

    for terme in BIAIS_METIER:
        ajoute(terme)
    for entree in index.entries:
        for gamme in entree["gamme"]:
            ajoute(gamme)
    for entree in index.entries:
        for systeme in entree["systeme"]:
            ajoute(systeme)
    tags = Counter(t for e in index.entries for t in e["tags"])
    for tag, _ in tags.most_common():
        ajoute(tag)
    return biais


_EXTENSIONS = {
    "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/webm": "webm",
    "audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/mp4": "m4a",
}
_STATUTS_A_REESSAYER = frozenset({429, 500, 502, 503, 504})


async def transcrire(audio: bytes, type_mime: str, biais: Sequence[str]) -> Dict[str, Any]:
    """Le texte d'un enregistrement, en français, avec le vocabulaire du wiki soufflé.

    Rend ``{"texte", "ms", "secondes_audio"}``. Un enregistrement sans parole (silence, bruit)
    rend un texte vide : ce n'est pas une erreur, c'est « je n'ai rien entendu ».
    """
    type_mime = (type_mime or "audio/wav").split(";")[0].strip().lower()
    champs: List[Tuple[str, Tuple[Optional[str], Any, Optional[str]]]] = [
        ("model", (None, settings.VOCAL_MODELE_TRANSCRIPTION, None)),
        ("language", (None, "fr", None)),
    ]
    champs += [("context_bias", (None, terme, None)) for terme in biais]
    champs.append(("file", (f"question.{_EXTENSIONS.get(type_mime, 'wav')}", audio, type_mime)))

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=20.0)) as client:
        for tentative in (1, 2):
            try:
                reponse = await client.post(
                    _url("/v1/audio/transcriptions"), headers=_autorisation(), files=champs
                )
            except httpx.RequestError as exc:
                if tentative == 2:
                    logger.error("[vocal] transcription : erreur réseau %s", exc)
                    raise TranscriptionImpossible("La transcription est indisponible pour le moment.") from exc
                await asyncio.sleep(1.0)
                continue
            if reponse.status_code in _STATUTS_A_REESSAYER and tentative == 1:
                logger.warning("[vocal] transcription HTTP %s, nouvel essai", reponse.status_code)
                await asyncio.sleep(1.0)
                continue
            break

    if reponse.status_code >= 400:
        logger.error("[vocal] transcription HTTP %s : %s", reponse.status_code, reponse.text[:500])
        raise TranscriptionImpossible("La transcription est indisponible pour le moment.")
    data = reponse.json()
    usage = data.get("usage") or {}
    ms = int((time.perf_counter() - t0) * 1000)
    texte = str(data.get("text") or "").strip()
    logger.info("[vocal] transcription : %d car. en %d ms (%s s d'audio)", len(texte), ms, usage.get("prompt_audio_seconds"))
    return {"texte": texte, "ms": ms, "secondes_audio": usage.get("prompt_audio_seconds")}


# ---------------------------------------------------------------------------
# Synthèse
# ---------------------------------------------------------------------------


async def synthese(texte: str) -> AsyncIterator[str]:
    """Les fragments audio (base64, float32 24 kHz mono) d'une phrase, au fil du flux Voxtral TTS.

    Le flux est un ``text/event-stream`` : ``speech.audio.delta`` porte ``audio_data`` (0,4 s
    d'audio par fragment), ``speech.audio.done`` clôt. Premier fragment après 0,4 s environ.
    """
    payload = {
        "model": settings.VOCAL_MODELE_SYNTHESE,
        "input": texte,
        "voice_id": settings.VOCAL_VOIX,
        "response_format": "pcm",
        "stream": True,
    }
    headers = {**_autorisation(), "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
        async with client.stream("POST", _url("/v1/audio/speech"), headers=headers, json=payload) as reponse:
            if reponse.status_code != 200:
                corps = (await reponse.aread()).decode(errors="replace")
                logger.error("[vocal] synthèse HTTP %s : %s", reponse.status_code, corps[:500])
                raise SyntheseImpossible(f"synthèse HTTP {reponse.status_code}")
            async for ligne in reponse.aiter_lines():
                if not ligne.startswith("data:"):
                    continue
                brut = ligne[5:].strip()
                if not brut or brut == "[DONE]":
                    continue
                try:
                    evenement = json.loads(brut)
                except json.JSONDecodeError:
                    continue
                if evenement.get("type") == "speech.audio.delta" and evenement.get("audio_data"):
                    yield evenement["audio_data"]
                elif evenement.get("type") == "speech.audio.done":
                    break


# ---------------------------------------------------------------------------
# Ce qui se dit
# ---------------------------------------------------------------------------

REGISTRES_PARLES = {
    "INC": "incohérence interne",
    "CTR": "contradiction entre sources",
    "VER": "information à vérifier",
}
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LIEN_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# « (/profiles/x.md) », « (/a.md, /b.md) » : la parenthèse entière disparaît.
_CITATION_PARENTHESEE_RE = re.compile(r"\(\s*(?:/(?:[a-z0-9_-]+/)*[a-z0-9_-]+\.md\s*[,;]?\s*)+\)")
_SEPARATEUR_TABLEAU_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$", re.MULTILINE)
_MARQUEURS_LIGNE_RE = re.compile(r"^[ \t]*(?:[-*•]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+)", re.MULTILINE)
_COEFFICIENTS_RE = re.compile(r"\b([USR])([wgf])\b")  # Uw → « U w », qui se lit lettre à lettre
_UNITES = [
    (re.compile(r"(?<=\d)\s?mm\b"), " millimètres"),
    (re.compile(r"(?<=\d)\s?cm\b"), " centimètres"),
    (re.compile(r"(?<=\d)\s?(?:m²|m2\b)"), " mètres carrés"),
    (re.compile(r"(?<=\d)\s?kg\b"), " kilogrammes"),
    (re.compile(r"(?<=\d)\s?daN\b"), " décanewtons"),
    (re.compile(r"(?<=\d)\s?Pa\b"), " pascals"),
    (re.compile(r"(?<=\d)\s?dB\b"), " décibels"),
    (re.compile(r"(?<=\d)\s?°C"), " degrés"),
    (re.compile(r"(?<=\d)\s?%"), " pour cent"),
    (re.compile(r"\bW/\(?m²\.?K\)?"), "watts par mètre carré kelvin"),
    (re.compile(r"\bkm/h\b"), "kilomètres par heure"),
    (re.compile(r"\bm/s\b"), "mètres par seconde"),
]
_SYMBOLES = [
    ("→", ", donc "), ("⚠️", "attention, "), ("⚠", "attention, "), ("≥", " au moins "),
    ("≤", " au plus "), ("±", " plus ou moins "), ("&", " et "), ("n°", "numéro "), ("N°", "numéro "),
]


def _ligne_de_tableau(ligne: str) -> str:
    """Une ligne de tableau que le modèle aurait quand même écrite devient une phrase."""
    if not ligne.strip().startswith("|"):
        return ligne
    cellules = [c.strip() for c in ligne.strip().strip("|").split("|") if c.strip()]
    return ", ".join(cellules) + "." if cellules else ""


def texte_parle(texte: str) -> str:
    """Ce que la synthèse doit dire : la réponse sans ce qui ne se lit pas à voix haute.

    Les chemins de pages cités sont retirés — l'écran les montre en pastilles —, les
    identifiants d'anomalie sont lus en clair (« contradiction entre sources 17 »), les
    unités et quelques symboles sont dépliés, le markdown résiduel est effacé.
    """
    t = texte or ""
    t = _IMAGE_RE.sub("", t)
    t = _LIEN_RE.sub(r"\1", t)
    t = _CITATION_PARENTHESEE_RE.sub("", t)
    t = CITATION_RE.sub("", t)
    t = ANOMALY_RE.sub(lambda m: f"{REGISTRES_PARLES[m.group(1)]} {int(m.group(2))}", t)
    t = _SEPARATEUR_TABLEAU_RE.sub("", t)
    t = "\n".join(_ligne_de_tableau(ligne) for ligne in t.splitlines())
    t = _MARQUEURS_LIGNE_RE.sub("", t)
    t = re.sub(r"[*_`#|]+", "", t)
    for symbole, mot in _SYMBOLES:
        t = t.replace(symbole, mot)
    for motif, mot in _UNITES:
        t = motif.sub(mot, t)
    t = _COEFFICIENTS_RE.sub(r"\1 \2", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"[ \t]+([,.;:!?])", r"\1", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return "\n".join(l.strip() for l in t.splitlines()).strip()


# ---------------------------------------------------------------------------
# Le découpage en phrases
# ---------------------------------------------------------------------------

_FIN_DE_PHRASE_RE = re.compile(r"(?<=[.!?…])[\"»)\]]*\s+")


class Phraseur:
    """Découpe le texte au fil du flux en segments prêts à dire.

    Le premier segment part dès que la première phrase est complète (et fait au moins
    ``minimum`` caractères : « Oui. » attend la phrase suivante) — c'est lui qui fixe le délai
    avant le premier son. Les suivants regroupent des phrases jusqu'à ``groupe`` caractères :
    une requête de synthèse par groupe, et une prosodie plus liée qu'en phrases isolées. Un
    saut de paragraphe force la coupe ; ``vider()`` rend ce qui reste à la fin du flux.
    """

    def __init__(self, groupe: int = 220, minimum: int = 40):
        self.groupe = groupe
        self.minimum = minimum
        self.reinitialiser()

    def reinitialiser(self) -> None:
        self.tampon = ""
        self.pretes: List[str] = []
        self.premier = True

    def pousser(self, delta: str) -> List[str]:
        self.tampon += delta
        sortie: List[str] = []
        while True:
            fin = _FIN_DE_PHRASE_RE.search(self.tampon)
            paragraphe = self.tampon.find("\n\n")
            if fin is None and paragraphe < 0:
                break
            if fin is not None and (paragraphe < 0 or fin.end() <= paragraphe + 2):
                phrase, self.tampon = self.tampon[: fin.end()].strip(), self.tampon[fin.end():]
                force = False
            else:
                phrase, self.tampon = self.tampon[:paragraphe].strip(), self.tampon[paragraphe + 2:]
                force = True
            if phrase:
                self.pretes.append(phrase)
            sortie += self._emettre(force)
        # Un long passage sans ponctuation (le modèle qui déroule) : on coupe à la virgule.
        if len(self.tampon) > self.groupe * 2:
            coupe = max(self.tampon.rfind(", ", 0, self.groupe), self.tampon.rfind(" ", 0, self.groupe))
            if coupe > self.minimum:
                self.pretes.append(self.tampon[: coupe + 1].strip())
                self.tampon = self.tampon[coupe + 1:]
                sortie += self._emettre(True)
        return sortie

    def _emettre(self, force: bool) -> List[str]:
        if not self.pretes:
            return []
        total = sum(len(p) + 1 for p in self.pretes) - 1
        if force or total >= self.groupe or (self.premier and total >= self.minimum):
            segment = " ".join(self.pretes)
            self.pretes = []
            self.premier = False
            return [segment]
        return []

    def vider(self) -> List[str]:
        reste = " ".join(self.pretes + ([self.tampon.strip()] if self.tampon.strip() else []))
        self.reinitialiser()
        return [reste] if reste else []


# ---------------------------------------------------------------------------
# Le tour
# ---------------------------------------------------------------------------

SyntheseFn = Callable[[str], AsyncIterator[str]]

# Dite pendant que le modèle lit le wiki, à son premier appel d'outil : le silence d'une
# recherche de plusieurs secondes est ce qui rend un assistant vocal pénible. Synthétisée une
# fois par processus et par voix, puis servie depuis la mémoire.
PHRASES_ATTENTE = (
    "Je regarde dans le wiki.",
    "Un instant, je consulte les pages.",
    "Je cherche ça dans le wiki.",
)
_attentes: Dict[str, List[str]] = {}
_compteur_attentes = 0


async def audio_attente(synthese_fn: SyntheseFn) -> Tuple[str, List[str]]:
    """Une phrase d'attente et ses fragments audio, en tournant entre les phrases."""
    global _compteur_attentes
    phrase = PHRASES_ATTENTE[_compteur_attentes % len(PHRASES_ATTENTE)]
    _compteur_attentes += 1
    cle = f"{settings.VOCAL_VOIX}|{phrase}"
    if cle not in _attentes:
        try:
            _attentes[cle] = [morceau async for morceau in synthese_fn(phrase)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vocal] phrase d'attente non synthétisée : %s", exc)
            return phrase, []
    return phrase, _attentes[cle]


async def prechauffer_attentes() -> None:
    """Synthétise les phrases d'attente une fois pour toutes, au démarrage, en arrière-plan."""
    for phrase in PHRASES_ATTENTE:
        cle = f"{settings.VOCAL_VOIX}|{phrase}"
        if cle in _attentes:
            continue
        try:
            _attentes[cle] = [morceau async for morceau in synthese(phrase)]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[vocal] préchauffage de « %s » impossible : %s", phrase, exc)
            return
    logger.info("[vocal] %d phrase(s) d'attente prêtes (voix %s)", len(PHRASES_ATTENTE), settings.VOCAL_VOIX)


@dataclass
class TourVocal:
    """Les événements du tour de chat, entrelacés avec les phrases dites.

    ``run()`` rend des événements SSE : ceux de ``WikiAnswer`` tels quels (``etape``,
    ``thinking``, ``message``, ``reset``, ``sources``), plus ``phrase`` (un segment part en
    synthèse : son index et son texte) et ``audio`` (un fragment base64 de ce segment). L'index
    -1 est la phrase d'attente. Les mesures du tour se lisent sur ``mesures`` ensuite.
    """

    answer: WikiAnswer
    # Résolu à l'exécution (``synthese`` du module) pour rester remplaçable dans les tests.
    synthese_fn: Optional[SyntheseFn] = None
    attente: bool = True
    mesures: Dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        synthese_fn = self.synthese_fn or synthese
        sortie: "asyncio.Queue[Optional[str]]" = asyncio.Queue()
        segments: "asyncio.Queue[Optional[Tuple[int, int, str]]]" = asyncio.Queue()
        phraseur = Phraseur()
        en_attente: List[str] = []
        etat: Dict[str, Any] = {
            "generation": 0, "pages_ok": False, "index": 0,
            "premier_son_ms": None, "attente_ms": None, "phrases": 0, "attente_emise": False,
        }

        def emettre(texte: str) -> None:
            segments.put_nowait((etat["generation"], etat["index"], texte))
            etat["index"] += 1

        async def texte_du_tour() -> None:
            try:
                async for brut in self.answer.run():
                    sortie.put_nowait(brut)
                    try:
                        data = json.loads(brut[6:]) if brut.startswith("data: ") else {}
                    except json.JSONDecodeError:
                        data = {}
                    if data.get("etape"):
                        if data["etape"].get("pages"):
                            etat["pages_ok"] = True
                        if self.attente and not etat["attente_emise"]:
                            etat["attente_emise"] = True
                            phrase, morceaux = await audio_attente(synthese_fn)
                            if morceaux:
                                etat["attente_ms"] = int((time.perf_counter() - t0) * 1000)
                                sortie.put_nowait(sse({"phrase": {"index": -1, "texte": phrase, "attente": True}}))
                                for morceau in morceaux:
                                    sortie.put_nowait(sse({"audio": {"index": -1, "data": morceau}}))
                        if etat["pages_ok"] and en_attente:
                            for segment in en_attente:
                                emettre(segment)
                            en_attente.clear()
                    elif data.get("reset"):
                        # Le modèle a finalement appelé un outil, ou est renvoyé lire : ce qu'il
                        # avait écrit n'est plus la réponse, et ne se dira pas.
                        etat["generation"] += 1
                        phraseur.reinitialiser()
                        en_attente.clear()
                    elif (data.get("message") or {}).get("content"):
                        for segment in phraseur.pousser(data["message"]["content"]):
                            if etat["pages_ok"]:
                                emettre(segment)
                            else:
                                en_attente.append(segment)
                    elif data.get("sources") is not None:
                        # La conclusion du tour : tout ce qui reste se dit.
                        for segment in en_attente + phraseur.vider():
                            emettre(segment)
                        en_attente.clear()
            except Exception as exc:  # noqa: BLE001
                logger.exception("[vocal] tour de chat en échec")
                sortie.put_nowait(sse({"error": f"Génération impossible : {exc}"}))
            finally:
                segments.put_nowait(None)

        async def synthese_des_segments() -> None:
            avertissement_fait = False
            while True:
                element = await segments.get()
                if element is None:
                    return
                generation, index, texte = element
                if generation != etat["generation"]:
                    continue
                a_dire = texte_parle(texte)
                if not a_dire:
                    continue
                sortie.put_nowait(sse({"phrase": {"index": index, "texte": texte}}))
                try:
                    async for morceau in synthese_fn(a_dire):
                        if generation != etat["generation"]:
                            break
                        if etat["premier_son_ms"] is None:
                            etat["premier_son_ms"] = int((time.perf_counter() - t0) * 1000)
                        sortie.put_nowait(sse({"audio": {"index": index, "data": morceau}}))
                    etat["phrases"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[vocal] synthèse en échec sur un segment : %s", exc)
                    if not avertissement_fait:
                        avertissement_fait = True
                        sortie.put_nowait(sse({
                            "avertissement": "La synthèse vocale est indisponible : la réponse reste à l'écran."
                        }))

        taches = [asyncio.create_task(texte_du_tour()), asyncio.create_task(synthese_des_segments())]

        async def clore() -> None:
            await asyncio.gather(*taches, return_exceptions=True)
            sortie.put_nowait(None)

        cloture = asyncio.create_task(clore())
        try:
            while True:
                evenement = await sortie.get()
                if evenement is None:
                    break
                yield evenement
        finally:
            for tache in taches + [cloture]:
                tache.cancel()
            self.mesures = {
                "voix": settings.VOCAL_VOIX,
                "modele_synthese": settings.VOCAL_MODELE_SYNTHESE,
                "attente_ms": etat["attente_ms"],
                "premier_son_ms": etat["premier_son_ms"],
                "phrases": etat["phrases"],
                "duree_ms": int((time.perf_counter() - t0) * 1000),
            }
            logger.info(
                "[vocal] %d segment(s) dit(s), attente à %s ms, premier son de réponse à %s ms",
                etat["phrases"], etat["attente_ms"], etat["premier_son_ms"],
            )
