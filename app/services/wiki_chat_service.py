"""Le tour de chat : une question, quelques pages, une réponse streamée, ses citations vérifiées.

Navigation outillée et non CAG : le wiki (198 pages) ne tient dans aucune fenêtre de contexte.
Le prompt permanent ne porte que les consignes, un vocabulaire et l'index des anomalies ; les
pages arrivent par trois outils — ``chercher`` (qui livre directement le contenu entier des
premières pages trouvées), ``lire_page`` et ``lire_anomalie``.

Deux garde-fous ne dépendent pas de la discipline du modèle :

* **les anomalies sont injectées par le serveur.** La règle 2 des consignes — donner la valeur
  *et* signaler la contradiction — est trop importante pour reposer sur la bonne volonté d'un
  petit modèle : les entrées rapprochées de la question et des pages chargées sont poussées dans
  le contexte à chaque tour ;
* **les coupes sont vérifiées sur disque.** Le modèle a sous les yeux une colonne de chemins
  d'images qui ne diffèrent que par la référence ; en fabriquer un lui coûte peu. Seule l'image
  réellement présente, et réellement rattachée à la référence demandée, passe.

Le seul contrôle de sortie reste déterministe : les chemins de pages cités (``/dossier/page.md``)
sont résolus dans le wiki. Une citation vers une page inexistante est rendue telle quelle mais
marquée ``exists: false`` : on constate et on montre, on ne réécrit pas la réponse.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Tuple

from sqlmodel import Session, select

from app.config import settings
from app.models.message import Message
from app.services import wiki_service
from app.services.mistral_service import chat_stream
from app.services.wiki_index import (
    REFERENCE_RE,
    WikiIndex,
    formate_resultats,
    trier_resultats,
)
from app.services.wiki_service import WikiSnapshot

logger = logging.getLogger(__name__)

# Un chemin de page tel que les consignes demandent de le citer : « (/profiles/perform76-parcloses.md) ».
CITATION_RE = re.compile(r"(?<![\w/.])(/(?:[a-z0-9_-]+/)*[a-z0-9_-]+\.md)(?![\w/])")
# Les identifiants des trois registres d'anomalies.
ANOMALY_RE = re.compile(r"\b(INC|CTR|VER)-(\d{1,3})\b")
ANOMALY_PAGES = {
    "INC": "/anomalies/incoherences-internes.md",
    "CTR": "/anomalies/contradictions-entre-sources.md",
    "VER": "/anomalies/informations-a-verifier.md",
}

# Un tour = plusieurs appels. Au-delà, la question est trop large pour être traitée en lisant :
# on le dit plutôt que de laisser filer le coût.
MAX_TOOL_ROUNDS = int(os.environ.get("CHAT_MAX_TOOL_ROUNDS", "6"))
# Nombre de pages que ``chercher`` livre en entier. Trois couvrent la très grande majorité des
# questions pour un contexte encore modeste ; monter à cinq gagne peu et coûte la moitié en plus.
PAGES_COMPLETES = int(os.environ.get("CHAT_PAGES_COMPLETES", "3"))

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "chercher",
            "description": (
                "Trouve les pages du wiki et renvoie le contenu ENTIER des premières, suivi des "
                "autres résultats en métadonnées seules. La recherche porte sur le texte intégral "
                "autant que sur les métadonnées : une référence (76526, NT1947, A076) se cherche "
                "directement. Premier outil à appeler pour toute question, et souvent le seul "
                "nécessaire. Commence par les mots-clés seuls : les facettes sont faites pour "
                "départager deux familles voisines une fois que tu as vu les résultats, pas pour "
                "un premier essai — une facette mal choisie écarte la bonne page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mots_cles": {
                        "type": "string",
                        "description": "Mots de la question, références comprises. Ex. : 'parclose vitrage 44 perform76'.",
                    },
                    "type": {
                        "type": "string",
                        "description": "Facette optionnelle : Profilé, Quincaillerie, Gamme, Procédure, Document source…",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Facette optionnelle : un tag du vocabulaire (ex. 'parclose').",
                    },
                    "gamme": {
                        "type": "string",
                        "description": "Facette optionnelle : une gamme du vocabulaire (ex. 'LUMINE').",
                    },
                    "systeme": {
                        "type": "string",
                        "description": "Facette optionnelle : un système du vocabulaire (ex. '76').",
                    },
                    "limite": {
                        "type": "integer",
                        "description": "Nombre de résultats, 10 par défaut, 15 au maximum.",
                    },
                },
                "required": ["mots_cles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lire_page",
            "description": (
                "Charge le contenu complet d'une page du wiki à partir d'un chemin rendu par "
                "chercher (par ex. /profiles/perform76-parcloses.md). À appeler pour chaque page "
                "pertinente avant d'affirmer un détail technique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chemin": {
                        "type": "string",
                        "description": "Chemin de la page, commençant par /, copié d'un résultat de chercher.",
                    }
                },
                "required": ["chemin"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lire_anomalie",
            "description": (
                "Renvoie le détail d'une entrée d'anomalie à partir de son identifiant "
                "(INC-01, CTR-03, VER-07), tel qu'il figure dans l'index des anomalies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "identifiant": {
                        "type": "string",
                        "description": "Identifiant de l'entrée, ex. CTR-03.",
                    }
                },
                "required": ["identifiant"],
            },
        },
    },
]

LIBELLES = {"chercher": "Recherche", "lire_page": "Lecture", "lire_anomalie": "Anomalie"}


def sse(obj: Dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Citations et anomalies citées
# ---------------------------------------------------------------------------


def extract_citations(text: str, snapshot: WikiSnapshot) -> List[Dict[str, Any]]:
    """Les pages citées dans la réponse, dans l'ordre d'apparition, dédoublonnées, vérifiées."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for match in CITATION_RE.finditer(text or ""):
        path = match.group(1)
        if path in seen:
            continue
        seen.add(path)
        page = snapshot.pages.get(path)
        if page is not None and not page.missing:
            out.append(
                {
                    "path": path,
                    "title": page.title,
                    "type": page.type,
                    "status": page.status,
                    "exists": True,
                }
            )
        else:
            out.append(
                {
                    "path": path,
                    "title": path.rsplit("/", 1)[-1][:-3].replace("-", " "),
                    "type": "",
                    "status": "",
                    "exists": False,
                }
            )
    return out


def extract_anomalies(text: str) -> List[Dict[str, str]]:
    """Les identifiants d'anomalie mentionnés, avec la page registre qui les porte."""
    out: List[Dict[str, str]] = []
    seen: set = set()
    for match in ANOMALY_RE.finditer(text or ""):
        ident = f"{match.group(1)}-{match.group(2)}"
        if ident in seen:
            continue
        seen.add(ident)
        out.append({"id": ident, "path": ANOMALY_PAGES[match.group(1)]})
    return out


def load_history(session: Session, conversation_id: int) -> List[Dict[str, str]]:
    """Les derniers échanges de la conversation, bornés en nombre et en caractères, et
    commençant par un message utilisateur (Mistral l'exige après le système)."""
    rows = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at, Message.id)
    ).all()
    items = [
        {"role": m.role, "content": m.content}
        for m in rows
        if m.role in ("user", "assistant") and (m.content or "").strip()
    ]
    items = items[-settings.CHAT_HISTORY_MAX_MESSAGES :] if settings.CHAT_HISTORY_MAX_MESSAGES > 0 else []
    while items and sum(len(i["content"]) for i in items) > settings.CHAT_HISTORY_MAX_CHARS:
        items.pop(0)
    while items and items[0]["role"] != "user":
        items.pop(0)
    return items


def _cached_tokens(usage: Optional[Dict[str, Any]]) -> Optional[int]:
    if not usage:
        return None
    for key in ("cached_tokens", "prompt_cached_tokens", "cache_read_input_tokens"):
        if isinstance(usage.get(key), int):
            return usage[key]
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        return details["cached_tokens"]
    return None


# ---------------------------------------------------------------------------
# Lecture du wiki et des coupes
# ---------------------------------------------------------------------------


def read_wiki_page(chemin: Any, snapshot: WikiSnapshot) -> str:
    """Le contenu d'une page, ou le message d'erreur que le modèle doit lire pour se reprendre.

    On sert depuis l'instantané, pas depuis le disque : c'est la même version que celle qui a
    été indexée, et le chemin ne peut désigner que ce que le wiki contient.
    """
    if not isinstance(chemin, str) or not chemin.startswith("/"):
        return f"Chemin invalide : {chemin!r}. Un chemin doit commencer par / et venir de chercher."
    page = snapshot.pages.get(chemin)
    if page is None or page.missing or page.reserved:
        return f"Page introuvable : {chemin}. Relance chercher pour obtenir le chemin exact."
    return page.raw_text


# Les coupes sont rangées dans wiki/assets/ et référencées depuis les pages par un lien image
# markdown à chemin absolu : ![Parclose 76507](/assets/profiles/perform76/parcloses/…png).
IMAGE_WIKI_RE = re.compile(r"!\[([^\]\n]*)\]\(\s*(/assets/[^)\s]+?)\s*\)")
TYPES_IMAGE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
}
# « profil » sans accent seulement : « quel profilé » n'est pas une demande de coupe.
DEMANDE_COUPE_RE = re.compile(
    r"\b(coupes?|sch[ée]mas?|dessins?|profils?|images?|visuels?|allure|ressemble)\b",
    re.IGNORECASE,
)


def fichier_asset(chemin: str, root: Path) -> Optional[Path]:
    """Le fichier disque derrière un ``/assets/…`` du wiki, ou None s'il n'est pas servable.

    On résout puis on vérifie l'appartenance à ``wiki/assets`` : un ``../..`` ne sort pas du
    bundle. Les extensions sont restreintes à l'image — ce point d'entrée ne lit pas le wiki.
    """
    from urllib.parse import unquote

    chemin = unquote(str(chemin).split("?", 1)[0].split("#", 1)[0])
    if not chemin.startswith("/assets/"):
        return None
    assets = (root / "wiki" / "assets").resolve()
    cible = (root / "wiki" / chemin.lstrip("/")).resolve()
    try:
        cible.relative_to(assets)
    except ValueError:
        return None
    if cible.suffix.lower() not in TYPES_IMAGE or not cible.is_file():
        return None
    return cible


def coupes_des_pages(pages_lues: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    """Référence → chemin de sa coupe, relevé dans les tableaux des pages chargées.

    Une ligne de tableau porte sa référence en première colonne et son image dans une autre :
    ce couple est ce qui permet de vérifier que l'image servie est bien celle de la référence
    demandée, sans croire le modèle sur parole.
    """
    coupes: Dict[str, str] = {}
    for page in pages_lues:
        for ligne in (page.get("corps") or "").splitlines():
            ligne = ligne.strip()
            if not ligne.startswith("|"):
                continue
            image = IMAGE_WIKI_RE.search(ligne)
            if not image:
                continue
            premiere = ligne.strip("|").split("|")[0]
            trouvees = REFERENCE_RE.findall(premiere)
            if trouvees:
                coupes.setdefault(trouvees[-1], image.group(2))
    return coupes


def preparer_images(
    texte: str, question: str, pages_lues: Sequence[Dict[str, Any]], root: Path
) -> Tuple[str, Dict[str, List[str]]]:
    """Ne laisse passer que la coupe de la référence demandée.

    Deux fautes à couvrir, observées l'une et l'autre :

    * **le chemin inventé.** Le modèle a sous les yeux une colonne entière de chemins qui ne
      diffèrent que par la référence ; en fabriquer un pour une référence non illustrée lui coûte
      peu, et une image cassée ressemble à une image manquante plutôt qu'à une invention. D'où la
      vérification sur disque ;
    * **l'image de la ligne voisine.** C'est la faute que la règle 9 décrit pour les colonnes,
      transposée aux lignes. Quand la question nomme une référence dont une page chargée donne la
      coupe, c'est ce couple qui fait foi, pas le choix du modèle.
    """
    journal: Dict[str, List[str]] = {"servies": [], "inexistantes": [], "hors_sujet": [], "ajoutees": []}

    coupes = coupes_des_pages(pages_lues)
    demandees: set = set()
    attendues: set = set()
    if DEMANDE_COUPE_RE.search(question or ""):
        demandees = set(REFERENCE_RE.findall(question or "")) & set(coupes)
        attendues = {coupes[r] for r in demandees}

    def remplace(m: "re.Match") -> str:
        chemin = m.group(2)
        if not fichier_asset(chemin, root):
            journal["inexistantes"].append(chemin)
            return ""
        if attendues and chemin not in attendues:
            journal["hors_sujet"].append(chemin)
            return ""
        journal["servies"].append(chemin)
        return m.group(0)

    texte = IMAGE_WIKI_RE.sub(remplace, texte)

    # La coupe demandée que le modèle a oubliée, ou qu'il a remplacée par une autre.
    for ref in sorted(demandees):
        chemin = coupes[ref]
        if chemin in journal["servies"] or not fichier_asset(chemin, root):
            continue
        texte = texte.rstrip() + f"\n\n![Coupe {ref}]({chemin})"
        journal["servies"].append(chemin)
        journal["ajoutees"].append(chemin)

    if journal["inexistantes"]:
        texte = texte.rstrip() + (
            "\n\n*(Coupe citée par le modèle mais absente du wiki, retirée : "
            + ", ".join(f"`{c}`" for c in dict.fromkeys(journal["inexistantes"]))
            + ".)*"
        )
    # La suppression laisse des lignes vides là où l'image occupait sa ligne.
    return re.sub(r"\n{3,}", "\n\n", texte), journal


# ---------------------------------------------------------------------------
# Le tour
# ---------------------------------------------------------------------------


def _tool_args(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    fonction = call.get("function") or {}
    nom = fonction.get("name") or ""
    args = fonction.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return nom, (args if isinstance(args, dict) else {})


def _detail(nom: str, args: Dict[str, Any]) -> str:
    if nom == "chercher":
        facettes = ", ".join(
            f"{k}={args[k]}" for k in ("type", "tags", "gamme", "systeme") if args.get(k)
        )
        mots = str(args.get("mots_cles") or "")
        return f"{mots} ({facettes})" if facettes else mots
    if nom == "lire_anomalie":
        return str(args.get("identifiant") or "")
    return str(args.get("chemin") or "")


StreamFn = Callable[..., AsyncIterator[str]]


@dataclass
class WikiAnswer:
    """Une instance par tour. ``run()`` est un générateur d'événements SSE ; l'état final
    (texte, raisonnement, sources, étapes, trace) se lit sur l'instance ensuite."""

    question: str
    history: List[Dict[str, str]]
    snapshot: WikiSnapshot
    model: str = ""
    # Résolu à l'exécution (``chat_stream`` du module) pour rester remplaçable dans les tests.
    stream_fn: Optional[StreamFn] = None
    # Le prompt permanent et sa clé de cache : ceux du chat par défaut, ceux du tour vocal quand
    # la réponse doit être dite (mêmes règles de vérité, forme parlée devant).
    system_prompt: Optional[str] = None
    cache_key: Optional[str] = None
    # Les coupes n'ont pas de sens à l'oral : le tour vocal les laisse de côté.
    images: bool = True

    text: str = ""
    thinking: str = ""
    usage: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, str]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = settings.MODEL_FAST
        if self.system_prompt is None:
            self.system_prompt = self.snapshot.system_prompt
        if self.cache_key is None:
            self.cache_key = self.snapshot.cache_key

    @property
    def index(self) -> WikiIndex:
        return self.snapshot.index

    def messages(self) -> List[Dict[str, Any]]:
        return (
            [{"role": "system", "content": self.system_prompt}]
            + list(self.history)
            + [{"role": "user", "content": self.question}]
        )

    # ---- outils --------------------------------------------------------

    def _executer(self, nom: str, args: Dict[str, Any], pages_lues: List[Dict[str, Any]]) -> str:
        if nom == "chercher":
            mots = str(args.get("mots_cles") or "")
            try:
                limite = min(int(args.get("limite") or 10), 15)
            except (TypeError, ValueError):
                limite = 10
            pages = self.index.search(
                mots_cles=mots,
                type=args.get("type"),
                tags=args.get("tags"),
                gamme=args.get("gamme"),
                systeme=args.get("systeme"),
                limite=limite,
            )
            # Les pages livrées entières comptent comme lues : elles alimentent le rapprochement
            # avec les registres d'anomalies, et le modèle a le droit de les citer sans repasser
            # par lire_page.
            livrees, _ = trier_resultats(pages, PAGES_COMPLETES)
            for page in livrees:
                if page not in pages_lues:
                    pages_lues.append(page)
            return formate_resultats(self.index, pages, mots, completes=PAGES_COMPLETES)
        if nom == "lire_page":
            chemin = args.get("chemin", "")
            entree = next((e for e in self.index.entries if e["chemin"] == chemin), None)
            if entree is not None and entree not in pages_lues:
                pages_lues.append(entree)
            return read_wiki_page(chemin, self.snapshot)
        if nom == "lire_anomalie":
            return self.index.anomalie(args.get("identifiant", ""))
        return f"Outil inconnu : {nom}"

    def _injecter_anomalies(
        self, working: List[Dict[str, Any]], pages_lues: List[Dict[str, Any]], deja: set
    ) -> None:
        entrees = [
            e for e in self.index.match_anomalies(self.question, pages_lues) if e["id"] not in deja
        ]
        if not entrees:
            return
        deja.update(e["id"] for e in entrees)
        corps = "\n".join(f"{e['registre']} :\n{e['ligne']}" for e in entrees)
        working.append(
            {
                "role": "system",
                "content": (
                    "===== ENTRÉES D'ANOMALIE À PRENDRE EN COMPTE =====\n\n"
                    "Ces entrées ont été rapprochées de la question et des pages que tu as "
                    "chargées. Signale celles qui concernent réellement la valeur que tu donnes, "
                    "avec leur identifiant. Passe les autres sous silence, sans les énumérer.\n\n"
                    "Elles ne dispensent pas de lire les pages : une entrée d'anomalie signale un "
                    "problème, elle n'est pas la source de la valeur. Charge la page concernée "
                    "avec lire_page avant de répondre, et cite son chemin.\n\n" + corps
                ),
            }
        )

    # ---- le tour -------------------------------------------------------

    async def run(self) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        first_token_ms: Optional[int] = None

        extra: Dict[str, Any] = {
            "prompt_cache_key": self.cache_key,
            "tools": TOOLS,
            "tool_choice": "auto",
        }
        if settings.GENERATION_REASONING_EFFORT:
            extra["reasoning_effort"] = settings.GENERATION_REASONING_EFFORT

        stream_fn = self.stream_fn or chat_stream
        working: List[Dict[str, Any]] = self.messages()
        pages_lues: List[Dict[str, Any]] = []
        deja_injectees: set = set()
        relance_faite = False
        # Les trois mesures sont CUMULÉES sur les appels du tour : un tour d'outils en fait
        # jusqu'à six, et chacun paie son prompt et réutilise le préfixe en cache. Rapporter le
        # cache d'un seul appel à la somme des prompts divisait le taux par le nombre d'appels.
        cumul = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "appels": 0}
        think_parts: List[str] = []

        # Injecté avant le premier appel : le modèle peut répondre sans outil, et la règle 2
        # doit tenir même dans ce cas.
        self._injecter_anomalies(working, pages_lues, deja_injectees)

        for _ in range(MAX_TOOL_ROUNDS):
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            usage: Dict[str, Any] = {}

            async for raw in stream_fn(
                "",
                model=self.model,
                context=working,
                max_tokens=settings.CHAT_MAX_TOKENS,
                temperature=settings.CHAT_TEMPERATURE,
                **extra,
            ):
                try:
                    data = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                if data.get("thinking"):
                    think_parts.append(data["thinking"])
                    yield sse({"thinking": data["thinking"]})
                    continue
                if data.get("tool_calls"):
                    tool_calls = data["tool_calls"]
                    continue
                if data.get("usage"):
                    usage = data["usage"]
                    continue
                content = (data.get("message") or {}).get("content")
                if not content:
                    continue
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - t0) * 1000)
                text_parts.append(content)
                yield sse({"message": {"content": content}})

            cumul["prompt_tokens"] += usage.get("prompt_tokens") or 0
            cumul["completion_tokens"] += usage.get("completion_tokens") or 0
            cumul["cached_tokens"] += _cached_tokens(usage) or 0
            cumul["appels"] += 1
            texte = "".join(text_parts)

            if tool_calls:
                # Le texte de ce tour n'était qu'un préambule : il est conservé dans le contexte
                # du modèle, mais effacé de l'écran, où seule la réponse finale a sa place.
                if texte.strip():
                    yield sse({"reset": True})
                working.append(
                    {"role": "assistant", "content": texte, "tool_calls": tool_calls}
                )
                for call in tool_calls:
                    nom, args = _tool_args(call)
                    deja_lues = len(pages_lues)
                    contenu = self._executer(nom, args, pages_lues)
                    etape = {
                        "outil": nom,
                        "libelle": LIBELLES.get(nom, nom),
                        "detail": _detail(nom, args),
                        "pages": [p["chemin"] for p in pages_lues[deja_lues:]],
                    }
                    self.steps.append(etape)
                    yield sse({"etape": etape})
                    message: Dict[str, Any] = {"role": "tool", "name": nom, "content": contenu}
                    if call.get("id"):
                        message["tool_call_id"] = call["id"]
                    working.append(message)
                self._injecter_anomalies(working, pages_lues, deja_injectees)
                continue

            # Répondre sur la foi d'un extrait, sans avoir ouvert la moindre page, est la faute
            # que les règles 0 et 5 interdisent — et celle que le modèle commet le plus. On le
            # renvoie lire, une fois.
            if not pages_lues and not relance_faite:
                relance_faite = True
                if texte.strip():
                    yield sse({"reset": True})
                working.append({"role": "assistant", "content": texte})
                working.append(
                    {
                        "role": "system",
                        "content": (
                            "Tu n'as chargé aucune page. Appelle chercher : il te renvoie "
                            "directement le contenu entier des premières pages, sur lequel tu "
                            "pourras t'appuyer.\n"
                            "Tu ne peux pas déclarer que le wiki ne couvre pas un sujet sans "
                            "avoir cherché au moins une fois. Le wiki ne contient pas que des "
                            "cotes de profilés : il porte aussi des tables réglementaires et de "
                            "référence — régions climatiques par département, classement AEV par "
                            "site, résistance au vent, glossaire du métier. Une question qui te "
                            "semble hors sujet y a souvent sa réponse.\n"
                            "Cherche, puis réponds en citant le chemin des pages utilisées."
                        ),
                    }
                )
                continue

            self.usage = usage
            async for event in self._conclure(texte, pages_lues, cumul, first_token_ms, t0, think_parts):
                yield event
            return

        yield sse(
            {
                "error": (
                    "Trop d'allers-retours de lecture pour cette question — essaie de la préciser "
                    "(gamme, référence) pour réduire le nombre de pages à consulter."
                )
            }
        )

    async def _conclure(
        self,
        texte: str,
        pages_lues: List[Dict[str, Any]],
        cumul: Dict[str, int],
        first_token_ms: Optional[int],
        t0: float,
        think_parts: List[str],
    ) -> AsyncIterator[str]:
        """Filtre les coupes, vérifie les citations, mesure le tour."""
        if self.images:
            texte, coupes = preparer_images(texte, self.question, pages_lues, self.snapshot.root)
            # Le texte servi peut différer de celui qui a été streamé : une coupe inventée a été
            # retirée, une coupe oubliée ajoutée. On le renvoie en entier, l'interface remplace.
            if any(coupes.values()):
                logger.info("[chat] coupes : %s", json.dumps(coupes, ensure_ascii=False))
                yield sse({"remplacer": texte})

        self.text = texte.strip()
        self.thinking = "".join(think_parts)
        self.sources = extract_citations(self.text, self.snapshot)
        self.anomalies = extract_anomalies(self.text)

        usage = self.usage or {}
        self.trace = {
            "model": self.model,
            "prompt_tokens": cumul["prompt_tokens"] or usage.get("prompt_tokens"),
            "cached_tokens": cumul["cached_tokens"],
            "completion_tokens": cumul["completion_tokens"] or usage.get("completion_tokens"),
            "appels": cumul["appels"],
            "first_token_ms": first_token_ms,
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "wiki_hash": self.cache_key,
            "wiki_pages": len(self.snapshot.concept_pages),
            "history_messages": len(self.history),
            "steps": self.steps,
            "pages_lues": [p["chemin"] for p in pages_lues],
            "cited_pages": [s["path"] for s in self.sources if s["exists"]],
            "unknown_citations": [s["path"] for s in self.sources if not s["exists"]],
            "anomalies": [a["id"] for a in self.anomalies],
        }
        wiki_service.record_call(
            {
                "at": datetime.now().isoformat(timespec="seconds"),
                "model": self.model,
                "prompt_tokens": self.trace["prompt_tokens"],
                "cached_tokens": self.trace["cached_tokens"],
                "completion_tokens": self.trace["completion_tokens"],
                "appels": cumul["appels"],
                "first_token_ms": first_token_ms,
                "duration_ms": self.trace["duration_ms"],
            }
        )
        if self.trace["unknown_citations"]:
            logger.info(
                "[chat] citations vers des pages inexistantes : %s",
                ", ".join(self.trace["unknown_citations"]),
            )
        logger.info(
            "[chat] %d appel(s), %d page(s) lue(s), %s tokens d'entrée",
            cumul["appels"],
            len(pages_lues),
            cumul["prompt_tokens"],
        )
        yield sse({"sources": self.sources, "anomalies": self.anomalies})
