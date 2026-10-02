"""Le tour de chat : une question, une carte, quelques lectures, une réponse streamée, ses citations vérifiées.

Navigation outillée : le wiki (~2 M tokens) ne tient dans aucune fenêtre de contexte. Le prompt
permanent porte l'index du wiki (``index.md``) et un vocabulaire ; le modèle navigue par trois
outils — ``chercher`` (qui rend une **carte** : douze résultats avec leurs sections et les lignes
qui répondent, et ne livre que la page qui domine nettement le classement), ``lire`` (les
sections ou les pages qu'il choisit, en un seul appel) et ``lire_anomalie``. Rien n'est livré
deux fois dans un tour (``Livraison``).

Seul ce que ``lire`` livre — et la page dominante — est **lu** : une ligne de carte situe une
réponse, elle ne la prouve pas. Les anomalies, les coupes et le contrôle des citations reposent
sur les pages lues.

Deux garde-fous ne dépendent pas de la discipline du modèle :

* **les anomalies sont injectées par le serveur.** La règle 2 des consignes — donner la valeur
  *et* signaler la contradiction — est trop importante pour reposer sur la bonne volonté d'un
  modèle : les entrées rapprochées des pages lues sont poussées dans le contexte après chaque
  lecture — jamais avant : on pose une question, il cherche dans le wiki ;
* **les coupes sont vérifiées sur disque.** Le modèle a sous les yeux une colonne de chemins
  d'images qui ne diffèrent que par la référence ; en fabriquer un lui coûte peu. Seule l'image
  réellement présente, et réellement rattachée à la référence demandée, passe.

Le serveur prépare aussi la question (``WikiIndex.fiche_question`` : les références rares et
leurs emplacements, celles qu'aucune page ne porte, ce que le tour précédent avait lu), tient le
budget du tour (``MAX_APPELS``, ``BUDGET_LECTURE`` : chaque résultat d'outil dit où on en est, et
le dernier appel part sans outils), et renvoie au modèle son propre raisonnement d'un appel à
l'autre, comme le recommandent Mistral et Z.ai.

Le seul contrôle de sortie reste déterministe : les chemins de pages cités (``/dossier/page.md``)
sont résolus dans le wiki. Une citation vers une page inexistante est rendue telle quelle mais
marquée ``exists: false`` ; une page citée sans avoir été lue est relevée dans la trace
(``citees_non_lues``) : on constate et on montre, on ne réécrit pas la réponse.
"""
from __future__ import annotations

import json
import logging
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
    Livraison,
    WikiIndex,
    cotes_de,
    tokenise,
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

# Un tour = au plus six appels au modèle, le dernier sans outils : il répond avec ce qu'il a lu
# plutôt que de laisser filer le coût. Trois suffisent quand la fiche ou une page dominante désigne
# la réponse : chercher, lire, répondre.
MAX_APPELS = 6
# Ce qu'un tour peut lire en tout (~45 k tokens). Au-delà, une lecture ne livre plus rien.
BUDGET_LECTURE = 150_000
REQUETES_MAX = 4
LECTURES_MAX = 6

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "chercher",
            "description": (
                "Cherche dans tout le wiki et rend une CARTE : douze résultats, chacun avec sa page, "
                "ses sections qui répondent et les lignes qui contiennent tes mots (avec l'en-tête de "
                "leur tableau). Ce n'est pas une lecture : une ligne de carte situe une réponse, elle "
                "ne la prouve pas — lis ensuite avec `lire`. Écris de une à quatre formulations avec "
                "les mots que le wiki emploie : l'objet (famille, pièce), la référence seule (76526, "
                "TGY3731), un synonyme du métier ; l'index du prompt te dit comment le wiki nomme les "
                "choses. La question de l'utilisateur est toujours cherchée aussi. Une page qui domine "
                "nettement le classement est livrée entière avec la carte. `gamme` et `systeme` "
                "font remonter les pages d'un produit sans exclure les autres."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "requetes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": REQUETES_MAX,
                        "description": "De une à quatre formulations. Ex. : ['parclose vitrage 44 PERFORM76', '76507'].",
                    },
                    "gamme": {
                        "type": "string",
                        "description": "Facette optionnelle : une gamme du vocabulaire (ex. 'LUMINE').",
                    },
                    "systeme": {
                        "type": "string",
                        "description": "Facette optionnelle : un système du vocabulaire (ex. '76').",
                    },
                },
                "required": ["requetes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lire",
            "description": (
                "Lit des pages ou des sections du wiki, en un seul appel (jusqu'à six lectures). "
                "`chemin` vient de la carte ou de l'index (ex. /profiles/perform76-parcloses.md). "
                "`sections` : « §13 », un mot du titre, ou « sommaire » ; sans `sections`, la page "
                "entière (si elle est trop grande, tu reçois sa fiche et son sommaire). Une section "
                "n'est pas la page : si la valeur dépend d'un autre tableau, d'une note ou d'une "
                "exception, lis aussi la section qui la porte. Ce qui a déjà été lu n'est pas redonné."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lectures": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": LECTURES_MAX,
                        "items": {
                            "type": "object",
                            "properties": {
                                "chemin": {
                                    "type": "string",
                                    "description": "Chemin de la page, commençant par /.",
                                },
                                "sections": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optionnel : « §13 », un mot du titre, ou « sommaire ».",
                                },
                            },
                            "required": ["chemin"],
                        },
                    }
                },
                "required": ["lectures"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lire_anomalie",
            "description": (
                "Renvoie le détail d'une entrée d'anomalie à partir de son identifiant "
                "(INC-01, CTR-03, VER-07), tel qu'il est écrit dans une page lue."
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

LIBELLES = {"chercher": "Carte", "lire": "Lecture", "lire_anomalie": "Anomalie"}

DERNIER_APPEL = (
    "C'est ton dernier appel : tu n'as plus d'outils. Réponds maintenant avec ce que tu as lu, en "
    "citant le chemin des pages lues, et dis clairement ce que tu n'as pas pu vérifier."
)

AUCUNE_PAGE_LUE = (
    "Tu n'as lu aucune page : `chercher` ne rend qu'une carte, qui situe une réponse sans la "
    "prouver. Appelle `chercher` si tu ne l'as pas fait, puis `lire` sur les sections qui portent "
    "la réponse ; si la carte ne montre rien de plausible, reformule avec d'autres mots.\n"
    "Tu ne peux pas déclarer que le wiki ne couvre pas un sujet sans avoir cherché au moins une "
    "fois. Le wiki ne contient pas que des cotes de profilés : il porte aussi des tables "
    "réglementaires et de référence — régions climatiques par département, classement AEV par "
    "site, résistance au vent, glossaire du métier. Une question qui te semble hors sujet y a "
    "souvent sa réponse.\n"
    "Cherche, lis, puis réponds en citant le chemin des pages lues."
)


CITATIONS_SANS_LECTURE = (
    "Tu cites des pages que tu n'as pas lues : {pages}. Elles ne te sont arrivées que dans une carte "
    "ou dans l'index, et une phrase qui s'appuie sur une ligne de carte n'a pas de source. Lis avec "
    "`lire` les sections dont tu as besoin, puis réponds de nouveau en ne gardant que ce que tu as "
    "lu — ou retire les phrases qui ne s'appuient sur aucune page lue."
)


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


def cout_estime(prompt_tokens: Optional[int], cached_tokens: Optional[int], completion_tokens: Optional[int]) -> Optional[float]:
    """Le coût d'un tour en dollars, au tarif du modèle (``MODEL_PRIX_*``), ou None sans tarif.

    ``prompt_tokens`` est cumulé sur les appels du tour et contient la part en cache, facturée au
    dixième : l'entrée au prix plein est la différence.
    """
    if not (settings.MODEL_PRIX_ENTREE or settings.MODEL_PRIX_SORTIE):
        return None
    cache = cached_tokens or 0
    entree = max((prompt_tokens or 0) - cache, 0)
    return round((entree * settings.MODEL_PRIX_ENTREE + cache * settings.MODEL_PRIX_CACHE
                  + (completion_tokens or 0) * settings.MODEL_PRIX_SORTIE) / 1_000_000, 4)


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


def _nom_outil(nom: Any) -> str:
    """Le nom d'outil tel que le serveur le comprend : coupé au premier espace ou à la première
    balise. Un nom suivi de ``</arg_value>`` est un bogue d'hébergement observé chez Mistral."""
    return re.split(r"[\s<]", str(nom or "").strip(), maxsplit=1)[0]


def _tool_args(call: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    """``(nom, arguments, erreur)``. L'erreur n'est pas vide quand les arguments ne sont pas un
    JSON lisible : le modèle la reçoit, au lieu d'une recherche vide qu'il prendrait pour une
    absence."""
    fonction = call.get("function") or {}
    nom = _nom_outil(fonction.get("name"))
    args = fonction.get("arguments")
    if isinstance(args, str):
        if not args.strip():
            return nom, {}, ""
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return nom, {}, "Arguments illisibles (JSON invalide) : réécris l'appel avec des arguments JSON complets."
    if not isinstance(args, dict):
        return nom, {}, "Arguments illisibles : un objet JSON est attendu."
    return nom, args, ""


def _liste(valeur: Any) -> List[Any]:
    """Une liste, qu'on ait reçu une liste, une valeur seule ou rien (tolérance de forme)."""
    if valeur is None:
        return []
    return list(valeur) if isinstance(valeur, (list, tuple)) else [valeur]


def _detail(nom: str, args: Dict[str, Any]) -> str:
    if nom == "chercher":
        facettes = ", ".join(f"{k}={args[k]}" for k in ("gamme", "systeme") if args.get(k))
        requetes = " | ".join(str(r) for r in _liste(args.get("requetes")))
        return f"{requetes} ({facettes})" if facettes else requetes
    if nom == "lire_anomalie":
        return str(args.get("identifiant") or "")
    vues = []
    for lecture in _liste(args.get("lectures")):
        if isinstance(lecture, dict):
            sections = ", ".join(str(s) for s in _liste(lecture.get("sections")))
            vues.append(str(lecture.get("chemin") or "") + (f" ({sections})" if sections else ""))
    return " ; ".join(vues)


def load_precedent(session: Session, conversation_id: int) -> Optional[Dict[str, Any]]:
    """La trace du dernier tour de LIA dans cette conversation (``Message.metadata_json``), ou None."""
    ligne = session.exec(
        select(Message)
        .where(Message.conversation_id == conversation_id, Message.role == "assistant")
        .order_by(Message.created_at.desc(), Message.id.desc())
    ).first()
    if ligne is None or not isinstance(ligne.metadata_json, dict):
        return None
    trace = ligne.metadata_json.get("trace")
    return trace if isinstance(trace, dict) else None


def resume_precedent(trace: Optional[Dict[str, Any]]) -> str:
    """Ce que le tour précédent avait cherché et lu, en une ligne : le fil d'une question à l'autre.

    Une question de suite (« et pour 48 mm ? ») relit la bonne section sans relancer de recherche.
    Cela vit dans la fiche de la question et non dans l'historique : l'historique reste du texte
    seul, sans ligne technique qu'un modèle pourrait recopier dans sa réponse.
    """
    if not trace:
        return ""
    requetes: List[str] = []
    lu: List[str] = []
    for etape in trace.get("steps") or []:
        if etape.get("outil") == "chercher":
            requetes += [r for r in etape.get("requetes") or [] if r not in requetes]
        for livree in etape.get("livraisons") or []:
            chemin, mode = livree.get("chemin"), livree.get("mode")
            if not chemin or mode == "sommaire":
                continue
            if mode == "page":
                ligne = f"{chemin} (page entière)"
            else:
                numeros = [m.group(0) for m in (re.match(r"§\d+", str(s)) for s in livree.get("sections") or []) if m]
                ligne = f"{chemin} {' '.join(numeros)}".strip()
            if ligne not in lu:
                lu.append(ligne)
    if not requetes and not lu:
        return ""
    texte = "Tour précédent — "
    if requetes:
        texte += "requêtes : " + " ; ".join(requetes[:4]) + ". "
    if lu:
        texte += "lu : " + " · ".join(lu) + "."
    return texte if len(texte) <= 600 else texte[:597].rstrip() + "…"


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
    # La trace du tour précédent de la conversation (``load_precedent``) : le fil d'une question à l'autre.
    precedent: Optional[Dict[str, Any]] = None

    text: str = ""
    thinking: str = ""
    usage: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    anomalies: List[Dict[str, str]] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)
    # Ce que le tour a déjà livré : rien n'est envoyé deux fois, et le budget du tour est tenu.
    livraison: Livraison = field(default_factory=Livraison)

    def __post_init__(self) -> None:
        if not self.model:
            self.model = settings.MODEL_FAST
        if self.system_prompt is None:
            self.system_prompt = self.snapshot.system_prompt
        if self.cache_key is None:
            self.cache_key = self.snapshot.cache_key
        # Calculées une fois, avant tout appel : pour toutes les recherches du tour.
        self._cotes = cotes_de(self.question)
        self._fiche = self.index.fiche_question(self.question)
        self._appel = 0
        self._livrees_etape: List[Dict[str, Any]] = []
        self._carte_etape: List[str] = []
        self._requetes_etape: List[str] = []
        self._relances_citations: List[str] = []

    @property
    def index(self) -> WikiIndex:
        return self.snapshot.index

    def message_utilisateur(self) -> str:
        """La question suivie de sa fiche (ce que le serveur sait d'elle) et du tour précédent."""
        bloc = self._fiche["texte"]
        precedent = resume_precedent(self.precedent)
        if precedent:
            entete = bloc + "\n" if bloc else "===== FICHE DE LA QUESTION (calculée par le serveur) =====\n"
            bloc = entete + "- " + precedent
        return self.question + ("\n\n" + bloc if bloc else "")

    def messages(self) -> List[Dict[str, Any]]:
        return (
            [{"role": "system", "content": self.system_prompt}]
            + list(self.history)
            + [{"role": "user", "content": self.message_utilisateur()}]
        )

    # ---- outils --------------------------------------------------------

    def _marquer_lues(self, livrees: Sequence[Dict[str, Any]], pages_lues: List[Dict[str, Any]]) -> None:
        """Une page livrée, entière ou en partie, compte comme lue : elle alimente le rapprochement
        avec les registres d'anomalies et le contrôle des coupes, et le modèle peut la citer. Un
        simple sommaire n'est pas une lecture."""
        for livree in livrees:
            if livree["mode"] == "sommaire":
                continue
            entree = self.index.par_chemin.get(livree["chemin"])
            if entree is not None and entree not in pages_lues:
                pages_lues.append(entree)
        self._livrees_etape.extend(livrees)

    def _pied(self, appel: int) -> str:
        """Où en est le tour : le modèle règle sa recherche sur ce qui lui reste."""
        pied = f"\n\n---\nTour : appel {appel}/{MAX_APPELS} · lu {self.livraison.caracteres // 1000} k car. sur {BUDGET_LECTURE // 1000} k"
        if appel == MAX_APPELS - 1:
            pied += " · au prochain appel tu n'auras plus d'outils : réponds-y"
        return pied

    def _executer(self, nom: str, args: Dict[str, Any], pages_lues: List[Dict[str, Any]]) -> str:
        if nom == "chercher":
            return self._chercher(args, pages_lues)
        if nom == "lire":
            return self._lire(args, pages_lues)
        if nom == "lire_anomalie":
            return self.index.anomalie(args.get("identifiant", ""))
        return f"Outil inconnu : {nom}. Les outils sont chercher, lire et lire_anomalie."

    def _chercher(self, args: Dict[str, Any], pages_lues: List[Dict[str, Any]]) -> str:
        brutes = [str(r).strip() for r in _liste(args.get("requetes")) if str(r).strip()][:REQUETES_MAX]
        if not brutes:
            return (
                "Aucune requête : écris de une à quatre formulations dans `requetes` (les mots du wiki "
                "pour l'objet, la référence seule, un synonyme du métier)."
            )
        self._requetes_etape = brutes
        # La question de l'utilisateur d'abord : l'ordre des formulations est une priorité (mesuré
        # le 02/10 : question + reformulations donnent 69,6 % de « toutes les pages de preuve »
        # sur le banc, la question seule 64,3 %). Une formulation qui répète une autre est ôtée.
        formulations: List[str] = []
        vues: set = set()
        for formulation in [self.question] + brutes:
            cle = tuple(sorted(set(tokenise(formulation))))
            if cle not in vues:
                vues.add(cle)
                formulations.append(formulation)
        classement = self.index.classer_multi(
            formulations, gamme=args.get("gamme"), systeme=args.get("systeme"), cotes=self._cotes
        )
        texte, listees = self.index.carte(classement, self.index.jetons_requete(formulations), self.livraison)
        self._carte_etape = listees
        # Une page qui domine nettement et qui tient entière arrive avec la carte : c'est un appel de moins.
        dominante = self.index.page_dominante(classement, self.livraison)
        if dominante is not None:
            chemin = dominante["entree"]["chemin"]
            if self.livraison.caracteres + len(dominante["entree"]["corps"].strip()) <= BUDGET_LECTURE:
                page, livree = self.index.page_entiere(chemin, self.livraison, entete="PAGE DOMINANTE (livrée entière)")
                self._marquer_lues([livree], pages_lues)
                texte += "\n\n" + page
        return texte

    def _lire(self, args: Dict[str, Any], pages_lues: List[Dict[str, Any]]) -> str:
        lectures = _liste(args.get("lectures"))[:LECTURES_MAX]
        if not lectures:
            return "Aucune lecture : `lectures` est une liste d'objets {chemin, sections}."
        morceaux: List[str] = []
        for lecture in lectures:
            if isinstance(lecture, str):
                lecture = {"chemin": lecture}
            if not isinstance(lecture, dict):
                morceaux.append("Lecture ignorée : un objet {chemin, sections} est attendu.")
                continue
            chemin = lecture.get("chemin", "")
            erreur = read_wiki_page(chemin, self.snapshot)
            if erreur.startswith(("Chemin invalide", "Page introuvable")):
                morceaux.append(erreur)
                continue
            sections = [str(s) for s in _liste(lecture.get("sections")) if str(s).strip()] or [None]
            for section in sections:
                reste = BUDGET_LECTURE - self.livraison.caracteres
                if reste <= 0:
                    morceaux.append(
                        "Budget de lecture du tour atteint : plus rien n'est livré. Réponds avec ce que "
                        "tu as lu, ou dis ce qui reste à vérifier."
                    )
                    break
                texte, livree = self.index.lire(self.snapshot.pages[chemin], section, self.livraison, reste)
                self._marquer_lues([livree], pages_lues)
                morceaux.append(texte)
        return "\n\n".join(morceaux)

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
                    "lues. Signale celles qui concernent réellement la valeur que tu donnes, "
                    "avec leur identifiant. Passe les autres sous silence, sans les énumérer.\n\n"
                    "Elles ne dispensent pas de lire les pages : une entrée d'anomalie signale un "
                    "problème, elle n'est pas la source de la valeur. Lis la page concernée "
                    "avec `lire` avant de répondre, et cite son chemin.\n\n" + corps
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
        relance_citations = False
        # Les mesures sont CUMULÉES sur les appels du tour : chacun paie son prompt et réutilise le
        # préfixe en cache. Rapporter le cache d'un seul appel à la somme des prompts divisait le
        # taux par le nombre d'appels.
        cumul = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "appels": 0}
        think_parts: List[str] = []  # tout le raisonnement du tour, pour l'écran
        finish_reason: Optional[str] = None
        vides = 0

        for appel in range(1, MAX_APPELS + 1):
            self._appel = appel
            dernier = appel == MAX_APPELS
            options = dict(extra)
            if dernier:
                # Plus d'outils : le modèle répond avec ce qu'il a lu.
                options.pop("tools")
                options.pop("tool_choice")
                working.append({"role": "system", "content": DERNIER_APPEL})
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            pensee: List[str] = []  # le raisonnement de CET appel : c'est lui qu'on renvoie
            usage: Dict[str, Any] = {}
            finish_reason = None

            async for raw in stream_fn(
                "",
                model=self.model,
                context=working,
                max_tokens=settings.CHAT_MAX_TOKENS,
                temperature=settings.CHAT_TEMPERATURE,
                **options,
            ):
                try:
                    data = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue
                if data.get("thinking"):
                    pensee.append(data["thinking"])
                    think_parts.append(data["thinking"])
                    yield sse({"thinking": data["thinking"]})
                    continue
                if data.get("tool_calls"):
                    tool_calls = data["tool_calls"]
                    continue
                if data.get("finish_reason"):
                    finish_reason = data["finish_reason"]
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

            if tool_calls and not dernier:
                # Le texte de ce tour n'était qu'un préambule : il est conservé dans le contexte
                # du modèle, mais effacé de l'écran, où seule la réponse finale a sa place.
                if texte.strip():
                    yield sse({"reset": True})
                contenu: Any = texte
                if pensee:
                    # Le raisonnement revient avec les résultats d'outils : Mistral et Z.ai le
                    # recommandent, et l'API l'accepte (essai du 02/10).
                    contenu = [{"type": "thinking", "thinking": [{"type": "text", "text": "".join(pensee)}]}]
                    if texte.strip():
                        contenu.append({"type": "text", "text": texte})
                working.append({"role": "assistant", "content": contenu, "tool_calls": tool_calls})
                for call in tool_calls:
                    nom, args, erreur = _tool_args(call)
                    deja_lues = len(pages_lues)
                    self._livrees_etape, self._carte_etape, self._requetes_etape = [], [], []
                    resultat = erreur or self._executer(nom, args, pages_lues)
                    resultat += self._pied(appel)
                    etape = {
                        "outil": nom,
                        "libelle": LIBELLES.get(nom, nom),
                        "detail": _detail(nom, args),
                        # Les pages LUES par cet outil ; la carte (pages listées) est à part.
                        "pages": [p["chemin"] for p in pages_lues[deja_lues:]],
                        "carte": list(self._carte_etape),
                        "requetes": list(self._requetes_etape),
                        # Page par page : livrée entière, ou les sections livrées.
                        "livraisons": list(self._livrees_etape),
                        "budget": {
                            "appel": appel,
                            "max": MAX_APPELS,
                            "lu": self.livraison.caracteres,
                            "total": BUDGET_LECTURE,
                        },
                    }
                    self.steps.append(etape)
                    yield sse({"etape": etape})
                    message: Dict[str, Any] = {
                        "role": "tool",
                        "name": (call.get("function") or {}).get("name") or nom,
                        "content": resultat,
                    }
                    if call.get("id"):
                        message["tool_call_id"] = call["id"]
                    working.append(message)
                # Les anomalies suivent une lecture, jamais une simple carte.
                if pages_lues:
                    self._injecter_anomalies(working, pages_lues, deja_injectees)
                continue

            # Répondre sur la foi d'une carte, sans avoir lu la moindre page, est la faute que les
            # règles 0 et 5 interdisent — et celle que le modèle commet le plus. On le renvoie
            # lire, une fois.
            if not pages_lues and not relance_faite and not dernier:
                relance_faite = True
                if texte.strip():
                    yield sse({"reset": True})
                working.append({"role": "assistant", "content": texte})
                working.append({"role": "system", "content": AUCUNE_PAGE_LUE})
                continue

            # Citer une page qu'on n'a pas lue, c'est s'appuyer sur une ligne de carte : la phrase n'a
            # pas de source (mesuré le 02/10 : 3 réponses sur 10, puis 4 malgré la consigne). Le
            # serveur le voit sans modèle — c'est une comparaison de chemins —, renvoie lire une
            # fois, et seulement s'il reste de quoi lire puis répondre.
            if not dernier and not relance_citations and appel <= MAX_APPELS - 2:
                lues = {p["chemin"] for p in pages_lues}
                sans_lecture = [
                    c["path"] for c in extract_citations(texte, self.snapshot)
                    if c["path"] not in lues and c["path"] != "/index.md" and not c["path"].startswith("/anomalies/")
                ]
                if sans_lecture:
                    relance_citations = True
                    self._relances_citations = sans_lecture
                    if texte.strip():
                        yield sse({"reset": True})
                    working.append({"role": "assistant", "content": texte})
                    working.append({"role": "system", "content": CITATIONS_SANS_LECTURE.format(pages=", ".join(sans_lecture))})
                    continue

            # Conclusion : la réponse, coupée ou non, vide ou non, est comptée dans la trace.
            if finish_reason == "length":
                if texte.strip():
                    note = "\n\n*(Réponse coupée : la limite de longueur est atteinte.)*"
                else:
                    # Le raisonnement a consommé toute la limite avant le premier mot de la réponse.
                    note = (
                        "*(Le modèle n'a pas pu écrire sa réponse : sa réflexion a épuisé la limite de "
                        "longueur. Relancez la question.)*"
                    )
                texte += note
                yield sse({"message": {"content": note}})
                logger.warning("[chat] réponse coupée par max_tokens (%s)", settings.CHAT_MAX_TOKENS)
            if not texte.strip():
                vides += 1
                logger.warning("[chat] réponse vide au dernier appel (finish_reason=%s)", finish_reason)
            self.usage = usage
            async for event in self._conclure(
                texte, pages_lues, cumul, first_token_ms, t0, think_parts, finish_reason, vides
            ):
                yield event
            return

    async def _conclure(
        self,
        texte: str,
        pages_lues: List[Dict[str, Any]],
        cumul: Dict[str, int],
        first_token_ms: Optional[int],
        t0: float,
        think_parts: List[str],
        finish_reason: Optional[str] = None,
        vides: int = 0,
    ) -> AsyncIterator[str]:
        """Filtre les coupes, vérifie les citations, mesure le tour."""
        coupes: Dict[str, List[str]] = {"servies": []}
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
        # Une coupe servie vient d'une page chargée : c'est sa source, que le modèle l'ait citée
        # ou non (le 30/09, les réponses réduites à une image n'avaient aucune source).
        citees = {s["path"] for s in self.sources}
        for image in dict.fromkeys(coupes["servies"]):
            page = next((p["chemin"] for p in pages_lues if image in (p.get("corps") or "")), None)
            if page and page not in citees:
                self.sources += extract_citations(f"({page})", self.snapshot)
                citees.add(page)
        self.anomalies = extract_anomalies(self.text)
        lues = {p["chemin"] for p in pages_lues}

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
            "caracteres_livres": self.livraison.caracteres,
            "cited_pages": [s["path"] for s in self.sources if s["exists"]],
            "unknown_citations": [s["path"] for s in self.sources if not s["exists"]],
            # Citées sans avoir été lues : on constate et on montre, on ne réécrit pas la réponse.
            "citees_non_lues": [s["path"] for s in self.sources if s["exists"] and s["path"] not in lues],
            "anomalies": [a["id"] for a in self.anomalies],
            "finish_reason": finish_reason,
            "reponses_vides": vides,
            "fiche": {"rares": self._fiche["rares"], "absentes": self._fiche["absentes"]},
            "avec_tour_precedent": bool(resume_precedent(self.precedent)),
            # Les pages que la réponse citait sans les avoir lues quand le serveur a renvoyé lire.
            "relance_citations": self._relances_citations,
        }
        self.trace["cout_estime_usd"] = cout_estime(
            self.trace["prompt_tokens"], self.trace["cached_tokens"], self.trace["completion_tokens"]
        )
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
        if self.trace["citees_non_lues"]:
            logger.info("[chat] pages citées sans avoir été lues : %s", ", ".join(self.trace["citees_non_lues"]))
        logger.info(
            "[chat] %d appel(s), %d page(s) lue(s), %s tokens d'entrée",
            cumul["appels"],
            len(pages_lues),
            cumul["prompt_tokens"],
        )
        yield sse({"sources": self.sources, "anomalies": self.anomalies})
