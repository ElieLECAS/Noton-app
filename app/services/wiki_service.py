"""Le wiki — la seule source de LIA.

Le wiki est un bundle OKF (``wiki_llm/wiki/*.md`` : frontmatter YAML + markdown, liens
``[..](/dossier/page.md)``). Il s'écrit HORS de l'application (Claude Code + git, protocole
``wiki_llm/CLAUDE.md``) ; l'application le lit. Ce module construit un instantané :

  * les pages (frontmatter, corps, liens sortants), les nœuds fantômes (liens vers des pages
    pas encore écrites — valides en OKF), les degrés, les pages périmées ;
  * l'**index de navigation** (``wiki_index.WikiIndex``) : recherche lexicale sur le texte
    intégral et registres d'anomalies, ce que l'outil ``chercher`` du chat interroge ;
  * le **prompt système** du chat : consignes + vocabulaire + index des anomalies — quelques
    milliers de tokens, PAS le wiki, qui ne tient dans aucune fenêtre — et sa clé de cache
    Mistral (sha256 du prompt entier, donc toute modification des consignes, du vocabulaire ou
    d'une entrée d'anomalie invalide proprement le cache) ;
  * un lint (orphelines, fantômes, périmées, frontmatter illisible, pages hors index) et des
    statistiques pour la carte d'administration.

L'instantané est reconstruit dès qu'un fichier change (signature = nombre, taille et date des
``.md`` + date des consignes + nombre de PDF de ``raw/``), vérifiée à chaque accès : quelques
dizaines de ``stat`` — sans bouton ni tâche planifiée.

Le 19/09/2026 le wiki est passé en transcription exhaustive des PDF : 198 pages, largement
au-delà de la fenêtre de 256 k. Le contexte augmenté (CAG) est donc abandonné au profit de la
navigation outillée — voir ``wiki_index`` et ``wiki_chat_service``.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from app.config import settings
from app.services import wiki_index
from app.services.wiki_index import WikiIndex

logger = logging.getLogger(__name__)

RESERVED = frozenset({"index.md", "log.md"})
LINK_RE = re.compile(r"\[([^\]]*)\]\((/[^)\s]+\.md)\)")
FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)
# Ratio caractères / token mesuré le 18/09 sur le prompt réel (Small, tokenizer tekken).
CHARS_PER_TOKEN = 2.924
# Le prompt permanent ne porte plus que les consignes, le vocabulaire et l'index des anomalies :
# il doit rester petit, puisqu'il est payé à chaque appel d'un tour d'outils. Au-delà de ce seuil
# ESTIMÉ on avertit (journal + admin) — c'est en général le vocabulaire ou un registre qui enfle.
TOKEN_WARNING_THRESHOLD = 20_000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONSIGNES_PATH = PROJECT_ROOT / "app" / "prompts" / "wiki_consignes.md"


class WikiUnavailable(RuntimeError):
    """Le dossier du wiki ou le fichier de consignes est introuvable."""


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------


@dataclass
class WikiPage:
    id: str  # identité OKF : "/dossier/page.md"
    title: str
    type: str
    description: str = ""
    status: str = ""
    tags: List[str] = field(default_factory=list)
    # Facettes de la recherche : saisies tantôt en chaîne, tantôt en entier, tantôt en liste.
    gamme: List[str] = field(default_factory=list)
    systeme: List[str] = field(default_factory=list)
    famille: List[str] = field(default_factory=list)
    stale_after: str = ""
    stale: bool = False
    folder: str = "."
    reserved: bool = False
    missing: bool = False
    sources: List[Dict[str, Any]] = field(default_factory=list)
    generated: Dict[str, Any] = field(default_factory=dict)
    body: str = ""
    raw_text: str = ""  # le fichier entier, frontmatter comprise : c'est ce que lit le modèle
    out_links: List[str] = field(default_factory=list)
    in_degree: int = 0
    out_degree: int = 0
    frontmatter_error: str = ""

    @property
    def is_concept(self) -> bool:
        return not self.reserved and not self.missing

    @property
    def source_titles(self) -> List[str]:
        return [str(s.get("title") or s.get("resource") or "") for s in self.sources if s]

    def node(self) -> Dict[str, Any]:
        """Le nœud du graphe : le contrat de ``graph.json`` moins le corps."""
        return {
            "id": self.id,
            "title": self.title,
            "type": self.type,
            "description": self.description,
            "status": self.status,
            "tags": list(self.tags),
            "staleAfter": self.stale_after,
            "stale": self.stale,
            "folder": self.folder,
            "reserved": self.reserved,
            "missing": self.missing,
            "sources": self.source_titles,
            "inDegree": self.in_degree,
            "outDegree": self.out_degree,
        }


@dataclass
class WikiSnapshot:
    root: Path
    pages: Dict[str, WikiPage]
    links: List[Dict[str, Any]]
    index: WikiIndex
    system_prompt: str
    cache_key: str
    raw_files: List[str]
    signature: Tuple
    loaded_at: datetime
    lint: Dict[str, Any]
    # Le texte cherchable de chaque page, en minuscules — construit à la première
    # recherche, jeté avec l'instantané dès qu'un fichier change.
    search_index: Dict[str, str] = field(default_factory=dict, repr=False)

    # ---- mesures -------------------------------------------------------

    @property
    def char_count(self) -> int:
        return len(self.system_prompt)

    @property
    def estimated_tokens(self) -> int:
        return round(self.char_count / CHARS_PER_TOKEN)

    @property
    def token_warning(self) -> bool:
        return self.estimated_tokens > TOKEN_WARNING_THRESHOLD

    @property
    def concept_pages(self) -> List[WikiPage]:
        return [p for p in self.pages.values() if p.is_concept]

    # ---- charges utiles ------------------------------------------------

    def graph_payload(self) -> Dict[str, Any]:
        return {
            "generated": self.loaded_at.isoformat(timespec="seconds"),
            "nodes": [p.node() for p in sorted(self.pages.values(), key=lambda p: p.id)],
            "links": [dict(l) for l in self.links],
        }

    def page_payload(self, page_id: str) -> Optional[Dict[str, Any]]:
        page = self.pages.get(page_id)
        if page is None or page.missing:
            return None
        ins = [self.pages[l["source"]] for l in self.links if l["target"] == page_id]
        outs = [self.pages[l["target"]] for l in self.links if l["source"] == page_id]
        payload = page.node()
        payload.update(
            {
                "body": page.body,
                "sources": [dict(s) for s in page.sources],
                "generated": dict(page.generated),
                "inLinks": [{"id": q.id, "title": q.title, "type": q.type} for q in ins],
                "outLinks": [
                    {"id": q.id, "title": q.title, "type": q.type, "missing": q.missing}
                    for q in outs
                ],
            }
        )
        return payload

    def search(self, query: str, limit: int = 120) -> List[Dict[str, Any]]:
        """Les pages où TOUS les mots de ``query`` apparaissent — titre, description,
        étiquettes, chemin **et corps**. C'est ce qui permet de retrouver depuis l'accueil
        une référence citée dans un tableau (``TGY3704``) et pas seulement dans un titre.
        Chaque page revient avec l'extrait de la ligne qui porte le mot, vide si le mot
        n'est que dans les métadonnées."""
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []
        if not self.search_index:
            for page in self.pages.values():
                if page.is_concept:
                    self.search_index[page.id] = " ".join(
                        (page.title, page.description, page.type, " ".join(page.tags), page.id, page.body)
                    ).lower()
        hits: List[Dict[str, Any]] = []
        for page_id in sorted(self.search_index):
            haystack = self.search_index[page_id]
            if any(term not in haystack for term in terms):
                continue
            hits.append({"id": page_id, "excerpt": _excerpt(self.pages[page_id].body, terms)})
            if len(hits) >= limit:
                break
        return hits

    def raw_path(self, name: str) -> Optional[Path]:
        """Le PDF ``name`` de ``raw/`` — résolu contre la LISTE des fichiers, jamais contre le
        disque : aucun ``..`` ni sous-chemin ne peut sortir du dossier."""
        if name in self.raw_files:
            return self.root / "raw" / name
        return None

    def stats(self) -> Dict[str, Any]:
        concept = self.concept_pages
        types = Counter(p.type for p in concept)
        return {
            "pages": len(concept),
            "reserved": sum(1 for p in self.pages.values() if p.reserved),
            "ghosts": len(self.lint["ghosts"]),
            "links": len(self.links),
            "drafts": len(self.lint["drafts"]),
            "types": dict(sorted(types.items(), key=lambda kv: (-kv[1], kv[0]))),
            "orphans": self.lint["orphans"],
            "stale": self.lint["stale"],
            "chars": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "token_warning": self.token_warning,
            "cache_key": self.cache_key,
            "loaded_at": self.loaded_at.isoformat(timespec="seconds"),
            "wiki_dir": str(self.root),
            "raw_files": len(self.raw_files),
            "lint": {
                "frontmatter_errors": self.lint["frontmatter_errors"],
                "not_in_index": self.lint["not_in_index"],
                "index_dead_links": self.lint["index_dead_links"],
            },
            "last_call": last_call(),
        }


# ---------------------------------------------------------------------------
# Lecture des fichiers
# ---------------------------------------------------------------------------


def _excerpt(body: str, terms: List[str], width: int = 170) -> str:
    """La ligne du corps qui porte le premier mot trouvé, resserrée autour de lui."""
    low = body.lower()
    found = [pos for pos in (low.find(t) for t in terms) if pos >= 0]
    if not found:
        return ""
    pos = min(found)
    start = low.rfind("\n", 0, pos) + 1
    end = low.find("\n", pos)
    line = body[start:] if end < 0 else body[start:end]
    if len(line) <= width:
        return line.strip()
    lead = max(0, (pos - start) - width // 3)
    chunk = line[lead:lead + width]
    return ("… " if lead else "") + chunk.strip() + (" …" if lead + width < len(line) else "")


def wiki_root() -> Path:
    """La racine de connaissance (``wiki_llm/`` par défaut) : ``wiki/`` et ``raw/`` dedans."""
    configured = Path(settings.WIKI_DIR)
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


def _page_id(path: Path, wiki_dir: Path) -> str:
    return "/" + path.relative_to(wiki_dir).as_posix()


def _parse_page(path: Path, wiki_dir: Path) -> WikiPage:
    raw = path.read_text(encoding="utf-8")
    page_id = _page_id(path, wiki_dir)
    reserved = path.name in RESERVED
    folder = path.parent.relative_to(wiki_dir).as_posix() or "."

    meta: Dict[str, Any] = {}
    body = raw
    error = ""
    match = FRONTMATTER_RE.match(raw)
    if match:
        body = match.group(2)
        try:
            loaded = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            error = f"YAML illisible : {str(exc).splitlines()[0]}"
        else:
            if isinstance(loaded, dict):
                meta = loaded
            else:
                error = "frontmatter qui n'est pas un dictionnaire"
    elif not reserved:
        error = "frontmatter absente"

    page_type = str(meta.get("type") or "").strip()
    if reserved:
        page_type = "Réservé"
    elif not page_type:
        if not error:
            error = "champ type vide"
        page_type = "Sans type"

    title = str(meta.get("title") or "").strip()
    if not title:
        if path.name == "index.md":
            title = "Table des matières"
        elif path.name == "log.md":
            title = "Journal des opérations"
        else:
            title = path.stem.replace("-", " ")

    tags_raw = meta.get("tags")
    tags = [str(t) for t in tags_raw] if isinstance(tags_raw, list) else []
    sources_raw = meta.get("sources")
    sources = [s for s in sources_raw if isinstance(s, dict)] if isinstance(sources_raw, list) else []
    generated = meta.get("generated") if isinstance(meta.get("generated"), dict) else {}
    stale_after = meta.get("stale_after")
    stale_after_str = str(stale_after) if stale_after else ""

    return WikiPage(
        id=page_id,
        title=title,
        type=page_type,
        description=str(meta.get("description") or "").strip(),
        status=str(meta.get("status") or "").strip(),
        tags=tags,
        gamme=wiki_index.normalise(meta.get("gamme")),
        systeme=wiki_index.normalise(meta.get("systeme")),
        famille=wiki_index.normalise(meta.get("famille")),
        stale_after=stale_after_str,
        stale=bool(stale_after_str and stale_after_str < date.today().isoformat()),
        folder=folder,
        reserved=reserved,
        sources=sources,
        generated=generated,
        body=body.strip(),
        raw_text=raw,
        out_links=[target for _, target in LINK_RE.findall(body)],
        frontmatter_error=error,
    )


def _ghost(target: str) -> WikiPage:
    """Une cible de lien qui n'existe pas : en OKF un lien cassé est valide, il marque une
    connaissance pas encore écrite."""
    stripped = target.strip("/")
    return WikiPage(
        id=target,
        title=target.rsplit("/", 1)[-1][:-3].replace("-", " "),
        type="À écrire",
        description="Page liée mais pas encore écrite.",
        folder=stripped.rsplit("/", 1)[0] if "/" in stripped else ".",
        missing=True,
    )


def _signature(root: Path) -> Tuple:
    wiki_dir = root / "wiki"
    count, latest, size = 0, 0, 0
    if wiki_dir.is_dir():
        for dirpath, _, files in os.walk(wiki_dir):
            for name in files:
                if name.endswith(".md"):
                    st = os.stat(os.path.join(dirpath, name))
                    count += 1
                    size += st.st_size
                    latest = max(latest, st.st_mtime_ns)
    consignes = CONSIGNES_PATH.stat().st_mtime_ns if CONSIGNES_PATH.exists() else 0
    raw_dir = root / "raw"
    raw_count = sum(1 for _ in raw_dir.glob("*.pdf")) if raw_dir.is_dir() else 0
    return (count, latest, size, consignes, raw_count)


# ---------------------------------------------------------------------------
# Prompt système
# ---------------------------------------------------------------------------


def build_system_prompt(index: WikiIndex, consignes: str) -> Tuple[str, str]:
    """Consignes, vocabulaire, index des anomalies. **Pas le wiki** : il ne tient pas.

    Le modèle n'a en permanence que de quoi *formuler une recherche* — les types de pages, les
    tags, les gammes, les systèmes — et de quoi *savoir qu'une anomalie existe* : un identifiant
    et un sujet par entrée. Les pages arrivent par l'outil ``chercher`` ; le détail d'une entrée
    par ``lire_anomalie``, et les entrées qui concernent la réponse sont de toute façon injectées
    par le serveur (règle 2).

    Retourne ``(prompt, clé de cache)``. La clé couvre le prompt ENTIER, consignes comprises :
    modifier une consigne invalide donc le cache au lieu de servir un préfixe périmé.
    """
    prompt = "\n\n".join([
        consignes.rstrip(),
        f"===== VOCABULAIRE DU WIKI ({len(index.entries)} pages indexées) =====",
        index.vocabulaire(),
        f"===== INDEX DES ANOMALIES ({len(index.anomalies)} entrées) =====",
        "Identifiant et sujet seulement. Le détail des entrées qui concernent ta réponse t'est "
        "fourni automatiquement ; pour les autres, appelle lire_anomalie(identifiant).",
        index.index_anomalies(),
    ])
    key = "lia-wiki-" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:32]
    return prompt, key


# ---------------------------------------------------------------------------
# Construction de l'instantané
# ---------------------------------------------------------------------------


def _lint(pages: Dict[str, WikiPage]) -> Dict[str, Any]:
    concept = [p for p in pages.values() if p.is_concept]
    index = pages.get("/index.md")
    index_links = set(index.out_links) if index else set()
    # Une orpheline n'est citée par aucune page CONCEPT : index.md cite tout le monde, le
    # compter rendrait le lint aveugle.
    cited_by_concepts = {
        target for p in concept for target in p.out_links
    }
    return {
        "orphans": sorted(p.id for p in concept if p.id not in cited_by_concepts),
        "ghosts": sorted(p.id for p in pages.values() if p.missing),
        "stale": sorted(p.id for p in concept if p.stale),
        "drafts": sorted(p.id for p in concept if p.status == "draft"),
        "frontmatter_errors": sorted(
            f"{p.id} — {p.frontmatter_error}" for p in concept if p.frontmatter_error
        ),
        "not_in_index": sorted(p.id for p in concept if p.id not in index_links),
        "index_dead_links": sorted(
            t for t in index_links if t not in pages or pages[t].missing
        ),
    }


def load_snapshot(root: Optional[Path] = None, signature: Optional[Tuple] = None) -> WikiSnapshot:
    root = root or wiki_root()
    wiki_dir = root / "wiki"
    if not wiki_dir.is_dir():
        raise WikiUnavailable(f"wiki/ introuvable sous {root}")
    if not CONSIGNES_PATH.is_file():
        raise WikiUnavailable(f"consignes introuvables : {CONSIGNES_PATH}")
    consignes = CONSIGNES_PATH.read_text(encoding="utf-8")

    pages: Dict[str, WikiPage] = {}
    for path in sorted(wiki_dir.rglob("*.md")):
        page = _parse_page(path, wiki_dir)
        pages[page.id] = page

    links: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for page in list(pages.values()):
        for target in page.out_links:
            if target not in pages:
                pages[target] = _ghost(target)
            key = (page.id, target)
            if key in seen:
                seen[key]["count"] += 1
            else:
                seen[key] = {"source": page.id, "target": target, "count": 1}
                links.append(seen[key])
    for link in links:
        pages[link["source"]].out_degree += 1
        pages[link["target"]].in_degree += 1

    index = WikiIndex(pages.values())
    prompt, cache_key = build_system_prompt(index, consignes)
    raw_dir = root / "raw"
    raw_files = sorted(p.name for p in raw_dir.glob("*.pdf")) if raw_dir.is_dir() else []

    snapshot = WikiSnapshot(
        root=root,
        pages=pages,
        links=links,
        index=index,
        system_prompt=prompt,
        cache_key=cache_key,
        raw_files=raw_files,
        signature=signature if signature is not None else _signature(root),
        loaded_at=datetime.now(),
        lint=_lint(pages),
    )
    logger.info(
        "[wiki] %d pages, %d liens, %d fantômes, %d entrées d'anomalie — prompt permanent "
        "%s car. (~%s tokens), clé %s",
        len(snapshot.concept_pages),
        len(links),
        len(snapshot.lint["ghosts"]),
        len(index.anomalies),
        f"{snapshot.char_count:,}".replace(",", " "),
        f"{snapshot.estimated_tokens:,}".replace(",", " "),
        cache_key,
    )
    if snapshot.token_warning:
        logger.warning(
            "[wiki] le prompt permanent dépasse %s tokens estimés : il est payé à chaque appel "
            "d'un tour d'outils — vérifier le vocabulaire et les registres d'anomalies",
            f"{TOKEN_WARNING_THRESHOLD:,}".replace(",", " "),
        )
    return snapshot


_lock = threading.Lock()
_current: Optional[WikiSnapshot] = None
_last_call: Optional[Dict[str, Any]] = None


def get_snapshot() -> WikiSnapshot:
    """L'instantané courant, reconstruit si un fichier a changé depuis."""
    global _current
    root = wiki_root()
    signature = _signature(root)
    current = _current
    if current is not None and current.root == root and current.signature == signature:
        return current
    with _lock:
        current = _current
        if current is not None and current.root == root and current.signature == signature:
            return current
        _current = load_snapshot(root, signature=signature)
        return _current


def reset_snapshot() -> None:
    global _current
    _current = None


def record_call(info: Dict[str, Any]) -> None:
    """Mémorise la mesure du dernier appel réel (tokens, cache, latence) pour l'admin."""
    global _last_call
    _last_call = dict(info)


def last_call() -> Optional[Dict[str, Any]]:
    return dict(_last_call) if _last_call else None
