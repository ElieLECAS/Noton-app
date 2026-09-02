"""Filtres de stream pour les blocs machine de fin de réponse : ``<sources>`` et ``<evidence>``.

Le modèle termine sa réponse par une ou deux lignes machine :
  * ``<sources>{"used":[{"doc_id":405,"pages":[3,4]}]}</sources>`` — documents/pages
    réellement utilisés (l'ancienne forme ``{"doc":1,...}`` par index de pack reste acceptée) ;
  * ``<evidence>["citation exacte (doc 405 p.111)", …]</evidence>`` — citations verbatim
    des pages lues, vérifiées ensuite par le contrôle programmatique.

Ces blocs ne doivent JAMAIS atteindre l'utilisateur : le filtre retient les chunks suspects
(préfixe de balise possible) tant que l'ambiguïté n'est pas levée, capture le bloc complet,
et expose le contenu parsé.

Robustesse : balise coupée entre deux chunks SSE, bloc absent, JSON invalide, balise
ouverte jamais fermée (plafond de capture → on relâche le texte tel quel).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Au-delà, ce n'est manifestement pas notre petit bloc JSON : on relâche tout.
_MAX_CAPTURE_CHARS = 4000


class MachineTagStreamFilter:
    """Filtre incrémental d'UNE balise : ``feed(chunk) -> texte affichable``, puis ``finalize()``."""

    def __init__(self, tag: str) -> None:
        self._open = f"<{tag}>"
        self._close = f"</{tag}>"
        self._pending = ""  # suffixe retenu : préfixe potentiel de la balise
        self._capturing = False
        self._captured = ""
        self._raw_block: Optional[str] = None

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def feed(self, chunk: str) -> str:
        """Traite un chunk de stream et retourne la partie affichable."""
        if not chunk:
            return ""
        out: List[str] = []
        text = self._pending + chunk
        self._pending = ""

        while text:
            if self._capturing:
                close_idx = text.find(self._close)
                if close_idx >= 0:
                    self._captured += text[:close_idx]
                    self._raw_block = self._captured
                    self._captured = ""
                    self._capturing = False
                    text = text[close_idx + len(self._close):]
                    continue
                # Fermeture peut-être coupée entre deux chunks : retenir son préfixe.
                hold = _longest_suffix_prefix(text, self._close)
                self._captured += text[: len(text) - hold]
                self._pending = text[len(text) - hold:] if hold else ""
                if len(self._captured) > _MAX_CAPTURE_CHARS:
                    # Faux positif (balise jamais fermée) : tout relâcher.
                    out.append(self._open + self._captured + self._pending)
                    self._captured = ""
                    self._pending = ""
                    self._capturing = False
                return "".join(out)

            open_idx = text.find(self._open)
            if open_idx >= 0:
                out.append(text[:open_idx])
                self._capturing = True
                text = text[open_idx + len(self._open):]
                continue

            # Balise d'ouverture peut-être coupée : retenir le suffixe ambigu.
            hold = _longest_suffix_prefix(text, self._open)
            out.append(text[: len(text) - hold])
            self._pending = text[len(text) - hold:] if hold else ""
            return "".join(out)

        return "".join(out)

    def finalize(self) -> str:
        """Fin de stream : relâche ce qui était retenu à tort (balise jamais complétée)."""
        if self._capturing:
            # Stream coupé au milieu du bloc : si le contenu ressemble au JSON attendu,
            # on l'avale (mieux vaut perdre le bloc que l'afficher) ; sinon on le relâche.
            candidate = self._captured + self._pending
            self._captured = ""
            self._pending = ""
            self._capturing = False
            if candidate.lstrip().startswith(("{", "[")):
                self._raw_block = candidate
                return ""
            return self._open + candidate
        tail = self._pending
        self._pending = ""
        return tail

    @property
    def raw_block(self) -> Optional[str]:
        return self._raw_block


class SourcesTagStreamFilter(MachineTagStreamFilter):
    """Bloc ``<sources>`` : documents et pages réellement utilisés."""

    def __init__(self) -> None:
        super().__init__("sources")

    @property
    def used_documents(self) -> List[Dict[str, Any]]:
        """Liste ``[{"doc": int, "pages": [...]}, …]`` (index de pack) ou
        ``[{"doc_id": int, "pages": [...]}, …]`` (identifiant de document), parsée du bloc
        (vide si absent/invalide). Un item peut porter les deux clés."""
        if not self._raw_block:
            return []
        try:
            data = json.loads(self._raw_block.strip())
        except (ValueError, TypeError):
            logger.info("[sources-filter] bloc <sources> illisible : %r", self._raw_block[:200])
            return []
        used = data.get("used") if isinstance(data, dict) else None
        if not isinstance(used, list):
            return []
        result: List[Dict[str, Any]] = []
        for item in used:
            if not isinstance(item, dict):
                continue
            entry: Dict[str, Any] = {}
            for key in ("doc", "doc_id"):
                if item.get(key) is None:
                    continue
                try:
                    entry[key] = int(item.get(key))
                except (TypeError, ValueError):
                    continue
            if not entry:
                continue
            entry["pages"] = [int(p) for p in (item.get("pages") or []) if isinstance(p, (int, float))]
            result.append(entry)
        return result


class EvidenceTagStreamFilter(MachineTagStreamFilter):
    """Bloc ``<evidence>`` : citations verbatim que le lecteur affirme avoir lues."""

    def __init__(self) -> None:
        super().__init__("evidence")

    @property
    def citations(self) -> List[str]:
        """Citations (chaînes non vides) parsées du bloc ; JSON attendu, texte toléré."""
        if not self._raw_block:
            return []
        raw = self._raw_block.strip()
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            # Repli : une citation par ligne, guillemets français/anglais retirés.
            lines = [ln.strip().strip("«»\"'-• ").strip() for ln in raw.splitlines()]
            return [ln for ln in lines if len(ln) >= 3]
        items: List[Any] = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("citations", "evidence", "quotes", "used"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
        result: List[str] = []
        for item in items:
            if isinstance(item, str):
                txt = item.strip()
            elif isinstance(item, dict):
                txt = str(item.get("quote") or item.get("text") or item.get("citation") or "").strip()
            else:
                continue
            if txt:
                result.append(txt)
        return result


def chain_filters(chunk: str, filters: List[MachineTagStreamFilter]) -> str:
    """Passe un chunk successivement dans chaque filtre (ordre : sources puis evidence)."""
    text = chunk
    for f in filters:
        text = f.feed(text)
        if not text:
            return ""
    return text


def finalize_filters(filters: List[MachineTagStreamFilter]) -> str:
    """Fin de stream : le reliquat du filtre N traverse les filtres N+1…, puis chacun finalise."""
    tail = ""
    for i, f in enumerate(filters):
        released = f.feed(tail) if tail else ""
        released += f.finalize()
        tail = released
    return tail


_CITATION_LOCATION_RE = re.compile(r"\((?:doc(?:ument)?\s*\d+[^)]*?)\)\s*$", re.IGNORECASE)


def strip_citation_location(citation: str) -> str:
    """Retire un suffixe « (doc 405 p.111) » pour ne garder que le texte cité."""
    return _CITATION_LOCATION_RE.sub("", citation or "").strip()


def _longest_suffix_prefix(text: str, tag: str) -> int:
    """Longueur du plus long suffixe de ``text`` qui est un préfixe strict de ``tag``."""
    max_k = min(len(text), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if text[-k:] == tag[:k]:
            return k
    return 0
