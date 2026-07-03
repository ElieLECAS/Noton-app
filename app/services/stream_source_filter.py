"""Filtre de stream pour le bloc final ``<sources>{...}</sources>`` de la génération CAG.

Le modèle termine sa réponse par une ligne machine listant les documents/pages réellement
utilisés. Ce bloc ne doit JAMAIS atteindre l'utilisateur : ce filtre retient les chunks
suspects (préfixe de balise possible) tant que l'ambiguïté n'est pas levée, capture le
bloc complet, et expose le JSON parsé pour construire les sources UI.

Robustesse : balise coupée entre deux chunks SSE, bloc absent, JSON invalide, balise
ouverte jamais fermée (plafond de capture → on relâche le texte tel quel).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_OPEN = "<sources>"
_CLOSE = "</sources>"
# Au-delà, ce n'est manifestement pas notre petit bloc JSON : on relâche tout.
_MAX_CAPTURE_CHARS = 4000


class SourcesTagStreamFilter:
    """Filtre incrémental : ``feed(chunk) -> texte affichable``, puis ``finalize()``."""

    def __init__(self) -> None:
        self._pending = ""  # suffixe retenu : préfixe potentiel de <sources>
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
                close_idx = text.find(_CLOSE)
                if close_idx >= 0:
                    self._captured += text[:close_idx]
                    self._raw_block = self._captured
                    self._captured = ""
                    self._capturing = False
                    text = text[close_idx + len(_CLOSE):]
                    continue
                # Fermeture peut-être coupée entre deux chunks : retenir son préfixe.
                hold = _longest_suffix_prefix(text, _CLOSE)
                self._captured += text[: len(text) - hold]
                self._pending = text[len(text) - hold:] if hold else ""
                if len(self._captured) > _MAX_CAPTURE_CHARS:
                    # Faux positif (balise jamais fermée) : tout relâcher.
                    out.append(_OPEN + self._captured + self._pending)
                    self._captured = ""
                    self._pending = ""
                    self._capturing = False
                return "".join(out)

            open_idx = text.find(_OPEN)
            if open_idx >= 0:
                out.append(text[:open_idx])
                self._capturing = True
                text = text[open_idx + len(_OPEN):]
                continue

            # Balise d'ouverture peut-être coupée : retenir le suffixe ambigu.
            hold = _longest_suffix_prefix(text, _OPEN)
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
            if candidate.lstrip().startswith("{"):
                self._raw_block = candidate
                return ""
            return _OPEN + candidate
        tail = self._pending
        self._pending = ""
        return tail

    @property
    def used_documents(self) -> List[Dict[str, Any]]:
        """Liste [{"doc": int, "pages": [int, …]}, …] parsée du bloc (vide si absent/invalide)."""
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
            try:
                doc = int(item.get("doc"))
            except (TypeError, ValueError):
                continue
            pages = [int(p) for p in (item.get("pages") or []) if isinstance(p, (int, float))]
            result.append({"doc": doc, "pages": pages})
        return result


def _longest_suffix_prefix(text: str, tag: str) -> int:
    """Longueur du plus long suffixe de ``text`` qui est un préfixe strict de ``tag``."""
    max_k = min(len(text), len(tag) - 1)
    for k in range(max_k, 0, -1):
        if text[-k:] == tag[:k]:
            return k
    return 0
