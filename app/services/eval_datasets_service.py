"""Jeux d'évaluation LIVRÉS avec l'application — plus besoin de glisser un JSON.

La page « Qualité RAG » de l'administration ne savait lire qu'un fichier déposé à la main.
Un jeu de référence qui vit sur le poste de quelqu'un n'est pas un jeu de référence : il se
perd, il diverge, et deux mesures ne se comparent plus. Les fichiers de
``tests/fixtures/golden/`` sont donc exposés tels quels, versionnés avec le code.

Deux familles cohabitent, et ce module les ramène à UNE forme :

  * **retrieval** — l'ancien format (liste plate, ``pages_attendues`` par titre de
    document). Mesure : la bonne page remonte-t-elle ?
  * **generation** — le format du 14/09 (``entries``, ``preuve`` par identifiant de
    document, ``attendu`` typé). Mesure : la réponse dit-elle la bonne valeur ?

Un jeu de génération sert aussi à mesurer le retriever seul : ses pages de preuve sont des
pages attendues. L'inverse est faux — sans ``attendu``, on ne peut rien dire de la réponse.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from app.models.document import Document

logger = logging.getLogger(__name__)

GOLDEN_DIR = Path("tests/fixtures/golden")

KIND_RETRIEVAL = "retrieval"
KIND_GENERATION = "generation"


def _safe_key(path: Path) -> str:
    return path.stem


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[jeux d'éval] %s illisible : %s", path.name, exc)
        return None


def _entries_of(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    if isinstance(raw, dict):
        for cle in ("entries", "questions", "items"):
            if isinstance(raw.get(cle), list):
                return [e for e in raw[cle] if isinstance(e, dict)]
    return []


def _kind_of(raw: Any, entries: List[Dict[str, Any]]) -> str:
    if isinstance(raw, dict) and raw.get("kind") in (KIND_RETRIEVAL, KIND_GENERATION):
        return str(raw["kind"])
    # Un « attendu » typé est la signature d'un jeu de génération.
    return KIND_GENERATION if any("attendu" in e for e in entries) else KIND_RETRIEVAL


@lru_cache(maxsize=1)
def _dataset_paths() -> tuple:
    if not GOLDEN_DIR.is_dir():
        return tuple()
    return tuple(sorted(p for p in GOLDEN_DIR.glob("*.json") if p.is_file()))


def clear_dataset_cache() -> None:
    """Après l'ajout d'un fichier dans le dossier (utile en développement)."""
    _dataset_paths.cache_clear()


def list_datasets() -> List[Dict[str, Any]]:
    """Inventaire des jeux disponibles, sans leurs questions (pour peupler un menu)."""
    out: List[Dict[str, Any]] = []
    for path in _dataset_paths():
        raw = _read_json(path)
        if raw is None:
            continue
        entries = _entries_of(raw)
        if not entries:
            continue
        entete = raw if isinstance(raw, dict) else {}
        out.append(
            {
                "key": _safe_key(path),
                "filename": path.name,
                "kind": _kind_of(raw, entries),
                "space_id": entete.get("space_id"),
                "questions": len(entries),
                "description": entete.get("description") or "",
                "created": entete.get("created") or "",
            }
        )
    return out


def _titles_by_id(session: Session, document_ids: List[int]) -> Dict[int, str]:
    if not document_ids:
        return {}
    rows = session.exec(select(Document).where(Document.id.in_(document_ids))).all()
    return {int(d.id): (d.title or f"Document {d.id}") for d in rows}


def _normalize_generation_entry(
    entry: Dict[str, Any], titres: Dict[int, str]
) -> Dict[str, Any]:
    """Un item de golden de génération → la forme attendue par la page d'administration.

    ``preuve`` (par identifiant) devient ``pages_attendues`` (par titre) : le premier
    format est stable dans le temps, le second est celui que l'évaluateur de retriever sait
    comparer. Les deux sont conservés — l'identifiant sert à mesurer le document packé,
    le titre à mesurer la page retrouvée.
    """
    pages_attendues: List[Dict[str, Any]] = []
    doc_ids: List[int] = []
    for item in entry.get("preuve") or []:
        did = item.get("document_id")
        if did is None:
            continue
        did = int(did)
        doc_ids.append(did)
        pages_attendues.append(
            {
                "document_id": did,
                "document_title": titres.get(did, f"Document {did}"),
                "pages": [int(p) for p in (item.get("pages") or [])],
            }
        )
    return {
        "id": entry.get("id"),
        "question": entry.get("question") or "",
        "type": entry.get("difficulte") or "mono-document",
        "pages_attendues": pages_attendues,
        "acceptable_document_ids": sorted(set(doc_ids)),
        "attendu": entry.get("attendu"),
        "verite": entry.get("verite") or "",
        "tags": entry.get("tags") or [],
        "difficulte": entry.get("difficulte") or "",
    }


def _normalize_retrieval_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(entry)
    out.setdefault("type", "mono-document")
    out.setdefault("pages_attendues", [])
    out.setdefault("tags", [])
    return out


def load_dataset(session: Session, key: str) -> Optional[Dict[str, Any]]:
    """Un jeu complet, questions normalisées. ``None`` si la clé est inconnue."""
    for path in _dataset_paths():
        if _safe_key(path) != key:
            continue
        raw = _read_json(path)
        if raw is None:
            return None
        entries = _entries_of(raw)
        kind = _kind_of(raw, entries)
        entete = raw if isinstance(raw, dict) else {}
        if kind == KIND_GENERATION:
            doc_ids = sorted(
                {
                    int(item["document_id"])
                    for e in entries
                    for item in (e.get("preuve") or [])
                    if item.get("document_id") is not None
                }
            )
            titres = _titles_by_id(session, doc_ids)
            questions = [_normalize_generation_entry(e, titres) for e in entries]
            manquants = [d for d in doc_ids if d not in titres]
        else:
            questions = [_normalize_retrieval_entry(e) for e in entries]
            manquants = []
        return {
            "key": key,
            "filename": path.name,
            "kind": kind,
            "space_id": entete.get("space_id"),
            "description": entete.get("description") or "",
            "questions": questions,
            # Un document du jeu absent de la base rendrait la mesure trompeuse (toutes
            # ses questions échoueraient sans que le retriever y soit pour rien).
            "documents_manquants": manquants,
        }
    return None
