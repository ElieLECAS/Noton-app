"""Index d'entrée sémantique des arbres SAV (niveau 0) — on cherche dans l'ARBRE.

Reconstruit à la publication d'un arbre (rebuild_tree_entries) et à chaque modification
du registre de symptômes (rebuild_symptom_entries). La correspondance d'une demande est
ÉTAGÉE et s'arrête au premier niveau concluant :
  1. slug exact   — detected_symptom de la compréhension fusionnée ;
  2. alias lexical — vocabulaire client normalisé (sans accents) contenu dans la demande ;
  3. sémantique   — cosinus pgvector sur l'index (pré-filtré par périmètre), seuil MIN_SIM.

La sémantique PROPOSE (chip / picker), elle ne démarre jamais un parcours seule.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlmodel import Session, select

from app.models.document_category import DocumentCategory
from app.models.guided_entry import GuidedEntryIndex, GuidedSymptomAlias
from app.models.guided_tree import GuidedTree

logger = logging.getLogger(__name__)

# Seuil de similarité cosinus pour l'entrée sémantique (en dur — pas de flag).
MIN_SIM = 0.55
# Axes de périmètre partagés avec Document.
PERIMETER_AXES = ("materials", "product_types", "source", "proferm_gammes")


class EntryMatch(BaseModel):
    tree_id: int
    tree_slug: str
    tree_title: str
    tree_version: int
    entry_node_key: Optional[str] = None  # None → racine
    symptom_slug: str = ""
    method: str = ""  # symptom_slug | alias | semantic
    score: float = 1.0


def normalize_text(value: str) -> str:
    """minuscules + sans accents + espaces normalisés (matching lexical robuste)."""
    v = unicodedata.normalize("NFD", (value or "").lower())
    v = "".join(c for c in v if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", v).strip()


def perimeter_compatible(filters: Optional[Dict[str, Any]], perimeter: Optional[Dict[str, Any]]) -> bool:
    """Compatible si, pour chaque axe renseigné DES DEUX côtés, l'intersection est non vide."""
    if not filters or not perimeter:
        return True
    for axis in PERIMETER_AXES:
        f = {normalize_text(str(x)) for x in (filters.get(axis) or []) if str(x).strip()}
        p = {normalize_text(str(x)) for x in (perimeter.get(axis) or []) if str(x).strip()}
        if f and p and not (f & p):
            return False
    return True


# ---------------------------------------------------------------------------
# Registre de symptômes
# ---------------------------------------------------------------------------


def symptom_catalog(session: Session) -> Dict[str, Dict[str, str]]:
    """{slug: {label, description}} — BDD (axis=symptom) avec repli statique."""
    rows = session.exec(
        select(DocumentCategory).where(
            DocumentCategory.axis == "symptom", DocumentCategory.is_active == True  # noqa: E712
        )
    ).all()
    if rows:
        return {r.slug: {"label": r.label, "description": r.description or ""} for r in rows}
    from app.services.category_catalog import SYMPTOM_DESCRIPTIONS, SYMPTOM_LABELS

    return {
        slug: {"label": label, "description": SYMPTOM_DESCRIPTIONS.get(slug, "")}
        for slug, label in SYMPTOM_LABELS.items()
    }


def symptom_aliases(session: Session) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for row in session.exec(select(GuidedSymptomAlias)).all():
        out.setdefault(row.symptom_slug, []).append(row.alias)
    return out


# ---------------------------------------------------------------------------
# Reconstruction de l'index
# ---------------------------------------------------------------------------


def _embed_rows(rows: List[GuidedEntryIndex]) -> None:
    """Embeddings en batch (best-effort : un échec API n'empêche pas la publication —
    le matching lexical continue de fonctionner, embedding=None)."""
    texts = [r.text for r in rows]
    if not texts:
        return
    try:
        from app.services.embedding_service import generate_embeddings_batch

        vectors = generate_embeddings_batch(texts)
        for row, vec in zip(rows, vectors):
            row.embedding = vec
    except Exception as exc:  # noqa: BLE001
        logger.warning("[guided_entry] embeddings d'index échoués (lexical seul) : %s", exc)


def remove_tree_entries(session: Session, tree_id: int) -> None:
    session.execute(
        sa_text("DELETE FROM guidedentryindex WHERE tree_id = :tid"), {"tid": tree_id}
    )
    session.commit()


def rebuild_tree_entries(session: Session, tree: GuidedTree, snapshot: Dict[str, Any]) -> int:
    """Une entrée « tree » + une entrée « node » par nœud non terminal significatif."""
    session.execute(
        sa_text("DELETE FROM guidedentryindex WHERE tree_id = :tid"), {"tid": tree.id}
    )

    catalog = symptom_catalog(session)
    aliases = symptom_aliases(session)
    sym = tree.entry_symptom or ""
    sym_label = catalog.get(sym, {}).get("label", sym)
    sym_aliases = " ".join(aliases.get(sym, []))

    # Les DESCRIPTIONS des cas (y compris les solutions) nourrissent le texte de l'arbre :
    # c'est ce qui permet de reconnaître la demande quand le client l'exprime avec ses mots,
    # sans pour autant faire entrer le parcours directement sur une solution.
    leaf_words = " ".join(
        str(node.get("message") or "").strip()
        for node in (snapshot.get("nodes") or {}).values()
        if node.get("is_terminal") and str(node.get("message") or "").strip()
    )[:2000]

    rows: List[GuidedEntryIndex] = []
    tree_text = " ".join(
        t for t in [tree.title, tree.description, sym_label, sym_aliases, leaf_words] if t
    ).strip()
    rows.append(
        GuidedEntryIndex(
            space_id=tree.space_id,
            tree_id=tree.id,
            tree_version=snapshot.get("version"),
            entry_kind="tree",
            ref_key=tree.slug,
            label=tree.title[:300],
            text=tree_text,
            filters=tree.perimeter,
        )
    )

    for node_key, node in (snapshot.get("nodes") or {}).items():
        if node.get("is_terminal"):
            continue
        choice_labels = " ".join(
            str(c.get("label") or "") for c in (node.get("choices") or [])
        )
        node_text = " ".join(
            t for t in [node.get("title") or "", node.get("message") or "", choice_labels] if t
        ).strip()
        if len(node_text) < 12:
            continue
        filters = dict(tree.perimeter or {})
        for axis, values in (node.get("perimeter_condition") or {}).items():
            if values:
                filters[axis] = values
        rows.append(
            GuidedEntryIndex(
                space_id=tree.space_id,
                tree_id=tree.id,
                tree_version=snapshot.get("version"),
                entry_kind="node",
                ref_key=str(node_key)[:160],
                label=(node.get("title") or node.get("message") or "")[:300],
                text=node_text,
                filters=filters or None,
            )
        )

    _embed_rows(rows)
    for r in rows:
        session.add(r)
    session.commit()
    logger.info("[guided_entry] index reconstruit tree=%s : %d entrées", tree.slug, len(rows))
    return len(rows)


def rebuild_symptom_entries(session: Session) -> int:
    """Entrées « symptom » globales (space_id null, tree_id null)."""
    session.execute(sa_text("DELETE FROM guidedentryindex WHERE entry_kind = 'symptom'"))
    catalog = symptom_catalog(session)
    aliases = symptom_aliases(session)
    rows: List[GuidedEntryIndex] = []
    for slug, info in catalog.items():
        txt = " ".join(
            t for t in [info.get("label", ""), info.get("description", ""), " ".join(aliases.get(slug, []))] if t
        ).strip()
        rows.append(
            GuidedEntryIndex(
                space_id=None,
                tree_id=None,
                entry_kind="symptom",
                ref_key=slug[:160],
                label=info.get("label", slug)[:300],
                text=txt,
            )
        )
    _embed_rows(rows)
    for r in rows:
        session.add(r)
    session.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Correspondance d'une demande entrante (étagée)
# ---------------------------------------------------------------------------


def _published_trees(session: Session, space_id: int) -> List[GuidedTree]:
    trees = session.exec(
        select(GuidedTree).where(GuidedTree.status == "published")
    ).all()
    return [t for t in trees if t.space_id is None or t.space_id == space_id]


def _tree_for_symptom(
    session: Session, space_id: int, symptom: str, perimeter: Optional[Dict[str, Any]]
) -> Optional[GuidedTree]:
    candidates = [
        t
        for t in _published_trees(session, space_id)
        if (t.entry_symptom or "").strip().lower() == symptom.strip().lower()
        and perimeter_compatible(t.perimeter, perimeter)
    ]
    if not candidates:
        return None
    # Espace spécifique avant global, puis priorité décroissante.
    candidates.sort(key=lambda t: (t.space_id is None, -(t.priority or 0)))
    return candidates[0]


def _match_alias_symptom(session: Session, query_text: str) -> str:
    """Retourne le slug de symptôme dont un alias (ou le label) apparaît dans la demande."""
    q = f" {normalize_text(query_text)} "
    if not q.strip():
        return ""
    catalog = symptom_catalog(session)
    for slug, alias_list in symptom_aliases(session).items():
        for alias in alias_list:
            a = normalize_text(alias)
            if a and f" {a} " in q or (a and a in q and len(a) > 6):
                return slug
    for slug, info in catalog.items():
        label = normalize_text(info.get("label", ""))
        if label and label in q:
            return slug
    return ""


def _match_semantic(
    session: Session,
    space_id: int,
    query_text: str,
    perimeter: Optional[Dict[str, Any]],
) -> Optional[EntryMatch]:
    from app.services.embedding_service import generate_embedding

    vec = generate_embedding(query_text)
    if not vec:
        return None
    vec_literal = "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"
    rows = session.execute(
        sa_text(
            """
            SELECT id, space_id, tree_id, tree_version, entry_kind, ref_key, label, filters,
                   1 - (embedding <=> CAST(:v AS vector)) AS sim
            FROM guidedentryindex
            WHERE embedding IS NOT NULL
              AND (space_id IS NULL OR space_id = :space_id)
            ORDER BY embedding <=> CAST(:v AS vector)
            LIMIT 12
            """
        ),
        {"v": vec_literal, "space_id": space_id},
    ).mappings().all()

    for row in rows:
        sim = float(row["sim"] or 0.0)
        if sim < MIN_SIM:
            break  # trié par similarité décroissante
        if not perimeter_compatible(row["filters"], perimeter):
            continue
        if row["entry_kind"] == "symptom":
            tree = _tree_for_symptom(session, space_id, row["ref_key"], perimeter)
            if tree is None:
                continue
            return EntryMatch(
                tree_id=tree.id,
                tree_slug=tree.slug,
                tree_title=tree.title,
                tree_version=tree.current_version,
                entry_node_key=None,
                symptom_slug=row["ref_key"],
                method="semantic",
                score=sim,
            )
        tree = session.get(GuidedTree, row["tree_id"])
        if tree is None or tree.status != "published":
            continue
        return EntryMatch(
            tree_id=tree.id,
            tree_slug=tree.slug,
            tree_title=tree.title,
            tree_version=int(row["tree_version"] or tree.current_version),
            entry_node_key=row["ref_key"] if row["entry_kind"] == "node" else None,
            symptom_slug=tree.entry_symptom or "",
            method="semantic",
            score=sim,
        )
    return None


def match_entry(
    session: Session,
    *,
    space_id: int,
    query_text: str,
    detected_symptom: str = "",
    perimeter: Optional[Dict[str, Any]] = None,
) -> Optional[EntryMatch]:
    """Correspondance étagée. Retourne None si rien d'assez sûr (→ GuidedGap côté appelant)."""
    # 1. Slug exact (compréhension fusionnée)
    if detected_symptom:
        tree = _tree_for_symptom(session, space_id, detected_symptom, perimeter)
        if tree is not None:
            return EntryMatch(
                tree_id=tree.id,
                tree_slug=tree.slug,
                tree_title=tree.title,
                tree_version=tree.current_version,
                symptom_slug=detected_symptom,
                method="symptom_slug",
            )

    # 2. Alias lexical (vocabulaire client, sans accents)
    alias_slug = _match_alias_symptom(session, query_text)
    if alias_slug:
        tree = _tree_for_symptom(session, space_id, alias_slug, perimeter)
        if tree is not None:
            return EntryMatch(
                tree_id=tree.id,
                tree_slug=tree.slug,
                tree_title=tree.title,
                tree_version=tree.current_version,
                symptom_slug=alias_slug,
                method="alias",
            )

    # 3. Sémantique (cosinus sur l'index, seuil MIN_SIM)
    try:
        return _match_semantic(session, space_id, query_text, perimeter)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[guided_entry] match sémantique échoué : %s", exc)
        return None


def list_sav_entries(session: Session, space_id: int) -> List[Dict[str, Any]]:
    """Contenu du picker « Diagnostic SAV » : symptômes → arbres publiés de l'espace."""
    catalog = symptom_catalog(session)
    out: List[Dict[str, Any]] = []
    for tree in _published_trees(session, space_id):
        sym = tree.entry_symptom or ""
        out.append(
            {
                "tree_id": tree.id,
                "tree_slug": tree.slug,
                "title": tree.title,
                "version": tree.current_version,
                "symptom_slug": sym,
                "symptom_label": catalog.get(sym, {}).get("label", sym or "Autre"),
                "description": tree.description or "",
                "perimeter": tree.perimeter,
            }
        )
    out.sort(key=lambda e: (e["symptom_label"], e["title"]))
    return out
