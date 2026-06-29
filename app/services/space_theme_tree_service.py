"""Arbre thématique (carte mentale) d'un espace : racine → familles → catégories.

Construit une hiérarchie imbriquée et cliquable à partir des catégories KAG présentes
dans l'espace, en repliant les 16 catégories ``task`` sous des familles curées
(:mod:`app.services.theme_tree_catalog`). Les axes plats (``doc_type``,
``lifecycle_phase``, ``symptom``) donnent un arbre à deux niveaux (racine → catégorie).

Toutes les valeurs de comptage sont des DISTINCT calculés à la requête : un chunk taggé
sur plusieurs catégories d'une même famille n'est compté qu'une fois au niveau famille.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import bindparam, text
from sqlmodel import Session

from app.config import settings
from app.services.category_catalog import AXIS_TASK
from app.services.theme_tree_catalog import (
    FALLBACK_FAMILY,
    FLAT_AXIS_LABELS,
    axes_for_payload,
    entity_icon,
    entity_types_for_category,
    family_for_task_slug,
    family_meta,
    family_order_index,
    is_valid_axis,
)

__all__ = [
    "build_space_theme_tree",
    "resolve_node_to_category_ids",
    "parse_entity_node_key",
    "chunk_ids_for_entity_category",
    "get_space_theme_node_pages",
]


def parse_entity_node_key(node_key: str) -> Optional[Tuple[int, int]]:
    """``ent:<entity_id>:<category_id>`` → (entity_id, category_id), sinon None."""
    if not node_key or not node_key.startswith("ent:"):
        return None
    parts = node_key.split(":")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def _empty_tree(space_id: int, axis: str, status: str) -> Dict[str, Any]:
    return {
        "space_id": space_id,
        "axis": axis,
        "axes": axes_for_payload(),
        "root": {
            "key": "root",
            "kind": "root",
            "label": "Tous les documents",
            "icon": "ti-folders",
            "axis": axis,
            "doc_count": 0,
            "page_count": 0,
            "chunk_count": 0,
            "children": [],
        },
        "status": status,
    }


def _present_categories(session: Session, space_id: int) -> List[Dict[str, Any]]:
    """Catégories présentes dans l'espace avec axe et compteurs DISTINCT."""
    rows = session.execute(
        text(
            """
            SELECT
                dc.id AS category_id,
                dc.slug,
                dc.label,
                dc.axis,
                COUNT(DISTINCT ccr.chunk_id) AS chunk_count,
                COUNT(DISTINCT (ccr.document_id, ccr.page_no)) AS page_count,
                COUNT(DISTINCT ccr.document_id) AS document_count
            FROM chunkcategoryrelation ccr
            INNER JOIN documentcategory dc ON dc.id = ccr.category_id
            INNER JOIN document_space ds ON ds.document_id = ccr.document_id
            WHERE ds.space_id = :space_id
            GROUP BY dc.id, dc.slug, dc.label, dc.axis
            """
        ),
        {"space_id": space_id},
    ).all()

    return [
        {
            "category_id": int(category_id),
            "slug": slug,
            "label": label,
            "axis": axis or AXIS_TASK,
            "chunk_count": int(chunk_count or 0),
            "page_count": int(page_count or 0),
            "doc_count": int(document_count or 0),
        }
        for category_id, slug, label, axis, chunk_count, page_count, document_count in rows
    ]


def _count_distinct_for_categories(
    session: Session,
    space_id: int,
    category_ids: Sequence[int],
) -> Dict[str, int]:
    """Comptes DISTINCT (chunk/page/document) sur l'union d'un ensemble de catégories."""
    if not category_ids:
        return {"chunk_count": 0, "page_count": 0, "doc_count": 0}

    stmt = text(
        """
        SELECT
            COUNT(DISTINCT ccr.chunk_id) AS chunk_count,
            COUNT(DISTINCT (ccr.document_id, ccr.page_no)) AS page_count,
            COUNT(DISTINCT ccr.document_id) AS document_count
        FROM chunkcategoryrelation ccr
        INNER JOIN document_space ds ON ds.document_id = ccr.document_id
        WHERE ds.space_id = :space_id
          AND ccr.category_id IN :category_ids
        """
    ).bindparams(bindparam("category_ids", expanding=True))

    row = session.execute(
        stmt,
        {"space_id": space_id, "category_ids": list(category_ids)},
    ).first()
    if not row:
        return {"chunk_count": 0, "page_count": 0, "doc_count": 0}
    chunk_count, page_count, document_count = row
    return {
        "chunk_count": int(chunk_count or 0),
        "page_count": int(page_count or 0),
        "doc_count": int(document_count or 0),
    }


def _documents_in_space(session: Session, space_id: int) -> int:
    row = session.execute(
        text("SELECT COUNT(*) FROM document_space WHERE space_id = :space_id"),
        {"space_id": space_id},
    ).first()
    return int(row[0]) if row else 0


def _category_node(cat: Dict[str, Any], axis: str) -> Dict[str, Any]:
    return {
        "key": f"cat:{cat['category_id']}",
        "kind": "category",
        "label": cat["label"],
        "slug": cat["slug"],
        "category_id": cat["category_id"],
        "axis": axis,
        "doc_count": cat["doc_count"],
        "page_count": cat["page_count"],
        "chunk_count": cat["chunk_count"],
        "children": [],
    }


def _build_task_tree(
    session: Session,
    space_id: int,
    present: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Niveau 1 = familles curées, niveau 2 = catégories `task`."""
    task_cats = [c for c in present if c["axis"] == AXIS_TASK]
    if not task_cats:
        return []

    families: Dict[str, Dict[str, Any]] = {}
    for cat in task_cats:
        fam_slug = family_for_task_slug(cat["slug"])
        bucket = families.setdefault(fam_slug, {"slug": fam_slug, "categories": []})
        bucket["categories"].append(cat)

    nodes: List[Dict[str, Any]] = []
    for fam_slug, bucket in families.items():
        cats = sorted(bucket["categories"], key=lambda c: c["label"])
        category_ids = [c["category_id"] for c in cats]
        counts = _count_distinct_for_categories(session, space_id, category_ids)
        meta = family_meta(fam_slug)
        nodes.append(
            {
                "key": f"group:{fam_slug}",
                "kind": "group",
                "label": meta["label"],
                "icon": meta.get("icon", FALLBACK_FAMILY["icon"]),
                "axis": AXIS_TASK,
                "doc_count": counts["doc_count"],
                "page_count": counts["page_count"],
                "chunk_count": counts["chunk_count"],
                "children": [_category_node(c, AXIS_TASK) for c in cats],
            }
        )

    nodes.sort(key=lambda n: family_order_index(n["key"].split(":", 1)[1]))
    return nodes


def _build_flat_tree(
    present: List[Dict[str, Any]],
    axis: str,
) -> List[Dict[str, Any]]:
    """Axe plat : racine → catégories directement (pas de familles)."""
    known = FLAT_AXIS_LABELS.get(axis, {})
    axis_cats = [
        c for c in present if c["axis"] == axis or (c["slug"] in known)
    ]
    axis_cats.sort(key=lambda c: c["label"])
    return [_category_node(c, axis) for c in axis_cats]


def _entity_children_for_category(
    session: Session,
    space_id: int,
    category_id: int,
    slug: str,
) -> List[Dict[str, Any]]:
    """Top-N entités co-occurrentes avec une catégorie (seuillées, plafonnées).

    Croise ChunkCategoryRelation ∩ ChunkEntityRelation : déterministe (tri par
    co-occurrence puis mentions), dynamique (entités issues de l'extraction KAG).

    Catégories mappées (CATEGORY_ENTITY_TYPES) → filtrées sur le(s) type(s) curé(s)
    pour rester nettes. Catégories non mappées → fallback « tout type » (en excluant
    le fourre-tout ``other``), pour qu'aucune catégorie non vide ne reste sans entités.
    """
    max_n = settings.THEME_ENTITY_MAX_PER_CATEGORY
    min_cooc = settings.THEME_ENTITY_MIN_COOCCURRENCE
    if max_n <= 0:
        return []

    def _query(types: Optional[Tuple[str, ...]]):
        params: Dict[str, Any] = {
            "space_id": space_id,
            "category_id": category_id,
            "min_cooc": min_cooc,
            "max_n": max_n,
        }
        if types:
            type_filter = "AND ke.entity_type IN :entity_types"
        else:
            # Tout type sauf le fourre-tout ``other``.
            type_filter = "AND ke.entity_type <> 'other'"
        stmt = text(
            f"""
            SELECT ke.id, ke.name, ke.entity_type,
                   COUNT(DISTINCT cer.chunk_id) AS cooc,
                   COUNT(DISTINCT ccr.document_id) AS doc_count
            FROM chunkcategoryrelation ccr
            INNER JOIN chunkentityrelation cer ON cer.chunk_id = ccr.chunk_id
            INNER JOIN knowledgeentity ke ON ke.id = cer.entity_id
            INNER JOIN document_space ds ON ds.document_id = ccr.document_id
            WHERE ds.space_id = :space_id
              AND ccr.category_id = :category_id
              {type_filter}
            GROUP BY ke.id, ke.name, ke.entity_type
            HAVING COUNT(DISTINCT cer.chunk_id) >= :min_cooc
            ORDER BY cooc DESC, ke.mention_count DESC, ke.name
            LIMIT :max_n
            """
        )
        if types:
            stmt = stmt.bindparams(bindparam("entity_types", expanding=True))
            params["entity_types"] = list(types)
        return session.execute(stmt, params).all()

    entity_types = entity_types_for_category(slug)
    rows = _query(entity_types if entity_types else None)
    # Catégorie curée mais sans entité du type attendu → retombe sur le tout-type.
    if not rows and entity_types:
        rows = _query(None)

    children: List[Dict[str, Any]] = []
    for entity_id, name, entity_type, cooc, doc_count in rows:
        children.append(
            {
                "key": f"ent:{int(entity_id)}:{int(category_id)}",
                "kind": "entity",
                "label": name or f"Entité {entity_id}",
                "entity_type": entity_type,
                "entity_id": int(entity_id),
                "category_id": int(category_id),
                "icon": entity_icon(entity_type or ""),
                "axis": AXIS_TASK,
                "doc_count": int(doc_count or 0),
                "page_count": 0,
                "chunk_count": int(cooc or 0),
                "children": [],
            }
        )
    return children


def _attach_entities(session: Session, space_id: int, node: Dict[str, Any]) -> None:
    """Greffe récursivement les feuilles entités sous chaque nœud catégorie."""
    if node.get("kind") == "category" and node.get("category_id") and node.get("slug"):
        ents = _entity_children_for_category(
            session, space_id, int(node["category_id"]), node["slug"]
        )
        if ents:
            node["children"] = ents
        return
    for child in node.get("children", []):
        _attach_entities(session, space_id, child)


def chunk_ids_for_entity_category(
    session: Session,
    space_id: int,
    entity_id: int,
    category_id: int,
) -> List[int]:
    """Chunks de l'espace portant À LA FOIS l'entité et la catégorie."""
    rows = session.execute(
        text(
            """
            SELECT DISTINCT ccr.chunk_id
            FROM chunkcategoryrelation ccr
            INNER JOIN chunkentityrelation cer ON cer.chunk_id = ccr.chunk_id
            INNER JOIN document_space ds ON ds.document_id = ccr.document_id
            WHERE ds.space_id = :space_id
              AND ccr.category_id = :category_id
              AND cer.entity_id = :entity_id
            """
        ),
        {"space_id": space_id, "category_id": category_id, "entity_id": entity_id},
    ).all()
    return [int(r[0]) for r in rows]


def build_space_theme_tree(
    session: Session,
    space_id: int,
    axis: str = AXIS_TASK,
) -> Dict[str, Any]:
    """Construit l'arbre thématique d'un espace pour un axe de regroupement donné."""
    if not is_valid_axis(axis):
        axis = AXIS_TASK

    if not settings.KAG_ENABLED:
        return _empty_tree(space_id, axis, "disabled")

    present = _present_categories(session, space_id)
    if not present:
        return _empty_tree(space_id, axis, "empty")

    if axis == AXIS_TASK:
        children = _build_task_tree(session, space_id, present)
    else:
        children = _build_flat_tree(present, axis)

    # Comptes racine : nombre de documents réellement dans l'espace (pour « tous les
    # documents »), et couverture catégorisée pour les pages/chunks.
    all_category_ids = [c["category_id"] for c in present]
    root_counts = _count_distinct_for_categories(session, space_id, all_category_ids)
    root_doc_count = _documents_in_space(session, space_id)

    tree = _empty_tree(space_id, axis, "ok")
    tree["root"]["children"] = children
    tree["root"]["doc_count"] = root_doc_count or root_counts["doc_count"]
    tree["root"]["page_count"] = root_counts["page_count"]
    tree["root"]["chunk_count"] = root_counts["chunk_count"]
    # Phase 3 : greffe les feuilles entités sous les catégories (croisement KAG).
    if settings.KAG_ENABLED:
        for child in children:
            _attach_entities(session, space_id, child)
    tree["status"] = "ok" if children else "empty"
    return tree


def resolve_node_to_category_ids(
    session: Session,
    space_id: int,
    node_key: str,
    axis: str,
) -> Optional[List[int]]:
    """Map un node_key vers un ensemble de category_id.

    Retourne ``None`` pour la racine (= tous les documents de l'espace, sans filtre
    catégorie). Retourne une liste (éventuellement vide) sinon.
    """
    if node_key == "root":
        return None
    if node_key.startswith("cat:"):
        try:
            return [int(node_key.split(":", 1)[1])]
        except (ValueError, IndexError):
            return []
    if node_key.startswith("group:"):
        fam = node_key.split(":", 1)[1]
        present = _present_categories(session, space_id)
        return [
            c["category_id"]
            for c in present
            if c["axis"] == AXIS_TASK and family_for_task_slug(c["slug"]) == fam
        ]
    return []


def _entity_node_pages(
    session: Session,
    space_id: int,
    entity_id: int,
    category_id: int,
) -> Dict[str, Any]:
    """Pages où l'entité ∩ la catégorie co-occurrent (catégorie représentative = category_id)."""
    rows = session.execute(
        text(
            """
            SELECT
                ccr.document_id,
                d.title AS document_title,
                ccr.page_no,
                COUNT(DISTINCT ccr.chunk_id) AS chunk_count,
                BOOL_OR(
                    d.source_file_path IS NOT NULL AND d.source_file_path <> ''
                ) AS has_source_file
            FROM chunkcategoryrelation ccr
            INNER JOIN chunkentityrelation cer ON cer.chunk_id = ccr.chunk_id
            INNER JOIN document d ON d.id = ccr.document_id
            INNER JOIN document_space ds ON ds.document_id = ccr.document_id
            WHERE ds.space_id = :space_id
              AND ccr.category_id = :category_id
              AND cer.entity_id = :entity_id
            GROUP BY ccr.document_id, d.title, ccr.page_no
            ORDER BY d.title, ccr.page_no
            """
        ),
        {"space_id": space_id, "category_id": category_id, "entity_id": entity_id},
    ).all()

    pages = [
        {
            "document_id": int(document_id),
            "document_title": document_title or "",
            "page_no": int(page_no or 0),
            "chunk_count": int(chunk_count or 0),
            "has_source_file": bool(has_source_file),
            "category_id": int(category_id),
        }
        for document_id, document_title, page_no, chunk_count, has_source_file in rows
    ]
    return {
        "space_id": space_id,
        "node_key": f"ent:{entity_id}:{category_id}",
        "page_count": len(pages),
        "pages": pages,
    }


def get_space_theme_node_pages(
    session: Session,
    space_id: int,
    node_key: str,
    axis: str = AXIS_TASK,
) -> Dict[str, Any]:
    """Pages (PDF + texte) rattachées à un nœud de l'arbre.

    Agrège l'union des catégories du nœud et renvoie, par (document, page), une
    catégorie représentative (``category_id``) pour réutiliser la modale de détail de
    page existante. La racine couvre toutes les catégories présentes dans l'espace.
    """
    if not is_valid_axis(axis):
        axis = AXIS_TASK

    # Nœud entité : pages où l'entité ET la catégorie co-occurrent.
    ent = parse_entity_node_key(node_key)
    if ent is not None:
        return _entity_node_pages(session, space_id, ent[0], ent[1])

    category_ids = resolve_node_to_category_ids(session, space_id, node_key, axis)
    if category_ids is None:
        category_ids = [c["category_id"] for c in _present_categories(session, space_id)]

    if not category_ids:
        return {"space_id": space_id, "node_key": node_key, "page_count": 0, "pages": []}

    stmt = text(
        """
        SELECT
            ccr.document_id,
            d.title AS document_title,
            ccr.page_no,
            COUNT(DISTINCT ccr.chunk_id) AS chunk_count,
            BOOL_OR(
                d.source_file_path IS NOT NULL AND d.source_file_path <> ''
            ) AS has_source_file,
            MIN(ccr.category_id) AS category_id
        FROM chunkcategoryrelation ccr
        INNER JOIN document d ON d.id = ccr.document_id
        INNER JOIN document_space ds ON ds.document_id = ccr.document_id
        WHERE ds.space_id = :space_id
          AND ccr.category_id IN :category_ids
        GROUP BY ccr.document_id, d.title, ccr.page_no
        ORDER BY d.title, ccr.page_no
        """
    ).bindparams(bindparam("category_ids", expanding=True))

    rows = session.execute(
        stmt,
        {"space_id": space_id, "category_ids": list(category_ids)},
    ).all()

    pages = [
        {
            "document_id": int(document_id),
            "document_title": document_title or "",
            "page_no": int(page_no or 0),
            "chunk_count": int(chunk_count or 0),
            "has_source_file": bool(has_source_file),
            "category_id": int(category_id),
        }
        for document_id, document_title, page_no, chunk_count, has_source_file, category_id in rows
    ]

    return {
        "space_id": space_id,
        "node_key": node_key,
        "page_count": len(pages),
        "pages": pages,
    }
