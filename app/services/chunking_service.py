from typing import Dict, List, Optional, Sequence
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.config import settings
import logging
import uuid
from llama_index.core.schema import Document, NodeRelationship, TextNode
from llama_index.core.node_parser import HierarchicalNodeParser

logger = logging.getLogger(__name__)

DEFAULT_HIERARCHICAL_CHUNK_SIZES = [3072, 1024, 384]


# ---------------------------------------------------------------------------
# Helpers partagés
# ---------------------------------------------------------------------------

def _resolve_chunk_sizes(text_length: int) -> List[int]:
    """Calcule les tailles hiérarchiques avec fallback sécurisé pour gros documents."""
    configured = settings.HIERARCHICAL_CHUNK_SIZES or DEFAULT_HIERARCHICAL_CHUNK_SIZES
    chunk_sizes = sorted({int(size) for size in configured if int(size) > 0}, reverse=True)
    if not chunk_sizes:
        chunk_sizes = DEFAULT_HIERARCHICAL_CHUNK_SIZES

    if text_length >= 200_000:
        largest = max(chunk_sizes[0], 4096)
        medium = max(chunk_sizes[min(1, len(chunk_sizes) - 1)], 1536)
        smallest = max(chunk_sizes[min(2, len(chunk_sizes) - 1)], 512)
        chunk_sizes = [largest, medium, smallest]

    return chunk_sizes


def _build_full_text(note: Note) -> str:
    if note.content and note.content.strip():
        return f"{note.title}\n\n{note.content}" if note.title else note.content
    return note.title or ""


def _build_parent_map(nodes: List) -> Dict[str, Optional[str]]:
    parent_map: Dict[str, Optional[str]] = {}
    for node in nodes:
        relationships = getattr(node, "relationships", {}) or {}
        parent_rel = relationships.get(NodeRelationship.PARENT)
        parent_map[node.node_id] = (
            getattr(parent_rel, "node_id", None) if parent_rel else None
        )
    return parent_map


def _build_level_map(parent_map: Dict[str, Optional[str]]) -> Dict[str, int]:
    level_map: Dict[str, int] = {}

    def _compute_level(node_id: str) -> int:
        if node_id in level_map:
            return level_map[node_id]
        parent_id = parent_map.get(node_id)
        if not parent_id:
            level_map[node_id] = 0
        else:
            level_map[node_id] = _compute_level(parent_id) + 1
        return level_map[node_id]

    for node_id in parent_map:
        _compute_level(node_id)
    return level_map


def _detect_leaf_ids(parent_map: Dict[str, Optional[str]]) -> set:
    parent_ids = {parent_id for parent_id in parent_map.values() if parent_id}
    return {node_id for node_id in parent_map.keys() if node_id not in parent_ids}


# ---------------------------------------------------------------------------
# Stratégie 1 : notes manuelles — HierarchicalNodeParser (taille de texte)
# ---------------------------------------------------------------------------

def chunk_note(note: Note) -> List[NoteChunk]:
    """
    Découper une note en nœuds hiérarchiques (parents/enfants) via LlamaIndex.
    Utilisé pour les notes créées manuellement (pas issues de Docling).
    """
    full_text = _build_full_text(note)
    if not full_text.strip():
        full_text = note.title or "Note sans titre"

    chunk_sizes = _resolve_chunk_sizes(len(full_text))
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    document = Document(
        text=full_text,
        metadata={
            "note_id": note.id,
            "project_id": note.project_id,
            "user_id": note.user_id,
            "note_title": note.title or "",
        },
    )

    nodes = parser.get_nodes_from_documents([document])
    if not nodes:
        return []

    parent_map = _build_parent_map(nodes)
    level_map = _build_level_map(parent_map)
    leaf_ids = _detect_leaf_ids(parent_map)

    nodes_sorted = sorted(
        nodes,
        key=lambda n: (
            level_map.get(n.node_id, 0),
            int((n.metadata or {}).get("start_char_idx", 0)),
            n.node_id,
        ),
    )

    doc_metadata = dict(document.metadata or {})
    chunks: List[NoteChunk] = []

    for idx, node in enumerate(nodes_sorted):
        metadata = dict(node.metadata or {})
        content = (node.get_content() or "").strip()
        if not content:
            continue

        node_id = node.node_id
        parent_node_id = parent_map.get(node_id)
        hierarchy_level = level_map.get(node_id, 0)
        is_leaf = node_id in leaf_ids
        start_char = int(metadata.get("start_char_idx", 0) or 0)
        end_char = int(
            metadata.get("end_char_idx", start_char + len(content))
            or (start_char + len(content))
        )

        metadata.update(doc_metadata)
        metadata.update(
            {
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "hierarchy_level": hierarchy_level,
                "is_leaf": "true" if is_leaf else "false",
            }
        )

        chunks.append(
            NoteChunk(
                note_id=note.id,
                chunk_index=idx,
                content=content,
                text=content,
                start_char=start_char,
                end_char=end_char,
                node_id=node_id,
                parent_node_id=parent_node_id,
                is_leaf=is_leaf,
                hierarchy_level=hierarchy_level,
                metadata_json=metadata,
                metadata_=metadata,
            )
        )

    logger.info(
        "Chunking hiérarchique (HierarchicalNodeParser) note=%s nodes=%s chunk_sizes=%s",
        note.id,
        len(chunks),
        chunk_sizes,
    )
    return chunks


# ---------------------------------------------------------------------------
# Stratégie 2 : documents importés — DoclingNodeParser (structure sémantique)
# ---------------------------------------------------------------------------

def _get_top_heading(headings: list) -> str:
    """Retourne le premier heading (section de niveau 1) ou une clé générique."""
    if headings and isinstance(headings, list) and headings[0]:
        return str(headings[0])
    return "__no_heading__"


def chunk_note_from_docling_docs(
    note: Note,
    llama_docs: Sequence,
) -> List[NoteChunk]:
    """
    Découper un document importé via Docling en NoteChunks sémantiques.

    Utilise DoclingNodeParser qui respecte la structure logique du document
    (paragraphes, tableaux, listes, sections) — contrairement à
    HierarchicalNodeParser qui découpe uniquement par taille de texte.

    Hiérarchie produite :
    - Leaves  : blocs sémantiques individuels (paragraphe, tableau, liste…)
    - Parents : regroupement de blocs consécutifs partageant le même heading
      de niveau 1 (ex : tous les paragraphes de la section « 2 Résultats »)

    Les métadonnées Docling riches (page_no, bbox, headings) sont propagées
    dans metadata_json de chaque chunk.

    Args:
        note     : La note cible (déjà enregistrée en base)
        llama_docs : Liste de LlamaIndex Document dont le .text contient le
                     JSON sérialisé d'un DoclingDocument (model_dump_json)

    Returns:
        Liste de NoteChunk (leaves + parents) prête à être sauvegardée
    """
    try:
        from llama_index.node_parser.docling import DoclingNodeParser
    except ImportError:
        logger.warning(
            "llama-index-node-parser-docling non installé — "
            "fallback sur HierarchicalNodeParser pour la note %s",
            note.id,
        )
        return chunk_note(note)

    node_parser = DoclingNodeParser()
    try:
        leaf_nodes: List[TextNode] = node_parser.get_nodes_from_documents(
            list(llama_docs)
        )
    except Exception as exc:
        logger.warning(
            "DoclingNodeParser a échoué pour la note %s (%s) — "
            "fallback sur HierarchicalNodeParser",
            note.id,
            exc,
        )
        return chunk_note(note)

    if not leaf_nodes:
        logger.warning(
            "DoclingNodeParser n'a produit aucun nœud pour la note %s — "
            "fallback sur HierarchicalNodeParser",
            note.id,
        )
        return chunk_note(note)

    doc_metadata_base = {
        "note_id": note.id,
        "project_id": note.project_id,
        "user_id": note.user_id,
        "note_title": note.title or "",
    }

    # ------------------------------------------------------------------
    # Grouper les leaves par heading de niveau 1 pour créer les parents
    # ------------------------------------------------------------------
    # Structure : { top_heading: [leaf_node, ...] }
    groups: Dict[str, List[TextNode]] = {}
    group_order: List[str] = []  # ordre d'apparition des headings

    for node in leaf_nodes:
        headings = (node.metadata or {}).get("headings") or []
        key = _get_top_heading(headings)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(node)

    # ------------------------------------------------------------------
    # Construire les NoteChunk leaves + parents
    # ------------------------------------------------------------------
    chunks: List[NoteChunk] = []
    chunk_index = 0

    for heading_key in group_order:
        group_leaves = groups[heading_key]

        # Créer le nœud parent pour ce groupe
        parent_text = "\n\n".join(
            (n.get_content() or "").strip() for n in group_leaves
        ).strip()
        if not parent_text:
            continue

        parent_node_id = str(uuid.uuid4())
        parent_metadata = dict(doc_metadata_base)
        parent_metadata.update(
            {
                "node_id": parent_node_id,
                "parent_node_id": None,
                "hierarchy_level": 0,
                "is_leaf": "false",
                "heading": heading_key if heading_key != "__no_heading__" else "",
            }
        )

        chunks.append(
            NoteChunk(
                note_id=note.id,
                chunk_index=chunk_index,
                content=parent_text,
                text=parent_text,
                start_char=0,
                end_char=len(parent_text),
                node_id=parent_node_id,
                parent_node_id=None,
                is_leaf=False,
                hierarchy_level=0,
                metadata_json=parent_metadata,
                metadata_=parent_metadata,
            )
        )
        chunk_index += 1

        # Créer les NoteChunk feuilles pour chaque bloc sémantique
        for leaf_node in group_leaves:
            content = (leaf_node.get_content() or "").strip()
            if not content:
                continue

            leaf_node_id = leaf_node.node_id or str(uuid.uuid4())
            docling_meta = dict(leaf_node.metadata or {})

            leaf_metadata = dict(doc_metadata_base)
            # Propager les métadonnées Docling riches (page_no, bbox, headings, doc_items…)
            leaf_metadata.update(docling_meta)
            leaf_metadata.update(
                {
                    "node_id": leaf_node_id,
                    "parent_node_id": parent_node_id,
                    "hierarchy_level": 1,
                    "is_leaf": "true",
                }
            )

            chunks.append(
                NoteChunk(
                    note_id=note.id,
                    chunk_index=chunk_index,
                    content=content,
                    text=content,
                    start_char=0,
                    end_char=len(content),
                    node_id=leaf_node_id,
                    parent_node_id=parent_node_id,
                    is_leaf=True,
                    hierarchy_level=1,
                    metadata_json=leaf_metadata,
                    metadata_=leaf_metadata,
                )
            )
            chunk_index += 1

    leaf_count = sum(1 for c in chunks if c.is_leaf)
    parent_count = sum(1 for c in chunks if not c.is_leaf)
    logger.info(
        "Chunking sémantique (DoclingNodeParser) note=%s : "
        "%d chunks total (%d leaves, %d parents, %d sections)",
        note.id,
        len(chunks),
        leaf_count,
        parent_count,
        len(group_order),
    )
    return chunks
