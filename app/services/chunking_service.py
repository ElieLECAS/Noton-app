from typing import Dict, List, Optional
from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.config import settings
import logging
from llama_index.core.schema import Document, NodeRelationship
from llama_index.core.node_parser import HierarchicalNodeParser

logger = logging.getLogger(__name__)

DEFAULT_HIERARCHICAL_CHUNK_SIZES = [3072, 1024, 384]


def _resolve_chunk_sizes(text_length: int) -> List[int]:
    """Calcule les tailles hiérarchiques avec fallback sécurisé pour gros documents."""
    configured = settings.HIERARCHICAL_CHUNK_SIZES or DEFAULT_HIERARCHICAL_CHUNK_SIZES
    chunk_sizes = sorted({int(size) for size in configured if int(size) > 0}, reverse=True)
    if not chunk_sizes:
        chunk_sizes = DEFAULT_HIERARCHICAL_CHUNK_SIZES

    # Pour les très gros documents, agrandir légèrement pour réduire le nombre total de chunks.
    if text_length >= 200_000:
        largest = max(chunk_sizes[0], 4096)
        medium = max(chunk_sizes[min(1, len(chunk_sizes) - 1)], 1536)
        smallest = max(chunk_sizes[min(2, len(chunk_sizes) - 1)], 512)
        chunk_sizes = [largest, medium, smallest]

    return chunk_sizes


def chunk_note(note: Note) -> List[NoteChunk]:
    """
    Découper une note en nœuds hiérarchiques (parents/enfants) via LlamaIndex.
    
    Args:
        note: La note à découper
        
    Returns:
        Liste de NoteChunk
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

    # Tri stable pour conserver un ordre cohérent en base
    nodes_sorted = sorted(
        nodes,
        key=lambda n: (
            level_map.get(n.node_id, 0),
            int((n.metadata or {}).get("start_char_idx", 0)),
            n.node_id,
        ),
    )

    chunks: List[NoteChunk] = []
    # Récupérer les métadonnées du document original pour les propager aux chunks
    doc_metadata = dict(document.metadata or {})
    
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
        end_char = int(metadata.get("end_char_idx", start_char + len(content)) or (start_char + len(content)))

        # Propager les métadonnées du document (project_id, user_id, note_id, note_title)
        # qui ne sont pas automatiquement propagées par le parser hiérarchique
        metadata.update(doc_metadata)
        metadata.update(
            {
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "hierarchy_level": hierarchy_level,
                "is_leaf": "true" if is_leaf else "false",  # Chaîne pour compatibilité avec MetadataFilter
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
        "Chunking hiérarchique note=%s nodes=%s chunk_sizes=%s",
        note.id,
        len(chunks),
        chunk_sizes,
    )
    return chunks


def _build_full_text(note: Note) -> str:
    if note.content and note.content.strip():
        return f"{note.title}\n\n{note.content}" if note.title else note.content
    return note.title or ""


def _build_parent_map(nodes: List) -> Dict[str, Optional[str]]:
    parent_map: Dict[str, Optional[str]] = {}
    for node in nodes:
        relationships = getattr(node, "relationships", {}) or {}
        parent_rel = relationships.get(NodeRelationship.PARENT)
        parent_map[node.node_id] = getattr(parent_rel, "node_id", None) if parent_rel else None
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


def _detect_leaf_ids(parent_map: Dict[str, Optional[str]]) -> set[str]:
    parent_ids = {parent_id for parent_id in parent_map.values() if parent_id}
    return {node_id for node_id in parent_map.keys() if node_id not in parent_ids}

