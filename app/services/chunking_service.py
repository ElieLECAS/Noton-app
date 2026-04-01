from typing import Dict, List, Optional, Sequence, Tuple
import logging
import re
import threading
import uuid

from app.models.note import Note
from app.models.note_chunk import NoteChunk
from app.models.document import Document as LibraryDocument
from app.models.document_chunk import DocumentChunk
from app.config import settings
from llama_index.core.schema import Document as LlamaDocument, NodeRelationship, TextNode
from llama_index.core.node_parser import HierarchicalNodeParser

logger = logging.getLogger(__name__)

# Versions de chunking stockées dans metadata_json (traçabilité / reindex sélectif)
CHUNKING_VERSION_DOCLING_HIERARCHICAL = "docling_hierarchical_v1"
CHUNKING_VERSION_FIXED_WINDOW = "fixed_window_v1"
CHUNKING_VERSION_MARKDOWN_H2 = "markdown_h2_sections_v1"
CHUNKING_VERSION_ADAPTIVE = "adaptive_window_v1"

# Chunking adaptatif (fallback document) — approx. tokens ≈ chars/4
ADAPTIVE_CHUNK_CHARS_PROCEDURE = (600, 1200)  # ~150–300 tokens
ADAPTIVE_OVERLAP_FRAC_PROCEDURE = 0.25
ADAPTIVE_CHUNK_CHARS_NORMATIVE = (1600, 2400)  # ~400–600 tokens
ADAPTIVE_OVERLAP_FRAC_NORMATIVE = 0.10
ADAPTIVE_CHUNK_CHARS_DESCRIPTION = (3200, 4800)  # ~800+ tokens
ADAPTIVE_OVERLAP_FRAC_DESCRIPTION = 0.15

_docling_node_parser = None
_docling_node_parser_lock = threading.Lock()

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
    llama_doc = LlamaDocument(
        text=full_text,
        metadata={
            "note_id": note.id,
            "project_id": note.project_id,
            "user_id": note.user_id,
            "note_title": note.title or "",
        },
    )

    nodes = parser.get_nodes_from_documents([llama_doc])
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

    doc_metadata = dict(llama_doc.metadata or {})
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

def _get_docling_node_parser():
    """
    Retourne un DoclingNodeParser optimisé pour les tableaux (singleton, thread-safe).

    Utilise un HierarchicalChunker avec MarkdownTableSerializer au lieu du
    TripletTableSerializer par défaut : les tableaux sont sérialisés en grille
    Markdown (| Colonne A | Colonne B |) ce qui réduit les confusions
    colonnes/lignes et améliore la précision des valeurs numériques pour le LLM.
    """
    global _docling_node_parser
    if _docling_node_parser is not None:
        return _docling_node_parser
    with _docling_node_parser_lock:
        if _docling_node_parser is not None:
            return _docling_node_parser
        from llama_index.node_parser.docling import DoclingNodeParser

        try:
            from docling_core.transforms.chunker import HierarchicalChunker
            from docling_core.transforms.chunker.hierarchical_chunker import (
                ChunkingDocSerializer,
                ChunkingSerializerProvider,
            )
            from docling_core.transforms.serializer.markdown import MarkdownTableSerializer

            class MDTableSerializerProvider(ChunkingSerializerProvider):
                """Provider qui sérialise les tableaux en Markdown (grille avec en-têtes explicites)."""

                def get_serializer(self, doc):
                    return ChunkingDocSerializer(
                        doc=doc,
                        table_serializer=MarkdownTableSerializer(),
                    )

            chunker = HierarchicalChunker(serializer_provider=MDTableSerializerProvider())
            _docling_node_parser = DoclingNodeParser(chunker=chunker)
        except (ImportError, AttributeError) as e:
            logger.warning(
                "MarkdownTableSerializer non disponible (%s) — repli sur le serializer par défaut "
                "(TripletTableSerializer). Les tableaux seront moins lisibles et leur atomicité "
                "n'est plus garantie. Vérifier l'installation de docling-core.",
                e,
            )
            _docling_node_parser = DoclingNodeParser()
        return _docling_node_parser


def _get_parent_heading_label(headings: list) -> str:
    """
    Construit le libellé complet de section à partir de tous les niveaux de headings.

    Chaque chunk est ainsi étiqueté par son sujet (ex. "1.3.1 Montage", "2 Drainage").
    Utilisé comme clé de regroupement et comme parent_heading dans les métadonnées.
    """
    if not headings or not isinstance(headings, list):
        return "__no_heading__"
    parts = [str(h).strip() for h in headings if h is not None and str(h).strip()]
    if not parts:
        return "__no_heading__"
    return " ".join(parts)


def _extract_caption_from_metadata(meta: dict) -> Optional[str]:
    """
    Extrait la légende (figure/table) des métadonnées Docling.

    Docling peut exposer la légende dans caption, figure_title, caption_text,
    ou dans les doc_items. On normalise en un seul texte.
    """
    if not meta:
        return None
    for key in ("caption", "figure_title", "caption_text", "image_caption"):
        val = meta.get(key)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
    doc_items = meta.get("doc_items") or meta.get("doc_items_refs")
    if isinstance(doc_items, list):
        for it in doc_items:
            if isinstance(it, dict) and it.get("caption"):
                cap = it["caption"]
                if isinstance(cap, str) and cap.strip():
                    return cap.strip()
            if isinstance(it, dict) and it.get("label") in ("picture", "table"):
                cap = it.get("caption") or it.get("title")
                if isinstance(cap, str) and cap.strip():
                    return cap.strip()
    return None


def _is_picture_or_table_chunk(meta: dict) -> bool:
    """Détermine si le chunk provient d'un bloc picture ou table (pour fusion légende)."""
    if not meta:
        return False
    label = meta.get("label")
    if label in ("picture", "table", "figure"):
        return True
    doc_items = meta.get("doc_items") or meta.get("doc_items_refs")
    if isinstance(doc_items, list):
        for it in doc_items:
            if isinstance(it, dict) and it.get("label") in ("picture", "table", "figure"):
                return True
    return False


def _is_markdown_table_text(text: str) -> bool:
    """Heuristique : grille Markdown présente (| … |)."""
    if not text or not text.strip():
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return lines[0].startswith("|") and "|" in lines[0]


def _parse_markdown_table(text: str) -> Optional[Tuple[List[str], List[List[str]]]]:
    """
    Parse un tableau Markdown simple en (en-têtes, lignes de données).
    Ignore la ligne séparatrice |---|---|.
    """
    if not _is_markdown_table_text(text):
        return None
    table_lines = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and ln.strip().startswith("|")
    ]
    if len(table_lines) < 2:
        return None

    def _split_row(line: str) -> List[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    headers = _split_row(table_lines[0])
    if not headers:
        return None

    idx = 1
    if len(table_lines) > 1:
        sep = table_lines[1].replace("|", "").replace("-", "").replace(":", "").strip()
        if not sep:
            idx = 2

    data_rows: List[List[str]] = []
    for i in range(idx, len(table_lines)):
        row = _split_row(table_lines[i])
        if not any(row):
            continue
        # Aligner longueur sur les en-têtes
        while len(row) < len(headers):
            row.append("")
        data_rows.append(row[: len(headers)])
    if not data_rows:
        return None
    return headers, data_rows


def _table_row_chunk_text(
    *,
    headers: List[str],
    cells: List[str],
    parent_heading: str,
    caption: Optional[str],
    page_no: Optional[int],
    table_id: str,
    row_index: int,
) -> str:
    """Phrase autonome par ligne avec réinjection des en-têtes de colonnes."""
    parts = []
    for h, c in zip(headers, cells):
        h = (h or "").strip()
        c = (c or "").strip()
        if h and c:
            parts.append(f"{h}: {c}")
        elif c:
            parts.append(c)
    if not parts:
        return ""
    ctx = []
    if caption:
        ctx.append(f"Tableau ({caption})")
    if page_no is not None:
        ctx.append(f"p.{page_no}")
    prefix = " — ".join(ctx) if ctx else "Tableau"
    if parent_heading:
        prefix = f"[{parent_heading}] {prefix}"
    body = " ; ".join(parts)
    return (
        f"{prefix}. Pour cette ligne du tableau (id={table_id}, ligne {row_index + 1}): {body}"
    )


def _should_expand_table_leaf(meta: dict, raw_content: str) -> bool:
    if meta.get("label") == "table":
        return True
    return _is_markdown_table_text(raw_content)


def _list_parent_title_from_headings(headings: list) -> Optional[str]:
    """Dernier titre de la hiérarchie pour préfixer les items de liste."""
    if not headings or not isinstance(headings, list):
        return None
    for h in reversed(headings):
        if h is not None and str(h).strip():
            return str(h).strip()
    return None


_NORMATIVE_PAT = re.compile(
    r"\b(DTU|NF\s|EN\s|CE\b|norme|article|décret|arrêté|conformément|ISO\s)\b",
    re.IGNORECASE,
)
_PROCEDURE_PAT = re.compile(
    r"(?im)^\s*\d+[\.\)]\s+|^\s*[-•]\s+|\b(visser|régler|monter|poser|contrôler|vérifier|serrer|ajuster)\b",
)


def _detect_content_type(text: str) -> str:
    """
    Étiquette grossière pour chunking adaptatif / métadonnées : procédure, normative, description.
    """
    if not text or not text.strip():
        return "description"
    t = text[:8000]
    if _NORMATIVE_PAT.search(t):
        return "normative"
    if _PROCEDURE_PAT.search(t):
        return "procedure"
    return "description"


def resolve_adaptive_chunk_params(content_type: str) -> Tuple[int, int]:
    """
    Retourne (chunk_size_chars, overlap_chars) selon le type de contenu.
    """
    if content_type == "procedure":
        lo, hi = ADAPTIVE_CHUNK_CHARS_PROCEDURE
        mid = (lo + hi) // 2
        return mid, max(80, int(mid * ADAPTIVE_OVERLAP_FRAC_PROCEDURE))
    if content_type == "normative":
        lo, hi = ADAPTIVE_CHUNK_CHARS_NORMATIVE
        mid = (lo + hi) // 2
        return mid, max(120, int(mid * ADAPTIVE_OVERLAP_FRAC_NORMATIVE))
    lo, hi = ADAPTIVE_CHUNK_CHARS_DESCRIPTION
    mid = (lo + hi) // 2
    return mid, max(200, int(mid * ADAPTIVE_OVERLAP_FRAC_DESCRIPTION))


def _page_range_from_docling_leaves(group_leaves: List[TextNode]) -> Tuple[Optional[int], Optional[int]]:
    """Min / max page_no issus des métadonnées Docling des feuilles d'une section."""
    pages: List[int] = []
    for leaf_node in group_leaves:
        m = dict(leaf_node.metadata or {})
        p = m.get("page_no")
        if p is None:
            continue
        try:
            pages.append(int(p))
        except (TypeError, ValueError):
            continue
    if not pages:
        return None, None
    return min(pages), max(pages)


def _build_docling_hierarchical_specs(
    doc_metadata_base: dict,
    leaf_nodes: List[TextNode],
) -> List[dict]:
    """
    Groupe les nœuds Docling par section et produit une liste de specs agnostiques
    (NoteChunk ou DocumentChunk). Même logique que l'ancien chunk_note_from_docling_docs.
    """
    base_meta = dict(doc_metadata_base)
    base_meta["chunking_version"] = CHUNKING_VERSION_DOCLING_HIERARCHICAL

    groups: Dict[str, List[TextNode]] = {}
    group_order: List[str] = []
    section_captions: Dict[str, List[str]] = {}

    for node in leaf_nodes:
        headings = (node.metadata or {}).get("headings") or []
        key = _get_parent_heading_label(headings)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(node)
        cap = _extract_caption_from_metadata(dict(node.metadata or {}))
        if cap:
            captions_list = section_captions.setdefault(key, [])
            if cap not in captions_list:
                captions_list.append(cap)

    specs: List[dict] = []
    chunk_index = 0

    for heading_key in group_order:
        group_leaves = groups[heading_key]
        parent_heading_display = (
            heading_key if heading_key != "__no_heading__" else ""
        )
        section_anchors = section_captions.get(heading_key) or []

        parent_text = "\n\n".join(
            (n.get_content() or "").strip() for n in group_leaves
        ).strip()
        if not parent_text:
            continue

        parent_node_id = str(uuid.uuid4())
        parent_metadata = dict(base_meta)
        parent_metadata.update(
            {
                "node_id": parent_node_id,
                "parent_node_id": None,
                "hierarchy_level": 0,
                "is_leaf": "false",
                "heading": parent_heading_display,
                "parent_heading": parent_heading_display,
            }
        )
        if section_anchors:
            parent_metadata["image_anchor"] = " ; ".join(section_anchors)
            parent_metadata["figure_title"] = section_anchors[0]
            parent_metadata["contains_image"] = True

        page_start, page_end = _page_range_from_docling_leaves(group_leaves)
        if page_start is not None:
            parent_metadata["page_start"] = page_start
            parent_metadata["page_end"] = page_end
            parent_metadata["page_no"] = page_start

        specs.append(
            {
                "chunk_index": chunk_index,
                "is_leaf": False,
                "content": parent_text,
                "text": parent_text,
                "start_char": 0,
                "end_char": len(parent_text),
                "node_id": parent_node_id,
                "parent_node_id": None,
                "hierarchy_level": 0,
                "metadata_json": parent_metadata,
            }
        )
        chunk_index += 1

        for leaf_node in group_leaves:
            raw_content = (leaf_node.get_content() or "").strip()
            if not raw_content:
                continue

            leaf_node_id_base = leaf_node.node_id or str(uuid.uuid4())
            docling_meta = dict(leaf_node.metadata or {})
            headings = (leaf_node.metadata or {}).get("headings") or []

            caption = _extract_caption_from_metadata(docling_meta)
            if _is_picture_or_table_chunk(docling_meta) and caption:
                raw_content = f"{raw_content}\n\n{caption}".strip()

            if docling_meta.get("label") == "list":
                list_title = _list_parent_title_from_headings(headings) or parent_heading_display
                if list_title:
                    raw_content = f"[Liste: {list_title}]\n{raw_content}"

            table_src = raw_content.split("\n\n")[0].strip()
            parsed_table = None
            if _should_expand_table_leaf(docling_meta, table_src):
                parsed_table = _parse_markdown_table(table_src)
            if parsed_table is None and _should_expand_table_leaf(docling_meta, raw_content):
                parsed_table = _parse_markdown_table(raw_content)

            if parsed_table:
                headers, data_rows = parsed_table
                table_id = str(uuid.uuid4())
                page_no = docling_meta.get("page_no")
                cap = caption or _extract_caption_from_metadata(docling_meta)

                for ri, cells in enumerate(data_rows):
                    row_text = _table_row_chunk_text(
                        headers=headers,
                        cells=cells,
                        parent_heading=parent_heading_display,
                        caption=cap,
                        page_no=page_no,
                        table_id=table_id,
                        row_index=ri,
                    )
                    if not row_text.strip():
                        continue

                    leaf_rid = str(uuid.uuid4())
                    leaf_metadata = dict(base_meta)
                    leaf_metadata.update(docling_meta)
                    leaf_metadata["parent_heading"] = parent_heading_display
                    leaf_metadata["heading"] = parent_heading_display
                    leaf_metadata["content_type"] = "table_row"
                    leaf_metadata["table_id"] = table_id
                    leaf_metadata["row_index"] = ri
                    leaf_metadata["column_headers"] = headers
                    leaf_metadata["raw_content"] = raw_content
                    if section_anchors:
                        leaf_metadata["image_anchor"] = " ; ".join(section_anchors)
                        leaf_metadata["figure_title"] = section_anchors[0]
                    elif cap:
                        leaf_metadata["image_anchor"] = cap
                        leaf_metadata["figure_title"] = cap
                    leaf_metadata.update(
                        {
                            "node_id": leaf_rid,
                            "parent_node_id": parent_node_id,
                            "hierarchy_level": 1,
                            "is_leaf": "true",
                        }
                    )
                    if page_no is not None:
                        leaf_metadata["page_no"] = page_no
                    leaf_metadata["contains_image"] = True

                    content = (
                        f"[{parent_heading_display}] {row_text}"
                        if parent_heading_display
                        else row_text
                    )

                    specs.append(
                        {
                            "chunk_index": chunk_index,
                            "is_leaf": True,
                            "content": content,
                            "text": content,
                            "start_char": 0,
                            "end_char": len(content),
                            "node_id": leaf_rid,
                            "parent_node_id": parent_node_id,
                            "hierarchy_level": 1,
                            "metadata_json": leaf_metadata,
                        }
                    )
                    chunk_index += 1
                continue

            leaf_node_id = leaf_node_id_base

            if parent_heading_display:
                content = f"[{parent_heading_display}] {raw_content}"
            else:
                content = raw_content

            leaf_metadata = dict(base_meta)
            leaf_metadata.update(docling_meta)
            leaf_metadata["parent_heading"] = parent_heading_display
            leaf_metadata["heading"] = parent_heading_display
            if parent_heading_display:
                leaf_metadata["raw_content"] = raw_content
            leaf_metadata["content_type"] = _detect_content_type(raw_content)
            if section_anchors:
                leaf_metadata["image_anchor"] = " ; ".join(section_anchors)
                leaf_metadata["figure_title"] = section_anchors[0]
            elif caption:
                leaf_metadata["image_anchor"] = caption
                leaf_metadata["figure_title"] = caption
            leaf_metadata.update(
                {
                    "node_id": leaf_node_id,
                    "parent_node_id": parent_node_id,
                    "hierarchy_level": 1,
                    "is_leaf": "true",
                }
            )
            if docling_meta.get("page_no") is not None:
                leaf_metadata["page_no"] = docling_meta["page_no"]
            if section_anchors or _is_picture_or_table_chunk(docling_meta):
                leaf_metadata["contains_image"] = True

            specs.append(
                {
                    "chunk_index": chunk_index,
                    "is_leaf": True,
                    "content": content,
                    "text": content,
                    "start_char": 0,
                    "end_char": len(content),
                    "node_id": leaf_node_id,
                    "parent_node_id": parent_node_id,
                    "hierarchy_level": 1,
                    "metadata_json": leaf_metadata,
                }
            )
            chunk_index += 1

    return specs


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
        note       : La note cible (déjà enregistrée en base)
        llama_docs : Liste de LlamaIndex Document dont le .text contient le
                     JSON sérialisé d'un DoclingDocument (model_dump_json)

    Returns:
        Liste de NoteChunk (leaves + parents) prête à être sauvegardée
    """
    try:
        node_parser = _get_docling_node_parser()
    except ImportError:
        logger.warning(
            "llama-index-node-parser-docling non installé — "
            "fallback sur HierarchicalNodeParser pour la note %s",
            note.id,
        )
        return chunk_note(note)

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

    specs = _build_docling_hierarchical_specs(doc_metadata_base, leaf_nodes)
    chunks: List[NoteChunk] = []
    for spec in specs:
        meta = spec["metadata_json"]
        chunks.append(
            NoteChunk(
                note_id=note.id,
                chunk_index=spec["chunk_index"],
                content=spec["content"],
                text=spec["text"],
                start_char=spec["start_char"],
                end_char=spec["end_char"],
                node_id=spec["node_id"],
                parent_node_id=spec["parent_node_id"],
                is_leaf=spec["is_leaf"],
                hierarchy_level=spec["hierarchy_level"],
                metadata_json=meta,
                metadata_=meta,
            )
        )

    leaf_count = sum(1 for c in chunks if c.is_leaf)
    parent_count = sum(1 for c in chunks if not c.is_leaf)
    logger.info(
        "Chunking sémantique (DoclingNodeParser) note=%s : "
        "%d chunks total (%d leaves, %d parents, %d sections)",
        note.id,
        len(chunks),
        leaf_count,
        parent_count,
        parent_count,
    )
    return chunks


def chunk_document_from_docling_docs(
    document: LibraryDocument,
    llama_docs: Sequence,
) -> List[DocumentChunk]:
    """
    Découpe un Document bibliothèque via DoclingNodeParser (même logique que les notes).

    Retourne des DocumentChunk hiérarchiques (parents + leaves) avec métadonnées Docling.
    En cas d'échec ou liste vide, l'appelant doit retomber sur create_chunks_for_document.
    """
    try:
        node_parser = _get_docling_node_parser()
    except ImportError:
        logger.warning(
            "llama-index-node-parser-docling non installé — "
            "chunk_document_from_docling_docs indisponible pour document %s",
            document.id,
        )
        return []

    try:
        leaf_nodes: List[TextNode] = node_parser.get_nodes_from_documents(
            list(llama_docs)
        )
    except Exception as exc:
        logger.warning(
            "DoclingNodeParser a échoué pour le document %s (%s)",
            document.id,
            exc,
        )
        return []

    if not leaf_nodes:
        logger.warning(
            "DoclingNodeParser n'a produit aucun nœud pour le document %s",
            document.id,
        )
        return []

    doc_metadata_base = {
        "document_id": document.id,
        "library_id": document.library_id,
        "user_id": document.user_id,
        "document_title": document.title or "",
    }

    specs = _build_docling_hierarchical_specs(doc_metadata_base, leaf_nodes)
    chunks: List[DocumentChunk] = []
    for spec in specs:
        meta = spec["metadata_json"]
        chunks.append(
            DocumentChunk(
                document_id=document.id,
                chunk_index=spec["chunk_index"],
                content=spec["content"],
                text=spec["text"],
                start_char=spec["start_char"],
                end_char=spec["end_char"],
                node_id=spec["node_id"],
                parent_node_id=spec["parent_node_id"],
                is_leaf=spec["is_leaf"],
                hierarchy_level=spec["hierarchy_level"],
                metadata_json=meta,
                metadata_=meta,
            )
        )

    leaf_count = sum(1 for c in chunks if c.is_leaf)
    parent_count = sum(1 for c in chunks if not c.is_leaf)
    logger.info(
        "Chunking sémantique (DoclingNodeParser) document=%s : "
        "%d chunks total (%d leaves, %d parents)",
        document.id,
        len(chunks),
        leaf_count,
        parent_count,
    )
    return chunks
