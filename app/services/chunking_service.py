from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import logging
import re
import threading
import traceback
import unicodedata
import uuid

from app.models.document import Document as LibraryDocument
from app.models.document_chunk import DocumentChunk
from app.config import settings
from app.library_document_logging import get_library_document_logger
from llama_index.core.schema import Document as LlamaDocument, NodeRelationship, TextNode
from llama_index.core.node_parser import HierarchicalNodeParser, MarkdownNodeParser

logger = logging.getLogger(__name__)

# Versions de chunking stockées dans metadata_json (traçabilité / reindex sélectif)
CHUNKING_VERSION_DOCLING_HIERARCHICAL = "docling_hierarchical_v1"
CHUNKING_VERSION_DOCLING_HIERARCHICAL_V2 = "docling_hierarchical_v2"
CHUNKING_VERSION_FIXED_WINDOW = "fixed_window_v1"
CHUNKING_VERSION_MARKDOWN_H2 = "markdown_h2_sections_v1"
CHUNKING_VERSION_ADAPTIVE = "adaptive_window_v1"
CHUNKING_VERSION_MARKDOWN_HIERARCHICAL = "markdown_hierarchical_v1"
CHUNKING_VERSION_MARKDOWN_HIERARCHICAL_V2 = "markdown_hierarchical_v2"
CHUNKING_VERSION_MARKDOWN_STRUCTURED = "markdown_structured_v1"
CHUNKING_VERSION_MARKDOWN_STRUCTURED_V2 = "markdown_structured_v2"
CHUNKING_VERSION_PYMUPDF4LLM_CLEAN = "pymupdf4llm_semantic_v2"

_H2_SPLIT_RE = re.compile(r"(?m)^(##\s+.+)$")
_NUMBERED_STEP_RE = re.compile(r"(?m)^(?:\*\*)?(\d+)\.\s+")

HEADER_PATH_SEP = " > "

# Tailles par défaut pour l'ingestion OCR (parents ~1024 tokens, feuilles ~256)
DEFAULT_OCR_HIERARCHICAL_CHUNK_SIZES = [1024, 256]

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


@dataclass
class TableParseResult:
    """Résultat enrichi du parsing d'un tableau Markdown."""
    headers: List[str]
    data_rows: List[List[str]]
    suspicious_row_indices: List[int] = field(default_factory=list)
    empty_cell_map: Dict[int, List[int]] = field(default_factory=dict)


_RE_SEPARATOR_LINE = re.compile(r"^\|?[\s\-:|]+\|[\s\-:|]*$")
_RE_NONBREAKING = re.compile(r"[\u00a0\u202f\u2009\u200b]")
_RE_LONG_DASHES = re.compile(r"[\u2013\u2014\u2015]")


def _normalize_cell(value: str) -> str:
    """Normalise une cellule : espaces insécables → espace, tirets longs → tiret ASCII."""
    value = _RE_NONBREAKING.sub(" ", value)
    value = _RE_LONG_DASHES.sub("-", value)
    return unicodedata.normalize("NFC", value).strip()


def _split_md_row(line: str) -> List[str]:
    """Découpe une ligne Markdown en cellules en gérant les pipes internes échappés."""
    # Retire les pipes de bordure
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    # Découpe sur | non précédé d'un backslash
    cells = re.split(r"(?<!\\)\|", line)
    return [_normalize_cell(c) for c in cells]


def _is_markdown_table_text(text: str) -> bool:
    """Heuristique : grille Markdown présente (| … |)."""
    if not text or not text.strip():
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return lines[0].startswith("|") and "|" in lines[0]


def _parse_markdown_table_robust(text: str) -> Optional[TableParseResult]:
    """
    Parse un tableau Markdown en TableParseResult enrichi.

    - Parsing robuste via regex (gère les pipes internes échappés)
    - Normalisation UTF-8 des cellules (espaces insécables, tirets longs)
    - Détection des lignes suspectes (nb colonnes ≠ nb en-têtes)
    - Inventaire des cellules vides par position (row_idx → [col_idx])
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

    headers = _split_md_row(table_lines[0])
    if not headers or not any(h for h in headers):
        return None
    nb_cols = len(headers)

    # Détecter et sauter la ligne séparatrice (|---|---|)
    start_idx = 1
    if len(table_lines) > 1 and _RE_SEPARATOR_LINE.match(
        table_lines[1].replace(" ", "")
    ):
        start_idx = 2

    data_rows: List[List[str]] = []
    suspicious_row_indices: List[int] = []
    empty_cell_map: Dict[int, List[int]] = {}

    for raw_line in table_lines[start_idx:]:
        cells = _split_md_row(raw_line)
        if not any(cells):
            continue

        row_idx = len(data_rows)

        if len(cells) != nb_cols:
            suspicious_row_indices.append(row_idx)

        # Aligner sur nb_cols
        while len(cells) < nb_cols:
            cells.append("")
        cells = cells[:nb_cols]

        # Inventaire des cellules vides
        empty_cols = [ci for ci, v in enumerate(cells) if not v]
        if empty_cols:
            empty_cell_map[row_idx] = empty_cols

        data_rows.append(cells)

    if not data_rows:
        return None

    return TableParseResult(
        headers=headers,
        data_rows=data_rows,
        suspicious_row_indices=suspicious_row_indices,
        empty_cell_map=empty_cell_map,
    )


def _parse_markdown_table_legacy(text: str) -> Optional[Tuple[List[str], List[List[str]]]]:
    """Ancien parser conservé pour rétrocompatibilité."""
    result = _parse_markdown_table_robust(text)
    if result is None:
        return None
    return result.headers, result.data_rows


# Alias public maintenu pour ne pas casser les appelants éventuels
_parse_markdown_table = _parse_markdown_table_legacy


def _serialize_markdown_table(headers: List[str], data_rows: List[List[str]]) -> str:
    """Re-sérialise un tableau parsé en Markdown canonique (colonnes alignées)."""
    col_widths = [len(h) for h in headers]
    for row in data_rows:
        for ci, cell in enumerate(row):
            if ci < len(col_widths):
                col_widths[ci] = max(col_widths[ci], len(cell))

    def _fmt_row(cells: List[str]) -> str:
        padded = [c.ljust(col_widths[ci]) for ci, c in enumerate(cells) if ci < len(col_widths)]
        return "| " + " | ".join(padded) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines = [_fmt_row(headers), sep]
    for row in data_rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


def _build_table_json(
    headers: List[str],
    data_rows: List[List[str]],
    caption: Optional[str],
    page_no: Optional[int],
    suspicious_rows: List[int],
) -> dict:
    """Construit le JSON canonique d'un tableau pour metadata_json.table_json."""
    return {
        "caption": caption or "",
        "page": page_no,
        "headers": headers,
        "nb_cols": len(headers),
        "nb_rows": len(data_rows),
        "suspicious_rows": suspicious_rows,
        "rows": [dict(zip(headers, row)) for row in data_rows],
    }


def _table_full_chunk_text(
    *,
    headers: List[str],
    data_rows: List[List[str]],
    parent_heading: str,
    caption: Optional[str],
    page_no: Optional[int],
) -> str:
    """Texte du chunk table_full : en-tête contextuel + tableau Markdown canonique."""
    ctx_parts = []
    if parent_heading:
        ctx_parts.append(f"[{parent_heading}]")
    if caption:
        ctx_parts.append(f"Tableau : {caption}")
    else:
        ctx_parts.append("Tableau")
    if page_no is not None:
        ctx_parts.append(f"(p.{page_no})")
    header_line = " ".join(ctx_parts)
    md_table = _serialize_markdown_table(headers, data_rows)
    return f"{header_line}\n\n{md_table}"


def _table_row_chunk_text(
    *,
    headers: List[str],
    cells: List[str],
    parent_heading: str,
    caption: Optional[str],
    page_no: Optional[int],
    table_id: str,
    row_index: int,
    total_rows: int = 0,
    suspicious: bool = False,
    empty_col_indices: Optional[List[int]] = None,
) -> str:
    """
    Phrase autonome par ligne avec réinjection des en-têtes de colonnes.

    - Numéro de colonne inclus dans chaque paire header=valeur
    - Cellule vide signalée par [vide] (indexable par le LLM)
    - Ligne suspecte signalée par [décalage probable] en préfixe
    """
    parts = []
    for ci, (h, c) in enumerate(zip(headers, cells)):
        h = (h or "").strip()
        c = (c or "").strip()
        col_label = f"col{ci + 1}:{h}" if h else f"col{ci + 1}"
        if c:
            parts.append(f"{col_label}={c}")
        else:
            parts.append(f"{col_label}=[vide]")

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

    row_label = (
        f"Ligne {row_index + 1}/{total_rows}"
        if total_rows > 0
        else f"Ligne {row_index + 1}"
    )
    body = " | ".join(parts)
    result = f"{prefix} — {row_label}: {body}"
    if suspicious:
        result = f"[décalage probable] {result}"
    return result


def _table_summary_chunk_text(
    *,
    headers: List[str],
    data_rows: List[List[str]],
    parent_heading: str,
    caption: Optional[str],
    page_no: Optional[int],
) -> str:
    """
    Chunk table_summary pour les tableaux 2 colonnes (type clé-valeur).
    Produit une liste de paires 'Clé → Valeur' lisible par le LLM.
    """
    ctx_parts = []
    if parent_heading:
        ctx_parts.append(f"[{parent_heading}]")
    if caption:
        ctx_parts.append(f"Tableau clé-valeur : {caption}")
    else:
        ctx_parts.append("Tableau clé-valeur")
    if page_no is not None:
        ctx_parts.append(f"(p.{page_no})")
    header_line = " ".join(ctx_parts)

    lines = [header_line, ""]
    key_col = headers[0] if headers else "Clé"
    val_col = headers[1] if len(headers) > 1 else "Valeur"
    lines.append(f"{key_col} → {val_col}")
    lines.append("")
    for row in data_rows:
        k = row[0] if row else ""
        v = row[1] if len(row) > 1 else ""
        if k or v:
            lines.append(f"{k or '[vide]'} → {v or '[vide]'}")
    return "\n".join(lines)


def _should_expand_table_leaf(meta: dict, raw_content: str) -> bool:
    if meta.get("label") == "table":
        return True
    return _is_markdown_table_text(raw_content)


def _heading_path_list(headings: list) -> List[str]:
    """Liste des titres de section normalisés (fil d'Ariane)."""
    if not headings or not isinstance(headings, list):
        return []
    return [str(h).strip() for h in headings if h is not None and str(h).strip()]


def _format_text_full_chunk_text(
    raw_body: str,
    headings: list,
    parent_heading_display: str,
    page_no: Optional[int],
) -> str:
    """
    Texte canonique pour embedding : contenu brut du chunk, sans préfixe.

    Le contexte (heading_path, page_no, etc.) est conservé uniquement dans metadata_json.
    """
    return (raw_body or "").strip()


def _split_text_into_windows(text: str, max_chars: int, overlap: int) -> List[str]:
    """Découpe un texte en fenêtres glissantes (max_chars, overlap)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return []
    out: List[str] = []
    start = 0
    overlap = max(0, min(overlap, max_chars // 2))
    while start < len(text):
        end = min(start + max_chars, len(text))
        out.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return out


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
    base_meta["chunking_version"] = CHUNKING_VERSION_DOCLING_HIERARCHICAL_V2

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
            parsed_table_result: Optional[TableParseResult] = None
            if _should_expand_table_leaf(docling_meta, table_src):
                parsed_table_result = _parse_markdown_table_robust(table_src)
            if parsed_table_result is None and _should_expand_table_leaf(docling_meta, raw_content):
                parsed_table_result = _parse_markdown_table_robust(raw_content)

            if parsed_table_result:
                headers = parsed_table_result.headers
                data_rows = parsed_table_result.data_rows
                suspicious_row_indices = parsed_table_result.suspicious_row_indices
                empty_cell_map = parsed_table_result.empty_cell_map

                table_id = str(uuid.uuid4())
                page_no = docling_meta.get("page_no")
                cap = caption or _extract_caption_from_metadata(docling_meta)
                total_rows = len(data_rows)

                # --- chunk table_full (niveau 1, parent = section_parent) ---
                table_full_node_id = str(uuid.uuid4())
                full_text = _table_full_chunk_text(
                    headers=headers,
                    data_rows=data_rows,
                    parent_heading=parent_heading_display,
                    caption=cap,
                    page_no=page_no,
                )
                table_json_obj = _build_table_json(
                    headers=headers,
                    data_rows=data_rows,
                    caption=cap,
                    page_no=page_no,
                    suspicious_rows=suspicious_row_indices,
                )
                full_metadata = dict(base_meta)
                full_metadata.update(docling_meta)
                full_metadata["parent_heading"] = parent_heading_display
                full_metadata["heading"] = parent_heading_display
                full_metadata["content_type"] = "table_full"
                full_metadata["table_id"] = table_id
                full_metadata["nb_rows"] = total_rows
                full_metadata["nb_cols"] = len(headers)
                full_metadata["suspicious_rows"] = suspicious_row_indices
                full_metadata["column_headers"] = headers
                full_metadata["table_json"] = table_json_obj
                full_metadata["raw_content"] = raw_content
                if section_anchors:
                    full_metadata["image_anchor"] = " ; ".join(section_anchors)
                    full_metadata["figure_title"] = section_anchors[0]
                elif cap:
                    full_metadata["image_anchor"] = cap
                    full_metadata["figure_title"] = cap
                full_metadata.update(
                    {
                        "node_id": table_full_node_id,
                        "parent_node_id": parent_node_id,
                        "hierarchy_level": 1,
                        "is_leaf": "false",
                    }
                )
                if page_no is not None:
                    full_metadata["page_no"] = page_no

                specs.append(
                    {
                        "chunk_index": chunk_index,
                        "is_leaf": False,
                        "content": full_text,
                        "text": full_text,
                        "start_char": 0,
                        "end_char": len(full_text),
                        "node_id": table_full_node_id,
                        "parent_node_id": parent_node_id,
                        "hierarchy_level": 1,
                        "metadata_json": full_metadata,
                    }
                )
                chunk_index += 1

                # --- chunk table_summary pour les tableaux 2 colonnes (clé-valeur) ---
                if len(headers) == 2:
                    summary_text = _table_summary_chunk_text(
                        headers=headers,
                        data_rows=data_rows,
                        parent_heading=parent_heading_display,
                        caption=cap,
                        page_no=page_no,
                    )
                    if summary_text.strip():
                        summary_node_id = str(uuid.uuid4())
                        summary_metadata = dict(base_meta)
                        summary_metadata.update(docling_meta)
                        summary_metadata["parent_heading"] = parent_heading_display
                        summary_metadata["heading"] = parent_heading_display
                        summary_metadata["content_type"] = "table_summary"
                        summary_metadata["table_id"] = table_id
                        summary_metadata["column_headers"] = headers
                        summary_metadata.update(
                            {
                                "node_id": summary_node_id,
                                "parent_node_id": table_full_node_id,
                                "hierarchy_level": 2,
                                "is_leaf": "true",
                            }
                        )
                        if page_no is not None:
                            summary_metadata["page_no"] = page_no

                        specs.append(
                            {
                                "chunk_index": chunk_index,
                                "is_leaf": True,
                                "content": summary_text,
                                "text": summary_text,
                                "start_char": 0,
                                "end_char": len(summary_text),
                                "node_id": summary_node_id,
                                "parent_node_id": table_full_node_id,
                                "hierarchy_level": 2,
                                "metadata_json": summary_metadata,
                            }
                        )
                        chunk_index += 1

                # --- chunks table_row (niveau 2, parent = table_full) ---
                for ri, cells in enumerate(data_rows):
                    is_suspicious = ri in suspicious_row_indices
                    empty_cols = empty_cell_map.get(ri, [])
                    row_text = _table_row_chunk_text(
                        headers=headers,
                        cells=cells,
                        parent_heading=parent_heading_display,
                        caption=cap,
                        page_no=page_no,
                        table_id=table_id,
                        row_index=ri,
                        total_rows=total_rows,
                        suspicious=is_suspicious,
                        empty_col_indices=empty_cols,
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
                    leaf_metadata["suspicious"] = is_suspicious
                    if empty_cols:
                        leaf_metadata["empty_col_indices"] = empty_cols
                    if section_anchors:
                        leaf_metadata["image_anchor"] = " ; ".join(section_anchors)
                        leaf_metadata["figure_title"] = section_anchors[0]
                    elif cap:
                        leaf_metadata["image_anchor"] = cap
                        leaf_metadata["figure_title"] = cap
                    leaf_metadata.update(
                        {
                            "node_id": leaf_rid,
                            "parent_node_id": table_full_node_id,
                            "hierarchy_level": 2,
                            "is_leaf": "true",
                        }
                    )
                    if page_no is not None:
                        leaf_metadata["page_no"] = page_no
                    leaf_metadata["contains_image"] = True

                    specs.append(
                        {
                            "chunk_index": chunk_index,
                            "is_leaf": True,
                            "content": row_text,
                            "text": row_text,
                            "start_char": 0,
                            "end_char": len(row_text),
                            "node_id": leaf_rid,
                            "parent_node_id": table_full_node_id,
                            "hierarchy_level": 2,
                            "metadata_json": leaf_metadata,
                        }
                    )
                    chunk_index += 1
                continue

            page_no_val = docling_meta.get("page_no")
            heading_path = _heading_path_list(headings)
            heading_depth = len(heading_path)
            semantic_kind = _detect_content_type(raw_content)

            full_formatted = _format_text_full_chunk_text(
                raw_content,
                headings,
                parent_heading_display,
                page_no_val,
            )

            text_full_node_id = str(uuid.uuid4())
            text_full_metadata = dict(base_meta)
            text_full_metadata.update(docling_meta)
            text_full_metadata["parent_heading"] = parent_heading_display
            text_full_metadata["heading"] = parent_heading_display
            text_full_metadata["heading_path"] = heading_path
            text_full_metadata["heading_depth"] = heading_depth
            text_full_metadata["content_type"] = "text_full"
            text_full_metadata["semantic_content_kind"] = semantic_kind
            if parent_heading_display:
                text_full_metadata["raw_content"] = raw_content
            if section_anchors:
                text_full_metadata["image_anchor"] = " ; ".join(section_anchors)
                text_full_metadata["figure_title"] = section_anchors[0]
            elif caption:
                text_full_metadata["image_anchor"] = caption
                text_full_metadata["figure_title"] = caption
            text_full_metadata.update(
                {
                    "node_id": text_full_node_id,
                    "parent_node_id": parent_node_id,
                    "hierarchy_level": 1,
                    "is_leaf": "true",
                }
            )
            if page_no_val is not None:
                text_full_metadata["page_no"] = page_no_val
            if section_anchors or _is_picture_or_table_chunk(docling_meta):
                text_full_metadata["contains_image"] = True

            specs.append(
                {
                    "chunk_index": chunk_index,
                    "is_leaf": True,
                    "content": full_formatted,
                    "text": full_formatted,
                    "start_char": 0,
                    "end_char": len(full_formatted),
                    "node_id": text_full_node_id,
                    "parent_node_id": parent_node_id,
                    "hierarchy_level": 1,
                    "metadata_json": text_full_metadata,
                }
            )
            chunk_index += 1

            tw_threshold = int(
                getattr(settings, "DOCLING_TEXT_WINDOW_CHAR_THRESHOLD", 0) or 0
            )
            tw_overlap = int(getattr(settings, "DOCLING_TEXT_WINDOW_OVERLAP", 200) or 0)
            if tw_threshold > 0 and len(raw_content) > tw_threshold:
                windows = _split_text_into_windows(raw_content, tw_threshold, tw_overlap)
                n_win = len(windows)
                for wi, wtext in enumerate(windows):
                    w_formatted = _format_text_full_chunk_text(
                        wtext,
                        headings,
                        parent_heading_display,
                        page_no_val,
                    )
                    win_body = (
                        f"(Fenêtre {wi + 1}/{n_win})\n\n{w_formatted}"
                        if n_win > 1
                        else w_formatted
                    )
                    win_id = str(uuid.uuid4())
                    win_meta = dict(base_meta)
                    win_meta.update(docling_meta)
                    win_meta["parent_heading"] = parent_heading_display
                    win_meta["heading"] = parent_heading_display
                    win_meta["heading_path"] = heading_path
                    win_meta["heading_depth"] = heading_depth
                    win_meta["content_type"] = "text_window"
                    win_meta["semantic_content_kind"] = semantic_kind
                    win_meta["text_window_index"] = wi
                    win_meta["text_window_count"] = n_win
                    win_meta["parent_text_full_node_id"] = text_full_node_id
                    if parent_heading_display:
                        win_meta["raw_content"] = wtext
                    win_meta.update(
                        {
                            "node_id": win_id,
                            "parent_node_id": text_full_node_id,
                            "hierarchy_level": 2,
                            "is_leaf": "true",
                        }
                    )
                    if page_no_val is not None:
                        win_meta["page_no"] = page_no_val
                    specs.append(
                        {
                            "chunk_index": chunk_index,
                            "is_leaf": True,
                            "content": win_body,
                            "text": win_body,
                            "start_char": 0,
                            "end_char": len(win_body),
                            "node_id": win_id,
                            "parent_node_id": text_full_node_id,
                            "hierarchy_level": 2,
                            "metadata_json": win_meta,
                        }
                    )
                    chunk_index += 1

    return specs



def chunk_document_from_docling_docs(
    document: LibraryDocument,
    llama_docs: Sequence,
) -> List[DocumentChunk]:
    """
    Découpe un Document bibliothèque via DoclingNodeParser (même logique que les notes).

    Retourne des DocumentChunk hiérarchiques (parents + leaves) avec métadonnées Docling.
    En cas d'échec ou liste vide, l'appelant doit retomber sur create_chunks_for_document.
    """
    ld = get_library_document_logger()
    ld.info(
        "[DoclingNodeParser] document_id=%s — étape : chargement du parser + "
        "get_nodes_from_documents(JSON Docling).",
        document.id,
    )
    try:
        node_parser = _get_docling_node_parser()
    except ImportError as exc:
        tb = traceback.format_exc()
        logger.warning(
            "llama-index-node-parser-docling non installé — "
            "chunk_document_from_docling_docs indisponible pour document %s",
            document.id,
        )
        logger.error(
            "Import DoclingNodeParser (document %s): %r\n%s",
            document.id,
            exc,
            tb,
        )
        ld.error(
            "[DoclingNodeParser] document_id=%s — ÉCHEC : import du parser : %r. "
            "Traceback (voir aussi logs applicatifs) :\n%s"
            "→ repli chunking markdown prévu.",
            document.id,
            exc,
            tb,
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
        ld.error(
            "[DoclingNodeParser] document_id=%s — ÉCHEC : exception dans get_nodes_from_documents "
            "(JSON incompatible, version docling/llama-index, etc.) : %s. → repli markdown.",
            document.id,
            exc,
            exc_info=True,
        )
        return []

    if not leaf_nodes:
        logger.warning(
            "DoclingNodeParser n'a produit aucun nœud pour le document %s",
            document.id,
        )
        ld.error(
            "[DoclingNodeParser] document_id=%s — ÉCHEC : 0 nœud feuille retourné "
            "(document vide côté parser ou filtre trop strict). → repli markdown.",
            document.id,
        )
        return []

    ld.info(
        "[DoclingNodeParser] document_id=%s — %d nœud(s) feuille(s) LlamaIndex reçus ; "
        "construction des specs parent/feuille (_build_docling_hierarchical_specs).",
        document.id,
        len(leaf_nodes),
    )

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
    ld.info(
        "[DoclingNodeParser] document_id=%s — succès : %d chunks SQL "
        "(%d feuilles is_leaf=True, %d parents is_leaf=False). "
        "Les parents permettent la résolution de contexte en recherche RAG.",
        document.id,
        len(chunks),
        leaf_count,
        parent_count,
    )
    return chunks


# ---------------------------------------------------------------------------
# Stratégie principale : markdown Mistral OCR → HierarchicalNodeParser
# ---------------------------------------------------------------------------

_PAGE_MARKER_RE = re.compile(r"<!--\s*page:(\d+)\s*-->", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class MarkdownSegment:
    """Segment texte ou tableau extrait du markdown OCR."""
    kind: str  # "text" | "table"
    content: str
    start_char: int
    end_char: int
    page_no: Optional[int] = None
    caption: Optional[str] = None
    parent_heading: Optional[str] = None


def _page_no_from_markdown_segment(text: str, fallback: Optional[int] = None) -> Optional[int]:
    m = _PAGE_MARKER_RE.search(text)
    if m:
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            pass
    return fallback


def _build_page_marker_index(markdown: str) -> List[Tuple[int, int]]:
    """Index (offset caractère, numéro de page) pour tous les marqueurs <!-- page:N -->."""
    index: List[Tuple[int, int]] = []
    for match in _PAGE_MARKER_RE.finditer(markdown or ""):
        try:
            page_no = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if page_no > 0:
            index.append((match.start(), page_no))
    return index


def _page_no_from_char_offset(
    offset: int,
    page_index: List[Tuple[int, int]],
    fallback: Optional[int] = None,
) -> Optional[int]:
    """Retourne la page active à un offset (dernier marqueur <= offset)."""
    page_no = fallback
    for pos, pno in page_index:
        if pos <= offset:
            page_no = pno
        else:
            break
    return page_no


def _parent_heading_from_markdown(text: str) -> Optional[str]:
    headings = _HEADING_RE.findall(text)
    if not headings:
        return None
    return headings[-1][1].strip()


def _strip_heading_markup(title: str) -> str:
    """Retire le gras markdown (**titre**) pour métadonnées et clés de regroupement."""
    return re.sub(r"\*+", "", (title or "").strip()).strip()


def _heading_path_list_from_meta(meta: dict) -> List[str]:
    """Extrait le chemin de titres depuis les métadonnées MarkdownNodeParser."""
    parts: List[str] = []
    for i in range(1, 10):
        key = f"Header {i}"
        val = meta.get(key)
        if val and str(val).strip():
            clean = _strip_heading_markup(str(val).strip())
            if clean:
                parts.append(clean)
    if parts:
        return parts
    hp = meta.get("header_path")
    if isinstance(hp, str) and hp.strip() and hp.strip() not in ("/", ""):
        return [
            _strip_heading_markup(p.strip())
            for p in hp.split("/")
            if p.strip()
        ] or [
            _strip_heading_markup(p.strip())
            for p in re.split(r"\s*>\s*", hp)
            if p.strip()
        ]
    return parts


def _format_heading_path(parts: Sequence[str]) -> str:
    cleaned = [p for p in (_strip_heading_markup(x) for x in parts) if p]
    return HEADER_PATH_SEP.join(cleaned)


_MAJOR_SECTION_MARKERS = (
    "CATALOGUE",
    "VITRAGES",
    "GARANTIES",
    "QUESTIONS",
    "RÉSUMÉ",
    "RESUME",
    "PRÉSENTATION",
    "PRESENTATION",
    "DOCUMENT STRUCTURÉ",
    "DOCUMENT STRUCTURE",
    "PERSONNALISATIONS",
)


def _is_major_section_heading(title: str) -> bool:
    """Repère les sections racine type catalogue (souvent ## seuls dans les plaquettes marketing)."""
    t = _strip_heading_markup(title).upper()
    return any(m in t for m in _MAJOR_SECTION_MARKERS)


def _infer_heading_stack_from_leaf_content(content: str, stack: List[str]) -> List[str]:
    """
    Met à jour la pile de titres à partir du premier header du chunk (ordre document).
    Gère les plaquettes avec plusieurs ## frères (INTERIEURES / EXTERIEURES / VITRAGES).
    """
    first_line = (content or "").strip().split("\n", 1)[0].strip()
    m = re.match(r"^(#{1,6})\s+(.+)$", first_line)
    if not m:
        return list(stack)
    level = len(m.group(1))
    title = _strip_heading_markup(m.group(2).strip())
    if not title:
        return list(stack)
    if level <= 2 and _is_major_section_heading(title):
        return [title]
    new_stack = stack[: level - 1]
    new_stack.append(title)
    return new_stack


def _enrich_leaf_specs_heading_paths(leaf_specs: List[dict]) -> None:
    """Ajoute heading_path / heading_path_list / scope_label sur chaque feuille (in-place)."""
    stack: List[str] = []
    for spec in leaf_specs:
        if not spec.get("is_leaf", True):
            continue
        content = spec.get("content") or ""
        stack = _infer_heading_stack_from_leaf_content(content, stack)
        meta = dict(spec.get("metadata_json") or {})
        path_parts = _heading_path_list_from_meta(meta)
        if not path_parts and stack:
            path_parts = list(stack)
        elif path_parts and stack:
            # Fusionner : préférer la pile document si plus profonde
            if len(stack) > len(path_parts):
                path_parts = list(stack)
        if not path_parts:
            immediate = _parent_heading_from_markdown(content)
            if immediate:
                path_parts = [_strip_heading_markup(immediate)]
        meta["heading_path_list"] = path_parts
        meta["heading_path"] = _format_heading_path(path_parts)
        if path_parts:
            meta["parent_heading"] = path_parts[-1]
            meta["scope_label"] = path_parts[0] if len(path_parts) >= 1 else path_parts[-1]
        spec["metadata_json"] = meta


def _parent_group_key(path_parts: List[str], parent_depth: int) -> str:
    """Clé de regroupement pour les chunks parents (ex. tout le catalogue EXTERIEURES)."""
    depth = max(1, int(parent_depth or 1))
    if not path_parts:
        return "__document__"
    return _format_heading_path(path_parts[: min(depth, len(path_parts))])


def _build_section_parent_chunks(
    leaf_specs: List[dict],
    base_meta: dict,
    *,
    parent_depth: int = 1,
) -> List[dict]:
    """
    Crée des parents is_leaf=False qui agrègent toutes les feuilles d'une même section.
    Les feuilles reçoivent parent_node_id et hierarchy_level=1.
    """
    from app.config import settings

    depth = int(
        getattr(settings, "MARKDOWN_STRUCTURED_PARENT_DEPTH", None) or parent_depth or 1
    )
    groups: Dict[str, List[dict]] = defaultdict(list)
    for spec in leaf_specs:
        if not spec.get("is_leaf", True):
            continue
        meta = spec.get("metadata_json") or {}
        path_parts = meta.get("heading_path_list") or []
        if isinstance(path_parts, str):
            path_parts = [p.strip() for p in path_parts.split(HEADER_PATH_SEP) if p.strip()]
        key = _parent_group_key(path_parts, depth)
        groups[key].append(spec)

    parent_specs: List[dict] = []
    for group_key, leaves in groups.items():
        if not leaves:
            continue
        leaves_sorted = sorted(
            leaves,
            key=lambda s: (int(s.get("start_char", 0) or 0), int(s.get("chunk_index", 0) or 0)),
        )
        path_parts = (leaves_sorted[0].get("metadata_json") or {}).get("heading_path_list") or []
        if group_key != "__document__" and path_parts:
            section_title = _format_heading_path(path_parts[:depth])
        else:
            section_title = group_key

        body_parts: List[str] = []
        for leaf in leaves_sorted:
            body_parts.append((leaf.get("content") or "").strip())
        body = "\n\n---\n\n".join(p for p in body_parts if p)
        if not body.strip():
            continue

        parent_node_id = str(uuid.uuid4())
        parent_content = f"## {section_title}\n\n{body}" if section_title != "__document__" else body

        start_char = min(int(s.get("start_char", 0) or 0) for s in leaves_sorted)
        end_char = max(int(s.get("end_char", 0) or 0) for s in leaves_sorted)
        page_nos = []
        for s in leaves_sorted:
            pn = (s.get("metadata_json") or {}).get("page_no")
            if pn is not None:
                try:
                    page_nos.append(int(pn))
                except (TypeError, ValueError):
                    pass

        parent_meta = dict(base_meta)
        parent_meta.update(
            {
                "node_id": parent_node_id,
                "parent_node_id": None,
                "hierarchy_level": 0,
                "is_leaf": "false",
                "heading_path_list": path_parts[:depth] if path_parts else [],
                "heading_path": section_title,
                "parent_heading": section_title,
                "scope_label": path_parts[0] if path_parts else section_title,
                "content_type": "section_parent",
                "nb_child_leaves": len(leaves_sorted),
            }
        )
        if page_nos:
            parent_meta["page_no"] = min(page_nos)
            parent_meta["page_start"] = min(page_nos)
            parent_meta["page_end"] = max(page_nos)

        parent_specs.append(
            {
                "chunk_index": 0,
                "is_leaf": False,
                "content": parent_content,
                "text": parent_content,
                "start_char": start_char,
                "end_char": end_char,
                "node_id": parent_node_id,
                "parent_node_id": None,
                "hierarchy_level": 0,
                "metadata_json": parent_meta,
            }
        )

        for leaf in leaves_sorted:
            leaf["parent_node_id"] = parent_node_id
            leaf["hierarchy_level"] = 1
            lmeta = dict(leaf.get("metadata_json") or {})
            lmeta["parent_node_id"] = parent_node_id
            lmeta["section_parent_heading"] = section_title
            leaf["metadata_json"] = lmeta

    return parent_specs


def _find_parent_heading_before(full_md: str, end_pos: int) -> Optional[str]:
    prefix = (full_md or "")[: max(0, end_pos)]
    return _parent_heading_from_markdown(prefix)


def _extract_caption_before_table(full_md: str, table_start: int) -> Optional[str]:
    prefix = (full_md or "")[: max(0, table_start)].rstrip()
    if not prefix:
        return None
    lines = [ln.strip() for ln in prefix.splitlines() if ln.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    if _PAGE_MARKER_RE.match(candidate):
        candidate = lines[-2] if len(lines) > 1 else ""
    if not candidate or candidate.startswith("#") or candidate.startswith("|"):
        return None
    if len(candidate) > 200:
        return None
    return candidate


def _line_is_table_row(line: str) -> bool:
    stripped = (line or "").strip()
    return bool(stripped.startswith("|") and "|" in stripped[1:])


def _find_markdown_table_spans(text: str) -> List[Tuple[int, int, str]]:
    """Retourne (start, end, table_text) pour chaque bloc tableau markdown."""
    if not text:
        return []
    lines = text.split("\n")
    spans: List[Tuple[int, int, str]] = []
    offset = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        line_start = offset
        line_len = len(line)
        next_offset = offset + line_len + (1 if i < len(lines) - 1 else 0)

        if not _line_is_table_row(line):
            offset = next_offset
            i += 1
            continue

        block_lines = [line]
        j = i + 1
        block_end_offset = next_offset
        while j < len(lines):
            ln = lines[j]
            stripped = ln.strip()
            if _line_is_table_row(ln) or (
                stripped and _RE_SEPARATOR_LINE.match(stripped.replace(" ", ""))
            ):
                block_lines.append(ln)
                block_end_offset += len(ln) + (1 if j < len(lines) - 1 else 0)
                j += 1
            else:
                break

        block_text = "\n".join(block_lines)
        if _is_markdown_table_text(block_text):
            spans.append((line_start, block_end_offset, block_text))
            offset = block_end_offset
            i = j
        else:
            offset = next_offset
            i += 1
    return spans


def _split_markdown_into_segments(markdown: str) -> List[MarkdownSegment]:
    """Découpe le markdown en segments texte / tableau en préservant les offsets."""
    md = (markdown or "").strip()
    if not md:
        return []

    page_index = _build_page_marker_index(md)
    table_spans = _find_markdown_table_spans(md)
    if not table_spans:
        return [
            MarkdownSegment(
                kind="text",
                content=md,
                start_char=0,
                end_char=len(md),
                page_no=_page_no_from_char_offset(0, page_index),
                parent_heading=_parent_heading_from_markdown(md),
            )
        ]

    segments: List[MarkdownSegment] = []
    cursor = 0
    for start, end, table_text in table_spans:
        if start > cursor:
            text_part = md[cursor:start]
            stripped = text_part.strip()
            if stripped:
                seg_start = cursor + (len(text_part) - len(text_part.lstrip()))
                seg_end = cursor + len(text_part.rstrip())
                segments.append(
                    MarkdownSegment(
                        kind="text",
                        content=stripped,
                        start_char=seg_start,
                        end_char=seg_end,
                        page_no=_page_no_from_char_offset(seg_start, page_index),
                        parent_heading=_find_parent_heading_before(md, seg_start),
                    )
                )
        segments.append(
            MarkdownSegment(
                kind="table",
                content=table_text,
                start_char=start,
                end_char=end,
                page_no=_page_no_from_char_offset(start, page_index),
                caption=_extract_caption_before_table(md, start),
                parent_heading=_find_parent_heading_before(md, start),
            )
        )
        cursor = end

    if cursor < len(md):
        text_part = md[cursor:]
        stripped = text_part.strip()
        if stripped:
            seg_start = cursor + (len(text_part) - len(text_part.lstrip()))
            seg_end = cursor + len(text_part.rstrip())
            segments.append(
                MarkdownSegment(
                    kind="text",
                    content=stripped,
                    start_char=seg_start,
                    end_char=seg_end,
                    page_no=_page_no_from_char_offset(seg_start, page_index),
                    parent_heading=_find_parent_heading_before(md, seg_start),
                )
            )
    return segments


def normalize_markdown_tables(text: str) -> str:
    """Re-sérialise les grilles markdown valides (post-OCR)."""
    if not text or not text.strip():
        return text or ""
    spans = _find_markdown_table_spans(text)
    if not spans:
        return text
    parts: List[str] = []
    cursor = 0
    for start, end, table_text in spans:
        parts.append(text[cursor:start])
        parsed = _parse_markdown_table_robust(table_text)
        if parsed is not None:
            if parsed.suspicious_row_indices:
                logger.warning(
                    "Tableau OCR : lignes suspectes aux indices %s",
                    parsed.suspicious_row_indices,
                )
            parts.append(_serialize_markdown_table(parsed.headers, parsed.data_rows))
        else:
            parts.append(table_text)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _propagate_page_metadata_to_parents(specs: List[dict]) -> None:
    """Propage page_start/page_end/page_no des feuilles vers les parents."""
    children_by_parent: Dict[str, List[dict]] = defaultdict(list)
    for spec in specs:
        parent_id = spec.get("parent_node_id")
        if parent_id:
            children_by_parent[parent_id].append(spec)

    def _collect_pages(node_id: str) -> List[int]:
        pages: List[int] = []
        for child in children_by_parent.get(node_id, []):
            meta = child.get("metadata_json") or {}
            pn = meta.get("page_no")
            if pn is not None:
                try:
                    pages.append(int(pn))
                except (TypeError, ValueError):
                    pass
            pages.extend(_collect_pages(child["node_id"]))
        return pages

    for spec in specs:
        if spec.get("is_leaf"):
            continue
        node_id = spec.get("node_id")
        if not node_id:
            continue
        pages = _collect_pages(node_id)
        if not pages:
            continue
        meta = dict(spec.get("metadata_json") or {})
        meta["page_start"] = min(pages)
        meta["page_end"] = max(pages)
        meta["page_no"] = min(pages)
        spec["metadata_json"] = meta


def _expand_table_to_specs(
    *,
    table_text: str,
    base_meta: dict,
    parent_heading: str = "",
    caption: Optional[str] = None,
    page_no: Optional[int] = None,
    section_parent_node_id: Optional[str] = None,
    hierarchy_base: int = 0,
    start_char: int = 0,
    raw_content: Optional[str] = None,
) -> List[dict]:
    """Produit table_full + table_row (+ table_summary si 2 colonnes) pour un bloc markdown."""
    parsed = _parse_markdown_table_robust(table_text)
    if parsed is None:
        logger.warning(
            "Expansion table échouée (parse) — fallback texte, start_char=%s",
            start_char,
        )
        return []

    headers = parsed.headers
    data_rows = parsed.data_rows
    suspicious_row_indices = parsed.suspicious_row_indices
    empty_cell_map = parsed.empty_cell_map
    parent_heading_display = (parent_heading or "").strip()
    cap = caption
    raw = raw_content or table_text
    table_id = str(uuid.uuid4())
    total_rows = len(data_rows)
    parent_node_id = section_parent_node_id
    specs: List[dict] = []

    table_full_node_id = str(uuid.uuid4())
    full_text = _table_full_chunk_text(
        headers=headers,
        data_rows=data_rows,
        parent_heading=parent_heading_display,
        caption=cap,
        page_no=page_no,
    )
    table_json_obj = _build_table_json(
        headers=headers,
        data_rows=data_rows,
        caption=cap,
        page_no=page_no,
        suspicious_rows=suspicious_row_indices,
    )
    full_metadata = dict(base_meta)
    full_metadata["parent_heading"] = parent_heading_display
    full_metadata["heading"] = parent_heading_display
    full_metadata["content_type"] = "table_full"
    full_metadata["table_id"] = table_id
    full_metadata["nb_rows"] = total_rows
    full_metadata["nb_cols"] = len(headers)
    full_metadata["suspicious_rows"] = suspicious_row_indices
    full_metadata["column_headers"] = headers
    full_metadata["table_json"] = table_json_obj
    full_metadata["raw_content"] = raw
    if cap:
        full_metadata["image_anchor"] = cap
        full_metadata["figure_title"] = cap
    full_metadata.update(
        {
            "node_id": table_full_node_id,
            "parent_node_id": parent_node_id,
            "hierarchy_level": hierarchy_base,
            "is_leaf": "false",
        }
    )
    if page_no is not None:
        full_metadata["page_no"] = page_no

    specs.append(
        {
            "chunk_index": 0,
            "is_leaf": False,
            "content": full_text,
            "text": full_text,
            "start_char": start_char,
            "end_char": start_char + len(full_text),
            "node_id": table_full_node_id,
            "parent_node_id": parent_node_id,
            "hierarchy_level": hierarchy_base,
            "metadata_json": full_metadata,
        }
    )

    row_level = hierarchy_base + 1

    if len(headers) == 2:
        summary_text = _table_summary_chunk_text(
            headers=headers,
            data_rows=data_rows,
            parent_heading=parent_heading_display,
            caption=cap,
            page_no=page_no,
        )
        if summary_text.strip():
            summary_node_id = str(uuid.uuid4())
            summary_metadata = dict(base_meta)
            summary_metadata["parent_heading"] = parent_heading_display
            summary_metadata["heading"] = parent_heading_display
            summary_metadata["content_type"] = "table_summary"
            summary_metadata["table_id"] = table_id
            summary_metadata["column_headers"] = headers
            summary_metadata.update(
                {
                    "node_id": summary_node_id,
                    "parent_node_id": table_full_node_id,
                    "hierarchy_level": row_level,
                    "is_leaf": "true",
                }
            )
            if page_no is not None:
                summary_metadata["page_no"] = page_no
            specs.append(
                {
                    "chunk_index": 0,
                    "is_leaf": True,
                    "content": summary_text,
                    "text": summary_text,
                    "start_char": start_char,
                    "end_char": start_char + len(summary_text),
                    "node_id": summary_node_id,
                    "parent_node_id": table_full_node_id,
                    "hierarchy_level": row_level,
                    "metadata_json": summary_metadata,
                }
            )

    for ri, cells in enumerate(data_rows):
        is_suspicious = ri in suspicious_row_indices
        empty_cols = empty_cell_map.get(ri, [])
        row_text = _table_row_chunk_text(
            headers=headers,
            cells=cells,
            parent_heading=parent_heading_display,
            caption=cap,
            page_no=page_no,
            table_id=table_id,
            row_index=ri,
            total_rows=total_rows,
            suspicious=is_suspicious,
            empty_col_indices=empty_cols,
        )
        if not row_text.strip():
            continue

        leaf_rid = str(uuid.uuid4())
        leaf_metadata = dict(base_meta)
        leaf_metadata["parent_heading"] = parent_heading_display
        leaf_metadata["heading"] = parent_heading_display
        leaf_metadata["content_type"] = "table_row"
        leaf_metadata["table_id"] = table_id
        leaf_metadata["row_index"] = ri
        leaf_metadata["column_headers"] = headers
        leaf_metadata["raw_content"] = raw
        leaf_metadata["suspicious"] = is_suspicious
        if empty_cols:
            leaf_metadata["empty_col_indices"] = empty_cols
        if cap:
            leaf_metadata["image_anchor"] = cap
            leaf_metadata["figure_title"] = cap
        leaf_metadata.update(
            {
                "node_id": leaf_rid,
                "parent_node_id": table_full_node_id,
                "hierarchy_level": row_level,
                "is_leaf": "true",
            }
        )
        if page_no is not None:
            leaf_metadata["page_no"] = page_no

        specs.append(
            {
                "chunk_index": 0,
                "is_leaf": True,
                "content": row_text,
                "text": row_text,
                "start_char": start_char,
                "end_char": start_char + len(row_text),
                "node_id": leaf_rid,
                "parent_node_id": table_full_node_id,
                "hierarchy_level": row_level,
                "metadata_json": leaf_metadata,
            }
        )
    return specs


def _chunk_markdown_text_to_specs(
    text: str,
    base_meta: dict,
    *,
    char_offset: int = 0,
    page_index: Optional[List[Tuple[int, int]]] = None,
) -> List[dict]:
    """Découpe un segment texte via HierarchicalNodeParser avec offsets absolus."""
    segment = (text or "").strip()
    if not segment:
        return []
    if page_index is None:
        page_index = _build_page_marker_index(segment)

    chunk_sizes = list(DEFAULT_OCR_HIERARCHICAL_CHUNK_SIZES)
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    llama_doc = LlamaDocument(text=segment, metadata=dict(base_meta or {}))
    nodes = parser.get_nodes_from_documents([llama_doc])
    if not nodes:
        return []

    parent_map = _build_parent_map(nodes)
    level_map = _build_level_map(parent_map)
    leaf_ids = _detect_leaf_ids(parent_map)
    def _sort_key(n):
        start_idx = getattr(n, "start_char_idx", None)
        if start_idx is None:
            start_idx = (n.metadata or {}).get("start_char_idx", 0)
        return (
            level_map.get(n.node_id, 0),
            int(start_idx or 0),
            n.node_id,
        )

    nodes_sorted = sorted(nodes, key=_sort_key)

    specs: List[dict] = []
    for node in nodes_sorted:
        content = (node.get_content() or "").strip()
        if not content:
            continue

        node_id = node.node_id
        parent_node_id = parent_map.get(node_id)
        hierarchy_level = level_map.get(node_id, 0)
        is_leaf = node_id in leaf_ids
        
        start_idx = getattr(node, "start_char_idx", None)
        if start_idx is None:
            start_idx = (node.metadata or {}).get("start_char_idx", 0)
        rel_start = int(start_idx or 0)

        end_idx = getattr(node, "end_char_idx", None)
        if end_idx is None:
            end_idx = (node.metadata or {}).get("end_char_idx", rel_start + len(content))
        rel_end = int(end_idx or (rel_start + len(content)))
        start_char = char_offset + rel_start
        end_char = char_offset + rel_end

        meta = dict(node.metadata or {})
        meta.update(base_meta)
        meta.update(
            {
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "hierarchy_level": hierarchy_level,
                "is_leaf": "true" if is_leaf else "false",
            }
        )
        page_no = _page_no_from_char_offset(start_char, page_index)
        if page_no is None:
            page_no = _page_no_from_markdown_segment(content)
        if page_no is not None:
            meta["page_no"] = page_no
        parent_heading = _parent_heading_from_markdown(content)
        if parent_heading:
            meta["parent_heading"] = parent_heading

        specs.append(
            {
                "chunk_index": 0,
                "is_leaf": is_leaf,
                "content": content,
                "text": content,
                "start_char": start_char,
                "end_char": end_char,
                "node_id": node_id,
                "parent_node_id": parent_node_id,
                "hierarchy_level": hierarchy_level,
                "metadata_json": meta,
            }
        )
    return specs


def chunk_markdown_hierarchical(markdown: str, metadata_base: dict) -> List[dict]:
    """
    Découpe un markdown (Mistral OCR) en specs hiérarchiques parents/feuilles.
    Compatible DocumentChunk / NoteChunk (node_id, parent_node_id, is_leaf).
    """
    text = (markdown or "").strip()
    if not text:
        return []

    base_meta = dict(metadata_base or {})
    base_meta["chunking_version"] = CHUNKING_VERSION_MARKDOWN_HIERARCHICAL
    page_index = _build_page_marker_index(text)
    specs = _chunk_markdown_text_to_specs(
        text, base_meta, char_offset=0, page_index=page_index
    )
    for idx, spec in enumerate(specs):
        spec["chunk_index"] = idx

    logger.info(
        "Chunking markdown hiérarchique : %d specs (%d feuilles, %d parents) chunk_sizes=%s",
        len(specs),
        sum(1 for s in specs if s["is_leaf"]),
        sum(1 for s in specs if not s["is_leaf"]),
        DEFAULT_OCR_HIERARCHICAL_CHUNK_SIZES,
    )
    return specs


def chunk_markdown_hierarchical_with_tables(markdown: str, metadata_base: dict) -> List[dict]:
    """
    Chunking markdown OCR avec expansion atomique des tableaux (table_full / table_row).
    """
    text = (markdown or "").strip()
    if not text:
        return []

    page_index = _build_page_marker_index(text)
    segments = _split_markdown_into_segments(text)
    base_meta = dict(metadata_base or {})
    base_meta["chunking_version"] = CHUNKING_VERSION_MARKDOWN_HIERARCHICAL_V2

    all_specs: List[dict] = []

    for seg in segments:
        if seg.kind == "text":
            all_specs.extend(
                _chunk_markdown_text_to_specs(
                    seg.content,
                    base_meta,
                    char_offset=seg.start_char,
                    page_index=page_index,
                )
            )
            continue

        parent_heading = seg.parent_heading or ""
        section_parent_node_id: Optional[str] = None
        hierarchy_base = 0

        if parent_heading:
            section_parent_node_id = str(uuid.uuid4())
            section_meta = dict(base_meta)
            section_meta.update(
                {
                    "node_id": section_parent_node_id,
                    "parent_node_id": None,
                    "hierarchy_level": 0,
                    "is_leaf": "false",
                    "parent_heading": parent_heading,
                    "heading": parent_heading,
                }
            )
            if seg.page_no is not None:
                section_meta["page_no"] = seg.page_no
            section_content = f"[{parent_heading}]\n\n{seg.content}"
            all_specs.append(
                {
                    "chunk_index": 0,
                    "is_leaf": False,
                    "content": section_content,
                    "text": section_content,
                    "start_char": seg.start_char,
                    "end_char": seg.end_char,
                    "node_id": section_parent_node_id,
                    "parent_node_id": None,
                    "hierarchy_level": 0,
                    "metadata_json": section_meta,
                }
            )
            hierarchy_base = 1

        table_specs = _expand_table_to_specs(
            table_text=seg.content,
            base_meta=base_meta,
            parent_heading=parent_heading,
            caption=seg.caption,
            page_no=seg.page_no,
            section_parent_node_id=section_parent_node_id,
            hierarchy_base=hierarchy_base,
            start_char=seg.start_char,
            raw_content=seg.content,
        )
        if table_specs:
            all_specs.extend(table_specs)
        else:
            logger.warning(
                "Tableau non parsé — segment traité comme texte, start_char=%s",
                seg.start_char,
            )
            all_specs.extend(
                _chunk_markdown_text_to_specs(
                    seg.content,
                    base_meta,
                    char_offset=seg.start_char,
                    page_index=page_index,
                )
            )

    for idx, spec in enumerate(all_specs):
        spec["chunk_index"] = idx
    _propagate_page_metadata_to_parents(all_specs)

    logger.info(
        "Chunking markdown+tableaux : %d specs (%d feuilles, %d parents, %d table_row)",
        len(all_specs),
        sum(1 for s in all_specs if s["is_leaf"]),
        sum(1 for s in all_specs if not s["is_leaf"]),
        sum(
            1
            for s in all_specs
            if (s.get("metadata_json") or {}).get("content_type") == "table_row"
        ),
    )
    return all_specs


def _chunk_markdown_with_node_parser(
    text: str,
    base_meta: dict,
    *,
    char_offset: int = 0,
    page_index: Optional[List[Tuple[int, int]]] = None,
) -> List[dict]:
    """
    Découpe un segment texte via MarkdownNodeParser (structure headers).
    Produit des chunks is_leaf=True sans hiérarchie parent/enfant.
    """
    segment = (text or "").strip()
    if not segment:
        return []
    if page_index is None:
        page_index = _build_page_marker_index(segment)

    parser = MarkdownNodeParser.from_defaults(
        include_metadata=True,
        include_prev_next_rel=False,
    )
    llama_doc = LlamaDocument(text=segment, metadata=dict(base_meta or {}))
    nodes = parser.get_nodes_from_documents([llama_doc])
    if not nodes:
        return []

    specs: List[dict] = []
    for node in nodes:
        content = (node.get_content() or "").strip()
        if not content:
            continue

        node_id = node.node_id
        start_idx = getattr(node, "start_char_idx", None)
        if start_idx is None:
            start_idx = (node.metadata or {}).get("start_char_idx", 0)
        rel_start = int(start_idx or 0)

        end_idx = getattr(node, "end_char_idx", None)
        if end_idx is None:
            end_idx = (node.metadata or {}).get("end_char_idx", rel_start + len(content))
        rel_end = int(end_idx or (rel_start + len(content)))
        start_char = char_offset + rel_start
        end_char = char_offset + rel_end

        meta = dict(node.metadata or {})
        meta.update(base_meta)
        meta.update({
            "node_id": node_id,
            "parent_node_id": None,
            "hierarchy_level": 0,
            "is_leaf": "true",
        })

        path_parts = _heading_path_list_from_meta(meta)
        if path_parts:
            meta["heading_path_list"] = path_parts
            meta["heading_path"] = _format_heading_path(path_parts)
            meta["parent_heading"] = path_parts[-1]
            if len(path_parts) >= 1:
                meta["scope_label"] = path_parts[0]

        page_no = _page_no_from_char_offset(start_char, page_index)
        if page_no is None:
            page_no = _page_no_from_markdown_segment(content)
        if page_no is not None:
            meta["page_no"] = page_no

        specs.append({
            "chunk_index": 0,
            "is_leaf": True,
            "content": content,
            "text": content,
            "start_char": start_char,
            "end_char": end_char,
            "node_id": node_id,
            "parent_node_id": None,
            "hierarchy_level": 0,
            "metadata_json": meta,
        })
    return specs


def chunk_markdown_structured(markdown: str, metadata_base: dict) -> List[dict]:
    """
    Chunking markdown structurel : MarkdownNodeParser (feuilles) + parents de section.

    Pipeline :
    1. pymupdf4llm / OCR → markdown
    2. MarkdownNodeParser → feuilles par header
    3. Enrichissement heading_path (pile de titres document)
    4. Parents is_leaf=False par section (ex. tout le catalogue EXTERIEURES)
    5. Retrieval : vectoriel sur feuilles → remplacement par parent (contexte complet)
    """
    text = (markdown or "").strip()
    if not text:
        return []

    base_meta = dict(metadata_base or {})
    base_meta["chunking_version"] = CHUNKING_VERSION_MARKDOWN_STRUCTURED_V2
    page_index = _build_page_marker_index(text)

    segments = _split_markdown_into_segments(text)
    leaf_specs: List[dict] = []

    for seg in segments:
        if seg.kind == "text":
            leaf_specs.extend(
                _chunk_markdown_with_node_parser(
                    seg.content,
                    base_meta,
                    char_offset=seg.start_char,
                    page_index=page_index,
                )
            )
            continue

        parent_heading = seg.parent_heading or ""
        table_specs = _expand_table_to_specs(
            table_text=seg.content,
            base_meta=base_meta,
            parent_heading=parent_heading,
            caption=seg.caption,
            page_no=seg.page_no,
            section_parent_node_id=None,
            hierarchy_base=0,
            start_char=seg.start_char,
            raw_content=seg.content,
        )
        if table_specs:
            for spec in table_specs:
                spec["parent_node_id"] = None
                spec["hierarchy_level"] = 0
                if parent_heading:
                    ph = _strip_heading_markup(parent_heading)
                    tmeta = dict(spec.get("metadata_json") or {})
                    tmeta["heading_path_list"] = [ph]
                    tmeta["heading_path"] = ph
                    tmeta["scope_label"] = ph
                    spec["metadata_json"] = tmeta
            leaf_specs.extend(table_specs)

    _enrich_leaf_specs_heading_paths(leaf_specs)
    parent_specs = _build_section_parent_chunks(leaf_specs, base_meta)

    combined: List[dict] = parent_specs + leaf_specs
    combined.sort(
        key=lambda s: (
            int(s.get("start_char", 0) or 0),
            0 if not s.get("is_leaf", True) else 1,
        )
    )
    _propagate_page_metadata_to_parents(combined)
    for idx, spec in enumerate(combined):
        spec["chunk_index"] = idx

    n_leaf = sum(1 for s in combined if s.get("is_leaf"))
    n_parent = sum(1 for s in combined if not s.get("is_leaf"))
    logger.info(
        "Chunking markdown structurel v2: %d chunks (%d feuilles, %d parents section, "
        "%d table_row)",
        len(combined),
        n_leaf,
        n_parent,
        sum(
            1
            for s in leaf_specs
            if (s.get("metadata_json") or {}).get("content_type") == "table_row"
        ),
    )
    return combined


def _step_number_from_heading(heading: Optional[str]) -> Optional[int]:
    if not heading:
        return None
    m = re.match(r"^(\d+)\.", heading.strip())
    return int(m.group(1)) if m else None


def _split_block_by_numbered_steps(
    block: str,
    section_heading: Optional[str],
) -> List[dict]:
    """
    Découpe un bloc en sections sémantiques : intro éventuelle + étapes numérotées (1. 2. 3.).
    Chaque entrée : {heading, step_number, content, section_type}.
    """
    block = (block or "").strip()
    if not block:
        return []

    from app.services.pdf_extraction_service import _unwrap_bold_line

    matches = list(_NUMBERED_STEP_RE.finditer(block))
    if not matches:
        content = block
        if section_heading:
            content = f"{section_heading}\n\n{block}".strip() if block else section_heading
        return [
            {
                "heading": section_heading,
                "step_number": _step_number_from_heading(section_heading),
                "content": content,
                "section_type": "document_header" if not section_heading else "section",
            }
        ]

    sections: List[dict] = []

    # Intro avant la première étape numérotée (ex. légendes sous ## 1.)
    if matches[0].start() > 0:
        intro = block[: matches[0].start()].strip()
        if intro:
            combined = intro
            if section_heading:
                combined = f"{section_heading}\n\n{intro}"
            sections.append(
                {
                    "heading": section_heading,
                    "step_number": _step_number_from_heading(section_heading),
                    "content": combined,
                    "section_type": "section",
                }
            )

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        step_text = block[start:end].strip()
        if not step_text:
            continue
        step_num = int(m.group(1))
        first_line = _unwrap_bold_line(step_text.split("\n", 1)[0])
        sections.append(
            {
                "heading": first_line[:200],
                "step_number": step_num,
                "content": step_text,
                "section_type": "step",
            }
        )

    return sections


def chunk_pymupdf4llm_page_clean(
    page_text: str,
    page_no: int,
    metadata_base: dict,
) -> List[dict]:
    """
    Chunking sémantique propre pour une page pymupdf4llm :
    - 1 chunk par section ## ou par étape numérotée
    - Contenu nettoyé (sans placeholders image, sans marqueurs page)
    - Uniquement des feuilles (is_leaf=True)
    """
    from app.services.pdf_extraction_service import clean_pymupdf4llm_markdown

    cleaned = clean_pymupdf4llm_markdown(page_text)
    if not cleaned:
        return []

    base_meta = dict(metadata_base or {})
    base_meta["chunking_version"] = CHUNKING_VERSION_PYMUPDF4LLM_CLEAN

    raw_sections: List[dict] = []

    # Découpe par titres ## (structure pymupdf4llm typique)
    parts = _H2_SPLIT_RE.split(cleaned)
    if len(parts) == 1:
        raw_sections.extend(_split_block_by_numbered_steps(cleaned, section_heading=None))
    else:
        preamble = parts[0].strip()
        if preamble and len(preamble) >= 20:
            raw_sections.append(
                {
                    "heading": "Informations document",
                    "step_number": None,
                    "content": preamble,
                    "section_type": "document_header",
                }
            )
        idx = 1
        while idx < len(parts):
            h2_line = parts[idx].strip()
            body = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
            from app.services.pdf_extraction_service import _unwrap_bold_line

            section_heading = _unwrap_bold_line(re.sub(r"^##\s+", "", h2_line))
            if body:
                raw_sections.extend(
                    _split_block_by_numbered_steps(body, section_heading=section_heading)
                )
            elif section_heading:
                raw_sections.append(
                    {
                        "heading": section_heading,
                        "step_number": _step_number_from_heading(section_heading),
                        "content": section_heading,
                        "section_type": "section",
                    }
                )
            idx += 2

    specs: List[dict] = []
    for sec in raw_sections:
        content = (sec.get("content") or "").strip()
        if not content or len(content) < 8:
            continue

        node_id = str(uuid.uuid4())
        meta = dict(base_meta)
        meta.update(
            {
                "node_id": node_id,
                "parent_node_id": None,
                "hierarchy_level": 1,
                "is_leaf": "true",
                "content_type": "semantic_leaf",
                "section_type": sec.get("section_type") or "section",
                "page_no": page_no,
                "page_start": page_no,
                "page_end": page_no,
            }
        )
        heading = sec.get("heading")
        if heading:
            meta["heading"] = heading
            meta["parent_heading"] = heading
        if sec.get("step_number") is not None:
            meta["step_number"] = sec["step_number"]

        specs.append(
            {
                "chunk_index": 0,
                "is_leaf": True,
                "content": content,
                "text": content,
                "start_char": 0,
                "end_char": len(content),
                "node_id": node_id,
                "parent_node_id": None,
                "hierarchy_level": 1,
                "metadata_json": meta,
            }
        )

    return specs


def specs_to_document_chunks(document: LibraryDocument, specs: List[dict]) -> List[DocumentChunk]:
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
    return chunks
