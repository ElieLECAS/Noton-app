"""
Extraction de chunks documentaires via la couche texte native (pymupdf4llm).

Voie « texte brut », alternative à ``vision_page_extraction_service`` :
  1. pymupdf4llm.to_markdown() sur TOUT le document en UN passage (page_chunks=True)
  2. Par page : séparation des blocs tableau et des blocs prose
  3. Tableaux : une ligne = un chunk autosuffisant (en-têtes réinjectés), + un chunk
     « tableau complet » pour le contexte — ou un seul chunk atomique selon le mode
  4. Prose : découpe structurelle (titres ##, étapes numérotées), cap tokens en filet

Pourquoi cette voie : sur une page dotée d'une couche texte, les chiffres et les
références sont lus dans les objets texte du PDF — exacts par construction, sans
transcription par un modèle. Les tableaux sortent entiers (pas de plafond de tokens
de sortie LLM). Coût nul, déterministe, réingestion quasi instantanée.

Limite : ne produit rien sur une page sans couche texte (scan, texte vectorisé) ;
l'appelant bascule alors sur la voie vision.

IMPORTANT : ``clean_pymupdf4llm_markdown`` convertit les ``<br>`` en sauts de ligne,
ce qui casse les lignes de tableau. Le nettoyage est donc appliqué UNIQUEMENT aux
segments de prose ; les cellules de tableau voient leurs ``<br>`` remplacés par des
espaces.

CHUNKING_VERSION = "text_page_v1"
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

CHUNKING_VERSION = "text_page_v1"
EXTRACTION_PROVIDER_TEXT = "pymupdf4llm_text"

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HEADING_RE = re.compile(r"(?m)^(#{1,6})\s+(.+)$")

# Nombre minimal de caractères pour qu'une page soit jugée exploitable en voie texte.
# Aligné sur MULTIMODAL_NATIVE_TEXT_MIN_CHARS (défaut 100).
MIN_PAGE_CHARS = 100


# ---------------------------------------------------------------------------
# Extraction brute (tableaux préservés)
# ---------------------------------------------------------------------------


# Balisage markdown / HTML injecté par pymupdf4llm. Il DOIT être retiré avant toute
# comparaison : `<sup>` laisse les lettres « sup » dans le texte normalisé, et un
# `TEXTURAL®` du PDF ne correspond alors plus à `TEXTURAL**<sup>®</sup>` du markdown.
# Conséquence observée le 2026-07-28 : des paragraphes DÉJÀ présents étaient déclarés
# perdus, puis réinjectés en vrac dans « Éléments hors flux ».
_MARKUP_RE = re.compile(r"</?(?:sup|sub|br|b|i|em|strong)\s*/?>", re.IGNORECASE)


def _normalize_for_compare(text: str) -> str:
    """Forme comparable : balisage retiré, minuscules, sans espaces ni ponctuation."""
    cleaned = _MARKUP_RE.sub("", text or "")
    return re.sub(r"[^a-z0-9]+", "", cleaned.lower())


def extract_page_raw_text(pdf_path: str) -> Dict[int, str]:
    """Texte natif par page, un BLOC de mise en page par ligne.

    ``get_text("blocks")`` regroupe les lignes d'un même bloc de mise en page ; les
    retours à la ligne internes (justification) sont donc recollés. ``get_text("text")``
    rendait des lignes PHYSIQUES : une phrase justifiée sur trois lignes produisait trois
    fragments (« Pivot pouvant » / « supporter le poids » / « d'une fenêtre jusqu'à »),
    qui remontaient ensuite en trois puces absurdes.
    """
    import fitz

    out: Dict[int, str] = {}
    doc = fitz.open(pdf_path)
    try:
        for idx, page in enumerate(doc):
            try:
                blocks = page.get_text("blocks") or []
            except Exception:  # noqa: BLE001
                out[idx + 1] = page.get_text("text") or ""
                continue
            lines = []
            for block in blocks:
                # (x0, y0, x1, y1, texte, block_no, block_type)
                if len(block) < 5 or block[4] is None:
                    continue
                flat = " ".join(str(block[4]).split())
                if flat:
                    lines.append(flat)
            out[idx + 1] = "\n".join(lines)
    finally:
        doc.close()
    return out


def recover_lost_lines(markdown: str, raw_text: str) -> List[str]:
    """Lignes présentes dans la couche texte MAIS absentes du markdown pymupdf4llm.

    pymupdf4llm classe comme « image » le texte posé sur un visuel et le supprime
    quand les images ne sont pas écrites. Mesuré le 2026-07-28 sur une plaquette
    commerciale : **12 % des lignes** disparaissaient ainsi — libellés de nuanciers
    (« Les teintés dans la masse »), codes RAL, épaisseurs (« 10 mm », « 18 mm »),
    titres d'encarts (« ACCESSOIRES »).

    Ces lignes sont du texte NATIF, donc exact : les perdre est inacceptable pour un
    extracteur dont l'argument est justement l'exactitude. On les récupère telles
    quelles, en conservant l'ordre de lecture du PDF.
    """
    if not raw_text:
        return []

    haystack = _normalize_for_compare(markdown)
    lost: List[str] = []
    seen: set[str] = set()

    for raw_line in _join_wrapped_fragments(raw_text.splitlines()):
        line = raw_line.strip()
        if len(line) < 3:
            continue
        needle = _normalize_for_compare(line)
        if not needle or needle in seen:
            continue
        if needle in haystack:
            continue
        seen.add(needle)
        lost.append(line)

    return lost


def _join_wrapped_fragments(lines: List[str]) -> List[str]:
    """Recolle les fragments d'une même phrase répartis sur plusieurs blocs.

    Les libellés d'encart d'une plaquette sont des boîtes de texte distinctes : une
    phrase courte s'y retrouve coupée (« Pivot pouvant » / « supporter le poids » /
    « d'une fenêtre jusqu'à » / « 130kg. »). Chacune deviendrait une puce séparée et
    illisible. On fusionne un fragment avec le suivant quand il ne se termine PAS par
    une ponctuation forte et que le suivant ne commence pas par une majuscule — la même
    heuristique que le recollage inter-pages.
    """
    out: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if out:
            prev = out[-1]
            prev_open = not prev.endswith((".", ":", "!", "?", ";", "»"))
            starts_low = line[0].islower() or line[0].isdigit()
            # Ne recoller que des fragments COURTS : deux vrais paragraphes qui se
            # suivent ne doivent pas fusionner.
            if prev_open and starts_low and len(prev) < 120 and len(line) < 120:
                out[-1] = f"{prev} {line}"
                continue
        out.append(line)
    return out


def extract_document_pages_markdown_raw(pdf_path: str) -> List[Tuple[int, str]]:
    """
    Markdown pymupdf4llm page par page, SANS nettoyage destructeur de tableaux.

    Un seul appel ``to_markdown`` pour tout le document (l'ancien fallback par page
    ré-extrayait le PDF entier à chaque page — coût quadratique).
    """
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
        write_images=False,
        show_progress=False,
    )
    pages: List[Tuple[int, str]] = []
    if not isinstance(chunks, list):
        return pages

    for i, chunk in enumerate(chunks):
        page_text = ""
        page_num = i + 1
        if isinstance(chunk, dict):
            page_text = (chunk.get("text") or "").strip()
            meta = chunk.get("metadata")
            if isinstance(meta, dict):
                p_num = meta.get("page_number") or meta.get("page")
                if p_num is not None:
                    try:
                        page_num = int(p_num)
                    except (TypeError, ValueError):
                        pass
        elif isinstance(chunk, str):
            page_text = chunk.strip()
        pages.append((page_num, page_text))
    return pages


# ---------------------------------------------------------------------------
# Découpe tableau / prose
# ---------------------------------------------------------------------------


def _split_table_prose_segments(markdown: str) -> List[Tuple[str, str]]:
    """
    Segmente une page en blocs ("table"|"prose", texte), dans l'ordre de lecture.
    """
    from app.services.chunking_service import _find_markdown_table_spans

    if not markdown or not markdown.strip():
        return []

    spans = _find_markdown_table_spans(markdown)
    if not spans:
        return [("prose", markdown)]

    segments: List[Tuple[str, str]] = []
    cursor = 0
    for start, end, table_text in spans:
        before = markdown[cursor:start]
        if before.strip():
            segments.append(("prose", before))
        segments.append(("table", table_text))
        cursor = end
    tail = markdown[cursor:]
    if tail.strip():
        segments.append(("prose", tail))
    return segments


def _table_br_to_space(table_text: str) -> str:
    """Remplace les <br> par des espaces (préserve l'intégrité des lignes de tableau)."""
    return _BR_RE.sub(" ", table_text or "")


def _last_heading(text: str) -> Optional[str]:
    """Dernier titre markdown rencontré dans un bloc de prose (contexte des tableaux)."""
    from app.services.pdf_extraction_service import _unwrap_bold_line

    matches = _HEADING_RE.findall(text or "")
    if not matches:
        return None
    return _unwrap_bold_line(matches[-1][1])


# ---------------------------------------------------------------------------
# Construction des specs
# ---------------------------------------------------------------------------


def _make_spec(content: str, meta: dict) -> dict:
    node_id = meta.get("node_id") or str(uuid.uuid4())
    meta["node_id"] = node_id
    return {
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


def _base_leaf_meta(
    metadata_base: dict,
    page_no: int,
    section_type: str,
    heading: Optional[str],
) -> dict:
    meta = dict(metadata_base)
    meta.update(
        {
            "chunking_version": CHUNKING_VERSION,
            "extraction_provider": EXTRACTION_PROVIDER_TEXT,
            "content_type": "semantic_leaf",
            "section_type": section_type,
            "page_no": page_no,
            "page_start": page_no,
            "page_end": page_no,
            "is_leaf": True,
            "parent_node_id": None,
        }
    )
    if heading:
        meta["heading"] = heading
        meta["parent_heading"] = heading
    return meta


def _promote_header_row(
    headers: List[str], data_rows: List[List[str]]
) -> Tuple[List[str], List[List[str]], Optional[str]]:
    """
    Corrige le cas « ligne de titre prise pour les en-têtes ».

    pymupdf4llm rend fréquemment un tableau dont la PREMIÈRE ligne est un titre
    fusionné sur plusieurs colonnes (cellules vides) et dont les vrais en-têtes sont
    sur la ligne suivante. Observé sur un DTA réel :

        |Tableau 1 - Compositions vinyliques||Comosition|      <- pris pour en-têtes
        |REFERENCE Matière (Certificat QB34)|CODE CSTB|Coloris| <- vrais en-têtes

    Sans correction, chaque chunk-ligne porte ``col1:Tableau 1 - Compositions
    vinyliques=4038-699`` au lieu de ``col1:REFERENCE Matière=4038-699``.

    Ne se déclenche QUE si la ligne d'en-têtes est visiblement défectueuse (au moins
    une cellule vide) et que la première ligne de données est complète.

    Returns:
        (headers, data_rows, caption) — caption = ancienne ligne d'en-têtes si promue.
    """
    if not data_rows:
        return headers, data_rows, None
    if all(h.strip() for h in headers):
        return headers, data_rows, None

    candidate = data_rows[0]
    if not all(c.strip() for c in candidate):
        return headers, data_rows, None

    caption = " ".join(h.strip() for h in headers if h.strip()).strip() or None
    logger.info(
        "[TextExtract] en-têtes promus depuis la 1re ligne de données (ligne de titre détectée) : %s",
        candidate,
    )
    return candidate, data_rows[1:], caption


def _build_table_specs(
    table_text: str,
    page_no: int,
    metadata_base: dict,
    parent_heading: Optional[str],
) -> List[dict]:
    """
    Chunks d'un tableau markdown.

    Mode "rows" (défaut) : une ligne = un chunk autosuffisant portant ses en-têtes
    de colonnes, plus un chunk « tableau complet » pour le contexte d'ensemble.
    Une question sur une référence précise matche alors UNE ligne, au lieu d'être
    diluée dans plusieurs centaines de tokens.

    Mode "atomic" : le tableau entier en un seul chunk (jamais coupé au token).
    """
    from app.services.chunking_service import (
        _build_table_json,
        _parse_markdown_table_robust,
        _serialize_markdown_table,
        _table_full_chunk_text,
        _table_row_chunk_text,
    )

    normalized = _table_br_to_space(table_text)
    parsed = _parse_markdown_table_robust(normalized)
    if parsed is None:
        # Grille non parsable : on conserve le bloc brut plutôt que de le perdre.
        content = normalized.strip()
        if not content:
            return []
        meta = _base_leaf_meta(metadata_base, page_no, "table", parent_heading)
        return [_make_spec(content, meta)]

    headers, data_rows, caption = _promote_header_row(parsed.headers, parsed.data_rows)
    if not data_rows:
        return []
    table_id = str(uuid.uuid4())
    mode = (settings.TEXT_EXTRACTION_TABLE_MODE or "rows").strip().lower()

    table_json = _build_table_json(
        headers, data_rows, caption, page_no, parsed.suspicious_row_indices
    )
    if parsed.suspicious_row_indices:
        logger.info(
            "[TextExtract] page %s — tableau : %d ligne(s) suspecte(s) (colonnes décalées)",
            page_no,
            len(parsed.suspicious_row_indices),
        )

    specs: List[dict] = []

    if mode == "atomic":
        content = _table_full_chunk_text(
            headers=headers,
            data_rows=data_rows,
            parent_heading=parent_heading or "",
            caption=caption,
            page_no=page_no,
        )
        meta = _base_leaf_meta(metadata_base, page_no, "table", parent_heading)
        meta["table_id"] = table_id
        meta["table_json"] = table_json
        return [_make_spec(content, meta)]

    # Mode "rows"
    total = len(data_rows)
    for row_idx, cells in enumerate(data_rows):
        content = _table_row_chunk_text(
            headers=headers,
            cells=cells,
            parent_heading=parent_heading or "",
            caption=caption,
            page_no=page_no,
            table_id=table_id,
            row_index=row_idx,
            total_rows=total,
            suspicious=row_idx in parsed.suspicious_row_indices,
            empty_col_indices=parsed.empty_cell_map.get(row_idx),
        )
        if not content.strip():
            continue
        meta = _base_leaf_meta(metadata_base, page_no, "table_row", parent_heading)
        meta["table_id"] = table_id
        meta["row_index"] = row_idx
        meta["column_headers"] = headers
        specs.append(_make_spec(content, meta))

    if settings.TEXT_EXTRACTION_TABLE_FULL_CHUNK and specs:
        full_content = _table_full_chunk_text(
            headers=headers,
            data_rows=data_rows,
            parent_heading=parent_heading or "",
            caption=caption,
            page_no=page_no,
        )
        meta = _base_leaf_meta(metadata_base, page_no, "table", parent_heading)
        meta["table_id"] = table_id
        meta["table_json"] = table_json
        specs.append(_make_spec(full_content, meta))
    elif not specs:
        # Aucune ligne exploitable : on retombe sur le markdown canonique.
        content = _serialize_markdown_table(headers, data_rows)
        meta = _base_leaf_meta(metadata_base, page_no, "table", parent_heading)
        meta["table_id"] = table_id
        meta["table_json"] = table_json
        specs.append(_make_spec(content, meta))

    return specs


def _split_prose_into_sections(text: str) -> List[dict]:
    """
    Découpe un bloc de prose en SECTIONS : un titre markdown et son corps.

    Reconnaît tous les niveaux de titre (``#`` à ``######``), y compris ceux
    enveloppés de gras (``###### **2.2.3. Eléments**``). L'ancienne découpe ne
    matchait que ``##`` exactement : sur un DTA titré en ``######``, aucune section
    n'était détectée et la page entière devenait UN chunk « document_header » de
    800+ tokens sans titre.

    Le texte précédant le premier titre forme une section sans titre (suite de la
    page précédente ou en-tête de page).
    """
    matches = list(_HEADING_RE.finditer(text or ""))
    if not matches:
        body = (text or "").strip()
        return [{"heading": None, "level": 0, "body": body}] if body else []

    sections: List[dict] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append({"heading": None, "level": 0, "body": preamble})

    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        heading = _clean_heading(match.group(2))
        body = text[match.end() : end].strip()
        if not heading and not body:
            continue
        sections.append(
            {"heading": heading, "level": len(match.group(1)), "body": body}
        )

    return sections


def _clean_heading(raw: str) -> str:
    """Titre nettoyé : gras markdown retiré, y compris NON APPARIÉ.

    pymupdf4llm produit fréquemment des titres à gras ouvert mais non fermé
    (``**INTÉRIEUR ET EXTÉRIEUR PVC``) quand la mise en gras déborde du titre :
    ``_unwrap_bold_line`` exige les deux délimiteurs et les laissait tels quels.
    """
    from app.services.pdf_extraction_service import _unwrap_bold_line

    cleaned = _unwrap_bold_line((raw or "").strip())
    return cleaned.strip("*").strip()


def _step_number(heading: Optional[str]) -> Optional[int]:
    if not heading:
        return None
    match = re.match(r"^(\d{1,2})[.)]\s", heading.strip())
    return int(match.group(1)) if match else None


def _build_prose_specs(
    prose_text: str,
    page_no: int,
    metadata_base: dict,
) -> List[dict]:
    """
    Chunks de prose : une SECTION = un chunk.

    Le cap de tokens n'est pas le critère de découpe, c'est un filet : une section
    qui tient sous ``TEXT_EXTRACTION_MAX_CHUNK_TOKENS`` reste entière. Au-delà, elle
    est scindée d'abord sur les étapes numérotées, puis au token — et le titre de la
    section est REPRIS en tête de chaque morceau pour qu'aucun fragment ne devienne
    orphelin de son contexte.
    """
    from app.services.multimodal_page_service import count_tokens, split_text_by_tokens
    from app.services.pdf_extraction_service import clean_pymupdf4llm_markdown

    cleaned = clean_pymupdf4llm_markdown(prose_text, page_no=page_no)
    if not cleaned:
        return []

    max_tokens = settings.TEXT_EXTRACTION_MAX_CHUNK_TOKENS
    specs: List[dict] = []

    for section in _split_prose_into_sections(cleaned):
        heading = section["heading"]
        body = section["body"]
        content = f"{heading}\n\n{body}".strip() if heading else body
        if not content or len(content) < 8:
            continue

        section_type = "section" if heading else "document_header"
        step = _step_number(heading)

        if max_tokens <= 0 or count_tokens(content) <= max_tokens:
            parts = [content]
        else:
            # Section trop longue : on scinde, en réinjectant le titre dans chaque part.
            raw_parts = [
                p for p in split_text_by_tokens(body or content, max_tokens=max_tokens)
                if p.strip()
            ]
            parts = [
                f"{heading}\n\n{p}".strip() if heading else p for p in raw_parts
            ] or [content]

        for part_idx, part in enumerate(parts):
            meta = _base_leaf_meta(metadata_base, page_no, section_type, heading)
            if step is not None:
                meta["step_number"] = step
            meta["token_count"] = count_tokens(part)
            if len(parts) > 1:
                meta["section_part"] = part_idx + 1
                meta["section_parts_total"] = len(parts)
            # Section sans titre en tête de page = très probablement la suite de la
            # page précédente : marqueur exploité par le recollage inter-pages.
            if heading is None and part_idx == 0:
                meta["orphan_section_start"] = True
            specs.append(_make_spec(part, meta))

    return specs


# ---------------------------------------------------------------------------
# Points d'entrée
# ---------------------------------------------------------------------------


def _normalize_for_dedup(text: str) -> str:
    """Forme comparable d'un contenu : minuscules, espaces et ponctuation écrasés."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _drop_duplicate_specs(specs: List[dict]) -> List[dict]:
    """Écarte les chunks dont le contenu est déjà couvert par un autre de la page.

    pymupdf4llm émet un bloc « picture text » par image : deux images qui se recouvrent
    (fréquent sur les plaquettes commerciales) produisent DEUX fois le même texte. Sans
    dédup, la même notion est indexée en double, ce qui la fait remonter deux fois et
    gonfle artificiellement le score d'élection de son document.

    On garde le chunk le plus LONG (le plus complet) et on écarte ceux dont le texte
    normalisé y est entièrement contenu.
    """
    if len(specs) < 2:
        return specs

    order = sorted(
        range(len(specs)), key=lambda i: len(specs[i].get("content") or ""), reverse=True
    )
    kept_indices: List[int] = []
    kept_norms: List[str] = []

    for idx in order:
        norm = _normalize_for_dedup(specs[idx].get("content"))
        if not norm:
            continue
        if any(norm in bigger for bigger in kept_norms):
            logger.info(
                "[TextExtract] chunk doublon écarté (contenu déjà couvert) : %r",
                (specs[idx].get("content") or "")[:60],
            )
            continue
        kept_norms.append(norm)
        kept_indices.append(idx)

    # Ordre de lecture d'origine restauré.
    return [specs[i] for i in sorted(kept_indices)]


def extract_page_chunk_specs_text(
    page_markdown: str,
    page_no: int,
    metadata_base: dict,
    raw_text: str = "",
) -> List[dict]:
    """Specs d'une page à partir de son markdown pymupdf4llm brut.

    ``raw_text`` (couche texte native de la page) sert à RÉCUPÉRER ce que
    pymupdf4llm a perdu — voir recover_lost_lines.
    """
    if not page_markdown or len(page_markdown.strip()) < MIN_PAGE_CHARS:
        return []

    lost = recover_lost_lines(page_markdown, raw_text) if raw_text else []
    if lost:
        logger.info(
            "[TextExtract] page %s — %d ligne(s) récupérée(s) hors flux pymupdf4llm",
            page_no,
            len(lost),
        )
        recovered = "\n".join(f"- {line}" for line in lost)
        page_markdown = (
            f"{page_markdown.rstrip()}\n\n#### Éléments hors flux\n{recovered}\n"
        )

    segments = _split_table_prose_segments(page_markdown)
    if not segments:
        return []

    specs: List[dict] = []
    current_heading: Optional[str] = None

    for kind, text in segments:
        if kind == "prose":
            heading = _last_heading(text)
            specs.extend(_build_prose_specs(text, page_no, metadata_base))
            if heading:
                current_heading = heading
        else:
            specs.extend(
                _build_table_specs(text, page_no, metadata_base, current_heading)
            )

    specs = _drop_duplicate_specs(specs)

    max_chunks = settings.TEXT_EXTRACTION_MAX_CHUNKS_PER_PAGE
    if max_chunks > 0 and len(specs) > max_chunks:
        logger.warning(
            "[TextExtract] page %s — %d chunks produits, tronqué à %d "
            "(TEXT_EXTRACTION_MAX_CHUNKS_PER_PAGE) : %d chunk(s) PERDU(S)",
            page_no,
            len(specs),
            max_chunks,
            len(specs) - max_chunks,
        )
        specs = specs[:max_chunks]

    return specs


def _merge_sections_across_pages(specs_by_page: Dict[int, List[dict]]) -> int:
    """
    Recolle une section coupée par un saut de page.

    Un saut de page ne coïncide presque jamais avec une frontière de section : la
    page N+1 commence alors par du texte sans titre (marqué ``orphan_section_start``),
    qui est la suite directe du dernier chunk de la page N.

    Le recollage n'a lieu que si le résultat reste sous
    ``TEXT_EXTRACTION_MAX_CHUNK_TOKENS`` × 1.6 — sinon on garde deux chunks (l'objectif
    « sections entières » ne doit pas fabriquer les chunks kilométriques qu'on cherche
    justement à éviter), mais la continuité reste signalée en métadonnée.

    Returns:
        Nombre de recollages effectués.
    """
    from app.services.multimodal_page_service import count_tokens

    max_tokens = settings.TEXT_EXTRACTION_MAX_CHUNK_TOKENS
    ceiling = int(max_tokens * 1.6) if max_tokens > 0 else 0
    merged = 0

    for page_no in sorted(specs_by_page):
        nxt = specs_by_page.get(page_no + 1)
        cur = specs_by_page.get(page_no)
        if not cur or not nxt:
            continue

        head = nxt[0]
        head_meta = head.get("metadata_json") or {}
        if not head_meta.get("orphan_section_start"):
            continue

        tail = cur[-1]
        tail_meta = tail.get("metadata_json") or {}
        # Ne jamais recoller sur un tableau : ses lignes sont des unités atomiques.
        if str(tail_meta.get("section_type", "")).startswith("table"):
            continue

        combined = f"{tail['content'].rstrip()}\n\n{head['content'].lstrip()}".strip()
        if ceiling and count_tokens(combined) > ceiling:
            head_meta["continues_from_previous_page"] = True
            tail_meta["continues_on_next_page"] = True
            continue

        tail["content"] = combined
        tail["text"] = combined
        tail["end_char"] = len(combined)
        tail_meta["page_end"] = page_no + 1
        tail_meta["cross_page_merge"] = True
        tail_meta["merged_pages"] = [page_no, page_no + 1]
        tail_meta["token_count"] = count_tokens(combined)
        nxt.pop(0)
        merged += 1

    return merged


def extract_document_chunk_specs_text(
    pdf_path: str,
    metadata_base: dict,
) -> Tuple[Dict[int, List[dict]], List[int]]:
    """
    Specs de tout le document par la voie texte.

    Returns:
        (specs_par_page, pages_sans_texte) — les pages sans couche texte exploitable
        sont retournées à part pour que l'appelant les traite en vision.
    """
    pages = extract_document_pages_markdown_raw(pdf_path)
    # Couche texte native : sert de FILET par rapport au markdown pymupdf4llm, qui
    # perd le texte posé sur les visuels (12 % des lignes sur une plaquette mesurée).
    try:
        raw_by_page = extract_page_raw_text(pdf_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TextExtract] texte natif indisponible (%s) — pas de filet", exc)
        raw_by_page = {}

    specs_by_page: Dict[int, List[dict]] = {}
    pages_without_text: List[int] = []

    for page_no, markdown in pages:
        try:
            specs = extract_page_chunk_specs_text(
                markdown, page_no, metadata_base, raw_text=raw_by_page.get(page_no, "")
            )
        except Exception as exc:
            logger.error(
                "[TextExtract] page %s — extraction échouée : %s", page_no, exc
            )
            specs = []
        if specs:
            specs_by_page[page_no] = specs
        else:
            pages_without_text.append(page_no)

    merged = _merge_sections_across_pages(specs_by_page)

    # NE PAS retirer les pages devenues vides après recollage : leur clé doit rester
    # présente, sinon l'appelant les croirait « sans couche texte » et les
    # ré-extrairait en vision — dupliquant un contenu déjà absorbé par la page
    # précédente.
    pages_extraites = sum(1 for s in specs_by_page.values() if s)

    logger.info(
        "[TextExtract] %d page(s) en texte natif, %d page(s) sans texte exploitable, "
        "%d section(s) recollée(s) entre pages",
        pages_extraites,
        len(pages_without_text),
        merged,
    )
    return specs_by_page, pages_without_text
