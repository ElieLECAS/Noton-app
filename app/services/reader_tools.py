"""Les quatre outils du lecteur agentique — wrappers fins sur des fonctions existantes.

Plan ``docs/plan_lecteur_agentique_2026-09-02.md`` (phase 3). Presque aucune logique
nouvelle de retrieval : chaque outil enveloppe une fonction déjà écrite et testée, et rend
un résultat **compact** (il reste dans l'historique des appels suivants) et **explicite**
sur ce qu'il n'a pas trouvé (une page muette, une référence absente), pour que le modèle
agisse au lieu de deviner.

  * ``rechercher``        → ``search_technical_passages`` (phase A ; phase B si document_id)
  * ``lire_pages``        → texte des chunks feuilles, ou lecture déléguée en image
                            (``page_reader_service``) pour les pages sans couche texte
  * ``chercher_code``     → SQL + ``code_in_text`` + ``spec_density`` (motif de reference_pinning)
  * ``plan_du_document``  → ``profile_document`` + titres de sections des chunks

``zoomer`` a été retiré le 2026-09-12. Deux mesures l'ont condamné : sur une planche
vectorielle sans couche texte — le seul cas où il servait — ``page.search_for`` ne rend
AUCUNE ancre, donc l'ancrage par code était structurellement impossible ; et le recadrage
n'apportait rien, la page entière à 150 dpi suffisant à lire une cotation. Son rôle est
repris par la lecture déléguée de ``lire_pages``.

Les fonctions ``format_*`` sont pures (testables sans DB) ; les handlers ouvrent leur
propre ``Session`` (les appels d'un même round s'exécutent en parallèle, et une Session
SQLAlchemy n'est pas concurrente).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sqlalchemy import text as sql_text
from sqlmodel import Session

from app.config import settings
from app.database import engine
from app.models.document import Document
from app.services.page_reader_service import PageReading, read_page_image, survey_entry_for
from app.services.reader_agent_service import ToolResult, ToolSpec

logger = logging.getLogger(__name__)

# Plafonds des résultats (compacité de l'historique).
SEARCH_MAX_RESULTS = 8
SEARCH_EXTRACT_CHARS = 300
READ_MAX_PAGES = 6
READ_MAX_CHARS = 20000
CODE_MAX_EXTRACTS = 3
CODE_EXTRACT_CHAR_CAP = 1200
PLAN_MAX_SECTIONS = 60

_PLACEHOLDER_MARKERS = ("contenu visuel uniquement", "[ColPali Indexed Page")

_MODE_LABELS = {
    "full_text": "texte — le texte extrait est fiable, lis-le",
    "windowed": "texte, document long — cible des pages précises",
    "image_first": "image — pages muettes en texte, lis-les avec lire_pages (lecture sur le dessin)",
}


@dataclass
class ToolContext:
    """Ce que les outils doivent savoir du tour (périmètre, signaux, documents connus)."""

    space_id: int
    user_id: int
    allowed_document_ids: Optional[List[int]] = None
    signals: Any = None
    known_documents: Dict[int, str] = field(default_factory=dict)  # id → titre (pack)
    matched_pages_by_doc: Dict[int, List[int]] = field(default_factory=dict)
    retrieval_k: int = SEARCH_MAX_RESULTS
    image_dpi: int = 150
    # Modèle qui LIT les planches en image (appel isolé, une page, une question). Petit et
    # rapide suffit : mesuré aussi juste que le grand sur la cotation, en deux fois moins
    # de temps — c'est le cadrage de l'appel qui fait la justesse, pas la taille du modèle.
    page_reader_model: str = ""
    _space_docs: Optional[Set[int]] = None

    def __post_init__(self) -> None:
        if not self.page_reader_model:
            self.page_reader_model = settings.READER_PAGE_MODEL

    def space_document_ids(self, session: Session) -> Set[int]:
        if self._space_docs is None:
            from app.services.page_retrieval_service import get_space_document_ids

            self._space_docs = {int(d) for d in get_space_document_ids(session, self.space_id, "technical")}
        return self._space_docs


# ---------------------------------------------------------------------------
# Helpers purs
# ---------------------------------------------------------------------------


def _compact(text: str, limit: int) -> str:
    flat = re.sub(r"\s+", " ", (text or "")).strip()
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _is_placeholder(text: str) -> bool:
    t = (text or "").strip()
    return not t or any(m in t for m in _PLACEHOLDER_MARKERS)


def _coerce_int(value: Any) -> Optional[int]:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _doc_header(doc: Document) -> str:
    bits: List[str] = []
    if getattr(doc, "source", None):
        bits.append(f"Source : {doc.source}")
    if getattr(doc, "proferm_gammes", None):
        bits.append(f"Gamme : {', '.join(doc.proferm_gammes)}")
    if getattr(doc, "materials", None):
        bits.append(f"Matériau : {', '.join(doc.materials)}")
    if getattr(doc, "product_types", None):
        bits.append(f"Type : {', '.join(doc.product_types)}")
    return " | ".join(bits)


def format_search_results(passages: Sequence[Dict[str, Any]], *, document_id: Optional[int] = None) -> Tuple[str, str, Dict[int, str]]:
    """(texte pour le modèle, preuve, documents rencontrés) à partir de passages du retriever."""
    if not passages:
        where = f" dans le document {document_id}" if document_id else " dans le périmètre"
        return (
            f"Aucune page trouvée pour cette question{where}. Reformule en vocabulaire métier "
            "(pose, montage, nomenclature, réglage…), ou cherche dans un autre document.",
            "",
            {},
        )
    lines: List[str] = []
    evidence: List[str] = []
    docs: Dict[int, str] = {}
    for i, p in enumerate(passages[:SEARCH_MAX_RESULTS], start=1):
        did = _coerce_int(p.get("document_id"))
        title = str(p.get("document_title") or "Document")
        page = p.get("page_no") or p.get("page_start")
        raw = str(p.get("passage_raw") or "")
        channels = ", ".join(str(c) for c in (p.get("retrieval_sources") or [])) or "—"
        try:
            score = float(p.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if _is_placeholder(raw):
            extract = "(page sans texte extrait — planche : lis-la avec lire_pages en posant ta question)"
        else:
            extract = _compact(raw, SEARCH_EXTRACT_CHARS)
            evidence.append(extract)
        lines.append(f"{i}. doc {did} « {title} » p.{page}  {score:.3f}  [{channels}]\n   « {extract} »")
        if did is not None:
            docs[did] = title
    return "\n".join(lines), "\n".join(evidence), docs


def format_page_reading(reading: PageReading, *, needle: str = "") -> str:
    """Rendu d'une page lue EN IMAGE (planche sans couche texte).

    Le bloc dit toujours d'où vient la valeur — « lue sur l'image » — et ce qui a été lu à
    côté d'elle : une valeur sans son repère est invérifiable, et c'est précisément la
    forme que prend une lecture inventée.
    """
    head = f"[page {reading.page_no}] PLANCHE TECHNIQUE (aucun texte dans le PDF) — lue en image"
    if reading.error:
        return head + f" : lecture indisponible ({reading.error}).\nUtilise chercher_code, ou lis une autre page."
    if reading.absent:
        return (
            head
            + " : ce que tu cherches n'apparaît PAS sur cette page.\n"
            "Ne déduis rien d'un repère voisin — vérifie la référence avec chercher_code, "
            "ou cherche la bonne page (rechercher, plan_du_document)."
        )

    def _entry_line(entry: Dict[str, Any]) -> str:
        bits = []
        for v in entry.get("valeurs") or []:
            value = v.get("valeur") or "?"
            couleur = v.get("couleur") or ""
            bits.append(f"{value} ({couleur})" if couleur else value)
        return f"  relevé pour « {entry.get('repere')} » : " + " · ".join(bits)

    entry = survey_entry_for(reading.survey, needle) if needle else None

    if reading.ambiguous or not reading.answer:
        out = [
            head + " : la planche ne permet pas d'attribuer une valeur avec certitude au "
            "repère demandé."
        ]
        if entry:
            out.append(_entry_line(entry))
        out.append(
            "  → n'en choisis AUCUNE au hasard : recoupe avec chercher_code (il rend souvent "
            "la valeur associée à une référence), ou dis à l'utilisateur que la planche ne "
            "permet pas de trancher."
        )
        return "\n".join(out)

    out = [head + " :", f"  → {reading.answer}"]
    if entry:
        out.append(_entry_line(entry))
    if reading.citations:
        out.append("  repères lus sur la page : " + " · ".join(f"« {c} »" for c in reading.citations))
    return "\n".join(out)


def format_pages_text(
    *,
    document_id: int,
    title: str,
    header: str,
    page_count: int,
    pages: Sequence[int],
    text_by_page: Dict[int, str],
    question: str = "",
    readings: Optional[Dict[int, PageReading]] = None,
    needle: str = "",
) -> str:
    """Bloc texte de ``lire_pages`` : en-tête + un marqueur ``[page N]`` par page demandée.

    Une page dont le PDF ne porte AUCUN texte (planche vectorielle, scan) n'est pas rendue
    en transcription : son texte extrait ne contient que des étiquettes, jamais les cotes,
    et il sert alors de béquille — le modèle croit avoir lu la page et ne regarde plus le
    dessin. Ces pages arrivent ici déjà lues en image (``readings``).
    """
    readings = readings or {}
    out: List[str] = [f"=== doc {document_id} « {title} » ==="]
    if header:
        out.append(header)
    if page_count:
        out.append(f"Document de {page_count} page(s).")
    if question:
        out.append(f"Ta question : « {_compact(question, 200)} »")
    total = 0
    truncated = False
    for p in pages:
        if page_count and p > page_count:
            out.append(f"[page {p}] (hors du document : {page_count} pages)")
            continue
        reading = readings.get(p)
        if reading is not None:
            out.append(format_page_reading(reading, needle=needle))
            continue
        txt = (text_by_page.get(p) or "").strip()
        if not txt:
            out.append(
                f"[page {p}] (page muette — ni texte extrait ni image exploitable ; "
                "essaie une page voisine ou chercher_code)"
            )
            continue
        if total + len(txt) > READ_MAX_CHARS:
            room = max(0, READ_MAX_CHARS - total)
            txt = txt[:room].rstrip() + "\n[… texte tronqué : demande moins de pages à la fois]"
            truncated = True
        total += len(txt)
        out.append(f"[page {p}]")
        out.append(txt)
        if truncated:
            break
    return "\n".join(out)


def format_code_occurrences(
    code: str,
    rows: Sequence[Dict[str, Any]],
    *,
    n_documents: int,
) -> Tuple[str, str, Dict[int, str], List[Tuple[int, int]]]:
    """(texte, preuve, documents, pages) pour ``chercher_code``.

    ``rows`` : {"document_id", "document_title", "page_no", "content"} contenant le code
    (frontières vérifiées). Trois issues : extraits courts qui font autorité ; présence
    seulement dans des passages longs (→ lire_pages) ; absence totale — dite explicitement,
    avec le nombre de documents fouillés, pour qu'un « non » soit vérifiable."""
    code_u = code.upper()
    if not rows:
        return (
            f"{code_u} — aucun chunk ne contient cette référence dans le périmètre "
            f"({n_documents} document(s)). Si tu comptais la citer, écris que les documents "
            "ne la mentionnent pas ; ne la déduis pas d'une référence voisine.",
            "",
            {},
            [],
        )
    from app.services.reference_codes import spec_density

    ranked = sorted(
        rows,
        key=lambda r: (-spec_density(r.get("content") or ""), len(r.get("content") or ""), int(r.get("chunk_id") or 0)),
    )
    short = [r for r in ranked if len((r.get("content") or "").strip()) <= CODE_EXTRACT_CHAR_CAP]
    docs: Dict[int, str] = {}
    pages: List[Tuple[int, int]] = []
    if short:
        lines: List[str] = []
        evidence: List[str] = []
        for r in short[:CODE_MAX_EXTRACTS]:
            did = _coerce_int(r.get("document_id"))
            title = str(r.get("document_title") or "Document")
            page = r.get("page_no")
            content = _compact(r.get("content") or "", CODE_EXTRACT_CHAR_CAP)
            loc = f"doc {did} « {title} »" + (f", p.{page}" if page else "")
            lines.append(f"{code_u} — {loc}\n« {content} »")
            evidence.append(content)
            if did is not None:
                docs[did] = title
                if page:
                    pages.append((did, int(page)))
        more = len(rows) - len(short[:CODE_MAX_EXTRACTS])
        if more > 0:
            lines.append(f"(+{more} autre(s) passage(s) mentionnant {code_u})")
        return "\n".join(lines), "\n".join(evidence), docs, pages

    # Présente, mais seulement dans des passages longs : dire OÙ, laisser lire.
    by_doc: Dict[int, Dict[str, Any]] = {}
    for r in ranked:
        did = _coerce_int(r.get("document_id"))
        if did is None:
            continue
        entry = by_doc.setdefault(did, {"title": str(r.get("document_title") or "Document"), "pages": set()})
        if r.get("page_no"):
            entry["pages"].add(int(r["page_no"]))
    locs = []
    for did, entry in by_doc.items():
        docs[did] = entry["title"]
        pg = sorted(entry["pages"])
        for p in pg:
            pages.append((did, p))
        locs.append(
            f"doc {did} « {entry['title']} »" + (f" p. {', '.join(str(p) for p in pg[:8])}" if pg else "")
        )
    return (
        f"{code_u} — présente dans {len(rows)} passage(s) long(s), sans ligne de spécification "
        f"isolée : {' ; '.join(locs)}. Lis la page avec lire_pages pour voir le contexte exact.",
        "",
        docs,
        pages,
    )


def format_plan(
    *,
    document_id: int,
    title: str,
    header: str,
    page_count: int,
    pages_with_text: int,
    mode: str,
    sections: Sequence[Tuple[int, str]],
    matched_pages: Sequence[int] = (),
) -> str:
    out = [f"doc {document_id} « {title} »"]
    if header:
        out.append(header)
    out.append(
        f"{page_count} page(s), {pages_with_text} avec texte · mode conseillé : "
        f"{_MODE_LABELS.get(mode, mode)}"
    )
    if matched_pages:
        out.append("Pages retrouvées par la recherche : " + ", ".join(str(p) for p in matched_pages))
    if sections:
        out.append("Sections : " + " · ".join(f"p.{p} {_compact(h, 60)}" for p, h in sections[:PLAN_MAX_SECTIONS]))
    else:
        out.append(
            "Aucun titre de section indexé pour ce document : utilise rechercher avec ce "
            "document_id, ou lire_pages par tranches de 6 pages."
        )
    return "\n".join(out)


def _collapse_sections(rows: Sequence[Tuple[Optional[int], Optional[str]]]) -> List[Tuple[int, str]]:
    """Titres par page → liste (première page, titre), sans répétitions consécutives."""
    out: List[Tuple[int, str]] = []
    last = None
    for page, heading in rows:
        h = re.sub(r"\s+", " ", str(heading or "")).strip(" #>|-")
        if not h or page is None:
            continue
        key = h.lower()
        if key == last:
            continue
        last = key
        out.append((int(page), h))
    return out


# ---------------------------------------------------------------------------
# Accès données (une Session par appel)
# ---------------------------------------------------------------------------


def _find_code_occurrences(session: Session, document_ids: Sequence[int], code: str, *, limit: int = 200) -> List[Dict[str, Any]]:
    """Chunks feuilles portant ``code`` (frontières alphanumériques), avec document et page."""
    from app.services.reference_codes import code_in_text

    if not code or not document_ids:
        return []
    rows = session.execute(
        sql_text(
            """
            SELECT dc.id, dc.document_id, dc.content, d.title AS document_title,
                   COALESCE(
                       dc.metadata_json->>'page_no',
                       dc.metadata_json->>'page_start',
                       dc.metadata_->>'page_no',
                       dc.metadata_->>'page_start'
                   ) AS page_no
            FROM documentchunk dc
            INNER JOIN document d ON d.id = dc.document_id
            WHERE dc.document_id IN :doc_ids
              AND dc.is_leaf = true
              AND dc.content ILIKE :needle
            LIMIT :lim
            """
        ),
        {"doc_ids": tuple(int(d) for d in document_ids), "needle": f"%{code}%", "lim": limit},
    ).all()
    out: List[Dict[str, Any]] = []
    for row in rows:
        content = (row.content or "").strip()
        if not content or not code_in_text(code, content):
            continue
        out.append(
            {
                "chunk_id": int(row.id),
                "document_id": int(row.document_id),
                "document_title": row.document_title or "Document",
                "page_no": _coerce_int(row.page_no),
                "content": content,
            }
        )
    return out


def _section_rows(session: Session, document_id: int) -> List[Tuple[Optional[int], Optional[str]]]:
    rows = session.execute(
        sql_text(
            """
            SELECT COALESCE(
                       (metadata_json->>'page_no')::int,
                       (metadata_json->>'page_start')::int
                   ) AS page,
                   COALESCE(
                       metadata_json->>'heading_path',
                       metadata_json->>'parent_heading',
                       metadata_json->>'heading',
                       metadata_json->>'section'
                   ) AS heading
            FROM documentchunk
            WHERE document_id = :doc_id
              AND is_leaf = true
              AND COALESCE(
                       metadata_json->>'heading_path',
                       metadata_json->>'parent_heading',
                       metadata_json->>'heading',
                       metadata_json->>'section'
                   ) IS NOT NULL
            ORDER BY page NULLS LAST, chunk_index, id
            LIMIT 400
            """
        ),
        {"doc_id": int(document_id)},
    ).all()
    return [(r.page, r.heading) for r in rows]


def _native_text_by_page(pdf_path: Optional[str], pages: Sequence[int]) -> Dict[int, str]:
    """Texte de la COUCHE TEXTE du PDF, page par page (vide = planche vectorielle ou scan).

    C'est le discriminateur de ``lire_pages`` : il sépare un texte réellement écrit dans le
    document d'une transcription produite par la passe vision à l'ingestion. Une seule
    ouverture du PDF pour toutes les pages demandées.
    """
    if not pdf_path or not pages:
        return {}
    import fitz

    out: Dict[int, str] = {}
    try:
        with fitz.open(pdf_path) as pdf:
            for page_no in pages:
                if page_no < 1 or page_no > len(pdf):
                    continue
                try:
                    out[page_no] = pdf[page_no - 1].get_text("text").strip()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[lire_pages] couche texte p.%s illisible : %s", page_no, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[lire_pages] ouverture PDF échouée (%s) : %s", pdf_path, exc)
        return {}
    return out


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def build_reader_tools(ctx: ToolContext) -> List[ToolSpec]:
    """Les quatre outils, liés au contexte du tour."""

    async def rechercher(args: Dict[str, Any]) -> ToolResult:
        question = str(args.get("question") or "").strip()
        if not question:
            return ToolResult(text="Paramètre « question » manquant.", error=True)
        doc_id = _coerce_int(args.get("document_id"))
        from app.services import space_search_service as sss

        with Session(engine) as session:
            allowed = ctx.allowed_document_ids
            if doc_id is not None:
                if doc_id not in ctx.space_document_ids(session):
                    return ToolResult(text=f"Document {doc_id} inconnu dans cet espace.", error=True)
                allowed = [doc_id]
            result = await sss.search_technical_passages(
                session=session,
                space_id=ctx.space_id,
                query_text=question,
                user_id=ctx.user_id,
                k=ctx.retrieval_k,
                signals=ctx.signals,
                anchor_document_ids=None,
                allowed_document_ids=allowed,
            )
        if (result or {}).get("status") == "disabled":
            return ToolResult(text=f"Recherche indisponible : {result.get('reason')}.", error=True)
        text, evidence, docs = format_search_results(result.get("passages") or [], document_id=doc_id)
        return ToolResult(text=text, evidence=evidence, documents=docs)

    async def lire_pages(args: Dict[str, Any]) -> ToolResult:
        doc_id = _coerce_int(args.get("document_id"))
        if doc_id is None:
            return ToolResult(text="Paramètre « document_id » manquant.", error=True)
        question = str(args.get("question") or "").strip()
        if not question:
            return ToolResult(
                text="Paramètre « question » manquant : dis ce que tu cherches sur ces pages "
                "(une planche est lue en image, la lecture a besoin d'une question précise).",
                error=True,
            )
        raw_pages = args.get("pages") or []
        if not isinstance(raw_pages, list):
            raw_pages = [raw_pages]
        pages: List[int] = []
        for p in raw_pages:
            v = _coerce_int(p)
            if v is not None and v not in pages:
                pages.append(v)
        pages = sorted(pages)[:READ_MAX_PAGES]
        if not pages:
            return ToolResult(text="Aucune page valide demandée (entiers ≥ 1).", error=True)

        from app.services.context_packer_service import _count_document_pages, _load_leaf_records

        with Session(engine) as session:
            if doc_id not in ctx.space_document_ids(session):
                return ToolResult(text=f"Document {doc_id} inconnu dans cet espace.", error=True)
            doc = session.get(Document, doc_id)
            if doc is None:
                return ToolResult(text=f"Document {doc_id} introuvable.", error=True)
            title = doc.title or f"Document {doc_id}"
            header = _doc_header(doc)
            page_count = _count_document_pages(session, doc_id)
            records = _load_leaf_records(session, doc_id)
            pdf_path = doc.source_file_path if doc.source_file_path and os.path.exists(doc.source_file_path) else None

        valid_pages = [p for p in pages if not page_count or p <= page_count]

        # Couche texte NATIVE du PDF : c'est elle qui décide comment la page est lue.
        # Vide ⇒ planche vectorielle ou scan : le texte « extrait » de ces pages vient d'une
        # transcription vision qui ne porte JAMAIS les cotes (mesuré doc 438 p.8 : 663
        # caractères d'étiquettes, une seule valeur, celle d'une note générale — et c'est
        # cette note que le modèle a transposée à tort sur un repère).
        native_text = await asyncio.to_thread(_native_text_by_page, pdf_path, valid_pages)

        text_by_page: Dict[int, List[str]] = {}
        for page, _, txt in records:
            if page in valid_pages and txt:
                text_by_page.setdefault(page, []).append(txt)
        joined = {p: "\n".join(parts) for p, parts in text_by_page.items()}
        for p, txt in native_text.items():
            if txt and not joined.get(p):
                joined[p] = txt

        # Repère cherché : sert au recoupement couleur du lecteur de page, puis à retrouver
        # la ligne utile du relevé (le relevé entier, 30 schémas, n'a rien à faire dans
        # l'historique du lecteur).
        needle = ""
        try:
            from app.services.coverage_service import extract_message_reference_codes

            codes = extract_message_reference_codes(question)
            needle = codes[0] if codes else ""
        except Exception:  # noqa: BLE001
            needle = ""

        plate_pages = [p for p in valid_pages if not (native_text.get(p) or "").strip()]
        readings: Dict[int, PageReading] = {}
        if plate_pages and pdf_path:
            results = await asyncio.gather(
                *(
                    read_page_image(
                        pdf_path=pdf_path,
                        document_id=doc_id,
                        page_no=p,
                        question=question,
                        needle=needle,
                        model=ctx.page_reader_model,
                        dpi=ctx.image_dpi,
                    )
                    for p in plate_pages
                )
            )
            readings = {r.page_no: r for r in results}

        text = format_pages_text(
            document_id=doc_id,
            title=title,
            header=header,
            page_count=page_count,
            pages=pages,
            text_by_page=joined,
            question=question,
            readings=readings,
            needle=needle,
        )
        if plate_pages and not pdf_path:
            text += "\n(Fichier PDF source indisponible : les planches de ce document ne peuvent pas être lues.)"

        # Corpus de preuve : le texte natif des pages lisibles, et la LECTURE des planches
        # (valeur + repères). C'est ce qui rend une cote lue sur un dessin vérifiable par le
        # contrôle de sortie, au lieu de reposer sur une exemption aveugle.
        evidence_parts: List[str] = []
        for p in valid_pages:
            reading = readings.get(p)
            if reading is not None:
                if reading.ok:
                    bits = [reading.answer] + list(reading.citations)
                    evidence_parts.append(f"[page {p}] (lu sur l'image) " + " · ".join(bits))
                continue
            if joined.get(p):
                evidence_parts.append(f"[page {p}]\n{joined[p]}")

        return ToolResult(
            text=text,
            pages_read=[(doc_id, p) for p in valid_pages],
            documents={doc_id: title},
            evidence="\n".join(evidence_parts),
        )

    async def chercher_code(args: Dict[str, Any]) -> ToolResult:
        code = str(args.get("code") or "").strip()
        if len(code) < 3:
            return ToolResult(text="Paramètre « code » manquant ou trop court.", error=True)
        doc_id = _coerce_int(args.get("document_id"))
        with Session(engine) as session:
            space_docs = ctx.space_document_ids(session)
            if doc_id is not None:
                if doc_id not in space_docs:
                    return ToolResult(text=f"Document {doc_id} inconnu dans cet espace.", error=True)
                doc_ids: List[int] = [doc_id]
            elif ctx.allowed_document_ids:
                doc_ids = [int(d) for d in ctx.allowed_document_ids if int(d) in space_docs]
            else:
                doc_ids = sorted(space_docs)
            rows = _find_code_occurrences(session, doc_ids, code) if doc_ids else []
        text, evidence, docs, pages = format_code_occurrences(code, rows, n_documents=len(doc_ids))
        return ToolResult(text=text, evidence=evidence, documents=docs, pages_read=pages)

    async def plan_du_document(args: Dict[str, Any]) -> ToolResult:
        doc_id = _coerce_int(args.get("document_id"))
        if doc_id is None:
            return ToolResult(text="Paramètre « document_id » manquant.", error=True)
        from app.services.context_packer_service import profile_document

        with Session(engine) as session:
            if doc_id not in ctx.space_document_ids(session):
                return ToolResult(text=f"Document {doc_id} inconnu dans cet espace.", error=True)
            doc = session.get(Document, doc_id)
            if doc is None:
                return ToolResult(text=f"Document {doc_id} introuvable.", error=True)
            title = doc.title or f"Document {doc_id}"
            header = _doc_header(doc)
            profile = profile_document(session, doc_id)
            try:
                sections = _collapse_sections(_section_rows(session, doc_id))
            except Exception as exc:  # noqa: BLE001
                logger.debug("[plan_du_document] sections indisponibles doc=%s : %s", doc_id, exc)
                sections = []
        text = format_plan(
            document_id=doc_id,
            title=title,
            header=header,
            page_count=profile.page_count,
            pages_with_text=profile.pages_with_text,
            mode=profile.mode,
            sections=sections,
            matched_pages=ctx.matched_pages_by_doc.get(doc_id) or [],
        )
        return ToolResult(text=text, documents={doc_id: title}, evidence="")

    def _title(doc_id: Any) -> str:
        did = _coerce_int(doc_id)
        return ctx.known_documents.get(did) if did is not None and did in ctx.known_documents else f"document {doc_id}"

    def _pages_label(pages: Any) -> str:
        vals = sorted({v for v in (_coerce_int(p) for p in (pages if isinstance(pages, list) else [pages])) if v})
        if not vals:
            return ""
        if len(vals) == 1:
            return f"page {vals[0]}"
        if vals == list(range(vals[0], vals[-1] + 1)):
            return f"pages {vals[0]}-{vals[-1]}"
        return "pages " + ", ".join(str(v) for v in vals)

    return [
        ToolSpec(
            name="rechercher",
            schema=TOOL_SCHEMAS["rechercher"],
            handler=rechercher,
            label=lambda a: "Recherche : « " + _compact(str(a.get("question") or ""), 70) + " »"
            + (f" dans {_title(a.get('document_id'))}" if a.get("document_id") else ""),
        ),
        ToolSpec(
            name="chercher_code",
            schema=TOOL_SCHEMAS["chercher_code"],
            handler=chercher_code,
            label=lambda a: f"Vérification de la référence {str(a.get('code') or '').upper()}",
        ),
        ToolSpec(
            name="plan_du_document",
            schema=TOOL_SCHEMAS["plan_du_document"],
            handler=plan_du_document,
            label=lambda a: f"Plan de « {_compact(_title(a.get('document_id')), 50)} »",
        ),
    ]


# ---------------------------------------------------------------------------
# Schémas (format Mistral / OpenAI function calling)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "rechercher": {
        "type": "function",
        "function": {
            "name": "rechercher",
            "description": (
                "Cherche des pages dans les documents de l'espace (ColPali + BM25). Sans "
                "document_id : tout le périmètre (coûteux, ne répète pas la même question). "
                "Avec document_id : recherche fine dans ce document. Rend au plus 8 pages avec "
                "un extrait court et les canaux qui ont matché ([colpali] seul = page "
                "probablement sans texte : demande son image)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Formulation autonome en vocabulaire métier (pose, montage, nomenclature, rallonge, crémone…).",
                    },
                    "document_id": {
                        "type": ["integer", "null"],
                        "description": "Restreindre à un document (identifiant « doc N » du contexte).",
                    },
                },
                "required": ["question"],
            },
        },
    },
    "lire_pages": {
        "type": "function",
        "function": {
            "name": "lire_pages",
            "description": (
                "Lit des pages précises d'un document et répond à TA question sur ces pages. "
                "Une page avec du texte est rendue telle quelle ; une PLANCHE (dessin sans "
                "texte : coupe, nomenclature cotée) est lue directement sur l'image, et la "
                "réponse cite le repère lu à côté de la valeur. Une planche dit explicitement "
                "quand ce que tu cherches n'y figure pas — ne déduis alors rien d'un repère "
                "voisin. C'est l'outil à utiliser pour toute COTE (épaisseur, section, entraxe)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer"},
                    "pages": {"type": "array", "items": {"type": "integer"}, "maxItems": READ_MAX_PAGES},
                    "question": {
                        "type": "string",
                        "description": (
                            "Ce que tu cherches sur ces pages, en une phrase précise et "
                            "autonome (ex. « épaisseur de vitrage de la parclose 2636 »)."
                        ),
                    },
                },
                "required": ["document_id", "pages", "question"],
            },
        },
    },
    "chercher_code": {
        "type": "function",
        "function": {
            "name": "chercher_code",
            "description": (
                "Vérifie si une référence produit exacte existe dans les documents et rend les "
                "extraits qui la portent (3 max, avec page). Rend explicitement l'absence. À "
                "appeler avant de citer toute référence absente du contexte."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "document_id": {"type": ["integer", "null"]},
                },
                "required": ["code"],
            },
        },
    },
    "plan_du_document": {
        "type": "function",
        "function": {
            "name": "plan_du_document",
            "description": (
                "Profil d'un document (pages, pages avec texte, mode de lecture conseillé) et, "
                "quand ils existent, ses titres de sections par page, plus les pages déjà "
                "retrouvées par la recherche."
            ),
            "parameters": {
                "type": "object",
                "properties": {"document_id": {"type": "integer"}},
                "required": ["document_id"],
            },
        },
    },
}
