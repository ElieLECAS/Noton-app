"""Markdown augmenté : une section par page, appariée au PNG de la même page.

Protocole : ``docs/protocole_markdown_augmente_2026-09-16.md``.

Ce que ce service fait, et rien d'autre :
  * **parser** un markdown augmenté de façon DÉTERMINISTE (aucun modèle, aucun appel) ;
  * le **valider** contre le PDF source (nombre de pages, doublons, trous, hachage) ;
  * le **stocker** dans ``media/page_markdown/<document_id>.md`` ;
  * le **servir** par page au packer, avec les conventions du document.

Pourquoi un fichier plutôt qu'une table : le markdown est écrit et corrigé à la main.
Un fichier se diffe, se relit, se corrige page par page et se réimporte — c'est le cycle
de travail réel. La base n'apporterait rien qu'un cache TTL ne donne déjà.

Séparateur reconnu (cf. protocole § 1.3) ::

    <!-- PageBreak page_pdf="8" page_imprimee="5" -->

``page_imprimee`` est facultatif. ``<!-- PageBreak -->`` nu est REFUSÉ : un séparateur
sans numéro ne survit ni à une troncature ni à une concaténation partielle, et
l'appariement au PNG a besoin d'un identifiant.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

PAGE_MARKDOWN_DIR = Path("media/page_markdown")

# <!-- PageBreak page_pdf="8" page_imprimee="5" -->  (page_imprimee facultatif)
_SEPARATOR_RE = re.compile(
    r"^[ \t]*<!--\s*PageBreak\s+"
    r"page_pdf\s*=\s*\"(?P<pdf>\d+)\""
    r"(?:\s+page_imprimee\s*=\s*\"(?P<printed>[^\"]*)\")?"
    r"\s*-->[ \t]*$",
    re.MULTILINE,
)
# Séparateur sans numéro : détecté pour pouvoir l'EXPLIQUER, jamais accepté.
_BARE_SEPARATOR_RE = re.compile(r"^[ \t]*<!--\s*PageBreak\s*-->[ \t]*$", re.MULTILINE)

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(?P<body>.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_CONVENTIONS_RE = re.compile(
    r"^##[ \t]+Conventions[^\n]*\n(?P<body>.*?)(?=^<!--\s*PageBreak|^##[ \t]|\Z)",
    re.MULTILINE | re.DOTALL,
)
_TITLE_RE = re.compile(r"^##[ \t]+(?P<title>[^\n]+)$", re.MULTILINE)


@dataclass
class PageSection:
    """Une page du markdown augmenté."""

    page_pdf: int
    page_imprimee: Optional[str]
    title: str
    body: str

    @property
    def char_count(self) -> int:
        return len(self.body)


@dataclass
class ParsedMarkdown:
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    conventions: str = ""
    pages: List[PageSection] = field(default_factory=list)

    @property
    def page_numbers(self) -> List[int]:
        return [p.page_pdf for p in self.pages]


@dataclass
class ValidationReport:
    """Verdict de l'import. ``ok`` faux ⇒ rien n'est écrit."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    pages_found: int = 0
    pages_expected: int = 0
    pages_vides: List[int] = field(default_factory=list)
    conventions_presentes: bool = False
    hash_source_ok: Optional[bool] = None
    apercu: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "pages_found": self.pages_found,
            "pages_expected": self.pages_expected,
            "pages_vides": self.pages_vides,
            "conventions_presentes": self.conventions_presentes,
            "hash_source_ok": self.hash_source_ok,
            "apercu": self.apercu,
        }


# ---------------------------------------------------------------------------
# Parsing — déterministe, sans modèle
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Front-matter YAML de tête (facultatif) + le reste du document."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw = m.group("body")
    data: Dict[str, Any] = {}
    try:
        import yaml

        loaded = yaml.safe_load(raw)
        if isinstance(loaded, dict):
            data = loaded
    except Exception as exc:  # noqa: BLE001 - front-matter illisible n'est pas bloquant
        logger.warning("[PageMarkdown] front-matter illisible (%s) — ignoré", exc)
    return data, text[m.end():]


def parse_markdown(text: str) -> ParsedMarkdown:
    """Découpe un markdown augmenté en sections de page. Ne lève jamais."""
    text = (text or "").replace("\r\n", "\n")
    frontmatter, body = _parse_frontmatter(text)

    conv_match = _CONVENTIONS_RE.search(body)
    conventions = conv_match.group("body").strip() if conv_match else ""

    matches = list(_SEPARATOR_RE.finditer(body))
    pages: List[PageSection] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        chunk = body[start:end].strip("\n")

        title_match = _TITLE_RE.search(chunk)
        title = title_match.group("title").strip() if title_match else ""

        printed = (m.group("printed") or "").strip() or None
        pages.append(
            PageSection(
                page_pdf=int(m.group("pdf")),
                page_imprimee=printed,
                title=title,
                body=chunk.strip(),
            )
        )

    return ParsedMarkdown(frontmatter=frontmatter, conventions=conventions, pages=pages)


# ---------------------------------------------------------------------------
# Validation — c'est ici que l'import se refuse
# ---------------------------------------------------------------------------


def sha256_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[PageMarkdown] hachage impossible (%s) : %s", path, exc)
        return None


def pdf_page_count(pdf_path: str) -> Optional[int]:
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[PageMarkdown] comptage de pages impossible (%s) : %s", pdf_path, exc)
        return None


def validate(
    text: str,
    parsed: ParsedMarkdown,
    *,
    expected_pages: Optional[int],
    source_sha256: Optional[str],
) -> ValidationReport:
    """Confronte le markdown au PDF. Toute ERREUR bloque l'import.

    La règle non négociable est le comptage : un markdown à 20 sections pour un PDF de
    26 pages décale TOUT à partir de la première erreur, et le décalage est silencieux.
    """
    report = ValidationReport(ok=True)
    report.pages_found = len(parsed.pages)
    report.pages_expected = expected_pages or 0
    report.conventions_presentes = bool(parsed.conventions)

    if _BARE_SEPARATOR_RE.search(text or ""):
        report.errors.append(
            "Séparateur sans numéro détecté (« <!-- PageBreak --> »). Utiliser la forme "
            "complète : <!-- PageBreak page_pdf=\"N\" page_imprimee=\"M\" -->."
        )

    if not parsed.pages:
        report.errors.append(
            "Aucune section de page trouvée. Attendu au moins un séparateur "
            "<!-- PageBreak page_pdf=\"1\" ... -->."
        )

    numbers = parsed.page_numbers
    seen: Dict[int, int] = {}
    for n in numbers:
        seen[n] = seen.get(n, 0) + 1
    duplicates = sorted(n for n, c in seen.items() if c > 1)
    if duplicates:
        report.errors.append(
            f"Numéro(s) de page PDF en double : {', '.join(map(str, duplicates))}."
        )

    if numbers and numbers != sorted(numbers):
        report.warnings.append(
            "Les pages ne sont pas dans l'ordre croissant — vérifier que ce n'est pas un "
            "copier-coller déplacé."
        )

    if expected_pages:
        if len(parsed.pages) != expected_pages:
            report.errors.append(
                f"{len(parsed.pages)} section(s) de page pour {expected_pages} page(s) dans le "
                "PDF. L'appariement page ↔ image serait faux : import refusé."
            )
        hors_bornes = sorted({n for n in numbers if n < 1 or n > expected_pages})
        if hors_bornes:
            report.errors.append(
                f"Page(s) hors du PDF : {', '.join(map(str, hors_bornes))} "
                f"(le PDF a {expected_pages} page(s))."
            )
        manquantes = sorted(set(range(1, expected_pages + 1)) - set(numbers))
        if manquantes and not duplicates:
            apercu = ", ".join(map(str, manquantes[:20]))
            suite = "…" if len(manquantes) > 20 else ""
            report.errors.append(f"Page(s) PDF absente(s) du markdown : {apercu}{suite}.")

    report.pages_vides = [p.page_pdf for p in parsed.pages if not p.body.strip()]
    if report.pages_vides:
        report.warnings.append(
            f"{len(report.pages_vides)} page(s) sans contenu : "
            f"{', '.join(map(str, report.pages_vides[:20]))}."
        )

    if not parsed.conventions:
        report.warnings.append(
            "Aucune section « ## Conventions … » : les unités et le code couleur de cotation "
            "ne seront pas joints aux pages."
        )

    declared = str(parsed.frontmatter.get("source_sha256") or "").strip()
    if declared and source_sha256:
        report.hash_source_ok = declared == source_sha256
        if not report.hash_source_ok:
            report.warnings.append(
                "Le hachage déclaré ne correspond pas au PDF actuel : le markdown a été écrit "
                "pour une AUTRE version du fichier."
            )

    report.apercu = [
        {
            "page_pdf": p.page_pdf,
            "page_imprimee": p.page_imprimee,
            "titre": p.title,
            "caracteres": p.char_count,
        }
        for p in parsed.pages[:400]
    ]
    report.ok = not report.errors
    return report


# ---------------------------------------------------------------------------
# Stockage et lecture
# ---------------------------------------------------------------------------


def markdown_path(document_id: int) -> Path:
    return PAGE_MARKDOWN_DIR / f"{int(document_id)}.md"


def has_markdown(document_id: int) -> bool:
    return markdown_path(document_id).exists()


def read_markdown(document_id: int) -> Optional[str]:
    path = markdown_path(document_id)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("[PageMarkdown] lecture impossible (doc %s) : %s", document_id, exc)
        return None


def write_markdown(document_id: int, text: str) -> Path:
    PAGE_MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    path = markdown_path(document_id)
    path.write_text((text or "").replace("\r\n", "\n"), encoding="utf-8")
    invalidate(document_id)
    return path


def delete_markdown(document_id: int) -> bool:
    path = markdown_path(document_id)
    if not path.exists():
        return False
    try:
        os.remove(path)
    except OSError as exc:
        logger.warning("[PageMarkdown] suppression impossible (doc %s) : %s", document_id, exc)
        return False
    invalidate(document_id)
    return True


# Cache TTL du markdown parsé — même réglage que le cache de feuilles du packer.
_cache: Dict[int, Tuple[float, ParsedMarkdown]] = {}


def invalidate(document_id: Optional[int] = None) -> None:
    if document_id is None:
        _cache.clear()
    else:
        _cache.pop(int(document_id), None)


def load_parsed(document_id: int) -> Optional[ParsedMarkdown]:
    """Markdown parsé d'un document, ou None s'il n'en a pas."""
    document_id = int(document_id)
    ttl = settings.CAG_FULLTEXT_CACHE_TTL
    now = time.monotonic()
    if ttl > 0:
        cached = _cache.get(document_id)
        if cached and cached[0] > now:
            return cached[1]

    text = read_markdown(document_id)
    if text is None:
        return None
    parsed = parse_markdown(text)
    if ttl > 0:
        _cache[document_id] = (now + ttl, parsed)
    return parsed


def load_page_records(document_id: int) -> Optional[List[Tuple[int, int, str]]]:
    """Enregistrements ``(page, index, texte)`` — même forme que les feuilles du packer.

    Une page = UN enregistrement (contre 4 à 10 fragments auparavant) : un tableau ne peut
    plus être coupé entre deux morceaux et perdre son en-tête.
    """
    parsed = load_parsed(document_id)
    if not parsed or not parsed.pages:
        return None
    return [(p.page_pdf, 0, p.body) for p in parsed.pages if p.body.strip()]


def page_section(document_id: int, page_no: int) -> Optional[str]:
    """Corps du markdown augmenté d'UNE page, ou None.

    Sert les vues de consultation (modale « document consulté », recherche, catégories) :
    quand un document a été retranscrit, c'est cette page-là qu'on montre à côté du PDF,
    pas les fragments de l'extraction automatique.
    """
    parsed = load_parsed(document_id)
    if not parsed:
        return None
    for page in parsed.pages:
        if page.page_pdf == int(page_no):
            body = page.body.strip()
            return body or None
    return None


def conventions_block(document_id: int) -> str:
    """Conventions du document, à placer en tête du bloc packé.

    Placées une fois par document plutôt que répétées sur chaque page : les pages d'un
    même document sont contiguës dans le bloc, et la répétition coûterait des tokens sans
    rien ajouter.
    """
    parsed = load_parsed(document_id)
    if not parsed or not parsed.conventions:
        return ""
    return f"Conventions de lecture de ce document :\n{parsed.conventions}"


# ---------------------------------------------------------------------------
# Corpus BM25 — une ligne par page dans documentchunk
# ---------------------------------------------------------------------------
# Le markdown vit dans un fichier, mais BM25 interroge ``documentchunk.tsv_content``,
# une colonne GÉNÉRÉE par Postgres à l'insertion et indexée en GIN. Il suffit donc d'y
# insérer des lignes : aucune colonne à ajouter, aucun index à créer.
#
# Une ligne = UNE page. C'est la même granularité que la matière lue, et c'est celle que
# le retrieval agrège de toute façon (``retrieve_bm25_pages`` groupe par page).

CONTENT_TYPE_PAGE_MARKDOWN = "page_markdown"
CHUNKING_VERSION_PAGE_MARKDOWN = "page_markdown_v1"


def delete_chunks(session, document_id: int) -> int:
    """Retire du corpus lexical les lignes de markdown d'un document."""
    from sqlalchemy import text as sql

    res = session.execute(
        sql(
            """
            DELETE FROM documentchunk
            WHERE document_id = :doc
              AND COALESCE(metadata_json->>'content_type', '') = :ct
            """
        ),
        {"doc": int(document_id), "ct": CONTENT_TYPE_PAGE_MARKDOWN},
    )
    return int(res.rowcount or 0)


def sync_chunks(session, document_id: int) -> int:
    """(Re)construit les lignes BM25 depuis le fichier markdown. Idempotent.

    Retourne le nombre de pages indexées ; 0 si le document n'a pas de markdown, après
    avoir retiré d'éventuelles lignes devenues orphelines.
    """
    from app.models.document import Document
    from app.models.document_chunk import DocumentChunk

    delete_chunks(session, document_id)

    parsed = load_parsed(document_id)
    if not parsed or not parsed.pages:
        session.commit()
        return 0

    document = session.get(Document, int(document_id))
    source = getattr(document, "source", None) if document else None

    # Index de chunk hors de portée de l'extraction, pour ne jamais entrer en collision
    # avec les feuilles produites par le pipeline.
    base_index = 900_000
    crees = 0
    for page in parsed.pages:
        corps = (page.body or "").strip()
        if not corps:
            continue
        meta = {
            "document_id": int(document_id),
            "content_type": CONTENT_TYPE_PAGE_MARKDOWN,
            "chunking_version": CHUNKING_VERSION_PAGE_MARKDOWN,
            "page_no": page.page_pdf,
            "page_start": page.page_pdf,
            "page_end": page.page_pdf,
            "page_imprimee": page.page_imprimee,
            "heading": page.title or None,
            "is_leaf": True,
        }
        session.add(
            DocumentChunk(
                document_id=int(document_id),
                chunk_index=base_index + page.page_pdf,
                content=corps,
                text=corps,
                start_char=0,
                end_char=len(corps),
                node_id=f"page-markdown-{document_id}-{page.page_pdf}",
                is_leaf=True,
                hierarchy_level=1,
                metadata_json=meta,
                metadata_=meta,
                source=source,
            )
        )
        crees += 1

    session.commit()
    logger.info(
        "[PageMarkdown] corpus BM25 — doc %s : %d page(s) indexée(s)", document_id, crees
    )
    return crees


def status(document_id: int, pdf_path: Optional[str]) -> Dict[str, Any]:
    """État du markdown d'un document, pour la pastille de l'interface."""
    parsed = load_parsed(document_id)
    if parsed is None:
        return {"present": False, "pages": 0, "pages_pdf": None, "hash_source_ok": None}

    expected = pdf_page_count(pdf_path) if pdf_path and os.path.exists(pdf_path) else None
    declared = str(parsed.frontmatter.get("source_sha256") or "").strip()
    hash_ok: Optional[bool] = None
    if declared and pdf_path and os.path.exists(pdf_path):
        actual = sha256_of_file(pdf_path)
        hash_ok = (actual == declared) if actual else None

    return {
        "present": True,
        "pages": len(parsed.pages),
        "pages_pdf": expected,
        "aligne": (expected is None) or (len(parsed.pages) == expected),
        "conventions": bool(parsed.conventions),
        "hash_source_ok": hash_ok,
        "titre": str(parsed.frontmatter.get("titre") or "") or None,
        "redige_le": str(parsed.frontmatter.get("redige_le") or "") or None,
    }
