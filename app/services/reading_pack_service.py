"""Pack de LECTURE — le lecteur reçoit des pages LUES, jamais du texte extrait.

Principe posé par Elie le 2026-09-13 : **le texte extrait sert à TROUVER la page (BM25),
jamais à répondre.** La page est lue en PNG par un modèle vision isolé (une page, une
question, aucun autre contexte), et c'est cette lecture — pas l'image, pas la
transcription — qui entre dans le contexte du générateur.

Ce que ce module remplace, et pourquoi. Le pack précédent (``context_packer_service``)
chargeait les chunks feuilles des pages trouvées et y joignait jusqu'à 6 PNG. Deux défauts
mesurés sur le corpus réel :

1. **Sur une planche sans couche texte, le texte indexé est une transcription vision qui ne
   porte AUCUNE cote.** Mesuré sur le dossier Perform76 (document 438), dont les 26 pages
   sont des planches vectorielles : la page 8 rend 663 caractères — la liste des libellés
   (« Parclose 2452 », « Parclose 2636 »…) et une note générale (« valable pour une
   feuillure de 62 mm »). Zéro valeur. Le 2026-09-13 le lecteur a répondu « 31,5 mm pour une
   feuillure de 62 mm, valable avec un joint post-extrudé » : la seconde moitié est
   recopiée mot pour mot de cette transcription, la première est la ligne VOISINE du dessin.

2. **Sur une page AVEC couche texte, la transcription fabrique des associations fausses.**
   Document 400 page 172 : le texte natif du PDF liste proprement « TGA3817 Cale de vitrage /
   TGY3600 Equerre 11x28 », le texte indexé rend
   « col1:**TGA3817** Cale de vitrage=**TGY3605** Butées multivantaux » — un signe égal entre
   deux références sans rapport, produit par la reconstruction de tableau.

Et l'image brute jointe au contexte n'est pas une solution : mesuré le 2026-09-12, la même
planche lue par le grand modèle au milieu du contexte du tour donne le mauvais chiffre,
alors que le lecteur délégué est juste. Ce n'est ni le modèle ni la résolution : c'est le
cadrage de l'appel.

Mesures qui fondent la conception (2026-09-13, corpus réel) :

    planche cotée    438 p.8    « épaisseur de vitrage parclose 2636 »  → lecture du dessin
    tableau          400 p.172  « rallonge 4 points pour TGY3702 »      → TGY3704
    procédure        405 p.13   « ordre de montage du bouclier »        → les 6 étapes
    6 pages en parallèle : 8,5 s de mur pour 26,4 s d'appels cumulés
    les 5 pages hors sujet se déclarent « absent » — elles n'ajoutent aucun bruit

Le texte NATIF du PDF (les caractères réellement imprimés, lus par PyMuPDF) est joint à la
lecture, jamais à sa place : il ne sert qu'à vérifier l'orthographe exacte d'une référence
ou d'une norme, là où un modèle vision peut confondre TGY3702 et TGY3704. La transcription
vision (``page_raw_enriched``), elle, n'apparaît nulle part.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlmodel import Session, select

from app.config import settings
from app.models.document import Document
from app.services.page_reader_service import PageReading, read_page_image, survey_entry_for

logger = logging.getLogger(__name__)

# Repères du relevé listés dans le bloc : ils disent au lecteur CE QUI EST SUR LA PAGE,
# lu sur le dessin et non sur la transcription. Sans eux, le bloc COUVERTURE déclarerait
# absente une référence pourtant présente sur la planche.
MAX_SURVEY_LABELS = 30
# Texte natif joint par page. Une page de catalogue en fait ~800, une notice ~2 000.
NATIVE_TEXT_CHARS = 1800


@dataclass
class PageRead:
    """Une page du pack : ce qui a été LU dessus, et le texte imprimé qui l'accompagne."""

    document_id: int
    page_no: int
    is_star: bool = True
    reading: Optional[PageReading] = None
    native_text: str = ""
    fallback_text: str = ""          # texte indexé, UNIQUEMENT si le PDF est indisponible

    @property
    def answered(self) -> bool:
        return self.reading is not None and self.reading.ok


@dataclass
class ReadingPackTrace:
    """Ce que le pack a lu — va dans la trace du tour, pas dans le contexte."""

    pages_requested: int = 0
    pages_read: int = 0
    pages_answered: int = 0
    pages_absent: int = 0
    pages_failed: int = 0
    wall_ms: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_trace(self) -> Dict[str, Any]:
        return {
            "pages_requested": self.pages_requested,
            "pages_read": self.pages_read,
            "pages_answered": self.pages_answered,
            "pages_absent": self.pages_absent,
            "pages_failed": self.pages_failed,
            "wall_ms": self.wall_ms,
            "details": self.details[:30],
        }


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


def select_pages_to_read(
    ranked_docs: Sequence[Tuple[int, Dict[str, Any]]],
    *,
    max_readings: int,
    max_per_doc: int,
) -> List[Tuple[int, int]]:
    """(document, page) à lire, au mérite, en servant chaque document élu à tour de rôle.

    Le tour de rôle (et non « le premier document d'abord ») garantit qu'un document élu en
    complément obtient au moins une page lue : c'est exactement ce que le plancher d'images
    par document assurait avant sa suppression, et son absence rendait les co-élus muets.
    Fonction pure, testable sans base ni réseau.
    """
    par_doc: List[List[Tuple[int, int]]] = []
    for did, meta in ranked_docs:
        pages = meta.get("matched_pages") or {}
        ordered = sorted(pages.items(), key=lambda kv: (-float(kv[1]), kv[0]))
        par_doc.append([(int(did), int(p)) for p, _ in ordered[:max_per_doc]])

    out: List[Tuple[int, int]] = []
    rang = 0
    while len(out) < max_readings and any(rang < len(lst) for lst in par_doc):
        for lst in par_doc:
            if rang < len(lst):
                out.append(lst[rang])
                if len(out) >= max_readings:
                    break
        rang += 1
    return out


def _native_text(pdf_path: Optional[str], pages: Sequence[int]) -> Dict[int, str]:
    """Couche texte NATIVE du PDF, page par page (vide = planche vectorielle ou scan)."""
    if not pdf_path or not pages:
        return {}
    import fitz

    out: Dict[int, str] = {}
    try:
        with fitz.open(pdf_path) as pdf:
            for page_no in pages:
                if 1 <= page_no <= len(pdf):
                    try:
                        out[page_no] = pdf[page_no - 1].get_text("text").strip()
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("[pack de lecture] couche texte p.%s illisible : %s", page_no, exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[pack de lecture] ouverture PDF échouée (%s) : %s", pdf_path, exc)
    return out


def render_page_block(entry: PageRead, *, needle: str = "") -> str:
    """Rend UNE page lue. La provenance de chaque ligne est explicite : ce qui vient de
    l'image le dit, ce qui vient des caractères imprimés le dit aussi."""
    head = f"\n[page {entry.page_no}{' — ★ page retrouvée par la recherche' if entry.is_star else ''}]"
    lines: List[str] = []
    r = entry.reading

    if r is None and entry.fallback_text:
        # PDF indisponible : on ne peut pas VOIR la page. Le texte indexé est dégradé
        # (transcription vision sur les planches) et doit se présenter comme tel.
        return (
            head
            + " — PAGE NON LUE (fichier source indisponible). Texte indexé, NON vérifié sur la "
            "page, à ne pas citer comme une valeur :\n"
            + entry.fallback_text
        )
    if r is None:
        return head + " — page non lue."

    if r.error:
        lines.append(f" — LECTURE INDISPONIBLE ({r.error}).")
    elif r.absent:
        lines.append(" — LUE EN IMAGE : ce que tu cherches n'apparaît PAS sur cette page.")
    elif r.ambiguous or not r.answer:
        lines.append(" — LUE EN IMAGE : la page ne permet pas de trancher avec certitude.")
    else:
        lines.append(" — LUE EN IMAGE :")
        lines.append(f"  → {r.answer}")

    if r.convention:
        lines.append(f"  convention lue sur la page : « {r.convention} »")

    # Deux pages peuvent répondre à la même question avec des valeurs différentes (mesuré :
    # la planche des parcloses donne une valeur AVEC sa convention de lecture, une planche
    # de meneaux en donne une autre SANS aucune règle). Les présenter à égalité rouvrirait
    # l'arbitrage au hasard : on signale donc la page qui ne dit pas comment se lire.
    if r.answer and not r.absent and not r.ambiguous and not r.convention:
        lines.append(
            "  fiabilité : FAIBLE — aucune règle de lecture n'est inscrite sur cette page, "
            "rien ne dit ce que ce nombre mesure. Si une autre page répond AVEC sa "
            "convention, c'est elle qui fait foi."
        )

    entree = survey_entry_for(r.survey, needle) if needle else None
    if entree:
        valeurs = " · ".join(
            f"{v.get('valeur')} ({v.get('couleur')})" if v.get("couleur") else str(v.get("valeur"))
            for v in (entree.get("valeurs") or [])
        )
        if valeurs:
            lines.append(f"  relevé pour « {entree.get('repere')} » : {valeurs}")

    if r.survey:
        # Les LIBELLÉS seulement : les étiquettes de couleur du relevé basculent en bloc
        # d'un run à l'autre (mesuré le 12/09) et ne valent rien. L'inventaire, lui, dit ce
        # qui est sur la page — et il est lu sur le dessin, pas sur une transcription.
        labels = [str(e.get("repere")) for e in r.survey if e.get("repere")]
        extrait = ", ".join(labels[:MAX_SURVEY_LABELS])
        reste = f" … ({len(labels)} au total)" if len(labels) > MAX_SURVEY_LABELS else ""
        lines.append(f"  repères lus sur la page : {extrait}{reste}")

    if entry.native_text:
        lines.append(
            "  Texte imprimé de la page (caractères exacts du PDF — sert à vérifier "
            "l'orthographe d'une référence, pas à déduire une valeur) :"
        )
        lines.append(entry.native_text[:NATIVE_TEXT_CHARS])

    # La première ligne prolonge l'en-tête de page ; les suivantes sont indentées dessous.
    return head + lines[0] + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")


def render_document_block(
    doc: Document, index: int, entries: Sequence[PageRead], *, needle: str = ""
) -> str:
    titre = doc.title or "Document sans titre"
    header = _doc_header(doc)
    out = [f"=== DOCUMENT {index} (id {doc.id}) : « {titre} » ==="]
    if header:
        out.append(header)
    out.append(
        "Pages LUES pour cette question : "
        + ", ".join(str(e.page_no) for e in entries)
    )
    for entry in entries:
        out.append(render_page_block(entry, needle=needle))
    return "\n".join(out)


PREAMBULE = (
    "\n\nPAGES LUES — chaque page ci-dessous a été RENDUE EN IMAGE et lue par un lecteur "
    "dédié, une page à la fois, avec ta question. Ce que tu lis ici est le résultat de cette "
    "lecture, pas un texte extrait automatiquement.\n"
    "RÈGLE : une valeur, une cote, une référence ou une consigne ne se cite QUE depuis une "
    "page lue. Une page qui dit « ce que tu cherches n'apparaît PAS » est une information "
    "fiable : cherche ailleurs (rechercher, lire_pages sur une autre page), ne transpose "
    "JAMAIS la valeur d'un repère voisin et ne reprends JAMAIS une note générale de la page "
    "comme si c'était la valeur du repère demandé.\n"
    "Avant d'attribuer une information à une gamme ou à un produit, vérifie l'en-tête du "
    "document : ne transfère jamais une valeur d'une gamme vers une autre.\n\n"
)


async def build_reading_pack(
    session: Session,
    passages: List[Dict[str, Any]],
    *,
    question: str,
    system_prompt: str,
    needle: str = "",
    elected_document_ids: Optional[List[int]] = None,
    max_documents: Optional[int] = None,
    max_readings: Optional[int] = None,
    max_pages_per_doc: Optional[int] = None,
    dpi: Optional[int] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Construit le message système du lecteur à partir de PAGES LUES EN IMAGE.

    Retourne la même forme que ``build_cag_context`` (``content``, ``cag_documents``,
    ``cag_document_blocks``) pour rester compatible avec les sources UI, le registre de la
    boucle et le contrôle de sortie, plus ``reading_trace`` pour le cheminement.
    """
    import time

    from app.services.context_packer_service import (
        _apply_document_election,
        aggregate_documents,
    )

    max_documents = max_documents or max(1, min(3, settings.CAG_MAX_DOCUMENTS))
    max_readings = max_readings or settings.READER_PACK_MAX_READINGS
    max_pages_per_doc = max_pages_per_doc or settings.READER_PACK_MAX_PAGES_PER_DOC

    system_message: Dict[str, Any] = {"role": "system", "content": system_prompt}
    trace = ReadingPackTrace()

    if not passages:
        system_message["content"] += "\n\nAucune page trouvée dans cet espace pour cette requête."
        system_message["cag_documents"] = []
        system_message["cag_document_blocks"] = []
        system_message["reading_trace"] = trace.to_trace()
        return system_message

    ranked_docs = aggregate_documents(passages, max_documents=max_documents)
    ranked_docs = _apply_document_election(
        ranked_docs, elected_document_ids, None, max_documents=max_documents
    )

    cibles = select_pages_to_read(
        ranked_docs, max_readings=max_readings, max_per_doc=max_pages_per_doc
    )
    trace.pages_requested = len(cibles)
    if not cibles:
        system_message["content"] += "\n\nAucune page exploitable pour cette requête."
        system_message["cag_documents"] = []
        system_message["cag_document_blocks"] = []
        system_message["reading_trace"] = trace.to_trace()
        return system_message

    doc_ids = [did for did, _ in ranked_docs]
    docs_by_id = {
        d.id: d for d in session.exec(select(Document).where(Document.id.in_(doc_ids))).all()
    }

    # Texte natif : une ouverture de PDF par document, hors boucle d'événements.
    pages_par_doc: Dict[int, List[int]] = {}
    for did, page in cibles:
        pages_par_doc.setdefault(did, []).append(page)
    natif_par_doc: Dict[int, Dict[int, str]] = {}
    for did, pages in pages_par_doc.items():
        doc = docs_by_id.get(did)
        chemin = (
            doc.source_file_path
            if doc and doc.source_file_path and os.path.exists(doc.source_file_path)
            else None
        )
        natif_par_doc[did] = await asyncio.to_thread(_native_text, chemin, pages)

    # ——— Les lectures, en PARALLÈLE ———
    # Mesuré : 6 pages = 8,5 s de mur pour 26,4 s d'appels cumulés. C'est le parallélisme
    # qui rend la lecture systématique payable ; en série elle serait inutilisable.
    t0 = time.perf_counter()

    async def _lire(did: int, page: int) -> Tuple[int, int, Optional[PageReading]]:
        doc = docs_by_id.get(did)
        chemin = (
            doc.source_file_path
            if doc and doc.source_file_path and os.path.exists(doc.source_file_path)
            else None
        )
        if not chemin:
            return did, page, None
        lecture = await read_page_image(
            pdf_path=chemin,
            document_id=did,
            page_no=page,
            question=question,
            needle=needle,
            model=model,
            dpi=dpi,
        )
        return did, page, lecture

    resultats = await asyncio.gather(*(_lire(did, page) for did, page in cibles))
    trace.wall_ms = int((time.perf_counter() - t0) * 1000)

    lectures: Dict[Tuple[int, int], Optional[PageReading]] = {
        (did, page): lecture for did, page, lecture in resultats
    }

    # ——— Assemblage ———
    blocs: List[str] = []
    cag_documents: List[Dict[str, Any]] = []
    position = 0
    for did, meta in ranked_docs:
        doc = docs_by_id.get(did)
        pages = sorted(pages_par_doc.get(did) or [])
        if doc is None or not pages:
            continue
        entrees: List[PageRead] = []
        for page in pages:
            lecture = lectures.get((did, page))
            entree = PageRead(
                document_id=did,
                page_no=page,
                reading=lecture,
                native_text=(natif_par_doc.get(did, {}).get(page) or "").strip(),
            )
            if lecture is None:
                entree.fallback_text = _texte_indexe(session, did, page)
            entrees.append(entree)

            trace.pages_read += 1 if lecture is not None else 0
            if lecture is None:
                trace.pages_failed += 1
            elif lecture.error:
                trace.pages_failed += 1
            elif lecture.absent:
                trace.pages_absent += 1
            elif lecture.ok:
                trace.pages_answered += 1
            trace.details.append(
                {
                    "document_id": did,
                    "page_no": page,
                    "etat": (
                        "non_lue" if lecture is None
                        else "erreur" if lecture.error
                        else "absent" if lecture.absent
                        else "ambigu" if lecture.ambiguous
                        else "repond"
                    ),
                    "ms": lecture.ms if lecture else 0,
                }
            )

        position += 1
        blocs.append(render_document_block(doc, position, entrees, needle=needle))
        cag_documents.append(
            {
                "index": position,
                "document_id": did,
                "document_title": doc.title,
                "pages": pages,
                "full_document": False,
                "score": round(float(meta.get("score_max") or 0.0), 4),
                "election_score": round(float(meta.get("election_score") or 0.0), 4),
                "matched_pages": sorted(meta.get("matched_pages") or {}),
                "seed_pages": pages,
                "has_source_file": bool(getattr(doc, "source_file_path", None)),
            }
        )

    system_message["content"] += PREAMBULE + "\n\n".join(blocs)
    system_message["cag_documents"] = cag_documents
    system_message["cag_document_blocks"] = blocs
    system_message["reading_trace"] = trace.to_trace()

    logger.info(
        "[pack de lecture] %d page(s) lue(s) en %.1fs (mur) — %d répondent, %d absentes, "
        "%d en échec | %d document(s), ~%d caractères",
        trace.pages_read,
        trace.wall_ms / 1000,
        trace.pages_answered,
        trace.pages_absent,
        trace.pages_failed,
        len(blocs),
        sum(len(b) for b in blocs),
    )
    return system_message


def _texte_indexe(session: Session, document_id: int, page_no: int) -> str:
    """Repli UNIQUEMENT quand le PDF source est absent : la page ne peut pas être VUE.

    Le texte indexé est alors la seule matière disponible ; il est rendu avec un avertissement
    explicite (cf. ``render_page_block``) parce que sur une planche il s'agit d'une
    transcription vision sans aucune cote.
    """
    from app.services.context_packer_service import _load_leaf_records

    try:
        records = _load_leaf_records(session, document_id)
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(t for (pg, _, t) in records if pg == page_no)[:NATIVE_TEXT_CHARS]
