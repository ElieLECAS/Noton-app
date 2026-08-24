"""Extraction d'atomes SAV depuis un document DÉJÀ traité de la bibliothèque.

Lot L1 de ``docs/plan_generation_graphe_depuis_pdf_2026-08-24.md`` — « temps A » de la
boucle : lire, par batch de pages, le texte déjà en base ET les images des pages, et
n'en retenir que ce qui aide un poseur sur un chantier.

Rien n'est ré-extrait : le texte vient de ``documentchunk`` (chunks L1 ``semantic_leaf``),
les PNG sont re-rendus localement depuis le PDF stocké (aucun appel API pour l'image).

Sortie = des **atomes**, pas des nœuds : constat / cause / vérification / geste, avec la
phrase source et la page.

``build_pivot_from_atoms`` (lot L2) fait l'assemblage : produit → symptôme → causes →
feuilles, **entièrement en Python, zéro appel LLM** — le modèle produit des faits, le
graphe est câblé en code (invariant I1 du plan). Le résultat est un JSON pivot, réimporté
par ``guided_json_import_service.import_from_json_text`` : un seul convertisseur, déjà
couvert par ``tests/test_guided_json_import.py``.

Trois garde-fous, tous repris de l'ingestion plutôt que réinventés :
  1. ``_ungrounded_numbers`` — un nombre absent du texte transcrit a été lu sur l'image
     (hors mandat) ou inventé : l'atome entier est écarté. Une cote de réglage fausse est
     pire qu'une cote absente.
  2. re-découpage du batch si la réponse est tronquée — sinon la réparation JSON rend un
     objet valide mais AMPUTÉ, et les atomes perdus le restent en silence.
  3. la page citée doit appartenir au batch — sinon l'atome n'est pas rattachable.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, ValidationError
from sqlmodel import Session

from app.config import settings
from app.database import engine
from app.models.document import Document

logger = logging.getLogger(__name__)

QUI_VALUES = ("poseur", "sav")
EFFORT_VALUES = ("visuel", "outil_simple", "demontage")


class SavAtom(BaseModel):
    """Un constat exploitable sur chantier, rattaché à une page."""

    symptome: str = Field(description="Ce que le client ou le poseur constate")
    cause: str = Field(default="", description="Origine probable, telle que la notice la donne")
    # B2 : le signe OBSERVABLE qui distingue cette cause de ses sœurs. C'est lui qui
    # devient le libellé du bouton — jamais `cause`, qui est un diagnostic qu'un client
    # ne sait pas poser (« polarité inversée des 24 V » n'est pas cliquable). Vide =
    # cause indiscernable sur place : elle sera fusionnée dans une feuille-checklist.
    constat_cause: str = Field(default="", description="Signe observable distinguant cette cause")
    verification: str = Field(default="", description="Ce qu'on peut regarder/essayer sur place")
    geste: str = Field(default="", description="Ce qu'on fait, ou la raison d'appeler le SAV")
    citation: str = Field(default="", description="Phrase de la notice, verbatim")
    page: int = Field(description="Rang de la page dans le fichier")
    produit: str = Field(default="", description="Désignation commerciale réelle si connue")
    qui: str = Field(default="poseur")
    effort: str = Field(default="visuel")

    def grounding_text(self) -> str:
        """Texte soumis au contrôle des valeurs numériques."""
        return " ".join([self.symptome, self.cause, self.verification, self.geste])

    def dedupe_key(self) -> Tuple[str, str]:
        return (_normalize(self.symptome), _normalize(self.cause))


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """Tu es technicien SAV en menuiserie (PVC, aluminium, hybride) et tu prépares
un assistant destiné aux POSEURS sur chantier et à leurs clients.

On te donne le texte transcrit de plusieurs pages consécutives d'un document technique ET les
images de ces pages. Tu en extrais uniquement ce qui aide QUELQU'UN QUI EST DEVANT LE MATÉRIEL.

CE QUI EST UTILE (à extraire) :
- pose et mise en œuvre, réglages, reprises de réglage ;
- pannes, dysfonctionnements, non-conformités, codes d'erreur ;
- entretien, accessoires et pièces de rechange ;
- manutention et stockage sur chantier ;
- garanties et limites d'emploi quand elles conditionnent une intervention.

CE QUI NE SERT À RIEN ICI (à ignorer complètement) :
- fabrication, usinage en atelier, cotes de débit, gammes de fabrication fournisseur ;
- tarifs, conditions commerciales, argumentaire de vente ;
- sommaires, pages de garde, mentions légales.

UN ATOME = UN CONSTAT, PAS UN CHAPITRE. Il faut qu'il tienne en cinq morceaux :
- "symptome" : ce que la personne CONSTATE (« le volet s'arrête avant le bas »), jamais la
  cause technique et jamais l'action à faire. **Réutilise EXACTEMENT la même formulation
  pour deux lignes qui parlent du même problème** (dans un tableau de pannes, toutes les
  lignes d'un même bloc « Erreur » partagent le même symptôme) ;
- "cause" : l'origine probable, telle que le document la donne ;
- "constat_cause" : le signe OBSERVABLE qui permet de reconnaître CETTE cause plutôt
  qu'une autre du même symptôme (« aucun bruit du tout », « il bipe 3 fois », « la porte
  n'est pas tout à fait fermée »). **Laisse-le VIDE si le document ne donne aucun signe
  qu'une personne sur place peut voir ou entendre** — c'est le cas des causes qui
  demandent un appareil de mesure (tension, polarité). Ne l'invente jamais ;
- "verification" : ce qu'on peut regarder, écouter ou essayer SUR PLACE pour trancher ;
- "geste" : ce qu'on fait pour corriger — ou, si ce n'est pas à la portée d'un poseur, la
  raison d'appeler le SAV.

RÈGLES ABSOLUES :
1. Zéro invention. Chaque atome doit être retrouvable dans le texte fourni. Dans le doute,
   n'écris pas l'atome.
2. "citation" = une phrase du texte RECOPIÉE CARACTÈRE POUR CARACTÈRE, celle qui fonde
   l'atome. Ce champ est comparé automatiquement au texte de la page : une phrase
   reformulée, recousue à partir de deux endroits, ou « nettoyée » est REJETÉE. Copie
   une seule phrase contiguë, telle quelle, même si elle est mal ponctuée. Si aucune
   phrase ne suffit à elle seule, prends la plus proche et laisse le reste de côté.
3. "page" = le numéro de page indiqué dans le texte fourni (--- PAGE n ---), jamais un
   numéro imprimé lu sur l'image.
4. Les valeurs chiffrées (cotes, couples, angles, durées) doivent figurer telles quelles
   dans le TEXTE. Ne lis jamais une cote sur une image.
5. Garde les désignations commerciales exactes (« Oximo 40 WF RTS »), jamais une paraphrase.
6. Les images servent à comprendre un schéma, un repérage, une séquence de gestes — pas à
   lire du texte fin.
7. "qui" : "poseur" si le geste est à sa portée, "sav" si ça touche au câblage, à un
   démontage de ferrure sous contrainte, ou si le document le réserve à un professionnel.
8. "effort" : "visuel" (il suffit de regarder/écouter), "outil_simple" (tournevis, clé),
   "demontage" (il faut déposer une pièce).
9. Si ces pages ne contiennent RIEN d'utile sur chantier, rends une liste vide. C'est un
   résultat normal et attendu, pas un échec — la plupart des pages d'un catalogue sont
   hors sujet.

SORTIE — JSON strict, rien d'autre :
{"atomes": [{"symptome": "...", "cause": "...", "constat_cause": "...",
"verification": "...", "geste": "...", "citation": "...", "page": 12, "produit": "...",
"qui": "poseur", "effort": "visuel"}]}
"""

_USER_TEMPLATE = """Document : {title}
Pages fournies : {page_range}

TEXTE TRANSCRIT DES PAGES
{page_text}
"""


# ---------------------------------------------------------------------------
# Fonctions pures (testables sans réseau ni base)
# ---------------------------------------------------------------------------


def _normalize(value: str) -> str:
    """Forme comparable d'un libellé : minuscules, sans accents ni ponctuation."""
    import unicodedata

    text = unicodedata.normalize("NFD", (value or "").lower())
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def build_sav_batches(
    page_numbers: Sequence[int],
    *,
    batch_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[List[int]]:
    """Fenêtre glissante de pages, aux réglages SAV (8/1 par défaut).

    Délègue à ``contextual_enrichment_service.build_page_batches`` : même mécanique que
    l'ingestion, seuls les défauts changent.
    """
    from app.services.contextual_enrichment_service import build_page_batches

    return build_page_batches(
        list(page_numbers),
        batch_size=batch_size if batch_size is not None else settings.SAV_EXTRACTION_BATCH_SIZE,
        overlap=overlap if overlap is not None else settings.SAV_EXTRACTION_BATCH_OVERLAP,
    )


def _strip_fences(raw: str) -> str:
    text = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_atoms_payload(raw: str) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    """Lit la réponse du modèle.

    Returns:
        ``(atomes, truncated)``. ``truncated=True`` quand le JSON n'est pas parsable tel
        quel : la réponse a été coupée par le plafond de sortie (ou est malformée). On ne
        répare pas ici — un objet réparé est valide mais AMPUTÉ, et le silence est
        précisément le mode d'échec qu'on veut éviter. L'appelant re-découpe le batch.
    """
    try:
        data = json.loads(_strip_fences(raw))
    except (json.JSONDecodeError, TypeError):
        return None, True

    if isinstance(data, list):
        return list(data), False
    if not isinstance(data, dict):
        return None, True

    for key in ("atomes", "atoms", "items", "resultats"):
        value = data.get(key)
        if isinstance(value, list):
            return list(value), False
    # Objet JSON valide sans liste d'atomes : le modèle a répondu « rien ici ».
    return [], False


def coerce_atoms(
    rows: Sequence[Dict[str, Any]],
    allowed_pages: Sequence[int],
) -> Tuple[List[SavAtom], List[str]]:
    """Valide les atomes bruts. Retourne (atomes retenus, motifs de rejet)."""
    pages = set(int(p) for p in allowed_pages)
    kept: List[SavAtom] = []
    rejected: List[str] = []

    for row in rows:
        if not isinstance(row, dict):
            rejected.append("entrée non-objet")
            continue
        try:
            atom = SavAtom.model_validate(row)
        except ValidationError as exc:
            rejected.append(f"champ manquant ou invalide : {exc.errors()[0].get('loc')}")
            continue

        if not atom.symptome.strip():
            rejected.append("symptôme vide")
            continue
        # La page doit appartenir au batch : un atome non rattachable n'est pas
        # vérifiable, et sa provenance ne pourra pas être affichée au relecteur.
        if atom.page not in pages:
            rejected.append(f"page {atom.page} hors du batch")
            continue
        if atom.qui not in QUI_VALUES:
            atom.qui = "poseur"
        if atom.effort not in EFFORT_VALUES:
            atom.effort = "visuel"
        kept.append(atom)

    return kept, rejected


def drop_ungrounded_atoms(
    atoms: Sequence[SavAtom],
    source_text: str,
) -> Tuple[List[SavAtom], List[SavAtom]]:
    """Écarte les atomes portant une valeur chiffrée absente du texte source.

    Réutilise le contrôle déterministe de l'ingestion (``_ungrounded_numbers``) : la
    vision est autorisée pour comprendre un schéma, jamais pour lire une cote.
    """
    from app.services.contextual_enrichment_service import _ungrounded_numbers

    kept: List[SavAtom] = []
    dropped: List[SavAtom] = []
    for atom in atoms:
        bad = _ungrounded_numbers(atom.grounding_text(), source_text)
        if bad:
            logger.warning(
                "[SAV] Atome écarté — valeur(s) non ancrée(s) %s (symptôme=%r, page=%s)",
                bad[:5],
                atom.symptome[:60],
                atom.page,
            )
            dropped.append(atom)
            continue
        kept.append(atom)
    return kept, dropped


def dedupe_atoms(atoms: Sequence[SavAtom]) -> List[SavAtom]:
    """Fusionne les atomes que le recouvrement des batches a produits deux fois.

    Clé = (symptôme, cause) normalisés. On garde le premier vu, en lui laissant la
    citation la plus longue — celle qui documentera le mieux le nœud plus tard.
    """
    by_key: Dict[Tuple[str, str], SavAtom] = {}
    for atom in atoms:
        key = atom.dedupe_key()
        current = by_key.get(key)
        if current is None:
            by_key[key] = atom
            continue
        if len(atom.citation or "") > len(current.citation or ""):
            current.citation = atom.citation
        if not current.produit and atom.produit:
            current.produit = atom.produit
    return list(by_key.values())


# ---------------------------------------------------------------------------
# Accès aux données déjà en base
# ---------------------------------------------------------------------------


def load_pages_text(session: Session, document_id: int) -> Dict[int, str]:
    """Texte par page depuis les chunks L1 déjà indexés. Aucune ré-extraction."""
    from app.services.contextual_enrichment_service import _load_semantic_chunks_by_page

    chunks_by_page = _load_semantic_chunks_by_page(session, document_id)
    pages: Dict[int, str] = {}
    for page_no, chunks in chunks_by_page.items():
        parts: List[str] = []
        for chunk in chunks:
            content = (chunk.content or chunk.text or "").strip()
            if not content:
                continue
            heading = (chunk.metadata_json or {}).get("heading") or ""
            parts.append(f"{heading}\n{content}".strip() if heading else content)
        if parts:
            pages[page_no] = "\n\n".join(parts)
    return pages


def format_batch_text(batch_pages: Sequence[int], pages_text: Dict[int, str]) -> str:
    return "\n\n".join(
        f"--- PAGE {pno} ---\n{pages_text.get(pno) or '(vide)'}" for pno in batch_pages
    )


def _render_batch_images(pdf_path: str, batch_pages: Sequence[int]) -> List[str]:
    """PNG base64 des pages du batch. Rendu LOCAL depuis le PDF stocké, best-effort."""
    import base64

    from app.services.multimodal_page_service import render_page_png_cached

    images: List[str] = []
    for pno in batch_pages:
        try:
            png = render_page_png_cached(pdf_path, pno, dpi=settings.SAV_EXTRACTION_DPI)
            images.append(base64.b64encode(png).decode("ascii"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SAV] Rendu PNG page %s échoué : %s", pno, exc)
    return images


# ---------------------------------------------------------------------------
# Appel modèle
# ---------------------------------------------------------------------------


def _call_extraction_api(
    document_title: str,
    batch_pages: Sequence[int],
    page_text: str,
    images_b64: Optional[List[str]],
) -> str:
    from app.services.multimodal_page_service import _mistral_chat_completion

    page_range = (
        f"{batch_pages[0]}-{batch_pages[-1]}" if len(batch_pages) > 1 else str(batch_pages[0])
    )
    user_text = _USER_TEMPLATE.format(
        title=document_title or "Document",
        page_range=page_range,
        page_text=page_text[:60000],
    )

    if images_b64:
        user_content: Any = [{"type": "text", "text": user_text}]
        for image_b64 in images_b64:
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            )
    else:
        user_content = user_text

    return _mistral_chat_completion(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        page_no=batch_pages[0],
        max_tokens=settings.SAV_EXTRACTION_MAX_TOKENS,
        temperature=0.0,
        response_format_json=True,
        timeout_seconds=settings.SAV_EXTRACTION_TIMEOUT,
        model=settings.SAV_EXTRACTION_MODEL or settings.MULTIMODAL_EXTRACT_MODEL,
    )


def extract_atoms_for_batch(
    batch_pages: List[int],
    document_title: str,
    pages_text: Dict[int, str],
    pdf_path: Optional[str],
    *,
    _depth: int = 0,
) -> Tuple[List[SavAtom], Dict[str, Any]]:
    """Un batch → ses atomes. Re-découpe le batch si la réponse est tronquée.

    Le re-découpage est le garde-fou des gros batches : sans lui, une réponse coupée par
    le plafond de sortie est « réparée » en JSON valide mais amputé, et les atomes perdus
    ne se voient nulle part.
    """
    report: Dict[str, Any] = {"pages": list(batch_pages), "splits": 0, "rejected": []}
    if not batch_pages:
        return [], report

    source_text = format_batch_text(batch_pages, pages_text)
    if not source_text.strip() or all(
        not (pages_text.get(p) or "").strip() for p in batch_pages
    ):
        report["status"] = "pages_muettes"
        return [], report

    images = _render_batch_images(pdf_path, batch_pages) if pdf_path else []

    try:
        raw = _call_extraction_api(document_title, batch_pages, source_text, images or None)
    except Exception as exc:  # noqa: BLE001
        logger.error("[SAV] Batch %s échoué : %s", batch_pages, exc)
        report["status"] = "erreur"
        report["error"] = str(exc)[:300]
        return [], report

    rows, truncated = parse_atoms_payload(raw)

    if truncated:
        # Une seule page et toujours tronqué : on ne peut plus découper. On tente la
        # réparation pour sauver ce qui est lisible, en le signalant.
        if len(batch_pages) <= 1 or _depth >= 3:
            from app.services.multimodal_page_service import _parse_json_with_repair

            try:
                repaired = _parse_json_with_repair(raw)
                rows = (repaired or {}).get("atomes") or []
                report["status"] = "tronque_repare"
                logger.warning(
                    "[SAV] Batch %s tronqué et non découpable — réparation partielle",
                    batch_pages,
                )
            except Exception:  # noqa: BLE001
                report["status"] = "tronque_perdu"
                logger.error("[SAV] Batch %s tronqué, réparation impossible", batch_pages)
                return [], report
        else:
            middle = len(batch_pages) // 2
            halves = [batch_pages[:middle], batch_pages[middle:]]
            logger.info(
                "[SAV] Batch %s tronqué — re-découpage en %s",
                batch_pages,
                [f"{h[0]}-{h[-1]}" for h in halves if h],
            )
            atoms: List[SavAtom] = []
            splits = 1
            for half in halves:
                sub_atoms, sub_report = extract_atoms_for_batch(
                    half, document_title, pages_text, pdf_path, _depth=_depth + 1
                )
                atoms.extend(sub_atoms)
                splits += int(sub_report.get("splits") or 0)
                report["rejected"].extend(sub_report.get("rejected") or [])
            report["splits"] = splits
            report["status"] = "decoupe"
            report["atoms"] = len(atoms)
            return atoms, report

    atoms, rejected = coerce_atoms(rows or [], batch_pages)
    atoms, ungrounded = drop_ungrounded_atoms(atoms, source_text)

    report["rejected"].extend(rejected)
    report["ungrounded"] = len(ungrounded)
    report["atoms"] = len(atoms)
    report.setdefault("status", "ok" if atoms else "vide")
    return atoms, report


# ---------------------------------------------------------------------------
# Orchestration document
# ---------------------------------------------------------------------------


def extract_atoms_for_document(
    document_id: int,
    *,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    batch_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Dict[str, Any]:
    """Inventorie la matière SAV d'un document déjà traité.

    N'écrit rien : la sortie est un rapport lisible, destiné à être relu avant de
    construire quoi que ce soit (le graphe arrive en L2).
    """
    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document introuvable : {document_id}")

        pages_text = load_pages_text(session, document_id)
        title = document.title or ""
        pdf_path = document.source_file_path or None

    if not pages_text:
        return {
            "document_id": document_id,
            "title": title,
            "status": "no_chunks",
            "atoms": [],
            "batches": [],
            "report": {"pages_lues": 0, "atomes": 0},
        }

    page_numbers = sorted(pages_text.keys())
    if page_start is not None:
        page_numbers = [p for p in page_numbers if p >= page_start]
    if page_end is not None:
        page_numbers = [p for p in page_numbers if p <= page_end]

    if len(page_numbers) > settings.SAV_EXTRACTION_MAX_PAGES:
        logger.warning(
            "[SAV] document_id=%s — %s pages ramenées à %s (SAV_EXTRACTION_MAX_PAGES)",
            document_id,
            len(page_numbers),
            settings.SAV_EXTRACTION_MAX_PAGES,
        )
        page_numbers = page_numbers[: settings.SAV_EXTRACTION_MAX_PAGES]

    batches = build_sav_batches(page_numbers, batch_size=batch_size, overlap=overlap)
    if not batches:
        return {
            "document_id": document_id,
            "title": title,
            "status": "no_pages",
            "atoms": [],
            "batches": [],
            "report": {"pages_lues": 0, "atomes": 0},
        }

    all_atoms: List[SavAtom] = []
    batch_reports: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=settings.SAV_EXTRACTION_CONCURRENCY) as pool:
        futures = {
            pool.submit(
                extract_atoms_for_batch, batch, title, pages_text, pdf_path
            ): batch
            for batch in batches
        }
        for future in as_completed(futures):
            batch = futures[future]
            try:
                atoms, report = future.result()
                all_atoms.extend(atoms)
                batch_reports.append(report)
            except Exception as exc:  # noqa: BLE001
                logger.error("[SAV] Batch %s échoué : %s", batch, exc)
                batch_reports.append({"pages": batch, "status": "erreur", "error": str(exc)[:300]})

    before_dedupe = len(all_atoms)
    atoms = sorted(dedupe_atoms(all_atoms), key=lambda a: (a.page, a.symptome))
    batch_reports.sort(key=lambda r: (r.get("pages") or [0])[0])

    pages_avec_matiere = sorted({a.page for a in atoms})
    logger.info(
        "[SAV] document_id=%s — %s pages, %s batches, %s atomes (%s avant dédoublonnage), modèle=%s",
        document_id,
        len(page_numbers),
        len(batches),
        len(atoms),
        before_dedupe,
        settings.SAV_EXTRACTION_MODEL,
    )

    return {
        "document_id": document_id,
        "title": title,
        "status": "ok" if atoms else "empty",
        "atoms": [a.model_dump() for a in atoms],
        "batches": batch_reports,
        "report": {
            "pages_lues": len(page_numbers),
            "batches": len(batches),
            "atomes": len(atoms),
            "doublons_fusionnes": before_dedupe - len(atoms),
            "pages_avec_matiere": pages_avec_matiere,
            "pages_sans_matiere": [p for p in page_numbers if p not in set(pages_avec_matiere)],
            "batches_decoupes": sum(1 for r in batch_reports if r.get("status") == "decoupe"),
            "modele": settings.SAV_EXTRACTION_MODEL,
        },
    }


# ---------------------------------------------------------------------------
# L2 — assemblage déterministe : atomes → JSON pivot (zéro LLM)
# ---------------------------------------------------------------------------

_EFFORT_ORDER = {"visuel": 0, "outil_simple": 1, "demontage": 2}
_ESCALATION_TITLE = "Rien de tout cela n'a résolu le problème"
_ESCALATION_DESC = (
    "Aucune des causes ci-dessus ne correspond, ou le geste proposé n'a pas résolu le "
    "problème. Contacter le SAV avec le symptôme observé et les vérifications déjà faites."
)
_FALLBACK_CAUSE_TITLE = "Cause à vérifier sur place"

# B3 : au-delà de ce nombre d'enfants, un étage intermédiaire apparaît — la profondeur
# est calculée à partir de la largeur, jamais décidée d'avance. MAX_SIBLINGS compte les
# causes sous UN symptôme, MAX_ROOT_SIBLINGS les symptômes à la racine.
MAX_SIBLINGS = 6
MAX_ROOT_SIBLINGS = 8

# Les familles ne sont PAS une taxonomie codée en dur : une liste de mots-clés bâtie sur
# une notice de serrure ne vaudrait rien pour un volet roulant ou un coulissant. C'est le
# temps B qui les nomme, depuis le vocabulaire du document lui-même (champ `famille`) ;
# le code décide seulement s'il faut un étage, et le valide.
FAMILY_FALLBACK = "Autres cas"
MIN_FAMILIES = 2
MAX_FAMILIES = 7

# Sous UN symptôme, en revanche, le document ne donne aucun thème : la `famille` du temps
# B nomme le symptôme, pas ses causes (« sert seulement à ranger la liste » dans le prompt
# de regroupement). Le seul axe que les atomes portent vraiment à ce niveau est CE QUE LA
# VÉRIFICATION COÛTE — `qui` et `effort`, déjà l'ordre d'affichage des causes. C'est aussi
# la coupure qui compte pour un client : ce qu'il peut regarder lui-même d'un côté, ce qui
# demande un pro de l'autre.
FAMILY_PRO = "Ce qui demande un professionnel"
FAMILY_TOOL = "Ce qui se vérifie avec un outil"
FAMILY_LOOK = "Ce qui se voit ou s'entend"


def _order_families(buckets: Dict[str, List[Any]]) -> List[Tuple[str, List[Any]]]:
    """Familles par ordre alphabétique, le fourre-tout toujours en dernier."""
    return sorted(buckets.items(), key=lambda kv: (kv[0] == FAMILY_FALLBACK, kv[0]))


def family_split_is_useful(buckets: Dict[str, List[Any]], total: int) -> bool:
    """Un étage famille n'aide que s'il range vraiment.

    Trois cas où il nuit : une seule famille (l'étage n'apporte rien) ; trop de familles
    (on remplace une longue liste par une autre) ; un fourre-tout qui avale plus d'un
    tiers des cas — « Autres cas » devient alors un choix de premier niveau où personne
    ne pense à chercher (constaté sur le 3e arbre réel).
    """
    if not MIN_FAMILIES <= len(buckets) <= MAX_FAMILIES:
        return False
    fallback = len(buckets.get(FAMILY_FALLBACK) or [])
    return fallback * 3 <= total


def _dedupe_sibling_titles(items: List[Dict[str, Any]]) -> None:
    """Rend les `nom` uniques parmi des frères — sinon `duplicate_choice_label` bloque
    la publication (seul lint bloquant qu'un pivot valide peut encore déclencher)."""
    seen: Dict[str, int] = {}
    for item in items:
        key = _normalize(item["nom"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            item["nom"] = f"{item['nom']} ({seen[key]})"


def _leaf_type(atom: SavAtom) -> str:
    return "sav" if (atom.qui == "sav" or atom.effort == "demontage") else "solution"


def _leaf_description(atom: SavAtom) -> str:
    parts = []
    if atom.cause and atom.constat_cause:
        # Le constat sert de libellé ; la cause technique n'a donc plus qu'ici pour vivre.
        parts.append(f"Origine probable : {atom.cause}")
    if atom.verification:
        parts.append(f"Vérification : {atom.verification}")
    if atom.geste:
        parts.append(atom.geste)
    return "\n".join(parts) or atom.symptome


# ---------------------------------------------------------------------------
# B5 — la citation doit être littéralement dans sa page, sinon ce n'est pas une citation
# ---------------------------------------------------------------------------


def _norm_verbatim(value: str) -> str:
    """Forme de comparaison d'une citation : sans accents, sans ponctuation, espaces
    écrasés. Tolère la typographie (apostrophes courbes, tirets longs, césures) sans
    tolérer une reformulation."""
    import unicodedata

    text = unicodedata.normalize("NFKD", value or "")
    text = (
        text.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("–", "-").replace("—", "-")
        .replace("­", "")  # trait d'union conditionnel des PDF
    )
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _clean_citation(citation: str) -> str:
    """Retire le balisage que l'extracteur d'ingestion a posé dans le texte transcrit
    (les tableaux de pannes reviennent en markdown : « … : **Solution** : … »). La
    citation reste fidèle, elle est juste lisible."""
    text = re.sub(r"[*_]{1,2}", "", citation or "")
    text = re.sub(r"\s*:\s*Solution\s*:\s*", " — ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def citation_is_verbatim(citation: str, page_text: str) -> bool:
    """Vrai si la citation figure littéralement dans le texte de sa page.

    Mesuré sur le premier arbre réel (Eneo CC) : 8 citations sur 23 étaient des
    paraphrases, dont une qui inversait le sens (« pontera l'interrupteur » devenu
    « activé par erreur »). Une citation qui n'en est pas une ruine la promesse de
    provenance — le SAV croit lire la notice.
    """
    needle = _norm_verbatim(citation)
    if len(needle) < 20:  # trop court pour être une preuve
        return False
    return needle in _norm_verbatim(page_text)


def _leaf_sources(
    atom: SavAtom,
    document_id: int,
    pages_text: Optional[Dict[int, str]],
) -> Tuple[List[Dict[str, Any]], str]:
    """Source du nœud + note interne si la citation n'est pas fiable.

    Sans ``pages_text`` (appel hors base), on ne peut pas vérifier : la citation passe
    telle quelle, comme avant.
    """
    src: Dict[str, Any] = {"document_id": document_id, "page_fichier": atom.page}
    note = ""
    if not atom.citation:
        return [src], note

    if pages_text is None or citation_is_verbatim(atom.citation, pages_text.get(atom.page, "")):
        # La vérification porte sur la citation BRUTE (comparée au texte transcrit, qui
        # contient parfois du markdown de l'extracteur) ; l'affichage, lui, est nettoyé.
        src["precision"] = _clean_citation(atom.citation)[:280]
    else:
        note = (
            f"⚠ Citation non retrouvée telle quelle page {atom.page} — reformulée par "
            f"l'IA, à vérifier : « {atom.citation[:200]} »"
        )
    return [src], note


# ---------------------------------------------------------------------------
# B6 — titre et symptôme de l'arbre : la désignation produit, pas le nom de fichier
# ---------------------------------------------------------------------------

_TITLE_NOISE = re.compile(
    r"\.(pdf|docx?|xlsx?|pptx?)$|\(\d+\)|\bv\d+\b|\b(19|20)\d{2}\b|"
    r"notice(\s+simplifi\w+)?|instructions?\s+de\s+montage",
    re.IGNORECASE,
)


def _clean_document_title(title: str) -> str:
    """Retire ce qui vient du fichier et non du produit : extension, « (1) » de
    téléchargement dupliqué, « v2 », millésime, le mot « notice »."""
    cleaned = _TITLE_NOISE.sub(" ", title or "")
    cleaned = re.sub(r"\s*[-—|]\s*", " - ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -–—|,;")
    return cleaned or (title or "").strip()


def canonical_produits(
    atoms: Sequence["SavAtom"],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Regroupe les désignations qui parlent DU MÊME matériel.

    Défaut mesuré sur le deuxième arbre réel : le modèle écrit tantôt « Eneo CC »,
    tantôt l'en-tête du document « Roto Safe E | Eneo CC ». Comparées par égalité exacte,
    ces deux chaînes créaient un faux étage produit — le client devait choisir entre deux
    noms du même matériel, et les symptômes se retrouvaient éclatés dans les deux
    branches (« la télécommande ne fonctionne pas » apparaissait deux fois).

    Règle déterministe : si une désignation normalisée est contenue dans une autre, c'est
    le même matériel. La forme canonique est la plus fréquente (à égalité, la plus
    courte) — même esprit que la résolution de document par inclusion.

    Returns:
        (clé normalisée → clé canonique, clé canonique → libellé affichable).
    """
    counts: Dict[str, int] = {}
    labels: Dict[str, str] = {}
    for atom in atoms:
        produit = (atom.produit or "").strip()
        if not produit:
            continue
        key = _normalize(produit)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
        labels.setdefault(key, produit)

    # Les plus fréquentes d'abord ; à égalité, la plus courte fait référence.
    ordered = sorted(counts, key=lambda k: (-counts[k], len(k)))
    canonical: Dict[str, str] = {}
    retenues: List[str] = []
    for key in ordered:
        match = next((c for c in retenues if key in c or c in key), None)
        canonical[key] = match or key
        if match is None:
            retenues.append(key)

    return canonical, {k: labels[k] for k in retenues}


def derive_tree_title(atoms: Sequence["SavAtom"], document_title: str) -> str:
    """Désignation produit la plus citée, à défaut le titre du document nettoyé.

    Le titre alimente le picker « Diagnostic SAV » côté client : « Proferm - Eneo CC -
    Notice simplifiée v2 (2022) (1) » n'y a pas sa place.
    """
    canon_of, canon_labels = canonical_produits(atoms)
    if canon_labels:
        counts: Dict[str, int] = {}
        for atom in atoms:
            key = canon_of.get(_normalize(atom.produit or ""))
            if key:
                counts[key] = counts.get(key, 0) + 1
        if counts:
            best, hits = max(counts.items(), key=lambda kv: (kv[1], -len(canon_labels[kv[0]])))
            # Un seul matériel identifié, ou un matériel cité plusieurs fois : c'est le
            # sujet du document. Une mention isolée parmi plusieurs ne l'est pas.
            if hits >= 2 or len(counts) == 1:
                return canon_labels[best]
    return _clean_document_title(document_title)


# ---------------------------------------------------------------------------
# B1 — temps B : canoniser les symptômes (1 appel, texte seul)
# ---------------------------------------------------------------------------

_GROUPING_PROMPT = """Tu regroupes des constats de panne qui parlent DU MÊME problème, pour un
assistant de diagnostic destiné aux poseurs.

On te donne une liste numérotée. Chaque ligne est un constat extrait d'une notice, avec la cause
associée. Plusieurs lignes décrivent souvent le même problème avec des mots différents (elles
viennent de lignes voisines d'un même tableau de pannes).

TA TÂCHE : rendre des groupes. Chaque groupe = un problème, tel que le CLIENT le décrirait.

RÈGLES ABSOLUES :
1. Tout numéro donné doit apparaître dans EXACTEMENT un groupe. N'en oublie aucun, n'en invente
   aucun. C'est vérifié automatiquement et un écart fait tout rejeter.
2. "label" = le problème dans les mots du client, court (moins de 60 caractères), au présent.
   Exemple : « La télécommande ne fait rien », pas « Absence de signal du récepteur radio ».
3. Regroupe ce qui est le même problème vu sous plusieurs angles : une panne de télécommande
   reste une panne de télécommande, que la cause soit la pile, l'association ou le récepteur.
4. Ne regroupe JAMAIS deux constats que le client distingue par un signe qu'il perçoit :
   nombre de bips, code d'erreur affiché, organe utilisé (clé / clavier / télécommande).
   Ces signes sont ce qui permet de trancher plus tard : les fondre ferait perdre l'information.
5. **La CAUSE ne doit JAMAIS apparaître dans le "label".** Le label est ce que le client dit
   AVANT de savoir pourquoi. N'ajoute donc aucune parenthèse de désambiguïsation :
   « ne se verrouille plus tout seul (mode jour) », « (porte mal fermée) », « (aimant mal
   aligné) » sont TROIS FOIS le même label — « Ça ne se verrouille plus tout seul » — et
   donc UN SEUL groupe de trois numéros. La cause se distingue plus loin, pas ici.
   Si tu hésites à écrire une parenthèse, c'est que tu dois fusionner.
6. Un groupe peut ne contenir qu'un seul numéro.
7. "famille" = le grand thème du groupe, tiré du VOCABULAIRE DE CE DOCUMENT, en 2 à
   4 mots ("Manœuvre du volet", "Étanchéité", "Serrure et verrouillage"…). Sert seulement
   à ranger la liste si elle est longue. Utilise le MÊME libellé de famille pour tous les
   groupes qui vont ensemble, et n'en crée pas plus de 6 en tout. Laisse vide si tu
   n'es pas sûr.

SORTIE — JSON strict, rien d'autre :
{"groupes": [{"label": "La télécommande ne fait rien", "famille": "Commande à distance",
"atomes": [3, 7, 12]}]}
"""


def _call_grouping_api(listing: str) -> str:
    from app.services.multimodal_page_service import _mistral_chat_completion

    return _mistral_chat_completion(
        [
            {"role": "system", "content": _GROUPING_PROMPT},
            {"role": "user", "content": listing},
        ],
        page_no=0,
        max_tokens=settings.SAV_EXTRACTION_MAX_TOKENS,
        temperature=0.0,
        response_format_json=True,
        timeout_seconds=settings.SAV_EXTRACTION_TIMEOUT,
        model=settings.SAV_EXTRACTION_MODEL or settings.MULTIMODAL_EXTRACT_MODEL,
    )


def _format_grouping_listing(atoms: Sequence[SavAtom]) -> str:
    lines = []
    for idx, atom in enumerate(atoms):
        cause = f" | cause : {atom.cause.strip()}" if atom.cause.strip() else ""
        lines.append(f"{idx}. {atom.symptome.strip()}{cause}")
    return "\n".join(lines)


def validate_groups(
    groups: Sequence[Dict[str, Any]],
    atom_count: int,
) -> Tuple[bool, str]:
    """Conservation des atomes — le garde-fou qui autorise à faire confiance au temps B.

    Regrouper ne doit JAMAIS perdre un constat : c'est la seule façon de gagner en
    lisibilité sans perdre en couverture. Tout écart → repli sur le regroupement par
    chaîne exacte.
    """
    seen: Dict[int, int] = {}
    for group in groups:
        if not isinstance(group, dict):
            return False, "groupe non-objet"
        indices = group.get("atomes")
        if not isinstance(indices, list) or not indices:
            return False, "groupe sans atome"
        if not str(group.get("label") or "").strip():
            return False, "groupe sans label"
        for raw in indices:
            try:
                idx = int(raw)
            except (TypeError, ValueError):
                return False, f"indice illisible : {raw!r}"
            if not 0 <= idx < atom_count:
                return False, f"indice hors bornes : {idx}"
            if idx in seen:
                return False, f"atome {idx} dans deux groupes"
            seen[idx] = 1

    if len(seen) != atom_count:
        manquants = sorted(set(range(atom_count)) - set(seen))
        return False, f"{len(manquants)} atome(s) non groupé(s) : {manquants[:10]}"
    return True, ""


def canonicalize_symptom_groups(
    atoms: Sequence[SavAtom],
) -> Optional[List[Dict[str, Any]]]:
    """Un appel, texte seul, sur la liste des constats. ``None`` si ça n'aboutit pas —
    l'appelant retombe alors sur le regroupement par chaîne exacte."""
    if len(atoms) < 2:
        return None

    try:
        raw = _call_grouping_api(_format_grouping_listing(atoms))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[SAV] Regroupement des symptômes échoué : %s", exc)
        return None

    rows, truncated = parse_atoms_payload(raw)
    if truncated:
        logger.warning("[SAV] Regroupement tronqué — repli sur le regroupement exact")
        return None
    if rows is None:
        return None
    if not rows:
        try:
            rows = (json.loads(_strip_fences(raw)) or {}).get("groupes") or []
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(rows, list) or not rows:
        return None

    ok, reason = validate_groups(rows, len(atoms))
    if not ok:
        logger.warning("[SAV] Regroupement rejeté (%s) — repli sur le regroupement exact", reason)
        return None
    return list(rows)


_PARENTHETICAL = re.compile(r"\s*[\(\[][^)\]]*[\)\]]\s*")


def merge_parenthetical_groups(
    groups: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fusionne les groupes dont les libellés ne diffèrent que par une parenthèse.

    Défaut mesuré sur le 3e arbre réel : au lieu de regrouper, le modèle désambiguïse —
    « ne se verrouille pas automatiquement (mode jour) », « (porte mal fermée) »,
    « (aimant mal aligné) ». Trois nœuds pour un symptôme, dispersés dans trois familles :
    le client devait devenir devin. Ce qui est entre parenthèses est la CAUSE, elle a sa
    place un étage plus bas, pas dans le libellé du symptôme.

    Le prompt l'interdit désormais explicitement ; ce garde-fou le rend vrai quoi qu'il
    arrive.
    """
    fusion: Dict[str, Dict[str, Any]] = {}
    ordre: List[str] = []
    for group in groups:
        label = str(group.get("label") or "").strip()
        nu = _PARENTHETICAL.sub(" ", label).strip(" -–—:,;")
        key = _normalize(nu)
        if not key:  # libellé entièrement parenthétique : on le garde tel quel
            key = _normalize(label)
            nu = label
        if key not in fusion:
            fusion[key] = {"label": nu or label, "atomes": [], "famille": ""}
            ordre.append(key)
        fusion[key]["atomes"].extend(group.get("atomes") or [])
        # La famille doit SURVIVRE à la fusion : sans elle, `resolve_symptom_groups` ne
        # voit plus aucune famille et l'étage racine de B3 ne se déclenche jamais. Des
        # groupes fusionnés parlent du même problème : la première famille connue vaut
        # pour tous.
        if not fusion[key]["famille"]:
            fusion[key]["famille"] = str(group.get("famille") or "").strip()
        # Le libellé le plus court fait référence (le moins bavard des trois).
        if len(nu) < len(fusion[key]["label"]) and nu:
            fusion[key]["label"] = nu
    return [fusion[k] for k in ordre]


def resolve_symptom_groups(
    atoms: Sequence[SavAtom],
    groups: Optional[Sequence[Dict[str, Any]]],
    warnings: List[str],
) -> Tuple[Dict[int, str], Dict[str, str], Dict[str, str]]:
    """(index d'atome → clé de groupe, clé → libellé, clé → famille).

    Avec ``groups`` valides : les classes du temps B. Sans : regroupement par chaîne
    exacte du symptôme — comportement d'origine, qui produisait 17 symptômes là où il en
    fallait 6, mais qui ne perd jamais rien. Sans temps B, aucune famille n'est connue :
    l'étage intermédiaire ne se déclenche pas, ce qui est le bon comportement (on ne va
    pas inventer une taxonomie).
    """
    if groups:
        ok, reason = validate_groups(groups, len(atoms))
        if ok:
            # La fusion des parenthèses conserve tous les atomes par construction
            # (elle ne fait que réunir des listes) : la validation reste vraie.
            groups = merge_parenthetical_groups(groups)
            group_of: Dict[int, str] = {}
            labels: Dict[str, str] = {}
            familles: Dict[str, str] = {}
            for g_idx, group in enumerate(groups):
                key = f"g{g_idx:03d}"
                labels[key] = str(group["label"]).strip()[:120]
                famille = str(group.get("famille") or "").strip()[:80]
                if famille:
                    familles[key] = famille
                for raw in group["atomes"]:
                    group_of[int(raw)] = key
            return group_of, labels, familles
        warnings.append(f"Regroupement ignoré ({reason}) — symptômes groupés à l'identique.")

    group_of = {}
    labels = {}
    for idx, atom in enumerate(atoms):
        key = _normalize(atom.symptome) or f"s{idx}"
        group_of[idx] = key
        labels.setdefault(key, atom.symptome.strip())
    return group_of, labels, {}


CONSTAT_SIMILARITY = 0.7


def _digits(value: str) -> frozenset:
    return frozenset(re.findall(r"\d+", value or ""))


def same_constat(a: str, b: str) -> bool:
    """Deux formulations désignent-elles le MÊME signe observable ?

    Défaut mesuré sur le 3e arbre réel : « Aucun bip de confirmation de 2 secondes après
    appui… » et « Aucun bip sonore de confirmation après appui… » donnaient deux boutons
    pour un seul signe — l'égalité de chaîne exacte ne voit pas la paraphrase.

    Garde-fou indispensable : **si les deux portent des chiffres et qu'ils diffèrent, ce
    ne sont PAS les mêmes** (« il bipe 2 fois » ≠ « il bipe 3 fois »). Ce nombre est
    précisément ce qui permet au client de trancher.
    """
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return False

    da, db = _digits(a), _digits(b)
    if da and db and da != db:
        return False

    return len(ta & tb) / len(ta | tb) >= CONSTAT_SIMILARITY


def cluster_by_constat(atoms: Sequence[SavAtom]) -> List[List[SavAtom]]:
    """Regroupe les causes qui partagent le même signe observable (paraphrases incluses)."""
    clusters: List[List[SavAtom]] = []
    for atom in atoms:
        cible = next(
            (c for c in clusters if same_constat(c[0].constat_cause, atom.constat_cause)),
            None,
        )
        if cible is None:
            clusters.append([atom])
        else:
            cible.append(atom)
    return clusters


def _merged_entry(
    entry_id: str,
    parent_id: str,
    atoms: List[SavAtom],
    document_id: int,
    pages_text: Optional[Dict[int, str]],
    *,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Une feuille unique pour PLUSIEURS causes que le client ne peut pas départager.

    Deux usages :
      * ``label=None`` — les causes sans signe observable. Cas réel : « ne réagit pas,
        aucun son » a cinq causes (220 V, 24 V ×2, polarité, unité motrice) qu'aucun
        client ne distingue sans multimètre. En faire cinq boutons, c'est lui demander
        de deviner ; on en fait une feuille qui énumère les contrôles.
      * ``label`` fourni — plusieurs causes derrière le MÊME signe observable. Cas réel :
        « l'Eneo bipe 3 fois » vaut pour un corps étranger ET pour un défaut
        d'alignement. Deux boutons identiques (« bipe 3 fois » et « bipe 3 fois (2) »)
        n'aident personne : un seul bouton, deux causes listées derrière.
    """
    lines: List[str] = []
    sources: List[Dict[str, Any]] = []
    notes: List[str] = []
    for atom in atoms:
        bullet = atom.cause.strip() or atom.symptome.strip()
        detail = atom.geste.strip() or atom.verification.strip()
        lines.append(f"- {bullet}" + (f" → {detail}" if detail else ""))
        atom_sources, note = _leaf_sources(atom, document_id, pages_text)
        sources.extend(atom_sources)
        if note:
            notes.append(note)

    # Le symptôme parent donne déjà le contexte : pas besoin d'y accoler un thème.
    nom = label or "Contrôles à effectuer sur place"
    entete = "Causes possibles :" if len(lines) > 1 else "À vérifier :"
    entry = {
        "id": entry_id,
        "nom": nom,
        # Un seul de ces contrôles réservé au SAV suffit à réserver la feuille au SAV :
        # on ne fait pas mesurer du 220 V à un client.
        "type": "sav" if any(_leaf_type(a) == "sav" for a in atoms) else "solution",
        "parents": [parent_id],
        "description": f"{entete}\n" + "\n".join(lines),
        "sources": sources,
    }
    if notes:
        entry["note_interne"] = "\n".join(notes)
    return entry


def _cause_entries_for_symptom(
    symptome_id: str,
    atoms: List[SavAtom],
    document_id: int,
    pages_text: Optional[Dict[int, str]],
) -> List[Dict[str, Any]]:
    """Les enfants d'un symptôme : un bouton par signe observable DISTINCT, plus une
    feuille unique regroupant les causes sans signe observable."""
    discernables = [a for a in atoms if a.constat_cause.strip()]
    indiscernables = [a for a in atoms if not a.constat_cause.strip()]

    # Du geste le plus léger au plus lourd : on ne fait pas démonter avant de faire regarder.
    discernables.sort(
        key=lambda a: (_EFFORT_ORDER.get(a.effort, 1), a.constat_cause or a.cause)
    )

    # Plusieurs causes derrière le même signe observable → UN bouton, pas deux boutons
    # que rien ne distingue à l'écran.
    clusters = cluster_by_constat(discernables)

    entries: List[Dict[str, Any]] = []
    for idx, group in enumerate(clusters, start=1):
        entry_id = f"{symptome_id}_c{idx}"
        if len(group) > 1:
            entries.append(
                _merged_entry(
                    entry_id,
                    symptome_id,
                    group,
                    document_id,
                    pages_text,
                    label=group[0].constat_cause.strip(),
                )
            )
            continue
        atom = group[0]
        source, note = _leaf_sources(atom, document_id, pages_text)
        entry = {
            "id": entry_id,
            "nom": atom.constat_cause.strip(),
            "type": _leaf_type(atom),
            "parents": [symptome_id],
            "description": _leaf_description(atom),
            "sources": source,
        }
        if note:
            entry["note_interne"] = note
        entries.append(entry)

    if indiscernables:
        entries.append(
            _merged_entry(
                f"{symptome_id}_chk", symptome_id, indiscernables, document_id, pages_text
            )
        )

    _dedupe_sibling_titles(entries)
    return entries


def build_pivot_from_atoms(
    raw_atoms: Sequence[Dict[str, Any]],
    *,
    document_id: int,
    document_title: str,
    tree_title: Optional[str] = None,
    pages_text: Optional[Dict[int, str]] = None,
    groups: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Assemble des atomes en JSON pivot. Aucun appel modèle ici.

    La FORME suit la matière (B3) au lieu d'un moule à trois niveaux :
      * causes indiscernables sur place → une feuille-checklist ;
      * une seule cause réelle → le symptôme DEVIENT la feuille (plus d'étage question
        inutile : sur le premier arbre réel, 14 symptômes sur 17 étaient dans ce cas) ;
      * plus de MAX_SIBLINGS enfants → un étage par famille de vérification ;
      * plus de MAX_ROOT_SIBLINGS symptômes → un étage famille à la racine.

    Args:
        pages_text: texte par page, pour vérifier que chaque citation est littérale (B5).
            Absent → pas de vérification (la citation passe telle quelle).
        groups: classes d'équivalence de symptômes issues du temps B (B1). Absent →
            regroupement par chaîne exacte, comme avant.

    Returns:
        (payload pivot, avertissements).
    """
    warnings: List[str] = []
    atoms: List[SavAtom] = []
    for row in raw_atoms:
        try:
            atoms.append(row if isinstance(row, SavAtom) else SavAtom.model_validate(row))
        except ValidationError as exc:
            warnings.append(f"Atome ignoré (invalide) : {exc.errors()[0].get('loc')}")

    title = tree_title or derive_tree_title(atoms, document_title) or "Diagnostic"
    if not atoms:
        return {"titre": title, "cas": []}, warnings

    group_of, group_labels, group_familles = resolve_symptom_groups(atoms, groups, warnings)

    # Étage produit : seulement s'il reste au moins DEUX matériels distincts après
    # regroupement des désignations (« Eneo CC » et « Roto Safe E | Eneo CC » sont le même).
    canon_of, canon_labels = canonical_produits(atoms)
    use_produit_level = len(canon_labels) >= 2

    cas: List[Dict[str, Any]] = []
    atoms_by_entry: Dict[str, SavAtom] = {}

    produit_buckets: Dict[str, List[int]] = {}
    produit_labels: Dict[str, str] = dict(canon_labels)
    for idx, atom in enumerate(atoms):
        key = ""
        if use_produit_level and atom.produit.strip():
            key = canon_of.get(_normalize(atom.produit), "")
        produit_buckets.setdefault(key, []).append(idx)
    produit_labels.setdefault("", "Autres cas")

    aiguillage_symptomes: List[str] = []

    for p_idx, (p_key, p_indices) in enumerate(sorted(produit_buckets.items()), start=1):
        parent_ids: List[str] = []
        prefix = ""
        if use_produit_level:
            produit_id = f"p{p_idx}"
            cas.append(
                {
                    "id": produit_id,
                    "nom": produit_labels[p_key],
                    "type": "aiguillage",
                    "parents": [],
                    "description": f"Matériel concerné : {produit_labels[p_key]}.",
                }
            )
            parent_ids = [produit_id]
            prefix = f"p{p_idx}_"

        # Symptômes de ce produit, dans l'ordre des groupes.
        symptome_buckets: Dict[str, List[SavAtom]] = {}
        for idx in p_indices:
            symptome_buckets.setdefault(group_of[idx], []).append(atoms[idx])

        symptome_entries: List[Dict[str, Any]] = []
        # `group_familles` est indexé par clé de groupe ; l'étage racine range des entrées.
        famille_of_entry: Dict[str, str] = {}
        for s_idx, (g_key, s_atoms) in enumerate(sorted(symptome_buckets.items()), start=1):
            symptome_id = f"{prefix}s{s_idx}"
            famille_of_entry[symptome_id] = group_familles.get(g_key, "")
            children = _cause_entries_for_symptom(
                symptome_id, s_atoms, document_id, pages_text
            )
            for entry in children:
                match = next(
                    (
                        a
                        for a in s_atoms
                        if a.constat_cause.strip() == entry["nom"]
                        or entry["id"].endswith("_chk")
                    ),
                    None,
                )
                if match:
                    atoms_by_entry[entry["id"]] = match

            label = group_labels.get(g_key) or s_atoms[0].symptome.strip()

            # UNE seule cause réelle → le symptôme devient la feuille. Pas de question
            # à un seul choix : c'est un clic qui n'apprend rien.
            if len(children) == 1:
                leaf = children[0]
                collapsed = {
                    "id": symptome_id,
                    "nom": label,
                    "type": leaf["type"],
                    "parents": list(parent_ids),
                    "description": leaf["description"],
                    "sources": leaf.get("sources") or [],
                }
                if leaf.get("note_interne"):
                    collapsed["note_interne"] = leaf["note_interne"]
                symptome_entries.append(collapsed)
                continue

            symptome_entries.append(
                {
                    "id": symptome_id,
                    "nom": label,
                    "type": "aiguillage",
                    "parents": list(parent_ids),
                    "description": f"Le client signale : {label}",
                }
            )
            aiguillage_symptomes.append(symptome_id)

            if len(children) > MAX_SIBLINGS:
                children = _split_by_family(symptome_id, children, atoms_by_entry)
            cas.extend(children)

        _dedupe_sibling_titles(symptome_entries)

        # B3 — trop de symptômes au premier niveau : un étage famille apparaît.
        if not use_produit_level and len(symptome_entries) > MAX_ROOT_SIBLINGS:
            symptome_entries = _split_root_by_family(symptome_entries, famille_of_entry)

        cas.extend(symptome_entries)

    if use_produit_level:
        _dedupe_sibling_titles([c for c in cas if c["parents"] == []])

    # B4 — UNE sortie SAV partagée, rattachée à tous les symptômes restés des questions.
    # Le format pivot gère le multi-parent (`parents` est une liste) : 17 nœuds → 1.
    # Si tous les symptômes se sont effondrés en feuilles, il ne reste aucune question à
    # qui la rattacher : elle passe alors au premier niveau, pour que le client garde une
    # porte de sortie et que `no_escalation_branch` ne se déclenche jamais.
    cas.append(
        {
            "id": "sav_global",
            "nom": _ESCALATION_TITLE if aiguillage_symptomes else "Aucun de ces cas",
            "type": "sav",
            "parents": list(aiguillage_symptomes),
            "description": _ESCALATION_DESC,
        }
    )

    # Sans `symptome`, l'arbre resterait catalogué « Autre » dans le picker client
    # (list_sav_entries retombe sur ce libellé quand `entry_symptom` est vide).
    # Sans `question_depart`, le nœud racine arrivait avec un message vide : LIA n'avait
    # aucune matière pour rédiger la première question.
    question = (
        "De quel matériel s'agit-il ?"
        if use_produit_level
        else "Qu'est-ce que vous constatez ?"
    )
    return (
        {"titre": title, "symptome": title, "question_depart": question, "cas": cas},
        warnings,
    )


def _verification_family(entry: Dict[str, Any], atom: Optional[SavAtom]) -> str:
    """Famille d'une cause : d'abord son coût de vérification, jamais un thème deviné."""
    # `type == "sav"` couvre déjà `qui == "sav"` ET `effort == "demontage"` (cf. _leaf_type),
    # y compris pour une feuille fusionnée dont une seule cause est réservée au SAV.
    if entry.get("type") == "sav":
        return FAMILY_PRO
    if atom is None:
        return FAMILY_FALLBACK
    return FAMILY_TOOL if atom.effort == "outil_simple" else FAMILY_LOOK


def _split_by_family(
    symptome_id: str,
    children: List[Dict[str, Any]],
    atoms_by_entry: Dict[str, SavAtom],
) -> List[Dict[str, Any]]:
    """Étage intermédiaire sous un symptôme qui a trop de causes discriminables.

    Même garde-fou qu'à la racine : si le rangement n'aide pas (une seule famille, ou un
    fourre-tout trop gros), la liste reste plate. Sept libellés courts à lire valent mieux
    qu'un clic de plus qui ne range rien.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for entry in children:
        famille = _verification_family(entry, atoms_by_entry.get(entry["id"]))
        buckets.setdefault(famille, []).append(entry)

    if not family_split_is_useful(buckets, len(children)):
        return children

    out: List[Dict[str, Any]] = []
    for f_idx, (famille, group) in enumerate(_order_families(buckets), start=1):
        family_id = f"{symptome_id}_f{f_idx}"
        out.append(
            {
                "id": family_id,
                "nom": famille,
                "type": "aiguillage",
                "parents": [symptome_id],
                "description": f"{famille}.",
            }
        )
        for child in group:
            child["parents"] = [family_id]
        out.extend(group)
    return out


def _split_root_by_family(
    symptome_entries: List[Dict[str, Any]],
    familles: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Étage intermédiaire à la racine quand les symptômes sont trop nombreux à scanner.

    Les libellés de famille viennent du temps B (donc du vocabulaire du document), pas
    d'une liste de mots-clés : une taxonomie écrite pour une serrure ne rangerait rien
    dans une notice de volet roulant.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for entry in symptome_entries:
        famille = (familles.get(entry["id"]) or "").strip() or FAMILY_FALLBACK
        buckets.setdefault(famille, []).append(entry)

    if not family_split_is_useful(buckets, len(symptome_entries)):
        return symptome_entries

    out: List[Dict[str, Any]] = []
    for f_idx, (famille, children) in enumerate(_order_families(buckets), start=1):
        family_id = f"fam{f_idx}"
        out.append(
            {
                "id": family_id,
                "nom": famille,
                "type": "aiguillage",
                "parents": [],
                "description": f"Ce qui concerne : {famille.lower()}.",
            }
        )
        for child in children:
            child["parents"] = [family_id]
        out.extend(children)
    return out
