"""
Chunks contextuels inter-pages : synthèse factuelle par thème.

Fenêtre glissante de 3 pages (overlap 1) :
  1. Texte L1 transcrit + IMAGES des pages (vision systématique depuis 2026-07-28)
  2. Appel LLM multimodal → chunks de synthèse documentaire (1 par notion/thème)
  3. Contrôle déterministe : tout nombre non ancré dans le texte L1 fait rejeter le chunk
  4. Persistance content_type=contextual_enrichment + embedding

C'est le SEUL étage de la pipeline où un modèle voit une page avec le droit de
synthétiser : la passe d'extraction, elle, a interdiction de décrire. D'où la vision
systématique — sans elle, aucune description de schéma n'existerait nulle part.

Ces chunks servent au retrieval (vectoriel + BM25) comme contexte complémentaire,
jamais comme preuve absolue — toujours rattachés aux pages sources.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text
from sqlmodel import Session, select

from app.config import settings
from app.database import engine
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.document_space import DocumentSpace
from app.models.chunk_category_relation import ChunkCategoryRelation

logger = logging.getLogger(__name__)

CONTEXTUAL_ENRICHMENT_VERSION = "contextual_enrichment_v3"
CONTENT_TYPE_CONTEXTUAL_ENRICHMENT = "contextual_enrichment"


def build_page_batches(
    page_numbers: List[int],
    *,
    batch_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[List[int]]:
    """Fenêtre glissante de pages : batches de ``batch_size`` avec ``overlap`` pages
    de recouvrement (3 et 1 par défaut → 1-2-3, 3-4-5, 5-6-7…).

    Le recouvrement coûte ~50 % d'appels supplémentaires mais évite de couper une
    notion sur une frontière de batch.

    Rapatrié de ``kag_extraction_service.build_kag_batches`` lors du retrait du KAG
    (2026-07-28) : c'est désormais l'enrichissement contextuel qui en est le seul
    consommateur.
    """
    if not page_numbers:
        return []

    size = batch_size if batch_size is not None else settings.CONTEXTUAL_ENRICHMENT_BATCH_SIZE
    overlap_val = (
        overlap if overlap is not None else settings.CONTEXTUAL_ENRICHMENT_BATCH_OVERLAP
    )
    size = max(1, size)
    overlap_val = max(0, min(overlap_val, size - 1))
    stride = max(1, size - overlap_val)

    sorted_pages = sorted(set(page_numbers))
    batches: List[List[int]] = []
    i = 0
    while i < len(sorted_pages):
        batch = sorted_pages[i : i + size]
        if batch:
            batches.append(batch)
        if i + size >= len(sorted_pages):
            break
        i += stride
    return batches

# Rôles de chunk L2 orientés accompagnement (au lieu d'une synthèse plate unique)
CHUNK_ROLE_SYNTHESIS = "synthesis"
CHUNK_ROLE_PROCEDURAL = "procedural_step"
CHUNK_ROLE_DIAGNOSTIC = "diagnostic_unit"
_VALID_CHUNK_ROLES = frozenset(
    {CHUNK_ROLE_SYNTHESIS, CHUNK_ROLE_PROCEDURAL, CHUNK_ROLE_DIAGNOSTIC}
)
# section_type L1 considérés "visuels" → déclenchent l'enrichissement multimodal sélectif

_ENRICHMENT_SYSTEM_PROMPT = """Tu es technicien expert en menuiserie (profilés PVC, aluminium et hybrides PVC-alu)
ET rédacteur d'une base de connaissances RAG destinée à des poseurs, techniciens SAV et conseillers.
On te donne le texte transcrit de plusieurs pages consécutives d'un document technique
ET les images de ces pages.

Ta mission n'est PAS de recopier ni de résumer platement : tu RÉÉCRIS et EXPLICITES le contenu avec
ton expertise métier. Tu rends explicite ce que le document tient pour implicite — la FONCTION et le
RÔLE de chaque élément dans la menuiserie, son emplacement, son matériau, ce qu'il permet de réaliser
et ses conditions de mise en œuvre. Le résultat doit se lire comme une fiche technique professionnelle,
dense et autonome : un assistant doit pouvoir y répondre sans avoir relu la page source.
Bannis le style « liste à plat » : une énumération de codes ou de cotes doit toujours être précédée de
ce que la famille d'éléments FAIT, puis détaillée élément par élément.

Si le message utilisateur contient une section « CONSIGNES DE RÉÉCRITURE PAR CATÉGORIE »,
applique-la ; sinon, adapte la réécriture au type de contenu que tu observes.

Produis des chunks factuels, un par thème/notion. Chaque chunk a un RÔLE adapté à son usage final :
accompagnement chantier (pose) ou SAV (diagnostic).

Choix du rôle (champ chunk_role) selon le CONTENU :
- "procedural_step" : le contenu décrit une ÉTAPE de pose/montage/réglage.
  Découpe la procédure en étapes ORDONNÉES (un chunk par étape).
- "diagnostic_unit" : le contenu décrit un PROBLÈME/SYMPTÔME SAV et sa résolution.
- "synthesis" : sinon (spécifications, commercial, garantie, normes…). Par défaut.

Règles absolues :
1. Renvoie UNIQUEMENT un objet JSON valide (aucun texte hors JSON).
2. Utilise les noms de produits, gammes, références et normes EXACTEMENT tels qu'ils apparaissent dans le texte source.
3. Jamais de déictiques : interdiction de "ce schéma", "ce profil", "celui-ci", "l'image montre", "voir ci-dessus".
   Remplace par le nom précis ("le profilé PVC Kömmerling 76 AD", "la gâche OB Droite", "la gamme Perform", etc.).
4. "content" = texte autonome, dense, TECHNIQUE et EXPLICITE, style fiche/notice professionnelle,
   directement réutilisable par un LLM RAG. Explique systématiquement la fonction et le rôle métier
   des éléments (jamais une simple liste : décris ce que la famille de pièces FAIT, puis détaille chaque référence).
5. Chaque chunk = 1 notion/étape/symptôme. Maximum 600 tokens (~2200 caractères) par chunk.
6. Tu peux mobiliser le savoir métier menuiserie STANDARD pour expliciter la FONCTION d'un type d'élément
   (ex. rôle d'un habillage, d'une garniture de joint, d'un renfort, d'un seuil PMR). Mais n'invente JAMAIS
   de valeur, cote, référence, performance, norme ou nom de gamme absent du texte source.
6bis. IMAGES — mandat STRICTEMENT limité. Quand des images de pages sont fournies, tu peux
   décrire UNIQUEMENT :
     - les VERDICTS visuels explicites : ce qui est coché, validé, barré, entouré, marqué
       d'une croix ou d'un pictogramme d'interdiction ;
     - les RELATIONS et l'ORDRE : quelle pièce se monte sur quelle autre, la séquence d'un
       geste, le repère chiffré qui désigne une étape.
   INTERDICTION ABSOLUE de produire un NOMBRE (cote, dimension, angle, couple, référence)
   qui ne figure pas déjà dans le texte transcrit fourni. Lire une valeur sur un dessin est
   hors mandat : tout nombre non présent dans le texte fera REJETER le chunk entier.
7. Croise les informations réparties sur les pages du batch quand elles concernent le même thème.
8. category_slug = FACULTATIF. Ne le renseigne que si une liste de catégories t'est
   fournie et qu'un slug s'applique exactement ; sinon laisse-le à null.
9. source_page = page principale où le thème est le plus documenté.
10. Champs structurés selon le rôle (laisse vide/null si non applicable) :
    - procedural_step : step_number (ordre), action (geste précis), components[], tools[], dimensions[], precaution, next_condition (condition de passage à l'étape suivante).
    - diagnostic_unit : symptom (slug ou libellé), probable_cause, verification (test à effectuer), resolution (correction).
    Le champ "content" reste TOUJOURS rempli (résumé dense), même quand les champs structurés le sont.

Format de réponse OBLIGATOIRE :
{{
  "enrichment_chunks": [
    {{
      "category_slug": "mounting",
      "chunk_role": "procedural_step",
      "theme": "Pose du profil seuil Profine 76 — étape 1",
      "content": "...",
      "source_page": 2,
      "step_number": 1,
      "action": "Positionner le seuil ... ",
      "components": ["seuil PMR Profine 76"],
      "tools": ["visseuse"],
      "dimensions": ["jeu 5 mm"],
      "precaution": "Ne pas percer la zone d'étanchéité",
      "next_condition": "Seuil calé et de niveau",
      "symptom": null,
      "probable_cause": null,
      "verification": null,
      "resolution": null
    }}
  ]
}}"""

_ENRICHMENT_USER_TEMPLATE = (
    "Document : {title}\nPages du batch : {page_range}\n\n"
    "Texte transcrit par page :\n{page_text}\n\n"
    "Catégories détectées par page :\n{categories_text}\n\n"
    "Entités extraites par page :\n{entities_text}\n\n"
    "CONSIGNES DE RÉÉCRITURE PAR CATÉGORIE (applique celles dont la catégorie est présente ci-dessus) :\n"
    "{playbooks}\n\n"
    "Réécris et explicite les chunks selon ces consignes et les règles du système."
)


# ---------------------------------------------------------------------------
# Consignes de réécriture par catégorie (« playbooks »)
# ---------------------------------------------------------------------------
# Chaque entrée dit au LLM SOUS QUEL ANGLE et AVEC QUELLE STRUCTURE réécrire le
# contenu d'une catégorie, pour passer d'une transcription plate à une fiche
# technique métier explicite. Seules les consignes des catégories présentes dans
# le batch sont injectées dans le prompt (voir `_build_playbook_section`).
# Clés = slugs de l'axe `task` (cf. category_catalog.CONTENT_CATEGORY_SLUGS).
CATEGORY_ENRICHMENT_PLAYBOOKS: Dict[str, str] = {
    "parts_references": (
        "Ne te contente JAMAIS de lister les codes. Regroupe les pièces par FAMILLE FONCTIONNELLE "
        "(habillages, garnitures de joint, renforts, tapées/appuis, embouts, profilés complémentaires…). "
        "Pour chaque famille, écris d'abord 1 phrase sur sa FONCTION dans la menuiserie (rôle, emplacement, "
        "matériau), puis détaille chaque référence : code exact + désignation + cotes (largeur/épaisseur/"
        "inertie avec unité) + matériau (PVC, alu, EPDM, TPE) + ce qu'elle permet de réaliser ou sa compatibilité."
    ),
    "mounting": (
        "Réécris en séquence opératoire de chantier. Pour chaque opération : le geste précis, l'outil, "
        "les cotes/jeux à respecter, le pas de vissage/fixation (en mm), et la précaution (zone à ne pas "
        "percer, sens du joint, ordre de calage). Explique le POURQUOI quand le texte le permet (report de "
        "charge, étanchéité, dilatation). Découpe en étapes ORDONNÉES (rôle procedural_step)."
    ),
    "hardware_adjustment": (
        "Décris l'organe de quincaillerie (gâche, roulette, charnière, compas, crémone…), son réglage "
        "(sens, amplitude, outil/clé), l'EFFET observable du réglage et le repère de bon réglage. "
        "Relie symptôme → réglage correctif quand c'est pertinent (utile au SAV)."
    ),
    "sealing": (
        "Explique la fonction d'étanchéité (air et/ou eau) de chaque garniture, joint ou bavette : où elle "
        "s'applique, comment elle se monte (sens, continuité, recouvrement, retours d'angle), le pas de "
        "fixation et le matériau (EPDM, TPE, mastic). Précise le rôle (barrière à l'eau, étanchéité à l'air, "
        "drainage) et la conséquence d'une pose défectueuse."
    ),
    "drilling_constraints": (
        "Formule sans ambiguïté ce qui PEUT et ce qui NE PEUT PAS être percé/usiné, les zones INTERDITES "
        "(chambres de renfort, canaux de drainage, plans d'étanchéité) et la CONSÉQUENCE d'un non-respect. "
        "Mets les interdictions en tête et garde les formulations impératives."
    ),
    "dimensions_tolerances": (
        "Restitue chaque cote chiffrée avec son unité, l'élément concerné et son contexte (mini/maxi, "
        "tolérance, faux-aplomb en mm/m, jeu de pose, entraxe). Indique la limite à ne pas dépasser et "
        "ce qui se passe au-delà (perte d'étanchéité, jeu de manœuvre, refus de garantie)."
    ),
    "load_capacity": (
        "Explique les limites structurelles : poids max de vantail/remplissage, report de charge, entraxe "
        "des pattes, inertie des renforts (en cm⁴) et ce que la valeur IMPLIQUE concrètement (rigidité, "
        "hauteur/largeur max admissible). Relie chaque renfort à la capacité qu'il autorise."
    ),
    "material_profile": (
        "Décris la nature et la composition du profilé/matériau (PVC, alu, hybride PVC-alu), sa structure "
        "(nombre de chambres, renfort acier, rupture de pont thermique) et ses propriétés fonctionnelles "
        "(isolation, rigidité, tenue). Donne les caractéristiques chiffrées et explique leur portée d'emploi."
    ),
    "glazing": (
        "Détaille le vitrage : composition (épaisseurs des verres, lame d'air/gaz), performances thermiques "
        "(Ug/Uw) et acoustiques (Rw), épaisseur de remplissage admissible et l'effet sur le confort. "
        "Relie chaque spécification à la performance qu'elle apporte."
    ),
    "product_range": (
        "Identifie la gamme/famille produit et ses variantes ; resitue chaque référence dans sa gamme et "
        "son domaine d'emploi (frappe, coulissant, PMR…). Caractérise ce qui distingue la gamme et son "
        "positionnement, sans argumentaire creux."
    ),
    "regulatory": (
        "Restitue l'exigence normative (DTU, NF EN, PMR, DTA, Avis Technique) précisément : ce qu'elle "
        "impose, à quel élément/ouvrage, et l'OBLIGATION concrète qui en découle pour la pose ou le produit. "
        "Cite la référence du document normatif EXACTEMENT telle qu'écrite."
    ),
    "warranty": (
        "Précise la durée, le périmètre couvert, les conditions à respecter et les EXCLUSIONS. Formule en "
        "termes d'engagement et de ce qui fait PERDRE la garantie (défaut de pose, modification non autorisée)."
    ),
    "certification": (
        "Précise le marquage/certificat (CE, label, PV d'essai, classement AEV), ce qu'il ATTESTE, "
        "l'organisme et la portée. Relie l'attestation à la performance prouvée (étanchéité, résistance)."
    ),
    "commercial": (
        "Reformule l'argumentaire en bénéfices CONCRETS et caractéristiques vérifiables (design, finitions, "
        "performances), sans emphase marketing. Garde uniquement les faits exploitables en conseil client."
    ),
    "product_comparison": (
        "Structure la comparaison PAR CRITÈRE (performance, dimensions, usage, finition) entre produits ou "
        "gammes, puis conclus sur le CAS D'EMPLOI de chacun pour guider le choix."
    ),
    "troubleshooting": (
        "Structure en unité de diagnostic (rôle diagnostic_unit) : symptôme observable côté client, cause "
        "probable, vérification à effectuer, correction. Emploie à la fois le vocabulaire client et le "
        "vocabulaire technique pour que la requête utilisateur matche."
    ),
}

_DEFAULT_PLAYBOOK = (
    "Réécris de façon technique, dense et explicite : nomme précisément chaque élément, explicite sa "
    "fonction et sa portée métier, conserve toutes les valeurs et références exactes du texte source."
)


class EnrichmentChunkItem(BaseModel):
    # Optionnel depuis le retrait du KAG (2026-07-28) : plus aucune catégorie n'est
    # détectée en amont, la catégorisation reviendra par un autre mécanisme.
    category_slug: Optional[str] = None
    theme: str
    content: str
    source_page: int = Field(ge=1)
    chunk_role: str = CHUNK_ROLE_SYNTHESIS
    # procedural_step
    step_number: Optional[int] = None
    action: Optional[str] = None
    components: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    precaution: Optional[str] = None
    next_condition: Optional[str] = None
    # diagnostic_unit
    symptom: Optional[str] = None
    probable_cause: Optional[str] = None
    verification: Optional[str] = None
    resolution: Optional[str] = None

    def structured_metadata(self) -> Dict[str, object]:
        """Champs structurés à sérialiser en metadata_json (selon le rôle)."""
        if self.chunk_role == CHUNK_ROLE_PROCEDURAL:
            return {
                "step_number": self.step_number,
                "action": self.action,
                "components": self.components,
                "tools": self.tools,
                "dimensions": self.dimensions,
                "precaution": self.precaution,
                "next_condition": self.next_condition,
            }
        if self.chunk_role == CHUNK_ROLE_DIAGNOSTIC:
            return {
                "symptom": self.symptom,
                "probable_cause": self.probable_cause,
                "verification": self.verification,
                "resolution": self.resolution,
            }
        return {}


class BatchEnrichmentResponse(BaseModel):
    enrichment_chunks: List[EnrichmentChunkItem] = Field(default_factory=list)


def _enrichment_model() -> str:
    return settings.CONTEXTUAL_ENRICHMENT_MODEL or settings.MODEL_FAST


def _load_semantic_chunks_by_page(session: Session, document_id: int) -> Dict[int, List[DocumentChunk]]:
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == True,  # noqa: E712
    )
    by_page: Dict[int, List[DocumentChunk]] = {}
    for chunk in session.exec(stmt).all():
        meta = chunk.metadata_json or {}
        if meta.get("content_type") != "semantic_leaf":
            continue
        page_no = meta.get("page_no") or meta.get("page_start")
        if page_no is None:
            continue
        by_page.setdefault(int(page_no), []).append(chunk)
    for pno in by_page:
        by_page[pno].sort(key=lambda c: (c.chunk_index or 0, c.id or 0))
    return by_page


def _load_page_anchors(session: Session, document_id: int) -> Dict[int, DocumentChunk]:
    stmt = select(DocumentChunk).where(
        DocumentChunk.document_id == document_id,
        DocumentChunk.is_leaf == False,  # noqa: E712
    )
    anchors: Dict[int, DocumentChunk] = {}
    for chunk in session.exec(stmt).all():
        meta = chunk.metadata_json or {}
        if meta.get("content_type") != "page_anchor":
            continue
        page_no = meta.get("page_no")
        if page_no is not None:
            anchors[int(page_no)] = chunk
    return anchors


def _load_categories_by_page(
    session: Session,
    document_id: int,
) -> Dict[int, List[str]]:
    rows = session.execute(
        text(
            """
            SELECT ccr.page_no, dc.slug
            FROM chunkcategoryrelation ccr
            INNER JOIN documentcategory dc ON dc.id = ccr.category_id
            WHERE ccr.document_id = :doc_id
            ORDER BY ccr.page_no, dc.slug
            """
        ),
        {"doc_id": document_id},
    ).all()
    by_page: Dict[int, List[str]] = {}
    for page_no, slug in rows:
        pno = int(page_no or 0)
        by_page.setdefault(pno, [])
        if slug and slug not in by_page[pno]:
            by_page[pno].append(slug)
    return by_page


def _load_entities_by_page(
    session: Session,
    document_id: int,
) -> Dict[int, List[str]]:
    rows = session.execute(
        text(
            """
            SELECT
                COALESCE(
                    (dc2.metadata_json->>'page_no')::int,
                    (dc2.metadata_->>'page_no')::int
                ) AS page_no,
                ke.name
            FROM chunkentityrelation cer
            INNER JOIN documentchunk dc2 ON dc2.id = cer.chunk_id
            INNER JOIN knowledgeentity ke ON ke.id = cer.entity_id
            WHERE dc2.document_id = :doc_id
            """
        ),
        {"doc_id": document_id},
    ).all()
    by_page: Dict[int, List[str]] = {}
    for page_no, name in rows:
        if not page_no or not name:
            continue
        pno = int(page_no)
        by_page.setdefault(pno, [])
        if name not in by_page[pno]:
            by_page[pno].append(name)
    return by_page


def _format_batch_context(
    batch_pages: List[int],
    chunks_by_page: Dict[int, List[DocumentChunk]],
    categories_by_page: Dict[int, List[str]],
    entities_by_page: Dict[int, List[str]],
) -> Tuple[str, str, str]:
    page_text_parts: List[str] = []
    cat_parts: List[str] = []
    ent_parts: List[str] = []

    for pno in batch_pages:
        chunks = chunks_by_page.get(pno) or []
        texts = []
        for idx, chunk in enumerate(chunks):
            content = (chunk.content or chunk.text or "").strip()
            if content:
                meta = chunk.metadata_json or {}
                heading = meta.get("heading") or "null"
                texts.append(f"[chunk_index={idx}] {heading}\n{content}")
        page_text_parts.append(f"--- PAGE {pno} ---\n" + ("\n\n".join(texts) if texts else "(vide)"))

        cats = categories_by_page.get(pno) or []
        cat_parts.append(f"Page {pno}: {', '.join(cats) if cats else 'aucune'}")

        ents = entities_by_page.get(pno) or []
        ent_parts.append(f"Page {pno}: {', '.join(ents[:15]) if ents else 'aucune'}")

    return (
        "\n\n".join(page_text_parts),
        "\n".join(cat_parts),
        "\n".join(ent_parts),
    )


def _collect_batch_slugs(
    batch_pages: List[int],
    categories_by_page: Dict[int, List[str]],
) -> List[str]:
    """Slugs de catégories présents dans le batch, dans l'ordre de première apparition."""
    seen: List[str] = []
    for pno in batch_pages:
        for slug in categories_by_page.get(pno) or []:
            if slug and slug not in seen:
                seen.append(slug)
    return seen


def _build_playbook_section(slugs: List[str]) -> str:
    """Concatène les consignes de réécriture des catégories présentes dans le batch.

    Seules les catégories du batch sont injectées (prompt focalisé). Si aucune n'a de
    consigne dédiée, on retombe sur une consigne générique de réécriture technique.
    """
    from app.services.category_catalog import DEFAULT_CATEGORY_LABELS

    lines: List[str] = []
    for slug in slugs:
        playbook = CATEGORY_ENRICHMENT_PLAYBOOKS.get(slug)
        if not playbook:
            continue
        label = DEFAULT_CATEGORY_LABELS.get(slug, slug)
        lines.append(f"- {label} ({slug}) : {playbook}")
    if not lines:
        return _DEFAULT_PLAYBOOK
    return "\n".join(lines)


def _call_enrichment_api(
    document_title: str,
    batch_pages: List[int],
    page_text: str,
    categories_text: str,
    entities_text: str,
    playbooks_text: str,
    images_b64: Optional[List[str]] = None,
) -> dict:
    from app.services.multimodal_page_service import (
        _mistral_chat_completion,
        _parse_json_with_repair,
    )

    page_range = f"{batch_pages[0]}-{batch_pages[-1]}" if len(batch_pages) > 1 else str(batch_pages[0])
    user_text = _ENRICHMENT_USER_TEMPLATE.format(
        title=document_title or "Document",
        page_range=page_range,
        page_text=page_text[:20000],
        categories_text=categories_text[:4000],
        entities_text=entities_text[:4000],
        playbooks=playbooks_text[:4000],
    )

    if images_b64:
        # Texte + PNG des pages : la séquence du geste et les verdicts visuels (coché/barré)
        # ne sont portés que par le schéma. Mandat borné par la règle 6bis du prompt système,
        # et vérifié en aval par _drop_ungrounded_numbers.
        user_content: object = [{"type": "text", "text": user_text}]
        for image_b64 in images_b64:
            user_content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            )
    else:
        user_content = user_text

    messages = [
        {"role": "system", "content": _ENRICHMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    raw = _mistral_chat_completion(
        messages,
        page_no=batch_pages[0],
        max_tokens=settings.CONTEXTUAL_ENRICHMENT_MAX_TOKENS,
        temperature=0.0,
        response_format_json=True,
        timeout_seconds=settings.CONTEXTUAL_ENRICHMENT_TIMEOUT,
        model=_enrichment_model(),
    )
    return _parse_json_with_repair(raw)


# Valeurs numériques « techniques » : une cote, un couple, un angle, une performance.
# On ignore volontairement les entiers courts isolés (numéros d'étape, de page, de repère),
# qui sont légitimement produits par la synthèse sans figurer tels quels dans le texte.
_MEASURE_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|m|kg|g|°|dan|n|nm|bar|%|mm²|mm2)\b",
    re.IGNORECASE,
)
_LONG_NUMBER_RE = re.compile(r"\b\d{3,}(?:[.,]\d+)?\b")


def _normalize_number(value: str) -> str:
    """Forme comparable d'une valeur numérique (virgule → point, espaces retirés)."""
    return value.replace(",", ".").replace(" ", "").lower()


def _ungrounded_numbers(content: str, source_text: str) -> List[str]:
    """Valeurs numériques du chunk absentes du texte source du batch.

    Garde-fou du mandat visuel : la vision est autorisée pour les verdicts et les
    relations, jamais pour lire une cote. Un nombre qui apparaît dans la synthèse sans
    figurer dans le texte transcrit a donc été lu sur une image — ou inventé.

    Même esprit que ``response_verification_service.check_grounding`` : contrôle de
    présence littérale, tolérant sur la ponctuation (virgule décimale, espaces).
    """
    if not content:
        return []

    haystack = _normalize_number(source_text or "")
    candidates = set(_MEASURE_RE.findall(content)) | set(_LONG_NUMBER_RE.findall(content))

    ungrounded: List[str] = []
    for raw_value in candidates:
        needle = _normalize_number(str(raw_value))
        # On compare la partie NUMÉRIQUE : « 70 mm » est ancré si « 70mm » ou « 70 » figure
        # dans la source (l'unité peut être écrite différemment).
        digits = re.sub(r"[^\d.]", "", needle)
        if not digits:
            continue
        if digits not in haystack:
            ungrounded.append(str(raw_value))
    return ungrounded


def _drop_ungrounded_numbers(
    items: List["EnrichmentChunkItem"],
    source_text: str,
) -> Tuple[List["EnrichmentChunkItem"], int]:
    """Écarte les chunks dont une valeur numérique n'est pas ancrée dans le texte source.

    Returns:
        (chunks conservés, nombre de chunks rejetés).
    """
    kept: List["EnrichmentChunkItem"] = []
    rejected = 0
    for item in items:
        bad = _ungrounded_numbers(item.content, source_text)
        if bad:
            rejected += 1
            logger.warning(
                "[Enrichment] Chunk rejeté — valeur(s) non ancrée(s) dans le texte source : %s "
                "(thème=%r)",
                bad[:5],
                (item.theme or "")[:60],
            )
            continue
        kept.append(item)
    return kept, rejected


def _coerce_enrichment_response(
    raw: dict,
    batch_pages: List[int],
    valid_category_slugs: frozenset[str],
    source_text: str = "",
) -> BatchEnrichmentResponse:
    payload = dict(raw or {})
    items_in = payload.get("enrichment_chunks")
    if not isinstance(items_in, list):
        raise ValueError("Réponse enrichissement invalide : champ 'enrichment_chunks' absent")

    items: List[EnrichmentChunkItem] = []
    for item in items_in:
        if not isinstance(item, dict):
            continue
        slug = (item.get("category_slug") or "").strip().lower().replace(" ", "_")
        # Slug hors catalogue → on l'IGNORE, sans jeter le chunk. Avant le retrait du
        # KAG, les catégories étaient fournies dans le prompt et un slug inconnu
        # signalait une hallucination ; désormais aucune catégorie n'est détectée, donc
        # rejeter sur ce critère supprimait la TOTALITÉ des chunks du batch.
        if slug and valid_category_slugs and slug not in valid_category_slugs:
            slug = ""
        content = (item.get("content") or "").strip()
        theme = (item.get("theme") or "").strip()
        if not content or not theme:
            continue
        source_page = int(item.get("source_page") or batch_pages[len(batch_pages) // 2])
        if source_page not in batch_pages:
            source_page = batch_pages[len(batch_pages) // 2]
        role = (item.get("chunk_role") or CHUNK_ROLE_SYNTHESIS).strip().lower()
        if role not in _VALID_CHUNK_ROLES:
            role = CHUNK_ROLE_SYNTHESIS
        normalized = dict(item)
        normalized.update(
            category_slug=slug or None,
            theme=theme,
            content=content,
            source_page=source_page,
            chunk_role=role,
        )
        try:
            items.append(EnrichmentChunkItem.model_validate(normalized))
        except ValidationError:
            # Champs structurés malformés → on conserve au moins le chunk dense.
            items.append(
                EnrichmentChunkItem(
                    category_slug=slug or None,
                    theme=theme,
                    content=content,
                    source_page=source_page,
                    chunk_role=role,
                )
            )

    if not items:
        raise ValueError(f"Aucun chunk d'enrichissement valide pour batch {batch_pages}")

    # Garde-fou du mandat visuel : un nombre absent du texte transcrit a été lu sur une
    # image (hors mandat) ou inventé. Le chunk entier est écarté — une cote hallucinée est
    # pire qu'une cote absente, puisqu'elle devient indexée et paraît faire autorité.
    if source_text:
        items, rejected = _drop_ungrounded_numbers(items, source_text)
        if rejected:
            logger.warning(
                "[Enrichment] Batch %s — %d chunk(s) rejeté(s) pour valeur non ancrée",
                batch_pages,
                rejected,
            )
        if not items:
            raise ValueError(
                f"Tous les chunks du batch {batch_pages} rejetés (valeurs non ancrées)"
            )

    return BatchEnrichmentResponse(enrichment_chunks=items)


def _render_batch_images(pdf_path: str, batch_pages: List[int]) -> List[str]:
    """Rend les PNG des pages du batch en base64 (best-effort)."""
    from app.services.multimodal_page_service import render_page_png_cached
    import base64

    images: List[str] = []
    for pno in batch_pages:
        try:
            png = render_page_png_cached(pdf_path, pno, dpi=settings.PAGE_EXTRACTION_DPI)
            images.append(base64.b64encode(png).decode("ascii"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Enrichment] Rendu PNG page %s échoué : %s", pno, exc)
    return images


def extract_batch_enrichment_response(
    batch_pages: List[int],
    document_title: str,
    chunks_by_page: Dict[int, List[DocumentChunk]],
    categories_by_page: Dict[int, List[str]],
    entities_by_page: Dict[int, List[str]],
    valid_category_slugs: frozenset[str],
    pdf_path: Optional[str] = None,
) -> Optional[BatchEnrichmentResponse]:
    if not batch_pages:
        return None

    page_text, categories_text, entities_text = _format_batch_context(
        batch_pages,
        chunks_by_page,
        categories_by_page,
        entities_by_page,
    )
    if not page_text.strip():
        return None

    playbooks_text = _build_playbook_section(
        _collect_batch_slugs(batch_pages, categories_by_page)
    )

    # Vision SYSTÉMATIQUE (2026-07-28) : la porte d'avant dépendait à moitié des
    # catégories KAG (supprimées) et à moitié d'un section_type `diagram` que la voie
    # d'extraction texte ne produit plus. Elle n'aurait donc plus jamais tiré sur les
    # pages de schémas — précisément celles où l'image est indispensable, puisque
    # l'enrichissement est désormais le SEUL étage qui voit une page avec droit de
    # synthétiser. Le garde-fou n'est plus la porte mais le mandat contraint du prompt
    # + le contrôle déterministe des nombres (voir _drop_ungrounded_numbers).
    images_b64: Optional[List[str]] = None
    if settings.CONTEXTUAL_ENRICHMENT_MULTIMODAL_ENABLED and pdf_path:
        rendered = _render_batch_images(pdf_path, batch_pages)
        images_b64 = rendered or None
        if images_b64:
            logger.info(
                "[Enrichment] Batch %s enrichi en multimodal (%s images)",
                batch_pages,
                len(images_b64),
            )

    try:
        for attempt in range(2):
            try:
                raw = _call_enrichment_api(
                    document_title,
                    batch_pages,
                    page_text,
                    categories_text,
                    entities_text,
                    playbooks_text,
                    images_b64=images_b64,
                )
                return _coerce_enrichment_response(
                    raw, batch_pages, valid_category_slugs, source_text=page_text
                )
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                if attempt == 0:
                    logger.warning(
                        "[Enrichment] Validation batch %s échouée (tentative 1) : %s — retry",
                        batch_pages,
                        exc,
                    )
                    continue
                logger.warning("[Enrichment] Validation batch %s échouée : %s", batch_pages, exc)
                return None
    except Exception as exc:
        logger.warning("[Enrichment] Extraction batch %s échouée : %s", batch_pages, exc)
        return None
    return None


def _get_document_space_ids(session: Session, document_id: int) -> List[int]:
    stmt = select(DocumentSpace.space_id).where(DocumentSpace.document_id == document_id)
    return list(session.exec(stmt).all())


def _next_chunk_index(session: Session, document_id: int) -> int:
    row = session.execute(
        text(
            "SELECT COALESCE(MAX(chunk_index), -1) FROM documentchunk WHERE document_id = :doc_id"
        ),
        {"doc_id": document_id},
    ).first()
    return int(row[0] if row else -1) + 1


def _persist_enrichment_chunks(
    session: Session,
    document: Document,
    space_ids: List[int],
    batch_pages: List[int],
    enrichment_items: List[EnrichmentChunkItem],
    page_anchors: Dict[int, DocumentChunk],
    category_id_by_slug: Dict[str, int],
    start_chunk_index: int,
) -> int:
    created = 0
    chunk_index = start_chunk_index
    central_page = batch_pages[len(batch_pages) // 2]
    anchor = page_anchors.get(central_page)
    parent_node_id = anchor.node_id if anchor else None

    for item in enrichment_items:
        source_page = item.source_page if item.source_page in batch_pages else central_page
        node_id = str(uuid.uuid4())
        meta = {
            "document_id": document.id,
            "document_title": document.title or "",
            "content_type": CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
            "chunk_type": CONTENT_TYPE_CONTEXTUAL_ENRICHMENT,
            "is_contextual_enrichment": True,
            "enrichment_version": CONTEXTUAL_ENRICHMENT_VERSION,
            "enrichment_model": _enrichment_model(),
            "category_slug": item.category_slug,
            "theme": item.theme,
            "chunk_role": item.chunk_role,
            "source_pages": batch_pages,
            "source_page": source_page,
            "page_no": source_page,
            "page_start": min(batch_pages),
            "page_end": max(batch_pages),
            "page_anchor_node_id": parent_node_id,
            "is_leaf": True,
            "node_id": node_id,
            "parent_node_id": parent_node_id,
        }
        # Champs structurés (procedural_step / diagnostic_unit) → exploités par le
        # générateur d'arbres d'accompagnement (Étape 8).
        structured = item.structured_metadata()
        if structured:
            meta["structured"] = structured

        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=chunk_index,
            content=item.content,
            text=item.content,
            start_char=0,
            end_char=len(item.content),
            node_id=node_id,
            parent_node_id=parent_node_id,
            is_leaf=True,
            hierarchy_level=2,
            metadata_json=meta,
            metadata_=meta,
            source=document.source,
        )
        session.add(chunk)
        session.flush()

        slug_meta = dict(meta)
        slug_meta["categories"] = [item.category_slug] if item.category_slug else []
        chunk.metadata_json = slug_meta
        chunk.metadata_ = slug_meta
        session.add(chunk)

        category_id = (
            category_id_by_slug.get(item.category_slug) if item.category_slug else None
        )
        if category_id is not None:
            # Lien au niveau document : une seule ligne par (chunk, catégorie), sans espace.
            exists = session.exec(
                select(ChunkCategoryRelation).where(
                    ChunkCategoryRelation.chunk_id == chunk.id,
                    ChunkCategoryRelation.category_id == category_id,
                )
            ).first()
            if exists is None:
                session.add(
                    ChunkCategoryRelation(
                        chunk_id=chunk.id,
                        category_id=category_id,
                        document_id=document.id,
                        page_no=source_page,
                        confidence=0.9,
                    )
                )

        chunk_index += 1
        created += 1

    return created


def cleanup_enrichment_for_document(session: Session, document_id: int) -> None:
    """Supprime les chunks d'enrichissement contextuel d'un document."""
    chunk_ids = [
        row[0]
        for row in session.execute(
            text(
                """
                SELECT id FROM documentchunk
                WHERE document_id = :doc_id
                  AND COALESCE(
                      metadata_json->>'content_type',
                      metadata_->>'content_type',
                      ''
                  ) = :content_type
                """
            ),
            {"doc_id": document_id, "content_type": CONTENT_TYPE_CONTEXTUAL_ENRICHMENT},
        ).all()
    ]
    if not chunk_ids:
        return

    chunk_ids_tuple = tuple(chunk_ids)
    session.execute(
        text("DELETE FROM chunkcategoryrelation WHERE chunk_id IN :chunk_ids"),
        {"chunk_ids": chunk_ids_tuple},
    )
    session.execute(
        text("DELETE FROM documentchunk WHERE id IN :chunk_ids"),
        {"chunk_ids": chunk_ids_tuple},
    )
    session.commit()
    logger.info(
        "[Enrichment] Nettoyage document_id=%s — %s chunks supprimés",
        document_id,
        len(chunk_ids),
    )


def run_contextual_enrichment_for_document(document_id: int) -> dict:
    """
    Génère et persiste les chunks d'enrichissement contextuel pour un document.
    Non bloquant : retourne des compteurs même si certains batches échouent.
    """
    if not settings.CONTEXTUAL_ENRICHMENT_ENABLED:
        return {"chunks": 0, "batches": 0, "status": "disabled"}

    with Session(engine) as session:
        document = session.get(Document, document_id)
        if not document:
            raise ValueError(f"Document introuvable : {document_id}")

        space_ids = _get_document_space_ids(session, document_id)
        if not space_ids:
            logger.warning(
                "[Enrichment] document_id=%s sans espace associé — ignoré",
                document_id,
            )
            return {"chunks": 0, "batches": 0, "status": "no_space"}

        chunks_by_page = _load_semantic_chunks_by_page(session, document_id)
        if not chunks_by_page:
            return {"chunks": 0, "batches": 0, "status": "no_chunks"}

        cleanup_enrichment_for_document(session, document_id)

        page_anchors = _load_page_anchors(session, document_id)
        categories_by_page = _load_categories_by_page(session, document_id)
        entities_by_page = _load_entities_by_page(session, document_id)

        from app.services.category_catalog import get_category_id_by_slug

        category_id_by_slug = get_category_id_by_slug(session)
        valid_category_slugs = frozenset(category_id_by_slug.keys())
        # pdf_path pour l'enrichissement multimodal (None → texte-seul partout)
        pdf_path = document.source_file_path or None

        page_numbers = sorted(chunks_by_page.keys())
        batches = build_page_batches(page_numbers)

        batch_responses: List[Tuple[List[int], BatchEnrichmentResponse]] = []
        concurrency = settings.CONTEXTUAL_ENRICHMENT_CONCURRENCY

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    extract_batch_enrichment_response,
                    batch,
                    document.title or "",
                    chunks_by_page,
                    categories_by_page,
                    entities_by_page,
                    valid_category_slugs,
                    pdf_path,
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    response = future.result()
                    if response and response.enrichment_chunks:
                        batch_responses.append((batch, response))
                except Exception as exc:
                    logger.error("[Enrichment] Batch %s échoué : %s", batch, exc)

        chunk_index = _next_chunk_index(session, document_id)
        total_created = 0
        batches_ok = 0

        for batch, response in sorted(batch_responses, key=lambda x: x[0][0]):
            created = _persist_enrichment_chunks(
                session,
                document,
                space_ids,
                batch,
                response.enrichment_chunks,
                page_anchors,
                category_id_by_slug,
                chunk_index,
            )
            chunk_index += created
            total_created += created
            batches_ok += 1

        session.commit()

        logger.info(
            "[Enrichment] Terminé document_id=%s batches=%s/%s chunks=%s model=%s",
            document_id,
            batches_ok,
            len(batches),
            total_created,
            _enrichment_model(),
        )
        return {
            "chunks": total_created,
            "batches": batches_ok,
            "status": "completed" if total_created else "empty",
        }


