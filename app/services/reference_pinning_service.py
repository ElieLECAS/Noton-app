"""
Épinglage de chunks par code de référence — sans graphe de connaissances.

Quand l'utilisateur demande une référence précise (« profil 76180 », « code RS100 »),
le retrieval sémantique peut très bien rapporter la bonne PAGE sans que le passage
contenant la valeur cherchée finisse dans le contexte. Ce module va chercher
directement le chunk qui fait autorité pour ce code et l'épingle verbatim.

Remplace ``kag_graph_service.select_authority_chunks`` / ``build_pinned_reference_block``
(retrait du KAG, 2026-07-28). L'ancienne version exigeait que le code ait été extrait
comme ``ref_code`` d'une entité par un LLM ; celle-ci lit le contenu des chunks
directement — même bénéfice, sans passe d'extraction, sans table d'entités.

Classement des candidats, par priorité décroissante :
  1. densité de spécification : nombre de valeurs unitaires (70 mm, 1,5 kg, 90°) —
     un chunk qui associe le code à des valeurs est celui qu'on veut ;
  2. brièveté : à densité égale, le chunk le plus court cible mieux le code
     (une ligne de tableau plutôt qu'une page entière) ;
  3. identifiant, pour un résultat déterministe.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlmodel import Session

from app.services.reference_codes import code_in_text, spec_density

logger = logging.getLogger(__name__)

# Plafond de caractères d'un extrait épinglé : au-delà, on n'épingle pas (le passage
# n'est plus un extrait ciblé mais une page entière, déjà couverte par le packing).
DEFAULT_CHAR_CAP = 1200

# Nombre de candidats lus en base par code avant classement en Python.
_SQL_CANDIDATE_LIMIT = 40


def select_authority_chunks(
    session: Session,
    document_ids: List[int],
    code: str,
    *,
    limit: int = 1,
    char_cap: int = DEFAULT_CHAR_CAP,
) -> List[Dict[str, Any]]:
    """Chunks faisant autorité pour ``code`` parmi ``document_ids``.

    Le filtrage SQL est volontairement large (ILIKE) : il sert à réduire le volume.
    La vraie sélection — frontières du code, densité de spécification — est faite en
    Python, où ``code_in_text`` évite le faux positif « 6111 trouvé dans 61110 ».
    """
    if not code or not document_ids:
        return []

    rows = session.execute(
        text(
            """
            SELECT dc.id, dc.content, d.title AS document_title,
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
        {
            "doc_ids": tuple(document_ids),
            "needle": f"%{code}%",
            "lim": _SQL_CANDIDATE_LIMIT,
        },
    ).all()

    candidates: List[Tuple[int, int, int, Dict[str, Any]]] = []
    for row in rows:
        content = (row.content or "").strip()
        if not content or len(content) > char_cap:
            continue
        if not code_in_text(code, content):
            continue

        page_no: Optional[int] = None
        try:
            page_no = int(row.page_no) if row.page_no is not None else None
        except (TypeError, ValueError):
            page_no = None

        candidates.append(
            (
                -spec_density(content),  # densité décroissante
                len(content),            # puis le plus court
                int(row.id),             # déterminisme
                {
                    "chunk_id": int(row.id),
                    "content": content,
                    "document_title": row.document_title or "Document",
                    "page_no": page_no,
                },
            )
        )

    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    return [c[3] for c in candidates[:limit]]


def build_pinned_reference_block(
    session: Session,
    document_ids: List[int],
    codes: List[str],
    *,
    max_codes: int = 2,
    char_cap: int = DEFAULT_CHAR_CAP,
) -> Tuple[str, List[str]]:
    """Bloc ``### EXTRAIT DE RÉFÉRENCE`` verbatim et sourcé pour les codes demandés.

    Returns:
        (bloc_texte, codes_effectivement_épinglés) — bloc vide si rien à épingler.
    """
    if not codes or not document_ids:
        return "", []

    parts: List[str] = []
    pinned: List[str] = []

    for code in codes[:max_codes]:
        try:
            chunks = select_authority_chunks(
                session, document_ids, code, char_cap=char_cap
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[pinning] Recherche du code %s échouée : %s", code, exc)
            continue
        if not chunks:
            continue
        pinned.append(code.upper())
        for chunk in chunks:
            page = f", p.{chunk['page_no']}" if chunk.get("page_no") else ""
            parts.append(
                f"### EXTRAIT DE RÉFÉRENCE — {code.upper()} "
                f"(source exacte : {chunk['document_title']}{page})\n"
                f"« {chunk['content']} »"
            )

    if pinned:
        logger.info("[pinning] %d code(s) épinglé(s) : %s", len(pinned), pinned)
    return ("\n\n".join(parts), pinned)
