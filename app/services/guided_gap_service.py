"""Angles morts du SAV guidé : enregistrement et backlog.

Un « gap » = une demande à connotation SAV (symptôme détecté, ou vocabulaire client)
pour laquelle AUCUN arbre publié ne matche. Le backlog trié par fréquence dit quels
arbres écrire en priorité ; « Créer l'arbre » depuis l'admin le préremplit.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from sqlmodel import Session, select

from app.models.guided_gap import GuidedGap
from app.services.guided_entry_index_service import normalize_text

logger = logging.getLogger(__name__)

MAX_SAMPLES = 10


def record_gap(
    session: Session,
    *,
    space_id: int,
    detected_symptom: str = "",
    query_text: str = "",
    conversation_id: Optional[int] = None,
) -> Optional[GuidedGap]:
    """Upsert : même espace + même symptôme (ou même question normalisée) → count++."""
    query_text = (query_text or "").strip()[:2000]
    if not query_text and not detected_symptom:
        return None

    gap: Optional[GuidedGap] = None
    if detected_symptom:
        gap = session.exec(
            select(GuidedGap).where(
                GuidedGap.space_id == space_id,
                GuidedGap.detected_symptom == detected_symptom,
                GuidedGap.status.in_(("open", "planned")),
            )
        ).first()
    else:
        norm = normalize_text(query_text)
        for candidate in session.exec(
            select(GuidedGap).where(
                GuidedGap.space_id == space_id,
                GuidedGap.detected_symptom.is_(None),
                GuidedGap.status.in_(("open", "planned")),
            )
        ).all():
            if normalize_text(candidate.query_text) == norm:
                gap = candidate
                break

    if gap is None:
        gap = GuidedGap(
            space_id=space_id,
            detected_symptom=detected_symptom or None,
            query_text=query_text,
            count=1,
            sample_conversation_ids=[conversation_id] if conversation_id else [],
        )
    else:
        gap.count = int(gap.count or 0) + 1
        gap.last_seen = datetime.utcnow()
        samples = list(gap.sample_conversation_ids or [])
        if conversation_id and conversation_id not in samples and len(samples) < MAX_SAMPLES:
            samples.append(conversation_id)
            gap.sample_conversation_ids = samples
        if query_text and len(gap.query_text or "") < 1500 and query_text not in (gap.query_text or ""):
            gap.query_text = ((gap.query_text or "") + "\n" + query_text)[:2000]

    session.add(gap)
    session.commit()
    return gap


def list_gaps(session: Session, *, space_id: Optional[int] = None, status: str = "") -> List[GuidedGap]:
    stmt = select(GuidedGap)
    if space_id is not None:
        stmt = stmt.where(GuidedGap.space_id == space_id)
    if status:
        stmt = stmt.where(GuidedGap.status == status)
    gaps = list(session.exec(stmt).all())
    gaps.sort(key=lambda g: (-(g.count or 0), g.last_seen or datetime.min), reverse=False)
    return gaps


def update_gap_status(session: Session, gap_id: int, status: str) -> Optional[GuidedGap]:
    gap = session.get(GuidedGap, gap_id)
    if gap is None:
        return None
    gap.status = status
    session.add(gap)
    session.commit()
    session.refresh(gap)
    return gap
