"""Lecture, amorçage et rendu des fiches de gammes commerciales.

Le rendu `build_gamme_knowledge_block` est le point d'entrée destiné aux prompts : il
produit un bloc COMPACT, injectable tel quel. Il est volontairement rendu EN ENTIER,
jamais filtré sur la gamme détectée — pour écarter une référence Technal quand on cherche
du PVC, le modèle doit connaître la règle Technal. Une injection sélective casserait
précisément la discrimination recherchée.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from sqlmodel import Session, select

from app.models.gamme_commerciale import (
    STATUT_BROUILLON,
    GammeCommerciale,
)

logger = logging.getLogger(__name__)


def list_gammes(session: Session, *, only_valid: bool = False) -> List[GammeCommerciale]:
    stmt = select(GammeCommerciale).order_by(GammeCommerciale.ordre, GammeCommerciale.slug)
    gammes = list(session.exec(stmt).all())
    if only_valid:
        gammes = [g for g in gammes if g.statut != STATUT_BROUILLON]
    return gammes


def seed_gammes(session: Session, *, overwrite: bool = False) -> dict:
    """Crée les fiches manquantes depuis GAMMES_SEED.

    Ne réécrit JAMAIS une fiche existante sauf `overwrite` explicite : une fois qu'un
    humain du métier a relu une fiche, c'est la base qui fait foi, pas le code.
    """
    from app.services.gamme_seed_data import GAMMES_SEED

    existing = {g.slug: g for g in session.exec(select(GammeCommerciale)).all()}
    created, updated, skipped = 0, 0, 0

    for payload in GAMMES_SEED:
        slug = payload["slug"]
        current = existing.get(slug)
        if current is None:
            session.add(GammeCommerciale(**payload))
            created += 1
            continue
        if not overwrite:
            skipped += 1
            continue
        for key, value in payload.items():
            if key != "slug":
                setattr(current, key, value)
        current.updated_at = datetime.utcnow()
        session.add(current)
        updated += 1

    session.commit()
    logger.info(
        "[gammes] amorçage — %d créée(s), %d mise(s) à jour, %d conservée(s)",
        created, updated, skipped,
    )
    return {"created": created, "updated": updated, "skipped": skipped}


def _fmt_list(values: Optional[List[str]], limit: int = 12) -> str:
    items = [str(v).strip() for v in (values or []) if str(v).strip()]
    return ", ".join(items[:limit]) if items else "—"


def build_gamme_knowledge_block(
    session: Session, *, only_valid: bool = False
) -> str:
    """Bloc de connaissance métier injectable dans un prompt.

    Retourne "" si aucune fiche : le prompt reste alors strictement identique à avant,
    aucun bloc vide ni consigne orpheline.
    """
    gammes = list_gammes(session, only_valid=only_valid)
    if not gammes:
        return ""

    lines: List[str] = [
        "### CONNAISSANCE MÉTIER — GAMMES PROFERM",
        "Le vocabulaire des utilisateurs et celui des documents fournisseurs NE SE "
        "RECOUPENT PAS : « Perform 76 » n'apparaît dans aucun document technique, qui "
        "parle de « TROCAL 76 ADVANCED ». Traduis toujours la gamme commerciale en termes "
        "documentaires avant de chercher, et sers-toi de ces fiches pour ÉCARTER les "
        "documents d'un autre univers.",
        "⚠️ Ces fiches routent la recherche. Elles ne font JAMAIS autorité sur une valeur : "
        "une cote, un Uw ou une référence se lisent uniquement dans les documents fournis.",
        "",
    ]

    for g in gammes:
        titre = f"{g.nom}"
        if g.accroche:
            titre += f" — {g.accroche}"
        lines.append(f"**{titre}**")
        meta = [f"matériau : {g.materiau or '—'}"]
        if g.familles:
            meta.append(f"familles : {_fmt_list(g.familles)}")
        if g.fournisseurs:
            meta.append(f"fournisseurs : {_fmt_list(g.fournisseurs)}")
        lines.append("  " + " · ".join(meta))
        if g.description:
            lines.append(f"  {g.description.strip()}")
        if g.alias_utilisateur:
            lines.append(f"  Dit par l'utilisateur : {_fmt_list(g.alias_utilisateur)}")
        if g.termes_documentaires:
            lines.append(f"  À CHERCHER dans les documents : {_fmt_list(g.termes_documentaires)}")
        if g.discriminants:
            for rule in g.discriminants.strip().splitlines():
                if rule.strip():
                    lines.append(f"  ⛔ {rule.strip()}")
        if g.statut == STATUT_BROUILLON:
            lines.append("  (fiche en brouillon — non validée par le métier)")
        lines.append("")

    return "\n".join(lines).strip()
