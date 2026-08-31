"""Bloc COUVERTURE : grounding mesuré injecté dans le contexte de génération.

Refonte routage/génération (plan 2026-07-21, C3) : le générateur ne peut respecter
« 0 info → ne pas inventer » que s'il SAIT ce que la recherche a trouvé. Ce module
calcule un mini-rapport factuel (passages, documents, références demandées trouvées/
absentes) contre le contexte RÉELLEMENT packé — pas contre une promesse.

Fonctions pures (hors DB) → testables unitairement.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from app.services.reference_codes import REF_CODE_RE

# Mots courants qui matchent le motif code mais n'en sont pas (faux positifs fréquents).
_CODE_STOPWORDS = frozenset({"pvc", "pmr", "sav", "faq", "dtu", "ral"})


def extract_message_reference_codes(*texts: Optional[str], limit: int = 8) -> List[str]:
    """Extrait les codes de référence d'un message utilisateur (et signaux), dédupliqués.

    Réutilise le motif partagé (reference_codes.REF_CODE_RE). Garde-fous :
    - numérique pur → ≥ 4 chiffres (« 155 » est une cote, pas une réf) ;
    - millésimes 1990-2035 exclus ;
    - mots-outils du domaine exclus (PVC, PMR…).
    """
    codes: List[str] = []
    seen: set = set()
    for text_val in texts:
        if not text_val:
            continue
        for m in REF_CODE_RE.finditer(str(text_val)):
            tok = m.group(0).strip().upper()
            if not tok or tok.lower() in _CODE_STOPWORDS:
                continue
            if tok.isdigit():
                if len(tok) < 4:
                    continue
                if len(tok) == 4 and 1990 <= int(tok) <= 2035:
                    continue
            if tok in seen:
                continue
            seen.add(tok)
            codes.append(tok)
            if len(codes) >= limit:
                return codes
    return codes


def _code_in_text(code: str, text_val: str) -> bool:
    """Match word-boundary alphanumérique (« 6111 » sans matcher « 61110 »)."""
    if not code or not text_val:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text_val, re.IGNORECASE) is not None


def build_coverage_block(
    *,
    context_text: str,
    requested_codes: Sequence[str],
    doc_passages: Sequence[dict],
    pinned_codes: Optional[Sequence[str]] = None,
    retrieval_status: Optional[str] = None,
    visual_context: bool = False,
) -> str:
    """Construit le bloc `### COUVERTURE DE LA RECHERCHE` injecté dans le contexte.

    - `context_text` : la MATIÈRE documentaire réellement fournie au modèle — la
      vérification TROUVÉE/ABSENTE se fait contre elle, pas contre les passages bruts.
      En mode 100 % PNG, l'appelant doit y joindre le texte des pages packées : le
      message système ne contient alors qu'un manifeste, et chercher une référence
      dedans la déclarerait absente à tort (la page part pourtant bien en image).
    - `visual_context` : les pages sont fournies en IMAGES. Une référence introuvable
      dans le texte extrait peut rester parfaitement lisible sur la planche — le verdict
      d'absence devient donc une invitation à vérifier l'image, jamais un ordre
      d'abstention.
    - Statut : ok (passages + tous codes trouvés) | partiel (passages mais un code
      absent) | vide (aucun passage pertinent).
    """
    pinned = {c.upper() for c in (pinned_codes or [])}
    n_passages = len(doc_passages)

    max_score = 0.0
    doc_titles: List[str] = []
    seen_titles: set = set()
    for p in doc_passages:
        try:
            max_score = max(max_score, float(p.get("score") or 0.0))
        except (TypeError, ValueError):
            pass
        title = (p.get("document_title") or "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            doc_titles.append(title)

    code_lines: List[str] = []
    missing_codes: List[str] = []
    for code in requested_codes:
        code_u = code.upper()
        found = _code_in_text(code_u, context_text)
        if found:
            suffix = " (chunk de référence épinglé)" if code_u in pinned else ""
            code_lines.append(f"- Référence {code_u} : TROUVÉE dans le contexte{suffix}")
        elif visual_context:
            missing_codes.append(code_u)
            code_lines.append(
                f"- Référence {code_u} : absente du TEXTE extrait — elle peut néanmoins "
                "figurer sur les images de pages fournies. Lis-les avant de conclure ; "
                "ne réponds à partir d'une référence voisine en aucun cas"
            )
        else:
            missing_codes.append(code_u)
            code_lines.append(
                f"- Référence {code_u} : ABSENTE du contexte — ne PAS répondre sur cette "
                "référence à partir d'une référence voisine"
            )

    if n_passages == 0:
        status = "vide"
    elif missing_codes:
        status = "partiel"
    else:
        status = "ok"

    lines = [
        "### COUVERTURE DE LA RECHERCHE (rapport factuel, fait foi)",
        f"- Passages pertinents trouvés : {n_passages}"
        + (f" (score max {max_score:.2f})" if n_passages else ""),
    ]
    if doc_titles:
        shown = ", ".join(doc_titles[:5])
        extra = f" (+{len(doc_titles) - 5} autres)" if len(doc_titles) > 5 else ""
        lines.append(f"- Documents couverts : {shown}{extra}")
    lines.extend(code_lines)
    if retrieval_status and retrieval_status not in ("ok", "completed"):
        lines.append(f"- Statut retrieval : {retrieval_status}")
    lines.append(f"- Statut couverture : {status}")

    return "\n".join(lines)


def coverage_status(
    *,
    context_text: str,
    requested_codes: Sequence[str],
    doc_passages: Sequence[dict],
) -> Dict[str, object]:
    """Version structurée (pour logs/télémétrie/tests) : {status, missing_codes}."""
    missing = [
        c.upper()
        for c in requested_codes
        if not _code_in_text(c.upper(), context_text)
    ]
    if not doc_passages:
        status = "vide"
    elif missing:
        status = "partiel"
    else:
        status = "ok"
    return {"status": status, "missing_codes": missing}
