"""
Codes de référence produit — motif partagé et détection.

Extrait de ``kag_extraction_service`` lors du retrait du KAG (2026-07-28) : le motif
reste nécessaire au bloc COUVERTURE (``coverage_service``) et à l'épinglage de chunks
par référence, tous deux conservés.

Le motif couvre les formes rencontrées dans les corpus menuiserie :
  - alphanumérique fournisseur : SL1600, BC01, TGY3702
  - numérique pur : 6111, 155, 76180
  - alterné : 6A20
"""
from __future__ import annotations

import re

# Motifs de code produit (référence, RAL, norme). Volontairement large : les
# garde-fous (longueur minimale, exclusion des millésimes, stopwords) sont portés
# par les appelants — voir coverage_service.extract_message_reference_codes.
#
# Les frontières alphanumériques ne sont PAS un détail : sans elles, le motif démarre au
# MILIEU d'un mot. « PERFORM76 » — le nom de gamme le plus fréquent du corpus — produisait
# le code fantôme « FORM76 » (les 4 lettres qui précèdent les chiffres), introuvable dans
# les documents par construction. Le contrôle de sortie déclenchait donc une correction à
# CHAQUE réponse de l'espace Perform (mesuré le 2026-09-12 : sur toutes les questions,
# quel que soit le sujet), et ces rounds de contrôle finissaient par déverser leur
# raisonnement dans la réponse montrée à l'utilisateur.
REF_CODE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z]{1,4}\d{2,6}[A-Za-z]?|\d[A-Z]\d{2,4}|\d{3,6}[A-Za-z]?)"
    r"(?![A-Za-z0-9])"
)

# Unités qui signalent une VALEUR technique à côté d'un code : sert à mesurer la
# « densité de spécification » d'un passage lors de l'épinglage par référence.
SPEC_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|m|kg|g|°|dan|n|nm|bar|mm²|mm2)\b",
    re.IGNORECASE,
)


def code_in_text(code: str, text: str) -> bool:
    """Présence d'un code dans un texte, avec frontières alphanumériques.

    Évite le faux positif « 6111 trouvé dans 61110 » que produit un simple ``in``.
    """
    if not code or not text:
        return False
    pattern = re.compile(
        r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    return bool(pattern.search(text))


def spec_density(text: str) -> int:
    """Nombre de valeurs unitaires dans un texte (proxy de densité technique)."""
    if not text:
        return 0
    return len(SPEC_UNIT_RE.findall(text))
