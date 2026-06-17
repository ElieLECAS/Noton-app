"""Helpers partagés pour les tests API bibliothèque."""

from __future__ import annotations

from typing import Any, Dict


def upload_form(**overrides: Any) -> Dict[str, str]:
    """Champs FormData classification obligatoires pour POST /api/library/upload."""
    base = {
        "space_ids": "[]",
        "is_paid": "false",
        "supplier": "Profine",
        "product_types": '["fenetre"]',
        "materials": '["pvc"]',
        "proferm_gammes": "[]",
    }
    base.update(overrides)
    return base
