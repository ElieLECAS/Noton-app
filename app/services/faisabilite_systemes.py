"""Les calculateurs de faisabilité et de débit, un par système (21/10/2026).

Jusqu'au 21/10 le vérificateur (``faisabilite.py``) et la fiche de débit (``debit_atelier.py``) ne
traitaient que la PERFORM76. Chaque autre système a maintenant son module ; ce registre les range
derrière une même porte (``/api/faisabilite/s/<id>/…``) et un même format de résultat, que la page
``/faisabilite`` affiche sans rien savoir du système.

Un calculateur est un module qui expose :

- ``ID``, ``LIBELLE`` (« Système 70 »), ``GAMMES`` (les gammes PROFERM concernées), ``PORTEE`` (une phrase
  honnête de ce qu'il calcule et de ce qu'il laisse « sur étude ») ;
- ``options(snap) -> dict`` : ``{"champs": [champ…], "configurations": {id: libellé}, "pages": [chemin…],
  "brouillons": [chemin…], "avertissements": [texte…]}`` ;
- ``verifier(snap, saisie: dict) -> dict`` : le résultat de ``synthese`` (ci-dessous), avec en plus ce que le
  système sait dire (``debit``, ``vitrage``…) ;
- ``fiche_debit(snap, saisie: dict) -> dict`` : ``{"lignes", "nomenclature", "accessoires", "notes",
  "brouillons", "verdict", "libelle"}`` — le format de ``debit_atelier.fiche`` ;
  ou ``{"indisponible": "raison"}`` quand le wiki ne permet pas la liste de coupe.

Un **champ** de formulaire : ``{"id", "libelle", "type": "select" | "nombre" | "case", "options": {valeur: libellé},
"defaut", "min", "max", "pas", "unite", "si": {champ: [valeurs]}, "aide"}``. Les champs ``configuration``,
``largeur_mm`` et ``hauteur_mm`` sont ceux de tout calculateur ; ``coercer`` vérifie une saisie contre ses champs.

Les règles de prudence sont celles de ``faisabilite.py`` : toutes les valeurs sont lues dans les tableaux du
wiki, jamais d'interpolation, à moins de la précision de lecture d'une limite « sur étude », une case absente
reste absente, les anomalies ouvertes qui touchent une valeur et les pages en brouillon sont signalées.

Le système ``perform76`` est servi par les modules d'origine et son formulaire d'origine (``legacy``).
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from app.services import debit_atelier, faisabilite
from app.services.wiki_service import WikiSnapshot

OK, HORS, ETUDE, INFO = faisabilite.OK, faisabilite.HORS, faisabilite.ETUDE, faisabilite.INFO

PERFORM76 = "perform76"
# Dans l'ordre où la page les propose. Un module absent (pas encore écrit) n'est pas proposé.
MODULES: Dict[str, str] = {
    "systeme70": "app.services.faisabilite_70",
    "soleal-fy": "app.services.faisabilite_fy",
    "soleal-gy": "app.services.faisabilite_gy",
    "lumeal-ga": "app.services.faisabilite_ga",
    "soleal-py": "app.services.faisabilite_py",
    "askey-oc": "app.services.faisabilite_askey_oc",
    "askey-nv": "app.services.faisabilite_askey_nv",
}
# Ce que le wiki ne permet pas du tout : dit sur la page plutôt que silencieusement absent.
INDISPONIBLES: Dict[str, Dict[str, str]] = {
    "askey-ov": {
        "libelle": "ASKEY Frappe 65 Ouvrant Visible",
        "raison": "Aucun calcul : le wiki n'a que le catalogue de fabrication de cette gamme, pas son catalogue de "
                  "conception (VER-970). Les débits de l'ouvrant caché ne s'y transposent pas.",
    },
}


def _module(systeme: str):
    nom = MODULES.get(systeme)
    if nom is None:
        raise KeyError(systeme)
    return importlib.import_module(nom)


def systemes() -> Dict[str, Any]:
    """Les systèmes proposés par la page : leur libellé, leur portée, et ceux qu'aucun calcul ne couvre."""
    liste: List[Dict[str, Any]] = [{
        "id": PERFORM76, "libelle": "PERFORM76 (système 76)", "gammes": ["PERFORM", "PERFORM+"], "legacy": True,
        "portee": "Faisabilité complète (débit, DTA, abaques, vitrage, ferrure Roto NX, pivot, tapée) et fiche de débit.",
    }]
    for id_, nom in MODULES.items():
        try:
            m = importlib.import_module(nom)
        except ModuleNotFoundError as exc:
            if exc.name != nom:
                raise
            continue
        liste.append({"id": id_, "libelle": m.LIBELLE, "gammes": list(m.GAMMES), "legacy": False, "portee": m.PORTEE})
    return {"systemes": liste,
            "indisponibles": [{"id": k, **v} for k, v in INDISPONIBLES.items()]}


def options(snap: WikiSnapshot, systeme: str) -> Dict[str, Any]:
    if systeme == PERFORM76:
        return faisabilite.options(snap)
    return _module(systeme).options(snap)


def verifier(snap: WikiSnapshot, systeme: str, saisie: Dict[str, Any]) -> Dict[str, Any]:
    if systeme == PERFORM76:
        return faisabilite.verifier(snap, faisabilite.Saisie(**saisie))
    return _module(systeme).verifier(snap, saisie)


def fiche_debit(snap: WikiSnapshot, systeme: str, saisie: Dict[str, Any]) -> Dict[str, Any]:
    if systeme == PERFORM76:
        saisie = dict(saisie)
        surcote = saisie.pop("surcote_soudure_mm", None)
        return debit_atelier.fiche(snap, faisabilite.Saisie(**saisie), surcote)
    return _module(systeme).fiche_debit(snap, saisie)


# ---- ce que les modules ont en commun ------------------------------------------------------


def synthese(controles: List[Dict[str, Any]], brouillons: Optional[List[str]] = None, **en_plus: Any) -> Dict[str, Any]:
    """Le verdict d'un calculateur : hors domaine si un contrôle l'est, sinon sur étude si l'un d'eux l'est,
    sinon réalisable ; le contrôle qui décide est celui que l'on montre en tête."""
    rang = [OK, INFO, ETUDE, HORS]
    decisifs = [c for c in controles if c["statut"] in (HORS, ETUDE)]
    verdict = HORS if any(c["statut"] == HORS for c in controles) else (ETUDE if decisifs else OK)
    return {
        "verdict": verdict,
        "libelle": {OK: "Réalisable", ETUDE: "Sur étude", HORS: "Hors domaine d'emploi"}[verdict],
        "contrainte": next(iter(sorted(decisifs, key=lambda c: -rang.index(c["statut"]))), None),
        "controles": controles,
        "brouillons": sorted(brouillons or []),
        **en_plus,
    }


def coercer(champs: List[Dict[str, Any]], brut: Dict[str, Any]) -> Dict[str, Any]:
    """Une saisie vérifiée contre les champs du formulaire : types, bornes, valeurs permises, défauts.

    ``ValueError`` (message lisible) à la première incohérence. Un champ dont la condition ``si`` n'est pas
    remplie est ignoré (``None``) : il n'a pas été montré à l'utilisateur."""
    sortie: Dict[str, Any] = {}
    for ch in champs:
        id_ = ch["id"]
        si = ch.get("si") or {}
        if any(sortie.get(k) not in v for k, v in si.items()):
            sortie[id_] = None
            continue
        v = brut.get(id_, ch.get("defaut"))
        genre = ch.get("type", "select")
        if genre == "case":
            sortie[id_] = bool(v)
        elif genre == "nombre":
            if v in (None, ""):
                if ch.get("requis", True) and ch.get("defaut") is None:
                    raise ValueError(f"{ch['libelle']} : valeur attendue")
                sortie[id_] = None
                continue
            try:
                v = float(str(v).replace(",", "."))
            except ValueError:
                raise ValueError(f"{ch['libelle']} : nombre attendu")
            if ch.get("min") is not None and v < ch["min"]:
                raise ValueError(f"{ch['libelle']} : au moins {ch['min']}")
            if ch.get("max") is not None and v > ch["max"]:
                raise ValueError(f"{ch['libelle']} : au plus {ch['max']}")
            sortie[id_] = v
        else:
            permis = ch.get("options") or {}
            if v is None or (permis and str(v) not in {str(k) for k in permis}):
                raise ValueError(f"{ch['libelle']} : « {v} » n'est pas une valeur proposée")
            sortie[id_] = str(v)
    return sortie
