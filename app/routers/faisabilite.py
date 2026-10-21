"""Le vérificateur de faisabilité et la fiche de débit : options du formulaire et verdict calculé.

Les routes sans préfixe de système sont celles de la PERFORM76 (la page d'origine) ; ``/s/<système>/…`` sert tous
les calculateurs du registre ``faisabilite_systemes``, PERFORM76 comprise.

Réservé au rôle admin : la dépendance est posée sur le routeur, une route ajoutée en hérite.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field

from app.routers.auth import require_role
from app.services import debit_atelier, faisabilite, faisabilite_systemes
from app.services.wiki_service import WikiUnavailable, get_snapshot

router = APIRouter(prefix="/api/faisabilite", tags=["faisabilite"],
                   dependencies=[Depends(require_role("admin"))])


class SaisieIn(BaseModel):
    configuration: str
    largeur_mm: float = Field(gt=0, le=10000)
    hauteur_mm: float = Field(gt=0, le=10000)
    dormant: str
    ouvrant: Optional[str] = None
    meneau: Optional[str] = None
    battement: Optional[str] = None
    bas: str = "dormant"
    couleur: str = "blanc"
    vitrage: str = "4-16-4"
    j079: bool = False
    vent: str = "0,8"
    isolant_mm: Optional[float] = Field(default=None, ge=0, le=500)
    paumelles: str = "P"
    version: str = "130"
    securite: str = "base"
    ferrure_poids: Optional[str] = None
    acier_dormant: str = "sans"
    distance_verrouillage_cm: Optional[float] = Field(default=None, gt=0, le=500)
    aluclip_pro: bool = False


def _snapshot():
    try:
        return get_snapshot()
    except WikiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/options")
async def faisabilite_options():
    try:
        return faisabilite.options(_snapshot())
    except faisabilite.DonneeIntrouvable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


class DebitIn(SaisieIn):
    surcote_soudure_mm: Optional[float] = Field(default=None, ge=0, le=20)


@router.post("/debit")
async def faisabilite_debit(saisie: DebitIn):
    """La fiche de débit : liste de coupe, nomenclature, accessoires (audit § 9.2)."""
    data = saisie.model_dump()
    surcote = data.pop("surcote_soudure_mm")
    try:
        return debit_atelier.fiche(_snapshot(), faisabilite.Saisie(**data), surcote)
    except faisabilite.DonneeIntrouvable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("")
async def faisabilite_verifier(saisie: SaisieIn):
    try:
        return faisabilite.verifier(_snapshot(), faisabilite.Saisie(**saisie.model_dump()))
    except faisabilite.DonneeIntrouvable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ---- un calculateur par système ------------------------------------------------------------


def _systeme(systeme: str) -> str:
    if systeme != faisabilite_systemes.PERFORM76 and systeme not in faisabilite_systemes.MODULES:
        raise HTTPException(status_code=404, detail=f"système inconnu : {systeme}")
    return systeme


def _appel(fonction, systeme: str, *args):
    try:
        return fonction(_snapshot(), _systeme(systeme), *args)
    except ModuleNotFoundError:
        raise HTTPException(status_code=404, detail=f"le calculateur du système {systeme} n'existe pas encore")
    except faisabilite.DonneeIntrouvable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/systemes")
async def faisabilite_systemes_liste():
    return faisabilite_systemes.systemes()


@router.get("/s/{systeme}/options")
async def systeme_options(systeme: str):
    return _appel(faisabilite_systemes.options, systeme)


@router.post("/s/{systeme}")
async def systeme_verifier(systeme: str, saisie: Dict[str, Any] = Body(...)):
    return _appel(faisabilite_systemes.verifier, systeme, saisie)


@router.post("/s/{systeme}/debit")
async def systeme_debit(systeme: str, saisie: Dict[str, Any] = Body(...)):
    return _appel(faisabilite_systemes.fiche_debit, systeme, saisie)
