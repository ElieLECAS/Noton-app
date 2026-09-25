"""Le vérificateur de faisabilité PERFORM76 : options du formulaire et verdict calculé."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services import debit_atelier, faisabilite
from app.services.wiki_service import WikiUnavailable, get_snapshot

router = APIRouter(prefix="/api/faisabilite", tags=["faisabilite"])


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


def _snapshot():
    try:
        return get_snapshot()
    except WikiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/options")
async def faisabilite_options(current_user: UserRead = Depends(get_current_user)):
    try:
        return faisabilite.options(_snapshot())
    except faisabilite.DonneeIntrouvable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


class DebitIn(SaisieIn):
    surcote_soudure_mm: Optional[float] = Field(default=None, ge=0, le=20)


@router.post("/debit")
async def faisabilite_debit(saisie: DebitIn, current_user: UserRead = Depends(get_current_user)):
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
async def faisabilite_verifier(saisie: SaisieIn, current_user: UserRead = Depends(get_current_user)):
    try:
        return faisabilite.verifier(_snapshot(), faisabilite.Saisie(**saisie.model_dump()))
    except faisabilite.DonneeIntrouvable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
