"""Le calculateur de parcloses et de joints : pour un vitrage donné, quoi monter, système par système."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.models.user import UserRead
from app.routers.auth import get_current_user
from app.services import parcloses
from app.services.wiki_service import WikiUnavailable, get_snapshot

router = APIRouter(prefix="/api/parcloses", tags=["parcloses"])


@router.get("")
async def parcloses_chercher(epaisseur: Optional[float] = Query(default=None, gt=0, le=120),
                             vitrage: Optional[str] = Query(default=None, max_length=60),
                             gamme: Optional[str] = Query(default=None, max_length=40),
                             current_user: UserRead = Depends(get_current_user)):
    try:
        snap = get_snapshot()
    except WikiUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    try:
        return parcloses.chercher(snap, epaisseur=epaisseur, vitrage=vitrage, gamme=gamme)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
