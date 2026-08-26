"""Fiche d'une gamme commerciale Proferm — couche de connaissance métier.

POURQUOI. Le vocabulaire des utilisateurs et celui des documents ne se recoupent pas :
« Perform 76 » n'apparaît dans AUCUN chunk de la documentation technique, qui parle de
« TROCAL 76 ADVANCED / KBE 76 ADVANCED / KÖMMERLING 76 ADVANCED ». Ce pont n'existe que
dans la tête des équipes. Ces fiches le rendent explicite et exploitable.

DEUX USAGES (aucun n'est du RAG) :
  1. injection dans les prompts de compréhension et de génération, pour que le modèle
     connaisse le paysage AVANT de chercher, et sache écarter un document hors-univers ;
  2. référence de curation pour l'attribution documentaire.

DIFFÉRENCE AVEC LE KAG (désactivé) : le KAG tentait de DÉCOUVRIR l'ontologie en extrayant
entités et relations par LLM sur des milliers de chunks — le bruit s'accumulait. Ici on la
DÉCLARE : une poignée de gammes, écrites et validées à la main.

⚠️ Ces fiches ne portent JAMAIS de valeur (cote, Uw, référence) faisant autorité : elles
routent la recherche, la valeur vient toujours du document.

Le rattachement aux systèmes fournisseurs (profine/Kömmerling, Technal…) viendra dans un
second temps ; `materiau` et `familles` sont donc déclarés ici, et deviendront dérivables.
"""
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlmodel import Column, Field, SQLModel

# Statuts de validation : une fiche pré-remplie depuis les documents reste un BROUILLON
# tant qu'un humain du métier ne l'a pas relue — c'est précisément ce que le KAG ne
# faisait pas.
STATUT_BROUILLON = "brouillon"
STATUT_VALIDE = "valide"
STATUTS = (STATUT_BROUILLON, STATUT_VALIDE)


class GammeCommerciale(SQLModel, table=True):
    """Fiche métier d'une gamme commercialisée par Proferm (Perform, Hybride, …)."""

    __tablename__ = "gamme_commerciale"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Identité — `slug` s'aligne sur Document.proferm_gammes (slot_catalog).
    slug: str = Field(index=True, unique=True, max_length=64)
    nom: str = Field(max_length=120)
    accroche: Optional[str] = Field(default=None, max_length=255)

    # Matériau et familles : déclarés ici, dérivables des systèmes plus tard.
    materiau: Optional[str] = Field(default=None, max_length=32)
    familles: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, server_default="{}"),
    )

    # Le corps de la fiche, injecté dans les prompts.
    description: Optional[str] = None

    # LE PONT : ce que dit l'utilisateur ↔ ce que disent les documents.
    # `alias_utilisateur` = formulations clients/commerciales ; `termes_documentaires` =
    # les mots réellement présents dans les PDF fournisseurs, à utiliser pour chercher.
    alias_utilisateur: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, server_default="{}"),
    )
    termes_documentaires: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, server_default="{}"),
    )

    # Fournisseurs en texte libre tant que l'entité « système » n'existe pas.
    fournisseurs: List[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(String), nullable=False, server_default="{}"),
    )

    # Règles de non-confusion (« Perform 70 ≠ Perform 76 », « ne pas mobiliser Lumine »).
    discriminants: Optional[str] = None

    # Documents qui font foi pour cette gamme. Les ids diffèrent entre bases : à
    # recalculer en prod, d'où le champ purement indicatif.
    document_ids: List[int] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(Integer), nullable=False, server_default="{}"),
    )

    # Points que les documents ne tranchent pas et qui attendent un avis métier.
    a_valider: Optional[str] = None

    statut: str = Field(default=STATUT_BROUILLON, max_length=16, index=True)
    ordre: int = Field(default=100)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[int] = Field(default=None, foreign_key="user.id")


class GammeCommercialeRead(SQLModel):
    id: int
    slug: str
    nom: str
    accroche: Optional[str] = None
    materiau: Optional[str] = None
    familles: List[str] = []
    description: Optional[str] = None
    alias_utilisateur: List[str] = []
    termes_documentaires: List[str] = []
    fournisseurs: List[str] = []
    discriminants: Optional[str] = None
    document_ids: List[int] = []
    a_valider: Optional[str] = None
    statut: str
    ordre: int
    updated_at: datetime


class GammeCommercialeUpdate(SQLModel):
    nom: Optional[str] = None
    accroche: Optional[str] = None
    materiau: Optional[str] = None
    familles: Optional[List[str]] = None
    description: Optional[str] = None
    alias_utilisateur: Optional[List[str]] = None
    termes_documentaires: Optional[List[str]] = None
    fournisseurs: Optional[List[str]] = None
    discriminants: Optional[str] = None
    document_ids: Optional[List[int]] = None
    a_valider: Optional[str] = None
    statut: Optional[str] = None
    ordre: Optional[int] = None
