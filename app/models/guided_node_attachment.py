"""Pièces jointes bibliothèque d'un nœud d'arbre SAV.

Le lien déterministe arbre → documentation : l'auteur attache un document de la
bibliothèque (et optionnellement une plage de pages) à un nœud. Au runtime, ces pièces
remplacent tout retrieval : elles sont affichées (ouverture PDF à la page) et servent
de contexte CAG/RAG scopé (guided_attachment_context_service).
"""

from typing import Optional

from sqlalchemy import ForeignKey
from sqlmodel import Column, Field, SQLModel

ATTACHMENT_KINDS = ("notice", "photo", "schema")


class GuidedNodeAttachment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    node_id: int = Field(
        sa_column=Column(ForeignKey("guidedtreenode.id", ondelete="CASCADE"), index=True)
    )
    document_id: int = Field(
        sa_column=Column(ForeignKey("document.id", ondelete="CASCADE"), index=True)
    )
    # Pages null = document entier.
    page_start: Optional[int] = Field(default=None)
    page_end: Optional[int] = Field(default=None)
    caption: str = Field(default="", max_length=300)
    kind: str = Field(default="notice", max_length=20)
    display_order: int = Field(default=0)
