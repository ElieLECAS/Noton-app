"""Contrat entre la compréhension de requête et le retrieval.

Module dédié plutôt qu'hébergé d'un côté ou de l'autre : la recherche ne dépend pas du
module de compréhension, et réciproquement — les deux importent ce schéma.

Historique : ces modèles vivaient dans `query_understanding_graph.py`, supprimé le
2026-08-26 (machine LangGraph sans appelant). Seul `RetrievalQueries` a survécu au
nettoyage — `QueryGroup` est parti avec la chaîne multi-groupe et `ClarificationResult`
avec le contrôle de vagueness.
"""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class RetrievalQueries(BaseModel):
    """Requêtes spécialisées par canal de retrieval.

    Deux canaux depuis le retrait de la voie dense texte (2026-08-25) :
      - ``colpali`` : question autonome, encodée par ColQwen2 (canal visuel) ;
      - ``lexical`` : question + entités/références extraites, pour le tsvector BM25.

    Le champ ``semantic`` a été retiré avec la voie dense : il n'avait plus de
    consommateur (son seul usage alimentait un reranker désactivé).
    """

    colpali: str
    lexical: str
    reasoning: str = ""
    slots_used: Dict[str, Any] = Field(default_factory=dict)
