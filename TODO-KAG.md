# TODO KAG / RAG - Checklist d'optimisation

## Priorité
- [ ] Augmenter `RAG_TOP_K` (de 1 vers 4-8) avec contrôle du budget tokens.
- [ ] Mettre en place un retriever hybride explicite: vectoriel + lexical (BM25/ILIKE pondéré) + KAG.
- [ ] Implémenter une fusion robuste des scores (RRF ou pondération) au lieu d'un fallback lexical tardif.
- [ ] Calibrer un score unifié: normaliser séparément score vectoriel, score graphe, score reranker avant fusion.
- [ ] Garder KAG en mode "assist" (pas "override"): cap de boost KAG + priorité aux matches d'entités exactes.
- [ ] Utiliser `summary` / `generated_questions` pour le retrieval seulement (pas d'injection directe au LLM).
- [ ] Conserver un contexte final LLM 100% source (passages documentaires uniquement).

## Qualité de ranking
- [ ] Rendre le rerank adaptatif: `top N` variable selon la confiance initiale.
- [ ] Ajouter déduplication + diversité (MMR) pour éviter les passages redondants.
- [ ] Ajuster la résolution parent: parent complet vs parent tronqué autour des leaves matchées.
- [ ] Ajouter un query rewrite léger (1 reformulation métier) pour améliorer le recall.

## Performance
- [ ] Ajouter un cache embedding de requête.
- [ ] Ajouter un cache des candidats rerankés pour requêtes similaires dans une conversation.

## Observabilité et évaluation
- [ ] Instrumenter des logs structurés par étape (`recall@k`, contribution KAG vs vectoriel, latence p95, fallback rate).
- [ ] Créer un benchmark offline (100-200 questions réelles + gold passages).
- [ ] Mesurer systématiquement avant/après chaque changement de pipeline.

## Quick wins
- [ ] `RAG_TOP_K` plus élevé.
- [ ] Fusion hybride vectoriel + lexical + KAG.
- [ ] Métadonnées utilisées pour scorer, mais contexte final LLM 100% source.