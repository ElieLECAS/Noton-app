# Audit pipeline RAG/KAG — Bibliothèque générale (2026-05-19)

## Portée auditée

Ce document audite la chaîne complète:

1. Ingestion/chunking Docling
2. Extraction entités et relations KAG
3. Embeddings
4. Retrieval hybride vectoriel + lexical + graphe
5. Multi-hop KAG
6. Diversification MMR
7. Reranking
8. Points de tuning pour viser un « RAG parfait » (pragmatique)

## 1) Chunking

### État actuel (points solides)
- Chunking hiérarchique avec tailles configurables et fallback robuste sur gros documents.
- Versions de chunking tracées (`docling_hierarchical_v1/v2`, etc.) pour reindex ciblée.
- Résolution de pages/offsets et métadonnées hiérarchiques (node_id/parent_node_id/is_leaf), utile pour citation et reconstruction contexte.

### Limites identifiées
- Les tailles par défaut (`[3072, 1024, 384]`) sont bonnes mais globales: pas de stratégie dynamique par type de contenu (tableau dense vs prose vs fiche normative) au moment retrieval.
- Le pipeline semble dépendre fortement du « parent enriched retrieval » pour compenser les pertes de contexte; cela fonctionne, mais masque un besoin de chunking sémantique encore plus fin sur certains artefacts Docling.

### Optimisations recommandées
- Ajouter un calibrage automatique par document (distribution longueur sections, ratio tableaux) qui pousse des tailles plus petites sur contenus normatifs/tests.
- Ajouter une feature de « contiguous chunk stitching »: regroupement post-retrieval de feuilles adjacentes quand elles partagent même parent + pages contiguës.

## 2) Entités et relations KAG

### État actuel (points solides)
- Taxonomie entités métier explicite et relativement riche (gamme, profil, norme, performance, etc.).
- Relations typées strictes avec blocklist de relations vagues.
- Seuil confiance entité configurable (`MIN_ENTITY_CONFIDENCE`, défaut 0.30).

### Limites identifiées
- Le seuil 0.30 est permissif: bon rappel, mais bruit possible en multi-hop.
- La résolution de coréférence/alias dépend encore de patterns regex et du LLM; risque de fragmentation des nœuds entités (ex: variantes orthographiques).

### Optimisations recommandées
- Passer en double seuil: seuil bas indexation (0.30), seuil plus haut pour expansion multi-hop (0.45–0.55).
- Mettre un score de « stabilité d’entité » (fréquence + cohérence type + alias consolidés) et l’injecter dans scoring hop.
- Ajouter un job de normalisation offline hebdo des alias les plus fréquents (merge assisté).

## 3) Embeddings

### État actuel (points solides)
- Modèle unique `BAAI/bge-m3` utilisé en singleton, batché avec retry/backoff.
- Vérification cache HF et mode offline activé pour éviter latence réseau inutile.
- Batch embeddings bien implémenté (anti-pattern one-by-one évité).

### Limites identifiées
- Batch size statique (`DEFAULT_BATCH_SIZE=16`) et pas d’auto-tuning selon device/VRAM.
- Pas de stratégie de quantization/accélération explicitée côté embedding (potentiel latence/coût).

### Optimisations recommandées
- Introduire auto batch sizing dynamique (warmup + adaptation).
- Instrumenter percentiles P50/P95 embedding par taille de texte.
- Ajouter un mode « fast query embedding » si charge concurrente élevée (même modèle mais queue prioritaire requête utilisateur).

## 4) Retrieval hybride (vectoriel + lexical + parent + KAG)

### État actuel (points solides)
- Pipeline mature avec fusion RRF sur plusieurs canaux.
- Parent enriched retrieval et fallback lexical ILIKE présents (robustesse recall).
- Seuils de filtrage hybrides et hooks de traçabilité bien exposés.

### Limites identifiées
- Poids RRF parent fixe (`0.50`) potentiellement sous-optimal selon intent de requête.
- Fallback ILIKE utile mais potentiellement bruyant sans garde-fou de proximité sémantique.

### Optimisations recommandées
- Rendre le poids parent adaptatif selon l’intent (`reason_query_intent`) et le type de question.
- Ajouter garde-fou sur fallback ILIKE: top lexical retenu seulement si reranker score minimal atteint.
- Enrichir le lexical avec expansion synonymique métier (ex: tarif/prix/coût; AEV/perméabilité air eau vent).

## 5) Multi-hop KAG

### État actuel (points solides)
- Déclenchement heuristique (patterns forts/faibles) + entités pivot.
- Orchestrateur borné (max 3 hops, budget candidats, patience, pénalité par hop).
- Traçabilité des chemins (`hop_traces`) et signaux par chunk.

### Limites identifiées
- Heuristique de déclenchement partiellement regex-driven: faux positifs/négatifs possibles.
- Pénalités par hop statiques (0.00/0.05/0.10/0.15) non calibrées sur qualité réelle.

### Optimisations recommandées
- Ajouter un classifieur léger « need_multi_hop » basé sur historique Q/A + features lexicales.
- Réapprendre périodiquement les pénalités hop depuis dataset d’évaluation (NDCG@k / answer faithfulness).
- Stop condition plus riche: arrêter si gain marginal reranker attendu < seuil.

## 6) MMR

### État actuel (points solides)
- MMR appliqué avant reranker (bonne pratique pour diversité).
- Embeddings candidats fetchés explicitement pour éviter surcoût requête initiale.

### Limites identifiées
- `MMR_K` et `MMR_LAMBDA` globaux: pas contextualisés par intent (comparatif vs factuel mono-point).

### Optimisations recommandées
- Rendre `lambda` dynamique: plus faible pour requêtes comparatives (plus de diversité), plus élevé pour requêtes factuelles.
- Exposer métrique de redondance intra-topK dans observabilité pour tuning continu.

## 7) Reranking

### État actuel (points solides)
- Reranker cross-encoder BGE v2 m3 (ou fallback ST) en singleton.
- Skip reranking si similarité vectorielle très haute (`SKIP_RERANK_THRESHOLD=0.85`) pour latence.
- Cap pool candidates et gestion erreurs fallback.

### Limites identifiées
- Seuil skip global peut produire des erreurs sur requêtes ambiguës (haute similarité ≠ bonne réponse finale).
- Possibilité de sous-exploiter reranker sur requêtes nécessitant arbitrage multi-sources.

### Optimisations recommandées
- Conditionner le skip reranker à intent + ambiguïté estimée (entropie top scores).
- Ajouter calibration score reranker pour filtrer passages trop incertains avant prompt final.

## 8) Plan concret d’optimisation (ordre priorisé)

### Phase A (impact élevé / faible risque)
1. Seuil KAG dual (indexation vs multi-hop expansion).
2. Lambda MMR dynamique par intent.
3. Skip reranker conditionnel (pas seulement seuil vectoriel).
4. Poids parent RRF adaptatif.

### Phase B (impact élevé / risque moyen)
1. Classifieur `need_multi_hop`.
2. Recalibrage hop penalties via jeu d’évaluation réel.
3. Expansion synonymique métier côté lexical.

### Phase C (plateforme / observabilité)
1. Dashboard retrieval end-to-end (recall@k, MRR, NDCG, redundancy, hallucination proxy).
2. Auto-tuning batch embeddings.
3. Job normalisation alias entités.

## Verdict global

- Le socle est déjà **avancé et bien structuré** (hybride + KAG + multi-hop + MMR + rerank).
- Ce n’est pas « sous-utilisé »: les briques clés sont en place.
- Le principal gap vers un RAG « quasi-parfait » est **le tuning adaptatif par intent**, et non un manque d’architecture.

## KPI cible recommandés

- Retrieval: NDCG@10 > 0.72, Recall@20 > 0.90
- Diversité: redondance top10 < 0.35
- Qualité réponse: faithfulness > 0.90 (éval LLM-as-judge + checks citations)
- Latence: P95 retrieval+rerkank < 1.5s (hors génération)
