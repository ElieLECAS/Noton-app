# Audit technique complet RAG/KAG — Bibliothèque générale (2026-05-19)

## Résumé exécutif

Le pipeline actuel est **déjà avancé**: chunking hiérarchique Docling, récupération hybride vectorielle+lexicale+KAG, orchestrateur multi-hop, MMR, reranker cross-encoder, et observabilité minimale sont en place. Les gains restants sont surtout des gains de **tuning adaptatif** (par intention de requête et niveau d’ambiguïté), plus que des gains d’architecture brute.

## 1) Pipeline actuel (lecture code)

### 1.1 Chunking
- Versions de chunking tracées (`docling_hierarchical_v1/v2`, etc.) et mode adaptatif déclaré.  
- Le chunking transporte une hiérarchie exploitable au retrieval (`node_id`, `parent_node_id`, `is_leaf`) et inclut des métadonnées riches.

### 1.2 Embeddings
- Embedding en singleton + batch avec retries/backoff (bon compromis robustesse/perf).  
- Modèle homogène (`BAAI/bge-m3`) et garde-fous hors-ligne.

### 1.3 Entités / relations KAG
- Taxonomie métier structurée + contraintes de relations typées strictes.  
- Blocage des relations vagues et normalisation relationnelle déjà intégrés.

### 1.4 Retrieval hybride
- Fusion multi-canaux avec RRF (vectoriel, lexical, KAG, parent enriched).  
- Seuils minimum RRF, fallback lexical, puis reranking.

### 1.5 Multi-hop
- Orchestrateur borné (max hops, budget candidats, patience) avec pénalités par profondeur.  
- Déclenchement heuristique (patterns forts/faibles) + signaux KAG.

### 1.6 MMR + reranking
- MMR activé avant reranker pour diversifier le top-k.  
- Reranking BGE (ou fallback ST), skip conditionnel basé sur seuil de similarité vectorielle.

## 2) Diagnostic d’optimisation (écarts vers “RAG parfait”)

### 2.1 Chunking: bon socle, mais pas encore “intent-aware”
**Constat**: le chunking est robuste, mais la granularité n’est pas encore pilotée dynamiquement par type de question (factuelle, comparative, procédure, normatif).  
**Impact**: rappel/precision variables selon type de document (tableaux denses vs prose).

### 2.2 KAG: excellent rappel, qualité hop perfectible
**Constat**: seuil de confiance entités global (unique) appliqué de façon uniforme.  
**Impact**: bon recall, mais expansion multi-hop parfois bruitée si entités faibles entrent trop tôt.

### 2.3 Hybride: poids statiques
**Constat**: certains poids sont fixes (ex. parent list, MMR lambda global, pénalités hop fixes).  
**Impact**: sous-optimal selon intention utilisateur (comparatif vs mono-factuel).

### 2.4 Skip reranker trop global
**Constat**: skip reranker basé principalement sur similarité vectorielle haute.  
**Impact**: risque de faux positifs sur requêtes ambiguës (similarité élevée ne garantit pas la bonne granularité de réponse).

### 2.5 Lexical fallback
**Constat**: fallback lexical indispensable, mais potentiellement bruyant sans garde post-rerank plus strict.  
**Impact**: montée de passages “lexicalement proches mais sémantiquement faibles”.

## 3) Plan d’optimisation priorisé (concret)

## Phase A — Quick wins (faible risque, fort ROI)
1. **Dual-threshold KAG**  
   - Seuil indexation (bas) conservé pour rappel.  
   - Seuil expansion multi-hop (plus haut) pour réduire le bruit de propagation.
2. **MMR adaptatif par intention**  
   - Requête comparative: lambda plus faible (diversité plus forte).  
   - Requête factuelle: lambda plus élevé (focus pertinence).
3. **Skip reranker “aware ambiguity”**  
   - Ajouter un critère d’entropie/disparité top scores + type d’intention.
4. **Poids parent enrichi adaptatif**  
   - Augmenter sur questions explicatives/synthèse; diminuer sur requêtes factuelles pointues.

## Phase B — Gains structurels (risque moyen)
1. **Classifier need_multi_hop** (léger, supervisé).  
2. **Recalibration des pénalités de hop** sur un dataset d’évaluation réel.  
3. **Expansion synonymique métier contrôlée** dans le canal lexical (dictionnaire interne).

## Phase C — Excellence opérationnelle
1. **Dashboard retrieval qualité/perf**: recall@k, ndcg@k, redundancy@k, latency p50/p95, rerank-skip rate.  
2. **Auto batch-sizing embeddings** selon device/latence courante.  
3. **Job de consolidation alias entités** (hebdo) avec score de stabilité.

## 4) Cibles KPI “RAG parfait pragmatique”

- Recall@20 ≥ 0.90  
- NDCG@10 ≥ 0.72  
- Redondance top10 ≤ 0.35  
- Faithfulness ≥ 0.90 (éval LLM-as-judge + vérification citationnelle)  
- P95 retrieval+rerrank ≤ 1.5s (hors génération)

## 5) Verdict

- Le pipeline n’est **pas sous-exploité**: les briques clés sont là et bien assemblées.
- Le principal levier est l’**adaptativité runtime** (intent/ambiguïté/type contenu), pas la réécriture complète.
- Objectif réaliste: passer de “très bon RAG” à “quasi parfait en prod” via tuning guidé métriques + protocole d’évaluation continu.
