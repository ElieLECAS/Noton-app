# Baseline Multimodal - MiniLM 512 tokens

**Date de baseline**: 2026-05-27  
**Objectif**: État des lieux avant optimisation pour `cross-encoder/ms-marco-MiniLM-L-6-v2` (max_length=512)

## 1. Paramètres Reranker actuels

### Configuration active (`config.py`)
```python
RERANKER_ENABLED: bool = False  # À activer pour les tests
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_POOL: int = 50
RERANK_CHAR_CAP: int = 8000  # ⚠️ MISMATCH avec max_length=512
RERANK_BATCH_SIZE: int = 16
MIN_DYNAMIC_K: int = 1
MAX_DYNAMIC_K: int = 10
SOFTMAX_CUM_THRESHOLD: float = 0.80
STUTTER_GAP: float = 0.05
ZSCORE_FLAT_THRESHOLD: float = 0.05
```

### Reranker service (`reranker_service.py`)
```python
_cross_encoder = CrossEncoder(
    settings.RERANKER_MODEL,
    device="cpu",
    max_length=512,  # ✓ Hardcodé, correct pour MiniLM L-6 v2
)
```

### ⚠️ **Problème critique identifié**
- **`RERANK_CHAR_CAP=8000`** chars ≈ **2000-2500 tokens** (ratio FR ≈ 3-4 chars/token)
- **`max_length=512`** tokens dans CrossEncoder
- **Conséquence**: Troncature silencieuse de ~75-80% du contenu par le tokenizer
- **Perte d'information**: Les passages longs perdent leur contexte final

## 2. Chunks multimodaux actuels

### Paramètres chunking (`multimodal_page_service.py`)
```python
MAX_SECTIONS_PER_PAGE = 5
MAX_CHUNK_CHARS = 1500  # ⚠️ En caractères, pas en tokens
CHUNKING_VERSION_MULTIMODAL = "multimodal_page_v2"
```

### Hiérarchie des chunks multimodaux
**Constat** (`DocumentChunk` + `multimodal_page_service.py`):
- `is_leaf=True` pour tous les chunks multimodaux
- `parent_node_id=None` systématiquement
- `hierarchy_level=0` systématiquement
- **Aucun parent hiérarchique** n'existe pour les chunks multimodaux

### Tailles effectives des chunks
- **Sections**: ≤1500 chars ≈ **375-500 tokens** (ratio variable selon densité texte)
- **`page_summary`**: **PAS de limite dure** → peut facilement dépasser 512 tokens
  - Le prompt système demande "documentation technique exhaustive (2000-4000 caractères si nécessaire)"
  - 4000 chars ≈ **1000-1300 tokens** → **dépasse largement la fenêtre MiniLM**

## 3. Pipeline de recherche actuel

### Retrieval (`space_search_service.py`)
1. **Recherche vectorielle** (pgvector) sur `is_leaf=true` → candidate_k résultats
2. **Recherche lexicale** (BM25 + tsvector) sur `is_leaf=true` → candidate_k résultats
3. **Fusion RRF** (Reciprocal Rank Fusion)
4. **Résolution parent** (pour chunks hiérarchiques legacy uniquement)
5. **Rerank cross-encoder** (pool de 50 candidats par défaut)
6. **Guardrails statistiques** (détection bégaiement)
7. **MMR** (diversification)

### Early stopping
- **Code présent** dans `reranker_service.py` (`should_early_stop()`)
- **NON branché** dans le pipeline de recherche
- Aucun appel à cette fonction dans `space_search_service.py`

### Troncature avant rerank
```python
# reranker_service.py ligne 131
truncated = text[:char_cap] if len(text) > char_cap else text
```
- Utilise `RERANK_CHAR_CAP=8000` chars
- Aucune mesure de la troncature réelle ni de son impact

## 4. Métriques et observabilité actuelles

### Logs disponibles
- Score top-1 et top-3 du rerank
- Nombre de candidats avant/après chaque étape
- Gap P@1-P@2 et z-score (guardrails)
- Statut du rerank (ok, low_confidence, early_stopped, disabled)

### ⚠️ **Métriques manquantes**
- **Aucune mesure** de la longueur réelle en tokens avant troncature
- **Aucune statistique** sur le ratio de chunks tronqués
- **Aucun tracking** de la perte d'information due à la troncature
- **Aucune latence P50/P95** détaillée du rerank

## 5. Problèmes identifiés

### Critique (bloquants qualité)
1. ❌ **Mismatch char_cap vs max_length**: 8000 chars >> 512 tokens
2. ❌ **`page_summary` non borné**: peut atteindre 1000+ tokens
3. ❌ **Pas de validation token**: chunks persistés sans vérification token count

### Majeurs (dégradation possible)
4. ⚠️ **MAX_CHUNK_CHARS en caractères**: pas de garantie <512 tokens
5. ⚠️ **RERANK_POOL=50**: peut-être trop large pour CPU MiniLM
6. ⚠️ **Early stop non utilisé**: latence CPU évitable sur requêtes faciles

### Mineurs (optimisation)
7. ℹ️ Aucune hiérarchie parent/leaf multimodale (déjà conforme Option A)
8. ℹ️ Métriques diagnostiques limitées

## 6. Compatibilité token estimée

### Ratio chars→tokens observé (français technique)
- **Texte dense** (normes, tableaux): ~3.0 chars/token
- **Texte normal**: ~3.5 chars/token
- **Texte avec espacements/listes**: ~4.0 chars/token

### Estimation chunks actuels
| Chunk type | Chars max | Tokens estimés (pessimiste) | Status MiniLM 512 |
|------------|-----------|------------------------------|-------------------|
| Section split | 1500 | **500** | ✅ OK limite |
| Section full | 1500 | **500** | ✅ OK limite |
| `page_summary` small | 2000 | **667** | ❌ Dépasse |
| `page_summary` exhaustive | 4000 | **1333** | ❌ Dépasse largement |

**Conclusion**: Les sections sont à la limite acceptable, mais les synthèses dépassent systématiquement.

## 7. Stratégie parent/leaf retenue

**Option A** (multimodal flat contrôlé, leaf-only):
- ✅ Chunks multimodaux sont déjà `is_leaf=True`, `parent_node_id=None`
- ✅ Aucun changement de hiérarchie nécessaire
- ✅ Focus sur la qualité des leafs et compatibilité MiniLM

**Décision**: Conserver le statut quo (Option A validée par la baseline).

## 8. Actions recommandées (par priorité)

### Immédiat (P0 - qualité)
1. Borner `page_summary` à 512 tokens max (token-aware split)
2. Valider sections ≤512 tokens (token-aware split si nécessaire)
3. Aligner `RERANK_CHAR_CAP` avec 512 tokens (~1500-1800 chars)

### Court terme (P1 - observabilité)
4. Ajouter métriques de troncature (chars avant, tokens estimés, ratio tronqué)
5. Logger latence P50/P95 du rerank

### Moyen terme (P2 - optimisation)
6. Tester `RERANK_POOL` réduit (20, 30, 40 vs 50)
7. Brancher early stopping si bénéfice latence démontré
8. Calibrer seuils dynamiques (SOFTMAX_CUM_THRESHOLD, MIN/MAX_DYNAMIC_K)

## 9. Next steps (plan d'implémentation)

Voir le plan détaillé dans `.cursor/plans/optimisation-multimodal-minilm-512_375412e9.plan.md`

**TODOs restants**:
1. ✅ Baseline validée (ce document)
2. ⏭️ Appliquer Option A (déjà OK, verrouiller explicitement)
3. ⏭️ Recalibrer tailles chunks (token-aware + validation)
4. ⏭️ Aligner troncature rerank (char_cap + métriques)
5. ⏭️ Optimiser retrieval/rerank pool
6. ⏭️ A/B test qualité/latence
7. ⏭️ Documentation finale

---

**Statut baseline**: ✅ VALIDÉ  
**Date**: 2026-05-27  
**Auteur**: Agent optimization pipeline
