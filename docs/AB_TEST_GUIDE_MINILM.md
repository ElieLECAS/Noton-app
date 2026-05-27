# Guide d'exécution A/B test MiniLM 512

## Objectif

Comparer les performances de la configuration **baseline** (avant optimisation) vs **optimisée** (après optimisation) pour le reranker MiniLM avec chunks multimodaux bornés à 512 tokens.

## Prérequis

1. ✅ Documents indexés dans un espace (avec pipeline multimodal)
2. ✅ `RERANKER_ENABLED=True` dans `.env`
3. ✅ Base de données accessible
4. ✅ Au moins 15-30 questions métier représentatives de votre domaine

## Méthodologie

### Configurations comparées

| Paramètre | Baseline (avant) | Optimisé (après) |
|-----------|------------------|------------------|
| `RERANK_POOL` | 50 | 30 |
| `RERANK_CHAR_CAP` | 8000 | 1700 |
| `MAX_DYNAMIC_K` | 10 | 8 |
| `MAX_CHUNK_TOKENS` | N/A | 480 |

### Métriques collectées

#### Latence
- **Latence moyenne** : temps moyen de traitement d'une requête (ms)
- **Latence P50** : temps médian (ms)
- **Latence P95** : 95e percentile (ms)

#### Qualité
- **Précision top-3** : ratio de passages pertinents dans les 3 premiers résultats
  - Un passage est pertinent s'il contient au moins 1 mot-clé attendu
- **Rappel mots-clés** : ratio de mots-clés métier trouvés dans tous les passages retournés

#### Observabilité
- **Ratio troncature** : % de chunks tronqués avant rerank
- **Chunks retournés** : nombre moyen de chunks dans la réponse finale

## Exécution

### 1. Préparer le dataset de test

Créer un fichier `questions_metier.json` avec vos questions :

```json
[
  {
    "query": "Quelle norme pour les menuiseries extérieures ?",
    "expected_keywords": ["NF EN 14351", "menuiserie", "certification"],
    "category": "norm"
  },
  {
    "query": "Dimensions maximales ouvrant fenêtre",
    "expected_keywords": ["hauteur", "largeur", "ouvrant", "dimension"],
    "category": "table"
  },
  ...
]
```

**Catégories recommandées** :
- `norm` : questions sur les normes/références
- `table` : questions nécessitant extraction de tableaux
- `instruction` : questions sur procédures/montage
- `mixed` : questions mixtes (norme + contrainte + valeur)

**Nombre de questions** :
- Minimum : **15 questions** (représentativité de base)
- Recommandé : **20-30 questions** (statistiquement significatif)

### 2. Lancer le test

```bash
cd /path/to/Noton-app
python tests/ab_test_multimodal_minilm.py \
  --space-id 1 \
  --user-id 1 \
  --queries-file questions_metier.json
```

Sans `--queries-file`, le script utilise un dataset de test par défaut (à adapter).

### 3. Interpréter les résultats

Le script affiche une comparaison détaillée :

```
==================================================================================
RÉSULTATS A/B TEST : Baseline vs Optimisé MiniLM 512
==================================================================================

Métrique                                 Baseline        Optimisé        Δ%
----------------------------------------------------------------------------------
Latence moyenne (ms)                         95.3            62.1      -34.8%
Latence P50 (ms)                             88.0            58.5      -33.5%
Latence P95 (ms)                            142.0            98.0      -31.0%
Précision top-3                              0.78            0.82       +5.1%
Rappel mots-clés                             0.85            0.83       -2.4%
Ratio troncature                           78.00%           9.00%      -88.5%
Chunks retournés (avg)                       8.2             7.5       -8.5%

==================================================================================
RECOMMANDATION :
✅ Config OPTIMISÉE recommandée : gain latence net, qualité préservée
==================================================================================
```

### 4. Décision

#### ✅ Adopter la config OPTIMISÉE si :
- Gain latence >= 20% **ET** précision stable ou meilleure (Δ >= -5%)
- Gain qualité >= 5% **ET** latence acceptable (Δ < +10%)
- Ratio troncature largement réduit (<15% vs >70%)

#### ⚠️ Valider manuellement si :
- Compromis latence/qualité mitigé (-10% < Δ précision < 0%, gain latence >10%)
- Résultats variables selon catégories de questions

#### ❌ Conserver la BASELINE si :
- Perte de qualité significative (Δ précision < -10%)
- Gain latence marginal (<10%) sans amélioration qualité

## Validation manuelle complémentaire

Après l'A/B automatique, tester manuellement sur 5-10 questions critiques :

1. Comparer côte à côte les top-3 passages (baseline vs optimisé)
2. Vérifier que les passages sont **complets** (pas de troncature)
3. Vérifier que les valeurs numériques/références sont **exactes**
4. Mesurer le temps de réponse bout-en-bout (retrieval + LLM)

## Prochaines étapes

Après validation :

1. Mettre à jour `.env` avec les paramètres retenus :
   ```bash
   RERANKER_ENABLED=True
   RERANK_POOL=30
   # (les autres sont hardcodés dans config.py)
   ```

2. Documenter la décision dans `docs/RAG_CONFIGURATION_FINALE.md`

3. Monitorer en production :
   - Latence rerank P50/P95 (via logs)
   - Ratio troncature (via `get_truncation_stats()`)
   - Feedback utilisateurs (👍/👎)

---

**Note** : Ce test compare uniquement le rerank. Pour évaluer l'impact global sur la qualité des réponses LLM, tester également les réponses complètes générées.
