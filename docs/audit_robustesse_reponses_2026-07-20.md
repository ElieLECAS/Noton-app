# Audit robustesse & qualité des réponses — 2026-07-20

_Branche `fix/retriever`, HEAD `afec509`. Complète l'audit du 17/07
(`audit_complet_refonte_2026-07-17.md`) : vérification de l'état RÉEL du code et,
nouveauté, du `.env` de prod. Axes demandés : retriever (entités/lexical),
chunks contextuels LLM, température & reasoning, variables d'environnement._

---

## 0. Résumé exécutif — les 5 constats qui expliquent les réponses observées

1. **Le reranker est COUPÉ en prod** (`.env` : `RERANKER_ENABLED=false`). Tout
   l'étage de jugement — seuil de pertinence 75 %, K dynamique, garde-fou
   « bégaiement » → clarification, abstention — n'existe plus en production. Le
   classement final = RRF + boosts, sans cross-encoder. C'est la cause n°1 du
   « hors-sujet confiant » (houssette → poignées). Ironie : un modèle de rerank
   **français** (`antoinelouis/crossencoder-camembert-L2-mmarcoFR`) est déjà
   configuré dans le `.env`… mais inactif.
2. **Génération à température 0.7 sur `mistral-large-latest`** (`.env`), avec un
   contexte CAG de dizaines de milliers de tokens et **aucun moment de jugement**
   entre retrieval et génération → le modèle brode et part sur le document voisin
   plausible. 4 valeurs de température coexistent (code 0.3 / compose 0.55 /
   `.env_exemple` 0.0 / `.env` 0.7).
3. **Le plan reasoning-first n'est PAS implémenté.** Aucun `reasoning_effort`,
   aucun ThinkChunk, aucun appel A/B dans le code. Le commit « reasoning-first
   workflow » (afec509) n'a livré que les docs de plan + du nettoyage.
4. **Le bug « une requête par mot, stopwords inclus » est confirmé et localisé** :
   matching d'entités KAG (`kag_retrieval_service.py:37-72`) — 1 requête SQL
   trigram par token, seul filtre `len ≥ 3`, donc « quelle », « comment »,
   « pour », « avec » partent en base. En plus : matching **sensible aux accents**
   (« reglage » ≠ « réglage »). BM25, lui, est sain (requête globale, dico FTS
   `french`).
5. **Config ingérable** : 198 settings, **~24 variables réglées dans
   `.env`/compose mais silencieusement ignorées** (`extra="ignore"`), **21 champs
   morts**, 5 sources de vérité de fait, **zéro validation au démarrage**
   (SECRET_KEY absente = boot OK, crash à la 1re requête). Divergences prod
   dangereuses, dont `COLPALI_MODEL_NAME` v1.0 en prod vs v0.1 par défaut code
   (risque index/requête incompatibles).

**Lien avec les conversations exemples :**

| Symptôme observé | Cause racine |
|---|---|
| « houssette » → réponse sur les poignées, avec aplomb | Reranker OFF (pas de seuil 75 %, pas d'abstention) + temp 0.7 |
| TGY3702 : la question demande la crémone 4 points, la réponse décrit longuement la 3 points | Aucun jugement « ces docs répondent-ils à CETTE question ? » avant rédaction + temp 0.7 |
| « Je n'ai pas accès aux images des catalogues » (faux) | `ILLUSTRATION_REFERENCE_PATTERN` = `\b\d{3,5}[A-Za-z]?\b` (config.py:221) ne matche pas TGY3702/TMX13 → 0 illustration ancrée → le prompt fait nier la capacité |
| Mode guidé Askey : 5 questions de suite, l'utilisateur doit couper | Budget de questions non borné dans le flux guidé (cadrage prévu au plan cas réels, non fait) |
| Réponses correctes mais noyées de détails hors périmètre | CAG jusqu'à 100k tokens collés en fin de system prompt, instruction diluée ; le hack « RAPPEL FINAL » (rag_generation_service.py:275) compense partiellement |

---

## 1. Retriever

### 1.1 Lexical / BM25 — sain, contrairement au diagnostic initial

- Canal lexical du RAG = BM25/tsvector Postgres
  (`page_retrieval_service.py::retrieve_bm25_pages`, l.1189). **Une seule
  `tsquery` globale** (`websearch_to_tsquery`/`plainto_tsquery`, dictionnaire
  `'french'` → stemming + stopwords gérés par Postgres). Chaîne de repli AND →
  OR avec set de stopwords FR+EN (`_BM25_STOPWORDS`, l.27-35) et tokens
  discriminants seulement.
- `lexical_search_service.py` (ILIKE par token, sans stopwords) n'est PAS le
  RAG : c'est la recherche mot-clé de l'UI. Coût faible (un seul OR), mais un
  filtrage stopwords y serait bienvenu par propreté.

### 1.2 KAG / entités — le vrai foyer du bug

`kag_retrieval_service.py::_query_entity_candidates` (l.27-104) :

- `tokens = [t for t in re.split(r"\W+", normalized_query) if len(t) >= 3][:8]`
  → **aucun stopword filtré** (« quelle », « dimension », « comment », « pour »,
  « avec », « régler »… passent tous).
- **Boucle `for token in tokens:` avec une requête SQL trigram par token**
  (jusqu'à 8 allers-retours, N+1) sur `knowledgeentity.name_normalized` +
  sous-requête `entityalias.alias_normalized`.
- `normalize_entity_name` (kag_extraction_service.py:287-296) = NFKC + lower,
  **sans suppression d'accents** ; aucun `unaccent` dans le repo ; index trigram
  sur texte accentué → « reglage » matche mal « réglage ».
- `EntityAlias` : contrainte unique `(space_id, alias_normalized)`
  (knowledge_entity.py:74-81) → un alias ne pointe qu'une entité par espace,
  **collisions abandonnées en silence** (logger.debug), jamais re-rattachées.

**Correctif cible** : une requête unique (`unnest` des tokens), stopwords FR
filtrés en amont, `unaccent` (ou déaccentuation Python à l'écriture ET à la
lecture de `*_normalized`), alias many-to-many ou résolution des collisions.

### 1.3 Fusion & boosts — double-comptages et étages neutralisés

- RRF (`fuse_multimodal_hits`, page_retrieval_service.py:1270-1347), poids
  identiques tous canaux, puis **6 étages de score** empilés : RRF → boost
  catégorie (×, cap 1.5) → boost ancre (×) → rerank (remplace le score) →
  soft boosts additifs (source/matériau/entité) → autorité de source (+0.8·conf).
- **Source comptée 2×** : `apply_soft_boosts_to_passages`
  (retrieval_boost_service.py:274-277) PUIS `refine_with_source_authority`
  (space_search_service.py:1300-1301) sur le même critère.
- **KAG compté 2×** : canal RRF ET `entity_boost` ensuite (l.285).
- **Boosts catégorie/ancre largement neutralisés quand le reranker est actif**
  (le cross-encoder re-score tout le pool) — et quand il est inactif (prod !),
  ils deviennent le signal de classement final sans avoir été calibrés pour ça.

### 1.4 Reranker

- Défaut code : `cross-encoder/ms-marco-MiniLM-L-6-v2` (**anglais**, config.py:257)
  sur du texte français. Le `.env` pointe déjà
  `antoinelouis/crossencoder-camembert-L2-mmarcoFR` (français) **mais
  `RERANKER_ENABLED=false`**.
- Les seuils actuels (`RAG_MIN_PERTINENCE`, calibration sigmoïde `+1.5`,
  `RERANKER_MIN_SCORE=-3.0`, `STUTTER_GAP`, `ZSCORE_FLAT_THRESHOLD`) ont été
  réglés pour les logits ms-marco → **à recalibrer sur le golden** avec le
  modèle FR (les valeurs prod actuelles 0.5 / 0.001 / 0.001 ont de fait
  désactivé les garde-fous).

### 1.5 Gating ColPali

- `should_use_colpali` (space_search_service.py:52-75) : marqueurs visuels en
  **substring** (« vue » matche « revue », « montre » matche « démontre ») →
  sur-déclenchement ; et **faux négatifs** si requête visuelle sans marqueur
  avec ≥5 hits texte (le fallback `COLPALI_GATING_FALLBACK_MIN_HITS=5` ne se
  déclenche pas).
- Le gate n'existe pas sur le chemin multi-groupe (`_retrieve_one_group_hits`,
  l.697) — OFF par défaut, mais incohérent.

### 1.6 Robustesse (retriever)

- Exceptions avalées en cascade : KAG → `[]`, BM25 → `[]`, embedding KO →
  pgvector silencieusement neutralisé, reranker KO → scores RRF bruts,
  pipeline entier → `status: disabled`. **Aucun signal remonté** à l'appelant ni
  métrique : une panne partielle ressemble à « peu de résultats ».
- N+1 : KAG par token ; `apply_soft_boosts_to_passages` fait `session.get`
  par passage (2× pour la source) ; `load_l1_chunks_for_page` recharge TOUTES
  les feuilles du document puis filtre en Python, appelé répétitivement.

---

## 2. Chunks contextuels LLM

### 2.1 Ce qui existe vraiment (≠ contextual retrieval Anthropic)

- **Chunking vivant** = extraction vision LLM page par page
  (`vision_page_extraction_service`, Ministral 8B, temp 0.0), 3 niveaux :
  L0 ancre de page (ID stable), L1 feuilles sémantiques, L2 enrichissements.
- **(A) Préfixe déterministe** (`_build_embed_text`,
  document_indexing_service.py:548-601) : métadonnées (doc/section/catégories/
  entités/matériau/source) préfixées **au seul texte embeddé**.
- **(B) Enrichissement LLM L2** (`contextual_enrichment_service`,
  mistral-small, fenêtre 3 pages, overlap 1) : crée des **chunks de synthèse
  séparés**, pas un contexte par chunk.
- Il n'y a donc **pas** de contexte LLM court préfixé à chaque chunk L1.

### 2.2 Problèmes concrets

1. **Asymétrie des trois textes** : l'embedding voit préfixe+contenu ; le BM25
   (`tsv_content`) voit le contenu seul ; le LLM de génération (CAG) voit le
   contenu seul. Le signal contextuel n'atteint ni le lexical ni la génération.
2. **IDs non stables** : `node_id = uuid4()` pour L1 et L2 (seul L0 est stable)
   → réindexation non idempotente, embeddings re-payés, impossible de rejouer
   KAG/catégories sans re-ingérer.
3. **Cap dur 12 chunks/page avec troncature silencieuse**
   (`vision_page_extraction_service.py:212,224`) → perte de contenu non tracée
   sur les pages denses (tableaux de références !).
4. **Aucun gate de fidélité numérique** sur les réécritures L2 : les consignes
   anti-invention sont dans le prompt, rien dans le code → risque de cotes/réfs
   inventées indexées.
5. **Dédup L2 absente** (fenêtres chevauchantes → triplons possibles).
6. **CAG** : jusqu'à 100k tokens (code) appendus au system prompt → dilution de
   l'instruction ; en prod le budget est réduit (50k / 3 docs) ET
   `SPACE_CONTEXT_MAX_PASSAGE_CHARS=4000` tronque les passages — deux réglages
   qui se contredisent (pack de documents entiers vs troncature à 4000 chars).
7. **Code mort** toujours là : bloc Docling de `chunking_service` (~1900 l.),
   fonctions markdown de `chunk_service`, `embed_enrichment_chunks_for_document`,
   chemins DEPRECATED de `document_service_new`, colonne JSONB dupliquée
   `metadata_`/`metadata_json`.

### 2.3 Pour un vrai contextual retrieval

Passe LLM par chunk L1 (résumé du doc + chunk → 1-2 phrases de contexte),
contexte injecté **dans l'embedding ET dans le tsvector ET disponible au
packing**. Pré-requis : IDs stables par hash (sinon chaque réindexation re-paye
la passe) ; sinon commencer par la symétrie lexical/dense du préfixe existant
(quick win sans LLM).

---

## 3. Génération : température, reasoning, vérification

### 3.1 Température — état et cible

| Source | Valeur |
|---|---|
| `config.py:72` | 0.3 |
| `docker-compose.yaml` | 0.55 |
| `.env_exemple` | 0.0 |
| **`.env` prod (gagnant)** | **0.7** |

De plus, les appels de compréhension (`lightweight_query_understanding`,
`query_reasoning_service`) ne passent **aucune température** → défaut serveur
Mistral (~0.7) pour des extractions JSON qui devraient être à 0.0-0.1. Les
commentaires `chat.py:23/38` (« 0.0 par défaut ») sont faux.

**Cible : 0.2 unifié partout** (une seule valeur, `.env` = compose = code),
0.0-0.1 explicite sur les appels JSON de compréhension. Garde-fou : si bascule
reasoning, vérifier la reco de sampling Mistral pour les modèles reasoning
(historiquement plus haute) sur le golden avant de conclure.

### 3.2 Reasoning : custom ou natif ? → NATIF, ne pas concevoir custom

Réponse à la question posée :

- `mistral-large-latest` (Large 3) ne supporte **pas** `reasoning_effort`
  (HTTP 422, vérifié le 17/07 sur docs.mistral.ai). Le custom « chain-of-thought
  par prompt » sur large = plus de code, pas de séparation thinking/réponse,
  pas de garantie de format → à garder uniquement comme repli.
- `mistral-small-latest` (Small 4) et `mistral-medium-3-5` supportent
  `reasoning_effort={high,none}` nativement, avec ThinkChunk séparé du texte.
- **Recommandation (déjà actée le 17/07, toujours valable)** :
  `mistral-small-latest` + `reasoning_effort="high"` sur les deux appels
  (compréhension = appel A, jugement+rédaction = appel B), `none` sur les fast
  paths. **Repli si small décroche en rédaction sur le golden :
  `mistral-medium-3-5` sur l'appel B uniquement** (une variable de config, pas
  un refactor). Medium 4 partout = plus cher pour un gain non démontré ;
  trancher au golden, pas a priori.

### 3.3 État de l'implémentation : rien n'est fait, et 3 pré-requis bloquants

`git grep reasoning_effort|ThinkChunk` → **0 résultat**. La génération est un
unique appel streaming sur `MODEL_FAST`. Pré-requis avant toute bascule
(= chantiers C0/C1/C3 du plan `plan_impl_reasoning_query_generation_2026-07-17.md`) :

1. **Client incompatible** : `chat_stream` (mistral_service.py:328-498) ne lit
   `delta.content` qu'en string ; en reasoning c'est une **liste** (ThinkChunk)
   pendant la phase de réflexion → à adapter + masquer le thinking du SSE.
2. **Bug images** : `is_vision_model` (rag_generation_service.py:27-36)
   reconnaît `pixtral|vision|large-latest|ministral|gpt-4o` mais **pas
   `small`** → basculer `MODEL_FAST` sur small **couperait les PNG de pages**
   de la génération alors que Small 4 est multimodal. À corriger AVANT la
   bascule.
3. **`max_tokens`** : le thinking consomme le budget de complétion →
   `CAG_MAX_COMPLETION_TOKENS` à relever (~8192) en mode reasoning.

### 3.4 Jugement & vérification — les deux chaînons manquants

- **Aucun appel LLM de jugement** entre retrieval et génération. Le seul
  jugement est le gate du reranker… désactivé en prod. Résultat : le retrieval
  renvoie toujours de la matière → le modèle répond toujours → hors-sujet
  confiant.
- **Aucune vérification post-génération** : pas de check programmatique des
  références/cotes citées (regex → présence littérale dans le contexte packé),
  pas de retry, pas d'abstention a posteriori. `chat_critique_service` est
  **orphelin** (défini, jamais importé). `FAQ_POST_DRAFT` réduit à un
  `nb_faq_passages: 0` codé en dur.
- La boucle d'outils (`chat()` non-stream, brave-only) n'est **jamais utilisée**
  par le pipeline RAG streaming.
- Robustesse stream : idle 90 s / durée max 400 s → `break` **silencieux**
  possible (réponse tronquée sans signal) ; retry streaming seulement si rien
  n'a été émis.

---

## 4. Variables d'environnement — 198 champs, 5 sources de vérité

### 4.1 Chiffres

- **198 champs** dans `Settings` (config.py), dont ~43 pour le seul domaine
  retrieval/reranking.
- **~24 variables réglées dans `.env`/compose mais ignorées** (absentes de
  `Settings`, supprimées par `extra="ignore"`) : `RERANKER_PROVIDER` (le
  `getattr` retombe TOUJOURS sur "local" — bug net), `MMR_*` (4),
  `RRF_DYNAMIC_*` (4), `DOCLING_*`/`OCR_*` (9), `RAG_RETRIEVAL_K`/`RAG_GRADE_K`/
  `RAG_CONTEXT_PAGES`/`RAG_ILLUSTRATION_PAGES`, `EMBEDDING_DEVICE`,
  `RERANK_GROUP_CHAR_CAP`, `USE_MULTIMODAL_RETRIEVAL`.
- **21 champs morts** (jamais lus) : `VISION_RERANK_*` (4, feature forcée
  `False` en dur dans space_search_service.py:661), `EARLY_STOP_ENABLED`/
  `MEAN_THRESHOLD` (l'interrupteur d'une feature à moitié câblée), `FAQ_*` (4),
  `VISION_MODEL`/`VISION_MAX_*` (3), `SPACE_CHAT_TOP_P`, `BM25_MAX_QUERY_TERMS`,
  etc.
- **5 sources de vérité** : config.py + compose (`x-app-env` avec ses propres
  défauts `${VAR:-…}`) + `.env_exemple` + `.env` + **défauts `getattr`/
  `os.getenv` éparpillés dans le code** (`WINDOW_MAX_PAGES` : 2 dans config,
  12 dans le getattr ; `SPACE_CONTEXT_MAX_CHARS`, `MIN_VECTOR_SIMILARITY`,
  `LOG_LEVEL` lus hors Settings…).

### 4.2 Divergences prod à trancher (extrait)

| Variable | code | compose | .env_ex | .env prod | Impact |
|---|---|---|---|---|---|
| `SPACE_CHAT_TEMPERATURE` | 0.3 | 0.55 | 0.0 | **0.7** | broderie SAV |
| `MODEL_FAST` | small | small | large | **large** | coût + pas de reasoning possible |
| `RERANKER_ENABLED` | true | true | true | **false** | plus aucun gate qualité |
| `RERANKER_MODEL` | ms-marco (EN) | idem | idem | **camembert mmarcoFR** | prêt mais inactif |
| `RAG_MIN_PERTINENCE` | 0.75 | — | 0.75 | **0.5** | seuil affaibli |
| `MIN_DYNAMIC_K` | 0 | 2 | 0 | **2** | 2 passages forcés même non pertinents → jamais d'abstention |
| `STUTTER_GAP` / `ZSCORE_FLAT` | 0.05 | 0.05 | 0.05 | **0.001** | clarification basse-confiance quasi jamais déclenchée |
| `COLPALI_MODEL_NAME` | colqwen2-**v0.1** | — | — | **v1.0** | index et requêtes potentiellement encodés par 2 modèles ≠ |
| `CAG_TOKEN_BUDGET` / `MAX_DOCS` | 100k / 8 | 100k / 8 | — | **50k / 3** | packing réduit |
| `SPACE_CONTEXT_MAX_PASSAGE_CHARS` | 12000 | 12000 | 4000 | **4000** | tronque les pages entières du CAG |
| `RAG_TOP_K` | 10 | 8 | 10 | **20** | pool élargi sans reranker derrière |
| `KAG_EXTRACTION_MODEL` | small | small | small | **ministral-8b** | qualité extraction entités |
| `SECRET_KEY` / `MISTRAL_API_KEY` | None accepté | — | — | fournis | aucun fail-fast |

### 4.3 Structure cible

1. `config.py` = source unique des défauts (supprimer les `os.getenv` dans les
   défauts, interdire `getattr(settings, "X", défaut)` dans le code).
2. `extra="forbid"` (au moins CI/staging) → les 24 orphelines explosent au boot
   au lieu d'être ignorées ; câbler ou supprimer chacune.
3. `@model_validator` fail-fast sur `SECRET_KEY` (SecretStr sans défaut),
   `MISTRAL_API_KEY`, `DATABASE_URL`.
4. Supprimer le bloc `x-app-env` de compose (pydantic lit déjà `.env`) — ne
   garder que POSTGRES_* et l'infra.
5. Régénérer `.env_exemple` par script depuis `Settings` (plus jamais de
   divergence manuelle).
6. Purger les 21 morts + 5 redondants (`REDIS_URL`/`CELERY_*`/`LANGSMITH_*`
   lus via os.getenv direct).

---

## 5. Plan d'action priorisé

### P0 — Config uniquement, effet immédiat, 0 refactor (½ journée)
1. `.env` : `SPACE_CHAT_TEMPERATURE=0.2` (garder 0.3 en option de test golden).
2. `.env` : `RERANKER_ENABLED=true` avec le camembert-mmarcoFR déjà configuré ;
   **recalibrer** `RAG_MIN_PERTINENCE` (retour vers 0.75 à valider),
   `MIN_DYNAMIC_K=0`, `STUTTER_GAP=0.05`, `ZSCORE_FLAT_THRESHOLD=0.05` sur le
   golden (les logits du modèle FR ≠ ms-marco, la calibration sigmoïde `+1.5`
   est à revérifier).
3. Trancher `COLPALI_MODEL_NAME` (aligner code sur v1.0 si l'index a été
   construit avec, sinon réindexer).
4. `ILLUSTRATION_REFERENCE_PATTERN` → couvrir les réfs alphanumériques
   (ex. `\b(?:[A-Z]{1,4}\d{2,6}[A-Za-z]?|\d{3,6}[A-Za-z]?)\b`) + retirer du
   prompt la négation de capacité d'affichage d'images.
5. Aligner compose et `.env_exemple` sur les mêmes valeurs (fin des 4 tables de
   vérité pour la température).

### P1 — KAG & fusion (2-4 j)
6. KAG : stopwords FR + requête unique `unnest` + déaccentuation
   (écriture + lecture) ; résoudre les collisions d'alias.
7. Fusion : supprimer le double-boost source et le double-comptage KAG ;
   décider du sort des boosts catégorie/ancre (post-rerank ou supprimés).
8. Remonter les échecs de canaux (log structuré + compteur) au lieu de `[]`.

### P2 — Génération reasoning natif (1-2 sem, plan C0→C5 existant)
9. C0 : client `chat_stream` compatible `delta.content` liste + masquage
   thinking ; **corriger `is_vision_model` pour small** ; spike compat
   reasoning × json_object × vision × tools.
10. Bascule `MODEL_FAST=mistral-small-latest` + `reasoning_effort` (A puis B),
    mesurée au golden à chaque palier ; repli `mistral-medium-3-5` sur l'appel
    B seul si la rédaction décroche. Pas de reasoning custom.
11. Vérification programmatique post-génération (réfs/cotes ∈ contexte packé,
    0 LLM) + brancher ou supprimer `chat_critique_service`.

### P3 — Chunks contextuels (1-2 sem, après P2)
12. IDs stables par hash (pré-requis économique de tout le reste).
13. Symétrie lexical/dense du préfixe (search_text ou 2e tsvector).
14. Lever le cap 12 chunks/page (split sans perte + ratio de couverture loggé).
15. Gate de fidélité numérique sur les L2 + dédup ; ensuite seulement, la passe
    « contexte par chunk » façon Anthropic si le golden montre un gain.

### P4 — Durcissement config (2-3 j, en parallèle de P1)
16. `extra="forbid"` + fail-fast secrets + purge morts/orphelines + compose
    dégraissé + `.env_exemple` généré.

**Mesure** : chaque palier passe par le golden (`tests/fixtures/golden/`,
`docker compose exec web pytest`) — à brancher en CI, sinon aucune de ces
améliorations n'est prouvable.
