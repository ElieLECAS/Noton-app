# Plan P0 + P1 — config, reranker, ColPali toujours actif

_2026-07-20 · branche `fix/retriever`. Concrétise le plan de
`audit_robustesse_reponses_2026-07-20.md`. Certaines actions sont **déjà
appliquées** (marquées ✅) ; les autres sont prêtes à exécuter avec le diff._

---

## 0. Réponse directe : pgvector fait-il doublon avec ColPali ?

**Non — complémentarité, prouvée par le code.**

- **ColPali indexe TOUTES les pages** : ses patches visuels sont attachés au chunk
  L0 `page_anchor`, créé pour chaque page même sans texte
  (`document_indexing_service.py:422-425`, `691-715`).
- **pgvector / BM25 n'indexent QUE les pages ayant produit du texte L1** :
  `content_type IN ('semantic_leaf','contextual_enrichment')` et un `content`
  non vide (`document_indexing_service.py:477-479`, filtre
  `page_retrieval_service.py:1046-1048`).

Conséquence dure : **une page purement visuelle (schéma, plan, photo, tableau non
océrisé) est invisible pour pgvector/BM25 mais visible pour ColPali.** Et sur une
page qui a du texte, les deux portent des signaux différents : pgvector = la
sémantique du **texte extrait** (+ préfixe déterministe), ColPali = la **mise en
page / le visuel / la position**. Inversement, ColPali (image DPI 150) peut rater
un texte fin/dense que pgvector capte parfaitement — le rerank ré-extrait
d'ailleurs le texte pymupdf pour les pages ColPali-dominantes
(`page_reranker_service.py:106-133`), preuve que ColPali seul ne suffit pas côté
texte. **On garde les deux.**

---

## 1. P0 — config (déjà appliqué dans `.env` et `.env_exemple`)

| Variable | Avant (prod) | Après | Pourquoi |
|---|---|---|---|
| `SPACE_CHAT_TEMPERATURE` | 0.7 | **0.2** ✅ | Broderie SAV sur contexte CAG massif |
| `RERANKER_ENABLED` | false | **true** ✅ | Rétablit seuil pertinence + K dynamique + abstention |
| `RERANKER_MODEL` | camembert FR (inactif) | **camembert FR (actif)** ✅ | Reranker français sur corpus français |
| `RERANK_POOL` | 100 | **40** ✅ | Précision sur un pool focalisé (cf. §3) |
| `RAG_MIN_PERTINENCE` | 0.5 | **0.75** ✅ ⚠ | Cible ; **à recalibrer sur golden** (logits camembert ≠ ms-marco) |
| `RERANK_CHAR_CAP` | 8000 | **1800** ✅ | Aligné sur `max_length=512` ; au-delà = tronqué en silence |
| `MIN_DYNAMIC_K` | 2 | **0** ✅ | 2 = 2 passages forcés même hors-sujet → jamais d'abstention |
| `STUTTER_GAP` / `ZSCORE_FLAT_THRESHOLD` | 0.001 | **0.05** ✅ | Restaure le garde-fou « bégaiement » → clarification |
| `COLPALI_GATING_ENABLED` | true | **false** ✅ | ColPali toujours actif (cf. §4) |

Défauts `config.py` alignés en parallèle (fin des divergences code↔prod) :
`RERANKER_MODEL`, `RERANK_CHAR_CAP=1800`, `COLPALI_MODEL_NAME=v1.0`,
`COLPALI_GATING_ENABLED` défaut `false`, et **ajout du champ manquant
`RERANKER_PROVIDER`** (auparavant lu via `getattr` → toujours `"local"` en
silence, un `RERANKER_PROVIDER=mistral` n'avait aucun effet). ✅

**Inchangé volontairement** : `MODEL_FAST=mistral-large-latest` (la bascule vers
`mistral-small` est **P2** — bloquée par le bug `is_vision_model` qui couperait
les images, cf. §5). La température 0.2 sur large est sûre dès maintenant.

### P0 restant (code, prêt à appliquer)

1. **Pattern d'illustration** (`config.py:221`) — cause du « je n'ai pas accès aux
   images ». Actuel `r"\b\d{3,5}[A-Za-z]?\b"` : rate les réfs alphanumériques
   (TGY3702, TMX13) **et** les réfs numériques à 6 chiffres (259879). Proposition :
   ```python
   ILLUSTRATION_REFERENCE_PATTERN: str = os.getenv(
       "ILLUSTRATION_REFERENCE_PATTERN",
       r"\b(?:[A-Z]{1,4}\d{2,6}[A-Za-z]?|\d{3,6}(?:\.\d{1,2})?[A-Za-z]?)\b",
   )
   ```
   Couvre TFZ60032, TGY3702, TMX13, T910002, 259879, 9718.3. À vérifier au golden
   pour les faux positifs (dimensions type 2200) — le préfixe lettres est
   haute-précision, le risque numérique existait déjà.
2. **Prompt génération** : retirer la formulation qui fait NIER au modèle sa
   capacité à montrer des images (le pipeline ancre bien des illustrations quand
   le pattern matche). À localiser dans `SPACE_CHAT_SYSTEM_PROMPT` (`chat.py:212`)
   et les templates de génération.

---

## 2. P1 — retriever KAG & fusion (code)

### 2.1 KAG : « une requête par mot y compris stopwords » + sensibilité aux accents
`kag_retrieval_service.py:37-72`. Trois défauts cumulés :
- **Aucun stopword filtré** : seul `len ≥ 3` → « quelle », « comment », « pour »,
  « avec », « régler » partent en base.
- **N+1** : une requête SQL trigram par token (jusqu'à 8).
- **Sensible aux accents** : `normalize_entity_name` (`kag_extraction_service.py:287`)
  ne déaccentue pas ; aucun `unaccent` → « reglage » matche mal « réglage ».

Correctif :
1. Set de stopwords FR (réutiliser `_BM25_STOPWORDS` de
   `page_retrieval_service.py:27-35` ou une liste dédiée).
2. **Une seule requête** : `unnest(:tokens)` + jointure trigram, au lieu de la
   boucle Python.
3. Déaccentuation à **l'écriture ET à la lecture** des colonnes `*_normalized`
   (extension Postgres `unaccent`, ou `unidecode` Python appliqué symétriquement),
   + reconstruire l'index trigram sur la valeur déaccentuée.
4. `EntityAlias` : résoudre les collisions `(space_id, alias_normalized)` au lieu
   de les abandonner en `logger.debug` (`kag_extraction_service.py:910-916`) —
   alias many-to-many ou politique de fusion explicite.

### 2.2 Fusion : double-comptages
- **Source comptée 2×** : `apply_soft_boosts_to_passages`
  (`retrieval_boost_service.py:274-277`) **et** `refine_with_source_authority`
  (`space_search_service.py:1300-1301`). → garder **un seul** des deux.
- **KAG compté 2×** : canal RRF **et** `entity_boost`
  (`retrieval_boost_service.py:285`). → retirer l'`entity_boost` ou le canal, pas
  les deux.
- Décider du sort des boosts catégorie/ancre : avec le reranker réactivé, ils sont
  re-écrasés par le cross-encoder (appliqués sur `rrf_score`, avant rerank). Les
  déplacer **après** le rerank (sur le score final) ou les supprimer.

### 2.3 Remonter les échecs de canaux
Aujourd'hui KAG/BM25/embedding/reranker échouent en `return []` silencieux → une
panne partielle ressemble à « peu de résultats ». Ajouter un statut par canal +
compteur/log structuré dans `_run_retrievers` et `fuse_multimodal_hits`.

---

## 3. Reranker « dans les règles de l'art »

**Chargement — ✅ conforme après patch.**
- **Singleton, jamais rechargé par requête** : `_get_cross_encoder`
  (`reranker_service.py:92`) est un singleton lazy thread-safe (globals +
  `threading.Lock`). C'était déjà le cas.
- **Chargé au démarrage** (nouveau ✅) : `warmup_cross_encoder`
  (`reranker_service.py`) + thread daemon dans `main.py:startup_event`, à l'image
  du préchargement ColPali. La 1re requête ne paie plus le téléchargement HF ni le
  1er forward ; un échec de chargement est loggé au boot (non bloquant) au lieu de
  surgir sur la 1re requête utilisateur.

**Entonnoir recall → précision** (répond à « la précision doit se faire sur les 20
derniers chunks »). L'ordre est :
1. Retrievers (colpali/pgvector/bm25/kag) → **recall**, pool fusionné par RRF.
2. `RERANK_POOL=40` candidats passés au cross-encoder → **précision** (re-score
   paires (question, passage)). 40 garde une marge de rappel pour le reranker ;
   100 diluait et coûtait du CPU pour rien.
3. K dynamique (`MAX_DYNAMIC_K=12`, `SOFTMAX_CUM_THRESHOLD=0.80`) + seuil
   `RAG_MIN_PERTINENCE` → **sélection finale** des passages réellement pertinents,
   avec **abstention** possible (`MIN_DYNAMIC_K=0`).

**`RERANK_CHAR_CAP=1800` (règle de l'art)** : le cross-encoder a `max_length=512`
tokens (`reranker_service.py:107`) — limite positionnelle de CamemBERT. 8000 chars
(~2300 tokens) étaient tronqués **par le tokenizer** → 70 % du passage ignoré sans
le savoir. 1800 chars ≈ 510 tokens = ce que le modèle voit réellement, et ça colle
aux chunks L1 (≤ 480 tokens). Aucune perte cachée.

**⚠ Recalibration obligatoire avant prod** : les seuils actuels
(`RAG_MIN_PERTINENCE=0.75`, calibration sigmoïde `+1.5` à
`reranker_service.py:351-361`, `RERANKER_MIN_SCORE=-3.0`) ont été réglés pour les
**logits ms-marco**. Le camembert-L2-mmarcoFR a une échelle de logits différente.
Procédure :
1. Passer le golden (`tests/fixtures/golden/`) avec le reranker FR, logguer la
   distribution des logits bruts et des scores sigmoïde.
2. Régler `RAG_MIN_PERTINENCE` pour que les bons passages passent et les
   hors-sujet soient coupés (viser précision, l'abstention `MIN_DYNAMIC_K=0`
   fait le reste).
3. Ajuster `RERANKER_MIN_SCORE` et la constante `+1.5` si la sigmoïde sature.
4. Ne pousser en prod qu'après ce palier. **Tant que non calibré, garder
   `MIN_DYNAMIC_K=0` sous surveillance** (un seuil trop haut + K=0 = abstentions
   massives ; un seuil trop bas = bruit).

---

## 4. ColPali toujours actif

**Fait ✅** : `COLPALI_GATING_ENABLED=false` (`.env`, `.env_exemple`, défaut
`config.py`). Sémantique vérifiée : `should_use_colpali`
(`space_search_service.py:67-68`) retourne `(True, "gating_off")` en sortie
anticipée → ColPali n'est **jamais** court-circuité. Le chemin multi-groupe
lançait déjà ColPali sans condition.

**Coût** : uniquement à la **requête** (encode ColQwen2 + MaxSim, CPU).
L'indexation est inchangée (les patches sont déjà calculés pour toutes les pages à
l'ingestion et stockés dans LanceDB). Le modèle ColQwen2 est déjà préchargé au
démarrage (`main.py:126-136`).

**Points à surveiller / régler (P1)** — l'absence de gate lève 3 garde-fous qui
comptaient sur lui :
1. **`protect_colpali_visual_hits`** (`page_reranker_service.py:164-206`) réinjecte
   des pages ColPali-dominantes que le cross-encoder (texte) a écartées, dans
   `COLPALI_PROTECTED_SLOTS=2`. Sur une requête **purement textuelle**, ça peut
   réintroduire des pages visuelles non pertinentes → bruit. **C'est le principal
   risque.** Options : (a) conditionner les slots protégés à un signal visuel dans
   la requête ; (b) réactiver un **juge vision** (voir point 3) ; (c) réduire
   `COLPALI_PROTECTED_SLOTS` et valider au golden.
2. **Seuils anti-bruit** : `COLPALI_MIN_THRESHOLD=0.30` (pré-fusion) et
   `COLPALI_POST_FUSION_MIN_SCORE=0.25` (ne filtre QUE les pages ColPali-only).
   Toujours actif = plus de pages ColPali-only en lice → probablement remonter ces
   deux seuils. À régler au golden.
3. **`VISION_RERANK` est du code mort** : le juge de pertinence vision LLM sur les
   PNG (censé filtrer le bruit ColPali) est forcé `False` en dur
   (`space_search_service.py:661`) alors qu'il est configuré dans le `.env`. C'est
   précisément l'outil qui sait juger une page **visuelle** (que le cross-encoder
   texte ne peut pas voir). **Recommandation** : le rebrancher (ou le supprimer et
   assumer les slots protégés). À trancher au golden — c'est le vrai complément de
   « ColPali toujours actif ».
4. **Code mort** : `COLPALI_GATING_FALLBACK_MIN_HITS` (le filet de rattrapage)
   devient inatteignable quand le gate est off. À nettoyer.

---

## 5. Rappel P2 (hors de ce lot, mais lié au reranker/génération)

- **Bug `is_vision_model`** (`rag_generation_service.py:27-36`) : ne reconnaît pas
  `small` → basculer `MODEL_FAST` sur `mistral-small` **couperait les PNG de
  pages** à la génération. À corriger AVANT tout switch de modèle.
- Bascule `mistral-small` + `reasoning_effort` (appel A/B), client `chat_stream`
  compatible ThinkChunk, `max_tokens` relevé. Repli `mistral-medium-3-5` sur
  l'appel de rédaction. Cf. `plan_impl_reasoning_query_generation_2026-07-17.md`.

---

## 6. Ordre d'exécution & validation

1. **Brancher le golden en CI** (`docker compose exec web pytest`) — sinon rien
   n'est mesurable.
2. **Recalibrer le seuil reranker FR** (§3) — palier bloquant avant prod.
3. Appliquer P0 restant (pattern illustration + prompt images).
4. P1 KAG + fusion + remontée d'erreurs (§2).
5. Régler les seuils ColPali + trancher le juge vision (§4).
6. Mesurer à chaque palier : hors-sujet, refs inventées, taux d'abstention/
   clarification, latence 1er token, rappel sur pages visuelles.
