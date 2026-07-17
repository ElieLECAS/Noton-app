# Audit complet avant refonte — 17 juillet 2026

> Branche auditée : `fix/retriever`. Six audits de code exhaustifs (ingestion, chunking,
> KAG/GraphRAG, catégories, retriever, configuration). Chaque constat est ancré fichier:ligne.
> Version web : https://claude.ai/code/artifact/3684a9d9-0292-4116-98dd-b61c24dd48a9
> Plan d'exécution des phases 0 et 1 : `docs/plan_phases_0_1_2026-07-17.md`

**Bilan : 13 constats critiques, 26 majeurs, ~4 500 lignes de code mort, ~30 variables d'env mortes.**

## Synthèse — trois problèmes structurels

1. **Les échecs sont silencieux et le réindexage détruit avant de reconstruire.** Une page en
   échec devient `[]` et le document finit `completed`. Les pages scannées perdent tout leur
   texte (le fallback OCR n'est jamais branché). Le réindexage supprime + commite l'index
   existant AVANT la reconstruction : tout crash laisse un document vide, irrécupérable.
2. **2 à 3 générations de pipelines coexistent.** ~1 900 lignes de chunking mort, un pipeline
   « multimodal v4 » débranché mais encore invocable via Celery, une famille de retrievers
   morts, le service OCR orphelin, le pipeline de candidats de catégories jamais persisté.
3. **Rien n'est mesuré.** `retriever_evaluator.py` (précision/recall/MRR par étage) est bien
   conçu, le jeu golden `tests/fixtures/golden/` existe — mais aucun test ne les relie.
   L'évaluateur n'est accessible que par un endpoint admin manuel.

### Vérification des intuitions initiales

| Intuition | Verdict | Réalité |
|---|---|---|
| « Le lexical requête chaque mot, stopwords compris » | Confirmé — côté **KAG** | `kag_retrieval_service.py:38-72` : jusqu'à 8 requêtes SQL trigram/message, sans filtre stopwords. Le BM25 fait UNE requête (Postgres `'french'` filtre les stopwords) mais reçoit la phrase entière en AND → rate presque toujours → fallback OR systématique. La recherche UI (`lexical_search_service.py:53-75`) garde bien « le/et/ou » en ILIKE. |
| « EntityAlias est cassé » | Confirmé | Alias unique par espace (collisions perdues en silence, `kag_extraction_service.py:910-916`), jamais peuplé pour les cas utiles, jamais re-lié après fusion. « K76 » ne sera jamais lié à « Kömmerling 76 ». |
| « Catégories imparfaites » | Confirmé | 3 systèmes parallèles dont 1 mort ; taxonomie 2 niveaux fictive ; aucune reclassification ; `is_primary` write-only. |
| « Chunks IA sur 3 pages » | Imprécis | Le chunking primaire est PAGE PAR PAGE (vision mistral-small). La fenêtre 3 pages/overlap 1 ne concerne que KAG + enrichissement L2, qui AJOUTENT des chunks (en double, sans dédup). |
| « Énormément de vars d'env à corriger » | Confirmé, structurel | ~225 vars, ~30 mortes/fantômes, 3 sources de vérité divergentes, AUCUNE validation au démarrage. |

---

## 1. Ingestion documentaire

### Pipeline réel

```
upload (library.py:1181, fichier entier en RAM, aucune limite de taille)
 └─ dispatch thread|celery → _process_document_for_id (document_service_new.py:1278)
     └─ process_document_indexing (document_indexing_service.py:55)
         1. SUPPRESSION chunks + LanceDB, commit immédiat            ⚠ destructif
         2. Vision : 1 appel mistral-small PAR PAGE (PNG 300dpi, JSON, concurrency 3)
            fallback pymupdf4llm si 2 échecs — AUCUN OCR pour les scans
            + merge inter-pages HEURISTIQUE (ponctuation, minuscule)
         3. KAG (fenêtre 3 pages, overlap 1)                — non bloquant
         4. Enrichissement contextuel L2 (fenêtre 3 pages)  — non bloquant
         5. Embeddings mistral-embed 1024d (L1+L2)
         6. ColPali ColQwen2 (PDF ENTIER rasterisé en RAM) → LanceDB
```

Il n'existe **aucune passe LLM dédiée** à la liaison du texte coupé entre deux pages :
flags `continues_on_next_page` devinés par un modèle page-locale + heuristiques Python
fragiles (`vision_page_extraction_service.py:380-425` : « pas de ponctuation finale »
fusionne les fins de tableaux à tort ; une page en échec casse l'adjacence).

### Critiques
- **C1. Réindexage destructif non atomique** — `document_indexing_service.py:94/275` :
  delete + commit avant rebuild. Avec `task_acks_late=True` (`celery_app.py:65`) et aucun
  `time_limit`, un OOM → redélivrance → boucle de destructions.
- **C2. Pages scannées : texte perdu, OCR débranché** — fallback = pymupdf
  (`vision_page_extraction_service.py:282`), vide sur un scan. `mistral_ocr_service.py`
  entièrement orphelin (seul appelant : `process_document_file`, DEPRECATED sans appelant).
- **C3. Échec de page silencieux → succès affiché** — `document_indexing_service.py:379-385` :
  exception par page → `[]`, statut final `completed`.

### Majeurs
- ColPali rasterise tout le PDF d'un coup (`colpali_service.py:69`) → OOM probable.
- Index LanceDB IVF reconstruit à CHAQUE insertion, erreurs avalées (`lancedb_service.py:73-121`) ;
  delete-then-add non atomique.
- Annulation inopérante en Celery : `_cancelled_document_ids` est un set en mémoire du
  process web (`document_service_new.py:97`).
- Second pipeline divergent encore invocable : `multimodal_reindex_library_document_task`
  produit des chunks `multimodal_page_v4` (texte brut, sans vision/KAG) incompatibles.
- Aucun timeout document : pire cas ~20 min/page (120s × 2 tentatives × 5 retries).
- Caches PNG jamais invalidés (clé `md5(chemin)` sur chemin déterministe,
  `multimodal_page_service.py:803`) ; `delete_document` ne purge ni `page_cache/` ni
  `illustration_cache/`.

### Mineurs
Triple rasterisation (300/150/200 dpi) ; barre de progression qui recule pendant ColPali ;
`ocr_preprocessing.py` (300 lignes) inutilisé ; embeddings vides silencieusement ignorés ;
pas de timeout LibreOffice (`file_conversion.py:59`) ; pas de limite de taille d'upload ;
artefacts de debug dans le package (`app/test_pymupdf.py`, `app/test_reranker.py`,
`app/debug_db.txt`).

---

## 2. Chunking & enrichissement contextuel

Le chunker de production est le LLM vision (1 appel/page, JSON Pydantic, ≤ 480 tok/chunk,
cap 12 chunks/page). L'enrichissement L2 n'est pas du contextual-retrieval Anthropic :
c'est une **réécriture domaine** (synthèse / étapes / diagnostic) pilotée par 15 playbooks,
sur fenêtres de 3 pages avec overlap 1 (stride 2).

### Critiques
- **C1. Aucune garantie de couverture** — cap dur 12 chunks/page, excédent tronqué sans log
  (`vision_page_extraction_service.py:212/224`). Rien ne compare la transcription au texte
  source. Des lignes de specs disparaissent sans trace.
- **C2. Non idempotent, non déterministe** — `node_id=uuid4()` partout + frontières LLM
  variables : chaque réindexation produit des IDs et frontières différents. Réindexation
  incrémentale impossible, tous les embeddings repayés.
- **C3. Valeurs numériques réécrites sans vérification** — l'enrichissement « RÉÉCRIT et
  EXPLICITE » (`contextual_enrichment_service.py:53`) : une cote fausse (Ug, Rw, entraxe)
  peut naître ici et devenir un fait retrievable. Aucun contrôle croisé avec le PDF.

### Majeurs
- ~1 900 lignes mortes : `chunk_service.py` entier + blocs Docling/tables/hiérarchie de
  `chunking_service.py`. Seule fonction vivante du fichier : `chunk_pymupdf4llm_page_clean:2318`
  (fallback). **Les tableaux ne sont pas gérés en prod** — que du texte libre LLM.
- Doublons d'enrichissement : fenêtres chevauchantes sans dédup + L1 verbatim + L2 paraphrase
  tous embarqués → le même fait remonte en 3 quasi-doublons.
- Asymétrie BM25/dense : préfixe contextuel (titre/section/catégories/entités) injecté
  seulement dans le texte d'embedding (`document_indexing_service.py:548`), le tsvector ne
  voit que le contenu brut.
- Colonnes JSONB dupliquées `metadata_json` + `metadata_` (`document_chunk.py:26-27`).
- Cap 600 tokens de l'enrichissement : prompt uniquement, jamais vérifié en code.

### Note importante — les chunks L2 dans la génération
Les chunks d'enrichissement sont `is_leaf=True`, `hierarchy_level=2`
(`contextual_enrichment_service.py:693,712`). Or le packer CAG charge TOUS les chunks
`is_leaf=True` d'un document sélectionné (`context_packer_service.py:100-103`).
**Conclusion : les paraphrases L2 sont packées dans le prompt de génération, mélangées aux
chunks verbatim L1 — le même contenu y figure 2 à 3 fois**, gonflant le budget de tokens
d'environ le volume de l'enrichissement.

---

## 3. KAG / GraphRAG — entités, relations, EntityAlias

**Verdict : un graphe de co-occurrence déguisé en KAG.** Extraction correcte
(mistral-small vision, fenêtres 3 pages, 11 types d'entités validés) ; résolution,
alias et exploitation des relations cassées.

### Critiques
- **C1. Dédup fragmentée par type** — clé `(space_id, name_normalized, entity_type)`
  (`knowledge_entity.py:31-39`) : « Kömmerling » typé `organization` puis `product` = 2
  entités. Chaque extrémité de relation sans correspondance exacte crée un doublon `other`
  (`kag_extraction_service.py:1046`). `recreate_kag_tables` a régressé depuis la bonne clé
  2 colonnes de `add_kag_space_constraints.py:20-25`.
- **C2. Pas d'accent-folding / abréviations / pluriels** — `normalize_entity_name:287-296`
  préserve les accents (NFKC) ; dictionnaire de 11 entrées mono-token matchées sur la chaîne
  ENTIÈRE. « K76 » / « Kömmerling 76 » / « Kommerling 76 » ne fusionneront jamais.
- **C3. Matching de requête mot-par-mot avec stopwords** — `kag_retrieval_service.py:38` :
  tokens ≥ 3 chars sans denylist (« pour », « avec », « quelle », « dimension »…), une
  requête trigram + sous-requête alias PAR token (≤ 8, ×G en multi-groupe).

### Majeurs
- **EntityAlias write-only** : alias créés seulement si le LLM les émet ; index unique
  `(space_id, alias_normalized)` → collisions silencieusement abandonnées
  (`kag_extraction_service.py:910-916`) ; jamais re-liés après fusion ; pas de type.
- **Direction des relations détruite** : `sorted((a,b))` avant stockage (`:970`) — fatal
  pour `remplace`, `symptome_cause`, `etape_precede`. Vocabulaire de relations non validé
  (`_normalize_relation_type:306-309` accepte tout) → « compatible » / « compatible_avec » /
  « est_compatible » = 3 arêtes distinctes.
- **Relations = poids mort au retrieval** : expansion 1-hop ignore type/direction/confiance,
  `LIMIT 100` global, facteur plat 0.65 (`kag_retrieval_service.py:155-184,216`). Le
  `kag_score` calculé est jeté par la fusion RRF (rang seul, `page_retrieval_service.py:1307`).
- `mention_count` incrémenté/décrémenté avec deux dénominateurs différents → GC d'orphelins
  faux (`:882` vs `:1436`).
- Relations orphelines après suppression (`ON DELETE SET NULL` sur `source_chunk_id`).

### Mineurs
Provenance grossière (relations attribuées au 1er chunk de page, `:1013`) ; seuil pg_trgm
jamais fixé ; document dans N espaces = N copies du graphe ; `feedback_knowledge_service`
non branché sur le KAG (canal idéal pour le gazetteer, inutilisé).

---

## 4. Catégories & classification

Trois systèmes : **(A)** catégories de chunks à vocabulaire fermé (axes task/doc_type/
lifecycle/symptom), assignées par le LLM KAG ; **(B)** candidats LLM — **mort** ;
**(C)** classification documentaire manuelle (produits/matériaux/gammes) — le signal le
plus fiable.

- **Critique. Pipeline « LLM propose, humain valide » inerte** : `_persist_symptom_candidates`
  jamais appelé, `candidates_touched=0` codé en dur (`kag_extraction_service.py:1602-1604`),
  aucun endpoint de promotion. Les nouveaux symptômes disparaissent en silence.
- **Majeur. Taxonomie 2 niveaux fictive** : `parent_slug`/`task_group` lus nulle part ;
  la mind map utilise `THEME_FAMILIES` codé en dur (`theme_tree_catalog.py:26-63`).
- **Majeur. Aucune reclassification** : `taxonomy_version` estampillé
  (`kag_extraction_service.py:1164-1165`) mais lu nulle part. Taxonomie write-once.
- **Majeur. Gate de confiance neutralisé pour 2 axes sur 4** : doc_type/lifecycle injectés
  à confiance 1.0 sur 100 % des chunks, en contournant seuil et top-K (`:1149-1152`).
  Le boost de retrieval (max de confiance par page) est saturé — et c'est le signal de
  classement principal quand le reranker est off (cas du déploiement actuel).
- Mineurs : N+1 dans `apply_soft_boosts_to_passages` (`retrieval_boost_service.py:262-282`) ;
  `is_primary` write-only ; `ChunkCategoryRelation.document_id` sans FK (orphelins invisibles) ;
  double gestion de schéma Alembic + `create_all` ; seuil prompt 0.5 ≠ code 0.55 ;
  confiance par défaut 0.7 déjà au-dessus du seuil.

---

## 5. Retriever

### Flux réel (2 appels LLM/message)

```
message → compréhension fusionnée (1 appel : route+signaux+condense)
        → 4 retrievers PARALLÈLES : ColPali (gaté) · pgvector · BM25 · KAG
        → fusion RRF k=60 (saine, basée rang)
        → boosts catégorie/ancre sur rrf_score      ⚠ inertes si reranker actif
        → rerank cross-encoder ms-marco-MiniLM      ⚠ modèle ANGLAIS
        → boosts additifs post-rerank (+0.8·conf peut dominer le reranker)
        → packing CAG (docs entiers, ancres forcées — fonctionne)
        → génération (1 appel streaming)
```

Bonnes nouvelles : RRF correct, gating ColPali et ancrage conversationnel bien câblés,
pas de boucle de critique LLM cachée. Coût dominant = CPU (encodage ColPali + MaxSim exact
qui charge tous les patchs en RAM sous 150k, cross-encoder).

### Critiques
- **C1. Reranker anglais sur du français** — `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (`config.py:257`). Signal de classement dominant → plafond de qualité global.
  Remplacer par `BAAI/bge-reranker-v2-m3` ou `mmarco-mMiniLMv2-L12`.
- **C2. KAG mot-par-mot** (cf. section 3, C3).
- **C3. Aucun harnais d'éval automatisé** — golden non chargé par les tests, évaluateur
  admin-HTTP seulement (`admin.py:783`).

### Majeurs
- BM25 reçoit la phrase naturelle entière en AND (`lightweight_query_understanding.py:653-656`
  + `websearch_to_tsquery`) → fallback OR quasi systématique. Alimenter en mots-clés +
  références OR avec boost de phrase (`<->`).
- Pile de boosts à 3 échelles incompatibles : multiplicatif sur rrf (inerte avec reranker),
  additif sur sigmoïde, `+0.8·confiance` qui écrase le reranker
  (`space_search_service.py:1292`). Nombres magiques : -8.0, +1.5, 0.65, 0.15, 0.75…
- Recherche UI : tokens ≥ 2 chars sans stopwords → `%le%`/`%et%` matchent tout
  (`lexical_search_service.py:53-75`).
- Famille de retrievers morts `PageRetrievalHit` (`page_retrieval_service.py:605,686,753,
  1608,1660,1936`) + échafaudage RRF/ColPali legacy dans `space_search_service.py:175,323,1309`.

### Mineurs
Chemin legacy de compréhension = 5+N appels LLM séquentiels si flag fusionné off ;
restes vision-rerank dans l'évaluateur ; étage éval « colpali_only » non isolé ;
traversée KAG `LIMIT 100` global.

---

## 6. Configuration

~225 noms de variables, ~198 champs `Settings`.

- **Critique. Aucune validation au démarrage** : `SECRET_KEY`/`DATABASE_URL` typées `str`,
  défaut `None`, pas de `validate_default` → démarrage OK, explosion différée.
  `extra="ignore"` → une var mal orthographiée est ignorée en silence. Aucun `SecretStr`.
- **Critique. Couche `int(os.getenv(...))` morte ET dangereuse** : en prod pydantic re-parse
  lui-même (les ~129 expressions ne s'exécutent jamais) ; hors Docker, une valeur vide crashe
  l'import entier avant tout logging.
- **Majeur. Lectures hors config.py** (8 modules : `embedding_config`, `logging_config`,
  `tracing`, `chat.py:186-206`, `space_search_service.py:147-149`,
  `library_document_logging`…). Pydantic n'injecte pas `.env` dans `os.environ` →
  **ces lectures ignorent le .env en dev local**.
- **Majeur. Variables fantômes** lues via `getattr(settings, ...)` sur des champs
  inexistants : `RERANKER_PROVIDER` (le provider Mistral de rerank est inatteignable),
  `DOCLING_TEXT_WINDOW_*`.
- **Majeur. 3 sources de vérité divergentes** malgré le commentaire « tenus ALIGNÉS » :
  `SPACE_CHAT_TEMPERATURE` 0.3/0.55/0.0 ; `RERANK_CHAR_CAP` 1700/8000/8000 ;
  `PAGE_EXTRACTION_DPI` 300/200/300 ; `MIN_DYNAMIC_K` 0/2/0 ; `TASK_BACKEND_MODE` 3 valeurs.
- **~30 vars mortes** : blocs MMR, RRF dynamique, `DOCLING_OCR_*`, `OCR_*` (Tesseract),
  `USE_MULTIMODAL_RETRIEVAL`, `EMBEDDING_DEVICE`, `RERANK_GROUP_CHAR_CAP`…
- Cible : `pydantic-settings` imbriqué par sous-système, `extra="forbid"`,
  `validate_default=True`, secrets requis en `SecretStr`, suppression de la couche
  `os.getenv`, poids/seuils de retrieval déplacés vers une table admin en DB,
  compose/.env.example régénérés depuis le modèle. Renommer `LANCED_DB_DIR` → `LANCEDB_DIR`.

---

## Choix Mistral : Small vs Large vs OCR 4

**Réponse : OCR 4 pour l'extraction, Small pour l'arbitrage, Large presque jamais.**

Mistral OCR 4 (sorti le 23/06/2026, 4 $/1000 pages, 2 $ en batch, auto-hébergeable) :

| Besoin actuel | Apport OCR 4 |
|---|---|
| Garantie de couverture | Extraction exhaustive + confiance PAR MOT (le doute devient mesurable) |
| Tableaux structurés (absents en prod) | Classification de blocs (titre/tableau/équation) + markdown → chunks ligne-par-ligne déterministes |
| Liaison inter-pages | Bounding boxes + types de blocs → continuation majoritairement géométrique/déterministe |
| Pages scannées (perdues aujourd'hui) | OCR natif 170 langues, préféré à 72 % par annotateurs humains |
| Citations | Bounding boxes → surlignage de la source exacte |

Répartition cible : **OCR 4** = extraction texte/structure/tableaux ; **Small 3.2** =
arbitrage des jonctions inter-pages ambiguës (~5 % des frontières), extraction KAG,
classification, enrichissement ; **Large/Medium** = uniquement là où le harnais d'éval
prouve un gain ; **vision (Small/Pixtral)** = description des pages purement visuelles
(schémas CAO). ColPali n'est PAS remplacé (extraction ≠ retrieval visuel) ; le gating
par intention est conservé.

---

## Plan de refonte en 5 phases

- **Phase 0 — Filet de sécurité (~1 sem)** : golden en pytest/CI avec seuils recall/MRR ;
  réindexage atomique + `time_limit` Celery + statut `partial` ; fallback OCR scans ;
  purge du code mort (~4 500 lignes, 30 vars, 2 pipelines legacy).
- **Phase 1 — Ingestion OCR 4 (~2-3 sem)** : OCR 4 extracteur primaire, chunking
  déterministe par structure (tableau → 1 chunk/ligne + résumé), liaison inter-pages
  géométrique + arbitrage Small, invariant de couverture + gate de fidélité numérique,
  IDs stables par hash → réindexation incrémentale, enrichissement additif dédupliqué,
  préfixe contextuel visible du BM25.
- **Phase 2 — KAG (~2 sem)** : canonicalisation (accent-folding, lemmatisation FR,
  gazetteer marques/gammes/codes), dédup sans type, alias many-to-many (sources : gazetteer,
  LLM, feedback), relations dirigées à vocabulaire fermé, traversée pondérée par
  type/confiance selon l'intention, matching de requête en 1 requête set-based.
- **Phase 3 — Retriever (~1-2 sem)** : reranker multilingue + recalibration ; BM25 par
  termes ; un seul étage de boost post-rerank à échelle unique ; stopwords UI ;
  suppression des retrievers morts.
- **Phase 4 — Catégories & config (~1-2 sem)** : reclassification versionnée, candidats
  branchés ou supprimés, fin de la confiance 1.0 systémique ; pydantic-settings
  `extra="forbid"`, poids retrieval en DB, compose/.env régénérés.
