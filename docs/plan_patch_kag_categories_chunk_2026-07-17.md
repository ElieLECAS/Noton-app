# Plan de patch — KAG robuste, Catégories, Chunk contextuel

_Date : 2026-07-17 · Branche : `fix/retriever`_

## Périmètre

**On garde** : l'extraction texte par LLM vision (mistral-small par page) — jugée fiable, non touchée.
**On ne fait pas** : migration Mistral OCR 4, refonte du retriever/reranker, refonte de la config (hors variables mortes croisées).
**On fait** : patcher l'existant sur (0) nettoyage Docling + code mort, (1) KAG robuste, (2) chunk contextuel / ingestion, (3) catégories.

**Décisions actées** :
- Canonicalisation KAG **100% automatique** (accent-fold + lemmatisation FR + fusion par embedding), **sans gazetteer** à maintenir.
- Catégories : **vocabulaire fermé amélioré** (on garde les axes curés, on supprime le pipeline candidats mort, on corrige la classification).

**Principe transversal** : chaque changement KAG/catégories doit être rejouable sur le corpus existant sans re-extraire la vision (coûteuse). On s'appuie sur les chunks déjà persistés.

---

## Phase 0 — Nettoyage (préalable, ~1 j)

Réduit la surface avant de patcher.

- **Supprimer Docling** : `chunking_service.py` (tout le bloc Docling/tables hiérarchiques mort), les refs dans `library.py`, et les vars fantômes `DOCLING_TEXT_WINDOW_*` (lues via `getattr` sur des champs inexistants). Ne garder que `chunk_pymupdf4llm_page_clean` (fallback vivant). Vérifier `mistral_ocr_service.py` : le retirer des imports Docling mais **le garder** (utile plus tard comme fallback OCR — hors périmètre ici).
- **Supprimer le pipeline candidats de catégories** (mort) : `CategoryCandidate` modèle + table + migration, champ prompt `symptom_candidates`, `_persist_symptom_candidates`, `_slugify_candidate`, l'accumulateur `symptom_candidates_all`. → cohérent avec la décision « vocabulaire fermé ».
- **Supprimer la taxonomie 2 niveaux morte** : colonnes `parent_slug`/`task_group` jamais lues (la mind map utilise `theme_tree_catalog.THEME_FAMILIES` en dur). Retirer `task_group` de `get_active_categories` pour qu'il ne pollue plus la validation de slugs.
- **Test de non-régression** : `docker compose exec web pytest` doit rester vert après suppressions.

---

## Phase 1 — KAG robuste (~3-4 j)

Cible : le graphe passe de « co-occurrence bruitée » à « index d'entités fiable ». 5 correctifs.

### 1.1 — Bug requête-par-mot (le plus urgent)
`kag_retrieval_service.py:38` tokenise la phrase brute (chaque mot ≥3 car., stopwords compris → jusqu'à 8 requêtes SQL trigram/message).

- **Alimenter le matching avec les entités déjà extraites**, pas la phrase. `_run_retrievers` / `_retrieve_one_group_hits` (`space_search_service.py:126,705`) reçoivent `semantic_q` ; leur passer aussi `entity_texts + detected_references` (déjà produits par la compréhension de requête, cf. `lightweight_query_understanding.py:653`).
- **Une seule requête set-based** : remplacer la boucle `for token in tokens` par un `WHERE ke.name_normalized % ANY(:terms)` + JOIN alias (au lieu de la sous-requête corrélée par token), `GREATEST(similarity(...))` agrégé. Garder la branche embedding.
- **Denylist stopwords FR** + fallback : si aucune entité extraite, retomber sur un extracteur de termes discriminants (réutiliser `_BM25_STOPWORDS`), pas la phrase entière.
- **Fixer `pg_trgm.similarity_threshold`** explicitement (aujourd'hui dépendant de l'environnement).

### 1.2 — Résolution d'entités (fusion des doublons)
`knowledge_entity.py:31` — clé unique `(space_id, name_normalized, entity_type)` fragmente une même entité par type.

- **Migration** : passer la clé unique à `(space_id, name_normalized)`. Résoudre `entity_type` par vote (priorité `product > reference > material > ... > other`) lors de l'upsert (`_upsert_entity:862`).
- **Ne plus forcer `type="other"`** pour les extrémités de relations (`kag_extraction_service.py:1046`) : résoudre le type comme pour toute entité.
- **Script de fusion rétroactif** : réunir les doublons existants (même `name_normalized`, types différents), re-pointer `chunkentityrelation` / `entityentityrelation` / `entityalias` vers le survivant. Idempotent, rejouable.

### 1.3 — Canonicalisation automatique (accent-fold + lemmatisation + fusion embedding)
`normalize_entity_name:287` préserve les accents (NFKC) → « Kömmerling » ≠ « Kommerling ».

- **Nouvelle normalisation** : NFKD + suppression d'accents + casefold pour la **clé** `name_normalized` ; garder le nom accentué en `name` (affichage). Un seul helper `normalize_slug/normalize_label` partagé (aussi utilisé côté catégories, cf. 3.4).
- **Lemmatisation FR légère** pour les pluriels techniques (profilé/profilés, gâche/gâches) : table de règles suffixes ou `fr_core_news_sm` si déjà dispo, sinon règles.
- **Fusion par embedding en écriture** : à l'upsert, avant de créer une entité, chercher les entités du même espace à cosine ≥ seuil (embedding déjà calculé en 2ᵉ passe — le déplacer/anticiper) ; si match, fusionner au lieu de créer. Couvre « K76 » ↔ « Kömmerling 76 » sans gazetteer, via proximité sémantique.
- **Note** : sans gazetteer, les codes très courts (« K76 ») dépendent de la qualité de l'embedding d'entité — à mesurer ; si insuffisant, on ré-ouvrira l'option hybride.

### 1.4 — EntityAlias many-to-many
`knowledge_entity.py:74` — unique `(space_id, alias_normalized)` → collisions perdues en silence (`_upsert_entity:910`), alias jamais re-liés après fusion.

- **Migration** : clé unique → `(space_id, alias_normalized, entity_id)` (many-to-many). Un même « 76 » peut pointer plusieurs entités ; la désambiguïsation se fait au scoring.
- **Peupler les alias utiles** : à l'upsert, générer automatiquement les variantes (accent-fold, sans espaces, forme titre) + brancher `feedback_knowledge_service` comme source d'alias correctifs (« K76 = Kömmerling 76 » saisi par un utilisateur devient un alias).
- **Re-lier après fusion** (1.2/1.3) : les alias du doublon suivent le survivant.

### 1.5 — Relations dirigées + exploitation au retrieval
`_upsert_entity_relation:970` — `sorted((a,b))` détruit la direction ; type non validé ; `_neighbor_entities:164` ignore type/confiance (`LIMIT 100` global).

- **Vocabulaire fermé de relations** : allow-list dans `_normalize_relation_type:306` + map de synonymes (« compatible »/« est_compatible » → `compatible_avec`). Stocker la direction `(source, target)` pour les types asymétriques (`remplace`, `symptome_cause`, `etape_precede`), garder le tri pour les symétriques (`co_occurs`).
- **Traversée pondérée** : `_neighbor_entities` par-seed, ordonné par `weight × confidence × prior(type)` selon l'intention (SAV → suivre `symptome_cause`/`cause_resolution`).
- **Intégrité** : GC d'orphelins basé sur l'existence réelle de `chunkentityrelation` (abandonner `mention_count` comme clé de GC) ; supprimer les arêtes dont une extrémité devient orpheline.

---

## Phase 2 — Chunk contextuel & ingestion (~2-3 j)

Vision conservée ; on fiabilise ce qui l'entoure.

- **Garantie de couverture** (`vision_page_extraction_service.py:212/224`) : le cap dur à 12 chunks/page tronque en silence. Remplacer par un **split** (jamais de perte) + log d'un ratio de couverture (caractères transcrits / caractères pymupdf par page) ; alerte sous seuil.
- **Gate de fidélité numérique** (le plus haut ROI domaine) : après enrichissement (`contextual_enrichment_service.py:53` réécrit le contenu), vérifier que chaque token numérique/référence de la sortie LLM existe dans la couche texte pymupdf de la page source ; sinon rejet/marquage. Empêche l'invention de cotes (Ug, Rw, entraxe).
- **Dédup enrichissement** : fenêtres chevauchantes + L1/L2 produisent des triplons. Clé de dédup `(thème, page source)` ou hash de contenu dans `_persist_enrichment_chunks:657` ; envisager `overlap=0` pour l'enrichissement.
- **Symétrie BM25/dense** : le préfixe contextuel (titre/section/catégories/entités) n'est injecté que dans l'embedding (`document_indexing_service.py:548`). Persister un `search_text` (ou 2ᵉ tsvector généré) qui le reflète, pour que le lexical voie les mêmes termes.
- **IDs de chunks stables** : `node_id = hash(doc_id, page, contenu normalisé)` au lieu de `uuid4()` → réindexation idempotente, embeddings non re-payés sur contenu inchangé. Permet aussi de rejouer KAG/catégories sans toucher la vision.
- **Nettoyage schéma** : supprimer la colonne JSONB dupliquée `metadata_` (garder `metadata_json`), migrer les rares lecteurs.

---

## Phase 3 — Catégories (vocabulaire fermé amélioré, ~2 j)

- **Reclassification versionnée** : `taxonomy_version` est estampillé (`kag_extraction_service.py:1164`) mais jamais lu. À chaque modif de catégorie/description/axe → bump de version + tâche de reclassification des chunks dont le stamp diffère. **Rejoue la classification sur les chunks existants** (LLM texte, pas vision) → pas de re-ingestion.
- **Confiance calibrée** : supprimer l'injection des tags `doc_type`/`lifecycle` à confiance 1.0 sur 100% des chunks (`kag_extraction_service.py:1149`) qui sature le gate. Aligner seuil prompt (0.5) et code (0.55) ; binning primary/secondary plutôt qu'un float auto-déclaré non calibré.
- **`doc_type` au niveau document** : réconcilier les tags `doc_type` par page en une valeur document (vote), au lieu de les laisser diverger page à page.
- **Normalisation unifiée** : réutiliser le helper de 1.3 (NFKD + casefold + accent-strip) pour les slugs de catégories (aujourd'hui juste `.lower().replace(" ","_")`, safe seulement parce que le vocab est ASCII).
- **Intégrité** : FK `ChunkCategoryRelation.document_id` → `document` avec `ON DELETE CASCADE` (aujourd'hui int nu, orphelins invisibles) ; retirer `is_primary` (écrit jamais lu) **ou** l'exploiter dans le boost. Choisir une seule autorité de schéma (Alembic, pas `create_all`).
- **N+1** : `apply_soft_boosts_to_passages:262` charge `Document` 2-3× par passage → bulk-load unique.

---

## Ordre d'exécution recommandé

1. **Phase 0** (nettoyage) — débloque le reste, faible risque.
2. **Phase 1.1** (bug requête-par-mot) — gain immédiat, isolé, peu risqué.
3. **Phase 1.2 → 1.4** (résolution + canonicalisation + alias) — le cœur du KAG robuste ; migrations + script de fusion rétroactif.
4. **Phase 1.5** (relations dirigées).
5. **Phase 3** (catégories) — dépend du helper de normalisation de 1.3.
6. **Phase 2** (chunk/ingestion) — indépendant, peut se faire en parallèle une fois la vision confirmée intacte.

**Effort total estimé** : ~10-12 jours. Filet de sécurité minimal recommandé avant 1.2/1.3 (migrations) : un petit jeu de tests sur la résolution d'entités (K76↔Kömmerling 76, accents, pluriels) pour valider la canonicalisation automatique.
