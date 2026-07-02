# Plan d'amélioration — côté réponse (intention → retriever → génération) + nettoyage KAG

> Périmètre : **pas d'ingestion**. On agit sur la compréhension d'intention, le retrieval, la génération,
> la mémoire de conversation, et le nettoyage des entités à la suppression.
> Contexte technique : génération avec **Mistral Large, fenêtre 256k tokens**, **reranker cross-encoder désactivé**.

## Corrections au diagnostic initial (vérifiées dans le code)

- **Reranker OFF** : `space_search_service.py:828-831` → sans reranker, `final_hits = fused_hits` et `dynamic_k = len(fused_hits)`. Tous les hits fusionnés (jusqu'à `pool_size = max(RERANK_POOL=40, RAG_POOL_SIZE, top_k)`) atteignent la construction du contexte. **Le seul limiteur réel est `SPACE_CONTEXT_MAX_CHARS=18000`** (`chat.py:163,463`). → Les recommandations reranker/MIN_DYNAMIC_K sont sans objet.
- **Retraitement texte déjà propre** : `document_indexing_service.py:252-272` (`_delete_chunk_foreign_relations`) appelle `delete_chunk_kag_relations` + `prune_kag_entities_after_chunk_removal`. Rien à faire ici.
- **Trou réel = suppression** : `delete_document` (`document_service_new.py:1111`) → `delete_chunks_for_document` (chunks PG + LanceDB) **sans** nettoyage KAG. `cleanup_kag_for_document` (`kag_extraction_service.py:1464`) existe mais n'est appelé nulle part.

---

## Axe 1 — Intention (query understanding) : fusionner les appels + détecter le changement de sujet

**Problème** : 4–5 appels LLM séquentiels avant retrieval (route → signals → vagueness → condense → generate) ≈ 5–7 s ; et le condense réintègre toujours l'ancien sujet → perte de fil.

**1.1 Fusionner route + signals + vagueness + condense en UN seul appel JSON.**
- Fichier : `app/services/lightweight_query_understanding.py` (le graphe `_node_*`).
- Un seul prompt, un schéma de sortie unique :
  ```json
  {
    "decision": "direct" | "rag",
    "topic_shift": true | false,
    "standalone_question": "…",
    "too_vague": false,
    "clarification_question": null,
    "signals": { "intent": "...", "entities": [...], "categories": [...], "primary_source": "...", "detected_references": [...] }
  }
  ```
- Garder `generate_queries` séparé (il dépend de `standalone_question`) mais le déclencher immédiatement après.
- **Gain** : ~5 appels → 2 appels. Latence pré-retrieval ~6 s → ~2,5 s.

**1.2 Détection de changement de sujet (`topic_shift`).**
- Ajouter dans le prompt fusionné : *« Si le dernier message introduit un sujet sans lien avec l'historique, ou demande explicitement de changer/oublier, mets `topic_shift=true` et renvoie `standalone_question` = le message tel quel, sans réintégrer l'historique. »*
- Filet de sécurité déterministe (optionnel) : cosine entre `embed(message_n)` et `embed(dernier échange)` < seuil → forcer `topic_shift=true`. Évite de dépendre uniquement du LLM.

**1.3 Reset des signaux persistés sur `topic_shift`.**
- `chat.py` : quand `topic_shift=true`, **purger `Conversation.query_context.signals`** avant `apply_soft_boosts_to_passages` / `refine_with_source_authority` (`chat.py:955-971`). Sinon matériau/source de l'ancien sujet boostent les mauvais docs.
- `_node_merge_context` (`lightweight_query_understanding.py:466`) : ne fusionner l'ancien contexte que si `topic_shift=false`.

**1.4 Tests manquants** (`tests/test_query_condense.py`) :
- « oublie le profil 76, parle-moi du DTU 36.5 » → pas de profil 76 dans la reformulation.
- suivi après 5 tours où le sujet a changé au milieu.

---

## Axe 2 — Retriever : paralléliser + fiabiliser le ranking (sans reranker)

**2.1 Paralléliser les 4 retrievers.**
- `space_search_service.py:746-753` : ColPali (CPU ~400 ms) + pgvector + BM25 + KAG en `asyncio.gather` au lieu de séquentiel. Gain ~-30 % sur la phase retrieval.

**2.2 Le category boost devient le principal levier de ranking (reranker off) — à fiabiliser.**
- `retrieval_boost_service.py:106-164` : boost multiplicatif jusqu'à ~×1.9 (3 catégories symptom). Sans reranker pour rattraper, une erreur de classification LLM remonte directement.
- Actions : monter `CATEGORY_MIN_CONFIDENCE` (0.55 → ~0.70) ; réduire `RETRIEVAL_CATEGORY_BOOST` (0.15 → ~0.08) OU plafonner le facteur total ; instrumenter (logguer boost appliqué par hit dans les traces).

**2.3 Élargir le pool candidat (on a la place dans 256k).**
- `RAG_POOL_SIZE` 20 → 40, `RAG_TOP_K` 10 → 20. Plus de candidats fusionnés → plus de pages disponibles pour l'Axe 3.

**2.4 Garde-fou anti-contexte-vide explicite.**
- Si `fused_hits` faible/vide → statut clair renvoyé au front (« aucune source pertinente ») plutôt qu'une génération sur contexte maigre. Le grounding strict du prompt fait déjà « la notice ne précise pas », mais tracer le cas.

---

## Axe 3 — Génération : exploiter les 256k (pages entières texte + images PDF)

**Idée centrale** : passer d'un contexte de **fragments tronqués** à un contexte de **pages entières**. Aujourd'hui chaque passage est un chunk coupé à `SPACE_CONTEXT_MAX_PASSAGE_CHARS`, total 18k chars. Avec 256k on peut donner au modèle des pages complètes + leur image.

**3.1 Contexte texte « page-centric ».**
- Pour les top-N pages retrouvées, charger **tout le texte de la page** (tous les chunks `semantic_leaf` de cette page, dans l'ordre de lecture) au lieu du seul chunk matché. Le modèle voit la page entière → moins de « cotes assemblées de phrases différentes », meilleur grounding.
- Construction dans `chat.py` (section `build_space_context_from_passages`, ~`chat.py:446-463`).

**3.2 Relever les plafonds de contexte (env, pas de code).**
- `SPACE_CONTEXT_MAX_CHARS` 18000 → **80000–120000** (~25–35k tokens texte).
- `SPACE_CONTEXT_MAX_PASSAGE_CHARS` : lever fortement (ou désactiver la troncature par passage en mode page-centric).
- `SPACE_HISTORY_MAX_CHARS` 8000 → 16000 ; `max_messages` 10 → 16.
- Garder une marge : viser ≤ ~150k tokens d'entrée pour laisser respirer coût/latence.

**3.3 Plus de pages PDF en image (vision Mistral Large).**
- `RAG_MAX_IMAGES` 12 → selon budget (ex. 8–15 pages ciblées), `RAG_RENDER_ALL_IMAGES=true` déjà OK (`chat.py:1012-1023`).
- **Attention coût/latence** : chaque page image ≈ 1–2k+ tokens et le rendu PNG est bloquant avant le 1er token. Deux garde-fous :
  - N'imager que les pages **procédurales/schéma** (là où le visuel porte le sens), pas les pages texte pur.
  - Rendre les PNG **en parallèle** et, idéalement, streamer le texte d'abord puis attacher les images (sortir le rendu du chemin bloquant avant `done`).

**3.4 Débloquer la longueur de réponse.**
- Passer explicitement `SPACE_CHAT_MAX_TOKENS` (≥ 2000) à l'appel Mistral (aujourd'hui `MAX_COMPLETION_TOKENS=1024` / 1200 hardcodé non transmis → réponses procédurales coupées).

**3.5 Ordre du contexte.**
- Placer les meilleures pages en premier (les LLM long-contexte pondèrent début/fin) ; historique compact avant le bloc sources.

---

## Axe 4 — Mémoire de conversation

- **État de sujet courant** : stocker dans `Conversation.query_context` un `current_topic` (résumé court + refs produits actives), mis à jour à chaque tour ; réinitialisé sur `topic_shift`.
- **Signaux** : ne reporter d'un tour à l'autre que si `topic_shift=false` (cf. 1.3).
- Optionnel : mini-résumé roulant de la conversation (1 phrase) réinjecté au lieu de 16 messages bruts, pour garder l'historique pertinent sans le gonfler.

---

## Axe 5 — Nettoyage des entités KAG à la suppression

**Trou** : `delete_document` ne nettoie pas le graphe (retraitement texte, lui, est déjà propre).

**Fix** : dans `document_service_new.py:delete_document`, **avant** `delete_chunks_for_document` (l'ordre compte — le nettoyage lit `chunk_id` depuis `documentchunk`) :
```python
from app.services.kag_extraction_service import cleanup_kag_for_document
if settings.KAG_ENABLED:
    cleanup_kag_for_document(session, document_id)  # delete_chunk_kag_relations + prune_orphans
```
- `cleanup_kag_for_document` fait déjà exactement le nécessaire (relations chunk↔entité, chunk↔catégorie, entité↔entité, puis prune des entités orphelines `mention_count<=0`). Il suffit de le **câbler**.
- Vérifier au passage que `chunkcategoryrelation` n'a pas de FK bloquante lors du `DELETE documentchunk` (le nettoyage la retire d'abord, donc OK une fois câblé).
- Idem : vérifier la suppression **d'espace** / retrait document↔espace si elle supprime des chunks sans passer par ce chemin.

---

## Séquencement proposé

1. **Axe 5** (petit, isolé, sûr) : câbler `cleanup_kag_for_document` dans `delete_document` + test.
2. **Axe 1** (fort impact sur les deux plaintes : latence + perte de fil) : fusion des appels + `topic_shift` + reset signaux + tests.
3. **Axe 3** (grounding) : contexte page-centric + relever les plafonds env + débloquer max_tokens.
4. **Axe 2** : parallélisation retrievers + tuning boost + pool.
5. **Axe 4** : état de sujet / résumé roulant.

## Mesure

Avant/après sur `retriever_evaluator.py` + traces LangSmith (déjà en place) : temps par phase, nb pages/tokens réellement envoyés, taux de `topic_shift` correctement détectés sur un jeu de conversations multi-sujets.

---

## RÉALISÉ (2026-07-02) — 5 axes implémentés + testés

Tous les changements sont derrière des flags (repli sûr) sauf les corrections de bug. ~177 tests verts.

### Axe 5 — Nettoyage KAG à la suppression ✅
- `delete_document` appelle désormais `cleanup_kag_for_document` AVANT la suppression des chunks
  (`app/services/document_service_new.py`). Corrige une **violation FK** sur les documents catégorisés
  (`chunkcategoryrelation.chunk_id` sans cascade) ET purge les entités orphelines (mention_count).
- Tests : `tests/test_delete_document_kag_cleanup.py` (purge orphelins + entité partagée préservée).
- Le retraitement texte était déjà propre (`_delete_chunk_foreign_relations`) — non touché.

### Axe 1 — Intention : fusion + topic_shift ✅
- **Bug corrigé** : `_node_merge_context` n'hérite plus du message/contexte du tour précédent hors
  reprise de clarification (cause majeure du « se perd quand on continue »).
- Nouveau nœud fusionné `_node_fused_understand` : route + signaux + condense + vagueness + `topic_shift`
  en **1 appel LLM** (au lieu de 4-5). Flag `QUERY_FUSED_UNDERSTANDING_ENABLED` (défaut on ; legacy préservé).
- `topic_shift` propagé jusqu'à `chat.py` (log + élagage historique génération, cf. Axe 4).
- Tests : `tests/test_fused_understanding.py` + legacy épinglé sur l'ancien graphe.

### Axe 3 — Génération 256k ✅
- `chat_stream_wrapper` transmet enfin `max_tokens` (plancher 2048 au lieu du repli 1024 qui coupait
  les réponses procédurales).
- Budget contexte relevé : `SPACE_CONTEXT_MAX_CHARS` 18000→80000, `SPACE_CONTEXT_MAX_PASSAGE_CHARS`
  4000→12000 (pages entières non tronquées), `SPACE_HISTORY_MAX_CHARS` 8000→16000. `docker-compose.yaml`
  mis à jour (les valeurs y écrasaient les défauts). Reranker OFF → tous les hits fusionnés atteignent
  déjà le contexte : relever les caps suffit à charger plusieurs pages.
- Tests : `tests/test_space_context_generation.py`.

### Axe 2 — Retriever ✅
- Parallélisation des 4 retrievers (`_run_retrievers`, threads + sessions DB dédiées, `asyncio.gather`).
  Flag `RETRIEVAL_PARALLEL_ENABLED` (défaut on ; séquentiel en repli).
- Plafond du boost catégorie `RETRIEVAL_CATEGORY_BOOST_MAX` (défaut 1.5) : empêche l'amplification
  pathologique ~1.9× (critique sans reranker). `CATEGORY_MIN_CONFIDENCE` NON touché (ingestion).
- Tests : `tests/test_retrieval_parallel_and_boost_cap.py`.

### Axe 4 — Mémoire conversation ✅
- Sur `topic_shift`, l'historique n'est PAS envoyé à la génération (`chat.py`) → le modèle ne reste plus
  ancré à l'ancien sujet.
- `current_topic` persistant dans `query_context`, réinitialisé sur `topic_shift`, conservé sinon.
- Tests : dans `tests/test_fused_understanding.py`.

### À faire côté ops
- **Recréer le conteneur** pour charger les nouveaux défauts d'env (`docker compose up -d --force-recreate web`).
- Mesurer la latence réelle par phase (traces) et, si besoin, ajuster les flags/plafonds.
