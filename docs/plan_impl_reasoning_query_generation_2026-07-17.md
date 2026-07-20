# Plan d'implémentation — UN raisonnement à la place de l'intention + réécriture, et génération raisonnée (Small 4, reasoning high)

_Date : 2026-07-17 · Branche : `fix/retriever`. Concrétise
`plan_reasoning_first_small_2026-07-17.md` (§2 appel A, §5 small partout) en plan
d'implémentation ancré dans le code. Décisions : **température 0.2-0.3**, test sur
**`mistral-small-latest` (= Mistral Small 4)**, `reasoning_effort="high"`._

---

## 1. Analyse — pourquoi la génération répond à côté malgré un bon CAG

Constats vérifiés dans le code ce jour :

1. **La prod tourne à température 0.7 sur `mistral-large-latest`** (`.env` :
   `SPACE_CHAT_TEMPERATURE=0.7`, `MODEL_FAST=mistral-large-latest`). Trois sources de
   vérité divergent : `config.py:72` dit 0.3 par défaut, `docker-compose.yaml:20` dit
   0.55, le `.env` (qui gagne) dit 0.7. À 0.7 sur un contexte CAG de dizaines de
   milliers de tokens, la variance produit exactement les symptômes observés :
   hors-sujet et hallucinations alors que l'info est dans le contexte.
   **Cause n°1, corrigeable immédiatement (chantier C1).**

2. **Aucun raisonnement en amont.** `_node_fused_understand`
   (`lightweight_query_understanding.py:470`) est une extraction JSON à plat : route
   direct/rag + signaux + condense + vagueness + topic_shift + guided remplis comme un
   formulaire, en un appel sans réflexion. La « réécriture » des requêtes retriever est
   ensuite **déterministe et sans LLM** (`_node_build_queries_fast:643` : concat
   standalone_question + entités) — personne ne se demande *ce qu'il faut vraiment
   chercher* ni *s'il faut chercher*.

3. **Aucun raisonnement en aval.** La génération (`chat.py:1473 generate()`) reçoit tout
   le contexte CAG packé + le system prompt grounding (`chat.py:212`) et rédige en un
   jet à temp 0.7. Aucune étape « quels documents répondent à LA question posée ? »
   avant rédaction → le modèle part sur un document voisin plausible.

4. **Le client streaming est incompatible reasoning tel quel.**
   `mistral_service.chat_stream:447` ne lit que `delta.content` en *string* ; en mode
   reasoning, `delta.content` devient une **liste** (ThinkChunk) pendant la phase de
   réflexion. À adapter avant toute bascule (C0).

5. **Code mort à purger avec la bascule** : `reason_query_intent` et
   `decide_retrieval_route` (`query_reasoning_service.py`) ne sont plus appelés sur le
   chemin fused (seul le modèle pydantic `QueryIntent` est réutilisé `chat.py:1301`
   pour `refine_with_source_authority`).

---

## 2. API Mistral — vérifié dans les docs officielles (2026-07-17)

- `reasoning_effort` est disponible sur **`mistral-small-latest` (Small 4)** et
  **`mistral-medium-3-5`**, endpoint chat completions standard. Valeurs :
  **`"high"`** (ThinkChunk complet avant la réponse, tokens en plus) et **`"none"`**
  (pas de trace, vitesse Small 3.2).
- Réponse non-stream avec `high` : `message.content` devient une **liste de chunks** —
  `{type:"thinking", thinking:[TextChunk…]}` puis `{type:"text", text:"…"}`.
- Stream : 3 phases — `delta.content` **liste** (ThinkChunk) → chunk de transition
  (fin thinking + premier TextChunk) → **string** (réponse).
- Multi-tours : Mistral recommande de **rejouer le message assistant complet, ThinkChunk
  inclus**, dans l'historique (sinon dégradation). Arbitrage coût acté : replay du
  **dernier tour uniquement** (cf. plan stratégique §3).
- **Non documenté** : compatibilité `reasoning_effort="high"` avec
  `response_format=json_object`, structured outputs, tools et vision → **spike C0**.
  Le design n'en dépend pas : en repli, on parse le JSON dans le TextChunk.

Sources : docs.mistral.ai (Reasoning / Adjustable reasoning, Chat Completions),
mistral.ai/news/mistral-small-4.

---

## 3. Architecture cible

```
message + historique + fil persistant (current_topic, focus_entities, ancre)
   │
   ▼
APPEL A — mistral-small-latest, reasoning_effort="high"          [C2]
   raisonne : résout les ellipses → juge (sens/précision/domaine) → conclut
   ├─ "direct"    → réponse directe (salutation/identité), 0 retrieval
   ├─ "clarifier" → 1 question courte, STOP (0 retrieval)
   ├─ "refuser"   → hors domaine / non-sens : refus poli, STOP
   └─ "chercher"  → requêtes RÉÉCRITES par canal :
        semantic (phrase dense pour l'embedding) · lexical (mots-clés + refs
        exactes) · colpali (description visuelle) + signaux + topic/focus
   │
   ▼
RETRIEVAL (inchangé : search_technical_passages + CAG packing + ancre)
   │
   ▼
APPEL B — mistral-small-latest, reasoning_effort="high", temp 0.2   [C3]
   raisonne : élit les documents qui répondent À CETTE question → rédige
   thinking MASQUÉ du stream · vérification programmatique des refs/cotes
```

L'appel A **remplace** : route direct/rag + extraction de signaux + condense
(standalone_question) + vagueness + décision guidée + génération des requêtes.
2 appels LLM par message, comme aujourd'hui.

---

## 4. Chantiers

### C0 — Client Mistral : support reasoning (préalable, ~0,5-1 j)

`app/services/mistral_service.py` :
- `chat()` : les kwargs passent déjà dans le payload (`payload.update(kwargs)`) →
  `reasoning_effort` transmissible sans modif. Ajouter un helper
  `split_reasoning_content(message) -> (text, thinking)` qui gère
  `content: str | list[chunk]`, utilisé par tous les appelants non-stream.
- `chat_stream()` : gérer `delta.content` en **liste** — extraire les deltas thinking et
  les émettre séparément (`{"thinking": …}`) ; le texte final reste
  `{"message": {"content": …}}`. Le frontend actuel n'affiche que `message.content` →
  le thinking est masqué par construction.
- `_clean_messages()` : tolérer les chunks `type:"thinking"` dans un message assistant
  (aplatis hors replay ; préservés pour C4).
- **Spike compat** (script scratch, ~1 h) : sur `mistral-small-latest` +
  `reasoning_effort="high"`, tester (a) `response_format json_object`,
  (b) vision `image_url`, (c) tools. Consigner les résultats ici.

### C1 — Température + modèle : quick win immédiat (~15 min)

- `.env` : `SPACE_CHAT_TEMPERATURE=0.2` (au lieu de **0.7**) et
  `MODEL_FAST=mistral-small-latest` (test Small 4 voulu).
- `docker-compose.yaml:20` : défaut `0.2` (au lieu de 0.55) ; `config.py:72` : défaut
  `0.2` (au lieu de 0.3). Une seule valeur partout — fini les 3 sources de vérité.
- ⚠ Garde-fou : les modèles reasoning Mistral ont historiquement des recos de sampling
  plus hautes (Magistral : 0.7). Si le golden montre un thinking dégradé à 0.2, tester
  0.3 avant de conclure. L'anti-hallucination durable est portée par C3 (vérification
  programmatique), pas par la température seule.
- C1 est mesurable **seul** : golden avant/après pour isoler l'effet température.

### C2 — Appel A : le raisonnement qui remplace intention + réécriture (~1 sem)

- Nouveau nœud `_node_reason_understand` dans `lightweight_query_understanding.py`,
  graphe court `merge_context → reason_understand → END`. Plus de
  `build_queries_fast` ni `plan_multi_query`/`generate_queries` sur ce chemin : **les
  requêtes sortent du raisonnement**.
- Flag `QUERY_REASONING_FIRST_ENABLED` (défaut `false`) + `QUERY_REASONING_TIMEOUT_S=45`
  (le reasoning high dépasse les 25 s actuels de `QUERY_UNDERSTANDING_TIMEOUT_S`).
  **Fallback automatique** sur le graphe fused actuel si timeout ou parse KO — jamais de
  blocage du pipeline.
- Appel : `mistral-small-latest`, `reasoning_effort="high"`. Sortie = un JSON unique
  dans le TextChunk (ou structured output si le spike C0 valide la compat).
- Prompt système : réutiliser `build_extract_signals_prompt(session)` pour le
  vocabulaire métier (catégories, symptômes, gammes) mais reformulé en **instructions de
  raisonnement**, pas de formulaire :
  1. Résous les références au fil (état persistant + historique) ; corrige les typos.
  2. Juge : compréhensible ? assez précis ? dans le domaine PROFERM ? → 4 verdicts.
     Le doute vaut mieux que l'invention : demander, jamais deviner. 1 question max.
  3. Si `chercher` : réécris les requêtes pour CHAQUE canal — `semantic` (question
     autonome dense optimisée embedding), `lexical` (mots-clés métier + références
     EXACTES telles quelles), `colpali` (description visuelle si schéma/tableau utile).
- Schéma de sortie, mappé sur `LightweightQueryResult` → **zéro changement dans
  chat.py** :

  ```json
  {
    "decision": "chercher | clarifier | refuser | direct",
    "message_utilisateur": "…",            // si clarifier / refuser
    "standalone_question": "…",            // si chercher
    "queries": { "semantic": "…", "lexical": "…", "colpali": "…" },
    "topic_shift": false,
    "current_topic": "…",
    "signals": { "...": "champs actuels validés par parse_and_validate_signals" },
    "is_guided": false, "flow_kind": "howto", "detected_symptom": "",
    "product_named": true, "needs_intent_clarification": false
  }
  ```

  Mapping : `direct` → `route="direct"` ; `clarifier`/`refuser` →
  `ready_for_retrieval=False` + `clarification.question=message_utilisateur` (réutilise
  le flux SSE clarification existant `chat.py:1156`) ; `chercher` →
  `retrieval_queries=RetrievalQueries(semantic, lexical, colpali)`.
- L'état du fil continue d'être alimenté : `build_conversation_state` consomme
  `current_topic` / `topic_shift` / entités comme aujourd'hui.
- Dépréciation après validation golden : `decide_retrieval_route`,
  `reason_query_intent`, nœuds condense/vagueness/plan_multi_query/generate_queries et
  leurs prompts (`query_understanding_graph.py`), graphe legacy.

### C3 — Appel B : génération raisonnée + anti-hallucination (~3-4 j)

- `_stream_llm_to_sse` (`chat.py:636`) : passer `reasoning_effort="high"` via
  `chat_stream_wrapper` ; ne relayer au SSE que les deltas texte (thinking filtré).
  Optionnel UX : émettre un statut « analyse des documents… » pendant la phase
  thinking. `SourcesTagStreamFilter` ne s'applique qu'au texte.
- `SPACE_CHAT_SYSTEM_PROMPT` (`chat.py:212`) : ajouter un bloc « MÉTHODE (dans ta
  réflexion) » : reformule la question posée → liste les documents dont l'EN-TÊTE
  correspond au produit/gamme/version demandés → rédige UNIQUEMENT à partir d'eux →
  relis chaque référence/cote citée. Et renforcer : « si les documents ne répondent pas
  À CETTE question précise, dis-le au lieu de répondre sur un sujet voisin ».
- **Vérification programmatique post-génération (0 LLM)** : extraire de la réponse les
  références/cotes (regex codes produits, dimensions mm, normes DTU/NF EN) et vérifier
  leur présence LITTÉRALE dans le contexte packé. Violation → log trace ; itération 2 :
  1 retry non-stream puis abstention honnête.
- ⚠ `max_tokens` : le thinking consomme le budget de complétion.
  `CAG_MAX_COMPLETION_TOKENS=3072` → passer à ~8192 en mode reasoning (variable dédiée,
  ex. `CAG_REASONING_MAX_COMPLETION_TOKENS`).

### C4 — Continuité multi-tours (~2 j)

- Persister le thinking du **dernier tour** (colonne `Message.thinking` ou champ dans
  `query_context`) et rejouer le message assistant complet (ThinkChunk inclus) dans
  l'historique de l'appel suivant — fenêtre = 1 tour (arbitrage coût vs reco Mistral).
- `_clean_messages()` : préserver ce format au replay (ne pas aplatir le dernier
  assistant). Les ThinkChunks complets vont dans les traces, pas dans l'historique.

### C5 — Tests, golden, bascule (~2-3 j, en continu)

- Nouveaux tests : nœud reasoning (mock `chat` retournant content liste
  ThinkChunk+TextChunk-JSON), parseur stream 3 phases, fallback timeout→fused, mapping
  4 verdicts → `LightweightQueryResult`, vérification programmatique C3.
- Les 6 fichiers qui mockent le fused (`test_lightweight_query_understanding`,
  `test_query_understanding_fusion`, `test_fused_understanding`, `test_guided_*`,
  `test_conversation_state`) restent verts tant que le flag est off.
- Golden `tests/fixtures/golden/` : A/B en 3 paliers — actuel (large, 0.7) → C1 seul →
  C1+C2+C3. Mesures : hors-sujet, refs inventées, latence 1er token, taux de
  clarification/refus. Exécution : `docker compose exec web pytest` (deps Docker-only).
- Bascule : flag on en staging → prod. Repli qualité rédaction :
  `mistral-medium-3-5` sur l'appel B uniquement (config, pas de refactor).

---

## 5. Risques et parades

| Risque | Parade |
|---|---|
| `reasoning` + `json_object` incompatibles | Design par défaut = JSON parsé dans le TextChunk ; spike C0 tranche |
| Latence reasoning high ×2 appels | Refus/clarification précoce = tour économisé ; fallback fused sur timeout ; mesuré au golden |
| Temp 0.2 dégrade le thinking | Tester 0.3 ; la sécurité vient de la vérification programmatique C3 |
| Small 4 rédige moins bien que large | `mistral-medium-3-5` sur B seul (variable de config) |
| Thinking gonfle les completion tokens | `max_tokens` dédié relevé (C3) + suivi usage dans les logs `[MISTRAL]` |
| Régression guidé/fiche technique (fast-paths) | Champs guided conservés dans la sortie A ; tests guided au vert flag off puis on |

---

## 6. Ordre d'exécution recommandé

1. **C1 aujourd'hui** (temp 0.2 + small) — quick win mesurable seul au golden.
2. **C0** (client + spike compat) en parallèle.
3. **C3** (génération raisonnée) — c'est la douleur exprimée (« répond à côté »).
4. **C2** (appel A remplace intention + réécriture) — le cœur du remplacement.
5. **C4** (replay dernier ThinkChunk), **C5** en continu à chaque palier.
