# Audit complet stack Retriever / Génération — 2026-07-03 (post-refonte phases A-D)

> Périmètre : compréhension de requête → 4 retrievers → fusion/boosts → packing CAG →
> génération → post-traitement (sources, guidé, fiche technique) → indexation/métadonnées.
> Méthode : 4 passes d'audit indépendantes (retrieval core, query understanding,
> génération, indexation) + vérification des flags RUNTIME dans le conteneur + mesures
> sur la base réelle. Chaque constat cité a été rapproché du code (fichier:ligne).

---

## 0. Photographie du runtime (vérifiée dans le conteneur, pas la config théorique)

| Flag | Valeur réelle | Commentaire |
|---|---|---|
| CAG_ENABLED / CONVERSATION_ANCHOR | true / true | pipeline refondu actif |
| QUERY_UNDERSTANDING_ENABLED | **true** (`.env`) | ⚠️ compose default = false : un déploiement sans `.env` perd TOUTE la compréhension (standalone question, topic_shift, intent, ancre) |
| QUERY_FUSED_UNDERSTANDING | true (défaut code) | 1 appel LLM fusionné ✓ |
| GUIDED_FLOW / FICHE_TECHNIQUE | true / true | actifs |
| RERANKER_ENABLED | **false** | ⚠️ le boost catégorie devient le seul juge post-RRF |
| VISION_RERANK_ENABLED | false | de toute façon chemin mort (cf. §5) |
| MODEL_FAST | mistral-large-latest | 256k ✓ |
| RAG_TOP_K / RAG_POOL_SIZE | 20 / (unset→20) | pool effectif = max(RERANK_POOL=40, 20, 20) = 40 |
| COLPALI_RELATIVE_MARGIN | 0.10 (`.env`) | ⚠️ compose dit 0.18, code dit 0.10 — dérive non documentée |

**Mesure base réelle** : 57 documents ; `proferm_gammes`/`materials`/`product_types`
remplis sur **22/57 (39 %)** ; `source` sur 39/57. → 61 % des documents produisent des
en-têtes CAG dégradés et des choix d'identification produit au titre brut.

---

## 1. Ce qui va (forces à préserver)

1. **Architecture retrieval page-centric 4 canaux** (ColPali gated + pgvector + BM25 + KAG
   → RRF k=60), retrievers en parallèle (threads + sessions SQL dédiées), gating ColPali à
   deux niveaux (32 marqueurs visuels + intents) avec filet de rattrapage si le texte est
   faible. `space_search_service.py:734-1184`.
2. **CAG packer** : agrégation par document, document entier vs fenêtré, budget par intent,
   rognage des pages les plus éloignées du match, en-têtes anti-confusion de gammes, cache
   TTL. C'est l'ossature correcte d'un assistant documentaire.
3. **Compréhension fusionnée** : 1 appel LLM pour route+signaux+condense+topic_shift
   (contre 5-6 historiquement), prompt clair, signaux qui BOOSTENT sans jamais FILTRER
   (pas de faux négatifs par sur-contrainte). `lightweight_query_understanding.py:320-521`.
4. **Continuité conversationnelle** : ancre documentaire persistée + réutilisée, topic_shift
   qui élague l'historique, garantie d'inclusion des documents d'ancre dans le contexte.
5. **mistral_service robuste** : 5 retries, backoff exponentiel + jitter, Retry-After
   respecté, timeout stream 400s, idle-break 90s. `mistral_service.py:14-95,328-499`.
6. **Indexation L0/L1/L2 propre** : page anchors + feuilles vision (Ministral, fallback
   pymupdf4llm fiable) + enrichissement contextuel versionné ; LanceDB delete-before-insert
   idempotent ; merge inter-pages calibré.
7. **Filtre `<sources>` du stream** robuste (balises coupées, JSON invalide, plafond
   anti-faux-positif) ; sources UI par document alignées sur le contexte réel (refonte B).
8. **Observabilité retrieval** : résumé une-ligne par étape (retrievers→RRF→rerank→final),
   statuts explicites (ok / low_confidence_clarification / no_results / disabled / error).
9. **Infra d'éval existante** : `retriever_evaluator.py` (precision/recall contexte, MRR,
   juge LLM) branchée sur des routes admin — il manque le dataset golden et le niveau
   document, pas l'outillage.

---

## 2. Constats critiques (P0) — ce qui empêche le niveau « vrai assistant SAV »

### P0.1 — Métadonnées produit absentes sur 61 % du corpus (LE plus gros levier qualité)
Aucune passe d'indexation ne remplit `proferm_gammes` / `materials` / `product_types`
(remplissage manuel uniquement, `document_service_new.py:115-141` n'infère que `source`).
Conséquences en cascade : en-têtes CAG vides (la règle anti-mélange de gammes n'a plus de
support), choix de l'étape 0 du guidé repliés sur des titres de fichiers, boosts
matériau/source inopérants, filtre produit du routeur guidé aveugle.
**Reco** : extraction LLM systématique à l'indexation (gamme/matériau/type depuis les
premières pages + nom de fichier, vocabulaire contrôlé par le catalogue), backfill des 35
documents existants, badge « métadonnées incomplètes » en admin (cf. P1.3
`classification_status`).

### P0.2 — Robustesse du flux de génération (5 défauts qui font perdre des réponses)
1. **Fallback Mistral 400 mal ciblé** : le niveau 1 retire l'historique mais garde les
   ~60-100k tokens de CAG — si le 400 vient de la taille, on rejoue le même échec
   (`chat.py` fallbacks). Le fallback devrait d'abord RE-PACKER plus petit (budget eco).
2. **Réponse partielle persistée sans marqueur** si le client coupe ou si une exception
   survient à mi-stream : l'utilisateur voit une réponse tronquée « propre ».
3. **`_persist_assistant_reply` avalé** (~10 sites try/except) : si la DB échoue, la
   réponse est perdue silencieusement, le front reçoit `done:true`.
4. **Filtre `<sources>` non réinitialisé entre fallbacks** : état de capture résiduel
   possible (cas rare mais réel).
5. **6 copies de la boucle `async generate()`** (chat générique, direct, clarification,
   guidé, fiche, RAG) : tout correctif doit être appliqué 6 fois — c'est la cause racine
   des 4 points précédents. **Reco** : factoriser UN helper de stream (émission, filtre,
   persistance, erreurs) puis corriger les 4 défauts dedans une seule fois.

### P0.3 — Pannes silencieuses et absence de budget temps
- Aucun timeout sur : encode ColPali, MaxSim LanceDB, rerank, appels LLM de compréhension
  (`colpali_service.py:117-146`, `lancedb_service.py`, 6 `await chat(...)` de
  `lightweight_query_understanding.py`). Un hang Mistral/CPU = requête bloquée sans fin.
- Exceptions avalées sans signal : embedding requête (pgvector rend `[]` sans marqueur),
  BM25, KAG — le pipeline « réussit » en mode dégradé invisible.
- JSON du fused understanding non validé (pas de Pydantic) : clé manquante → défauts
  silencieux ; JSON invalide → aucune retry.
**Reco** : `asyncio.wait_for` généralisé (30 s), statut `degraded_channels=[...]` remonté
dans la réponse retrieval + loggé, validation Pydantic du JSON fused avec 1 retry.

### P0.4 — Gouvernance des appels LLM pré-retrieval
Pire cas actuel : `decide_guided_mode` + fused understanding + génération de requêtes =
**3-4 appels LLM avant le premier hit**, avec **3 taxonomies d'intents disjointes** sans
mapping (guided: howto/diagnostic ; signals: specification/installation/... ; reasoning:
company_info/supplier_info) et aucune réconciliation si les classifieurs divergent.
Les salutations paient aussi un appel LLM complet (aucun court-circuit regex).
**Reco** : fusionner la décision guidée DANS le fused understanding (champs is_guided /
flow_kind / product_named / needs_intent_clarification dans le même JSON → 1 appel au lieu
de 2), source de vérité unique des intents dans `slot_catalog.py`, regex de court-circuit
salutations avant tout LLM.

### P0.5 — Ranking sans arbitre (config runtime actuelle)
`RERANKER_ENABLED=false` en prod : après le RRF, le boost catégorie ×1.5 plafonné et les
soft boosts additifs NON bornés (`retrieval_boost_service.py:323-326`) sont les seuls
signaux d'ordre. Une page sur-catégorisée peut passer devant un vrai match texte, sans
cross-encoder pour rattraper. Le CAG amortit (agrégation document), mais l'ordre des
documents packés en dépend.
**Reco court terme** : borner les soft boosts (plafond absolu), bonus multi-canaux
explicite dans le RRF (page vue par ≥2 canaux), corriger le N+1 SQL
(`apply_soft_boosts_to_passages` charge Document par passage). **Décision à prendre** :
réactiver le reranker MiniLM local (coût faible, il y a déjà tout le code + K dynamique)
une fois l'éval en place pour mesurer.

---

## 3. Constats importants (P1)

1. **Cache fulltext cross-process** : l'invalidation depuis le worker Celery ne touche pas
   le cache mémoire du process web (TTL 300 s = borne du staleness). Redis est déjà dans
   la stack → clé de version par document dans Redis, vérifiée au hit (1 GET), ou pub/sub.
   *(Introduit par la refonte du 2026-07-03 — assumé, à durcir.)*
2. **`QUERY_UNDERSTANDING_ENABLED` défaut compose = false** : un environnement recréé sans
   `.env` complet perd silencieusement la moitié du pipeline (et le budget CAG par intent
   retombe sur « default »). Passer le défaut compose à true, ou fail-fast au boot si CAG
   actif sans compréhension.
3. **`classification_status` jamais mis à jour par l'indexation** (`document.py:36`) :
   reste « incomplete » à vie ; impossible de distinguer en UI un document complet d'un
   document sans KAG/enrichissement/métadonnées (cf. P0.1). Idem : document marqué
   `completed` même si KAG et enrichissement L2 ont échoué (corpus tronqué invisible).
4. **Message statique « seuil minimum de 75% »** (`chat.py`) : ce seuil n'existe nulle
   part dans le retrieval — message mensonger côté utilisateur. À reformuler (« aucune
   source suffisamment pertinente ») + proposer reformulation/clarification.
5. **Fiche technique — regex trop large** : `\b\d{3,5}[A-Za-z]?\b` matche « 123 », des
   quantités, des numéros de page → risque de court-circuiter le RAG sur un faux positif.
   Resserrer (≥4 chiffres OU alphanum à préfixe lettre, + garde contextuelle).
6. **Historique tronqué à 12 messages / topic perdu sur route directe** : un tour
   « direct » (salutation) ne persiste pas le query_context → le fil peut se dégrader.
7. **Métadonnées L2 inexploitées** : `chunk_role` (procedural_step / diagnostic_unit) et
   champs structurés (symptom, probable_cause, tools…) sont indexés mais jamais utilisés
   par le ranking — levier SAV évident (booster diagnostic_unit quand flow diagnostic).
8. **Page anchors L0 non embeddés pgvector** (by design, ColPali les couvre) : si le
   gating écarte ColPali ET que le fallback (seuil dur < 5 pages texte) ne se déclenche
   pas, une page 100 % visuelle est introuvable. Le seuil 5 mérite d'être relatif à la
   taille du corpus.
9. **Lost-in-the-middle partiel** : ordre actuel system(docs ~60k) → historique → user
   (+rappel). Le rappel final (refonte A) mitige ; si l'éval montre des pertes, tester
   docs dans un message user dédié juste avant la question, historique AVANT les docs.

---

## 4. Dette / mineurs (P2)

- Logs : `prompt_preview[:100]` (contenu utilisateur dans les logs), traces
  pipeline/inputs verbeuses.
- `DocumentChunk.text` ≡ `content` (duplication sans rôle).
- `.env_exemple` : `MODEL_FAST` défini deux fois (l.39 vide, l.153 rempli).
- Concurrency d'extraction vision sans throttle API (OK à 2-4, risqué au-delà).
- KAG hop_limit=1 : entités à 2 sauts perdues (assumer ou documenter).
- Condition fallback 400 `not assistant_response` fragile si chunks vides.

---

## 5. Code mort — à trancher (supprimer ou réactiver, pas laisser)

| Élément | État | Reco |
|---|---|---|
| `vision_reranker_service.py` | jamais appelé par le chemin multimodal (seul actif) ; `VISION_RERANK_ENABLED` est un no-op | supprimer ou re-brancher post-RRF |
| `_retrieve_leaves_sql()` (`space_search_service.py:175-310`) | appelée nulle part | supprimer |
| `search_corrective_faq_passages()` | hardcodé « disabled » | supprimer ou réactiver |
| `search_relevant_passages` chemin hybrid legacy | doublonne le multimodal | isoler derrière flag + date de retrait |
| `decide_retrieval_route` | redondant avec le fused (utilisé seulement si QU off) | absorber dans P0.4 |
| Multi-query groups | implémenté, OFF, quasi toujours « single » | garder OFF, documenter |

---

## 6. Plan d'action priorisé

| # | Chantier | Contenu | Effort | Impact SAV |
|---|---|---|---|---|
| 0 | **Quick wins** (1 j) | défaut compose QUERY_UNDERSTANDING→true + COLPALI_RELATIVE_MARGIN unifié ; message « 75% » ; regex fiche resserrée ; court-circuit regex salutations ; N+1 soft boosts ; plafond soft boosts ; `.env_exemple` nettoyé | S | ★★★ |
| 1 | **Métadonnées produit** (P0.1) | extraction LLM gamme/matériau/type à l'indexation + backfill 35 docs + `classification_status` mis à jour + badge admin | M | ★★★★★ |
| 2 | **Robustesse stream** (P0.2) | helper de stream unique (émission/filtre/persist/erreurs) ; fallback 400 = re-pack budget eco ; marqueur réponse tronquée ; persist avec retry | M | ★★★★ |
| 3 | **Budgets temps + dégradation visible** (P0.3) | timeouts partout ; `degraded_channels` dans le statut retrieval ; validation Pydantic du fused + 1 retry | M | ★★★★ |
| 4 | **1 seul appel LLM pré-retrieval** (P0.4) | guided-decision fusionnée dans le fused understanding ; taxonomie d'intents unique | M | ★★★ (latence -1 à -2 s/tour) |
| 5 | **Ranking** (P0.5/P1.7) | bonus multi-canaux RRF ; boost `chunk_role` diagnostic en flow SAV ; décision reranker MiniLM après éval | M | ★★★ |
| 6 | **Cache cross-process** (P1.1) | version de document dans Redis, check au hit | S | ★★ |
| 7 | **Nettoyage code mort** (§5) | suppression/re-branchement | S | ★ (maintenabilité) |
| 8 | **Éval (Phase E)** | golden set 30-50 questions (dont poignée, suivi elliptique, multi-gammes, page visuelle pure), recall@doc, groundedness, taux de clarification à bon escient, coût/latence — **condition de tuning des chantiers 3-5** | M | ★★★★★ (condition de vérité) |

Ordre recommandé : 0 → 1 → 2 → 8 (l'éval dès que possible, en parallèle de 3-4) → 3 → 4 → 5 → 6 → 7.

---

## 7. Notes de vérification (croisement des passes d'audit)

- L'affirmation « `has_source_file` absent du packing CAG » (passe génération) est
  **fausse depuis la refonte du 2026-07-03** — le packer renseigne le champ
  (`context_packer_service.py`, bloc `cag_documents`). Constat écarté.
- `COLPALI_RELATIVE_MARGIN` : la dérive compose (0.18) vs code (0.10) est réelle mais le
  `.env` runtime force 0.10 — à unifier quand même (chantier 0).
- Couverture métadonnées mesurée sur la base réelle le 2026-07-03 : 22/57 documents
  complets (39 %) — chiffre à re-mesurer après le chantier 1.
