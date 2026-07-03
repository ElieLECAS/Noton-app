# Refonte professionnelle de la stack Retriever / Génération — 2026-07-03

> Objectif : passer d'un pipeline « RAG bricolé par couches successives » à un assistant
> de qualité Gemini/ChatGPT : contexte cohérent, prompt cohérent, clarification quand il
> faut, et jamais de plongée dans un produit que l'utilisateur n'a pas nommé.

---

## 0. État des lieux exact (vérifié dans le code au 2026-07-03)

### 0.1 Ce que fait réellement le pipeline aujourd'hui

```
message utilisateur
  → compréhension légère (1 appel LLM : route, signaux, standalone_question, topic_shift)
  → 4 retrievers parallèles : ColPali (gated par intent), pgvector, BM25, KAG — pool par canal
  → fusion RRF → boost catégories → coupe RAG_TOP_K (8 compose / 20 env) → boost ancre conversation
  → CAG packer (context_packer_service.build_cag_context) :
      · agrégation des passages PAR DOCUMENT (score = max + 0.2·reste)
      · top CAG_MAX_DOCUMENTS=8 documents (+ documents d'ancre toujours inclus)
      · document ENTIER si ≤ CAG_FULL_DOC_MAX_TOKENS=20k tokens estimés,
        sinon fenêtre pages matchées ± CAG_PAGE_RADIUS=3
      · budget global CAG_TOKEN_BUDGET=100k tokens — rendu TEXTE avec en-têtes
        (source/gamme/matériau) + marqueurs [page N], injecté dans le message SYSTEM
  → bloc « fil de conversation » (sujet courant, standalone_question) en FIN de system
  → historique (élagué si topic_shift)
  → message USER en dernier (enriched_user_message) + images PNG attachées
  → génération Mistral Large (stream)
```

### 0.2 Réponses aux questions posées

**« Dans le contexte c'est uniquement le texte extrait ? »**
Oui. Les blocs CAG sont 100 % texte (chunks L1, ou texte pymupdf pour les placeholders
ColPali). Les images sont un canal séparé : les PNG de pages issus du pipeline multimodal
(`retrieval["images"]`, plafonné `RAG_MAX_IMAGES=12`) sont attachés au **message user**,
uniquement si le modèle est vision.

**« Dans le top-k 20 on envoie les pages en PNG ? »**
Non, pas les 20. Les PNG sont sélectionnés à partir des passages top-k du pipeline
multimodal, tronqués à 12 (`RAG_MAX_IMAGES`). ⚠️ Incohérence : cette sélection est faite
sur les **passages** top-k, PAS sur les documents réellement packés par le CAG. On peut
donc envoyer le PNG d'une page dont le document n'est pas dans le contexte, et inversement
ne pas envoyer le schéma clé d'un document packé en entier.

**« Quel intérêt d'avoir "Sources textuelles (20)" si on balance des documents entiers ? »**
Aucun — c'est un reliquat. `sources_data` (chat.py ~1360) est toujours construit depuis
les 20 passages, alors que le contexte réel est décrit par `cag_documents` (déjà retourné
par le packer mais jamais affiché). La Phase 2 du plan CAG (sources par document) n'a pas
été câblée. L'UI ment donc sur ce que le modèle a vraiment lu.

**« Faut-il donner tout le document ? ou page n-5 / n+5 ? petit PDF entier ? »**
C'est déjà la logique implémentée : document entier si ≤ ~20k tokens (la quasi-totalité
des notices 4–20 pages), sinon fenêtre ±3 pages autour des pages matchées. Le vrai
problème n'est pas le rayon mais le **budget fixe 100k** : une question simple
(« quelle vis pour X ? ») paie le même prefill (coût + latence 1er token) qu'un
diagnostic complexe. → budget adaptatif par intent (voir Phase C).

**« Faut-il placer la question après le contexte ? »**
C'est déjà partiellement le cas : le message user (la question) est le **dernier** message,
et le bloc « fil de conversation » (standalone_question) est en fin de system. Ce qui
manque : une **restatement explicite** de la tâche après les ~100k tokens de documents
(« Réponds UNIQUEMENT à la question suivante : … ») et un system allégé (voir Phase A).

### 0.3 Les 5 incohérences structurelles identifiées

| # | Incohérence | Fichier | Effet |
|---|---|---|---|
| 1 | Le system prompt parle encore de « chunks » et « PASSAGES » alors que le CAG envoie des DOCUMENTS avec en-têtes | `chat.py:206-223` | Les règles de grounding/conflit ne s'appliquent pas au bon objet ; le modèle n'exploite pas les en-têtes |
| 2 | Le prompt INTERDIT de demander une clarification (« sans demander de précision ou de clarification à l'utilisateur », ligne 213) | `chat.py:213` | Cause directe du comportement INNOSLIDE : le modèle DOIT choisir un produit même quand l'utilisateur n'en a nommé aucun |
| 3 | Sources UI = passages top-k, contexte réel = documents CAG | `chat.py:1360+`, templates | « Sources textuelles (20) » sans rapport avec ce que le modèle a lu |
| 4 | Images PNG sélectionnées sur les passages, pas sur les documents packés | `chat.py:1178-1198`, `space_search_service.py:1126-1140` | Contexte visuel et contexte textuel désalignés |
| 5 | Le mode guidé verrouille un « topic » produit (SUJET) déduit du retrieval/LLM, jamais confirmé par l'utilisateur, et ignore l'étape de clarification | `guided_flow_service.py`, `procedural_router_service.py`, `query_reasoning_service.decide_guided_mode` | « comment régler la hauteur de poignée » → plonge dans INNOSLIDE ; et l'intention (définir vs régler) n'est pas désambiguïsée |

---

## 1. Principes de la refonte

1. **Un seul objet de contexte** : le document packé (CAG). Prompt, sources UI, images,
   citations — tout se réfère aux `cag_documents`, plus jamais aux passages bruts.
2. **Clarifier avant de plonger** : si la réponse dépend d'un produit/gamme/config que
   l'utilisateur n'a pas donné ET que le corpus en contient plusieurs → UNE question de
   clarification courte (ou présentation des 2-3 cas si la réponse est courte). C'est le
   comportement Gemini/ChatGPT attendu.
3. **Le contexte s'adapte à la question**, pas l'inverse : budget de packing par intent.
4. **Sandwich robuste** : instructions courtes → documents → rappel de tâche + question
   en dernier (anti lost-in-the-middle).
5. **Mesurer** : aucun réglage sans jeu d'éval golden (l'exemple « poignée » en fait partie).

---

## 2. Phase A — Refonte du prompt & de la structure des messages (le plus gros levier)

### A1. Réécriture de `SPACE_CHAT_SYSTEM_PROMPT` (chat.py:206)

Problèmes actuels : ~20 règles empilées, contradictoires (« concision stricte » vs
« présente systématiquement tous les cas » ; « ne demande pas de clarification » vs
« propose une étape de vérification »), vocabulaire chunk/passage obsolète.

Nouveau prompt, hiérarchisé en 4 blocs courts :

```
1. IDENTITÉ & TON (3 lignes) — LIA, PROFERM, prose naturelle, concision.
2. CONTEXTE — « Tu reçois des DOCUMENTS complets ou étendus, chacun avec un en-tête
   (source, gamme, matériau, type) et des marqueurs [page N]. Vérifie TOUJOURS l'en-tête
   avant d'attribuer une valeur à une gamme. Documents classés par pertinence. »
3. POLITIQUE DE RÉPONSE (l'ordre est une priorité) :
   a. Si la question est sans ambiguïté et couverte → réponds directement, concis.
   b. Si la réponse DÉPEND d'un produit/gamme/version non précisé par l'utilisateur
      et que les documents en couvrent PLUSIEURS → pose UNE question de clarification
      courte (propose les options trouvées dans les documents), OU si la réponse tient
      en 2-3 lignes par cas, donne les cas. NE choisis JAMAIS un produit à la place
      de l'utilisateur.
   c. Si l'information est absente → dis-le et propose la vérification.
4. GROUNDING (inchangé sur le fond, réécrit pour « documents ») : mot-pour-mot pour
   gestes/cotes, pas d'assemblage inter-sections, conflit → document le plus spécifique.
```

- Supprimer la ligne 213 (« sans demander de précision ou de clarification ») — remplacée
  par la politique 3b.
- Garder « aucune citation dans le texte » tant que le bloc `<sources>` (Phase B3) n'est
  pas là.

### A2. Sandwich de génération

Structure cible des messages :

```
system : prompt court (A1) + DOCUMENTS CAG + bloc fil de conversation (existant)
history : inchangé (élagage topic_shift conservé)
user   : question enrichie + RAPPEL FINAL : "Réponds uniquement à la question
         ci-dessus. Question autonome reformulée : «…». Si le produit concerné
         n'est pas identifiable sans ambiguïté, demande d'abord lequel."
         + images PNG (alignées Phase B2)
```

Implémentation : enrichir `build_rag_user_message` (rag_generation_service.py:237) avec un
paramètre `task_reminder` construit depuis `standalone_question`.

### A3. Montée `max_tokens` génération en mode CAG
`CAG_MAX_COMPLETION_TOKENS=3072` (réponses procédurales complètes, prévu au plan CAG
initial, jamais câblé).

**Tests** : prompts snapshot ; cas golden « poignée » (sans produit nommé → le one-shot
doit poser la question du produit, pas choisir INNOSLIDE).

---

## 3. Phase B — Un seul objet de contexte : le document CAG

### B1. Sources UI par document
- `sources_data` construit depuis `system_message["cag_documents"]` quand CAG actif :
  `[1] Notice seuil PMR 76100 (document complet)` / `[2] DTA 6/16-2335 (p.5-9)`.
- Garder le lien PDF + page d'atterrissage (première page matchée).
- Templates `space_detail.html` / `project_detail.html` : le libellé « Sources
  textuelles (N) » devient « Documents consultés (N) ».

### B2. Images alignées sur les documents packés
- Nouvelle fonction dans le packer : `select_cag_images(cag_documents, passages)` —
  PNG uniquement pour des pages INCLUSES dans le contexte, priorisées par
  `needs_page_image` / `section_type ∈ {diagram, step}` puis score, cap
  `CAG_MAX_IMAGES=8` (nouvelle clé, remplace l'usage de RAG_MAX_IMAGES ici).
- Chaque image légendée dans le texte du message user : « Image k = document i, page p »
  pour que le modèle relie PNG ↔ bloc texte.

### B3. Sources réellement utilisées (fin de réponse)
- Le modèle termine par un bloc masqué `<sources>{"used":[{"doc":1,"pages":[3,4]}]}</sources>`
  parsé côté stream (retiré de l'affichage) → l'UI n'affiche que les documents/pages
  réellement utilisés ; fallback = cag_documents. Remplace le fallback « par score ».

**Tests** : mapping cag_documents→sources, cap images, parsing `<sources>` robuste
(bloc absent, JSON invalide, bloc coupé par le stream).

---

## 4. Phase C — Packing adaptatif (coût/latence sans perte de rappel)

### C1. Budget par intent (remplace le 100k fixe)
| Intent (déjà extrait par la compréhension légère) | Budget | Max docs |
|---|---|---|
| lookup simple / fiche / valeur unique | 25k | 4 |
| howto / installation / procédure | 60k | 6 |
| diagnostic SAV / comparaison / synthèse | 100k | 8 |
| fallback (intent inconnu) | 60k | 6 |

Clé : `CAG_BUDGET_BY_INTENT` (JSON env) avec défauts ci-dessus. Effet direct sur le
prefill (latence 1er token) et le coût par requête.

### C2. Escalade automatique
Si la réponse contient « la notice ne précise pas » ET que des documents candidats sont
restés hors budget → seconde passe au budget max (1 seule fois, loggée). Alternative UI :
bouton « chercher plus largement ».

### C3. Affinages du fenêtrage (petit, pas urgent)
- Fusion des fenêtres chevauchantes (déjà implicite via set de pages — OK).
- Le rognage actuel `selected[:-1]` coupe les DERNIÈRES pages : rogner plutôt les pages
  les plus ÉLOIGNÉES des pages matchées.
- `CAG_PAGE_RADIUS=3` est bon ; ne pas passer à ±5 par défaut (coût sans gain mesuré) —
  à trancher par l'éval (Phase E).

### C4. Cache fulltext par document
Table ou colonne `document_fulltext` assemblée à l'indexation → le packer ne re-concatène
plus 60 chunks par requête (latence packing ≈ 0, moins de SQL).

---

## 5. Phase D — Refonte du mode guidé (le bug INNOSLIDE)

Le comportement rapporté : « comment régler la hauteur de poignée ? » → le guidé
verrouille INNOSLIDE (jamais nommé par l'utilisateur) et déroule un réglage à outil,
alors que l'utilisateur voulait peut-être *définir* une hauteur (dimensionnement).

### D1. Étape 0 obligatoire : identification du produit (slot-filling)
- `decide_guided_mode` retourne en plus `product_named: bool` (le produit/gamme est-il
  explicitement nommé dans le message ou l'historique ?).
- Si `product_named=false` : la PREMIÈRE étape du parcours est une question
  d'identification dont les choix sont générés depuis le retrieval (les gammes/produits
  distincts des documents candidats) + « Autre / je ne sais pas ». Le `topic` n'est
  verrouillé qu'après cette réponse. Interdiction au routeur (prompt
  `GUIDED_ROUTER_SYSTEM_PROMPT`) de fixer un SUJET non confirmé.
- L'ancre de conversation (`current_documents`) compte comme « nommé » UNIQUEMENT si elle
  vient d'un tour où l'utilisateur a lui-même cité le produit.

### D2. Désambiguïsation de l'intention
- « régler / ajuster » vs « définir / choisir / quelle hauteur » : `decide_guided_mode`
  retourne `needs_intent_clarification` quand le verbe est ambigu ET que le corpus couvre
  les deux lectures ; l'étape 0 pose alors la question (« Vous voulez ajuster une poignée
  existante, ou déterminer à quelle hauteur la poser ? »).
- Une question de dimensionnement (« quelle hauteur ? ») n'est PAS un howto → doit rester
  dans le pipeline one-shot. Renforcer le prompt de `decide_guided_mode` : guidé seulement
  si l'utilisateur veut être accompagné pas à pas, pas pour une valeur unique.

### D3. Le guidé respecte l'ancre et peut sortir
- `run_guided_turn` : passer `anchor_document_ids` au retrieval par étape (déjà prévu au
  plan CAG, jamais câblé).
- Choix permanent « Ce n'est pas mon produit / recommencer » qui ré-ouvre l'étape 0.

**Tests** : scénario golden complet de l'exemple poignée (message sans produit → étape 0
d'identification ; réponse « définir la hauteur » → sortie vers one-shot).

---

## 6. Phase E — Éval & pilotage (condition de toutes les autres phases)

- **Golden set** : 30-50 questions réelles (inclure : poignée sans produit, suivi
  elliptique « tu as ses dimensions ? », référence nue, question multi-gammes, question
  dont la réponse est 2 pages après le match).
- **Métriques** : recall@doc (le bon document est-il packé ?), groundedness (juge LLM),
  taux de clarification à bon escient (question posée quand ambigu / PAS posée quand
  clair), coût tokens/requête, latence 1er token P50/P95.
- Réutiliser `retriever_evaluator.py` ; ajouter un stage « packing » (docs packés vs
  attendus) et un stage « comportement » (clarification attendue ?).
- Chaque phase A→D est validée par un run A/B flag ON/OFF avant d'être défaut.

---

## RÉALISÉ (2026-07-03) — Phases A, B, C, D implémentées (E volontairement exclue)

- **A1** ✅ `SPACE_CHAT_SYSTEM_PROMPT` réécrit (chat.py) : 4 blocs, politique de clarification
  (l'interdiction « sans demander de précision » est supprimée), vocabulaire documents/en-têtes.
- **A2** ✅ Sandwich : `build_cag_task_reminder` (rag_generation_service) — rappel final +
  question autonome en toute fin de message user ; légendes d'images (`image_captions`).
- **A3** ✅ `CAG_MAX_COMPLETION_TOKENS=3072` passé à la génération en mode CAG.
- **B1** ✅ Sources UI par document : `_build_document_sources` (chat.py) depuis `cag_documents`
  (+ `matched_pages`/`has_source_file` ajoutés par le packer) ; libellé « Documents consultés »
  dans space_detail.html / project_detail.html ; badges avec pages utilisées / « complet ».
- **B2** ✅ `select_cag_images` (context_packer_service) : PNG uniquement pour des pages packées,
  priorité `needs_page_image`, cap `CAG_MAX_IMAGES=8`, légendes Image↔Document.
- **B3** ✅ Bloc `<sources>{"used":[...]}</sources>` : instruction dans le préambule CAG,
  filtrage du stream par `stream_source_filter.SourcesTagStreamFilter` (robuste aux balises
  coupées / JSON invalide / balise jamais fermée), sources UI filtrées par les documents
  réellement utilisés (fallback : tous les documents packés).
- **C1** ✅ `CAG_BUDGET_BY_INTENT` (30k/4 spec-doc, 60k/6 install-regul-défaut, 100k/8
  troubleshooting-comparatif) via `budget_for_intent`, intent passé depuis chat.py.
- **C3** ✅ Rognage des pages les plus ÉLOIGNÉES des pages matchées (`_trim_records_to_budget`)
  au lieu des dernières du document.
- **C4** ✅ Cache TTL des feuilles par document (`CAG_FULLTEXT_CACHE_TTL=300`), invalidé par
  l'indexation (suppression + finalisation) ; fixture autouse côté tests.
- **D1** ✅ Étape 0 d'identification produit : `decide_guided_mode` retourne `product_named`
  (true UNIQUEMENT si l'utilisateur a nommé le produit) ; `_build_identification_step` propose
  les gammes des documents candidats + « Autre / je ne sais pas » ; le SUJET n'est verrouillé
  qu'après confirmation ; « je ne sais pas » → questions d'observation sans présomption.
- **D2** ✅ `needs_intent_clarification` (régler vs définir) → pas de mode guidé, le one-shot
  clarifie (politique n°2 du prompt) ; dimensionnement explicitement exclu du guidé.
- **D3** ✅ Ancre de conversation passée au retrieval de chaque étape guidée ; choix permanent
  « Ce n'est pas mon produit / recommencer » (rouvre l'étape 0) quand le produit vient de l'étape 0.
- **C2** ⏸ non implémenté (escalade auto post-stream = 2ᵉ réponse après affichage — à traiter
  côté UI, bouton « chercher plus largement », avec la Phase E).
- Tests : `test_stream_source_filter.py`, `test_cag_generation_helpers.py`,
  `test_guided_product_identification.py`, extensions `test_cag_context_packer.py`
  (budget/intent, rognage, cache, images alignées).
- Ops : `docker compose up -d --force-recreate web` pour recharger ; nouvelles clés env
  documentées dans compose (CAG_MAX_COMPLETION_TOKENS, CAG_MAX_IMAGES, CAG_BUDGET_BY_INTENT,
  CAG_FULLTEXT_CACHE_TTL).

---

## 7. Ordre d'exécution recommandé

| Ordre | Chantier | Effort | Impact qualité perçue |
|---|---|---|---|
| 1 | A1+A2 prompt & sandwich (+ suppression interdiction de clarifier) | S | ★★★★★ |
| 2 | D1+D2 étape 0 produit + désambiguïsation intent du guidé | M | ★★★★★ (corrige l'exemple rapporté) |
| 3 | B1 sources par document (supprime « Sources textuelles (20) ») | S | ★★★★ |
| 4 | B2 images alignées sur docs packés | S/M | ★★★ |
| 5 | C1+C2 budget par intent + escalade | M | ★★★ (coût/latence) |
| 6 | E golden set + métriques | M | condition de vérité des 5 précédents |
| 7 | B3, C3, C4, D3 | M | finitions |

Notes ops : tout derrière flags existants ou nouveaux (`CAG_*`, `GUIDED_*`) ; redémarrage
`docker compose up -d --force-recreate web` ; tests via `docker compose exec web pytest`.
