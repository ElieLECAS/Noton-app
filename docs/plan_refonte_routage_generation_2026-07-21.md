# Refonte routage & génération — moteur de recherche IA à voie unique

> Date : 2026-07-21 · Branche : fix/retriever

## ✅ RÉALISÉ le 21/07 (C1-C7)

- **C4** : `SPACE_CHAT_SYSTEM_PROMPT` refondu — sections SUJET DEMANDÉ / FORMAT ADAPTATIF /
  GROUNDING DUR (le bloc COUVERTURE fait foi) / ANTI-DIGRESSION (MÉTHODE+POLITIQUE remplacées).
- **C3** : nouveau `app/services/coverage_service.py` — `extract_message_reference_codes`
  (regex partagée KAG, garde-fous cotes/millésimes/mots-domaine) +
  `build_coverage_block` (TROUVÉE/ABSENTE contre le contexte RÉELLEMENT packé,
  statuts ok/partiel/vide). Injecté en fin de message système (zone de forte attention).
- **C6** : `select_authority_chunks` + `build_pinned_reference_block` (kag_graph_service) —
  subject d'abord, bonus densité spec (code+unité), verbatim sourcé, cap 1200 c ;
  injecté avant le bloc couverture (les codes épinglés y sont signalés).
- **C1** : compréhension = seul routeur ; QU off ou en échec → route=search directe
  (plus d'appel `decide_retrieval_route` autonome dans chat.py).
- **C2** : fast-path fiche SUPPRIMÉ de chat.py (~110 lignes) ; `FICHE_TECHNIQUE_ENABLED`
  déprécié (.env commenté). Le service fiche_technique reste (module non branché).
- **C5** (version light) : le générateur direct passe par `_stream_llm_to_sse` (même
  moteur de stream/thinking que la voie RAG). La fusion totale direct→RAG reste possible
  plus tard ; avec la fiche supprimée il ne reste que 3 voies (direct/clarif/search).
- **Tests** : `tests/test_routing_generation_refonte.py` (20 tests : codes, couverture,
  prompt, anti-régression route fiche, pinning DB) ; `test_query_routing.py` réécrit au
  nouveau contrat ; `test_space_context_generation.py` mis à jour (reasoning none/high) ;
  `test_category_retrieval_boost.py` réparé (imports morts préexistants).

Reste (hors périmètre de cette passe) : fusion SSE totale (C5 complet), instruction
retrieval multi-query par code (la couverture + pinning couvrent le besoin), guidé SAV.

> Objectif : remplacer l'empilement de routes scriptées par UNE voie
> comprendre → chercher → générer (reasoning), où le FORMAT de la réponse est décidé
> par le générateur à la fin — jamais par un regex au début.
> Périmètre exclu : le parcours guidé SAV (intact derrière son flag, traité plus tard).

## Les 4 exigences (critères d'acceptation du chantier)

1. **Grounding dur** : 0 info trouvée → le dire, ne jamais compléter par connaissance générale.
2. **Cohérence conversationnelle** : les suivis elliptiques restent sur le sujet ; un changement
   de sujet est détecté et repart proprement.
3. **Retrieval maximal** : retrouver le(s) bon(s) documents/passages et packer un maximum de
   contexte utile (CAG + pinning par code).
4. **Réponse cadrée** : répondre à LA question posée — sujet exact, format adapté, zéro
   information superflue, zéro « fiche bonus » sur une référence citée en passant.

---

## 1. État actuel (cartographié le 21/07)

```
POST /chat/stream                                     [chat.py]
 1. Guidé ACTIF (état persistant)      → run_guided_turn        (~l.815)
 2. REGEX fiche technique               → generateur SSE fiche   (~l.851)   ← court-circuite tout
 3. Query understanding (LLM fusionné)  → route {direct,rag} + signaux + clarif + guidé (~l.971)
 4. NOUVEAU départ guidé                → run_guided_turn        (~l.1048)
 5. route=direct                        → generateur SSE n°2     (~l.1104)
 6. not ready_for_retrieval             → generateur SSE n°3     (~l.1209)
 7. RAG : retrieval → boosts → ancre → CAG → sandwich → generateur SSE n°4 (~l.1261+)
```

### Défauts structurels

- **D1. Ordre inversé** : le regex fiche (2) décide AVANT la compréhension (3). La décision
  la moins informée a un veto sur la plus informée. Bug type : « SEUILS compatibles avec
  6111 » → fiche 6111 (le mot « seuil » est un marqueur documentaire du regex !).
- **D2. Format figé au routage** : « fiche » est un format de réponse, pas une route. Le
  figer avant d'avoir compris la question et vu les documents produit des réponses
  hors-sujet. Presque toutes les questions sont « proches d'une fiche » → preuve que ce
  n'est pas un critère discriminant.
- **D3. 4 générateurs SSE** (direct, clarification, RAG, fiche) : prompts, relais thinking,
  persistance et vérification divergents. `verify_response` n'est pas branché partout.
- **D4. Combinatoire de flags** : QUERY_UNDERSTANDING × FICHE_TECHNIQUE × GUIDED_FLOW ×
  CONVERSATION_ANCHOR → 16 chemins, fallbacks doublés (decide_retrieval_route seul si QU off).
- **D5. Grounding déclaratif** : le prompt exhorte (« ne réponds pas à côté ») mais le
  générateur ne reçoit AUCUN signal mesuré de ce que la recherche a trouvé.

### Ce qui est bon et reste

`run_lightweight_understanding` (question autonome, signaux, topic_shift, décisions fusionnées
en 1 appel) · ancre documentaire conversationnelle · CAG packer + images alignées + sandwich
anti-lost-in-the-middle · `verify_response` · le traversal KAG propre (post-refonte R1-R6).

---

## 2. Architecture cible

```
POST /chat/stream
 0. Guidé ACTIF          → inchangé (SAV, hors périmètre)
 1. COMPRENDRE           → run_lightweight_understanding (nettoyé)
                           route ∈ {direct, clarify, search} — les 2 seules bifurcations
                           + standalone_question + signaux + topic_shift + refs codes
 2. CHERCHER (search)    → retrieval hybride 4 canaux + ancre conv
                           + si codes détectés : boost lexical + CHUNK PINNING (G1)
 3. PACKER               → CAG + bloc COUVERTURE (signal mesuré)
 4. GÉNÉRER              → générateur UNIQUE, prompt UNIQUE, reasoning HIGH
                           format décidé par le reasoning (fiche/étapes/tableau/court)
 5. VÉRIFIER             → verify_response sur TOUTES les voies
```

**Principe cardinal** : la seule décision prise AVANT la recherche est « faut-il chercher ? »
(direct/clarify/search). Toute décision de FORME est prise APRÈS, par le générateur, qui est
le seul à voir : la question comprise + les documents trouvés + la couverture.

---

## 3. Chantiers

### C1 — La compréhension devient le SEUL routeur

**Fichiers** : `chat.py`, `lightweight_query_understanding.py`.

- Supprimer le fast-path fiche de sa position actuelle (chat.py ~l.851-958) — voir C2 pour
  la relocalisation de sa valeur.
- La route de compréhension devient tri-état : `direct | clarify | search`
  (aujourd'hui : `direct | rag` + clarification portée à part par `ready_for_retrieval`).
  Unifier : `clarify` = l'actuel `not ready_for_retrieval` + clarification présente.
- **Fallback robuste** : compréhension en échec/timeout → `route=search` avec le message brut
  comme standalone_question. Chercher ne coûte presque rien et ne peut pas inventer ;
  c'est le défaut le plus sûr. (Supprime le fallback `decide_retrieval_route` séparé.)
- La compréhension extrait aussi les **références codées** du message (regex `REF_CODE_RE`
  déjà existante côté KAG — réutiliser, pas dupliquer) → `signals.reference_codes`.
  Zéro appel LLM supplémentaire : post-traitement du même passage.

### C2 — La fiche technique cesse d'être une route, ses 2 valeurs sont relogées

**Fichiers** : `fiche_technique_service.py`, `space_search_service.py`, `chat.py`.

- **Valeur 1 — résolution par code → retrieval standard.** `_resolve_passages` (recherche
  posée sur la référence, boost lexical) devient un enrichissement de
  `search_technical_passages` : quand `signals.reference_codes` est non vide, le retrieval
  ajoute une sous-requête par code (le canal BM25/lexical fait déjà remonter les codes
  exacts) et fusionne. Le graphe KAG propre (ref_code) sert de résolveur : code → entité
  → chunks `subject` prioritaires.
- **Valeur 2 — format fiche → règle du prompt unique** (voir C4-b). Le générateur produit
  une fiche structurée quand le reasoning constate un lookup nu — même rendu qu'aujourd'hui,
  mais décidé au bon moment.
- Supprimer : `detect_reference_query` comme aiguillage (la détection de codes reste,
  relogée en C1), le générateur SSE fiche, `FichePrepared`/`prepare_fiche_technique`
  (le packing CAG standard + pinning les remplace).
- `FICHE_TECHNIQUE_ENABLED` : déprécié (la voie unique couvre le besoin).

### C3 — Le bloc COUVERTURE : grounding mesuré, pas déclaré

**Fichiers** : `chat.py` (ou `context_packer_service.py`).

Après retrieval + packing, calculer un mini-rapport factuel injecté EN TÊTE du contexte
système (avant les documents) :

```
### COUVERTURE DE LA RECHERCHE
- Passages pertinents : 14 (score max 0.87)
- Documents couverts : DTA 6/16-2335_V5, Catalogue PROFINE
- Références demandées : 6111 → TROUVÉE (3 passages, dont 1 chunk de référence épinglé)
                          9F67 → ABSENTE du contexte
- Statut : ok | partiel | vide
```

- « Références demandées » croise `signals.reference_codes` avec le contenu réellement packé
  (string-match word-boundary — même regex que le linking lexical R2).
- Règles associées côté prompt (C4-c) : statut `vide` → abstention obligatoire ;
  code ABSENT → le dire explicitement au lieu d'extrapoler depuis un code voisin.
- C'est le mécanisme anti-6110/6111 côté génération : le modèle SAIT si le code demandé
  est dans son contexte.

### C4 — Prompt de génération UNIQUE (le cœur)

**Fichiers** : `chat.py` (SPACE_CHAT_SYSTEM_PROMPT refondu).

Un seul prompt système pour direct/search (le mode direct = même prompt, section documents
vide). Sections, dans l'ordre :

- **(a) SUJET DEMANDÉ** : « Identifie d'abord le sujet exact de la question. Dans une
  question relationnelle ("X compatible avec Y", "quel X pour Y"), le sujet est X ; Y est
  un filtre. Ta réponse porte sur X. Ne produis JAMAIS de présentation non demandée de Y. »
- **(b) FORMAT ADAPTATIF** (remplace la route fiche) :
  - lookup de référence nue (« 6111 », « profil 6111 ») → fiche structurée sourcée ;
  - question relationnelle → liste des X trouvés, avec leurs références et sources ;
  - procédure (« comment poser… ») → étapes ordonnées ;
  - question ponctuelle (une cote, une norme) → réponse en 1-3 phrases, pas de fiche ;
  - comparaison → tableau court.
  « Le format le plus COURT qui répond complètement est le bon. »
- **(c) GROUNDING DUR** : « Le bloc COUVERTURE fait foi. Statut vide → réponds
  exactement : les documents de l'espace ne couvrent pas cette question. Référence marquée
  ABSENTE → dis qu'elle n'est pas documentée ici ; interdiction d'utiliser les valeurs
  d'une référence voisine. Jamais de connaissance générale pour combler un trou factuel. »
- **(d) ANTI-DIGRESSION** : « Aucune information non nécessaire à la question. Pas de
  sections "À noter" hors sujet, pas de caractéristiques non demandées, pas de rappel de
  la fiche complète quand on demande UN attribut. »
- (e) Style/citations : conservé de l'actuel (### STYLE, ### IMAGES, sources).

`GENERATION_REASONING_EFFORT=high` s'applique à cette voie unique (déjà câblé dans
chat_stream_wrapper). Le reasoning a 3 jobs explicites : sujet (a), format (b),
« ai-je de quoi répondre » (c).

### C5 — Générateur SSE unique

**Fichiers** : `chat.py`.

Une seule fonction `generate_answer_stream(...)` paramétrée :
- `context_docs` (None pour direct, packé pour search), historique (élagué si topic_shift),
  images CAG, sandwich final — logique actuelle du chemin RAG conservée ;
- relais `thinking` systématique (bulle UI) ;
- `verify_response` systématique avant persistance (aujourd'hui absent des voies
  direct/fiche) ; clarification = un yield simple (pas de LLM), reste un cas trivial à part ;
- persistance/metadata uniformes (message_id, sources, verification).

Supprime les générateurs direct (l.1129) et fiche ; le générateur RAG actuel devient LA
fonction, le direct l'appelle avec contexte vide.

### C6 — Chunk pinning (G1) au packing

**Fichiers** : `context_packer_service.py` (+ `kag_graph_service.py`).

- `select_authority_chunks(session, space_id, ref_code, limit=2)` : chunks liés à l'entité
  du code, `relation_role='subject'` d'abord puis `mention`, bonus « code + unité (mm/kg) »,
  cap ~1 200 caractères.
- Injectés en tête du contexte sous `### EXTRAITS DE RÉFÉRENCE — <code> (source : doc, p.N)`,
  texte VERBATIM. Le bloc COUVERTURE mentionne l'épinglage.
- Dépend du graphe retraité (R1-R6 livrés) ; dégrade proprement : pas d'entité → pas de bloc.

### C7 — Nettoyage

- `decide_retrieval_route` : ne reste que comme nœud interne de la compréhension
  (suppression du fallback autonome dans chat.py).
- Flags : `FICHE_TECHNIQUE_ENABLED` déprécié ; `QUERY_UNDERSTANDING_ENABLED` reste mais son
  chemin « off » se réduit à : route=search + message brut (plus de pipeline parallèle).
- Supprimer le code mort après migration (générateur fiche, ancien prompt direct).

---

## 4. Ordre d'implémentation

```
1. C4 prompt unique + C3 couverture      (le cœur, testable seul sur la voie RAG actuelle)
2. C5 générateur unique                  (fusion direct → RAG paramétré, verify partout)
3. C1 routeur tri-état + fallback search
4. C2 retrait de la route fiche + migration résolution par code dans le retrieval
5. C6 chunk pinning                      (après retraitement KAG de l'espace — bouton R6)
6. C7 nettoyage flags + code mort
```

Chaque étape est shippable indépendamment ; 1-2 puis 3-4 forment deux PR cohérents.

## 5. Validation (avant/après sur les mêmes questions)

| Question | Attendu après refonte |
|---|---|
| « parle moi en détail du profil 6111 » | Fiche structurée (décidée par le reasoning), 155 mm, sourcée |
| « LES SEUILS compatibles avec 6111 » | Liste de SEUILS (9F67, 9F68, Z043…) ; PAS de fiche 6111 |
| « quelle est la longueur du 6111 ? » | 1-3 phrases : 155 mm + source. PAS de fiche complète |
| « bonjour » | Direct, poli, zéro recherche |
| « garantie des vitrages teintés or » (hors corpus) | Abstention explicite (couverture vide), zéro invention |
| Suivi : « et ses dimensions ? » | Reste sur le sujet courant (ancre + standalone_question) |
| « profil 9999 » (code inexistant) | « 9999 → ABSENTE » → le dit, ne recycle pas le 6111 |
| Golden set space 28 (20 questions) | Doc-recall ≥ actuel ; réponses « answers_question » ↑ au juge |

Mesures : les 2 métriques de `verify_response` (grounded, answers_question) loggées par
message + l'éval CAG existante (« Qualité RAG ») pour la partie retrieval.

## 6. Risques & garde-fous

- **Régression lookup fiche** (« profil 76180 » doit rester excellent) : le format fiche est
  dans le prompt (C4-b) + le pinning ramène les chunks du code → à valider explicitement
  dans le tableau ci-dessus AVANT de supprimer l'ancien chemin (C2 après C4/C5).
- **Latence** : reasoning high sur toutes les réponses ; mitigé par le streaming du thinking
  (déjà en place, l'utilisateur voit l'activité) et GENERATION_REASONING_EFFORT réglable.
- **Prompt unique trop long** : les sections (a)-(d) remplacent MÉTHODE actuelle, elles ne
  s'y ajoutent pas — viser une taille équivalente.
- **Compréhension en échec** : fallback route=search (jamais de mur), à tester en coupant
  la clé API du nœud compréhension.
