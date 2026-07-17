# Plan d'amélioration piloté par les cas réels — 17 juillet 2026

_Branche : `fix/retriever`. Complète `docs/plan_patch_kag_categories_chunk_2026-07-17.md`
(périmètre patch acté : vision conservée, pas d'OCR 4, pas de refonte retriever/config)
en le croisant avec 9 conversations réelles (Technal, Askey, Soprofen, Kömmerling,
gamme 70, SAV). Les cas réels révèlent des priorités que l'audit code ne voyait pas —
notamment les images, demande utilisateur n°1, absente des deux plans précédents._

---

## 1. Diagnostic des conversations réelles

### Conversation → cause racine → couverture par le plan de patch actuel

| # | Conversation | Symptôme observé | Cause racine (vérifiée code) | Couvert par le patch ? |
|---|---|---|---|---|
| 1 | Crémone 4 points (Technal) | La bonne réponse (TGY3704) est **enterrée** au milieu d'une fiche technique de la 3 points que l'utilisateur connaît déjà | Packing CAG de docs entiers + chunks L2 dupliqués dans le prompt ; aucune consigne « réponds d'abord à la question » | Partiel (dédup L2, Phase 2) |
| 2 | Réglage roulettes (Askey, mode guidé) | 5 tours de questions avant toute aide ; citation `[Doc, page 102]` identique à chaque tour (suspecte) ; changement d'intention (« comment démonter l'ouvrant ») → réponse générique probablement non sourcée | Mode guidé 100 % dynamique (`guided_flow_service.py`, Phase 1 du guidage) : pas de budget de questions, pas de vérification de citation, sortie de mode non gérée | ❌ Non |
| 3 | Embouts galandage (Technal) | ✅ Bonnes réponses, citations précises | — cas de référence à protéger | (non-régression) |
| 4 | Tulipe TMX13 (Soprofen) | « As-tu un visuel » → rien, alors que la page existe en cache PNG | Motif d'illustration `\b\d{3,5}[A-Za-z]?\b` (`config.py:221`) ne matche pas TMX13 ; abstention stricte sans code ancrable (`illustration_service.py:18`) ; aucun fallback « servir la page citée » | ❌ Non |
| 5 | Gondage paumelle (Technal) | « MONTRE-MOI LES IMAGES » → le modèle répond « je n'ai pas accès aux images » (faux : le produit sait servir des crops) puis **dérive** vers LUMEAL GA hors sujet ; « SOLEAL FY » → répond paumelles standard alors que le fil portait sur les paumelles **invisibles** | (a) Le prompt de génération n'informe pas le modèle de la capacité illustration → il la nie ; (b) l'ancrage conversationnel garde le document mais perd la contrainte d'attribut (« invisible ») au changement de gamme | ❌ Non |
| 6 | « loqueteaiu » (Kömmerling 76) | Typo d'un composant → réponse sur les **poignées** (hors sujet, confiant) | Aucune correction de typo côté requête ; reranker ANGLAIS incapable de juger la pertinence FR ; boosts confiance saturés masquent le hors-sujet | Partiel (KAG 1.1/1.3) |
| 7 | « houssette / houssete » (gamme 70) | `houssete` (1 lettre) → échec ; `houssette` → succès. Puis « montre-moi le dessin » → rien | Idem #6 pour le fuzzy ; images cf. #4 | Partiel / ❌ |
| 8 | « trouve moi la houssette » (Kömmerling 76) | Répond **poignées Sécustik** — mauvais composant, aucune expression de doute | Pas d'abstention calibrée : retrieval vide de pertinence → le reranker anglais + boosts font remonter n'importe quoi et le modèle brode | ❌ Non |
| 9 | « photo houssette mécanique » (SAV) | Renvoi vers le service technique | Images cf. #4 + vocabulaire cf. #7 | ❌ Non |

### Synthèse — 5 axes de douleur réels, par fréquence

1. **Images / visuels (4 conversations sur 9)** — demande n°1, capacité codée à ~70 %
   (`illustration_service.py` + `page_cache/`) mais inatteignable : mauvais motif de
   référence + abstention stricte + le modèle nie la capacité. *Absent de l'audit et du patch.*
2. **Robustesse vocabulaire** (typos, variantes) — 3 conversations. Le patch KAG (1.1,
   1.3, 1.4) traite le côté index ; il manque le côté **requête** (corriger « houssete »
   avant retrieval).
3. **Hors-sujet confiant / dérive** — 3 conversations. Cause principale : reranker
   anglais + absence d'abstention. *Exclu du périmètre patch — à ré-arbitrer, voir §3.*
4. **Réponse indirecte** (dump au lieu de répondre) — cas #1, probablement systémique.
5. **Mode guidé SAV** — trop de questions, citations non vérifiées, sortie de mode ratée.

---

## 2. Dispositif « voir ce qui pèche en cas réel » (à monter en PREMIER)

Le patch prévoit un « filet de sécurité minimal » sur la résolution d'entités ; les cas
réels justifient plus large. Trois étages, du moins cher au plus cher :

### 2.1 Golden conversationnel (S1, ~2 j)
- Convertir **ces 9 conversations** en cas d'éval versionnés à côté du golden retrieval
  existant (`tests/fixtures/golden/`). Par cas : question(s), gamme, réponse attendue
  (référence exacte ou abstention), pages sources attendues, type d'échec historique.
- Brancher `retriever_evaluator.py` (recall/MRR) en pytest/CI — il existe, il n'est
  accessible que par endpoint admin (`admin.py:783`).
- Ajouter un étage **answer-level** : un juge LLM (mistral-small) note chaque réponse
  générée sur 4 axes binaires — *répond à la question posée*, *citation valide (la page
  citée contient l'info)*, *abstention correcte si l'info est absente*, *image servie si
  demandée*. Les cas #1/#5/#8 auraient un retrieval « correct » aux métriques MRR et une
  réponse pourtant inutilisable : le retrieval-only ne suffit pas.
- Cas #3 = non-régression obligatoire à chaque phase.

### 2.2 Traces de production exploitables (S1-S2, ~2 j)
- Log JSON structuré par message : requête, route, top-10 par retriever avant fusion,
  top-5 après rerank, boosts appliqués, docs packés, images servies/refusées **et
  pourquoi**. Endpoint admin « rejouer ce message ». (La matière existe en `logger.info`
  épars — la structurer.)
- Feedback 👍/👎 + motif (mauvaise réponse / pas d'image / mauvaise gamme / trop long).
  `feedback_knowledge_service.py` existe et n'est branché nulle part — c'est son point
  d'entrée, et le patch KAG 1.4 prévoit déjà d'en faire une source d'alias.

### 2.3 Rituel hebdo
- Extraire les 👎 et les conversations à reformulations répétées (signal de frustration —
  cf. l'utilisateur en majuscules du cas #5), rejouer, **promouvoir les échecs reproduits
  en cas golden**. Le jeu d'éval grandit avec la prod au lieu de rester figé.

---

## 3. Chantiers cas-réels (en plus du plan de patch)

### CR1 — Débloquer les images (cas #4, #5, #7, #9 — priorité n°1, ~2-3 j)
1. **Motif de référence** : étendre `ILLUSTRATION_REFERENCE_PATTERN` aux références
   alphanumériques : `\b(?:[A-Z]{1,4}[- ]?\d{2,6}[A-Za-z]{0,3}|\d{3,5}[A-Za-z]?)\b`
   (TGY3702, TMX13, T910002…). ⚠ Le motif sert aussi aux labels frères du crop Voronoï :
   trop large = schémas fragmentés → valider sur le golden avant merge. À terme
   (Phase 4 config du patch), le motif devient réglable par espace (les familles de
   références diffèrent par fournisseur).
2. **Fallback page entière** : quand l'intention visuelle est détectée (les marqueurs
   `_COLPALI_VISUAL_MARKERS` de `space_search_service.py:43` le font déjà) et qu'aucun
   code n'est ancrable, servir la **page PNG citée entière** (déjà dans `page_cache/`)
   avec bandeau « page N de <doc> », au lieu de l'abstention stricte. C'est exactement
   ce que l'utilisateur du cas #5 réclamait.
3. **Informer le modèle** : le prompt de génération doit décrire la capacité
   (« une illustration de la page citée peut être jointe ; ne JAMAIS dire que tu n'as
   pas accès aux images ; si on te demande un visuel, cite la page qui le contient »).
   Corrige le déni du cas #5.
4. **Requête visuelle sans code** (« photo houssette mécanique ») : résoudre le terme
   via le retrieval texte → meilleure page → servir page/crop.

### CR2 — Reranker multilingue + abstention (cas #6, #8 — à ré-arbitrer, ~1-2 j)
Le périmètre patch excluait la « refonte retriever ». Les cas #6/#8 montrent que le
pire mode d'échec (hors-sujet confiant) vient directement du reranker anglais. Or le
correctif n'est **pas une refonte** :
- Swap de modèle : `cross-encoder/ms-marco-MiniLM-L-6-v2` → `BAAI/bge-reranker-v2-m3`
  (une ligne de config + recalibration des seuils sur le golden). Mesurer la latence CPU ;
  repli `mmarco-mMiniLMv2-L12` si rédhibitoire.
- **Abstention calibrée** : si le meilleur score post-rerank < seuil (calibré sur le
  golden), basculer le prompt en « je n'ai pas trouvé X dans les documents de <gamme> ;
  voulais-tu dire <Y> ? » (suggestions <Y> par trigram sur `KnowledgeEntity`). Prérequis :
  neutraliser le boost `+0.8·confiance` qui écrase le reranker
  (`space_search_service.py:1292`) — déjà affaibli par le patch catégories (fin de la
  confiance 1.0 systémique), à finir ici.
- **Recommandation : intégrer au périmètre.** Sans cela, la canonicalisation KAG du patch
  améliorera le retrieval mais le hors-sujet confiant persistera.

### CR3 — Correction de requête (cas #6, #7, ~1 j)
Le patch KAG 1.1/1.3 canonicalise l'index ; il faut le miroir côté **requête** :
- Passe pg_trgm `similarity() > seuil` de chaque token requête ≥ 4 chars contre
  `KnowledgeEntity.name_normalized` (post-canonicalisation 1.3) : « houssete » →
  « houssette », « loqueteaiu » → « loqueteau », corrigé AVANT retrieval, avec mention
  discrète (« j'ai compris : loqueteau »). S'appuie sur la requête set-based de 1.1.
- Denylist stopwords FR dans la recherche UI (`lexical_search_service.py:53`) — le patch
  la prévoit pour le KAG, l'étendre à l'UI (3 h).

### CR4 — Répondre d'abord + packing propre (cas #1, ~1 j)
- Prompt de génération : « Première phrase = la réponse directe (référence, valeur,
  oui/non). Détails ensuite. »
- Exclure les chunks L2 du packing CAG (`context_packer_service.py:100-103` charge tous
  les `is_leaf=True`, donc les paraphrases L2 en double/triple) — complémentaire de la
  dédup d'enrichissement du patch Phase 2, et gain immédiat de budget tokens.

### CR5 — Ancrage d'attribut conversationnel (cas #5, ~1 j)
L'ancre suit le document/la gamme mais perd les contraintes d'attribut (« paumelle
**invisible** ») au changement de gamme. Reporter les attributs discriminants de la
question initiale dans les `detected_references` de tour en tour ; expiration seulement
au changement explicite de sujet.

### CR6 — Mode guidé SAV (cas #2, ~2-3 j, après CR1-CR5)
- **Budget de questions** : max 3 questions de diagnostic avant un premier élément
  actionnable ; chaque question doit éliminer au moins une branche.
- **Vérification de citation** : une étape ne peut citer une page que si un passage
  récupéré à CE tour la contient (le `[Doc, page 102]` répété doit devenir impossible).
- **Sortie de mode** : détecter le changement d'intention (« comment démonter
  l'ouvrant ») et basculer en réponse documentaire au lieu de rester dans la boucle.
- Accélérer la bascule vers les **arbres validés** (`authored_tree_service`, Phase 2 du
  guidage prévue dans le code) pour les gammes à fort trafic SAV.

---

## 4. Plan consolidé (patch + cas réels)

| Ordre | Chantier | Source | Effort | Jalon mesurable |
|---|---|---|---|---|
| 1 | Golden conversationnel + CI (§2.1) | cas réels | 2 j | 9 cas en CI, baseline chiffrée |
| 2 | Phase 0 patch (nettoyage Docling, candidats, taxonomie morte) | patch | 1 j | pytest vert, surface réduite |
| 3 | **CR1 images** | cas réels | 2-3 j | Cas #4/#7/#9 verts (image servie) |
| 4 | **CR2 reranker + abstention** (ré-arbitrage périmètre) | cas réels | 1-2 j | Cas #8 vert (abstention), #6 amélioré |
| 5 | CR4 réponse directe + packing | cas réels | 1 j | Cas #1 vert (réponse en 1re phrase) |
| 6 | Phase 1.1 patch (requête-par-mot KAG) | patch | 1 j | Bruit KAG éteint |
| 7 | Phases 1.2-1.4 patch (résolution, canonicalisation, alias) | patch | 2-3 j | K76↔Kömmerling 76 fusionnés |
| 8 | **CR3 correction de requête** (dépend de 1.1/1.3) | cas réels | 1 j | Cas #6/#7 verts (typo corrigée) |
| 9 | CR5 ancrage d'attribut | cas réels | 1 j | Cas #5 vert |
| 10 | Phase 1.5 patch (relations dirigées) | patch | 1-2 j | Traversée pondérée par intention |
| 11 | Phase 3 patch (catégories) | patch | 2 j | Reclassification versionnée, confiance calibrée |
| 12 | Phase 2 patch (couverture, gate numérique, dédup L2, IDs stables) | patch | 2-3 j | Couverture loggée, plus d'invention de cotes |
| 13 | §2.2 traces + feedback | cas réels | 2 j | Tout message rejouable, 👎 exploitables |
| 14 | **CR6 mode guidé** | cas réels | 2-3 j | Cas #2 vert (≤3 questions, citations vérifiées) |

**Effort total : ~20-25 j** (patch ~10-12 j + cas réels ~10-12 j). Les items 3-5 sont
avancés avant le cœur KAG parce qu'ils portent l'impact utilisateur le plus visible et
n'en dépendent pas.

**Règle de gate** : aucun item ne se ferme sans (a) golden CI vert, (b) cas #3 vert,
(c) pour les items 6+ : revue d'un échantillon de traces prod de la semaine.

---

## 5. Ce que les cas réels changent aux plans précédents

1. **Les images deviennent le chantier n°1 visible** — 4 conversations sur 9, capacité
   à 70 % codée, absente de l'audit ET du patch.
2. **Le swap reranker + abstention mérite de revenir au périmètre** : ce n'est pas la
   « refonte retriever » écartée (BM25, boosts, retrievers morts restent hors scope),
   c'est un changement de config + un seuil — et c'est la cause du pire mode d'échec
   (hors-sujet confiant, cas #6/#8).
3. **L'éval s'étend de retrieval-only à answer-level** : MRR correct n'empêche pas une
   réponse inutilisable (cas #1, #5, #8).
4. **Le mode guidé SAV entre au périmètre** (absent des deux documents précédents).
5. **La boucle prod → golden est institutionnalisée** (feedback 👎 → cas d'éval), et le
   feedback branche enfin `feedback_knowledge_service` que le patch destinait déjà aux alias.
