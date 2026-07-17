# Plan opérationnel — Génération mistral-small + reasoning, pipeline en 6 phases

_Date : 2026-07-17 · Branche : `fix/retriever`. Concrétise
`plan_generation_agentic_2026-07-17.md` (plan de pensée) avec les décisions actées :
**génération sur `mistral-small-latest` + `reasoning_effort`**, température 0.2-0.3,
HyDE optionnel. Le pipeline par message passe de 2 étapes LLM à un flux en 6 phases
calqué sur les 6 temps d'un SAV humain._

**Décisions actées** : génération = small + reasoning (un seul modèle sur tout le
pipeline = ops/coût simplifiés) · température unifiée 0.2-0.3 · HyDE sélectif ·
le golden answer-level reste l'arbitre de chaque choix.

---

## Vue d'ensemble — le flux par message

```
message utilisateur
  │
  ├─ PHASE 1 · COMPRENDRE  (small, reasoning adapté)
  │    intent enrichi + gamme/attributs + correction typos + ambiguïté bloquante ?
  │    ├─ ambiguïté bloquante → QUESTION DE CLARIFICATION (fin du tour, pas de retrieval)
  │    └─ sinon → requêtes optimisées (+ HyDE si signaux faibles)
  │
  ├─ PHASE 2 · CHERCHER  (0 LLM — retrievers existants, mieux alimentés)
  │    4 retrievers parallèles → RRF → reranker multilingue (CR2)
  │
  ├─ PHASE 3 · JUGER  (gate dur gratuit + small reasoning="high" si zone grise)
  │    RÉPONDRE / DEMANDER / ABSTENIR-ESCALADER / MONTRER
  │    + sélection des documents/passages UTILES (le jugement élit le contexte)
  │
  ├─ PHASE 4 · RÉDIGER  (small + reasoning, temp 0.2-0.3, contexte élu et budgété)
  │    ThinkChunk masqué · contrat answer-first · profil interne/client
  │
  ├─ PHASE 5 · VÉRIFIER  (programmatique d'abord, LLM ensuite)
  │    refs/cotes de la réponse ∈ passages ? répond à la question ? → 1 retry max
  │
  └─ PHASE 6 · LIVRER  (stream + sources + image proactive + suite proposée)
```

**Fast path** : question claire + retrieval confiant (~70-80 % des messages) → les
phases 3-4-5 fusionnent en UN appel (le reasoning fait juger-rédiger-vérifier dans le
même ThinkChunk). Coût : 2 appels LLM/message, comme aujourd'hui.
**Chemin délibéré** : zone grise, ambiguïté, multi-étapes → phases séparées, 3-4 appels.
Le routeur = signaux de la phase 1 + score post-rerank de la phase 2.

---

## PHASE 1 — Comprendre la question AVANT de chercher

_Objectif : ne plus jamais lancer un retrieval sur une question mal comprise — la
clarification arrive AVANT la recherche, pas après 4 retrievers et un packing._

Étend l'appel fused understanding existant (`lightweight_query_understanding`, déjà sur
small) — un seul appel, sortie structurée enrichie :

1. **Intent typé** : `lookup_reference` / `procédure` / `diagnostic_sav` / `comparaison`
   / `demande_visuelle` / `hors_domaine` / `conversationnel`. (Le routage guidé/chat
   existe ; on le précise et il pilote désormais le budget de packing ET le routeur
   fast/délibéré.)
2. **Extraction cible** : gamme/produit, référence(s), attributs discriminants
   (« invisible », « 24 mm », « semi-fixe ») — reportés dans l'ancre conversationnelle
   (CR5) pour ne plus les perdre entre les tours.
3. **Correction de typos** (CR3) : passe trigram contre `KnowledgeEntity.name_normalized`
   AVANT tout : « houssete »→« houssette », « loqueteaiu »→« loqueteau ». Mention
   discrète dans la réponse finale.
4. **Test d'ambiguïté bloquante** : la réponse dépend-elle d'un paramètre absent
   (gamme non précisée alors que l'espace en couvre plusieurs) ? Si OUI et que ça change
   la réponse → **poser LA question maintenant** (1 question max, avec les options
   listées) et FIN DU TOUR. On économise tout le retrieval/packing d'une réponse fausse.
   Si l'ambiguïté est non bloquante → continuer en la notant pour la phase 3.
5. **HyDE sélectif** (« réfléchir à la réponse voulue ») : si les signaux lexicaux sont
   faibles (pas de référence détectée, requête courte/jargon), le modèle rédige en 2-3
   phrases la **réponse hypothétique idéale** (« La houssette mécanique est un patin
   d'étanchéité référencé XXXX, monté sur… »).
   - Usage 1 : embeddée et **combinée** à l'embedding de la question (moyenne ou double
     requête) pour le canal dense UNIQUEMENT — jamais le lexical (une hallucination de
     HyDE ne doit pas polluer BM25/KAG).
   - Usage 2 : conservée comme **« attente de réponse »** pour la phase 5 (la réponse
     finale doit traiter le même objet).
   - Gaté : PAS de HyDE quand une référence exacte est détectée (le lexical suffit et
     HyDE peut dériver).

`reasoning_effort` : `"none"` par défaut (latence) ; `"high"` si le message est long,
multi-questions ou contradictoire avec l'historique.

## PHASE 2 — Chercher (structure inchangée, mieux alimentée)

Aucun nouveau composant. Les 4 retrievers parallèles + RRF restent ; ils reçoivent
désormais : question condensée corrigée (dense) + mots-clés/références en OR avec boost
de phrase (BM25, cf. audit §5) + entités extraites (KAG, cf. patch 1.1) + embedding
HyDE combiné (dense, si généré). **Prérequis absolu : CR2 (reranker multilingue +
neutralisation du boost +0.8)** — tout le jugement de la phase 3 s'appuie sur ce score.

## PHASE 3 — Juger avant de parler (le cœur du système)

1. **Gate dur (gratuit, déterministe)** sur le score post-rerank :
   - fort → `RÉPONDRE` (fast path, phases 3-5 fusionnées dans l'appel de rédaction)
   - faible → `ABSTENIR-ESCALADER` d'office (on n'entre jamais en mode affirmatif sur du
     bruit ; suggestions trigram « vouliez-vous dire… »)
   - intermédiaire (zone grise) → délibération LLM ↓
   - `demande_visuelle` (phase 1) → `MONTRER` (pipeline illustration/page, CR1)
2. **Délibération** (small, `reasoning_effort="high"`, ~top-10 passages seulement — PAS
   le packing entier) : « Ai-je de quoi répondre ? À quelle question exactement ?
   Quels passages sont utiles (ids) ? Manque-t-il une info que l'utilisateur doit
   fournir ? La gamme est-elle confirmée par les en-têtes des passages ? »
   → sortie structurée : décision + **liste des passages/documents élus** + plan de
   réponse en 2-3 points.
3. **Le jugement élit le contexte** : seuls les documents des passages élus partent au
   packing CAG (dédup L2, budget par intent). C'est la discipline de contexte réalisée
   par le raisonnement plutôt que par une heuristique — la réponse ne sera plus enterrée.

## PHASE 4 — Rédiger

- **Modèle** : `mistral-small-latest`, `reasoning_effort="high"` sur le chemin délibéré,
  `"none"` sur le fast path simple (lookup direct où le contexte est évident).
- **Température** : cible 0.2-0.3, unifiée dans les 3 sources (`.env` 0.7 / compose 0.55 /
  `config.py` 0.3 — réconcilier, c'est un bug de config connu). ⚠ Point à vérifier :
  la reco d'échantillonnage Mistral pour les appels AVEC reasoning (les modèles à
  thinking recommandent parfois une température plus haute pour la phase de réflexion) —
  suivre la doc pour les appels `"high"`, imposer 0.2-0.3 sur les appels sans reasoning ;
  le déterminisme final est de toute façon garanti par la phase 5, pas par la température.
- **Contexte** : les documents élus en phase 3, packés avec budget (par intent), dédup
  L2, cible marquée en tête (« L'utilisateur cherche : … »). Le 256K annoncé donne de la
  marge, mais on budgète quand même : l'attention se dilue bien avant la limite, et les
  tokens packés = le coût et la latence par message. (Vérifier au passage la fenêtre
  réelle de la version de small servie — elle a varié selon les versions.)
- **Contrat de réponse** (prompt système actuel conservé, il est bon) + 1re phrase =
  réponse directe + interdiction de nier la capacité images (CR1.3).
- **ThinkChunk** : parser le flux, NE JAMAIS streamer le `type:"thinking"` comme réponse.
  V1 : masqué (loggé dans les traces §2.2 — or de diagnostic). V2 : repliable dans l'UI
  (« voir le raisonnement ») — différenciant pour le profil interne.
- **Profils** : interne (dense, candidats multiples OK) vs SAV client (guidé, escalade
  nette, critique de sortie obligatoire) — mêmes phases, contrats/seuils différents.

## PHASE 5 — Vérifier avant d'envoyer (anti-hallucination)

Trois contrôles, du gratuit au payant :

1. **Vérification programmatique (0 LLM, systématique)** : extraire de la réponse toutes
   les références (regex du motif élargi CR1) et valeurs numériques avec unité (cotes,
   mm, kg) → chacune doit apparaître **telle quelle** dans les passages fournis.
   Une réf/cote absente = violation. C'est le filet le plus fiable contre l'hallucination
   de références — déterministe, imparable, gratuit.
2. **Auto-contrôle dans le reasoning (déjà payé)** : dernière consigne du prompt —
   « vérifie que ta 1re phrase répond à la question posée ; que chaque affirmation
   technique est traçable à un passage ; sinon corrige ou signale le doute ».
3. **Contrôle d'adéquation (LLM léger, chemin délibéré et profil client)** : la réponse
   traite-t-elle l'objet attendu (comparaison avec le plan de réponse de la phase 3 et/ou
   l'attente HyDE de la phase 1) ? `chat_critique_service` généralisé joue ce rôle en
   sortie du profil client.

**Politique d'échec : 1 retry max.** Violation détectée → regénérer UNE fois avec le
constat injecté (« la réf X n'apparaît dans aucun passage — retire-la ou reformule ») ;
si la violation persiste → réponse d'abstention honnête (« je ne peux pas le confirmer
dans les documents ») + escalade selon profil. Jamais de 2e boucle (latence).

## PHASE 6 — Livrer et proposer la suite

- Stream de la réponse finale (après le gate de la phase 5 — le stream démarre donc
  après vérification ; le TTFB est couvert par les événements SSE de statut des phases
  1-5 : « je vérifie la référence… », le front les gère déjà).
- Sources + **image proactive** (profil client : si une page citée contient un schéma
  pertinent, la joindre sans attendre la demande — CR1) ; sur demande côté interne.
- **Proposer la suite** (temps 6 du SAV) : « Voulez-vous le schéma de montage ? / la
  page du catalogue ? » — piloté par l'intent et ce que le jugement a vu passer.
- Capture du feedback 👍/👎 → boucle golden (§2.3 plan cas réels).

---

## Chantiers d'implémentation (ordre)

| # | Chantier | Dépend de | Effort | Validation |
|---|---|---|---|---|
| 0 | Golden answer-level en CI (l'arbitre de tout) | — | 2 j | baseline chiffrée |
| 1 | Température unifiée 0.2-0.3 + `GENERATION_MODEL` dédié en config | — | 2 h | 3 sources alignées |
| 2 | CR2 : reranker multilingue + neutraliser boost +0.8 | 0 | 1-2 j | golden retrieval |
| 3 | Phase 5.1 : vérification programmatique refs/cotes | 0 | 1 j | 0 réf inventée sur golden |
| 4 | Bascule génération small+reasoning + parsing ThinkChunk | 0,1 | 2 j | golden A/B vs large actuel |
| 5 | Phase 1 enrichie : clarification précoce + typos + HyDE gaté | 0,2 | 3-4 j | cas #6/#7 verts |
| 6 | Phase 3 : gate dur + délibération + contexte élu | 2,4 | 1 sem | cas #1/#8 verts |
| 7 | Phase 5.3 + profils interne/client | 4,6 | 3-4 j | golden par profil |
| 8 | Phase 6 : images proactives + suite proposée | CR1 | 2-3 j | cas #4/#5/#9 verts |

L'étape 4 (bascule modèle) se fait **tôt et isolément** : on mesure small+reasoning à
architecture constante avant d'attribuer les gains aux phases. Si small décroche en
rédaction sur le golden → repli acté : small pour juger (phases 1/3/5), medium-3-5 pour
rédiger (phase 4), sans rien changer d'autre au plan.

## Budget latence/coût par message (ordre de grandeur)

| Chemin | Appels LLM | vs aujourd'hui |
|---|---|---|
| Clarification précoce (phase 1 stoppe) | 1 | moins cher (pas de retrieval) |
| Fast path | 2 (compréhension + rédaction fusionnée) | identique, contexte réduit → moins cher |
| Délibéré | 3-4 (+ délibération, + critique client) | +1 à 2 appels small courts ; compensés par le packing élu (÷2-3 tokens) |

## Risques et parades

- **Small rédige moins bien que large** (nuance, ton) → mesuré à l'étape 4 sur le golden ;
  repli split juge/rédacteur sans toucher au reste.
- **Sur-clarification** (assistant qui questionne trop, cf. mode guidé Askey) → 1 question
  max par conversation hors mode guidé ; seulement si l'ambiguïté change la réponse.
- **HyDE qui dérive** (hypothèse fausse ancre le dense sur le mauvais contenu) → gaté
  (jamais avec référence exacte), combiné à l'embedding question, canal dense seulement.
- **ThinkChunk qui fuit dans l'UI** → test dédié au parsing du stream (chunk `thinking`
  jamais émis vers le client). Mécanique concrète du reasoning Mistral : `content` passe
  de string à LISTE de chunks typés (`{type:"thinking",thinking:[...]}` + `{type:"text"}`) ;
  en streaming, `delta` est une LISTE pendant la réflexion puis une STRING pour la réponse
  → aiguiller `isinstance(delta, list)` (log/statut) vs string (stream client). Impacts
  code : `mistral_service.py` (parsing), streaming `chat.py`, persistance conversation.
- **Piège multi-tours (reasoning)** → Mistral impose de REJOUER le message assistant
  complet, ThinkChunk INCLUS, dans l'historique ; le retirer dégrade nettement la qualité
  aux tours suivants. La persistance de conversation doit donc stocker et renvoyer les
  ThinkChunks → coût qui s'accumule sur un long fil SAV. Parades : `high` seulement sur le
  chemin délibéré (pas chaque tour) ; troncature de l'historique par fenêtre, pas par
  suppression des thinking.
- **Le ThinkChunk n'est PAS un journal fidèle** du calcul du modèle (c'est une
  génération) → il peut raisonner juste et conclure faux. La phase 5 (vérification
  programmatique des refs/cotes) reste indispensable MÊME avec reasoning ; on ne fait pas
  confiance au brouillon, on vérifie la sortie.
- **Reasoning + vision** (images de page dans le message) → à confirmer sur l'API dès
  l'étape 4 ; si incompatible, les pages ColPali-only passent par une description texte
  préalable.

---

## Annexe A — Trace concrète bout-en-bout

_Cas réel qui échouait aujourd'hui (gamme 70 : « houssete » non corrigé → échec ; puis
« montre-moi le dessin » → rien). Espace = « Gamme 70 » (interne PROFERM). Valeurs
illustratives mais réalistes._

### Config par phase

| Phase | Modèle | reasoning_effort | temp | appel LLM ? |
|---|---|---|---|---|
| 1 Comprendre | mistral-small | none (msg court) | 0.2 | 1 |
| 2 Chercher | — | — | — | 0 |
| 3 Juger | small (si zone grise) | high | — | 0 ou 1 |
| 4 Rédiger | mistral-small | high | 0.25 | 1 (fusionne 3+5 en fast path) |
| 5 Vérifier | regex puis small | — | — | 0 (+1 si retry) |
| 6 Livrer | — | — | — | 0 |

---

### TOUR 1 — l'utilisateur tape : `houssete`

**PHASE 1 · COMPRENDRE** — `small`, `reasoning_effort=none`, sortie structurée
```json
// entrée : {message:"houssete", space:"Gamme 70", history:[]}
// correction typo (trigram vs KnowledgeEntity.name_normalized) AVANT tout :
//   "houssete" → "houssette" (similarity 0.89)
// sortie structurée :
{
  "intent": "lookup_reference",
  "corrected_query": "houssette",
  "typo_fixed": {"from": "houssete", "to": "houssette"},
  "target": {"gamme": "70", "reference": null, "attributs": []},
  "ambiguite_bloquante": false,          // espace déjà scopé gamme 70
  "hyde_declenche": true,                // 1 seul terme, jargon, aucune réf → HyDE
  "hyde": "La houssette est un patin d'étanchéité souple inséré dans une rainure du
           profilé pour assurer l'étanchéité entre ouvrant et dormant ; elle porte une
           référence de type XXXX.",
  "queries": {
     "dense": "embed('houssette étanchéité gamme 70')  ⊕  embed(hyde)",  // combiné
     "bm25":  "houssette",                                                // lexical PUR
     "kag":   ["houssette"]
  }
}
```
> Le HyDE n'alimente QUE le canal dense. BM25/KAG ne voient jamais l'hypothèse (une
> invention du HyDE ne doit pas polluer le lexical). L'hypothèse est aussi gardée comme
> « attente de réponse » pour la phase 5.

**PHASE 2 · CHERCHER** — 4 retrievers ∥ → RRF → reranker `bge-reranker-v2-m3`
```
rang  score  source
[1]   0.83   DTD Gamme 70 Plateforme, p.34 : "...patin d'étanchéité (houssette)
             réf. 9718.3 pour le profil 6127..."
[2]   0.58   DTD Gamme 70, p.12 : "...joints périphériques du dormant..."
[3]   0.24   Doc Gamme 76, p.8 : "...étanchéité ouvrant..."   ← autre gamme
```

**PHASE 3 · JUGER** — gate dur (gratuit)
```
best_score 0.83 > seuil_fort 0.55  →  décision = RÉPONDRE (fast path)
passages élus = [1] (+[2] contexte)   [3] écarté : 0.24 < plancher ET gamme ≠ 70
→ phases 3-4-5 FUSIONNÉES dans l'appel de rédaction (pas de délibération séparée)
```

**PHASE 4 · RÉDIGER** — `small`, `reasoning_effort=high`, `temp=0.25`, contexte = [1],[2] dédupés, cible marquée
```
── ThinkChunk (type:"thinking" — MASQUÉ à l'utilisateur, loggé dans les traces) ──
"houssete → houssette. Gamme 70. Passage [1] : houssette = patin d'étanchéité
 réf. 9718.3, profil 6127 (DTD p.34). Passage [2] = joints périphériques, sujet
 différent → ne pas mélanger. Aucun autre profil cité. Vérif interne : '9718.3' et
 '6127' sont bien dans [1]. 1re phrase = la réponse. Proposer le schéma (intent lookup
 + composant illustrable)."

── TextChunk (type:"text" — LA réponse streamée) ──
"Sur la gamme 70 Plateforme, la houssette — j'ai compris « houssette » — correspond au
 patin d'étanchéité référence 9718.3, prévu pour le profil 6127. Il s'insère dans la
 rainure du profilé pour assurer l'étanchéité de l'ouvrant.
 Souhaitez-vous le schéma de montage ?"
```

**PHASE 5 · VÉRIFIER** — programmatique, avant flush du stream
```
refs extraites de la réponse  : ["9718.3"]      → "9718.3" ∈ passage[1] ?  ✓
codes/profils extraits        : ["6127"]        → "6127"   ∈ passage[1] ?  ✓
cotes numériques (mm/kg)      : []              → rien à vérifier
VERDICT : PASS — aucun retry
```

**PHASE 6 · LIVRER**
```
stream réponse ✓
sources UI : [DTD Gamme 70 Plateforme — page 34]
suite proposée : "schéma de montage ?"  (déjà dans la réponse)
feedback 👍/👎 armé
```

---

### TOUR 2 — l'utilisateur tape : `montre moi le dessin`

**PHASE 1** — `intent = demande_visuelle`. Cible **héritée de l'ancre** (CR5) :
`{houssette, réf 9718.3, profil 6127, gamme 70, doc=DTD p.34}`. Pas d'ambiguïté, pas de HyDE.
**PHASE 2** — léger : on réutilise la page ancrée (p.34) + pages contenant `6127`/`9718.3`.
**PHASE 3** — `demande_visuelle` → décision = **MONTRER**.
**PHASE 4 (illustration, CR1)** —
```
illustration_service(target_codes=["9718.3","6127"], page=34)
  motif élargi CR1 : "9718.3" ✓ et "6127" ✓ matchés comme labels de code
  → crop Voronoï ancré sur 6127/9718.3, contrôle anti-fuite, garde vision
  ├─ ancrable → crop détouré servi
  └─ non ancrable → FALLBACK CR1.2 : page 34 entière (page_cache) + bandeau
                    "page 34 — DTD Gamme 70"     (au lieu de l'abstention actuelle)
```
**PHASE 5** — la page servie est bien la page citée (34) ✓.
**PHASE 6** — image + légende : *« Voici le patin d'étanchéité 9718.3 sur le profil 6127
(page 34 de la DTD Gamme 70). »*

---

### Avant / après (même entrée)

| | Système actuel | Pipeline 6 phases |
|---|---|---|
| `houssete` | échec (typo non corrigée) | corrigé → patin **9718.3**, profil **6127** |
| `montre le dessin` | « je n'ai pas accès » / rien | crop ou page 34 servie |
| réf. inventée ? | possible (temp 0.7, pas de contrôle) | **impossible** (phase 5 : 9718.3/6127 ∈ passage sinon retrait) |
| gamme 76 (score 0.24) | pouvait fuiter dans le contexte | écartée (plancher + en-tête) |

---

### Variante — l'ambiguïté bloque AVANT la recherche

_Même terme, mais dans un espace multi-gammes (« SAV général »)._ L'utilisateur tape
`quelle houssette pour mon coulissant ?`

**PHASE 1** détecte : `intent=lookup_reference`, `target.gamme=null`, et surtout
`ambiguite_bloquante=true` (la réf houssette diffère selon la gamme, l'espace en couvre
plusieurs). **Le tour s'arrête ici — aucun retrieval, aucun packing.** Réponse :

> « Pour vous donner la bonne référence de houssette, sur quelle gamme êtes-vous —
> Gamme 70, Kömmerling 76, ou une autre ? »

Coût : **1 appel LLM court**, zéro retrieval gaspillé sur une réponse qui aurait été
fausse. C'est le « comprendre avant de chercher » — la clarification précède la
recherche au lieu de la suivre.
