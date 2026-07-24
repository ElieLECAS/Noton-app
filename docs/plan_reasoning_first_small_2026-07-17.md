# Plan — Assistant reasoning-first sur mistral-small (comportement Gemini/ChatGPT)

_Date : 2026-07-17 · Branche : `fix/retriever`. Fait suite à
`plan_generation_6_phases_2026-07-17.md`. Décisions utilisateur actées :
**`mistral-small-latest` partout**, **`reasoning_effort="high"`**, **le reasoning
REMPLACE l'étape de query-intention** (plus une extraction JSON séparée), refus humain
si la requête n'a pas de sens, continuité multi-messages garantie._

---

## 0. Ce que j'ai vérifié dans le système actuel (état des lieux)

| Brique | Fichier | Constat | Verdict |
|---|---|---|---|
| État conversationnel | `conversation_state_service.py` | `current_topic` + `focus_entities` persistés dans `Conversation.query_context`, réinjectés en compréhension / retrieval / génération. Fusion propre, reset sur `topic_shift`. | **Solide — on garde** |
| Ancrage documentaire | `chat.py` (`compute_anchor_documents`, `anchor_document_ids` → `build_cag_context`) | Les docs du sujet courant sont toujours packés (continuité même sous 100k tokens). | **Solide — on garde** |
| Historique brut | `chat.py:_load_conversation_context` | Chargé, plafonné `SPACE_HISTORY_MAX_CHARS=16000`. | On garde, à réconcilier avec le replay ThinkChunk (§3) |
| Compréhension fused | `lightweight_query_understanding.py:_node_fused_understand` | 1 appel small JSON : route + signaux + `standalone_question` (condense) + `topic_shift` + `too_vague` + `clarification_question` + `current_topic`. | **C'est CE que le reasoning remplace (§2)** |
| Refus / vagueness | idem (`too_vague`, `clarification_question`) | Le garde-fou EXISTE mais c'est une extraction JSON sans raisonnement → rate en pratique (répond à « loqueteaiu », « houssette » → poignées). | **À refaire en reasoning (§4)** |
| Génération | `rag_generation_service.py`, `chat.py` | `MODEL_FAST` (large en prod), temp 0.7, contexte CAG. | Bascule small+reasoning, temp 0.25 (§5) |

**Conclusion** : la mécanique de continuité est bonne. Ce qui manque, c'est le
*jugement* — remplacer l'extraction JSON par un vrai raisonnement qui comprend, décide
de chercher ou de refuser, puis répond ou s'abstient. C'est précisément ce que font
Gemini/ChatGPT : ils *réfléchissent* avant et pendant, ils ne remplissent pas un
formulaire de signaux.

---

## 1. Le principe cible

Un seul modèle (`small`), deux moments de raisonnement autour du retrieval :

```
message + historique + ÉTAT DU FIL (topic + focus_entities + ancre)
   │
   ▼
┌─ APPEL A · RAISONNEMENT AMONT (small, reasoning="high")  ── REMPLACE le query-intent
│    Le ThinkChunk réfléchit comme un humain :
│      « Que veut-il vraiment ? (résolution des ellipses via l'état du fil)
│        Est-ce que ça a un sens / est-ce assez précis / est-ce mon domaine ?
│        → si NON : je pose UNE question ou je refuse poliment, et je m'ARRÊTE.
│        → si OUI : que dois-je chercher ? »
│    Sortie : soit CLARIFICATION/REFUS (fin du tour), soit REQUÊTES de recherche.
│
├─ RETRIEVAL (inchangé, mieux alimenté par les requêtes du raisonnement)
│
└─ APPEL B · RAISONNEMENT AVAL (small, reasoning="high")
     Le ThinkChunk juge les passages, élit les utiles, décide répondre/abstenir,
     rédige. Le TextChunk = la réponse. Vérification programmatique par-dessus (§6).
```

- **Le reasoning REMPLACE la compréhension JSON** : l'appel A n'extrait plus des champs,
  il *raisonne* et produit soit une clarification, soit des requêtes. Le `topic_shift`,
  le condense (`standalone_question`), la détection de vague deviennent des *conclusions
  du raisonnement*, pas des cases à cocher.
- **2 appels LLM/message** comme aujourd'hui — on ne rallonge pas, on remplace un appel
  JSON par un appel reasoning et l'appel de génération par un appel reasoning.
- Pourquoi 2 appels et pas 1 : le retrieval est au milieu. La version « 1 seul flux qui
  cherche en cours de pensée » (le vrai comportement Gemini/ChatGPT) = **v2 agentique
  §7**, une fois v1 stabilisée.

---

## 2. Appel A — le raisonnement qui remplace le query-intent

Remplace `_node_fused_understand`. `small`, `reasoning_effort="high"`.

**Entrée** : dernier message + historique (fenêtré) + bloc « FIL DE LA CONVERSATION »
(déjà produit par `format_state_facts` / `format_generation_state_block`) + description
de l'espace (gammes couvertes, pour juger domaine et ambiguïté).

**Le raisonnement (ThinkChunk) déroule les temps 1-3 du SAV humain :**
1. Résoudre l'ellipse : « et le TGY3834 ? » / « celui-ci » / « ses dimensions » →
   via `focus_entities` + `current_topic`. Corriger les typos (trigram entités, CR3).
2. Juger le sens : la requête est-elle **compréhensible**, **assez précise**, **dans le
   domaine** ? (cf. §4 pour les 4 verdicts.)
3. Décider :
   - **REFUS/CLARIFICATION** → produire le message (1 question OU refus poli) et
     **STOP** : pas de retrieval, pas de génération. (1 seul appel LLM sur ce tour.)
   - **CHERCHER** → produire les requêtes optimisées (dense condensé + mots-clés/refs
     BM25 en OR + entités KAG + HyDE gaté) et l'intent typé.

**Sortie structurée** (le TextChunk porte un petit JSON, OU function-calling en v2) :
```json
{ "decision": "chercher | clarifier | refuser",
  "message_utilisateur": "...",           // si clarifier/refuser
  "standalone_question": "...",           // si chercher (condense résolu)
  "topic_shift": false,
  "current_topic": "houssette gamme 70",
  "focus_entities_maj": ["houssette","9718.3"],
  "queries": { "dense": "...", "bm25": "...", "kag": ["..."] },
  "intent": "lookup_reference" }
```
→ `build_conversation_state` consomme `current_topic`/`focus_entities_maj`/`topic_shift`
comme aujourd'hui : **l'état du fil continue d'être alimenté**, juste par un raisonnement
au lieu d'une extraction.

**`reasoning_effort`** : `"high"` (le jugement du sens en dépend). C'est le changement de
coût vs l'actuel appel JSON — assumé, c'est là que se gagne le « comme un humain ».

---

## 3. La continuité multi-messages (ce qui existe, ce qu'on ajuste)

La continuité est déjà portée par 3 canaux ; on les conserve et on les branche sur le
raisonnement :

1. **État compact du fil** (`current_topic` + `focus_entities`) — le « de quoi
   parle-t-on » durable, cheap, réinjecté dans l'appel A ET l'appel B. **C'est le
   meilleur mécanisme de continuité de SENS** (mieux qu'un historique brut tronqué).
2. **Ancre documentaire** (`anchor_document_ids`) — les docs du sujet toujours packés en
   génération. On y ajoute (CR5) le report des **attributs discriminants** (« invisible »,
   « 24 mm ») dans `focus_entities`, pour ne plus dériver (cas SOLEAL FY).
3. **Historique brut fenêtré** (16k) — pour les tournures fines.

**Décision sur le replay ThinkChunk (piège Mistral multi-tours)** : Mistral recommande de
rejouer les ThinkChunks dans l'historique. Mais sur un long fil SAV, ça explose le coût.
Arbitrage retenu :
- La **continuité de sens** repose sur l'état compact (canal 1), PAS sur le replay des
  brouillons — c'est notre mécanisme domaine, déjà tuné.
- On **rejoue uniquement le ThinkChunk du DERNIER tour** (fenêtre = 1) dans l'appel
  suivant, pour le bénéfice qualité immédiat sans l'accumulation.
- Les ThinkChunks complets vont dans les **traces** (diagnostic), pas dans l'historique
  renvoyé au modèle au-delà du dernier tour.

**À corriger côté persistance** : aujourd'hui l'historique stocke du texte. Il faut
stocker la réponse assistant sous une forme qui distingue thinking/texte (au moins pour
le dernier tour) — sinon le replay est impossible. Point d'implémentation §8.

---

## 4. Le refus humain — « ne pas répondre si ça n'a aucun sens »

Le cœur de votre demande. Aujourd'hui `too_vague` est une case JSON ; on la remplace par
un jugement de raisonnement à **4 verdicts**, produit dans l'appel A :

| Verdict | Exemple réel | Comportement |
|---|---|---|
| **Compréhensible + précis + dans le domaine** | « référence crémone 4 points coulissant » | → CHERCHER |
| **Ambigu** (manque un paramètre qui change la réponse) | « quelle houssette ? » (gamme non dite, espace multi-gammes) | → CLARIFIER (1 question) |
| **Incompréhensible / non-sens** | « loqueteaiu » non résolu même après trigram, suite de caractères | → « Je ne suis pas sûr de comprendre — parlez-vous du loqueteau (le petit verrou) ? » |
| **Hors domaine** | question météo, ou produit d'un concurrent non documenté | → refus poli + recadrage sur le périmètre PROFERM |

Principes (dans le prompt de l'appel A) :
- **Le doute vaut mieux que l'invention** : en cas d'incertitude sur le sens, DEMANDER,
  jamais deviner. C'est la posture SAV humaine.
- **Ne jamais entrer en retrieval sur un non-sens** : économise le pipeline ET empêche la
  réponse hors sujet confiante (le retrieval renvoie toujours *quelque chose*).
- **Anti sur-clarification** (le travers inverse, cf. mode guidé Askey) : 1 question max,
  et seulement si la réponse en dépend réellement. Si le raisonnement peut répondre pour
  les 2-3 cas, il présente les cas au lieu de questionner.

Un 2e filet en aval (appel B) : si malgré tout le retrieval ne ramène rien de pertinent
(score < seuil), le raisonnement aval **s'abstient** (« je ne trouve pas X dans les
documents ; escalade / vouliez-vous dire Y »).

---

## 5. `small` partout — implications concrètes

- **Un seul modèle** sur tout le pipeline (compréhension, KAG, enrichissement, extraction
  ET génération). Simplification ops/coût majeure. `GENERATION_MODEL` en config pointe sur
  small ; on garde la variable pour pouvoir tester medium en repli sans refactor.
- **Vision** : small est vision-capable (déjà `PAGE_EXTRACTION_MODEL`) → sert les images
  de page. ⚠ **Vérifier reasoning + vision simultanés** dès le 1er chantier.
- **Température unifiée 0.2-0.3** (réconcilier `.env` 0.7 / compose 0.55 / config 0.3).
  Suivre la reco Mistral pour les appels reasoning (le sampling recommandé peut différer) ;
  le déterminisme final est garanti par la vérification §6, pas par la température.
- **Contexte 256k** : suffisant même avec small, MAIS on budgète quand même (dédup L2,
  passages élus) — l'attention se dilue avant la limite et les tokens = coût/latence.
- **Repli qualité acté** : si small décroche en RÉDACTION sur le golden, on garde small
  pour A (juger/comprendre) et on bascule B (rédiger) sur `mistral-medium-3-5`. Aucun
  autre changement de plan. À trancher par le golden, pas a priori.

---

## 6. Vérification anti-hallucination (inchangée, indispensable)

Le ThinkChunk n'est PAS un journal fidèle du calcul → on ne lui fait pas confiance, on
vérifie la SORTIE. Après l'appel B, avant le flush :
1. **Programmatique (0 LLM)** : chaque référence/cote de la réponse doit apparaître telle
   quelle dans un passage élu. Sinon = violation.
2. **Auto-contrôle** dans le reasoning de B (déjà payé).
3. **1 retry max**, puis abstention honnête.

---

## 7. v2 — le vrai comportement Gemini/ChatGPT (agentique, après v1)

Fusionner A + retrieval + B en **un seul flux de raisonnement outillé** : le modèle
réfléchit, et **au milieu de sa pensée** décide d'appeler `chercher_documents(...)`,
lit les résultats, continue de raisonner, re-cherche si besoin, puis répond. C'est
littéralement ce que font ChatGPT/Gemini (« searching… », reasoning, answer).

- Outils : `chercher_documents`, `trouver_reference`, `lire_page`,
  `montrer_illustration`, `demander_precision` (cf. plan de pensée §E).
- **Budget 3 appels d'outils**, streaming des statuts pendant le reasoning.
- ⚠ **À vérifier avant de s'engager** : compatibilité `reasoning_effort="high"` +
  function-calling sur small chez Mistral (certaines API restreignent l'un avec l'autre).
  Si incompatible → rester en v1 (2 appels reasoning), qui délivre déjà 90 % du
  bénéfice. C'est pourquoi v1 est le socle et v2 l'extension.

---

## 8. Chantiers ordonnés

| # | Chantier | Dépend de | Effort | Validation |
|---|---|---|---|---|
| 0 | Golden answer-level en CI (arbitre) + traces (ThinkChunk loggé) | — | 2-3 j | baseline chiffrée |
| 1 | Bascule génération → small + parsing `content` liste + streaming ThinkChunk masqué | 0 | 2-3 j | pas de fuite thinking ; golden A/B vs large |
| 2 | Température unifiée 0.2-0.3 ; vérifs reasoning+vision, reasoning+sampling | 1 | 1 j | images servies, réponses stables |
| 3 | Appel B en reasoning (juger/élire/rédiger) + vérif programmatique §6 | 1 | 3-4 j | 0 réf inventée ; cas #1/#8 |
| 4 | **Appel A en reasoning — REMPLACE `_node_fused_understand`** (comprendre/décider/refuser + requêtes) | 3 | 1 sem | cas #6/#7 (typo/refus) ; #2 (clarif) |
| 5 | Refus 4 verdicts + anti sur-clarification (§4) | 4 | 2-3 j | non-sens refusé, hors-domaine recadré |
| 6 | Continuité : persistance thinking (replay dernier tour) + report attributs (CR5) | 3,4 | 2-3 j | cas #5 (pas de dérive) ; suivi elliptique OK |
| 7 | (optionnel) v2 agentique : outils + budget, si reasoning+tools OK | 4 | 1-2 sem | parité Gemini-like sur golden |

**Prérequis transverse** : CR2 (reranker multilingue) avant le chantier 3 — le jugement
aval s'appuie sur le score.

---

## 9. Risques et parades

- **small rédige moins bien que large** → mesuré chantier 1 ; repli medium sur B seul.
- **reasoning + vision incompatibles** → vérifié chantier 2 ; repli : description texte
  de la page avant l'appel.
- **reasoning + function-calling incompatibles** → bloque seulement v2 ; v1 intacte.
- **coût multi-tours (replay thinking)** → replay du dernier tour uniquement + état
  compact comme continuité principale (§3).
- **sur-clarification** (questionner trop) → 1 question max, seulement si ça change la
  réponse (§4).
- **latence reasoning="high" ×2 appels** → acceptable pour du SAV ; le refus précoce
  (§4) économise un tour entier sur les non-sens ; fast path possible plus tard
  (effort=none quand le score amont est trivialement fort).

---

## 9bis. Cas cible de référence — réponse Gemini (cas #1, crémone 4 points)

Preuve terrain : sur les MÊMES documents (SOLEAL GY-55, LUMEAL GA, DTA Lumeal uploadés),
Gemini produit la réponse idéale là où le système actuel enterrait l'info. C'est la
barre du golden answer-level.

**Question** : « pour un coulissant, il y a une crémone 3 point ref TGY3702, je cherche
la référence de la crémone 4 points pour coulissant. »

**Réponse cible (Gemini)** — traits à reproduire :
1. **Reformule la prémisse fausse** : « il n'existe pas de référence unique pour une
   crémone 4 points complète » → pour verrouiller 4 points, on associe la 3 points + une
   extension. (= raisonnement domaine, PAS retrieval brut → justifie Appel A+B reasoning.)
2. **Answer-first** puis références exactes : TGY3702 (3 pts) / TGY3703 (3 pts à clé) /
   **TGY3704 (rallonge inox 4ᵉ point)**, sans mélange.
3. **Conclusion actionnable** : « ajoutez TGY3704 à votre crémone 3 points ».
4. **Sourcé** (SOLEAL-GY-55-Catalogue-conception) → chaque réf traçable = phase 5.
5. **Concis**, périmètre exact, pas de dump de fiche.

**Enseignements** :
- L'info EST dans le corpus → écart 100 % pipeline, pas données. Dé-risque le plan.
- C'est le cas de rédaction/synthèse où small peut trailer un frontier → si small+reasoning
  ne l'atteint pas, bascule Appel B sur `medium-3-5`. Le golden tranche.
- Objectif = cette qualité AVEC notre moteur (retrieval à l'échelle, KAG, crops, profils,
  données maîtrisées), pas via Gemini.

## 10. En une phrase

On ne réécrit pas la continuité (elle est bonne) : on remplace les deux extractions
naïves — comprendre et rédiger — par deux temps de *raisonnement* sur un seul modèle
`small`, dont le premier a le droit de dire « ça n'a pas de sens, précisez » au lieu de
foncer. C'est ça, un assistant qui pense avant de parler.
