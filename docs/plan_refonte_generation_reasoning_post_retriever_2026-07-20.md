# Plan — Refonte de la génération en reasoning POST-retriever

_2026-07-20 · branche `fix/retriever`. **Supersède** l'architecture « 2 appels
reasoning (Appel A pré + Appel B post) » de
`plan_impl_reasoning_query_generation_2026-07-17.md`. Décision d'architecture
prise après les investigations de cette session (hallucination « hauteur de
poignée », cas « 6101 », désync ColPali) : **le reasoning va UNIQUEMENT
post-retriever**. Le pré-retrieval reste léger et non-reasoning._

---

## 1. Décision d'architecture (et pourquoi)

### Deux options considérées

- **Option A** — reasoning en amont (comprendre / décider chercher-ou-non /
  réécrire les requêtes) **ET** reasoning en génération. C'était le design du
  17/07.
- **Option B** — requête utilisateur quasi inchangée pour le retrieval,
  reasoning **uniquement** post-retriever (jugement + rédaction + auto-contrôle).

### Retenu : Option B. Justification ancrée dans les faits de cette session

1. **Le goulot est la génération, pas le retrieval.** L'hallucination
   « hauteur de poignée » est survenue avec le doc 411 **correctement packé** ;
   le cas « 6101 » a bien traversé KAG → fusion → rerank. Le reasoning coûteux
   doit aller là où le problème se manifeste : après le retrieval.
2. **Réécrire la requête = haut risque sur un corpus à références exactes**
   (6101, TGY3702, 9F68). La chaîne brute utilisateur est souvent la meilleure
   requête lexicale/KAG. Un reasoning qui « améliore » la requête peut perdre le
   code exact ou dériver. Recall déjà bon (4 canaux, KAG corrigé, ColPali
   toujours actif).
3. **Le CAG rend la précision passage inutile en amont** : il packe des
   documents ENTIERS, donc il suffit que le bon DOCUMENT soit dans le pool —
   barre de recall déjà atteinte, rien à optimiser par pré-reasoning.
4. **Latence/coût** : reasoning=high ≈ ×3 tokens de complétion. UN appel
   reasoning/message (post) plutôt que deux (pré+post) sur chaque tour.
5. **Le refus/abstention précoce est déjà couvert** par le gate du reranker
   (P0). Chercher un non-sens puis abstenir coûte peu.

### Ce que le pré-retrieval garde (léger, SANS reasoning)

`lightweight_query_understanding` (modèle `MODEL_QUERY_UNDERSTANDING`, JSON, pas
de reasoning) reste **inchangé** pour :
- résolution d'ellipse / condensation multi-tours (`standalone_question`) —
  DOIT précéder le retrieval ;
- route `direct` / `clarifier` / `rag` ;
- signaux (intent, références détectées) qui alimentent CAG budget + boosts.

**On ne l'upgrade PAS en reasoning=high.** La requête envoyée aux retrievers
préserve les termes exacts de l'utilisateur (surtout les codes de référence).

---

## 2. Architecture cible

```
message + historique + état du fil (topic, focus_entities, ancre)
   │
   ▼  [INCHANGÉ, léger, non-reasoning]
lightweight_query_understanding  →  route + standalone_question + signals
   ├─ direct     → réponse directe (0 retrieval)
   ├─ clarifier  → 1 question, STOP
   └─ rag        │
                 ▼
   RETRIEVAL (inchangé : 4 canaux → RRF → rerank → dynamic-K)
                 │  gate d'abstention du reranker conservé
                 ▼
   CAG packing (documents entiers, budget par intent)  [INCHANGÉ]
                 │
                 ▼  ★ SEUL POINT NOUVEAU ★
   APPEL GÉNÉRATION avec reasoning_effort="high"
     thinking (MASQUÉ du stream) : « quels documents packés répondent à CETTE
        question ? que disent-ils EXACTEMENT ? qu'est-ce que je serais tenté
        d'ajouter depuis mes connaissances générales et que je NE dois pas ? »
     text (STREAMÉ) : la réponse, strictement fondée
                 │
                 ▼
   VÉRIFICATION post-génération  [DÉJÀ IMPLÉMENTÉE ce jour]
     response_verification_service : contrôle programmatique + juge LLM
```

Un seul appel reasoning par message (au lieu de deux). Le reasoning natif est
l'endroit où le « moment de jugement » (élire les bons documents avant de
rédiger) se fait proprement, séparé de la prose via le ThinkChunk.

---

## 3. Chantiers

### C0 — Client streaming compatible reasoning (PRÉ-REQUIS BLOQUANT, ~0,5-1 j)

**Vérifié en direct 2026-07-20 :** `mistral-small-latest` + `reasoning_effort="high"`
retourne bien `content = [{"type":"thinking",...}, {"type":"text",...}]` (200 OK),
et **vision + reasoning cohabitent** (testé sur une page réelle du corpus — lève
l'inconnue « non documenté » du plan du 17/07).

Le verrou est côté client : `chat_stream` ([mistral_service.py:446-449](app/services/mistral_service.py:446))
fait `content = delta.get("content")` puis `if content: yield {"message":{"content": content}}`.
En reasoning, pendant la phase thinking, `delta.content` est une **liste** →
il streamerait la liste ThinkChunk brute au frontend (cassé).

À faire dans `chat_stream()` :
- Détecter `delta.content` de type `list` (phase thinking) → **ne pas streamer
  au client** ; optionnellement émettre un statut `{"status":"reasoning"}` pour
  un indicateur « analyse… » UX.
- Détecter le chunk de transition (fin thinking + 1er TextChunk).
- Reprendre le yield normal `{"message":{"content": str}}` uniquement pour la
  phase texte.
- `SourcesTagStreamFilter` continue de ne s'appliquer qu'au texte.
- Idem `chat()` non-stream : déjà OK (testé), mais ajouter un helper
  `split_reasoning_content(message) -> (text, thinking)` pour les appelants qui
  veulent le thinking (traces).

### C1 — Génération raisonnée (~2-3 j)

- `_stream_llm_to_sse` (chat.py) : passer `reasoning_effort="high"` sur l'appel
  de génération quand un flag `GENERATION_REASONING_ENABLED` est actif (défaut
  `false`, bascule progressive).
- `max_tokens` : le thinking consomme le budget. Variable dédiée
  `CAG_REASONING_MAX_COMPLETION_TOKENS` (~8192) quand le reasoning est actif.
- `SPACE_CHAT_SYSTEM_PROMPT` : ajouter un bloc « MÉTHODE (dans ta réflexion) »
  qui cadre le thinking — reformuler la question, lister les documents dont
  l'en-tête correspond, rédiger UNIQUEMENT à partir d'eux, ne rien ajouter
  depuis les connaissances générales (normes, cotes non écrites). Le prompt
  answer-first + concision (P2) reste.
- Le hack « RAPPEL FINAL » (`rag_generation_service.py:275`) devient
  probablement inutile avec le thinking (à retester au golden ; retirer si le
  reasoning suffit à garder le focus).

### C2 — Repli modèle configurable (option « medium » réservée, ~0,5 j)

- Variable `GENERATION_REASONING_MODEL` (défaut = `MODEL_FAST`). Permet de
  pointer l'appel de génération raisonnée vers `mistral-medium-3-5`
  **uniquement** si le golden montre que small décroche en rédaction raisonnée
  — sans toucher au reste du pipeline (le pré-retrieval et la vérif restent sur
  small). C'est le « droit réservé » exprimé : un seul point de bascule, une
  variable de config.

### C3 — Vérification (DÉJÀ FAITE ce jour, à articuler)

`response_verification_service` (contrôle programmatique + juge LLM) est déjà
branché post-génération. Avec le reasoning en place :
- Le contrôle programmatique reste le backstop déterministe (détecte les
  inventions totales de normes/cotes).
- **Décision à prendre au golden** : si la vérif signale un problème, faut-il
  (a) juste tracer (comportement actuel), (b) tenter 1 retry non-stream avec un
  thinking renforcé « tu as inventé X, refais en te limitant au contexte », ou
  (c) basculer en abstention honnête ? Le reasoning rend le retry (b) crédible
  (le modèle peut se corriger dans son thinking). À trancher avec des données.

### C4 — (Optionnel, plus tard) Re-recherche agentique

Une fois C1 stable : autoriser le reasoning post-retriever à **déclencher une
2ᵉ recherche** s'il juge que le contexte packé ne répond pas (vrai « Gemini-like »
tool-use mid-reasoning). Capte la valeur d'Option B **plus** un filet si le 1er
retrieval a raté, sans pré-reasoning gâché. **Gaté par vérification préalable de
la compat reasoning + function-calling sur Mistral** (à tester, non garanti).
Hors périmètre du lot initial.

---

## 4. Ordre et bascule

1. **C0** (client) — pré-requis, testable isolément (mock content liste).
2. **C1** derrière `GENERATION_REASONING_ENABLED=false` → golden A/B :
   reasoning off vs on, sur les cas réels (hauteur de poignée, 6101, houssette).
   Mesures : hallucinations (via `response_verification`), hors-sujet, longueur,
   latence 1er token, coût tokens.
3. **C2** seulement si small décroche en rédaction (repli medium sur ce seul
   appel).
4. **C3** décision retry/abstention selon les données golden.
5. **C4** plus tard, si compat tools confirmée.

Repli global : `GENERATION_REASONING_ENABLED=false` restaure exactement le
pipeline actuel (small, temp 0.2, prompts P2, vérification). Aucun risque de
blocage — le reasoning est une couche activable, pas une réécriture destructive.

---

## 5. Ce que ce plan NE fait PAS (décisions actées)

- **Pas de reasoning pré-retrieval** ni de réécriture de requête par LLM
  (préserve le recall et les codes de référence exacts).
- **Pas de refonte « toute la pipeline »** — refonte chirurgicale sur le seul
  appel de génération.
- Pas de blocage du streaming pour la vérification (inchangé depuis P2).
- Pas de bascule medium par défaut — small partout, medium en repli configurable
  sur le seul appel de génération.
