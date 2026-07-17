# Plan de pensée — Une génération qui raisonne comme un SAV humain

_Date : 2026-07-17 · Branche : `fix/retriever`. Remplace la version précédente de ce
document (dont le diagnostic « la génération tourne sur mistral-small » était FAUX pour
le déploiement réel — voir §0). Complète `plan_amelioration_cas_reels_2026-07-17.md` et
`plan_patch_kag_categories_chunk_2026-07-17.md`. Objectif : deux usages, un seul cerveau
— moteur de recherche interne ET assistant SAV client, avec une qualité de réponse de
classe Gemini/ChatGPT._

---

## 0. Re-diagnostic (correction)

Vérifié dans le code et le `.env` du déploiement réel :

- **Le modèle de génération est `mistral-large-latest`** (`.env` : `MODEL_FAST=mistral-large-latest`).
  Le modèle n'est PAS le goulot. _(Correction de la note précédente qui lisait le défaut
  du code `mistral-small` ; le `.env` prod surcharge à large.)_
- **Le prompt système est déjà bon** (`chat.py:212-251`) : answer-first, politique de
  clarification si gamme ambiguë, abstention si info absente, grounding strict, anti-mélange
  de références. Aucune consigne clé ne manque.
- **La température prod est 0.7** (`.env`), contre 0.55 (compose) et 0.3 (`config.py`) —
  3 sources divergentes. 0.7 sur du SAV technique = invitation à broder.

**Alors pourquoi ça déraille quand même ?** Parce qu'un bon modèle + un bon prompt ne
suffisent pas si :

1. **La consigne est noyée.** Politique en tête, puis 100k-256k tokens de documents
   entiers collés (`build_cag_context`). Le modèle attend sur la masse, pas sur la
   politique. Preuve : le hack « RAPPEL FINAL » (`rag_generation_service.py:275`) existe
   parce que la consigne du haut ne tient plus après le mur de texte.
2. **La température invite à combler les trous.** Quand le retrieval a raté, 0.7 change
   un manque d'information en invention confiante.
3. **Il n'y a AUCUN moment de jugement.** Architecture actuelle = retrieve → tout coller
   → générer, en un jet. Or le retrieval ne renvoie jamais vide (toujours un plus proche
   voisin) → il y a toujours quelque chose dans le contexte → sans étape qui juge la
   pertinence, le modèle traite des passages hors sujet comme matière à répondre.

Le vrai déficit n'est ni le modèle ni le prompt : **c'est l'absence du temps d'arrêt
qu'un SAV humain prend avant de parler.**

---

## 1. Le modèle mental d'un SAV humain

Un bon conseiller SAV, face à une question, déroule 6 temps — la plupart en une fraction
de seconde, mais ils existent :

1. **Écouter** — que veut-il vraiment ? Sur quel produit / quelle gamme / quel symptôme ?
   La question est-elle même claire ?
2. **Jauger ce qu'il sait** — « est-ce dans ma doc ou pas ? ai-je de quoi répondre avec
   certitude, ou seulement une vague idée ? »
3. **Choisir un mode** :
   - question claire + info fiable → **répondre**
   - question dépend d'un paramètre non précisé (gamme, version) → **demander** (1 question)
   - info absente / doute → **le dire honnêtement**, proposer une vérif ou **escalader**
   - hors périmètre → **rediriger**
4. **Répondre** — la réponse d'ABORD (la référence, la cote, le oui/non), puis le détail
   utile. Jamais réciter le manuel entier.
5. **Se relire avant d'envoyer** — « est-ce que je réponds vraiment à SA question ? suis-je
   sûr de ce que j'avance ? »
6. **Proposer la suite** — « voulez-vous que je vous montre le schéma / la page ? ».

Le système actuel écrase les 6 temps en un seul appel de génération sur un blob de texte.
Il saute le 1 (écoute active), le 2 (jauge), le 3 (choix de mode), le 5 (relecture). Il ne
fait que le 4, mal, parce qu'il n'a pas fait les autres.

### Preuve par les cas réels
| Cas réel | Temps SAV manquant | Ce qu'un humain aurait fait |
|---|---|---|
| Crémone 4 points → dump de la 3 points | 2 + 4 | « La 4 points, c'est TGY3704 (rallonge 4e point). » (l'info ÉTAIT dans le contexte, enterrée) |
| houssette → poignées (confiant) | 2 + 3 | « Houssette… le patin d'étanchéité ? Sur quelle gamme ? » |
| « montre-moi les images » → « je n'ai pas accès » | 6 | montre la page / le schéma |
| SOLEAL FY après « paumelle invisible » → dérive | 1 (mémoire du fil) | garde en tête qu'on parle d'invisibles |
| Askey roulettes : 5 questions | 3 (dosé) | 1-2 questions max, puis une action |

---

## 2. Le plan de pensée : installer le raisonnement SAV

L'idée directrice : **ajouter un temps de jugement entre le retrieval et la réponse**, et
donner à l'assistant les moyens d'agir sur ce jugement. Du moins cher au plus profond.

### Étape A — Le moment de jugement (le cœur, ~1 sem)
Avant de rédiger, une délibération explicite. Deux façons, à combiner :

- **Signal dur (gratuit)** : à partir du meilleur score post-rerank (une fois le reranker
  multilingue en place, cf. CR2) + des signaux de compréhension déjà calculés, classer la
  situation : `RÉPONDRE` / `DEMANDER` / `ABSTENIR-ESCALADER` / `MONTRER`. C'est le
  garde-fou : sous un seuil de pertinence, on n'entre même pas en mode « réponse
  affirmative ».
- **Délibération LLM (le levier qualité)** — DÉCISION MISE À JOUR (reasoning natif Mistral,
  cf. https://docs.mistral.ai/studio-api/conversations/reasoning) :
  - Le reasoning natif de Mistral (`reasoning_effort="high"`) EST exactement cette étape,
    mais faite proprement : une vraie passe de réflexion, sortie dans un `ThinkChunk`
    (`type:"thinking"`) séparé du `TextChunk` final. Plus besoin de bricoler un « plan de
    réponse » par prompt.
  - **MAIS `mistral-large-latest` (le modèle de génération actuel) NE le supporte PAS**
    (HTTP 422). Seuls `mistral-small-latest` et `mistral-medium-3-5` l'exposent ; les
    modèles `magistral-*` sont dépréciés. Le paramètre marche sur le chat standard et sur
    Agents/Conversations (dans `completion_args`).
  - **Conséquence : swap génération `mistral-large-latest` → `mistral-medium-3-5`.** Ce
    n'est pas un downgrade : medium-3.5 est plus RÉCENT que large (large = fin 2024),
    multimodal (indispensable — on sert des images de page), agentique, et moins cher.
    Mistral recommande `reasoning_effort="high"` pour l'agentique.
  - **Intégration parfaite au router 2 chemins (§E)** : `reasoning_effort="none"` sur le
    fast path (~75 % des messages simples, pas de surcoût) ; `"high"` sur le chemin
    agentique et les cas ambigus, là où le jugement compte.
  - **À vérifier avant bascule** (sur le golden answer-level) : (a) parser le `ThinkChunk`
    et NE PAS le streamer comme réponse (masquer ou repliable) — le streaming actuel attend
    du texte ; (b) confirmer que reasoning + vision (images de page dans le message)
    cohabitent ; (c) mesurer latence/coût du thinking chunk.
  - **Option `mistral-small-latest` + reasoning** (à benchmarker vs medium sur le golden) :
    small supporte aussi `reasoning_effort` et est DÉJÀ le modèle de tout le reste du
    pipeline (understanding/KAG/enrichissement/extraction) + vision-capable
    (`PAGE_EXTRACTION_MODEL`). Distinction clé : le reasoning rapproche fortement small
    d'un gros modèle pour DÉCIDER (jugement, choix d'outil), mais pas pour SYNTHÉTISER une
    réponse finale nuancée sur long contexte FR. D'où deux architectures :
    - _Option 1 — small+reasoning partout_ : simplicité/coût max, un seul modèle ; risque =
      qualité de synthèse en profil SAV client.
    - _Option 2 (recommandée par défaut)_ : `small+reasoning="high"` pour le **jugement /
      gate / décisions d'outils** (cheap, haute fréquence) ; `medium-3-5` pour la
      **rédaction finale** (synthèse, adhérence consignes, ton). Profil interne peut passer
      à small+reasoning si le golden l'égale ; profil SAV client = medium au moins au début.
    - Décider sur le golden answer-level (4 axes). Si small égale medium → Option 1.
  - _Repli si le reasoning natif déçoit partout : garder large SANS reasoning et installer
    le « plan de réponse » par prompt (préfixe structuré : intention / ai-je l'info ? /
    ambiguïté ? / décision / éléments à citer)._

Effet : le temps 2 et le temps 3 existent enfin. Le modèle ne peut plus glisser
directement du blob à une réponse affirmative hors sujet.

### Étape B — Discipliner le contexte pour que la réponse ne soit plus enterrée (~2 j)
Le jugement de A ne sert à rien si la matière est un mur redondant.

- **Dédup** : exclure les chunks L2 du packing (ils dupliquent les L1 — audit §2) ; dédup
  par hash sur le reste.
- **Budget par intention** : document entier seulement s'il tient dans un budget (ex.
  ~15k tokens) OU si l'intention est « procédure complète » ; sinon top-passages +
  voisinage (page ±1). Ordonner par pertinence.
- **Marquer la cible** : quand la compréhension a extrait une référence/entité précise,
  la signaler en tête du contexte (« L'utilisateur cherche : crémone 4 points coulissant »)
  pour que la réponse s'ancre dessus au lieu de résumer le document.
- Mesure : tokens packés/message ÷2 à ÷3 sur le golden sans perte de recall answer-level.

### Étape C — Calmer la génération (~1 h)
- **Température unifiée à ~0.2-0.3** pour la génération SAV (réconcilier les 3 sources
  divergentes ; 0.7 est la valeur qui fait broder). La chaleur du ton vient du prompt, pas
  de la température.
- Éventuellement 2 températures : un peu plus haute pour le profil « interne » exploratoire,
  basse pour le profil « SAV client » (cf. §3).

### Étape D — La relecture avant d'envoyer (temps 5, ~2-3 j)
- **Self-check intégré** : dernière consigne du plan de réponse — « avant de finaliser,
  vérifie que la 1re phrase répond à la question, et que chaque référence/cote citée
  apparaît telle quelle dans un passage fourni ; sinon retire-la ou signale le doute ».
- **Critique pour le profil client** : `chat_critique_service` existe déjà (comparaison
  aux FAQ correctives). Le généraliser en garde-fou de sortie pour le SAV client (pas
  seulement FAQ) : un 2e regard rapide qui bloque une réponse non fondée avant envoi.

### Étape E — Agir sur le jugement : les outils (temps 4 et 6, ~1-2 sem)
Quand le jugement de A dit « je n'ai pas de quoi répondre » ou « il manque une info »,
l'assistant doit pouvoir AGIR comme un humain qui rouvre un classeur — pas broder. La
boucle de function calling **existe déjà** (`mistral_service.py:190`, aujourd'hui limitée
à Brave Search, non-streaming `chat.py:348`). On y branche le retrieval interne :

| Outil | Réutilise | Temps SAV |
|---|---|---|
| `chercher_documents(query, gamme?)` | pipeline retrieval existant | reformuler et re-chercher (crémone 4 points) |
| `trouver_reference(code)` | entité KAG post-patch | vérifier TGY3704 avant de l'affirmer (temps 5) |
| `lire_page(doc, page)` | texte L1 + PNG `page_cache/` | lire la page voisine d'un tableau coupé |
| `montrer_illustration(code)` | `illustration_service` post-CR1 | « montre-moi le dessin » (temps 6) |
| `demander_precision(question)` | UI | 1 question de clarification (temps 3) |

- **Budget dur : 3 appels** puis réponse obligatoire. Pas de planner ni multi-agents — un
  seul conseiller outillé à budget court. Au-delà = latence et debug pour un gain marginal.
- **Router 2 chemins** : ~75 % des messages (jugement A = `RÉPONDRE`, score fort) passent
  en génération directe disciplinée (rapide) ; le reste bascule en boucle outillée.
- **Streaming du statut** : émettre « je vérifie la référence TGY3704… » pendant les
  appels (le front gère déjà des events `status`/`cropping`), puis streamer la réponse.

### Étape F — La mémoire du fil (temps 1, ~1 j)
Pour ne plus dériver (cas SOLEAL FY / paumelle invisible) : reporter d'un tour à l'autre
non seulement le document ancré (déjà fait) mais les **attributs discriminants** de la
demande (« invisible », « coulissant », « 24 mm »), et les faire expirer seulement au
changement explicite de sujet. C'est CR5 du plan cas réels.

---

## 3. Un cerveau, deux personnalités : interne vs SAV client

Même retrieval, mêmes outils, même moment de jugement — deux **contrats de réponse**
(profil par espace ou par rôle) :

| | Moteur de recherche interne | Assistant SAV client |
|---|---|---|
| Public | expert qui trie vite | client qui a besoin d'être guidé |
| Jugement `DEMANDER` | présente les 2-3 candidats | pose 1 question, attend |
| Jugement `ABSTENIR` | « pas trouvé ; le plus proche : … » | « je transmets au service technique » (escalade nette) |
| Ton | dense, technique, answer-first | chaleureux, pas à pas, rassurant |
| Température | ~0.3 | ~0.2 |
| Self-check (D) | léger | critique de sortie obligatoire |
| Web search | autorisé | interdit (docs validés only) |
| Images | sur demande | proactives (le client ne pense pas à les demander) |
| Latence | fast path prioritaire | l'agentique peut prendre +5-10 s, acceptable |

C'est le split ChatGPT consumer / enterprise : le cerveau ne change pas, le contrat et les
seuils oui.

---

## 4. Pourquoi cet ordre (dépendances)

1. **CR2 (reranker multilingue + seuil)** d'abord — le moment de jugement (A) s'appuie sur
   un score de pertinence fiable. Un jugement basé sur le reranker anglais actuel jugerait
   mal. _Un agent qui itère sur un retrieval bruité converge vers du bruit, plus lentement._
2. **B (contexte) + C (température)** — quick wins qui débloquent le modèle déjà en place.
3. **A (jugement) + D (relecture)** — le cœur du raisonnement SAV.
4. **E (outils) + F (mémoire)** — agir sur le jugement, tenir le fil.
5. **Profils (3)** en dernier — une fois le cerveau bon, décliner les 2 contrats.

Tout est arbitré par le **golden answer-level** (4 axes : répond / citation valide /
abstention correcte / image servie — §2.1 du plan cas réels), et A/B fast-path vs outillé
avant tout élargissement.

---

## 5. Ce qu'on ne fait PAS

- **Multi-agents / planner-executor / graphe** : un conseiller outillé à budget court
  couvre les cas observés ; au-delà = latence, coût, debug.
- **Fine-tuning** : le levier prompt + jugement + contexte + température n'est pas épuisé,
  et le corpus évolue plus vite qu'un cycle de fine-tuning.
- ~~**Changer de modèle de génération** : mistral-large convient.~~ **RÉVISÉ (§A)** :
  swap `large` → `mistral-medium-3-5` recommandé, pour obtenir le reasoning natif
  (`reasoning_effort`) que large ne supporte pas. Ce n'est pas une montée en puissance
  brute mais l'accès au « moment de jugement » natif ; à valider sur le golden.
- **Abandonner le CAG** : packer la bonne notice reste bon pour les procédures ; il
  devient un seed budgété (B) que le jugement (A) complète via les outils (E).
- **Agentiser la compréhension de requête** : le fused understanding (1 appel) marche ;
  on investit sur la génération, là où est le déficit.

---

## 6. Synthèse en une phrase

Le système sait chercher et sait rédiger ; il ne sait pas **juger** ce qu'il a trouvé
avant de parler, ni **agir** quand il n'a pas de quoi répondre. Installer ce temps de
jugement (A), lui donner de quoi agir dessus (E), le nourrir d'un contexte propre (B) à
basse température (C), et le faire se relire (D) — voilà le chemin d'une réponse de SAV
humain, sans refondre le retrieval ni changer de modèle.
