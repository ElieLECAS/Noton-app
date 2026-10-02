# Étape 2 — le tour de chat pour GLM : lots 2, 3 et 4 (plan, 02/10/2026)

Suite de `docs/plan_glm_first_2026-10-02.md` (§ 2 à 6) et de `docs/plan_etape1_wiki_index_2026-10-02.md`.
Les lots 0 et 1 sont faits (non commités) : `WikiIndex` sait construire la **fiche de la question**, fusionner
plusieurs formulations (`classer_multi`), repérer la **page dominante**, rendre la **carte** (pages, sections,
lignes qui répondent avec l'en-tête de leur tableau). Le chat n'en utilise rien encore : il livre toujours six
pages par recherche.

Reste : brancher tout cela dans le tour (lot 2), réécrire le prompt (lot 3), refaire l'écran (lot 4). Rien de ce
plan n'est codé.

## 1. Ce que le tour devient

```
question ─► serveur : FICHE (références rares et leurs emplacements, absentes du wiki, cotes, produits nommés,
            │          et ce que le tour précédent avait lu) ─► ajoutée au message de l'utilisateur
            ▼
   appel 1  GLM : comprend, lit l'index du wiki (prompt permanent), écrit 2 à 4 formulations
            ──► chercher(requetes, gamme?, systeme?)
            ▼
   serveur : formulations = [question brute] + [références rares] + [celles de GLM] → fusion (RRF)
            → CARTE (12 résultats) ; si une page domine et tient entière : elle est livrée avec
            ▼
   appel 2  GLM : choisit ce qu'il lit ──► lire(lectures=[{chemin, sections}, …])   (un seul appel)
            ▼
   serveur : livre les sections ; rapproche les anomalies des pages LUES ; marque l'état du tour
            ▼
   appel 3  GLM : répond (citations, coupes, nuances) — ou relance UNE fois si la carte ne montre rien de plausible
            ▼
   serveur : vérifie coupes et citations, relève « citée sans être lue », écrit la trace
```

Cible : 3 appels, 2 quand la fiche ou une page dominante suffit, 6 au plus. Le dernier appel autorisé part
**sans outils**.

## 2. Décisions de conception

| # | Décision | Pourquoi |
|---|---|---|
| 1 | **Lue ≠ listée.** Seules les sections livrées par `lire` (et la page dominante) comptent comme lues | Une ligne de carte situe une réponse, elle ne la prouve pas. Les anomalies, les coupes, les citations et le tour vocal reposent sur « lu » |
| 2 | **Les anomalies ne sont injectées qu'après une lecture** | Décision du 30/09 : « je pose une question, il cherche dans le wiki ». Sans pages lues, le rapprochement ne se ferait que sur les références de la question |
| 3 | **Trois outils, toujours** : `chercher`, `lire` (en lot), `lire_anomalie` | Règle de la maison. Un lot dans un seul appel plutôt que des appels parallèles : les appels d'outils streamés de GLM sont fragiles chez Mistral |
| 4 | **Le serveur fournit les formulations de base** : la question brute d'abord, les références rares ensuite, celles de GLM après | Mesuré le 02/10 : la fusion garde le score de la première formulation qui classe une section ; la question seule + 3 reformulations donne 69,6 % de « toutes les pages de preuve » contre 64,3 % pour la question seule |
| 5 | **Le raisonnement est renvoyé dans le tour** | Recommandation de Mistral et de Z.ai ; accepté par l'API (testé le 02/10, HTTP 200) |
| 6 | **Le fil d'un tour à l'autre est la fiche**, pas l'historique enrichi | « Tour précédent : requêtes …, lu : … » dans la fiche, tirée de la trace enregistrée. L'historique reste du texte : pas de ligne technique qu'un modèle pourrait recopier dans sa réponse |
| 7 | **Aucune reprise automatique sur réponse vide avant mesure** | On compte (journal, trace) et on décide au test réel. Règle de la maison : mesurer avant de proposer |
| 8 | **Le code remplacé est supprimé** | Pas de drapeau : `livrer` (sauf la page entière), `search`, `formate_liste`, les facettes `type`/`tags`/`limite`, leurs constantes et leurs tests |

## 3. Lot 2 — le tour (`wiki_chat_service.py` et ses voisins)

### 3.1 Les trois outils

```
chercher(requetes: [string] 1..4, gamme?: string, systeme?: string)
lire(lectures: [{chemin: string, sections?: [string]}] 1..6)
lire_anomalie(identifiant: string)
```

- `chercher` : la description dit qu'il rend une **carte** et pas des pages ; qu'il faut écrire les mots du wiki
  pour l'objet, la référence seule, un synonyme du métier ; que `gamme` et `systeme` font remonter, sans exclure.
  Une chaîne à la place de la liste est acceptée (tolérance de forme, pas de compatibilité).
  `type`, `tags` et `limite` disparaissent (le `type` ne classait déjà plus).
- `lire` : `sections` = « §13 », un mot du titre, ou « sommaire » ; sans `sections`, la page entière dans la
  limite du budget (sinon fiche + sommaire, comme aujourd'hui). Une lecture en erreur (chemin inconnu) ne fait
  pas échouer le lot : son message est rendu à sa place.
- `strict` n'est jamais envoyé (bogue d'hébergement observé chez Mistral).
- Le nom d'outil est normalisé (coupé au premier espace ou `<`) : un nom suivi de balises est un bogue connu.
  Un outil inconnu ou des arguments JSON illisibles rendent un message clair au modèle, au lieu d'un `{}` muet.

### 3.2 La boucle

- **Fiche** (`WikiIndex.fiche_question` + tour précédent) ajoutée au dernier message utilisateur du contexte.
  Le message enregistré en base reste la question seule.
- **Budget affiché** : chaque résultat d'outil se termine par `Tour : appel k/6 · lu X k car. sur 150 k`.
  `MAX_TOOL_ROUNDS` (8) devient `MAX_APPELS` (6) ; `BUDGET_TOUR` (200 k) devient `BUDGET_LECTURE` (150 k).
- **Dernier appel sans outils** : à l'appel 6, `tools` est retiré et un message système dit « réponds avec ce
  que tu as lu, et dis ce qui reste à vérifier ». L'erreur « trop d'allers-retours » disparaît.
- **Raisonnement renvoyé** : le message assistant du tour d'outils porte `content: [{"type":"thinking", …},
  {"type":"text", …}]` avec ses `tool_calls`. `_clean_messages` laisse déjà passer tels quels les messages
  qui portent des `tool_calls`. Le raisonnement est mémorisé par appel (aujourd'hui il est cumulé pour l'écran
  seulement).
- **Garde « aucune page lue »** conservée, avec un message mis à jour : « chercher ne livre qu'une carte ; appelle
  `lire` sur les sections qui portent la réponse ». Une fois seulement, et jamais au dernier appel.
- **`finish_reason`** : `mistral_service.chat_stream` le lit et l'émet ; la trace le garde ; une réponse coupée
  (`length`) affiche une ligne « réponse coupée ». `CHAT_MAX_TOKENS` passe de 4 096 à 16 384.
- **Réponse vide** : comptée dans la trace et le journal (`reponses_vides`), sans reprise (décision 7).
- **Citée sans être lue** : à la conclusion, les pages citées qui ne sont ni lues ni dominantes sont relevées dans
  la trace (`citees_non_lues`). On constate et on montre, on ne réécrit pas la réponse.

### 3.3 Ce que « lue » commande

| Consommateur | Aujourd'hui | Lot 2 |
|---|---|---|
| Anomalies | injectées après chaque outil | après chaque outil qui a livré une page ou une section |
| Coupes | `coupes_des_pages(pages_lues)` | inchangé : les pages lues |
| Sources citées | pages citées existantes | inchangé, plus `citees_non_lues` |
| Tour vocal | `etape["pages"]` non vide → on peut parler | `pages` reste **les pages lues** ; la carte va dans un champ à part (`carte`), `vocal_service.py` ne change pas |

### 3.4 Événements et trace

- `etape` : `outil`, `libelle` (« Carte », « Lecture », « Anomalie »), `detail` (les formulations, ou les
  lectures), `pages` (lues), **`carte`** (pages listées), `livraisons`, **`budget`** (appel, max, lu, total).
  Le chat et le vocal consomment déjà `etape` : `chat.html` et `vocal.html` changent au lot 4.
- Trace : + `finish_reason`, `reponses_vides`, `citees_non_lues`, `requetes` par étape, budget lu.
  `trace["steps"]` est ce que la fiche du tour suivant relit.

### 3.5 L'état d'un tour à l'autre

`routers/chat.py` et `routers/vocal.py` lisent la trace du dernier message de la réponse (`metadata_json`) et la
passent à `WikiAnswer(precedent=…)`, qui la rend dans la fiche : « Tour précédent — requêtes : …
· lu : /profiles/perform76-parcloses.md §3 §4 · /gammes/perform.md (page entière) », 600 caractères au plus.
Une conversation neuve n'a pas de tour précédent.

### 3.6 Fichiers

| Fichier | Changement |
|---|---|
| `app/services/wiki_chat_service.py` | cœur réécrit : `TOOLS`, `_executer`, `run`, `_chercher`, `_lire`, état du tour. Gardés : citations, anomalies, coupes (`preparer_images`), `load_history`, SSE |
| `app/services/wiki_index.py` | supprimés : `search`, `formate_liste`, `livrer` multi-pages (reste la livraison d'une page entière), `PAGES_PAR_RECHERCHE`, `SECTIONS_PAR_PAGE`, `BUDGET_RECHERCHE`, facette `tags` de `classer`. Gardés : `classer`, `classer_multi`, `carte`, `fiche_question`, `lire`, `Livraison`, `match_anomalies` |
| `app/services/mistral_service.py` | `finish_reason` lu et émis |
| `app/config.py` | `CHAT_MAX_TOKENS = 16384` |
| `app/routers/chat.py`, `vocal.py` | `precedent` tiré de la trace |
| `app/scripts/naviguer_wiki.py` | outils `chercher` / `lire` / `lire_anomalie` du nouveau tour |
| `app/scripts/mesurer_recuperation.py` | les lignes « livraison actuelle » et « 3 pages entières » partent avec `livrer` ; la carte reste, avec sa grille ; les chiffres anciens restent dans les documents |

### 3.7 Tests

- `tests/test_wiki_chat.py` réécrit : tour normal (carte → lecture → réponse) ; fiche dans le premier message
  et message enregistré inchangé ; page dominante livrée avec la carte ; `lire` en lot, budget et chemin
  invalide ; raisonnement renvoyé dans le message assistant ; dernier appel sans `tools` ; garde « aucune page
  lue » ; anomalies injectées après lecture et jamais avant ; nom d'outil normalisé et arguments illisibles ;
  `finish_reason` `length` visible ; `citees_non_lues`. Les tests de coupes (`preparer_images`,
  `coupes_des_pages`) ne changent pas.
- `tests/test_wiki_index.py` : les tests de `search`, `formate_liste` et de la facette `type` sont supprimés ou
  réécrits sur `classer` ; `tests/test_wiki_sections.py` : ceux de `livrer`.
- `tests/test_conversations_chat.py` et `tests/test_vocal.py` : leurs faux flux appellent `chercher` avec
  `mots_cles` ; ils passent à `requetes` et finissent par un `lire`.
- `tests/test_mistral_messages.py` : message assistant à contenu en blocs `thinking`, avec `tool_calls`.

**Critère de passage du lot 2 :** la suite complète passe, un tour simulé produit la trace attendue, et un
rejeu hors ligne des 4 recherches réelles de GLM sur TGY3731 montre ce que le tour livrerait. Aucun appel au modèle.

## 4. Lot 3 — le prompt

- **`index.md` dans le prompt permanent**, lu dans l'instantané (`pages["/index.md"].raw_text`, classé réservé
  mais chargé). ~36 000 tokens, 0,005 $ par appel en cache. Le prompt du tour vocal le reçoit aussi.
- **Consignes réécrites** (`wiki_consignes.md`) pour un modèle qui navigue : rôle ; méthode en quatre temps
  (comprendre, une recherche riche, lire en une fois, répondre) ; critère d'arrêt (« quand la section qui porte
  l'objet de la question est lue, réponds ; relance seulement si la carte ne montre aucune section plausible,
  jamais deux fois pour le même objet ; une ligne de carte situe, elle ne prouve pas ») ; règles de vérité
  condensées, **une ligne et un exemple chacune** : 2, 3, 4, 5, 6, 8, 9, 10, 13 et 14 ; forme.
  Elles perdent le ton « contre les fautes de Small » et la description des anciens outils. Une règle ne change
  pas de sens : on reformule, on ne retire rien (relecture règle par règle contre l'ancien texte).
- **Vocabulaire** : gammes, systèmes et les 649 tags ; la ligne `TYPES` part avec la facette.
- **Clé de cache** : déjà le hachage du prompt entier (`build_system_prompt`) ; un dépôt de wiki qui modifie
  `index.md` change la clé, comme aujourd'hui pour les consignes.
- **Tests** : `test_wiki_service.py` (le prompt commence par les consignes, ne porte ni corps de page ni
  anomalies) est mis à jour : il porte désormais l'index. Un test vérifie qu'`index.md` y figure en entier.
- **Mesure** : taille du prompt en caractères et en tokens (comptés par l'API au premier appel réel).

Un seul appel réel à la fin de ce lot, sur accord : la question TGY3731, à comparer à la trace de ce matin
(5 appels, 201 097 tokens, 181 553 car. livrés).

## 5. Lot 4 — l'interface

- **`chat.html`** : les étapes « Carte » (« 12 pages listées » avec les trois premières) et « Lecture » (les
  sections lues, comme aujourd'hui) ; la ligne de budget ; la page dominante affichée comme une lecture.
- **Fenêtre « Comment cette réponse a été produite »** (`chat.html`, ~l. 1098) : le texte « Tout le wiki était
  dans le contexte ; aucune recherche » est faux depuis le 22/09 ; il devient : appels au modèle, pages lues,
  caractères livrés, tokens d'entrée cumulés (« chaque appel relit tout ce qui précède »), part en cache, coût
  estimé, pastille « citée sans être lue » et « réponse coupée » quand elles s'appliquent.
- **Coût estimé** : trois tarifs en configuration (entrée, entrée en cache, sortie, par million de tokens) pour le
  modèle du chat ; sans tarif, la ligne n'est pas affichée. Valeurs de GLM 5.3 : 1,40 / 0,14 / 4,40 $.
- **`vocal.html`** : les libellés « Je cherche » / « Je lis » suivent les nouveaux noms d'outils.
- `admin.html` : rien, `last_call` garde ses champs.

## 6. Ordre d'exécution et points d'arrêt

| Étape | Contenu | Point d'arrêt |
|---|---|---|
| 2a | Vérification hors ligne de l'ordre des formulations (question, références rares, GLM) | chiffres montrés avant de le figer |
| 2b | `wiki_index.py` : suppressions + tests ; `mistral_service.py` (`finish_reason`) ; `config.py` | suite verte |
| 2c | `wiki_chat_service.py` : outils, boucle, état, trace ; routers ; scripts | suite verte, rejeu TGY3731 hors ligne montré |
| 3 | Prompt : consignes, index, vocabulaire ; tests | taille du prompt montrée ; **accord pour 1 appel réel** |
| 4 | Interface | capture de l'écran sur une réponse réelle |
| 5 | Mesure réelle (hors de ce plan, sur accord) | voir § 7 |

Je ne commite rien sans demande ; les modifications restent dans l'arbre de travail, par lot.

## 7. La mesure réelle (sur accord, après le lot 4)

1. **Un appel** : TGY3731. On compare à la trace du 02/10 matin : appels, tokens, caractères livrés, coût, réponse.
2. **Trois questions** : une référence, une faisabilité, une contradiction.
3. **Le golden de 40 questions**, une conversation par question, avec et sans `index.md` dans le prompt (~1 à 2 $ par passe).
4. **Variantes, une à la fois, sur trois questions** : `reasoning_effort` `high` contre `max` sur la faisabilité ;
   température 0,2 contre 1,0 (recommandation de Z.ai) ; avec et sans la ligne « tour précédent » sur une suite.

Critères proposés, à valider : médiane de **3 appels ou moins**, **60 000 tokens d'entrée cumulés ou moins**, coût
**0,03 $ par question ou moins**, et un score au golden de 40 questions **au moins égal à celui de Small** :
50 points sur 78 (64 %) en une passe le 01/10, après l'indexation par sections. Ce n'est pas la référence de
90 % du 30/09, qui portait sur l'ancien golden de 20 questions.

## 8. Risques

| Risque | Parade |
|---|---|
| GLM répond sur la foi d'une ligne de carte | consigne + `citees_non_lues` dans la trace ; à regarder dans le golden |
| L'index de 36 k tokens distrait GLM, comme un index de 35 k car. le faisait avec Small | test avec et sans `index.md` (§ 7.3) |
| Appel d'outil à tableau d'objets (`lire`) mal formé par GLM chez Mistral | tolérance de forme, message d'erreur clair, journal ; repli possible vers `lire(chemin, section)` à un seul objet si la mesure le montre |
| Identifiants d'appel d'outil changeants ou réponses vides (bogues d'hébergement signalés) | le client recolle déjà les fragments ; les réponses vides sont comptées |
| Raisonnement renvoyé : coût en tokens | quelques centaines par appel, en cache aux appels suivants ; mesuré dans la trace |
| Un tour coupé par `max_tokens` | 16 384 et `finish_reason` visible |
| Régression du tour vocal | `pages` reste « lues » ; les tests vocaux sont repris |

## 9. Décisions attendues

1. **Budgets** : 6 appels et 150 000 caractères lus par tour (aujourd'hui 8 et 200 000).
2. **Suppressions** : `WikiIndex.search`, `formate_liste`, `livrer` multi-pages et leurs tests (aucun consommateur
   de production ne reste ; `/api/wiki/search` utilise une autre fonction).
3. **Le fil d'un tour à l'autre** dans la fiche (décision 6) plutôt que dans l'historique.
4. **Pas de reprise sur réponse vide** avant la mesure réelle (décision 7).
5. **Tarifs en configuration** pour le coût affiché dans la fenêtre de trace.
6. **Schéma de `lire`** en tableau d'objets, avec repli à un seul objet si GLM s'y perd.

## 10. Réalisé le 02/10 (lots 2, 3 et 4) — et ce qui a changé en route

Fait, non commité, **204 tests verts** : le tour (3 outils, carte, lecture en lot, raisonnement renvoyé, dernier appel
sans outils, `finish_reason`, `citees_non_lues`, fil d'un tour à l'autre par la fiche), le prompt (consignes,
`index.md`, vocabulaire), l'interface (étapes « Carte » et « Lecture », budget, fenêtre de trace, coût estimé,
libellés du vocal), les deux routeurs, les scripts de banc.

Écarts au plan, par mesure ou par prudence :

- **Pas de formulation « références rares » à part.** Mesuré hors ligne (étape 2a) : question + références + GLM,
  question + GLM + références et question + GLM donnent le même résultat. Les formulations sont la question brute,
  puis celles de GLM.
- **La fiche ne dit que ce qui change la recherche** : références rares, absentes, citées seulement dans un registre,
  tour précédent. Les cotes et les produits nommés restent calculés, mais ne sont pas écrits dans le message.
- **Consignes : le cadre est réécrit, les règles de vérité sont gardées** avec leur numérotation (le vocal renvoie
  aux règles 11 et 13) et seulement ajustées aux nouveaux outils. Les condenser aurait gagné ~1 500 tokens sur un
  prompt de ~44 000 : à décider après le golden, pas avant.
- **Probes API avant de coder** (~1 000 tokens) : un message système en milieu de conversation est accepté et suivi,
  un dernier appel sans `tools` avec un historique d'appels d'outils est accepté.
- **Configuration** : le défaut `CHAT_MAX_TOKENS` de `docker-compose.yaml` (4 096) l'emportait sur celui de
  `config.py` ; il est à 16 384. Tarifs `MODEL_PRIX_*` dans `.env` (local, ignoré par git) et en défaut à 0 dans le
  compose. **Un changement de `docker-compose.yaml` ne s'applique qu'à la recréation du conteneur.**

### Premier essai réel, et son échec

Question de faisabilité PERFORM76 (oscillo-battante 1 300 × 1 450, 76171 / 76281, triple vitrage) : le tour s'est
terminé sur « Réponse coupée » **sans une ligne de réponse**. Cause : le conteneur tournait encore avec
`max_tokens` = 4 096 (le journal le dit) ; le raisonnement de GLM a consommé la limite du dernier appel avant le
premier mot. Corrigé en recréant le service web. Le message d'échec dit maintenant que la réflexion a épuisé la
limite quand rien n'a été écrit.

Rejeu réel, une seule fois, après correction (6 appels, 27 s) : réponse **correcte et sourcée** (limites du DTA, abaque
d'ouvrant et règle des 25 %, parclose 76503 pour 36 mm, poids du vantail, anomalie CTR-01, nuance « fabrications
certifiées »). La sortie la plus longue d'un appel est de 1 948 tokens, donc 16 384 laisse de la marge.

| Mesure | Résultat | Cible du plan |
|---|---|---|
| Appels | **6** (une carte, quatre lectures, la réponse) | médiane ≤ 3 |
| Tokens d'entrée cumulés | 359 639, dont 260 928 en cache (73 %) | ≤ 60 000 |
| Coût estimé | **0,195 $** à froid | ≤ 0,03 $ |
| Question suivante (« montre moi sa coupe », 2 min plus tard) | 3 appels, 141 375 tokens, 92 % en cache, ~0,04 $ | |

Lecture de ces chiffres : (1) le premier appel d'une conversation ne trouve pas le prompt en cache après quelques
minutes d'inactivité : ~44 000 tokens au prix plein, soit ~0,06 $ ; une question suivante le trouve (92 %).
(2) Une faisabilité à cinq sources fait lire GLM en quatre fois au lieu d'une, malgré la consigne : chaque tour de
lecture ajoute ~10 000 tokens relus ensuite. (3) Les cibles de coût et d'appels ne valent que pour la **médiane du
golden**, pas pour la question la plus dure ; elles restent à mesurer (§ 7).
