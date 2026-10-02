# LIA « GLM 5.3 first » — refonte de la navigation (proposition, 02/10/2026)

Suite de `docs/audit_glm_5_3_2026-10-02.md`. On ne corrige plus l'existant : on refait le tour de
chat pour un modèle qui **sait naviguer**. Small 4 ne lisait pas ce qu'on ne lui donnait pas : il
fallait tout lui livrer d'avance, 6 pages par recherche. GLM lit une carte, choisit et va lire.
Ce qui change : le serveur ne livre plus des pages, il **cartographie** ; GLM comprend la question,
cherche une fois, lit en une fois, répond.

Rien n'est codé. Les chiffres viennent de mesures hors ligne faites le 02/10 (golden : 20 questions
avec preuves ; banc du 30/09 : 56 questions), sur le code d'index actuel.

## 1. Ce qu'on garde, ce qu'on jette

**On garde — les garanties qui ne dépendent pas du modèle :**

- le wiki comme seule source, une recherche lexicale (BM25 par sections et par page), des
  facettes qui remontent une page sans en exclure ;
- pas de juge, pas de reranker, pas d'index vectoriel : deux pages qui se contredisent doivent
  remonter toutes les deux ;
- les anomalies injectées par le serveur après lecture, les coupes vérifiées sur disque, les
  citations vérifiées ;
- la découpe en sections, le sommaire, le budget du tour (`Livraison`).

**On jette — l'héritage de Small :**

| Héritage de Small | Pourquoi il existait | Remplacé par |
|---|---|---|
| 6 pages livrées d'office par recherche (52 k car.) | Small ne lisait pas ce qui n'était pas livré | une **carte** de 15 à 20 k car., puis une lecture choisie |
| Une relance livre les 6 pages suivantes | « rien deux fois » | une carte qui marque « déjà lu » |
| Références cherchées à part, après la recherche | Small oubliait la référence dans sa requête | une **fiche de la question**, avant le premier appel |
| « Absent de tout le wiki » ajouté après la recherche | six recherches pour une absence | la fiche, avant le premier appel |
| Consignes de 145 lignes contre les fautes de Small | — | un prompt écrit pour un navigateur |
| Erreur « trop d'allers-retours » | coût | dernier appel sans outils : il répond avec ce qu'il a lu |
| Raisonnement jeté entre deux appels | — | raisonnement renvoyé (recommandé par Mistral et Z.ai) |
| Facettes `type`, `tags`, `limite` | — | `gamme` et `systeme` seulement (le `type` ne classe plus depuis le 30/09) |

## 2. Le tour cible

```
question
  │
  ├─ 0. SERVEUR — fiche de la question (déterministe, ajoutée au message utilisateur)
  │     références rares → leurs occurrences exactes (page, §, ligne + en-tête du tableau)
  │     référence absente → « absente du wiki » · cotes · gamme/système reconnus
  │     pages lues au tour précédent (questions de suite : « et la rallonge TGY3710 ? »)
  │
  ├─ 1. GLM — comprend (raisonnement) puis chercher(requetes = 2 à 4 formulations)
  │          ou lire directement si la fiche désigne déjà la section
  │
  ├─ 2. SERVEUR — carte fusionnée (RRF des requêtes) : 12 à 15 pages, ≤ 6 sections chacune,
  │     les lignes qui répondent (avec l'en-tête de leur tableau), la taille, « déjà lu »
  │
  ├─ 3. GLM — lire(lectures = [{chemin, sections}, …]) en UN appel → sections complètes
  │     SERVEUR — anomalies rapprochées des pages lues, injectées
  │
  └─ 4. GLM — répond ; coupes et citations vérifiées ; « citée sans être lue » relevé
```

**La carte n'envoie pas 12 à 15 pages.** Elle envoie 12 à 15 **résultats**, comme une page de
moteur de recherche : pour chaque page, une ligne (chemin, titre, taille, produit), les titres
de ses sections qui répondent, et pour les trois meilleures, les quelques lignes qui contiennent
les mots cherchés, avec l'en-tête du tableau. Cela fait ~1 000 car. par résultat. Aujourd'hui, la
recherche livre 6 pages **entières** ou leurs sections complètes (~54 k car.). La vraie carte de
la question TGY3731 (12 résultats, générée le 02/10) fait **12 k car.**, et son premier résultat
(830 car.) porte déjà la preuve :

```
1. /quincaillerie/soleal-gy-roulements-et-fermetures.md — Chariots, crémones… SOLEAL GY 55 (8 k car., 14 sections, score 1.00)
   §8 Systèmes de fermeture et verrouillage > 3. Fermeture pompier (ERP)
      - Référence : **TGY3731** (avec vis de fixation T770059).
      - Permet l'ouverture d'urgence depuis l'extérieur à l'aide d'une clé spéciale pompier.
   §6 … > 1. Gamme de crémones encastrées
      | Référence | Points de fermeture | Condamnation clé | Hauteur min. châssis | … |
…
3. /quincaillerie/askey-coulissant-65-nv-quincaillerie.md — Quincaillerie ASKEY Coulissant 65 NV (6 k car.)
   §8 Ce que la source ne donne pas
      * La source ne précise pas la référence de la clé de manœuvre d'urgence pour le bloc verrou W6050111 …
```

GLM ne lit ensuite que ce qu'il choisit : ici le §8 de la page 1, ou la page entière (8 k car.).
Pourquoi 12 résultats et pas 6 : un résultat coûte ~1 k car. là où une page livrée en coûte 5 à
15 k. Passer de 8 à 12 résultats fait monter la section désignée sur le banc de 89 à 93 %, et
« toutes » de 59 à 64 %. Avec 8 résultats (~10 k car.), le golden est déjà à 100 %.

Cible : 3 appels, 2 quand la fiche ou une page dominante suffit. Au plus 6 appels. Chaque
résultat d'outil se termine par l'état du tour, par exemple « appel 3/6 · lu 24 k sur 150 k
car. ». Le dernier appel part **sans outils** : le modèle répond avec ce qu'il a lu et dit ce
qui reste à vérifier.

## 3. Les mesures qui fondent ce choix (hors ligne, 02/10)

**La carte désigne la bonne section aussi bien que la livraison actuelle la contient**, pour un
tiers du volume.

| Ce qui arrive au modèle | Golden : preuve / toutes | Banc : preuve / toutes | Volume |
|---|---|---|---|
| Actuel : 6 pages livrées | 95 % / 40 % (contenue) | 91,1 % / 60,7 % (contenue) | 52-55 k |
| Carte 8 p. × 3 s., lignes qui répondent | 95 % / 25 % (dans les lignes) | 82,1 % / 42,9 % | **10 k** |
| Carte 12 p. × 5 s. : **section désignée** | 100 % / 40 % | 92,9 % / 64,3 % | 16-18 k |
| Idem, 4 requêtes fusionnées | — | 92,9 % / 69,6 % | ~17 k |
| Carte 15 p. × 8 s., 4 requêtes fusionnées | — | 92,9 % / **75,0 %** | ~20 k |

« Toutes » compte les questions dont **chaque** page de preuve est atteinte : c'est la mesure des
contradictions. La carte large fait mieux que la livraison actuelle (75 % contre 60,7 %), parce
qu'une ligne de carte coûte peu là où une page livrée coûte cher.

**Plusieurs requêtes dans un même appel valent mieux qu'une relance.** Sur le banc, la question
et ses 3 reformulations fusionnées (RRF) donnent, à volume égal, 94,6 % / 66,1 % au lieu de
91,1 % / 60,7 %. Une seule reformulation en mots-clés fait 89,3 % / 57,1 %. La fusion rattrape
une mauvaise formulation sans payer de relance, alors qu'une relance coûte un appel de plus et
renvoie tout le contexte.

**La référence exacte, avant toute recherche.** Sur les vraies références de pièces du banc (25
questions) : une référence n'apparaît que dans 2 sections en médiane, et sa recherche exacte touche
la section de preuve **20 fois sur 25**. 4 références sont absentes du wiki, et le serveur peut le
dire avant le premier appel. Le détecteur actuel doit être filtré par rareté : il prend
`perform76`, `2024`, `7016` ou `240` pour des références (médiane de 103 sections sur le golden).
Règle proposée : un jeton à chiffres présent dans 10 sections au plus est une référence ; au-delà,
c'est un terme.

**Page dominante.** Quand la 2e page vaut moins de la moitié de la 1re (8 questions sur 76), la
preuve est dans la 1re page 8 fois sur 8. Si cette page fait moins de 15 k car., elle est livrée
entière avec la carte, et on économise un appel (cas TGY3731 : 1,00 contre 0,21).

**L'index de navigation, c'est `wiki_llm/wiki/index.md`** (décidé le 02/10 : pas d'index
généré, on part du fichier du wiki). Il est dans le prompt permanent, et il se teste avec et sans.

Vérifié le 02/10 : 321 entrées pour 321 pages de contenu (les 318 pages plus les 3 registres
d'anomalies), aucune page absente, aucune entrée orpheline ; 2 descriptions et 34 titres diffèrent
de la frontmatter, à la marge (non regardés en détail). Il est groupé par dossier en 17
sections (Profilés 77, Documents sources 62, Procédures 54, Quincaillerie 45…), à ~322 car. par
entrée : 109 k car. ≈ 36 k tokens, soit 0,005 $ par appel en cache et ~0,05 $ à froid. Le
protocole du wiki (`wiki_llm/CLAUDE.md`) le prévoit déjà comme point d'entrée : « Read
`wiki/index.md` first to find relevant pages ».

| Variante écartée | Taille | Raison |
|---|---|---|
| Plan généré par produit (chemin + titre, + 4 tags) | 30-44 k car. | une vue de plus à maintenir, gain non prouvé |
| Carte mentale telle quelle | 78 k car. | pages en double (produit et métier) |
| Glossaire | 73 k car. | il reste une page qu'on cherche |

Ce que le fichier n'a pas : le regroupement par produit et les tags. Les tags (649) restent dans le
vocabulaire que l'application génère déjà (~1,3 k car.). Si le test montre que le regroupement par
produit aide, une vue générée s'ajoutera alors, et pas avant. `index.md` est classé « réservé » :
il faudra le lire dans l'instantané du wiki pour le prompt (`lire` continue de le refuser). Un
contrôle au chargement signale dans les logs toute page absente de l'index ou toute entrée qui
pointe sur rien.

**L'index oriente ; il ne remplace pas la recherche.** Lu comme un moteur lexical (titre,
description, section), `index.md` place la page attendue dans ses 5 premiers résultats pour 79 %
du golden et 62 % du banc (« toutes les pages attendues » : 26 % et 23 %). La recherche par
sections désigne la section de preuve pour 100 % et 93 %. La réponse vit dans les tableaux
(références, valeurs), qu'une ligne de 300 caractères ne peut pas porter. GLM lit le sens et fera
mieux qu'un score sur les mots : le chiffre réel est inconnu, c'est le test avec et sans qui le
donnera.

**Optimiser `index.md` : une piste à part, sur le wiki et pas sur le code.** Le fichier est la
copie des `description` de la frontmatter (le protocole impose qu'elles soient identiques), donc
l'optimiser, c'est réécrire des descriptions selon `wiki_llm/CLAUDE.md` (« Writing to be found »),
hors de l'application, puis déposer le wiki par `/admin`. Les ratés mesurés montrent où :
- un thème **enfoui dans une page longue**, absent de la description : « effet bilame »,
  « zones sismiques », « retombée de membrane », « température de soudage » ;
- les **pages jumelles** que la description ne distingue pas : `/gammes/perform.md` et
  `/commercial/perform.md` ;
- ce qu'**aucune description ne peut porter** : une valeur dans un tableau (76576, TGY3701,
  6106). C'est le rôle de la recherche lexicale, pas de l'index.

Un seul levier compte : pour chaque page, dire en une phrase **quelles questions elle tranche**,
dans les mots du métier et leurs synonymes. Avant/après : le même calcul lexical, sans modèle,
puis le test réel.

**Navigation descendante (page de gamme → sous-pages) : permise, sans rien ajouter à la stack.**
Avec l'index sous les yeux, GLM peut orienter sa recherche par la facette `gamme`/`systeme`,
lire directement une page que l'index désigne (`/profiles/perform76-parcloses.md`), ou ouvrir la
page de gamme pour ce qu'elle porte en propre : les dimensions maximales, les options. Les 11
pages de gamme font 4 à 18 k car. et ont 5 à 17 liens sortants. Ce n'est pas le chemin par
défaut, parce que chaque niveau coûte un appel. Un routeur à règles « intention → pages » a fait
moins bien que BM25 le 01/10 (46 % contre 57 %). C'est GLM qui choisit, question par question.
La carte affiche le produit de chaque résultat (« PERFORM › Système 76 ») pour départager les
familles jumelles.

**Ce que coûte le contexte qu'on garde.** Sur le tour réel TGY3731, le contenu livré pour la
première fois fait 81 % du coût, la relecture en cache 14 % et la sortie 5 %. Relire coûte 10 %
du prix ; c'est le volume livré qui coûte.

## 4. Les outils (trois, toujours)

```
chercher(requetes: [string] 1..4, gamme?: string, systeme?: string)
lire(lectures: [{chemin: string, sections?: [string]}] 1..6)
lire_anomalie(identifiant: string)
```

- **`chercher`** rend la carte. GLM écrit 2 à 4 formulations : les mots du wiki pour l'objet
  (famille, gamme, valeur), la référence seule, un synonyme pris dans le lexique. Le serveur les
  fusionne par RRF, et les références rares de la fiche s'y ajoutent d'office. La carte ne livre
  pas de page, sauf la page dominante courte.
- **`lire`** prend un lot : plusieurs pages et plusieurs sections en un seul appel (« §13 », un
  mot du titre, « sommaire » ; sans `sections`, la page entière dans la limite du budget). On
  passe par un tableau dans un seul appel plutôt que par des appels parallèles, parce que les
  appels d'outils streamés de GLM sont fragiles chez Mistral (identifiants qui changent,
  fragments). Si GLM en émet plusieurs quand même, ils sont traités.
- **`lire_anomalie`** reste tel quel.

Format de la carte (exemple réduit) :

```
===== CARTE — 3 requêtes fusionnées — 12 pages =====
1. /quincaillerie/soleal-gy-roulements-et-fermetures.md — Chariots, crémones… SOLEAL GY 55 (11 k car.) ★ dominante, livrée entière ci-dessous
2. /quincaillerie/roto-nx-accessoires-et-gabarits.md — Accessoires Roto NX (96 k car., 64 sections)
   §32 Serrures de condamnation OF
     | Référence | Désignation | … |
     | 489 112 | Serrure de condamnation à clé … |
   §62 Outils
…
Tour : appel 2/6 · lu 11 k sur 150 k car.
```

## 5. La boucle de raisonnement

- **Raisonnement renvoyé dans le tour** : le message assistant garde son bloc `thinking` avec ses
  `tool_calls`. L'API l'accepte (testé le 02/10). Entre deux questions, l'historique reste du
  texte seul, comme le veut l'usage chat de Z.ai (`clear_thinking=true`).
- **`reasoning_effort=high`** par défaut ; `max` à tester sur la faisabilité. La sortie pèse ~5 %
  du coût, donc `max` coûte surtout en latence.
- **`max_tokens` à 16 384**, et `finish_reason` relevé dans la trace : `length` devient visible.
- **Température** : 0,2 au départ, puis 1,0 (la recommandation de Z.ai) testée dans le même essai.
- **Le budget est affiché** dans chaque résultat d'outil, et le dernier appel part sans outils :
  il n'y a plus d'erreur « trop d'allers-retours ».
- **Robustesse** : le nom d'outil est normalisé (on coupe à la première espace ou au premier
  `<`), une réponse vide est comptée dans le journal, et l'option `strict` n'est jamais envoyée.
- La garde « aucune page lue → on renvoie lire une fois » est gardée : elle protège la règle 0,
  quel que soit le modèle.

## 5 bis. Ce qu'on garde dans le contexte

**Dans un tour : tout.** Le prompt permanent (index compris), le raisonnement, la carte et les
lectures. Retirer l'index après la recherche :

- **casserait le cache.** Il est en tête du prompt, donc tout ce qui suit perdrait son préfixe et
  serait refacturé au prix plein, 10 fois le prix en cache. Garder l'index (~36 k tokens en cache,
  ~0,005 $ par appel) coûte moins que de repayer le reste du contexte à neuf ;
- **priverait GLM de l'index au moment où il en a besoin** : après la carte, pour voir la page
  voisine (le DTA, l'autre gamme, le second versant d'une contradiction).

On ne compacte pas non plus la carte une fois lue : en cache, elle coûte 10 %, et la réécrire
ferait tout repayer au prix plein.

**D'une question à l'autre : un état compact.** C'est là que vaut l'idée « question, requête
reformulée, pages et sections trouvées ». Aujourd'hui, l'historique ne porte que le texte des
questions et des réponses. Il porterait aussi, sous chaque réponse, une ligne comme
« [requêtes : … · lu : /profiles/perform76-parcloses.md §3 §4] », soit ~200 car. par tour, tirée
de la trace déjà enregistrée. Une question de suite (« et pour 48 mm ? ») relit alors la bonne
section sans relancer de recherche. Le raisonnement des tours précédents n'est pas renvoyé
(usage chat de Z.ai).

## 6. Le prompt permanent, réécrit

Environ 10 à 12 k tokens, en cache, dans cet ordre :

1. **Rôle** : PROFERM, menuiserie, assistant documentaire du métier.
2. **L'index du wiki** : le contenu d'`index.md` tel quel (~36 k tokens), lu dans l'instantané. S'y
   ajoutent 10 à 15 lignes de conventions écrites à la main : où vivent les limites
   dimensionnelles (page de gamme, DTA), les parcloses (`/profiles/<gamme>-parcloses.md`), la
   quincaillerie, les registres d'anomalies. *À tester avec et sans l'index.*
3. **Lexique de recherche** (généré, déjà en place) : les gammes et leurs graphies, les systèmes
   et les tags (~1,3 k car., à étendre aux 649 tags, 7,5 k car.). Le glossaire (73 k car.) n'y
   entre pas : il reste une page qu'on cherche.
4. **Méthode**, en 4 temps : comprendre, une recherche riche, lire en une fois, répondre. Le
   critère d'arrêt : « quand la section qui porte l'objet de la question est lue, réponds. Ne
   relance que si la carte ne montre aucune section plausible, et jamais deux fois pour le même
   objet. Une ligne de carte situe une réponse ; elle ne la prouve pas : lis la section. »
5. **Règles de vérité condensées** : une ligne et un exemple pour chacune des règles actuelles
   2, 3, 4, 5, 6, 8, 9, 10, 13 et 14. Les règles de vérité ne changent pas ; elles perdent le ton
   « contre les fautes de Small ».
6. **Forme** : français du métier, concis, citations par chemin, tableau pour comparer.

`vocal_consignes.md` reste devant, pour le tour vocal.

## 7. Lots

| Lot | Contenu | Mesure |
|---|---|---|
| **0** | `mesurer_recuperation` v2 : section désignée, preuve dans les lignes, fusion de requêtes, fiche de la question | hors ligne, golden + banc |
| **1** | Index : lignes qui répondent avec en-tête (markdown et HTML), références exactes filtrées par rareté, fusion RRF, `carte()`, `lire_lot()` ; suppression de `livrer` et des références « à part » | hors ligne + tests |
| **2** | Tour : les 3 outils, la fiche de la question, le budget affiché, le raisonnement renvoyé, le dernier appel sans outils, `max_tokens` et `finish_reason`, la normalisation des noms d'outil, l'état compact d'un tour à l'autre (requêtes, pages et sections lues) | tests (modèle simulé) |
| **3** | Prompt : consignes réécrites, `index.md` et lexique ; contrôle de l'index au chargement ; une nouvelle clé de cache | 1 appel réel |
| **4** | Interface : étapes « Carte » et « Lecture », fenêtre de trace juste (appels, pages lues, car., coût), pastille « citée sans être lue » | — |
| **5** | Mesure réelle, sur accord : TGY3731 seule, puis 3 questions (référence, faisabilité, contradiction), puis le golden de 40 questions avec et sans l'index | appels GLM |
| **W** | Piste wiki, en parallèle et sans code : réécrire les `description` (donc `index.md`) selon `wiki_llm/CLAUDE.md`, en commençant par les ratés mesurés ; avant/après par le calcul lexical | hors ligne, puis lot 5 |

Le golden complet au coût cible coûte ~1 à 2 $. Une conversation par question.

## 8. Ce qu'on attend, et les risques

| | Aujourd'hui (GLM, TGY3731) | Cible (estimation) |
|---|---|---|
| Appels | 5 | 2 à 3 |
| Caractères livrés | 181 k | 25-35 k |
| Tokens d'entrée cumulés | 201 k | 35-50 k (prompt plus gros, mais en cache) |
| Coût | ~0,128 $ | ~0,02-0,03 $ |

Risques :

- **GLM répond sur la foi d'une ligne de carte sans lire.** Parade : la consigne, plus un
  contrôle déterministe qui marque une page citée mais jamais lue (on constate et on montre,
  comme pour les citations aujourd'hui).
- **Trois appels au lieu de deux** sur les questions simples, soit environ 2 s de plus. La page
  dominante et la fiche ramènent une partie de ces questions à deux appels.
- **Les lignes sont moins bonnes sur une question en langue naturelle** (banc : 82-86 % dans les
  lignes, 93 % en section désignée). C'est la lecture qui ferme l'écart, et la mesure réelle le
  dira.
- **L'index du wiki peut distraire**, comme un index de 35 000 caractères le 30/09 avec Small (c'est
  l'index qui lui faisait perdre la référence). `index.md` est trois fois plus gros, et avec GLM
  rien ne le prouve : d'où le test avec et sans.
- **Bogues d'hébergement chez Mistral** (appels d'outils, réponses vides) : le lot en un seul
  appel, la normalisation et le journal les couvrent.
