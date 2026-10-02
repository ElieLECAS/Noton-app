# Étape 1 — un wiki qui se trouve : descriptions, titres, tags et `index.md` (plan, 02/10/2026)

Première des trois étapes de la refonte « GLM first » (`docs/plan_glm_first_2026-10-02.md`) :
1. le wiki se trouve ; 2. GLM récupère vite les bonnes pages et sections ; 3. la génération.

`index.md` sera lu en entier par GLM, dans le prompt permanent, pour choisir où chercher. Chaque
ligne y est la copie de la `description` d'une page (le protocole impose l'identité). Optimiser
`index.md`, c'est donc écrire, page par page, un titre, une description et des tags qui disent
**quelles questions la page tranche**, et qui la distinguent de ses voisines.

Rien n'est fait. Aucun appel au modèle dans cette étape. Le protocole d'écriture qui fait foi
reste `wiki_llm/CLAUDE.md` (*Writing to be found*, *Page format*, *Lint*).

## 1. Périmètre

**On touche :**

- la frontmatter : `title` (90 car. au plus), `description` (300 car. au plus), `tags` (4 à 10,
  liste fermée `wiki_llm/tags.md`), et `gamme` / `systeme` quand elles manquent ;
- `index.md`, recopié à l'identique ;
- `log.md`, une entrée par lot ;
- `wiki_llm/CLAUDE.md`, *Writing to be found* : le texte décrit encore la livraison d'avant le
  01/10 (« The first three results come back whole »). Il doit dire que la description est lue
  par le modèle dans `index.md` pour choisir ses pages. Ce changement de protocole est à valider.

**On ne touche pas :** le corps des pages (tableaux, phrases, titres de sections), les découpes
(la découpe des 88 pages de plus de 20 k car. reste en pause, décidée le 30/09), et les pages
Technal. Les 13 fiches Technal ne sont pas retraitées : leurs descriptions seront refaites au
retraitement.

Conséquence : les extraits de preuve du golden et du banc ne bougent pas, et les mesures avant et
après restent comparables.

## 2. Situation de départ (mesurée le 02/10, sans modèle)

**Conformité au protocole** (321 pages, dont 256 de contenu hors `sources/` et `anomalies/`) :

| Écart | Pages |
|---|---|
| `description` de plus de 300 car. (jusqu'à 683) | 55 |
| Nombre de tags hors 4-10 | 81 |
| Tag hors liste fermée | 0 |
| Titre de plus de 90 car. | 2 |
| Titre d'`index.md` différent de la frontmatter | 34 (ex. « Terminologie et légendes profine » contre « Terminologie, légendes et calcul d'une cote d'élément, directives profine ») |
| Description d'`index.md` différente de la frontmatter | 2 |
| Page de contenu sans `gamme` ni `systeme` | 24 |

**Trouvabilité par l'index.** On lit `index.md` comme un moteur lexical, avec le titre, la
description et la section du fichier. C'est un indicateur à la louche : GLM lit le sens. La
recherche par sections sert de comparaison.

| Jeu | Questions | Index : page attendue top 5 / top 10 | Recherche : top 5 / top 10 |
|---|---|---|---|
| Golden | 38 | 79 % / 84 % (toutes : 26 % / 39 %) | 89 % / 97 % |
| Banc du 30/09 | 56 | 62 % / 71 % (toutes : 23 % / 34 %) | 93 % / 95 % |
| Tenu à l'écart (gammes 70/76) | 10 | 60 % / 80 % (toutes : 0 % / 10 %) | 90 % / 90 % |

Le jeu tenu à l'écart vient de `docs/questions_gammes_70_76_2026-09-30.md`. Ses 20 questions n'ont
jamais été posées à un modèle ; 10 ont des sources que le calcul lit automatiquement.

**Les ratés** (page attendue hors des 10 premières lignes de l'index) : 124 cas sur 72 pages.
**18 pages hors Technal font la moitié des cas (61).**

| Ratés | Page |
|---|---|
| 12 | `/gammes/perform.md` |
| 5 | `/gammes/coulissants-aluminium.md` |
| 5 | `/certifications/dta-6-16-2334.md` |
| 4 | `/reference/glossaire.md` |
| 4 | `/profiles/systeme-76-profiles-principaux.md` |
| 4 | `/vitrages/performances-vitrages.md` |
| 3 | `/commercial/perform.md` |
| 3 | `/procedures/percage-montage-roto-nx.md` |
| 3 | `/procedures/directives-generales-systeme-70-evo2008.md` |
| 2 | `/quincaillerie/poignees-et-croisillons.md`, `/quincaillerie/roto-nx-champs-application.md`, `/fournisseurs/roto.md`, `/procedures/fabrication-profiles-pvc.md`, `/garanties/garanties-par-composant.md`, `/certifications/labels-et-certifications.md`, `/profiles/perform76-dormants.md`, `/procedures/fabrication-coulissant-askey-65-nv.md`, `/profiles/systeme-70-profiles-et-renforts.md` |

Les ratés tombent dans trois familles :

- **la page généraliste à description trop courte**, qui porte pourtant beaucoup de réponses :
  la page PERFORM (Uw, dimensions, différences entre le 70 et le 76, options), le glossaire, la
  page des coulissants alu ;
- **le thème enfoui** qu'aucune description ne nomme : « effet bilame », « zones sismiques »,
  « retombée de membrane », « température de soudage » ;
- **les pages jumelles** qu'une ligne ne départage pas. 126 paires ont des lignes d'index presque
  identiques. Les plus serrées sont les séries du système 70 (inerties VA1 à VC5, abaques VA2 et
  VA3) : pages légitimes, mais dont l'axe qui les distingue doit ouvrir la description. Puis
  `/gammes/perform.md` et `/commercial/perform.md`.

Ce que l'index ne peut pas faire, et ne doit pas chercher à faire : porter une valeur de tableau
(76576, TGY3701, 6106). C'est le travail de la recherche lexicale.

## 3. La règle d'écriture d'une description

Elle complète la règle 3 de *Writing to be found*, sans la contredire :

1. **Elle dit les questions que la page tranche**, dans les mots du métier : les familles
   couvertes, la plage de références, la nature des valeurs (cote, charge, Uw, épaisseur de
   vitrage, compatibilité, procédure).
2. **Elle ouvre sur ce qui la distingue de ses jumelles** : la classe (VA2 et VA3), le système
   (70 et 76), l'usage (gamme et argumentaire commercial), ouvrant et dormant.
3. **Elle nomme les thèmes que la page est seule à porter**, même enfouis dans une section :
   « dont l'effet bilame des panneaux ».
4. **Chaque mot est pris dans la page.** Un synonyme du métier n'entre que s'il est sourcé
   (règle 5) ; sinon il va dans les tags. Un script vérifie que les mots de la description
   figurent dans le corps de la page.
5. **Elle est écrite depuis la page, jamais depuis les questions de test.** Celui qui écrit lit
   la page entière (sommaire, tableaux) et ne voit ni le golden ni le banc. C'est ce qui empêche
   d'apprendre le test par cœur.
6. **300 caractères au plus, une phrase.** Une description qui veut tout dire fait remonter sa
   page pour toutes les questions et noie les autres. C'est le risque à mesurer.

Titres : objets, puis produit ou système, puis fournisseur (règle 2), 90 car. au plus. Tags : 4 à
10, dans la liste fermée ; un tag nouveau entre d'abord dans `tags.md`.

## 4. Déroulé

**Lot 1.0 — outillage de mesure, lecture seule.** Le calcul de ce document devient un script :
indicateur de l'index, recherche par sections, ratés, paires jumelles, conformité, et contrôle
« mots de la description présents dans la page ». Il rejoint `mesurer_recuperation` sans rien
changer à l'application. Il doit aussi lire les 20 questions tenues à l'écart, et pas seulement
10 : le format de leurs sources est à harmoniser.

**Lot 1.1 — conformité, sans réécrire.**

- Recopier dans `index.md` les 34 titres et 2 descriptions de la frontmatter.
- Signaler, sans les corriger, les 55 descriptions trop longues et les 81 nombres de tags hors
  fourchette : elles seront réécrites aux lots 1.2 et 1.3.
- Mesurer : c'est la nouvelle base.

**Lot 1.2 — pilote, 18 pages.** Les 18 pages du tableau, plus leurs jumelles directes (par
exemple `/commercial/perform.md` pour `/gammes/perform.md`). Pour chaque page :

1. lire la page entière ;
2. écrire titre, description et tags, en suivant la règle du § 3 ;
3. montrer l'avant et l'après ;
4. recopier dans `index.md` ;
5. ajouter l'entrée de `log.md`.

Écrit par Claude, ou par des agents sous ses consignes puis relu par Claude. Les 18 couples
avant/après te sont montrés avant le dépôt.

**Critère de passage du pilote** (proposé, à valider) :

- **index** : au moins la moitié des 61 cas visés entrent dans le top 10, et aucune page déjà
  dans le top 5 n'en sort ;
- **recherche** : `mesurer_recuperation` ne recule pas (aujourd'hui, preuve livrée 95 % sur le
  golden et 91,1 % sur le banc, toutes 40 % et 60,7 %), et on ne perd pas plus d'une question.
  Le titre, les tags et la description pèsent ×5, ×4 et ×3 dans le score de page : une
  description trop large ferait remonter sa page partout ;
- **jeu tenu à l'écart** : pas pire qu'avant. S'il ne bouge pas alors que le golden monte, on a
  appris le test, et on s'arrête ;
- **lint** : les écarts au protocole des 18 pages tombent à zéro.

**Lot 1.3 — généralisation, par famille.** Seulement si le pilote passe. On suit l'ordre du
protocole : `profiles/` et `quincaillerie/`, puis `procedures/`, puis les gammes et le reste,
`sources/` en dernier. Les séries du système 70 sont traitées en une fois, pour que leurs
descriptions se départagent. On mesure et on dépose à chaque famille. Technal attend son
retraitement.

**Lot 1.4 — protocole.** On met à jour *Writing to be found* : le rôle de l'index lu par GLM,
la règle du § 3, et la description de la livraison actuelle. Le groupe *Findability* du lint y
gagne le contrôle « mots de la description présents dans la page ».

## 5. Ce qu'on attend, ce qu'on ne promet pas

- L'indicateur lexical de l'index doit monter. GLM lit le sens et fera mieux que lui, mais le
  chiffre réel ne viendra qu'à l'étape 2, au test avec et sans l'index.
- La recherche par sections ne doit pas reculer. Elle peut monter un peu, puisque le score de
  page pèse dans le classement des sections.
- L'étape ne règle pas les valeurs absentes du wiki (Technal non retraité), ni les pages trop
  grosses (découpe en pause).

Coût : aucun appel au modèle. Le pilote tient en une demi-journée. La généralisation (~250 pages
hors Technal) se fait par familles et se dépose par `/admin` au fil des lots.

## 6. Décisions attendues

1. **Le critère de passage du pilote** (§ 4) : les seuils te conviennent-ils ?
2. **Qui écrit** : Claude seul (plus lent, une seule main) ou des agents relus par Claude (plus
   rapide, plus de relecture).
3. **Le changement de *Writing to be found*** (lot 1.4) : maintenant, ou à la fin de l'étape 2,
   quand la nouvelle livraison existera.
4. **L'outillage du lot 1.0** : un script dans `app/scripts/` (lecture seule, comme
   `mesurer_recuperation`), ou seulement dans un brouillon hors du dépôt.
