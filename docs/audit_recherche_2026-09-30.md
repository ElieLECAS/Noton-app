# Audit de la recherche du chat — 30/09/2026

Déclencheur : « crémone 3 points TGY3702, je cherche la 4 points pour coulissant ». Juste le
22/09 (TGY3704, page Technal SOLEAL GY), fausse le 30/09 : 771923 lu dans la page Roto NX en
local, gamme ASKEY en prod ; au tour suivant (« chez Technal »), appel d'outil écrit en texte
puis réponse inventée citant neuf pages qui n'existent pas.

Tout ce qui suit est mesuré (logs, base, rejeu du premier appel contre Mistral, index hors
ligne). Scripts de mesure : scratchpad de la session, non versionnés.

## 1. Ce qui a changé entre le 22/09 et le 30/09

Le code du chat n'a pas bougé. Le wiki, si : 196 → 297 pages (retraitement Système 70,
catalogue Roto NX).

| | 23/09 | 30/09 |
| --- | --- | --- |
| Pages indexées | 196 | 297 |
| Entrées d'anomalie | 77 | 479 |
| Prompt permanent | 13 567 car. (~4 600 tokens) | 44 170 car. (~15 100 tokens) |
| dont index des anomalies | 4 388 car. | **34 840 car.** |
| Tour « TGY3702 » (base, conv. 877 / 1054) | 41–62 k tokens d'entrée | **221 k tokens** |

## 2. Cause 1 — la référence de la question n'arrive pas à la recherche

Le 22/09, le modèle cherchait `crémone 4 points coulissant TGY3702` : la page Technal sort
première. Le 30/09 il cherche `crémone 4 points coulissant référence` : la référence a disparu.

Premier appel rejoué 8 fois par configuration (même code, mêmes consignes) :

| Configuration du prompt | TGY3702 dans la requête |
| --- | --- |
| Wiki du 23/09 | 7 / 8 |
| Wiki du 30/09 | **0 / 8** |
| Wiki du 30/09, index des anomalies réduit à 77 entrées | 0 / 8 |
| Wiki du 30/09, sans index des anomalies | 4 / 8 |

Le prompt a déplacé la décision, mais aucune configuration ne la rend sûre : **garder la
référence dépend de la discipline du modèle.** Or c'est le signal le plus fort qu'un menuisier
donne, et l'index le connaît exactement (TGY3702 figure dans une seule page).

Rang de la page Technal pour les requêtes réellement écrites par le modèle :

| Requête du modèle | Telle quelle | + `tgy3702` |
| --- | --- | --- |
| crémone 4 points coulissant référence (Quincaillerie) | 8e | **1re, livrée entière** |
| crémone 4 points coulissant 65 NV ASKEY référence (prod) | absente | **1re, livrée entière** |
| crémone 4 points coulissant Technal (Quincaillerie) | 4e | **1re, livrée entière** |

Aggravant côté wiki : la page qui porte la réponse s'intitule « Roulements, fermetures et
manœuvres Technal SOLEAL GY 55 », sans le mot crémone, et son tag est `cremona`.

## 3. Cause 2 — le volume : des pages géantes livrées entières

`chercher` livre les trois premières pages **entières**. Le choix tenait avec des pages de
5 à 20 k caractères. Aujourd'hui :

- 25 pages dépassent 50 000 car., 9 dépassent 100 000 (Roto NX crémones 141 k, Système 70
  plans de combinaison 129 k, Roto NX aperçu côté P 124 k…) ;
- sur les 62 questions du golden prises comme requête, **une seule recherche** livre
  médiane 64 k car., p90 137 k, max 194 k ;
- tour TGY3702 du 30/09 : 4 appels, dernier contexte 276 k car. Le modèle y a pris une valeur
  de la page Roto NX et l'a attribuée à une référence Technal.

## 4. Cause 3 — l'appel d'outil écrit en texte, et la relance qui l'entretient

Au deuxième tour, le modèle écrit l'appel en texte au lieu de l'émettre (2 / 8 au rejeu avec le
wiki du 30/09, 0 / 8 avec celui du 23/09, 0 / 8 sans l'index des anomalies). La relance
(« tu n'as chargé aucune page ») remet ce texte **dans le contexte** comme message assistant :
le second appel le poursuit (`chercher"]"}]}]{"mots_cles": …}}]}}]…`), puis invente une
réponse. Comme aucune page n'a été lue, le tour se conclut quand même, et l'écran montre le
tout.

## 5. Les anomalies avant la recherche

Avant même le premier appel, le serveur injecte les entrées rapprochées de la **question
seule** : 60 questions du golden sur 62 en reçoivent 6. Le modèle lit donc des anomalies avant
d'avoir cherché quoi que ce soit — et l'index permanent lui en listait 479 de plus.

## 6. Décision appliquée (30/09)

« Je pose une question et il cherche dans le wiki » : plus aucune anomalie dans le prompt.

- l'index des anomalies sort du prompt permanent ;
- plus d'injection avant la première recherche ; les entrées rapprochées des pages lues
  arrivent avec elles (règle 2 inchangée).

Mesuré après (mêmes rejeux, 8 par tour) :

| | Avant | Après |
| --- | --- | --- |
| Prompt permanent | 44 170 car. | **9 265 car.** (~3 200 tokens) |
| Premier appel du tour 1 | 20 745 tokens | **6 929 tokens** |
| Tour 2 : appel d'outil réel | 6 / 8 (1 fuite, 1 vide) | **8 / 8** |
| Tour 1 : TGY3702 gardé dans la requête | 0 / 8 | 3 / 8 |

La fuite disparaît ; la référence reste perdue plus d'une fois sur deux. R1 reste nécessaire.

Le golden lancé avant le changement (3 questions en parallèle) s'est arrêté sur des **429**
de Mistral : à 100–250 k tokens par tour, trois tours simultanés dépassent le débit autorisé.
En prod, trois utilisateurs en même temps produisent la même chose.

## 7. Propositions — à valider une par une, chacune mesurée avant / après

| # | Où | Changement | Ce qu'il corrige |
| --- | --- | --- | --- |
| R1 | code | Les références de la question (tokens connus de l'index) sont ajoutées à toute requête `chercher` qui les omet | Cause 1 : ne dépend plus du modèle ; mesuré ci-dessus |
| R2 | code | La relance ne remet pas le texte raté dans le contexte ; un tour qui finit sans aucune page lue et avec un appel écrit en texte affiche un message clair au lieu de la réponse | Cause 3 |
| R3 | wiki | Découper les pages de plus de ~20 000 car. selon le protocole (une donnée = une page) : Roto NX, Système 70 d'abord | Cause 2, à la source ; « on ne raccourcit jamais le wiki » reste vrai, on le range |
| R4 | code | Lint : pages au-delà du budget, tags quasi-doublons (`cremona` / `cremone`), dans l'onglet Wiki de l'administration | Empêche le retour de la cause 2 et des tags fautifs |
| R5 | wiki | Titre et tags de la page Technal SOLEAL GY : le mot crémone | La page remonte même sans référence |
| R6 | code | Facette `fournisseur` dans `chercher` (le frontmatter la porte déjà ; le modèle bricole `tags=technal`) | Relances du type « chez Technal » |
| R7 | éval | Golden « navigation » : 15–20 questions à référence sur plusieurs fournisseurs (Technal, ASKEY, Roto, profine), avec relances | Le golden actuel est 100 % PERFORM76 : il ne voit pas ces régressions |
| R8 | code | Pluriel ramené au singulier dans `tokenise` (crémone / crémones, renfort / renforts) — à mesurer, le classement bouge partout | La recherche est exacte au caractère près : « crémone » ne trouve pas le titre « Crémones Roto NX » |

Ordre proposé : R7 (pour mesurer), R1 + R2, R3 + R5 (wiki), puis R4, R6, R8.

## 8. Faut-il une passe sur le wiki ?

Oui, mais **sur la forme, pas sur le fond** : la valeur juste (TGY3704) était dans la page ;
c'est sa taille, son titre et ses tags qui l'empêchaient de sortir. Et la passe ne corrige pas
la cause 1 : une page parfaite reste introuvable si la requête perd la référence (R1).

| Chantier | État mesuré | Règle à écrire dans `wiki_llm/CLAUDE.md` |
| --- | --- | --- |
| Taille | 25 pages > 50 k car., 9 > 100 k ; médiane 11,9 k | Une page ≤ ~20 k car. ; au-delà, une page par famille ou par tableau |
| Titre et description | La page Technal ne dit pas « crémone » ; son titre décrit une rubrique de catalogue | Le titre nomme les produits dont la page donne les références, avec les mots du menuisier |
| Tags | 736 tags distincts, 385 employés une seule fois, 24 couples singulier / pluriel (renfort / renforts, profile / profilé / profilés…), fautes (cremona) | Liste fermée de tags au singulier ; un tag nouveau s'ajoute à la liste avant usage |

Ordre : pages Roto NX (5 pages de 90 à 141 k), puis Système 70 (6 pages de 80 à 129 k), puis la
normalisation des tags sur tout le wiki, que le lint R4 tient ensuite.
