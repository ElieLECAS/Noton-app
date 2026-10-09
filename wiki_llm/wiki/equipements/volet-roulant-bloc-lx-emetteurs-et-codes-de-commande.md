---
type: Équipement
title: "Émetteurs et codes de commande du Bloc LX SOPROFEN : M-Soft², Oximo RTS, iO, Smoove"
description: Codes des émetteurs muraux des moteurs radio du Bloc LX SOPROFEN (M-Soft² 68 à 65, Oximo RTS SM et 7 à 10, S&So RS100 iO QM, DM, S1P2…), Smoove collection couleurs (codes TM à PG par cadre mural), moteur Hybrid sans émetteur et câble seul code C.
resource: https://www.soprofen.com
tags: [volet-roulant, bloc-lx, motorisation, radio, somfy, emetteur, soprofen]
systeme: Bloc LX
fournisseur: [SOPROFEN, SOMFY]
usage: chiffrage
famille: bloc-lx
status: draft
sources:
  - resource: raw/moustiquaires/exports/133.pdf
    id: soprofen-guide-technique-bloc-lx
    title: Guide technique Bloc LX, extrait du catalogue SOPROFEN (pages imprimées 110 à 125)
    last_modified: 2024-01-01
  - resource: raw/moustiquaires/exports/132.pdf
    id: soprofen-guide-technique-blocs-baies-2022
    title: Guide technique Blocs-baies 2022, SOPROFEN, réf. DOC86505 (09/2022)
    last_modified: 2022-09-01
source_pages:
  - resource: raw/moustiquaires/exports/133.pdf
    pages: 9-11
  - resource: raw/moustiquaires/exports/132.pdf
    pages: 78-80
generated:
  by: process:claude-code
  at: 2026-10-09T02:30:00Z
---

La rubrique 8 du bon de commande du volet roulant
[Bloc LX](/equipements/volet-roulant-bloc-lx-demi-linteau.md) de [SOPROFEN](/fournisseurs/soprofen.md)
permet de choisir le type de commande des moteurs radio, c'est-à-dire l'émetteur mural (la
télécommande fixée au mur) livré avec le moteur. Chaque émetteur a un code, qu'on écrit dans la
colonne « type de commande moteur » de la commande. Le moteur, ses codes R1, IO1, R11 et sa largeur
minimale sont sur [Manœuvres et motorisations](/equipements/volet-roulant-bloc-lx-manoeuvres-et-motorisations.md).
Les commandes radio RTS, iO et M-SOFT 2 sont incompatibles entre elles [1 p. 9, 10].

![Bon de commande, rubrique 8 : type de commande de moteur](/assets/equipements/volets-roulants/bloc-lx/bon-de-commande-rubrique-8-type-de-commande-de-moteur.png)

# Émetteur de série et émetteurs au choix

Un tableau par famille de moteur. Chaque ligne donne un émetteur et son code. L'émetteur « par
défaut » est celui livré si on n'écrit rien d'autre. Pour les émetteurs « autres choix », le guide
renvoie, pour la description détaillée, au tarif (« page des émetteurs complémentaires ») [1 p. 9].

## Moteur radio M-Soft 2 (code R11)

L'émetteur par défaut est livré « avec 1 émetteur radio mural / portable pour la commande
individuelle d'un moteur MVM » ; il est à programmer par vos soins au chantier [1 p. 9].

| Émetteur | Code | Remarque |
| --- | --- | --- |
| MVM-E1CS Monotec blanc | 68 | par défaut |
| MVM STM 2N, noir | 61 | autre choix |
| MVM STM 2GM, gris métal | 62 | autre choix |
| MVM MTP 2W, Mono soft blanc | 63 | autre choix |
| MVM MTP 2GM, Mono soft gris métal | 64 | autre choix |
| MVM MTP 2GT, Mono soft gris titane | 65 | autre choix |

![Émetteur mural MVM-E1CS Monotec blanc](/assets/equipements/volets-roulants/bloc-lx/emetteur-mural-mvm-e1cs-monotec-blanc.png)

## Moteur radio Oximo RTS (codes R1 et R1S)

L'émetteur par défaut est livré « avec 1 émetteur radio mural MVO-ESM pour la commande individuelle
pré-programmée en usine d'un moteur MVO ou MVCR » [1 p. 9].

| Émetteur | Code | Remarque |
| --- | --- | --- |
| MVO-ESM (pure) | SM | par défaut |
| Situo 1 RTS, Pure 2 | 7 | autre choix |
| Situo 1 RTS, Iron 2 | 8 | autre choix |
| Situo 1 RTS, Artic 2 | 9 | autre choix |
| Situo 1 RTS, Naturel 2 | 10 | autre choix |

![Émetteur mural MVO-ESM pure](/assets/equipements/volets-roulants/bloc-lx/emetteur-mural-mvo-esm-pure.png)

## Moteur radio S&So RS100 (codes IO1 et IO1S)

L'émetteur par défaut est livré « avec 1 émetteur radio mural MVRS-MM1 pour la commande individuelle
pré-programmée en usine d'un moteur MVIO ou MVIO 06/17P ». **Il est impératif d'avoir 1 commande
individuelle MVRS-MM1 par moteur : c'est la seule télécommande permettant les réglages et
programmations individuels et complémentaires** [1 p. 9].

| Émetteur | Code | Remarque |
| --- | --- | --- |
| MVRS-MM1 (pure) | QM | par défaut |
| Smoove IO, Pure MVRS-ESM | DM | autre choix |
| Situo 1 IO, Pure 2 | S1P2 | autre choix |
| Situo 1 IO, Iron 2 | S1F | autre choix |
| Situo 1 IO, Artic 2 | S1A | autre choix |
| Situo 1 IO, Naturel 2 | S1N | autre choix |

![Émetteur mural MVRS-MM1 pure](/assets/equipements/volets-roulants/bloc-lx/emetteur-mural-mvrs-mm1-pure.png)

## Moteur Hybrid, et sans émetteur

Les moteurs Hybrid sont fournis sans inverseur ni télécommande ; selon le besoin et la
configuration des commandes des volets, les inverseurs et télécommandes sont à commander en
accessoires. Sans émetteur, on écrit le code 0 : si l'émetteur individuel correspondant au type
de motorisation est commandé en accessoire ou déjà existant, la programmation sera réalisée par vos
soins [1 p. 9, 10].

(schéma: raw/moustiquaires/exports/133.pdf, p. 9)

# Smoove collection couleurs

La collection couleurs Smoove est la gamme d'émetteurs muraux Smoove proposée avec un choix de
cadres. L'émetteur se commande avec deux choix : le module central (la partie qui porte les
touches, sans cadre) et le cadre mural. Le guide le présente pour les moteurs radio Oximo RTS
(codes R1 et R1S) et Oximo iO (« codes IO et IOS », écrits ainsi : **INC-321**) [1 p. 10].

Modules centraux : blanc, noir, acier. Cadres muraux : Soprofen blanc (exemple « blanc » imprimé),
Somfy noir mat (exemple « noir mat ») ; huit cadres au choix : Soprofen blanc, noir mat, acier
laqué, acier mat, bambou clair, bambou foncé, merisier, noyer [1 p. 10].

![Modules centraux Smoove : blanc, noir, acier](/assets/equipements/volets-roulants/bloc-lx/modules-centraux-smoove-blanc-noir-acier.png)

![Cadres muraux Smoove collection couleurs](/assets/equipements/volets-roulants/bloc-lx/cadres-muraux-smoove-collection-couleurs.png)

Code de l'émetteur selon le module central (ligne) et le cadre mural (colonne). Un tiret signifie
que cette combinaison n'existe pas. Le code marqué (\*) est celui de l'émetteur par défaut. Les
lignes « Silence » valent « uniquement pour moteurs S&SO-RS100 IO » [1 p. 10].

| Module central | Soprofen blanc | Noir mat | Acier laqué | Acier mat | Bambou clair | Bambou foncé | Merisier | Noyer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Radio RTS, mécanique 1 canal RTS blanc | SM (\*) | - | - | - | - | - | - | - |
| Radio RTS, tactile 1 canal RTS blanc | TM | TA | TB | TC | TD | TE | TF | TG |
| Radio RTS, tactile 1 canal RTS noir | AM | AA | AB | AC | AD | AE | AF | AG |
| Radio RTS, tactile 1 canal RTS acier | BM | BA | BB | BC | BD | BE | BF | BG |
| Radio iO, mécanique 1 canal iO blanc | DM (\*) | - | - | - | - | - | - | - |
| Radio iO, tactile 1 canal iO blanc | EM | EA | EB | EC | ED | EE | EF | EG |
| Radio iO, tactile 1 canal iO noir | FM | FA | FB | FC | FD | FE | FF | FG |
| Radio iO, tactile 1 canal iO acier | GM | GA | GB | GC | GD | GE | GF | GG |
| Radio iO, tactile 1 canal iO Silence blanc | MM | MA | MB | MC | MD | ME | MF | MG |
| Radio iO, tactile 1 canal iO Silence noir | NM | NA | NB | NC | ND | NE | NF | NG |
| Radio iO, tactile 1 canal iO Silence acier | PM | PA | PB | PC | PD | PE | PF | PG |

(schéma: raw/moustiquaires/exports/133.pdf, p. 10)

# Moteurs filaires : câble seul

Pour les moteurs filaires, le seul choix de commande du guide est le câble seul, code C,
« variante sans interrupteur individuel ». Les moteurs filaires sont fournis sans inverseur (voir
le tarif pour le choix des inverseurs). Tous les inverseurs ne conviennent pas pour des commandes
multiples : SOPROFEN demande à être consultée pour étude. La zone accessoire sert à commander les
inverseurs, les relais RMC, CD4… [1 p. 11].

![Moteur filaire, câble seul, variante sans interrupteur individuel, code C](/assets/equipements/volets-roulants/bloc-lx/moteur-filaire-cable-seul-variante-c.png)

# Citations

[1] Guide technique Bloc LX, SOPROFEN, extrait du catalogue — `raw/moustiquaires/exports/133.pdf`,
p. 9 à 11 (numérotation du PDF ; pages imprimées 118 à 120)

# Voir aussi

- [Volet roulant Bloc LX : cotes, tabliers, lames, coulisses et lames finales](/equipements/volet-roulant-bloc-lx-demi-linteau.md)
- [Manœuvres et motorisations du Bloc LX](/equipements/volet-roulant-bloc-lx-manoeuvres-et-motorisations.md)
- [SOPROFEN](/fournisseurs/soprofen.md)
- [SOMFY](/fournisseurs/somfy.md)
- [Guide technique volet roulant demi-linteau Bloc LX SOPROFEN](/sources/soprofen-guide-technique-bloc-lx.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
