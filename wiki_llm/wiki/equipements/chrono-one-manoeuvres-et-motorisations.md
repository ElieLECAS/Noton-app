---
type: Équipement
title: "Chrono ONE SOPROFEN : treuils, cardans, sangle, tirage direct, moteurs radio et filaires"
description: "Manœuvres du Chrono ONE : treuils TX13, TX16, TXL10, cardans, tiges oscillantes, sangle, tirage direct, moteurs radio RTS, iO, M-Soft², autonomes solaires, Hybrid et filaires avec codes R1 à IOHS, largeurs L mini, poids maxi, multi-tabliers et manœuvre de secours."
tags: [volet-roulant, chrono-one, manoeuvre, moteur, treuil, cardan, sangle, tirage-direct, tige-oscillante, soprofen]
systeme: Chrono One
fournisseur: SOPROFEN
usage: chiffrage
famille: chrono-one
status: draft
sources:
  - resource: raw/moustiquaires/exports/132.pdf
    id: soprofen-guide-technique-blocs-baies-2022
    title: Guide technique Blocs-baies 2022, SOPROFEN, réf. DOC86505 (09/2022)
    last_modified: 2022-09-01
source_pages:
  - resource: raw/moustiquaires/exports/132.pdf
    pages: 20-26, 28-29
generated:
  by: process:claude-code
  at: 2026-10-09T02:30:00Z
---

La manœuvre est ce qui fait monter et descendre le tablier du volet roulant Chrono ONE de [SOPROFEN](/fournisseurs/soprofen.md) : une sangle, un treuil à manivelle, un tirage direct à la main, ou un moteur. Le guide technique la présente à la rubrique 16, « Type de manœuvre ». Un seul type de manœuvre par volet, à l'exception de la manœuvre de secours par un treuil sur un volet motorisé. La largeur L, la hauteur H et le poids du tablier limitent le choix : les limites de poids ci-dessous sont celles d'un tablier par manœuvre [1 p. 20 à 29]. Les émetteurs radio livrés avec les moteurs (M-Soft², Oximo RTS, S&So RS100 iO, Smoove collection couleurs) et leurs codes de commande sont sur [Émetteurs et codes de commande du Bloc LX SOPROFEN](/equipements/volet-roulant-bloc-lx-emetteurs-et-codes-de-commande.md) : le guide technique Blocs-baies 2022 donne les mêmes émetteurs et les mêmes codes pour le Chrono ONE aux pages 26 à 28.

# Manœuvres manuelles

## Treuils

Le treuil est associé à un cardan et à une tige oscillante, qui se choisissent aux paragraphes suivants. Largeur mini pour une manœuvre par treuil : 350 mm. Les trois treuils sont sans fin de course et avec verrous tasseurs TLX. Poids maxi du tablier en kilogrammes selon la limite de la certification NF CSTB (M1 et M2) ou la limite SOPROFEN [1 p. 20].

| Treuil | Sortie | Couple (N.m) | Rapport | Poids maxi, NF CSTB M1 (kg) | Poids maxi, NF CSTB M2 (kg) | Poids maxi, SOPROFEN (kg) |
| --- | --- | --- | --- | --- | --- | --- |
| TX13/5-7 | sous-face | 13 | 1/5,25 | 9 | 18 | - |
| TX16/8D-7 | sous-face | 16 | 1/8 | 9 | 18 | 24 |
| TXL10/4 | latérale, avec cardan de sortie latérale | 10,4 | 1/4 | 7 | 14 | - |

## Cardans

Le cardan se monte avec le treuil et s'utilise avec une tige oscillante de Ø 12. EM maxi est l'épaisseur de menuiserie maximale, en millimètres [1 p. 20].

| Cardan | Description | Utilisation | Code | EM maxi (mm) |
| --- | --- | --- | --- | --- |
| MCS 12 (gris) | cardan simple à 45°, support intégré | tige oscillante Ø 12 | T | 176 |
| MCD 12 (gris) | cardan relevable, support intégré | tige oscillante Ø 12 | TD | 176 |
| TSL10-KIT (blanc) | cardan simple, plaquette injectée PVC de 40 × 80 mm | tige oscillante Ø 12 | TSL | 126 (coffre de 200) ; 140 (coffre de 230) |

Les cotes EM ne tiennent pas compte d'un espace entre la plaquette du cardan et le nu intérieur de la menuiserie : pour un joint de finition intérieur filant, il faut déduire 10 mm aux dimensions indiquées (voir [Chrono ONE : coulisses, adaptateurs et joint de finition](/equipements/chrono-one-coulisses-et-adaptateurs.md), épaisseur de menuiserie) [1 p. 20].

![Treuils TX13/5-7, TX16/8D-7, TXL10/4 et cardans MCS12, MCD12, TSL10-KIT avec cotes EM et EC](/assets/equipements/volets-roulants/blocs-baies-2022/chrono-one/treuils-et-cardans-avec-cotes-em-et-ec.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 20)

## Tiges oscillantes

Sans indication, le guide livre le choix par défaut ; le code 0 désigne un volet sans tige oscillante [1 p. 21].

| Tige oscillante Ø 12 | Longueur (mm) | Coloris | Code |
| --- | --- | --- | --- |
| TO120/18 | 1 230 | blanc, beige, gris, brun, caramel | 8 (par défaut) |
| TO150/18 | 1 530 | blanc, beige, gris, brun, caramel | 9 |
| TO190/18 | 1 930 | blanc, beige, gris | 10 |

## Sangle

L'enrouleur de sangle BES 14/5,5 est un enrouleur de sangle pivotant, à poser en applique ; sangle de 15 mm de large et de 5,60 m de long ; code S ; EM maxi 176 mm (coffres de 200 et de 230). Largeur mini : 450 mm. Poids maxi du tablier : 6 kg (NF CSTB M1), 3,75 kg (NF CSTB M2, coffre de 230), 10 kg (SOPROFEN) [1 p. 21].

![Enrouleur de sangle BES 14/5,5, poids maxi du tablier et cote EM](/assets/equipements/volets-roulants/blocs-baies-2022/chrono-one/enrouleur-de-sangle-bes-14-5-5-et-cote-em.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 21)

## Tirage direct

Le tirage direct est une manœuvre manuelle avec axe à ressort compensateur CR50G, des poignées (saillie de 24 mm) sur la lame finale, une lame intermédiaire avec serrure à clés, et une canne de tirage fournie. Il ne se fait qu'avec un tablier en lame aluminium LA37 ou LATH37 sans ajours. Largeur L de 720 à 1 500 mm. Poids maxi du tablier : 7 kg pour une largeur de 720 à 900 mm, 12 kg pour une largeur de 901 à 1 500 mm [1 p. 22].

La lame intermédiaire se place à environ 1 m du bas, ou juste au-dessus de la lame finale. Le code TGDF impose la lame intermédiaire au-dessus de la lame finale ; le code TGD est le code du tirage direct [1 p. 22].

| Position de la lame intermédiaire | Coffre | Hauteur | Disposition |
| --- | --- | --- | --- |
| à environ 1 m du bas | 200 | de 1 600 à 1 900 mm maxi | standard |
| à environ 1 m du bas | 230 | de 1 600 à 2 500 mm maxi | standard |
| juste au-dessus de la lame finale | 200 | inférieure à 1 600 mm, ou supérieure à 1 900 mm jusqu'à 2 200 mm maxi | standard |
| juste au-dessus de la lame finale | 200 | de 1 600 à 1 900 mm maxi | option, code TGDF |
| juste au-dessus de la lame finale | 230 | inférieure à 1 600 mm, ou supérieure à 2 450 mm jusqu'à 2 550 mm maxi | standard |
| juste au-dessus de la lame finale | 230 | de 1 600 à 2 500 mm maxi | option, code TGDF |

Coloris de la lame intermédiaire et des poignées de tirage en fonction du tablier : lame intermédiaire blanche pour un tablier blanc ou LATH37 ; brune pour brun ; beige pour beige et beige 1015 ; gris métal pour les autres coloris ; caramel pour chêne doré ; noire 9005 pour noir 9005. Poignées de tirage sur la lame finale : blanches pour une lame finale blanche ; brunes pour brune ; grises anodisées pour grise 7038 et les autres coloris ; caramel pour chêne doré ou caramel ; RAL identique à la lame finale [1 p. 22].

![Tirage direct : lame intermédiaire à environ 1 m du bas et juste au-dessus de la lame finale](/assets/equipements/volets-roulants/blocs-baies-2022/chrono-one/tirage-direct-lame-intermediaire-a-1-m-et-au-dessus-de-la-lame-finale.png)

![Tirage direct : poignées et canne de tirage](/assets/equipements/volets-roulants/blocs-baies-2022/chrono-one/tirage-direct-poignees-ptd-et-canne-de-tirage.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 22)

# Manœuvres motorisées

Les codes ci-dessous sont ceux à écrire sur la commande. Les commandes radio RTS, iO et M-SOFT 2 sont incompatibles entre elles [1 p. 24 à 26].

## Largeur L mini de chaque moteur

Largeur L mini en millimètres, pour un tablier. Un moteur court est le moteur de petite largeur, utilisé quand la largeur L est faible ; sa largeur maximale est donnée. Le tiret signale une case sans valeur [1 p. 23].

| Commande | Code | Moteur court : L mini (mm) | Moteur court : L maxi (mm) | Moteur normal : L mini (mm) |
| --- | --- | --- | --- | --- |
| Radio RTS, Oximo S RTS ou Oximo RTS | R1 | 500 | 789 | 790 |
| Radio RTS avec manœuvre de secours | R1S | 735 | 934 | 935 |
| Radio RTS autonome, Oximo 40 WF RTS (panneau déporté) | A35 | - | - | 610 |
| Radio iO, Oximo S iO ou Oximo iO | IO | 500 | 789 | 790 |
| Radio iO, Oximo S iO ou S&So RS100 IO | IO1 | 500 | 599 | 600 |
| Radio iO, S&So RS100 IO avec manœuvre de secours | IO1S | - | - | 745 |
| Radio iO autonome, Oximo 40 WF IO (panneau déporté) | A45 | - | - | 610 |
| Radio M-Soft², MVMC 06/23 ou MVM | R11 | 540 | 619 | 620 |
| Radio M-Soft² autonome (panneau déporté) | A55 | - | - | 600 |
| Radio SIMU autonome (panneau déporté) | A65 | - | - | 610 |
| Filaire WT, Ilmo S 12 trs (court MVC06) ou Ilmo 12 tours | M1 | 500 | 739 | 740 |
| Filaire WT avec manœuvre de secours | M1S | 735 | - | 885 |
| Filaire M-Soft², MVEC 06/23 ou MVEF | M11 | 540 | 619 | 620 |
| Hybrid filaire et radio, S&So RS100 IO Hybrid | IOH | - | - | 600 |
| Hybrid avec manœuvre de secours | IOHS | - | - | 745 |

Les moteurs autonomes (A35, A45, A55, A65) sont des moteurs sur batterie rechargée par un panneau solaire déporté. Pour le S&So RS100 IO, le guide ajoute « Oximo iO pour moteur > 20 Nm » [1 p. 23, 25].

## Plusieurs tabliers

On choisit entre un ou deux moteurs pour deux ou trois tabliers ; seuls des tabliers de même hauteur peuvent avoir une manœuvre commune. Poids maximal en kilogrammes de tablier avec une manœuvre, lames de 37 mm [1 p. 23].

| Moteur | Largeur L mini, côté moteur (mm) | H maxi (mm) | Poids maxi, 2 tabliers (kg) | Poids maxi, 3 tabliers (kg) |
| --- | --- | --- | --- | --- |
| Radio MVM M-Soft² (20 N.m) | 760 | 3 300 | 26,4 | 21,8 |
| Radio MVM M-Soft² (30 N.m) | 750 | 3 300 | 39,7 | 32,7 |
| Radio Oximo RTS ou Oximo iO (30 N.m) | 890 | 3 300 | 39,7 | 32,7 |
| Radio S&So RS100 IO (20 N.m) | 780 | 3 300 | 26,4 | 21,8 |
| Filaire MVE M-Soft² (20 N.m) | 760 | 3 300 | 26,4 | 21,8 |
| Filaire MVE M-Soft² (30 N.m) | 750 | 3 300 | 39,7 | 32,7 |
| Filaire Ilmo (30 N.m) | 890 | 3 300 | 39,7 | 32,7 |
| Filaire Ilmo 12 tours (20 N.m) | 840 | 3 300 | 26,4 | 21,8 |
| Hybrid (20 N.m) | 780 | 3 300 | 26,4 | 21,8 |

## Moteurs à commande radio

Tous les moteurs radio ont un verrouillage automatique du tablier par verrous TLX, un arrêt sur obstacle à la descente et une recherche automatique des fins de course. Un récepteur radio est intégré au moteur [1 p. 24, 25].

| Code | Moteur | Ce que le guide imprime |
| --- | --- | --- |
| R1 | Oximo RTS (MVO) | récepteur radio RTS intégré ; émetteur individuel MVO-ESM pré-programmé fourni en standard ; jusqu'à 12 points de commande ; câble de 3 m fourni. Petites largeurs L : moteur radio court Oximo S RTS (MVCR06/17), avec platine électronique radio commandée dans la joue du coffre |
| R1S | Oximo RTS ou moteur radio court Oximo S RTS | comme R1, avec une manœuvre de secours |
| A35 | autonome Oximo 40 WF RTS | fonctionne à l'énergie solaire captée par un panneau solaire et stockée dans une batterie intégrée dans le coffre ; récepteur radio RTS intégré, jusqu'à 12 points de commande ; émetteur individuel MVO-ESM préprogrammé fourni ; TLX et arrêt sur obstacle à la descente ; recherche automatique des fins de course |
| A65 | autonome SIMU | comme A35 avec l'émetteur individuel MVSI-EM1C préprogrammé |
| IO | Oximo iO (MVIO) | récepteur radio iO intégré ; émetteur individuel MViO-ESM pré-programmé ; jusqu'à 9 points de commande ; câble de 3 m fourni en standard. Petites largeurs : moteur radio court Oximo S IO (MVIO06/17P), avec platine électronique radio commandée dans la joue du coffre |
| IOS | Oximo iO ou moteur radio court Oximo S iO | comme IO, avec une manœuvre de secours |
| IO1 | S&So RS100 iO (MVRS) | récepteur radio iO intégré ; émetteur individuel MViO-ESM pré-programmé ; jusqu'à 9 points de commande ; câble de 3 m fourni. Petites largeurs : moteur radio court Oximo S IO (MVIO06/17P) |
| IO1S | S&So RS100 iO\* ou moteur radio court Oximo S iO | comme IO1, avec une manœuvre de secours ; l'astérisque n'a pas de note (**VER-138**) |
| A45 | autonome Oximo WF Radio IO | comme A35, avec récepteur radio IO et l'émetteur individuel MVIO-ESM préprogrammé |
| R11 | M-Soft² (MVM) | récepteur radio intégré ; émetteur individuel Monotec blanc pré-programmé ; jusqu'à 16 points de commande ; câble de 3 m fourni en standard ; petites largeurs possibles avec le moteur radio court MVM 06 ; pas de manœuvre de secours possible |

Moteurs autonomes : le panneau solaire se fixe par le poseur lors de l'installation du volet au chantier. Il nécessite obligatoirement un ensoleillement direct pour assurer l'autonomie du système ; il est connecté lors de l'installation pour maintenir la batterie en charge ; la batterie est intégrée dans le coffre et livrée chargée. Conditions : respecter les limites d'utilisation du tarif ; pas de manœuvre de secours possible ; uniquement avec un seul tablier en lames de 37 mm ; non compatible avec les multi-tabliers. L'accessoire indispensable pour l'installateur est le chargeur de batterie, réf. MVOS-ALI ; toute utilisation d'un autre chargeur est interdite ; le temps de charge maximal est de 3 h 30. Un stockage prolongé à l'abri de la lumière peut nécessiter de recharger la batterie [1 p. 24, 25].

![Panneau solaire du moteur A35 : bons emplacements et mauvaises expositions](/assets/equipements/volets-roulants/blocs-baies-2022/chrono-one/panneau-solaire-a35-emplacements-a-eviter-et-a-choisir.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 24)

![Panneau solaire du moteur A45 : bons emplacements et mauvaises expositions](/assets/equipements/volets-roulants/blocs-baies-2022/chrono-one/panneau-solaire-a45-emplacements-a-eviter-et-a-choisir.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 25)

![Schéma d'un moteur à commande radio : émetteur radio, moteur 230 V 50 Hz](/assets/equipements/volets-roulants/blocs-baies-2022/chrono-one/schema-d-un-moteur-a-commande-radio.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 24)

## Moteurs Hybrid, filaires et radio iO

Le moteur IOH, S&So RS100 IO Hybrid, est dédié au fonctionnement par une commande individuelle filaire ; il comporte un récepteur radio IO pour pouvoir être piloté par une télécommande radio IO complémentaire ou générale. Il est fourni sans inverseur ni télécommande, à commander en accessoires ; il fonctionne en filaire avec les principaux inverseurs individuels et mécaniques (à position fixe ou momentanée) pour moteur 230 V pour volet roulant. Il gère jusqu'à 9 points de commande, avec verrouillage automatique TLX et arrêt sur obstacle à la descente ; recherche automatique des fins de course ; câble de 3 m fourni en standard. Le moteur IOHS est le même avec une manœuvre de secours [1 p. 26].

![Moteur Hybrid : trois schémas de câblage, commande individuelle et émetteur radio iO](/assets/equipements/volets-roulants/blocs-baies-2022/chrono-one/moteur-hybrid-trois-schemas-de-cablage.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 26)

## Moteurs à commande filaire

Possibilité de piloter radio les moteurs filaires avec un récepteur RRE et un ou plusieurs émetteurs au choix, à commander en accessoires. Les moteurs filaires sont fournis sans inverseur ; voir le tarif pour le choix des inverseurs. Tous les inverseurs ne conviennent pas pour des commandes multiples : SOPROFEN demande à être consultée pour étude [1 p. 29].

| Code | Moteur | Ce que le guide imprime |
| --- | --- | --- |
| M1 | Ilmo 12 tours | commande filaire ; fonctionnement à couple réduit avec arrêt sur obstacle à la descente ; verrouillage automatique TLX en série ; arrêts automatiques aux fins de course sur appui et butoirs |
| M1S | Ilmo 12 tours | comme M1, avec une manœuvre de secours |
| M10 | M-Soft² | commande filaire ; arrêts automatiques aux fins de course sur appui et butoirs par détection électronique ; pas de manœuvre de secours possible ; arrêt sur obstacle à la descente et à la montée ; verrouillage automatique TLX en série ; câble de 3 m fourni |

Le tableau des largeurs de la page 23 donne au moteur filaire M-Soft² le code M11, alors que la page 29 l'appelle M10 (**INC-334**).

## Manœuvre de secours

La manœuvre de secours est un treuil monté du côté opposé au moteur, avec un cardan MCS12 (code « T1 »). Elle est fournie avec une manivelle (Ø 12) amovible et des adaptateurs MA-DEC-12 et MA-DE12-A. Elle n'est possible qu'avec un seul tablier. Elle n'est disponible qu'en moteur Somfy. Le code « T1 » est écrit ainsi alors que le cardan MCS12 porte le code « T » à la page 20 (**INC-335**) [1 p. 29].

# Ce que la source ne donne pas

La largeur maximale du moteur court Ilmo avec manœuvre de secours (M1S) n'est pas donnée, ni la description du moteur filaire Ilmo simple (codes M et MS) pour le Chrono ONE. L'astérisque de « S&So RS100 iO\* » n'a pas de note (**VER-138**).

# Citations

[1] Guide technique Blocs-baies 2022, SOPROFEN, réf. DOC86505 (09/2022) — `raw/moustiquaires/exports/132.pdf`, p. 20 à 26, 28 et 29
(le numéro de page imprimé est celui du PDF)

# Voir aussi

- [Volet roulant Chrono ONE SOPROFEN : cotes, lames, tabliers et coloris du coffre](/equipements/chrono-one-cotes-tabliers-et-coloris.md)
- [Émetteurs et codes de commande du Bloc LX SOPROFEN : M-Soft², Oximo RTS, iO, Smoove](/equipements/volet-roulant-bloc-lx-emetteurs-et-codes-de-commande.md)
- [Manœuvres et motorisations du Bloc LX](/equipements/volet-roulant-bloc-lx-manoeuvres-et-motorisations.md)
- [Chrono ONE : options, renforts, brides et croquage](/equipements/chrono-one-options-renforts-et-croquage.md)
- [SOPROFEN](/fournisseurs/soprofen.md)
- [SOMFY](/fournisseurs/somfy.md)
- [Guide technique Blocs-baies 2022, SOPROFEN (148 pages)](/sources/soprofen-guide-technique-blocs-baies-2022.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
