---
type: Équipement
title: "Manœuvres et motorisations du volet roulant Bloc LX SOPROFEN : treuil, moteurs, secours"
description: "Treuil T et T10, moteurs filaires Ilmo, radio Oximo RTS et iO, M-Soft², Hybrid, codes de commande et largeurs L mini, manœuvre de secours, côté de manœuvre G ou D, déport par barrettes BTD, tiges oscillantes de 12 et 15 mm et conseils de mise en œuvre du Bloc LX SOPROFEN."
resource: https://www.soprofen.com
tags: [volet-roulant, bloc-lx, motorisation, treuil, somfy, radio, tige-oscillante, soprofen]
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
    pages: 4-8, 11-12, 14
  - resource: raw/moustiquaires/exports/132.pdf
    pages: 75-77, 80-81
generated:
  by: process:claude-code
  at: 2026-10-09T02:30:00Z
---

La manœuvre du volet roulant [Bloc LX](/equipements/volet-roulant-bloc-lx-demi-linteau.md) de
[SOPROFEN](/fournisseurs/soprofen.md) est soit un treuil (une manœuvre à manivelle), soit un
moteur à commande filaire ou radio. La rubrique 7 du bon de commande en donne le type et le
coloris : **un seul type de manœuvre par volet, à l'exception de la manœuvre de secours par un
treuil d'un volet motorisé** [1 p. 4].

# Moteurs, codes et largeurs L mini

Largeur L mini de chaque moteur, en millimètres, relevée sur le tableau « Type de manœuvre
motorisé » du guide. Le code est celui qu'on écrit sur la commande ; « court » désigne le moteur de
petite largeur, dont la largeur maximale est donnée. La largeur L est la largeur de dos à dos des
coulisses ([cotes de fabrication](/equipements/volet-roulant-bloc-lx-demi-linteau.md)) [1 p. 4].

| Commande | Moteur | Code | Largeur L mini (mm) | Largeur maxi du moteur court (mm) |
| --- | --- | --- | --- | --- |
| Radio RTS | Oximo court RTS | R1 | 500 | 799 |
| Radio RTS | Oximo RTS | R1 | 750 | - |
| Radio RTS | Oximo court RTS avec manœuvre de secours | R1S | 600 | 944 |
| Radio RTS | Oximo RTS avec manœuvre de secours | R1S | 850 | - |
| Radio iO | Oximo court IO | IO1 | 500 | 659 |
| Radio iO | S&SO RS100 IO | IO1 | 610 | - |
| Radio iO | Oximo court IO avec manœuvre de secours | IO1S | 600 | 804 |
| Radio iO | S&SO RS100 IO avec manœuvre de secours | IO1S | 710 | - |
| Radio M-SOFT² | M-Soft² MVM court | R11 | 530 | 659 |
| Radio M-SOFT² | M-Soft² MVM | R11 | 610 | - |
| Filaire WT | Ilmo court | M | 500 | 699 |
| Filaire WT | Ilmo | M | 650 | - |
| Filaire WT | Ilmo court avec manœuvre de secours | MS | 600 | 844 |
| Filaire WT | Ilmo avec manœuvre de secours | MS | 750 | - |
| Filaire WT | Ilmo 12 trs | M1 | 650 | - |
| Filaire WT | Ilmo 12 trs avec manœuvre de secours | M1S | 750 | - |
| Filaire M-SOFT² | M-Soft² MVEC 06/23 | M11 | 530 | - |
| Filaire M-SOFT² | M-Soft² MVE | M11 | 610 | - |
| Hybrid filaire et radio iO | S&SO RS100 IO Hybrid | IOH | 610 | - |
| Hybrid filaire et radio iO | S&SO RS100 IO Hybrid avec manœuvre de secours | IOHS | 710 | - |

(schéma: raw/moustiquaires/exports/133.pdf, p. 4)

![Bon de commande, rubrique 7 : type de manœuvre](/assets/equipements/volets-roulants/bloc-lx/bon-de-commande-rubrique-7-type-de-manoeuvre.png)

Le guide ne nomme pas le fabricant de chaque moteur : il imprime les logos RTS, iO, WT et M-SOFT² en
regard des familles, et « SOMFY » ou « M-SOFT 2 » au-dessus des deux câbles de réglage (voir les
conseils de mise en œuvre). La fiche du
[TRADI](/equipements/volets-roulants-tradi-manoeuvres-et-garanties.md) range les moteurs « Radio RTS
et Oximo IO » et « Radio IO S&SO RS100 IO et IO Hybrid » chez [SOMFY](/fournisseurs/somfy.md), et les
moteurs « M-Soft² » chez SOPROFEN.

# Les manœuvres manuelles

Les descriptions détaillées sont renvoyées aux plans de fin de guide. Les limites de fabrication
sont indiquées sur les grilles du tarif « à l'aide d'un petit t » [1 p. 5].

- T, comme treuil : treuil TX13/5-7 sans fin de course, avec un verrouillage automatique TLX en
  série. Avec un cardan double CS20/12-7 (tige de 500 mm en 6 pans de 7) pour une sortie sur la
  retombée inférieure du coffre. Par défaut, avec un cardan, une tige oscillante ∅ 12 mm et un clip
  en coloris blanc. Pour une tige oscillante, un clip et un cardan bruns, noter le code « 9 » en
  colonne coloris [1 p. 5].
- T10, comme treuil à fin de course : treuil TX13/5-7FC à fin de course et attaches tablier
  souples [1 p. 5].

**La manœuvre par treuil est incompatible avec les demi-linteaux sans espace latéral (réservation)
pour le déport, tel que le demi-linteau STRADAL** [1 p. 5].

![Manœuvre par treuil : vue éclatée](/assets/equipements/volets-roulants/bloc-lx/manoeuvre-par-treuil-vue-eclatee.png)

# Les manœuvres motorisées

## Moteurs à commande filaire

Moteurs filaires, 230 V, 50 Hz. Les moteurs filaires sont fournis sans inverseur (voir le tarif
pour le choix des inverseurs) ; tous les inverseurs ne conviennent pas pour des commandes multiples,
auquel cas SOPROFEN demande de la consulter pour étude. La zone accessoire de la commande sert à
commander les inverseurs et les relais RMC, CD4… Un pilotage radio des moteurs filaires est possible
avec un récepteur RRE et un ou des émetteurs au choix, à commander en accessoires ; **la commande
radio RTS n'est pas compatible avec les commandes iO ou M-SOFT 2** [1 p. 5, 8, 11].

![Schéma d'un moteur à commande filaire et de son inverseur](/assets/equipements/volets-roulants/bloc-lx/schema-moteur-a-commande-filaire.png)

| Code | Moteur | Ce que le guide imprime |
| --- | --- | --- |
| M | Ilmo (MVI) | commande filaire ; arrêt automatique aux fins de course sur appui et butoirs par détection électronique ; verrouillage automatique TLX en série ; arrêt sur obstacle et protection contre le gel ; petites largeurs avec le moteur filaire court MVC06/17, recherche automatique des fins de course, platine électronique filaire dans la joue du coffre (**INC-317**) [1 p. 5, 8] |
| MS | Ilmo | comme M, avec une manœuvre de secours [1 p. 5, 8] |
| M1 | Ilmo 12 tours | commande filaire ; fonctionnement à couple réduit avec arrêt sur obstacle à la descente ; verrouillage automatique TLX en série ; arrêts automatiques aux fins de course sur appui et butoirs [1 p. 5, 8] |
| M1S | Ilmo 12 tours | comme M1, avec une manœuvre de secours [1 p. 5, 8] |
| M11 | M-Soft² | commande filaire ; arrêts automatiques aux fins de course sur appui et butoirs par détection électronique ; arrêt sur obstacle à la descente et à la montée ; verrouillage automatique TLX en série ; câble de 3 m fourni ; petites largeurs possibles avec le moteur filaire court M-Soft² ; platine électronique filaire dans la joue du coffre ; pas de manœuvre de secours possible [1 p. 5, 8] |
| C | câble seul | « variante sans interrupteur individuel » (rubrique 8, voir [Émetteurs et codes de commande](/equipements/volet-roulant-bloc-lx-emetteurs-et-codes-de-commande.md)) [1 p. 11] |

## Moteurs à commande radio

Moteurs 230 V, 50 Hz, avec un émetteur radio. **Les commandes radio RTS, iO et M-SOFT 2 sont
incompatibles entre elles** [1 p. 6].

![Schéma d'un moteur à commande radio](/assets/equipements/volets-roulants/bloc-lx/schema-moteur-a-commande-radio.png)

| Code | Moteur | Ce que le guide imprime |
| --- | --- | --- |
| R1 | Oximo RTS (MVO) | récepteur radio RTS intégré au moteur ; émetteur individuel MVO-ESM pré-programmé fourni en standard ; gère jusqu'à 12 points de commande ; verrouillage automatique TLX et arrêt sur obstacle à la descente ; recherche automatique des fins de course ; câble de 3 m fourni. Petites largeurs L : moteur radio court MVCR06/17 / Oximo S RTS, platine électronique radio commandée dans la joue du coffre [1 p. 6] |
| R1S | Oximo RTS ou MVCR 06/17 | comme R1, avec une manœuvre de secours [1 p. 6] |
| IO1 | S&So RS100 iO (MVRS) | récepteur radio iO intégré ; émetteur individuel MViO-ESM pré-programmé fourni en standard ; gère jusqu'à 9 points de commande ; verrouillage automatique TLX, arrêt sur obstacle à la descente ; recherche automatique des fins de course ; câble de 3 m fourni en standard. Petites largeurs : moteur radio court Oximo S IO / MVIO06/17P, platine électronique radio commandée dans la joue du coffre [1 p. 6] |
| IO1S | S&So RS100 iO\* ou moteur radio court Oximo S iO | comme IO1, avec une manœuvre de secours ; l'astérisque n'a pas de note (**VER-138**) [1 p. 6] |
| R11 | M-Soft 2 (MVM) | récepteur radio intégré ; émetteur individuel Solo-soft blanc pré-programmé fourni en standard ; gère jusqu'à 16 points de commande ; verrouillage automatique TLX, arrêt sur obstacle ; recherche automatique des fins de course ; câble de 3 m fourni. Petites largeurs possibles avec le moteur radio court MVM 06. Pas de manœuvre de secours possible [1 p. 6] |

## Moteurs à commande Hybrid, filaire et radio iO

IOH, S&SO RS100 IO Hybrid : dédié au fonctionnement par une commande individuelle filaire, ce
moteur comporte un récepteur radio iO pour pouvoir être piloté par une télécommande radio iO
complémentaire ou générale. Il est fourni sans inverseur ni télécommande, à commander en
accessoires ; il fonctionne en filaire avec les principaux inverseurs individuels et mécaniques (à
position fixe ou maintenue) pour moteur 230 V pour volet roulant. Il gère jusqu'à 9 points de
commande, avec verrouillage automatique TLX et arrêt sur obstacle à la descente ; recherche
automatique des fins de course ; câble de 3 m fourni en standard. IOHS : le même moteur avec une
manœuvre de secours [1 p. 7].

Trois schémas de câblage sont dessinés, en mm² : secteur 230 V = 50 Hz en 3 × 1,5 jusqu'à une boîte
de jonction, puis 4 × 0,75 vers le moteur ; commande individuelle en 3 × 0,75 ; troisième schéma : un
émetteur iO individuel intérieur, et une commande individuelle à inverseur à clé extérieur en
4 × 0,75 avec son secteur 3 × 1,5 ; un émetteur iO de commande générale les pilote tous [1 p. 7].

![Moteur Hybrid : commande filaire individuelle et émetteur radio iO](/assets/equipements/volets-roulants/bloc-lx/schema-moteur-hybrid-filaire-et-radio-io.png)

![Moteur Hybrid : trois schémas de câblage](/assets/equipements/volets-roulants/bloc-lx/schemas-de-cablage-moteur-hybrid.png)

# La manœuvre de secours

La manœuvre de secours est un treuil monté du côté opposé au moteur. Elle est fournie avec une
manivelle amovible et un adaptateur MA-DEC. Il faut indiquer la position de sortie du cadran (le
guide renvoie au « chapitre F ») ; elle n'est possible qu'avec un seul tablier [1 p. 5].

La vue éclatée de la manœuvre de secours nomme : le moteur MVX (ex. MVX6/17) ; le différentiel 410727
(à clipper sur la sortie du moteur) ; la roue MVX-R50 (à clipper sur le différentiel, et à monter sur
la bague de guidage) ; le tube de liaison TMS en acier 16 × 16, épaisseur 1 mm (à fixer sur la
sortie du différentiel et à emboîter dans le palier escamotable) ; la bague de guidage (livrée avec
le palier escamotable) ; le palier escamotable 410731 ; le manchon d'accouplement ATX/3 ; le cardan
CS20-7 ; l'adaptateur MA-DEC ; la tige oscillante TO../.. de diamètre 15 mm [1 p. 5].

![La manœuvre de secours : vue éclatée, cardan CS20-7, adaptateur MA-DEC et tige oscillante](/assets/equipements/volets-roulants/bloc-lx/manoeuvre-de-secours-vue-eclatee-et-cardan.png)

![Moteur filaire et console : vue éclatée](/assets/equipements/volets-roulants/bloc-lx/moteur-filaire-et-console-vue-eclatee.png)

# Côté de manœuvre (rubrique 9)

On indique le code du côté de la manœuvre ; le côté de manœuvre est toujours vu de l'intérieur.
Dans le cas d'une manœuvre de secours, on indique uniquement le côté moteur, G ou D [1 p. 11].

| Côté de manœuvre | Code |
| --- | --- |
| à gauche | G |
| à droite | D |

![Côté de manœuvre : G et D, vus de l'intérieur](/assets/equipements/volets-roulants/bloc-lx/cote-de-manoeuvre-g-et-d-vue-de-l-interieur.png)

# Déport de manœuvre (rubrique 10)

Le déport s'indique par le code correspondant au nombre de barrettes BTD (appelées « entretoises
BTD » dans le tableau) pour la manœuvre par treuil, ou le treuil de secours. Il permet d'adapter
latéralement la position de sortie du cardan afin que la tige oscillante ne gêne pas l'ouverture du
battant. Pas de déport pour le côté moteur [1 p. 11].

Code de la position du support, en millimètres.

| Position | A | B | C |
| --- | --- | --- | --- |
| Nombre d'entretoises BTD | 7 | 9 | 11 |
| DC, distance de dos de la coulisse à l'axe de la tige oscillante (mm) | 15 | 25 | 35 |
| R, réservation nécessaire dans le coffre pour le débord côté de la manœuvre (mm) | 50 | 60 | 70 |

La coupe porte « R mini = 50 (si pos. A) » et « R = 70 (standard) ». **Il est impératif de vérifier
la réservation du gros œuvre, qui est variable selon le type de demi-linteau** ; la manœuvre par
treuil est incompatible avec les demi-linteaux sans espace latéral pour le déport, tel que le
STRADAL [1 p. 11].

![Déport de manœuvre : console et barrettes BTD](/assets/equipements/volets-roulants/bloc-lx/deport-de-manoeuvre-console-et-barrettes-btd.png)

![Coupe du déport : réservation R et distance DC](/assets/equipements/volets-roulants/bloc-lx/coupe-du-deport-reservation-r-dc.png)

(schéma: raw/moustiquaires/exports/133.pdf, p. 11)

Le guide technique de 2022 porte la même rubrique 10 (trois codes A à C) et, sur son bon de commande, cinq codes A à E : D (distance DC 45 mm, R mini 80 mm) et E (50 mm, 85 mm) s'ajoutent, et le code C y est donné avec R mini 60 mm ; voir [Bons de commande des blocs-baies SOPROFEN (09/2022) : colonnes et codes](/equipements/blocs-baies-bons-de-commande-soprofen.md) et l'anomalie **INC-354** [2 p. 80, 139].

# Tiges oscillantes (rubrique 12)

La tige oscillante est la tige qui prolonge le cardan du treuil jusqu'à la main. **Sans indication,
c'est le choix standard ; code 0 : sans tige oscillante.** Par défaut, la tige oscillante est de
coloris blanc ; pour une tige oscillante, un clip et un cardan bruns, noter le code 9 en colonne
coloris de la manœuvre. La tige oscillante est de diamètre 12 mm, en acier revêtu polyester ; la
manivelle a des poignées en polypropylène, le bras et le manchon sont en acier laqué [1 p. 12].

| Code | Tige oscillante ∅ 12 (longueur en mm) | Coloris blanc W | Coloris brun BR |
| --- | --- | --- | --- |
| 8 | TO 120/18 (1200) | par défaut | autre choix |
| 9 | TO 150/18 (1500) | autre choix | par défaut |
| 10 | TO 190/18 (1900) | autre choix | autre choix |

La grille imprime « par défaut » en brun pour le code 9, alors que le texte dit la tige blanche par
défaut (**VER-140**). **Toutes les manœuvres de secours sont fournies avec des tiges oscillantes de
15 mm** [1 p. 12].

## Support de cardan selon l'épaisseur du doublage

Code du support de cardan en fonction de l'épaisseur du doublage, en millimètres. **Les supports
sont en fourniture seule** [1 p. 12].

| Épaisseur du doublage (mm) | 100 | 120 | 140 | 160 |
| --- | --- | --- | --- | --- |
| Code du support | 1 | 2 | 3 | 4 |
| Support de cardan (dessin) | DLSC10 / DLTC10 | DLSC12 / DLTC12 | DLSC14 / DLTC14 | DLSC16 / DLTC16 |

![Supports de cardan DLSC et DLTC selon l'épaisseur du doublage](/assets/equipements/volets-roulants/bloc-lx/supports-de-cardan-dlsc-et-dltc-selon-l-epaisseur-du-doublage.png)

(schéma: raw/moustiquaires/exports/133.pdf, p. 12)

# Conseils de mise en œuvre

Toujours suivre la notice fournie avec les volets motorisés [1 p. 14].

- Motorisation et programmation : alimentation des moteurs 230 V / 50 Hz (3 × 1,5 mm²) à relier à la
  terre ; câbles moteurs de 2,5 m de long (filaire) et de 3 m (radio) (**INC-320**) ; s'assurer d'avoir
  un boîtier de dérivation toujours accessible à proximité du volet [1 p. 14].
- Pour effectuer facilement et rapidement les réglages et programmations complémentaires en toute
  sécurité, utiliser le « câble de réglage tout en un » : pour les opérateurs RTS, iO électroniques
  filaires WT et mécaniques filaires, le câble [SOMFY](/fournisseurs/somfy.md) de référence 015971,
  avec indicateur secteur et bornes de connexion rapide ; pour les opérateurs M-Soft² électroniques
  filaires et radio, avec bornes de connexion rapide, le câble de référence MVM-CC, **non compatible
  avec le moteur Hybrid RS100**. Les deux se commandent en accessoires [1 p. 14].

![Câble de réglage Somfy, référence 015971](/assets/equipements/volets-roulants/bloc-lx/cable-de-reglage-somfy-015971.png)

![Câble de réglage M-Soft 2, référence MVM-CC](/assets/equipements/volets-roulants/bloc-lx/cable-de-reglage-m-soft-2-mvm-cc.png)

# Citations

[1] Guide technique Bloc LX, SOPROFEN, extrait du catalogue — `raw/moustiquaires/exports/133.pdf`,
p. 4 à 8, 11, 12 et 14 (numérotation du PDF ; pages imprimées 113 à 118, 120, 121 et 123)
[2] Guide technique Blocs-baies 2022, SOPROFEN, réf. DOC86505 (09/2022) — `raw/moustiquaires/exports/132.pdf`, p. 80 et 139

# Voir aussi

- [Volet roulant Bloc LX : cotes, tabliers, lames, coulisses et lames finales](/equipements/volet-roulant-bloc-lx-demi-linteau.md)
- [Émetteurs et codes de commande du Bloc LX](/equipements/volet-roulant-bloc-lx-emetteurs-et-codes-de-commande.md)
- [Options du Bloc LX](/equipements/volet-roulant-bloc-lx-options.md)
- [Intégration du Bloc LX dans les demi-linteaux](/equipements/volet-roulant-bloc-lx-integration-demi-linteaux.md)
- [SOPROFEN](/fournisseurs/soprofen.md)
- [SOMFY](/fournisseurs/somfy.md)
- [Guide technique volet roulant demi-linteau Bloc LX SOPROFEN](/sources/soprofen-guide-technique-bloc-lx.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
