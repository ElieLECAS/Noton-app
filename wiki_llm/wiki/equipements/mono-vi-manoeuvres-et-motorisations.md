---
type: Équipement
title: "Mono VI SOPROFEN : moteurs radio, filaires, Hybrid, solaire A35, secours, treuil et tiges"
description: "Manœuvres du Mono VI : moteurs R1, R1S, A35, IO1, IO1S, R11, M, MS, M1, M1S, M11, IOH, IOHS avec largeurs L mini, multi-tabliers, câbles, manœuvre de secours, treuil avec surlongueur de trappe de 52 mm, côté de manœuvre, tiges oscillantes Ø 12 et type de commande moteur."
tags: [volet-roulant, mono-vi, volet-inverse, manoeuvre, moteur, treuil, tige-oscillante, radio, somfy, soprofen]
systeme: Mono VI
fournisseur: SOPROFEN
usage: chiffrage
famille: mono-vi
status: draft
sources:
  - resource: raw/moustiquaires/exports/132.pdf
    id: soprofen-guide-technique-blocs-baies-2022
    title: Guide technique Blocs-baies 2022, SOPROFEN, réf. DOC86505 (09/2022)
    last_modified: 2022-09-01
source_pages:
  - resource: raw/moustiquaires/exports/132.pdf
    pages: 129-135
generated:
  by: process:claude-code
  at: 2026-10-09T02:30:00Z
---

Les rubriques 10 à 15 du bon de commande du volet roulant inversé Mono VI de [SOPROFEN](/fournisseurs/soprofen.md) traitent de la manœuvre (type, côté, tiges oscillantes), du type de commande des moteurs radio, de la surlongueur de câble et de l'isolation phonique. Les lames, les cotes et les adaptateurs sont sur [Mono VI SOPROFEN : bloc-baie inversé à enroulement extérieur, lames, cotes et adaptateurs](/equipements/volets-roulants-mono-vi.md) [1 p. 129 à 135].

# Manœuvres motorisées (rubrique 10)

Largeur L mini en millimètres, pour un tablier, ou pour le tablier du côté gauche ou droit en multi-tabliers. Le tiret remplace la barre oblique des cases sans valeur ; un moteur court est le moteur de petite largeur, dont la largeur maximale est donnée [1 p. 129].

| Commande | Code | Moteur | L maxi du moteur court (mm) | L mini, 1 tablier (mm) | L mini, multi-tabliers, L gauche ou droite (mm) |
| --- | --- | --- | --- | --- | --- |
| Radio RTS | R1 | Oximo court RTS | 799 | 550 | 550 |
| Radio RTS | R1 | Oximo RTS | - | 800 | 800 |
| Radio RTS | R1S | Oximo court RTS avec manœuvre de secours | 944 | 695 | - |
| Radio RTS | R1S | Oximo RTS avec manœuvre de secours | - | 945 | - |
| Radio RTS autonome | A35 | Oximo 40WF RTS | - | 590 | - |
| Radio iO | IO1 | Oximo court IO | 659 | 550 | 550 |
| Radio iO | IO1 | S&SO RS100 IO | - | 660 | 660 |
| Radio iO | IO1S | Oximo court IO avec manœuvre de secours | 804 | 695 | - |
| Radio iO | IO1S | S&SO RS100 IO avec manœuvre de secours | - | (case vide) | (case vide) |
| Radio M-Soft² | R11 | M-Soft² MVM court | 659 | 590 | 590 |
| Radio M-Soft² | R11 | M-Soft² MVM | - | 660 | 660 |
| Filaire WT | M | Ilmo court | 699 | 550 | 550 |
| Filaire WT | M | Ilmo | - | 700 | 700 |
| Filaire WT | MS | Ilmo court avec manœuvre de secours | 844 | 695 | - |
| Filaire WT | MS | Ilmo avec manœuvre de secours | - | 845 | 845 |
| Filaire WT | M1 | Ilmo 12 trs | - | 700 | 700 |
| Filaire WT | M1S | Ilmo 12 trs avec manœuvre de secours | - | 845 | - |
| Filaire M-Soft² | M11 | M-Soft² MVE court | 659 | 590 | 590 |
| Filaire M-Soft² | M11 | M-Soft² MVE | - | 660 | 660 |
| Hybride filaire et radio | IOH | S&SO RS100 IO Hybrid | - | 660 | 660 |
| Hybride filaire et radio | IOHS | S&SO RS100 IO Hybrid avec manœuvre de secours | - | 805 | - |

La ligne « S&SO RS100 IO avec manœuvre de secours » n'a aucune valeur dans le tableau (**VER-149**) [1 p. 129].

Plusieurs tabliers : largeur mini à gauche ou à droite côté moteur ; hauteur maxi inférieure à 2 500 mm ; poids maximal de tablier avec une manœuvre, lames de 37 mm, de 30,7 kg pour deux tabliers et de 25,3 kg pour trois tabliers pour chacun des moteurs suivants, tous de 20 N.m [1 p. 129].

| Moteur | Largeur mini côté moteur (mm) |
| --- | --- |
| Radio Oximo RTS | 850 |
| Radio S&SO RS100 IO ou S&SO RS100 IO Hybrid | 740 |
| Radio M-SOFT² MVM | 720 |
| Filaire M-SOFT² MVE | 720 |
| Filaire Ilmo | 850 |
| Filaire Ilmo 12 tours | 850 |

## Moteurs radio

Chaque moteur radio a un verrouillage automatique du tablier par les verrous TLX, un arrêt sur obstacle à la descente du tablier, une recherche automatique des fins de course (moteur « Plug & Play ») et un câble de 3 mètres. Les commandes radio RTS, iO et M-SOFT² sont incompatibles entre elles. On peut piloter par radio les moteurs filaires avec un récepteur RRE ou RRM et des émetteurs au choix, à commander en accessoires [1 p. 130].

| Code | Moteur | Ce que le guide imprime |
| --- | --- | --- |
| R1 | Oximo RTS (MVO) | récepteur radio RTS intégré ; émetteur individuel SMOOVE MVO-ESM pré-appairé en usine au moteur. Petites largeurs : moteur court Oximo S RTS (réf. MVCR06/17), platine électronique radio déportée dans la joue du coffre |
| R1S | Oximo RTS | comme R1, avec une manœuvre de secours |
| IO1 | S&SO RS100 IO (MVRS), moteur de classe 2 | récepteur radio iO intégré ; émetteur individuel SMOOVE MVRS-ESM (sauf moteur court, fourni avec SMOOVE MVIO-ESM) pré-appairé en usine. Petites largeurs : moteur court Oximo S IO (réf. MVIO06/17P). Le moteur S&SO RS100 IO est limité à 20 N.m ; les volets qui nécessitent un moteur de plus de 20 N.m sont fournis avec un moteur Oximo IO |
| IO1S | S&SO RS100 IO | comme IO1, avec une manœuvre de secours |
| R11 | M-SOFT² (MVM) | récepteur radio intégré ; émetteur individuel Monotec MVM-E1C pré-appairé. Petites largeurs : moteur court réf. MVMC06/23 |

## Moteur autonome A35 et moteur Hybrid

Le moteur A35, Oximo 40 WF RTS, fonctionne à l'énergie solaire, captée par un panneau solaire et stockée dans une batterie intégrée dans le coffre ; le récepteur RTS est intégré au moteur et gère jusqu'à 12 points de commande ; un émetteur individuel MVO-ESM préprogrammé est fourni en standard, avec verrouillage automatique TLX, arrêt sur obstacle à la descente et recherche automatique des fins de course. Le panneau solaire nécessite obligatoirement un ensoleillement direct ; il est fixé par le poseur lors de l'installation et connecté pour maintenir la batterie en charge. Conditions : respecter les limites d'utilisation du tarif ; pas de manœuvre de secours possible ; uniquement avec un seul tablier en lames de 37 mm ; non compatible avec les multi-tabliers. Le chargeur de batterie, réf. MVOS-ALI, est l'accessoire indispensable de l'installateur ; toute utilisation d'un autre chargeur est interdite ; temps de charge maximal 3 h 30 [1 p. 131].

![Panneau solaire du moteur A35 : bons emplacements et mauvaises expositions](/assets/equipements/volets-roulants/blocs-baies-2022/mono-vi/panneau-solaire-a35-emplacements-a-eviter-et-a-choisir.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 131)

Le moteur IOH, S&SO RS100 IO Hybrid (MVRSH, moteur de classe 2), allie les commandes filaires et la technologie radio iO : la commande individuelle se fait par un interrupteur filaire, et le moteur comporte un récepteur radio iO intégré, pilotable par une télécommande iO complémentaire ou générale. Verrouillage automatique du tablier par les verrous TLX, arrêt sur obstacle à la descente, recherche automatique des fins de course, câble de 2,5 mètres fourni (**INC-356**). Fourni sans inverseur ni télécommande, à commander en accessoire. Le moteur est limité à 20 N.m ; au-delà, on choisit un moteur commandé soit uniquement en filaire, soit uniquement en radio. Le moteur IOHS est le même avec une manœuvre de secours [1 p. 131].

![Moteur Hybrid : trois schémas de câblage](/assets/equipements/volets-roulants/blocs-baies-2022/mono-vi/moteur-hybrid-trois-schemas-de-cablage.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 131)

## Moteurs filaires

Moteurs filaires fournis sans inverseur, à commander en accessoire ; tous les inverseurs ne conviennent pas pour des commandes multiples. Câble de 2,5 mètres pour les moteurs Ilmo, de 3 mètres pour les moteurs M-SOFT² [1 p. 132].

| Code | Moteur | Ce que le guide imprime |
| --- | --- | --- |
| M | Ilmo (MVI), « filaire Ilmo 17trs » | arrêt automatique aux fins de course ; verrouillage automatique du tablier par les verrous TLX ; arrêt sur obstacle à la descente et protection contre le gel. Moteur court Ilmo (réf. MVC06/17) : platine électronique filaire déportée, intégrée dans la joue du coffre |
| MS | Ilmo (MVI) | comme M, avec une manœuvre de secours |
| M1 | Ilmo 12 trs | mêmes caractéristiques que M |
| M1S | Ilmo 12 trs | comme M1, avec une manœuvre de secours |
| M11 | M-SOFT² (MVE) | moteur filaire M-SOFT² ; mêmes caractéristiques ; moteur court M-SOFT² réf. MVEC06/23 |

# Manœuvre de secours et treuil

La manœuvre de secours est un treuil monté du côté opposé au moteur, fourni avec une manivelle (Ø 12) amovible ; elle n'est possible qu'avec un seul tablier. La vue éclatée porte le palier escamotable 410737, le tube de liaison TMS16, la roue MVX-RP54, le moteur, le différentiel 410727, le flasque FC160/54-M et la pièce d'accouplement 410629 [1 p. 133].

![Manœuvre de secours : vue éclatée](/assets/equipements/volets-roulants/blocs-baies-2022/mono-vi/manoeuvre-de-secours-vue-eclatee.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 133)

Pour la manœuvre par treuil, on indique le code T. Surlongueur de trappe minimale en manœuvre par treuil : 52 mm ; le dessin porte 52 mm mini, 40 mm mini et 12 mm [1 p. 133].

![Manœuvre par treuil : surlongueur de trappe mini de 52 mm](/assets/equipements/volets-roulants/blocs-baies-2022/mono-vi/manoeuvre-treuil-surlongueur-de-trappe-mini-de-52-mm.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 133)

Une surlongueur de trappe de 40 mm est demandée pour les manœuvres par treuil et par manœuvre de secours à la page 127, et de 52 mm mini pour le treuil à la page 133 (**INC-350**) [1 p. 127, 133].

# Côté de manœuvre et tiges oscillantes (rubriques 11 et 12)

On indique le côté de la manœuvre vu de l'intérieur : G, D ; pour deux tabliers, E (extrémités) ; pour trois tabliers, Eg (manœuvre de gauche pour les tabliers de gauche et du milieu) et Ed (manœuvre de droite pour les tabliers de droite et du milieu) [1 p. 133].

![Côté de manœuvre : G, D, E, Eg, Ed](/assets/equipements/volets-roulants/blocs-baies-2022/mono-vi/cote-de-manoeuvre-g-d-e-eg-ed.png)

(schéma: raw/moustiquaires/exports/132.pdf, p. 133)

Tiges oscillantes pour treuil : sans indication, le choix est le choix par défaut ; code 0, sans tige oscillante. Tige de Ø 12 mm en tube acier revêtu polyester, manivelle à poignée en polypropylène, bras et manchon en acier laqué [1 p. 133].

| Tige oscillante Ø 12 | Longueur (mm) | Blanc W | Code |
| --- | --- | --- | --- |
| TO 120/18 | 1 230 | ● | 8 (par défaut) |
| TO 150/18 | 1 530 | ○ | 9 |
| TO 190/18 | 1 930 | ○ | 10 |

# Type de commande moteur, câble et isolation (rubriques 13 à 15)

Les émetteurs livrés avec les moteurs radio ont les mêmes références et les mêmes codes que pour le Bloc LX : M-SOFT² (68 par défaut, 61 à 65), Oximo RTS (SM par défaut, 7 à 10), S&So RS100 IO (QM par défaut, DM, S1P2, S1F, S1A, S1N), voir [Émetteurs et codes de commande du Bloc LX SOPROFEN](/equipements/volet-roulant-bloc-lx-emetteurs-et-codes-de-commande.md). Sans émetteur : code 0 ; si l'émetteur individuel correspondant est commandé en accessoire ou déjà existant, la programmation est réalisée par le client. Les moteurs S&SO RS100 IO Hybrid sont fournis sans inverseur ni télécommande [1 p. 134].

Le code du moteur radio M-SOFT² est écrit « R5 » à la page 134, alors que le tableau des moteurs de la page 129 le note « R11 » (**INC-349**). La grille des codes Smoove de la page 135 compte sept colonnes de cadre, sans la colonne « acier laqué » des autres pages (**INC-351**) [1 p. 134, 135].

Surlongueur de câble (rubrique 14), option : câble de 5 mètres, uniquement pour le moteur S&SO RS100 IO Hybrid ; câble de 10 mètres, uniquement pour les moteurs M-SOFT², M-SOFT² court, Oximo RTS, Oximo IO, S&SO RS100 IO et Ilmo, à l'exception des moteurs courts [1 p. 135].

# Ce que la source ne donne pas

La largeur minimale du moteur S&SO RS100 IO avec manœuvre de secours est absente du tableau (**VER-149**).

# Citations

[1] Guide technique Blocs-baies 2022, SOPROFEN, réf. DOC86505 (09/2022) — `raw/moustiquaires/exports/132.pdf`, p. 129 à 135
(le numéro de page imprimé est celui du PDF)

# Voir aussi

- [Mono VI SOPROFEN : bloc-baie inversé à enroulement extérieur, lames, cotes et adaptateurs](/equipements/volets-roulants-mono-vi.md)
- [Émetteurs et codes de commande du Bloc LX SOPROFEN : M-Soft², Oximo RTS, iO, Smoove](/equipements/volet-roulant-bloc-lx-emetteurs-et-codes-de-commande.md)
- [Chrono PSE² : manœuvres et motorisations](/equipements/chrono-pse2-manoeuvres-et-motorisations.md)
- [SOPROFEN](/fournisseurs/soprofen.md)
- [SOMFY](/fournisseurs/somfy.md)
- [Guide technique Blocs-baies 2022, SOPROFEN (148 pages)](/sources/soprofen-guide-technique-blocs-baies-2022.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
