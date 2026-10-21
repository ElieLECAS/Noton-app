---
type: Quincaillerie
title: 'Compas Roto NX : lecture des tableaux, tolérance de châssis fixe, têtière de compas côté P'
description: 'Comment lire les tableaux de compas du catalogue Roto NX, tolérance de châssis fixe côté paumelles P et Designo (BA 13), puis têtière de compas côté paumelles P : sécurité de base, CDR 1 N, CDR 2 / CDR 2 N, CDR 3, vantail pivotant.'
tags: [roto, roto-nx, compas, palier, pivot, designo, quincaillerie, reference, charge, tetiere, bras-de-compas, equerre-de-compas, cache, tolerance-de-chassis-fixe, tiltfirst]
systeme: Roto NX
fournisseur: ROTO
usage: atelier
status: draft
sources:
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
source_pages:
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    pages: 277-312
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    pages: 35, 36, 68-73
generated:
  by: process:claude-code
  at: 2026-10-21T10:43:50Z
---

# Compas, paliers et pivots Roto NX

# Lire les tableaux de compas du catalogue

Le **compas** est la ferrure de l'angle haut de l'ouvrant ; le compas OF équipe un ouvrant à la
française, le compas OB un oscillo-battant, dont il retient le vantail basculé en soufflet (voir le
[glossaire](/reference/glossaire.md)). Dans le catalogue Roto NX PVC,
le chapitre 6 « Compas » range ses pièces en tolérance de châssis fixe, têtière de compas (la
longue ferrure plate vissée le long du haut de l'ouvrant), bras de compas, compas articulé à
commande forcée, équerre de compas OF, bras de compas en feuillure, charnière en feuillure,
paumelle de recouvrement, palier compas, broches de palier et caches, côté paumelles P et, pour
une partie d'entre elles, côté paumelles Designo (BA 13) [1 p. 277].

Les tableaux ont des **pictogrammes en en-tête de colonne**, définis dans la légende du catalogue
([Légende des tableaux du catalogue Roto NX](/quincaillerie/roto-nx-legende-catalogue.md))
[1 p. 10-12]. Les tables de cette page reprennent la référence (N° de matériel) en première
colonne, puis les colonnes dans l'ordre du catalogue, nommées d'après la légende : aération
intégrée, blocage, LFF (largeur fond de feuillure d'ouvrant, plage en mm), longueur (mm), zone de
recoupe (mm, longueur dont la pièce peut être raccourcie), taille (taille de compas, nombre
imprimé sans unité), nombre de galets, type de goujon de verrouillage (galets E, P, V, voir
[Roto NX](/quincaillerie/roto-nx.md)), système (système de profilé, voir
[Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)), traitement de
surface, DIN gauche / droite. Comme sur les tableaux de crémones, les colonnes oui / non portent
**J** ou **N**, reportés tels qu'imprimés (entrée **INC-288**). Une plage ou une valeur imprimée
une seule fois pour un groupe de lignes (cellule fusionnée) est répétée ici sur chaque ligne.

Au-dessus de chaque tableau, le catalogue dessine la pièce et les pictogrammes des types
d'ouverture auxquels elle s'applique ; ils sont nommés d'après la p. 12 du catalogue.

# Tolérance de châssis fixe

La *tolérance de châssis fixe* est l'encombrement de la paumelle, côté paumelles, entre
l'ouvrant et l'embrasure (le tableau de la maçonnerie) ; elle est donnée caches compris (voir le
[glossaire](/reference/glossaire.md)) [1 p. 278].

## Côté paumelles P

Tolérance de châssis fixe pour 20 mm de largeur de recouvrement. La coupe horizontale montre le
dormant en rose saumon posé contre la maçonnerie (hachurée à droite) et l'ouvrant en gris : [A] est
la tolérance de châssis fixe, cotée sur le dessin entre l'ouvrant et la maçonnerie, [B]
la hauteur de recouvrement, [C] l'espace extérieur, en profondeur de l'embrasure [1 p. 278].

![Tolérance de châssis fixe, côté paumelles P, coupe](/assets/quincaillerie/roto-nx-catalogue/compas/tolerance-chassis-fixe-cote-p-coupe.png)

Une ligne par poids d'ouvrant, cotes en mm [1 p. 278] :

| Poids d'ouvrant (kg) | Angle d'ouverture | Tolérance de châssis fixe [A] (mm) | Hauteur de recouvrement [B] (mm) | Espace extérieur [C] (mm) |
| --- | --- | --- | --- | --- |
| 130 | env. 180° (note [3]) | ≥ 21,0 | ≥ 16 | 100 |
| 150 | env. 150° (note [4]) | ≥ 26,5 | ≥ 16 | 100 |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 278)

Notes du tableau : [3] selon l'embrasure, l'angle d'ouverture peut être inférieur à 180° ; [4]
selon l'embrasure, l'angle d'ouverture peut être inférieur à 150° [1 p. 278].

**INFO.** Tolérance, caches compris. Angle d'ouverture jusqu'à 20 mm de hauteur de recouvrement
[1 p. 278].

Le manuel de montage KSR donne la même tolérance avec un angle d'ouverture d'environ 180° pour
130 kg comme pour 150 kg, et appelle [C] et [D] les jeux en haut et en bas du vantail
(entrée **CTR-118**) ; ses valeurs sont sur
[Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md).

**DANGER — Danger de mort par endommagement des composants du palier !** Des hauteurs de
recouvrement importantes [B] ou des éléments de construction dans la zone de l'embrasure (comme
les plinthes) peuvent générer des forces de levier trop élevées du côté paumelles. Cela peut
endommager les composants du palier et entraîner la chute de l'ouvrant [1 p. 278].

- Dans la zone libre [C], l'ouvrant ne doit pas toucher l'embrasure ou les éléments dans la zone
  de l'embrasure.
- Si la profondeur de l'embrasure est < 100 mm, vérifier l'utilisation du limiteur d'ouverture.

**ATTENTION — Dommages matériels dus à l'endommagement des composants du palier !** Même texte et
mêmes deux consignes que l'encadré DANGER [1 p. 278].

### Vues de face, côté paumelles P

Quatre vues de face : le vantail en gris dans le dormant rose saumon, côté paumelles à droite, une
pièce dessinée en haut et une en bas de ce côté. Chaque vue porte la cote horizontale au droit de la
pièce haute, la cote au droit de la pièce basse, la cote verticale sous le vantail et, sur les deux premières, une cote verticale
« ≥ 10* » au-dessus de la pièce haute. L'astérisque renvoie à la note : **pour le démontage de la broche,
laisser un espace d'au moins 10 mm jusqu'à l'embrasure** [1 p. 279].

![Tolérance de châssis fixe, poids d'ouvrant 130 kg](/assets/quincaillerie/roto-nx-catalogue/compas/tolerance-chassis-fixe-cote-p-130-kg.png)

![Tolérance de châssis fixe, poids d'ouvrant 150 kg](/assets/quincaillerie/roto-nx-catalogue/compas/tolerance-chassis-fixe-cote-p-150-kg.png)

![Tolérance de châssis fixe, cintré, poids d'ouvrant 80 kg](/assets/quincaillerie/roto-nx-catalogue/compas/tolerance-chassis-fixe-cote-p-cintre.png)

![Tolérance de châssis fixe, ferrure soufflet, poids d'ouvrant 80 kg](/assets/quincaillerie/roto-nx-catalogue/compas/tolerance-chassis-fixe-cote-p-soufflet.png)

Cotes portées sur les vues, en mm ; « - » : cote non portée sur la vue [1 p. 279] :

| Vue | Poids d'ouvrant (kg) | Au droit de la pièce haute (mm) | Au droit de la pièce basse (mm) | Sous le vantail (mm) | Au-dessus de la pièce haute (mm) |
| --- | --- | --- | --- | --- | --- |
| Rectangulaire | 130 | 21 | 19 | 8 | ≥ 10* |
| Rectangulaire | 150 | 26,5 | 19 | 8 | ≥ 10* |
| Cintré | 80 | 25 | 19 | 8 | - |
| Ferrure soufflet | 80 | - | - | 23 | - |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 279)

Sur la vue « Ferrure soufflet », deux pièces sont dessinées sous le vantail, le long du bas du
dormant, et la seule cote est la cote verticale de 23 mm à gauche [1 p. 279].

## Côté paumelles Designo (BA 13)

Tolérance de châssis fixe (en mm) pour un angle d'ouverture de 90°. La coupe montre le dormant
rose saumon et l'ouvrant gris côté paumelles Designo (BA 13) : [A] tolérance de châssis fixe, [B]
hauteur de recouvrement, [C] largeur de recouvrement, [D] axe de ferrure [1 p. 280].

![Tolérance de châssis fixe, côté paumelles Designo (BA 13), coupe](/assets/quincaillerie/roto-nx-catalogue/compas/tolerance-chassis-fixe-designo-coupe.png)

Une ligne par combinaison de recouvrement, côté paumelles Designo (BA 13), cotes en mm
[1 p. 280] :

| Angle d'ouverture | Tolérance de châssis fixe [A] (mm) | Hauteur de recouvrement [B] (mm) | Largeur de recouvrement [C] (mm) | Axe de ferrure [D] (mm) |
| --- | --- | --- | --- | --- |
| 90° | 5,2 | 16 | 18 | 13 |
| 90° | 7,6 | 20 | 18 | 13 |
| 90° | 10,6 | 24 | 18 | 13 |
| 90° | 4,9 | 16 | 20 | 13 |
| 90° | 7,2 | 20 | 20 | 13 |
| 90° | 9,9 | 24 | 20 | 13 |
| 90° | 4,7 | 16 | 22 | 13 |
| 90° | 6,8 | 20 | 22 | 13 |
| 90° | 9,3 | 24 | 22 | 13 |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 280)

# Têtière de compas, côté paumelles P

La têtière de compas est la longue ferrure plate vissée le long du haut de l'ouvrant, sur laquelle
se monte le compas (voir le [glossaire](/reference/glossaire.md)) ; les tableaux la classent par
niveau de sécurité et par largeur fond de feuillure (LFF) [1 p. 281-282].

## Sécurité de base – CDR 1 N, côté paumelles P

![Têtière de compas côté P, sécurité de base – CDR 1 N](/assets/quincaillerie/roto-nx-catalogue/compas/tetiere-compas-cote-p-securite-de-base-cdr-1-n.png)

Types d'ouverture : fenêtre oscillo-battante ; fenêtre à la française / oscillo-battante à
ouvrant semi-fixe. Une ligne par référence, cotes en mm [1 p. 281] :

| N° de matériel | Aération intégrée | Blocage | LFF (mm) | Longueur (mm) | Zone de recoupe (mm) | Taille | Nombre de galets | Type de goujon de verrouillage | Traitement de surface |
| --- | :---: | :---: | --- | ---: | ---: | ---: | :---: | :---: | --- |
| 787345 | J | J | 290 – 410 | 300 | 120 | 150 | – | – | Roto Sil |
| 787346 | J | J | 411 – 600 | 490 | 190 | 250 | – | – | Roto Sil |
| 787347 | J | J | 601 – 800 | 690 | 200 | 350 | – | – | Roto Sil |
| 787348 | J | J | 801 – 1000 | 890 | 200 | 500 | – | – | Roto Sil |
| 788617 | J | J | 801 – 1000 | 890 | 200 | 500 | 1 | E | Roto Sil |
| 787350 | J | J | 1001 – 1200 | 1090 | 200 | 500 | – | – | Roto Sil |
| 787349 | J | J | 1001 – 1200 | 1090 | 200 | 500 | 1 | E | Roto Sil |
| 787351 | J | J | 1201 – 1400 | 1290 | 200 | 500 | 1 | E | Roto Sil |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 281)

**INFO.** Pour les LFF à partir de 1 400 mm, 2ème compas requis [1 p. 281].

## Sécurité – CDR 2 / CDR 2 N, côté paumelles P

![Têtière de compas côté P, sécurité – CDR 2 / CDR 2 N](/assets/quincaillerie/roto-nx-catalogue/compas/tetiere-compas-cote-p-cdr-2-cdr-2-n.png)

Types d'ouverture : fenêtre oscillo-battante ; fenêtre à la française / oscillo-battante à
ouvrant semi-fixe. Une ligne par référence, cotes en mm [1 p. 281] :

| N° de matériel | Aération intégrée | Blocage | LFF (mm) | Longueur (mm) | Zone de recoupe (mm) | Taille | Nombre de galets | Type de goujon de verrouillage | Traitement de surface |
| --- | :---: | :---: | --- | ---: | ---: | ---: | :---: | :---: | --- |
| 787360 | J | J | 801 – 1000 | 890 | 200 | 500 | 1 | V | Roto Sil |
| 787361 | J | J | 1001 – 1200 | 1090 | 200 | 500 | 1 | V | Roto Sil |
| 787362 | J | J | 1201 – 1400 | 1290 | 200 | 500 | 1 | V | Roto Sil |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 281)

## Sécurité – CDR 3, côté paumelles P

![Têtière de compas côté P, sécurité – CDR 3](/assets/quincaillerie/roto-nx-catalogue/compas/tetiere-compas-cote-p-cdr-3.png)

Types d'ouverture : fenêtre oscillo-battante ; fenêtre à la française / oscillo-battante à
ouvrant semi-fixe. Une ligne, cotes en mm [1 p. 282] :

| N° de matériel | Aération intégrée | Blocage | LFF (mm) | Longueur (mm) | Zone de recoupe (mm) | Taille | Nombre de galets | Type de goujon de verrouillage | Traitement de surface |
| --- | :---: | :---: | --- | ---: | ---: | ---: | :---: | :---: | --- |
| 787358 | J | J | 801 – 1000 | 890 | 200 | 350 | 1 | V | Roto Sil |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 282)

**INFO.** Pour les LFF à partir de 1 001 mm, verrouilleur médian (200 mm, galet 1V) requis. En
supplément pour les LFF à partir de 1 201 mm, 2ème compas requis [1 p. 282].

## Vantail pivotant, côté paumelles P

![Têtière de compas côté P, vantail pivotant](/assets/quincaillerie/roto-nx-catalogue/compas/tetiere-compas-cote-p-vantail-pivotant.png)

Type d'ouverture : fenêtre à la française / oscillo-battante à ouvrant semi-fixe. La première
colonne est la classe de sécurité ; il n'y a pas de colonne aération intégrée ni blocage. Une ligne
par référence, cotes en mm [1 p. 282] :

| N° de matériel | Classe de sécurité | LFF (mm) | Longueur (mm) | Zone de recoupe (mm) | Taille | Nombre de galets | Type de goujon de verrouillage | Traitement de surface |
| --- | --- | --- | ---: | ---: | ---: | :---: | :---: | --- |
| 787366 | Sécurité de base | 290 – 410 | 300 | 120 | 150 | – | – | Roto Sil |
| 787367 | Sécurité de base | 411 – 600 | 490 | 190 | 250 | – | – | Roto Sil |
| 787368 | Sécurité de base | 601 – 800 | 690 | 200 | 350 | – | – | Roto Sil |
| 787369 | Sécurité de base | 801 – 1000 | 890 | 200 | 500 | 1 | E | Roto Sil |
| 787370 | Sécurité | 801 – 1000 | 890 | 200 | 500 | 1 | V | Roto Sil |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 282)

**INFO.** « Pour les LVFF à partir de 1001 mm, allonge de crémone requise. » [1 p. 282] L'abréviation
LVFF n'est pas dans la liste des abréviations du catalogue, qui ne connaît que LFF et HFF
(entrée **INC-293**).

# Citations

[1] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 — `raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 10-13, 277 à 312
[2] Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf. IMO_180_NX_FR_v2, novembre 2022 — `raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, p. 35, 36 et 68 à 73 (numérotation du PDF)

# Voir aussi

- [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)
- [Légende des tableaux du catalogue Roto NX](/quincaillerie/roto-nx-legende-catalogue.md)
- [Aperçu des ferrures Roto NX, côté paumelles P](/quincaillerie/roto-nx-apercu-ferrures-cote-p.md)
- [Aperçu des ferrures Roto NX, côté paumelles Designo (BA 13)](/quincaillerie/roto-nx-apercu-ferrures-designo.md)
- [Finitions Roto NX](/quincaillerie/roto-nx-finitions.md)
- [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md)
- [Configurations Roto NX KSR — Confort et NT Designo II](/quincaillerie/roto-nx-ksr-confort-designo.md)
- [Pivots et paliers d'angle Roto NX](/quincaillerie/roto-nx-pivots-et-paliers-angle.md)
- [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md)
- [Renvois d'angle et verrouilleurs Roto NX](/quincaillerie/roto-nx-renvois-angle-et-verrouilleurs.md)
- [Pièces de fermeture et gâches Roto NX](/quincaillerie/roto-nx-pieces-fermeture-gaches.md)
- [Deuxième compas, compas d'arrêt, limiteurs et compas d'aération Roto NX](/quincaillerie/roto-nx-compas-complementaires.md)
- [Catalogue Roto NX pour profils PVC](/sources/roto-nx-catalogue-pvc.md)

Autres parties de ce sujet :

- [Têtière de compas côté Designo (BA 13) et bras de compas standard côté P, Roto NX](/quincaillerie/roto-nx-compas-et-paliers-tetiere-designo-et-bras-standard-cote-p.md)
- [Bras de compas Roto NX côté P : TiltFirst, fenêtre inclinée et cintré](/quincaillerie/roto-nx-compas-et-paliers-bras-tiltfirst-inclinee-cintre-cote-p.md)
- [Bras de compas standard Roto NX côté Designo (BA 13) : tailles 250, 350 et 500](/quincaillerie/roto-nx-compas-et-paliers-bras-standard-designo.md)
- [Bras de compas TiltFirst Roto NX côté Designo (BA 13) : tailles 250, 350 et 500](/quincaillerie/roto-nx-compas-et-paliers-bras-tiltfirst-designo.md)
- [Compas articulé à commande forcée, équerre de compas OF et bras en feuillure Roto NX](/quincaillerie/roto-nx-compas-et-paliers-compas-commande-forcee-equerre-of-bras-feuillure.md)
- [Charnière en feuillure, paumelle de recouvrement et palier de compas Roto NX côté P](/quincaillerie/roto-nx-compas-et-paliers-charniere-feuillure-paumelle-recouvrement-palier-compas.md)
- [Caches de paliers et paumelles Roto NX côté P, longueurs de palier P3 et P6, réglages](/quincaillerie/roto-nx-compas-et-paliers-caches-longueurs-de-palier-reglages.md)
