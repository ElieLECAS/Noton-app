---
type: Profilé
title: "Inerties des ouvrants 24 mm, 28/32 mm et des chicanes du coulissant ASKEY 65 NV"
description: "Moments d'inertie Ix (1 200 à 3 000 mm), Iy et modules de résistance W des montants et traverses d'ouvrant 24 mm et 28/32 mm (W1041260 à W1041276, Y1740028 à Y1740041, W1040048) et des chicanes 24 mm et 28-32 mm (W3041240 à W3041256, Y3740019, Y3740020) du Coulissant 65 NV ASKEY."
tags: [askey, coulissant, inertie, ouvrant, chicane, montant, traverse, profile, conception, 65mm]
gamme: Coulissant 65 NV
fournisseur: ASKEY
usage: chiffrage
famille: valeurs-statiques
status: draft
sources:
  - resource: raw/askey1.pdf
    title: Coulissant 65 NV ASKEY - Conception
    last_modified: 2022-12-13
source_pages:
  - resource: raw/askey1.pdf
    pages: 17-21
generated:
  by: process:claude-code
  at: 2026-10-10T19:22:07Z
---

# Valeurs statiques des ouvrants 24 mm

L'ouvrant est le cadre mobile qui porte le vitrage et coulisse sur le rail du dormant ; le Coulissant 65 NV ASKEY ([gamme](/gammes/askey-coulissant-65-nv.md)) a deux modules d'ouvrant, 24 mm et 28/32 mm. Le moment d'inertie d'un profilé (en cm⁴) mesure sa résistance à la flexion ; le module de résistance W (en cm³) sert à contrôler la contrainte (règle de calcul : voir [Inerties des dormants coupe droite et des seuils](/profiles/askey-coulissant-65-nv-valeurs-statiques-dormants-et-seuils.md)). La planche est titrée « Ouvrants 24 mm » [1 p. 17].

Lecture : une ligne par profilé, la référence en première colonne. Les six colonnes « Ix » donnent le moment d'inertie I selon l'axe XX' (axe horizontal de la section dessinée) pour la longueur de profilé indiquée en mm ; « Iy » est le moment d'inertie I selon l'axe YY' ; « W selon YY' » est le module de résistance W, chiffré seulement pour l'axe YY' (ses six cases de l'axe XX' sont « - » sur la planche). Aucune valeur n'est interpolée entre deux longueurs. La colonne « Coupe » est la vignette de la section imprimée sur la planche.

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| W1041260 | 5,53 | 6,64 | 7,31 | 7,60 | 7,83 | 8,14 | 12,69 | 3,56 | ![W1041260](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041260.png) |
| W1041261 | 20,86 | 24,91 | 27,66 | 28,91 | 29,89 | 31,31 | 34,48 | 9,95 | ![W1041261](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041261.png) |
| W1041262 | 14,05 | 16,76 | 18,55 | 19,34 | 19,96 | 20,85 | 20,63 | 5,76 | ![W1041262](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041262.png) |
| W1041263 | 15,19 | 17,75 | 19,41 | 20,14 | 20,71 | 21,52 | 18,23 | 4,65 | ![W1041263](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041263.png) |
| W1041264 | 2,34 | 3,18 | 3,83 | 4,14 | 4,40 | 4,80 | 8,13 | 2,51 | ![W1041264](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041264.png) |
| W1041265 | 2,30 | 3,11 | 3,71 | 4,00 | 4,24 | 4,60 | 6,59 | 2,06 | ![W1041265](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041265.png) |
| Y1740028 | 11,08 | 13,24 | 14,64 | 15,25 | 15,73 | 16,40 | 18,55 | 4,73 | ![Y1740028](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740028.png) |
| Y1740029 | 18,45 | 22,68 | 25,62 | 26,97 | 28,05 | 29,61 | 24,42 | 5,53 | ![Y1740029](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740029.png) |

(schéma: raw/askey1.pdf, p. 17 pour W1041260 à W1041265 ; p. 18 pour Y1740028 et Y1740029)

# Valeurs statiques des ouvrants 28-32 mm

Même lecture. Les planches portent le titre « Ouvrants 28-32 mm » : les profilés du module 28/32 mm reçoivent des vitrages de 28 mm ou de 32 mm [1 p. 19].

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| W1040048 | 25,63 | 32,80 | 38,80 | 41,90 | 44,56 | 48,74 | 23,77 | 5,39 | ![W1040048](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1040048.png) |
| W1041270 | 6,19 | 8,15 | 9,56 | 10,22 | 10,75 | 11,54 | 12,69 | 3,56 | ![W1041270](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041270.png) |
| W1041271 | 19,62 | 25,09 | 29,49 | 31,71 | 33,58 | 36,46 | 34,48 | 9,95 | ![W1041271](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041271.png) |
| W1041272 | 13,75 | 17,72 | 20,82 | 22,34 | 23,61 | 25,54 | 20,63 | 5,75 | ![W1041272](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041272.png) |
| W1041273 | 14,97 | 18,79 | 21,71 | 23,14 | 24,32 | 26,10 | 18,23 | 4,65 | ![W1041273](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041273.png) |
| W1041274 | 3,76 | 5,14 | 6,19 | 6,71 | 7,14 | 7,78 | 8,13 | 2,51 | ![W1041274](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041274.png) |
| W1041275 | 3,70 | 5,00 | 5,98 | 6,46 | 6,85 | 7,43 | 6,59 | 2,06 | ![W1041275](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041275.png) |
| W1041276 | 6,73 | 8,22 | 9,16 | 9,57 | 9,89 | 10,34 | 28,91 | 6,02 | ![W1041276](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041276.png) |
| Y1740030 | 11,15 | 14,46 | 16,99 | 18,22 | 19,24 | 20,77 | 18,55 | 4,73 | ![Y1740030](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740030.png) |
| Y1740031 | 17,16 | 22,73 | 27,33 | 29,69 | 31,70 | 34,83 | 24,42 | 5,53 | ![Y1740031](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740031.png) |
| Y1740040 | 22,11 | 29,05 | 34,66 | 37,49 | 39,87 | 43,56 | 5,89 | 3,20 | ![Y1740040](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740040.png) |
| Y1740041 | 29,33 | 38,63 | 46,60 | 50,80 | 54,43 | 60,21 | 19,58 | 5,26 | ![Y1740041](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740041.png) |

(schéma: raw/askey1.pdf, p. 19 pour W1040048 et W1041270 à W1041275 ; p. 20 pour W1041276, Y1740030, Y1740031, Y1740040, Y1740041)

Le module de résistance W de W1041262 (ouvrant 24 mm) est 5,76 cm³ et celui de W1041272 (ouvrant 28-32 mm) est 5,75 cm³, pour la même inertie Iy de 20,63 cm⁴ (**VER-725**). Les désignations des profilés (montant latéral, traverse, montant central) ne sont pas imprimées sur ces planches, qui portent la référence et la coupe.

# Valeurs statiques des chicanes 24 mm et 28-32 mm

La chicane est le montant central où se rejoignent deux vantaux (voir [Profilés dormants et ouvrants ASKEY Coulissant 65 NV](/profiles/askey-coulissant-65-nv-dormants-et-ouvrants.md)). La planche présente côte à côte les chicanes 24 mm (colonne de gauche) et 28-32 mm (colonne de droite) [1 p. 21].

Lecture : une ligne par profilé. Les valeurs sont imprimées sans longueur, en quatre colonnes : Ixx' et Iyy' (moments d'inertie, cm⁴) et Ixx'/v et Iyy'/v (cm³ ; le « v » n'est pas défini sur la planche). La colonne « Coupe » est la vignette de la section imprimée sur la planche.

## Chicanes 24 mm

| Référence | Ixx' (cm⁴) | Iyy' (cm⁴) | Ixx'/v (cm³) | Iyy'/v (cm³) | Coupe |
| --- | --- | --- | --- | --- | ---: |
| W3041240 | 5,02 | 2,57 | 2,19 | 1,00 | ![W3041240](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041240.png) |
| W3041241 | 29,22 | 8,36 | 6,78 | 3,72 | ![W3041241](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041241.png) |
| W3041242 | 29,00 | 6,36 | 6,74 | 2,59 | ![W3041242](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041242.png) |
| W3041245 | 54,80 | 10,74 | 10,36 | 4,89 | ![W3041245](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041245.png) |
| W3041246 | 58,45 | 8,56 | 10,50 | 3,61 | ![W3041246](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041246.png) |
| Y3740019 | 16,36 | 6,42 | 4,48 | 2,80 | ![Y3740019](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y3740019.png) |

## Chicanes 28-32 mm

| Référence | Ixx' (cm⁴) | Iyy' (cm⁴) | Ixx'/v (cm³) | Iyy'/v (cm³) | Coupe |
| --- | --- | --- | --- | --- | ---: |
| W3041250 | 6,39 | 2,70 | 2,56 | 1,04 | ![W3041250](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041250.png) |
| W3041251 | 33,25 | 8,52 | 7,30 | 3,77 | ![W3041251](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041251.png) |
| W3041252 | 33,36 | 6,51 | 7,38 | 2,64 | ![W3041252](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041252.png) |
| W3041255 | 60,54 | 10,89 | 10,94 | 4,93 | ![W3041255](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041255.png) |
| W3041256 | 65,94 | 8,73 | 11,40 | 3,66 | ![W3041256](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041256.png) |
| Y3740020 | 19,35 | 6,57 | 4,97 | 2,85 | ![Y3740020](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y3740020.png) |

(schéma: raw/askey1.pdf, p. 21)

# Citations

[1] [Coulissant 65 NV ASKEY - Conception](raw/askey1.pdf), p. 17-21

# Voir aussi

- [Inerties des dormants coupe droite et des seuils du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-valeurs-statiques-dormants-et-seuils.md)
- [Inerties des renforts, angles variables et jonctions de vantaux du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-valeurs-statiques-renforts-et-jonctions.md)
