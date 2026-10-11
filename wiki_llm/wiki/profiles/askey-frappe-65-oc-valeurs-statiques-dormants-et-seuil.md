---
type: Profilé
title: "Inerties des dormants, monoblocs et seuil PMR de la Frappe 65 Ouvrant Caché ASKEY"
description: "Moments d'inertie Ix (1 200 à 3 000 mm), Iy et modules W des dormants 65 mm, des dormants pour vitrage 36 mm, des monoblocs (W1010059 à Y1715153, W1010120 à Y1715130) et du seuil PMR W1010384 de la Frappe 65 OC ASKEY."
tags: [askey, frappe, ouvrant-cache, inertie, statique, dormant, monobloc, seuil-pmr, profile, 65mm]
gamme: Frappe 65 Ouvrant Caché
fournisseur: ASKEY
usage: chiffrage
famille: valeurs-statiques
status: draft
sources:
  - resource: raw/askey3.pdf
    title: Frappe 65 Ouvrant Caché ASKEY - Conception
    last_modified: 2023-02-20
source_pages:
  - resource: raw/askey3.pdf
    pages: 12-17
generated:
  by: process:claude-code
  at: 2026-10-10T21:13:51Z
---

# Valeurs statiques des dormants 65 mm

Le dormant est le cadre fixe de la fenêtre ou de la porte-fenêtre, scellé dans la baie ; l'ouvrant est le cadre mobile (voir la [Gamme ASKEY Frappe 65 Ouvrant Caché](/gammes/askey-frappe-65-oc.md)). Le moment d'inertie d'un profilé (en cm⁴) mesure sa résistance à la flexion : plus il est grand, moins le profilé fléchit sous la charge. Le module de résistance W (en cm³) sert à contrôler la contrainte dans le profilé. Les termes du métier sont dans le [glossaire](/reference/glossaire.md).

Pour la vérification en déformation d'un profilé à rupture de pont thermique, le calcul de la flèche, nécessite la connaissance du moment d'inertie Ix qui varie en fonction de la longueur. Cette valeur est calculée conformément à l'annexe C de la norme EN 14024.

La preuve statique consiste à vérifier que la contrainte résultant de la combinaison d'efforts de chargement et de température hiver/été n'excède pas en tout point du profilé la contrainte admissible et que les efforts de cisaillement induits ne dépassent pas la tenue en glissement du sertissage. Dans un souci de simplification, il est fourni par longueur un module de résistance Wx permettant de limiter la vérification à un calcul de contrainte en flexion, la tenue selon les autres critères étant garantie. Dès lors, en fonction du cas de charge, il suffit de calculer le moment de flexion maximum Mfmax et de vérifier que Mfmax/Wx < σ0.2 (classiquement, σ0.2 = 160 N/mm2). [1 p. 12]

Lecture des tableaux : une ligne par profilé, la référence en première colonne. Les six colonnes « Ix » donnent le moment d'inertie I selon l'axe XX' (repères x et x' dessinés dans la case en haut à gauche de la planche : x en haut, x' en bas) pour la longueur de profilé indiquée en mm ; « Iy » est le moment d'inertie I selon l'axe YY' (repères y à gauche et y' à droite) ; « W » est le module de résistance W selon YY'. La planche imprime pour chaque profilé deux lignes, « Inertie I (cm4) » et « Module de résistance W (cm3) » : la ligne W est « - » aux six longueurs selon XX' pour tous les profilés de ces planches, de sorte que la règle Mfmax/Wx ne dispose d'aucune valeur Wx imprimée (**VER-775**). Les valeurs sont recopiées avec la virgule décimale ; la planche écrit le point.

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| W1010059 | 13,41 | 16,93 | 19,46 | 20,64 | 21,60 | 23,00 | 13,78 | 3,71 | ![W1010059](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010059.png) |
| W1010064 | 12,86 | 16,20 | 18,59 | 19,71 | 20,61 | 21,93 | 13,53 | 3,67 | ![W1010064](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010064.png) |
| W1010067 | 14,83 | 18,99 | 22,06 | 23,52 | 24,70 | 26,47 | 16,18 | 4,74 | ![W1010067](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010067.png) |
| W1010130 | 13,69 | 17,32 | 19,93 | 21,15 | 22,14 | 23,60 | 13,65 | 3,66 | ![W1010130](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010130.png) |
| W1010131 | 15,26 | 19,57 | 22,77 | 24,30 | 25,54 | 27,40 | 16,26 | 4,72 | ![W1010131](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010131.png) |
| W1010132 | 13,33 | 16,84 | 19,36 | 20,54 | 21,50 | 22,90 | 13,60 | 3,64 | ![W1010132](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010132.png) |
| W1010136 | 13,87 | 17,56 | 20,22 | 21,47 | 22,47 | 23,96 | 13,84 | 3,68 | ![W1010136](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010136.png) |
| W1010364 | 16,70 | 21,78 | 25,68 | 27,58 | 29,14 | 31,51 | 21,50 | 6,18 | ![W1010364](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010364.png) |
| W1010373 | 16,69 | 21,73 | 25,57 | 27,44 | 28,98 | 31,30 | 24,39 | 6,93 | ![W1010373](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010373.png) |
| W1010390 | 13,33 | 16,84 | 19,36 | 20,54 | 21,50 | 22,90 | 13,60 | 3,64 | ![W1010390](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010390.png) |
| W1010504 | 20,22 | 27,29 | 33,20 | 36,25 | 38,87 | 42,98 | 90,61 | 15,03 | ![W1010504](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010504.png) |
| W1715110 | 13,09 | 16,53 | 19,00 | 20,16 | 21,09 | 22,46 | 13,54 | 3,67 | ![W1715110](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1715110.png) |
| W1715111 | 14,99 | 19,23 | 22,36 | 23,86 | 25,08 | 26,89 | 16,19 | 4,75 | ![W1715111](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1715111.png) |
| Y1715125 | 16,70 | 21,78 | 25,68 | 27,58 | 29,14 | 31,51 | 21,50 | 6,18 | ![Y1715125](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/y1715125.png) |

(schéma: raw/askey3.pdf, p. 12 pour W1010059, W1010064, W1010067, W1010130, W1010131 ; p. 13 pour W1010132, W1010136, W1010364, W1010373, W1010390, W1010504 ; p. 14 pour W1715110, W1715111, Y1715125)

Les références W1715110 et W1715111 sont imprimées ainsi sur la planche, avec le préfixe W alors que les autres références de la famille Y17151xx portent le préfixe Y (**VER-768**). Les valeurs de W1010390 sont identiques à celles de W1010132 à toutes les longueurs et selon YY', et celles de Y1715125 à celles de W1010364 (**VER-769**). Les désignations des profilés (C45, CD, rénovation, ailette) ne sont pas imprimées sur ces planches : la planche porte la référence et la coupe cotée ; les désignations sont sur les planches à l'échelle 1 des dormants (voir [Profilés dormants et ouvrants ASKEY Frappe 65 Ouvrant Caché](/profiles/askey-frappe-65-oc-dormants-et-ouvrants.md)).

# Valeurs statiques des dormants pour vitrage 36 mm

Les dormants pour vitrage 36 mm sont les dormants de la Frappe 65 Ouvrant Caché ASKEY réunis sous ce titre sur les planches ; ils se lisent comme le tableau précédent. [1 p. 14-15]

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| Y1715147 | 19,94 | 25,24 | 29,18 | 31,07 | 32,61 | 34,90 | 16,15 | 4,71 | ![Y1715147](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/y1715147.png) |
| Y1715150 | 117,82 | 133,73 | 146,42 | 152,78 | 158,11 | 166,32 | 32,58 | 6,62 | ![Y1715150](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/y1715150.png) |
| Y1715151 | 79,54 | 92,53 | 102,82 | 107,96 | 112,26 | 118,86 | 31,47 | 6,53 | ![Y1715151](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/y1715151.png) |
| Y1715152 | 50,26 | 60,40 | 68,37 | 72,32 | 75,60 | 80,63 | 29,77 | 6,39 | ![Y1715152](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/y1715152.png) |
| Y1715153 | 30,45 | 37,95 | 43,71 | 46,53 | 48,85 | 52,37 | 24,68 | 5,75 | ![Y1715153](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/y1715153.png) |
| W1010410 | 17,77 | 22,11 | 25,23 | 26,70 | 27,88 | 29,63 | 13,50 | 3,63 | ![W1010410](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010410.png) |

(schéma: raw/askey3.pdf, p. 14 pour Y1715147, Y1715150, Y1715151 ; p. 15 pour Y1715152, Y1715153, W1010410)

# Valeurs statiques des dormants monoblocs

Les dormants monoblocs sont les dormants portant le doublage, titrés « Dormants monoblocs » sur les planches ; même lecture que les tableaux précédents (Ix aux six longueurs, Iy et W selon YY'). [1 p. 15-17]

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| W1010120 | 62,29 | 71,11 | 77,73 | 80,92 | 83,54 | 87,45 | 18,13 | 3,91 | ![W1010120](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010120.png) |
| W1010334 | 70,72 | 80,20 | 87,33 | 90,77 | 93,58 | 97,81 | 19,08 | 4,03 | ![W1010334](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010334.png) |
| W1010337 | 152,61 | 167,39 | 178,66 | 184,14 | 188,65 | 195,45 | 20,73 | 4,18 | ![W1010337](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010337.png) |
| W1715114 | 149,50 | 165,13 | 175,29 | 180,72 | 185,19 | 191,93 | 19,96 | 4,08 | ![W1715114](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1715114.png) |
| W1010123 | 26,15 | 31,33 | 34,98 | 36,66 | 38,01 | 39,98 | 10,93 | 2,91 | ![W1010123](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010123.png) |
| W1010124 | 43,48 | 50,53 | 55,56 | 57,90 | 59,79 | 62,55 | 11,67 | 3,01 | ![W1010124](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010124.png) |
| W1010336 | 69,11 | 78,43 | 85,13 | 88,28 | 90,81 | 94,54 | 12,32 | 3,10 | ![W1010336](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010336.png) |
| W1010125 | 106,11 | 118,44 | 127,39 | 131,62 | 135,04 | 140,10 | 13,00 | 3,19 | ![W1010125](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010125.png) |
| W1010133 | 13,33 | 16,78 | 19,24 | 20,39 | 21,31 | 22,66 | 11,59 | 3,19 | ![W1010133](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010133.png) |
| Y1715130 | 14,58 | 18,62 | 21,62 | 23,05 | 24,22 | 25,96 | 18,21 | 4,72 | ![Y1715130](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/y1715130.png) |

(schéma: raw/askey3.pdf, p. 15 pour W1010120, W1010334, W1010337 ; p. 16 pour W1715114, W1010123, W1010124, W1010336, W1010125, W1010133 ; p. 17 pour Y1715130)

La référence W1715114 est imprimée avec le préfixe W (**VER-768**).

# Valeurs statiques du seuil PMR

Le seuil PMR est le seuil bas à faible hauteur de la porte-fenêtre, qui facilite le passage des personnes à mobilité réduite (PMR). Même lecture que les tableaux précédents. Ix vaut 1,91 cm⁴ aux six longueurs ; W selon XX' est « - » aux six longueurs. [1 p. 17]

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| W1010384 | 1,91 | 1,91 | 1,91 | 1,91 | 1,91 | 1,91 | 0,79 | 0,79 | ![W1010384](/assets/profiles/askey-frappe-65-oc/valeurs-statiques/w1010384.png) |

(schéma: raw/askey3.pdf, p. 17)

Le module de résistance W selon YY' du seuil W1010384 (0,79 cm³) est égal à son Iy (0,79 cm⁴) (**VER-770**).

# Citations

[1] [Frappe 65 Ouvrant Caché ASKEY - Conception](raw/askey3.pdf), p. 12-17

# Voir aussi

- [Inerties des ouvrants, meneaux, feuillures rapportées et traverses, Frappe 65 OC ASKEY](/profiles/askey-frappe-65-oc-valeurs-statiques-ouvrants-meneaux-et-traverses.md)
- [Inerties des montants serrures et des renforts de la Frappe 65 Ouvrant Caché ASKEY](/profiles/askey-frappe-65-oc-valeurs-statiques-montants-serrures-et-renforts.md)
- [Profilés dormants et ouvrants ASKEY Frappe 65 Ouvrant Caché](/profiles/askey-frappe-65-oc-dormants-et-ouvrants.md)
- [Inerties des dormants coupe droite et des seuils du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-valeurs-statiques-dormants-et-seuils.md)
