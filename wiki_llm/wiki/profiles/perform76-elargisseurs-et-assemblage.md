---
type: Profilé
title: Élargisseurs et assemblage PERFORM76
description: Les quatre élargisseurs PERFORM76, les profils d'assemblage et les trois poteaux d'angle, réservés aux dormants sans aile 76171 et 76172.
tags: [perform76, elargisseur, poteau-angle, assemblage, renfort, profile]
gamme: PERFORM
systeme: 76
fournisseur: KÖMMERLING
usage: atelier
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
    last_modified: 2026-09-02
source_pages:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    pages: 22, 23
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
---

# Réservés aux dormants sans aile

Un **élargisseur** est un profil accolé au dormant pour agrandir la menuiserie sans changer de
dormant ; un **profil d'assemblage** relie deux menuiseries entre elles, côte à côte ou en angle.
Tous les profils de cette page se montent **uniquement sur les dormants 76171 et 76172**, les deux
dormants neufs sans aile. Aucun ne se monte sur un dormant rénovation ni sur le 76180 [1 p. 22 et
23]. Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

# Cotes des élargisseurs

Les quatre élargisseurs PERFORM76, cotes en mm.

| Élargisseur | Élargissement (mm) | Hauteur (mm) | Renfort acier nominal | Inertie $I_W$ / $I_G$ ($\text{cm}^4$) | Embout droit / biais | Coupe |
| --- | --- | --- | --- | --- | --- | ---: |
| 76700 | 15 | 76 | sans renfort | — | — | ![Élargisseur 76700](/assets/profiles/perform76/elargisseurs/elargisseur-76700.png) |
| 76701 | 30 | 76 | V312.Z (1,5 mm) | $I_W = 1,5$ / $I_G = 0,3$ | M302 / M303 | ![Élargisseur 76701](/assets/profiles/perform76/elargisseurs/elargisseur-76701.png) |
| 76702 | 60 | 76 | V314.Z (2,0 mm soudé) | $I_W = 5,7$ / $I_G = 8,4$ | M306 / M307 | ![Élargisseur 76702](/assets/profiles/perform76/elargisseurs/elargisseur-76702.png) |
| 76703 | 120 | 76 | V314.Z × 2 (2,0 mm soudé) | $I_W = 5,7$ / $I_G = 8,4$ | M308 / M309 | ![Élargisseur 76703](/assets/profiles/perform76/elargisseurs/elargisseur-76703.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 22 et raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, reg. 2.1.3 p. 2-3)

Sur la PERFORM76, le **renfort acier est en option sur les élargisseurs 76702 et 76703** : mention
en rouge de la planche, où le V314.Z est dessiné dans le 76702 et deux fois dans le 76703
[1 p. 22]. Chaque coupe montre l'élargisseur accolé au dormant, avec sa largeur d'élargissement
en haut et la hauteur de 76 mm à gauche.

**Règles de fixation et ventilation d'atelier** :
- Les élargisseurs empilés/couplés doivent être vissés entre eux : entraxe max. **400 mm** en blanc, **300 mm** en profilé filmé/couleur.
- À partir de 60 mm d'élargissement, prévoir une fixation à la maçonnerie par console ou équerre.
- Sur profilés filmés/couleur, percer impérativement un trou de ventilation de **Ø 5 mm à 100 mm des extrémités** de chaque préchambre extérieure pour éviter toute surchauffe thermique.

# Cotes des profils d'assemblage et liaisons de couplage

| Référence | Type | Cotes (mm) | Renfort associé | Inertie $I_W$ ($\text{cm}^4$) |
| --- | --- | --- | --- | --- |
| 76600 | profil d'assemblage de deux dormants | liaison 4,8 × 17 | - | - |
| 76606 | profilé de liaison plat (180°) | 4,8 × 17 | sans | — (reprise par dormants) |
| 76604 | profilé de liaison vertical (180°) | épaisseur 5,5 mm | entretoise | $I_W$ totale accouplement 6,4 à 11,4 |
| 76605 | profilé de liaison en H (180°) | largeur 48 mm, cote débit +31,5 mm | V330 (2,5 mm) ou V331 (2,5 mm) | $I_W = 5,5$ (V330) ou $10,0$ (V331) |
| 76608 | profilé de liaison en H lourd (180°) | largeur 48 mm, cote débit +30 mm | V288 (2,0 mm) | $I_W = 20,4$ ($I_W$ totale 26,8 à 33,2) |
| A250 | profilé contreventement aluminium | largeur 75,2 mm, cote débit +17,6 mm | V264 (jusqu'à 2 aciers) | $I_W = 17,9$ à $35,8$ ($I_W$ totale jusqu'à 74,6) |
| V477 | renfort d'accouplement extérieur | capot A235 + habillage 93000/93001 | V477 (2,5 mm) | $I_W = 49,3$ ($I_W$ totale jusqu'à 62,1) |
| 76821 | adaptateur de poteau d'angle | 55,5 / 12 / 12 / 13,5, hors tout 75,5 | — | — |
| 76822 | clip cornière | 72, décomposé 21 / 7 / 21 | — | — |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, reg. 2.5.2 p. 1-29 ; raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 23 pour les coupes)

# Cotes et renforts des poteaux d'angle

Les trois poteaux d'angle PERFORM76, cotes en mm et spécifications de renforcement d'atelier.

| Poteau | Angle | Référence | Dimensions (mm) | Renfort acier | Inertie $I_W$ / $I_G$ ($\text{cm}^4$) | Accessoires et isolants |
| --- | --- | --- | --- | --- | --- | --- |
| Poteau d'angle à 90° | 90° | 8355 | 104 × 88 | V263 (2,0 mm soudé) | $I_W = 14,4$ / $I_G = 14,4$ | Embout 90° M344, isolant I040, clameau S081 |
| Poteau d'angle à 135° | 135° | 8356 | 54 × 88 (onglet 45°, retombée 31) | V262 (2,0 mm) | $I_W = 7,6$ / $I_G = 7,6$ | Isolant I041, clameau S082 |
| Poteau d'angle variable | 90° à 180° | 8340 + 8341 | 84 × 84 (R42,5) | V265 (2,0 mm tubulaire soudé) | $I_W = 8,7$ / $I_G = 8,7$ | Isolants I042 (8340) et I043 (8341) en blanc uniquement, clameau S081 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, reg. 2.1.3 p. 14-15 et reg. 2.5.2 p. 31-40 ; raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 23 pour les coupes)

Cotes de débit X et Y du poteau d'angle variable 8340 + 8341 selon l'angle, en mm :

| Angle (°) | X (mm) | Y (mm) |
| --- | --- | --- |
| 90 | 104,8 | 14,8 |
| 100 | 97,0 | 21,5 |
| 110 | 90,3 | 27,3 |
| 120 | 84,3 | 32,4 |
| 130 | 79,0 | 37,0 |
| 135 | 76,4 | 39,2 |
| 140 | 74,0 | 41,2 |
| 150 | 69,4 | 45,3 |
| 160 | 65,0 | 49,1 |
| 170 | 60,8 | 52,9 |
| 180 | 56,4 | 56,4 |

# Coupes d'assemblage

Les profils d'assemblage relient deux menuiseries entre elles, côte à côte ou en angle. Chaque
coupe ci-dessous est une coupe horizontale vue de dessus : on y voit les deux dormants et la pièce
qui les relie [1 p. 23].

## Profil d'assemblage 76600

![Profil 76600](/assets/profiles/perform76/assemblage/profil-76600.png)

Deux dormants posés côte à côte, reliés par le profil 76600 (en haut et en bas de la jonction,
en jaune). La loupe montre la liaison, cotée **4,8 × 17 mm**.

## Clip cornière 76822

![Clip cornière 76822](/assets/profiles/perform76/assemblage/clip-corniere-76822.png)

Le clip cornière se clipse sur le dormant et reçoit une **cornière** (profil en L) de chaque
côté. Cotes : **72 mm** au total, décomposés en 21 / 7 / 21.

## Adaptateur 76821

![Adaptateur 76821](/assets/profiles/perform76/assemblage/adaptateur-76821.png)

L'adaptateur, qui raccorde le dormant au poteau d'angle, dessiné seul : 55,5 + 12 de long, soit
**75,5 mm** hors tout ; 12 et 13,5 mm de haut.

## Poteau d'angle à 90°, référence 8355

![Poteau d'angle à 90° 8355](/assets/profiles/perform76/assemblage/poteau-angle-8355.png)

Deux dormants à angle droit, reliés par le poteau 8355 (104 × 88 mm) ; l'adaptateur 76821 fait
la jonction avec le second dormant, et l'écart de **12 mm** est coté à gauche.

## Poteau d'angle à 135°, référence 8356

![Poteau d'angle à 135° 8356](/assets/profiles/perform76/assemblage/poteau-angle-8356.png)

Même principe à 135° : poteau 8356 de 54 × 88 mm, coupe à 45° et retombée de 31 mm, adaptateur
76821, écart de 12 mm.

## Poteau d'angle variable, références 8340 + 8341

![Poteau d'angle variable 8340 + 8341](/assets/profiles/perform76/assemblage/poteau-angle-8340-8341.png)

Le poteau variable se compose de deux profils, **8340** et **8341**, qui pivotent l'un dans
l'autre autour d'un rayon de **42,5 mm** : il règle tout angle de 90° à 180°. Cotes : 84 et 88 mm
pour le 8340, 84 mm pour le 8341, adaptateur 76821, écart de 12 mm.

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 23)

# Compatibilités

| Référence | 76171 | 76172 | 76177 | 76180 | 76185 |
| --- | --- | --- | --- | --- | --- |
| Élargisseurs 76700, 76701 | oui | oui | non | non | non |
| Élargisseurs 76702, 76703 | oui | oui | non | non | non |
| Renfort V314.Z | sur 76702 et 76703 uniquement | idem | non | non | non |
| Profil 76600 | oui | oui | non | non | non |
| Adaptateur 76821 | oui | oui | non | non | non |
| Clip cornière 76822 | oui | oui | non | non | non |
| Poteaux d'angle 8355, 8356, 8340 + 8341 | oui | oui | non | non | non |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 22 et 23)

# Ce que la source ne donne pas

Les références **8340, 8341, 8355 et 8356** sortent de la numérotation 76xxx de la gamme. Elles
appartiennent au catalogue [profine](/fournisseurs/profine.md), comme toutes les références du
cahier technique, sans que leur famille d'origine soit précisée.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, pages du PDF 22 et 23

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md)
- [KÖMMERLING](/fournisseurs/kommerling.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
