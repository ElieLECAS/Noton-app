---
type: Profilé
title: "Inerties des renforts, angles variables et jonctions de vantaux du coulissant ASKEY 65 NV"
description: "Moments d'inertie Ix (1 200 à 3 000 mm), Iy et modules W des profilés renforts W1916244, W1916272, W1916273, W1010144, W1010145, W001078L, Y1740042, Y1740043, des profilés d'angles variables W1910139, W1910140 et des jonctions de vantaux W3041249, Y3740022."
tags: [askey, coulissant, inertie, renfort, profile, angle-variable, conception, 65mm]
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
    pages: 22-23
generated:
  by: process:claude-code
  at: 2026-10-10T19:22:07Z
---

# Valeurs statiques des profilés renforts

Les profilés renforts du Coulissant 65 NV ASKEY ([gamme](/gammes/askey-coulissant-65-nv.md)) sont réunis sur des planches titrées « Les Profilés renforts » [1 p. 22]. Le moment d'inertie d'un profilé (en cm⁴) mesure sa résistance à la flexion ; le module de résistance W (en cm³) sert à contrôler la contrainte (règle de calcul : voir [Inerties des dormants coupe droite et des seuils](/profiles/askey-coulissant-65-nv-valeurs-statiques-dormants-et-seuils.md)).

Lecture : une ligne par profilé, la référence en première colonne. Les six colonnes « Ix » donnent le moment d'inertie I selon l'axe XX' (axe horizontal de la section dessinée) pour la longueur de profilé indiquée en mm ; « Iy » est le moment d'inertie I selon l'axe YY' ; « W selon YY' » est le module de résistance W, chiffré seulement pour l'axe YY' (ses six cases de l'axe XX' sont « - » sur la planche). Aucune valeur n'est interpolée entre deux longueurs. La colonne « Coupe » est la vignette de la section imprimée sur la planche.

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| W1916244 | 51,57 | 60,40 | 67,00 | 70,17 | 72,76 | 76,65 | 7,52 | 5,01 | ![W1916244](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1916244.png) |
| W1916272 | 32,55 | 34,46 | 35,92 | 36,63 | 37,22 | 38,11 | 27,41 | 6,75 | ![W1916272](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1916272.png) |
| W1916273 | 12,68 | 15,19 | 16,98 | 17,82 | 18,49 | 19,48 | 10,74 | 3,71 | ![W1916273](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1916273.png) |
| W1010144 | 10,28 | 12,35 | 13,67 | 14,25 | 14,71 | 15,34 | 1,02 | 0,85 | ![W1010144](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1010144.png) |
| W1010145 | 85,52 | 95,73 | 102,96 | 106,33 | 109,03 | 112,98 | 8,13 | 2,51 | ![W1010145](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1010145.png) |
| W001078L | 101,49 | 113,54 | 122,22 | 126,30 | 129,58 | 134,42 | 8,13 | 2,51 | ![W001078L](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w001078l.png) |
| Y1740042 | 52,62 | 63,54 | 71,93 | 76,04 | 79,42 | 84,55 | 10,54 | 5,14 | ![Y1740042](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740042.png) |
| Y1740043 | 19,83 | 26,48 | 32,27 | 35,35 | 38,03 | 42,34 | 134,32 | 20,96 | ![Y1740043](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740043.png) |

(schéma: raw/askey1.pdf, p. 22 pour W1916244 à Y1740042 ; p. 23 pour Y1740043)

Les valeurs Iy et W de W1010145 et de W001078L sont identiques (8,13 cm⁴ et 2,51 cm³) alors que leurs inerties Ix diffèrent (**VER-726**). Les désignations des profilés ne sont pas imprimées sur ces planches.

# Valeurs statiques des profilés pour angles variables

Les deux profilés pour angles variables portent la même lecture que le tableau précédent (planche titrée « Profilés pour angles variables ») [1 p. 23]. Les références imprimées sont W1910139 et W1910140.

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| W1910139 | 12,75 | 16,14 | 18,55 | 19,67 | 20,57 | 21,90 | 11,69 | 2,98 | ![W1910139](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1910139.png) |
| W1910140 | 11,98 | 15,20 | 17,49 | 18,56 | 19,42 | 20,68 | 10,38 | 2,81 | ![W1910140](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1910140.png) |

(schéma: raw/askey1.pdf, p. 23)

# Valeurs statiques des jonctions de vantaux en ligne

Les profilés de jonction de vantaux en ligne sont donnés sans longueur, comme les chicanes ; la planche est titrée « Jonction vantaux en ligne » [1 p. 23].

Lecture : une ligne par profilé. Les valeurs sont imprimées sans longueur, en quatre colonnes : Ixx' et Iyy' (moments d'inertie, cm⁴) et Ixx'/v et Iyy'/v (cm³ ; le « v » n'est pas défini sur la planche). La colonne « Coupe » est la vignette de la section imprimée sur la planche.

| Référence | Ixx' (cm⁴) | Iyy' (cm⁴) | Ixx'/v (cm³) | Iyy'/v (cm³) | Coupe |
| --- | --- | --- | --- | --- | ---: |
| W3041249 | 2,23 | 0,37 | 1,11 | 0,46 | ![W3041249](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3041249.png) |
| Y3740022 | 0,26 | 0,20 | 0,25 | 0,25 | ![Y3740022](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y3740022.png) |

(schéma: raw/askey1.pdf, p. 23)

# Citations

[1] [Coulissant 65 NV ASKEY - Conception](raw/askey1.pdf), p. 22-23

# Voir aussi

- [Inerties des ouvrants 24 mm, 28/32 mm et des chicanes du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-valeurs-statiques-ouvrants-et-chicanes.md)
- [Profilés dormants et ouvrants ASKEY Coulissant 65 NV](/profiles/askey-coulissant-65-nv-dormants-et-ouvrants.md)
