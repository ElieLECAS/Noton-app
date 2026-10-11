---
type: Profilé
title: "Inerties des dormants coupe droite et des seuils du coulissant ASKEY 65 NV"
description: "Moments d'inertie Ix à 1 200 à 3 000 mm, Iy et modules de résistance W des dormants coupe droite Y1740019 à Y1740027, W1041279, W1041280, W1041284, du seuil réduit RPT W1040050 et du seuil PMR rapporté W3090163 du Coulissant 65 NV ASKEY."
tags: [askey, coulissant, inertie, dormant, seuil, seuil-pmr, rpt, profile, conception, 65mm]
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
    pages: 14-16
generated:
  by: process:claude-code
  at: 2026-10-10T19:22:07Z
---

# Valeurs statiques des dormants coupe droite

Le dormant est le cadre fixe de la fenêtre coulissante, scellé dans la baie ; en coupe droite, ses traverses et ses montants sont assemblés par rapprochement, bout à bout (voir le [Coulissant 65 NV ASKEY](/gammes/askey-coulissant-65-nv.md)). Le moment d'inertie d'un profilé (en cm⁴) mesure sa résistance à la flexion : plus il est grand, moins le profilé fléchit sous la charge. Le module de résistance W (en cm³) sert à contrôler la contrainte dans le profilé. Les définitions des sigles du métier sont dans le [glossaire](/reference/glossaire.md).

Pour les profilés à rupture de pont thermique (barrettes isolantes de 30 mm entre les deux demi-coquilles d'aluminium), « pour la vérification en déformation d'un profilé à rupture de pont thermique, le calcul de la flèche, nécessite la connaissance du moment d'inertie Ix qui varie en fonction de la longueur. Cette valeur est calculée conformément à l'annexe C de la norme EN 14024. » [1 p. 14]

« La preuve statique consiste à vérifier que la contrainte résultant de la combinaison d'efforts de chargement et de température hiver/été n'excède pas en tout point du profilé la contrainte admissible et que les efforts de cisaillement induits ne dépassent pas la tenue en glissement du sertissage. Dans un souci de simplification, il est fourni par longueur un module de résistance Wx permettant de limiter la vérification à un calcul de contrainte en flexion, la tenue selon les autres critères étant garantie. Dès lors, en fonction du cas de charge, il suffit de calculer le moment de flexion maximum Mfmax et de vérifier que Mfmax/Wx < σ0.2 (classiquement, σ0.2 = 160 N/mm2) » [1 p. 14].

Lecture des tableaux : une ligne par profilé, la référence en première colonne. Les six colonnes « Ix » donnent le moment d'inertie I selon l'axe XX' (axe horizontal de la section dessinée, repères x et x' sur la planche) pour la longueur de profilé indiquée en mm ; « Iy » est le moment d'inertie I selon l'axe YY' (repères y et y'). La planche imprime pour chaque profilé deux lignes : « Inertie I (cm⁴) » et « Module de résistance W (cm3) ». La ligne du module de résistance porte « - » dans les six colonnes de l'axe XX' pour tous les profilés ; seule sa colonne de l'axe YY' est chiffrée : elle est donnée ici dans la colonne « W selon YY' ». Aucune valeur n'est interpolée entre deux longueurs. La colonne « Coupe » est la vignette de la section imprimée sur la planche. Les valeurs de W selon XX' ne sont pas fournies (**VER-723**).

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| Y1740019 | 45,33 | 53,49 | 59,52 | 62,40 | 64,74 | 68,23 | 5,53 | 2,51 | ![Y1740019](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740019.png) |
| Y1740020 | 71,30 | 82,23 | 90,42 | 94,36 | 97,57 | 102,39 | 5,96 | 2,74 | ![Y1740020](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740020.png) |
| Y1740021 | 108,76 | 123,48 | 134,65 | 140,07 | 144,52 | 151,22 | 6,45 | 2,95 | ![Y1740021](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740021.png) |
| Y1740023 | 72,84 | 87,75 | 100,14 | 106,52 | 111,96 | 120,51 | 14,23 | 4,66 | ![Y1740023](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740023.png) |
| Y1740024 | 78,78 | 94,38 | 107,38 | 114,09 | 119,81 | 128,81 | 14,67 | 4,76 | ![Y1740024](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740024.png) |
| Y1740025 | 150,77 | 174,14 | 194,00 | 204,39 | 213,33 | 227,54 | 15,36 | 4,83 | ![Y1740025](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740025.png) |
| Y1740026 | 161,33 | 185,65 | 206,34 | 217,18 | 226,51 | 241,36 | 15,72 | 4,90 | ![Y1740026](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740026.png) |
| Y1740027 | 14,00 | 18,17 | 21,28 | 22,78 | 24,00 | 25,82 | 10,92 | 3,79 | ![Y1740027](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/y1740027.png) |
| W1041279 | 22,89 | 29,21 | 34,17 | 36,63 | 38,69 | 41,82 | 14,52 | 5,07 | ![W1041279](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041279.png) |
| W1041280 | 22,89 | 29,21 | 34,17 | 36,63 | 38,69 | 41,82 | 14,52 | 5,07 | ![W1041280](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041280.png) |
| W1041284 | 10,81 | 13,46 | 15,28 | 16,11 | 16,76 | 17,71 | 3,33 | 1,75 | ![W1041284](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1041284.png) |

(schéma: raw/askey1.pdf, p. 14 pour Y1740019 à Y1740025 ; p. 15 pour Y1740026, Y1740027, W1041279, W1041280, W1041284)

Les valeurs de W1041279 et de W1041280 sont identiques à toutes les longueurs et selon YY', alors que leurs sections dessinées diffèrent (**VER-724**). Les désignations des profilés (monobloc, traverse basse, traverse haute, rejingot) ne sont pas imprimées sur ces planches : la planche porte la référence et la coupe, les désignations sont sur les planches à l'échelle 1/1 de la suite du document.

# Valeurs statiques du seuil réduit RPT

Le seuil réduit à rupture de pont thermique (RPT) est le profilé de seuil bas, à faible hauteur, du coulissant ; la planche le titre « Seuil réduit Rpt » [1 p. 16]. Même lecture que le tableau précédent : une ligne par profilé, Ix selon XX' aux six longueurs, Iy et W selon YY'. Le module de résistance W selon XX' est « - » aux six longueurs.

| Référence | Ix à 1 200 mm (cm⁴) | Ix à 1 600 mm (cm⁴) | Ix à 2 000 mm (cm⁴) | Ix à 2 250 mm (cm⁴) | Ix à 2 500 mm (cm⁴) | Ix à 3 000 mm (cm⁴) | Iy selon YY' (cm⁴) | W selon YY' (cm³) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| W1040050 | 6,22 | 6,22 | 6,22 | 6,22 | 6,22 | 6,22 | 2,01 | 1,40 | ![W1040050](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w1040050.png) |

(schéma: raw/askey1.pdf, p. 16)

L'inertie Ix du W1040050 est la même, 6,22 cm⁴, aux six longueurs imprimées [1 p. 16].

# Valeurs statiques du seuil PMR rapporté

Le seuil PMR rapporté est le profilé de seuil bas rapporté pour le passage des personnes à mobilité réduite (PMR). Les valeurs sont imprimées sans longueur : quatre colonnes, Ixx' et Iyy' en cm⁴, Ixx'/v et Iyy'/v en cm³ (le « v » n'est pas défini sur la planche).

| Référence | Ixx' (cm⁴) | Iyy' (cm⁴) | Ixx'/v (cm³) | Iyy'/v (cm³) | Coupe |
| --- | --- | --- | --- | --- | ---: |
| W3090163 | 71,62 | 1,92 | 10,77 | 1,82 | ![W3090163](/assets/profiles/askey-coulissant-65-nv/valeurs-statiques/w3090163.png) |

(schéma: raw/askey1.pdf, p. 16)

# Citations

[1] [Coulissant 65 NV ASKEY - Conception](raw/askey1.pdf), p. 14-16

# Voir aussi

- [Inerties des ouvrants 24 mm, 28/32 mm et des chicanes du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-valeurs-statiques-ouvrants-et-chicanes.md)
- [Profilés dormants et ouvrants ASKEY Coulissant 65 NV](/profiles/askey-coulissant-65-nv-dormants-et-ouvrants.md)
