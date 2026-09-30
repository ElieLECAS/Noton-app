---
type: Profilé
title: Assemblages du système 70
description: Les méthodes d'assemblage autorisées entre dormants, ouvrants et traverses du système 70 Plateforme, les seuils aluminium 9F67, 9F68, 9F69 et Z043, et les embouts, sets d'assemblage mécanique, patins d'étanchéité et équerres de chaque dormant, meneau et traverse.
tags: [systeme-70, e-volution, profine, assemblage, soudure, traverse, meneau, seuil, dormant, ouvrant]
systeme: 70
fournisseur: KÖMMERLING
usage: atelier
status: draft
sources:
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    id: dtd-6-16-2335-v5
    title: DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION
    last_modified: 2024-12-19
  - resource: raw/poster-kommerling-70-principaux-2025-03.pdf
    id: poster-kommerling-70-principaux
    title: Poster Kömmerling Gamme 70, profilés principaux, mars 2025
    last_modified: 2025-03-31
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés, système F 91, édition août 2008
    last_modified: 2008-08-31
source_pages:
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    pages: 3-5, 9, 12-13, 16, 24, 26-27, 30-31
  - resource: raw/poster-kommerling-70-principaux-2025-03.pdf
    pages: 1
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 12, 14, 22-23, 46-47, 108
generated:
  by: process:claude-code
  at: 2026-09-19T09:00:00Z
---

# Méthodes d'assemblage des traverses et meneaux

Dans le **système 70 Plateforme** de [profine](/fournisseurs/profine.md) (TROCAL e.XCLUSIVE, KBE
e.MOTION, KÖMMERLING e.VOLUTION), une traverse ou un meneau (le profilé horizontal ou vertical
qui recoupe un cadre) s'assemble sur un dormant ou sur un ouvrant mécaniquement ou par
thermosoudure, et **le couple profilé / traverse détermine les méthodes admises** : c'est l'objet
des tableaux 2 et 3 du DTD [1 p. 3, 12-13]. Chaque tableau porte sa propre légende, reprise
telle quelle.

| Code | Tableau 2, dormants / traverses | Tableau 3, ouvrants / traverses |
| --- | --- | --- |
| M | assemblage mécanique | assemblage mécanique |
| S | soudure en V | soudure |
| SP\* | soudure à plat avec équerres | - |
| SP | - | soudure à plat |

(schéma: raw/dtd-6-16-2335-v5-e-volution.pdf, p. 12 et 13)

# Assemblage dormant / traverse

Tableau 2 du DTD : méthodes admises entre chaque dormant du système 70 (une ligne par dormant,
groupés comme la source en standard, large et rénovation) et chacune des quatre traverses. Une
cellule énonce toutes les méthodes possibles, séparées par `/`.

| Dormant | Famille | Traverse 6127 | Traverse 2427 | Traverse 2425 | Traverse 6157 |
| --- | --- | --- | --- | --- | --- |
| 6100 | standard | M/S/SP\* | M | M | M/S |
| 6101 | standard | M/S/SP\* | M | M | M/S |
| 2502 | standard | M/SP\* | M | M | M |
| 2501 | standard | M/SP\* | M | M | M |
| 6104 | large | M/S/SP\* | M | M | M/S |
| 6108 | large | M/S/SP\* | M | M | M/S |
| 6109 | large | M/S/SP\* | M | M | M/S |
| 6110 | large | M/S/SP\* | M | M | M/S |
| 6111 | large | M/S/SP\* | M | M | M/S |
| 6158 | large | M/S/SP\* | M | M | M/S |
| 6102 | rénovation | M/S/SP\* | M | M | M/S |
| 6105 | rénovation | M/S/SP\* | M | M | M/S |
| 6106 | rénovation | M/S/SP\* | M | M | M/S |
| 6107 | rénovation | M/S/SP\* | M | M | M/S |
| 6155 | rénovation | M/S/SP\* | M | M | M/S |
| 6156 | rénovation | M/S/SP\* | M | M | M/S |
| 6159 | rénovation | M/S/SP\* | M | M | M/S |

(schéma: raw/dtd-6-16-2335-v5-e-volution.pdf, p. 12)

Les dormants 2501 et 2502 sont les seuls du tableau 2 sans soudure en V (S) sur la traverse 6127,
et les seuls à n'avoir que l'assemblage mécanique sur la traverse 6157. La soudure à plat avec
équerres (SP\*) n'est donnée que pour la traverse 6127 [1 p. 12].

# Assemblage ouvrant / traverse

Tableau 3 du DTD : méthodes admises entre chaque ouvrant du système 70 (une ligne par ouvrant) et
chacune des cinq traverses.

| Ouvrant | Traverse 6126 | Traverse 6127 | Traverse 2427 | Traverse 2425 | Traverse 6157 |
| --- | --- | --- | --- | --- | --- |
| 6112C | M/SP | M/S/SP | M | M | M/S |
| 6113 | M/SP | M/S/SP | M | M | M/S |
| 6115 | M/SP | M/S/SP | M | M | M/S |
| 6116 | M/SP | M/S/SP | M | M | M/S |
| 6117 | M/SP | M/SP | M | M | M |
| 6118 | M/SP | M/SP | M | M | M |
| 6119 | M/SP | M/SP | M | M | M |
| 6120 | M/SP | M/SP | M | M | M |
| 6121 | - | M/SP | M | M | M |
| 6122 | - | M/SP | M | M | M |
| 6123 | - | M/SP | M | M | M |
| 6124 | - | M/SP | M | M | M |
| 6150 | - | M/SP | M | M | M |
| 6151 | - | M/SP | M | M | M |
| 6152 | - | M/SP | M | M | M |
| 6153 | - | M/SP | M | M | M |
| 2416 | M/SP | M/S/SP | M | M | M/S |

(schéma: raw/dtd-6-16-2335-v5-e-volution.pdf, p. 13)

Le tableau 3 ne donne **aucune méthode d'assemblage des ouvrants 6121 à 6124 et 6150 à 6153 sur
la traverse 6126** (case marquée d'un tiret). La soudure (S) n'y est donnée que sur les traverses
6127 et 6157, et seulement pour les ouvrants 6112C, 6113, 6115, 6116 et 2416 [1 p. 13]. Le tableau 3
donne la soudure à plat (SP) sur la traverse 6126, alors que le § 2.8.5 réserve les soudures à plat
à l'assemblage du meneau 6127 (**INC-84**).

# Set d'assemblage dormant / seuil

Tableau 4 du DTD : pièce d'assemblage à employer entre chaque dormant et chacun des trois seuils
aluminium, dont la hauteur est écrite dans l'en-tête. Un `ou` dans une cellule signale deux pièces
également admises.

| Dormant | Famille | Seuil 9F67 (20 mm) | Seuil 9F68 (36 mm) | Seuil Z043 (20 mm) |
| --- | --- | --- | --- | --- |
| 6100 | standard | 9F57 ou 9F72 | 9F61 ou 9F72 | 9F72 |
| 6101 | standard | 9F65 ou 9F71 | 9F66 ou 9F71 | 9F71 |
| 2502 | standard | 9F65 ou J077 | 9F66 ou J077 | J077+M002 |
| 2501 | standard | 9F65 | 9F66 | - |
| 6104 | large | 9F65 ou 9F71 | 9F66 ou 9F71 | 9F71 |
| 6108 | large | 9F65 ou 9F71 | 9F66 ou 9F71 | 9F71 |
| 6109 | large | 9F65 ou 9F71 | 9F66 ou 9F71 | 9F71 |
| 6110 | large | 9F65 ou 9F71 | 9F66 ou 9F71 | 9F71 |
| 6111 | large | 9F65 ou 9F71 | 9F66 ou 9F71 | 9F71 |
| 6158 | large | 9F65 ou 9F71 | 9F66 ou 9F71 | 9F71 |
| 6102 | rénovation | 9F57 ou 9F72 | 9F61 ou 9F72 | 9F72 |
| 6105 | rénovation | 9F58 ou 9F72 | 9F62 ou 9F72 | 9F72 |
| 6106 | rénovation | 9F59 ou 9F72 | 9F63 ou 9F72 | 9F72 |
| 6107 | rénovation | 9F60 ou 9F72 | 9F64 ou 9F72 | 9F72 |
| 6155 | rénovation | 9F58 ou 9F72 | 9F62 ou 9F72 | 9F72 |
| 6156 | rénovation | J087 ou 9F72 | J088 ou 9F72 | 9F72 |
| 6159 | rénovation | 9F72 | 9F72 | 9F72 |

(schéma: raw/dtd-6-16-2335-v5-e-volution.pdf, p. 13)

Le tableau 4 ne donne **aucun set pour le dormant 2501 sur le seuil Z043** (tiret). Le 2502 y
demande la pièce J077 complétée de l'embout M002, seul cas de la table où deux pièces sont
cumulées. Excepté dans le cas d'un oscillo-coulissant, le cadre dormant peut être muni d'un seuil
aluminium selon le tableau 4 [1 p. 4, 13].

La table du DTD donne au dormant **6159** la seule pièce 9F72 sur les trois seuils (relu en image
le 28/09/2026) ; le poster des profilés principaux lui donne les embouts **M833** (seuil 9F67) et
**M834** (seuil 9F68) et ne le nomme pas parmi les dormants du set 9F72 — entrée **CTR-39** du
registre [Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

# Seuils aluminium

Le **seuil** est le profilé aluminium qui remplace la traverse basse du dormant d'une porte-fenêtre.
Le poster des profilés principaux dessine quatre seuils en coupe hachurée ; chaque coupe porte la
largeur en haut et la hauteur à droite, en mm.

| Seuil | Largeur (mm) | Hauteur totale (mm) | Hauteur de la partie arrière (mm) | Coupe |
| --- | --- | --- | --- | ---: |
| 9F67 | 70 | 20 | 16 | ![Seuil 9F67](/assets/profiles/systeme70/seuils/seuil-9f67.png) |
| 9F68 | 70 | 36 | 16 | ![Seuil 9F68](/assets/profiles/systeme70/seuils/seuil-9f68.png) |
| 9F69 | 125,5 | - | 16 | ![Seuil 9F69](/assets/profiles/systeme70/seuils/seuil-9f69.png) |
| Z043 | 125 | - | 16 | ![Seuil Z043](/assets/profiles/systeme70/seuils/seuil-z043.png) |

(schéma: raw/poster-kommerling-70-principaux-2025-03.pdf, p. 1, coin inférieur droit)

Le 9F67 porte en plus une cote de 1,5 mm, à gauche. Les 9F69 et Z043 ne portent pas de hauteur
totale sur le poster. Le **9F69** n'apparaît dans aucune table d'assemblage du DTD.

L'annexe du DTD dessine trois seuils, sans hachure, avec leur largeur en bas et leur hauteur à
gauche, en mm [1 p. 16] :

| Seuil | Largeur (mm) | Hauteur (mm) |
| --- | --- | --- |
| 9F68 | 70 | 36 |
| 9F67 | 70 | 20 |
| Z043 | 125 | - |

(schéma: raw/dtd-6-16-2335-v5-e-volution.pdf, p. 16)

La cote verticale du Z043 est portée à droite du dessin et coupée par le bord de l'image sur la
page ; seule la hauteur de 20 mm de l'en-tête du tableau 4, « Z043 (20mm) », la donne [1 p. 13, 16].

## Seuil 9C42 et profilé 9F46 des plans de 2008

Le classeur KÖMMERLING « Système e.VOLUTION, plan des profilés » d'août 2008 dessine, sur les
planches des ouvrants de porte 2416 et 2415 (registre 1.2, planche des ouvrants), le profilé
**9C42** et le profilé **9F46**, en coupe hachurée, sans légende de fonction. La planche des
seuils de porte d'entrée du registre 2 les légende « Seuil alu F95-50- 9C42 » et « Rejet d'eau alu
F95-50- 9F46 », sous les ouvrants 2416 et 2415 (coupes sur [Types d'ouverture et plans de combinaison du système 70](/profiles/systeme-70-plans-de-combinaison.md)) [3 p. 108]. Le 9C42 a la forme
d'un seuil : une partie arrière de 20 mm de haut et un nez incliné vers l'extérieur ; le 9F46
porte en pied une brosse dessinée en noir. Cotes en mm [3 p. 12, 14].

| Profilé | Désignation complète | Largeur (mm) | Hauteur (mm) | Autre cote (mm) | Page PDF | Coupe |
| --- | --- | --- | --- | --- | --- | ---: |
| 9C42 | F95-50- 9C42 | 70 | 20 | - | 12 | ![Profilé 9C42](/assets/profiles/systeme70/seuils/seuil-9c42-evo2008.png) |
| 9C42 | F95-50- 9C42 | 70 | 20 | - (dessin inversé, nez à droite) | 14 | ![Profilé 9C42, dessin inversé](/assets/profiles/systeme70/seuils/seuil-9c42-inverse-evo2008.png) |
| 9F46 | F95-50- 9F46 | 44,5 hors tout | 18 | 16,8 : largeur de la partie inclinée, à gauche | 12 | ![Profilé 9F46](/assets/profiles/systeme70/accessoires/profil-9f46-evo2008.png) |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 12 et 14)

Le 9C42 et le 9F46 ne sont dessinés ni au poster, ni au DTD, ni au manuel de mise en œuvre de
2023, qui dessinent les seuils 9F67, 9F68, 9F69 et Z043. Le montage du seuil 9C42, du rejet d'eau
9F46 et leurs débits par dormant sont sur
[Seuil aluminium de porte 9C42 du système 70, 2008](/procedures/seuil-alu-9c42-systeme-70-evo2008.md).

# Pièces d'assemblage dessinées sur le poster

La colonne de droite du poster des profilés principaux porte les pièces qui assemblent les
profilés entre eux et sur les seuils, chacune avec la liste des profilés auxquels elle est
destinée. Une ligne par pièce et par groupe de profilés, dans les termes de la planche [2 p. 1].

## Embouts pour seuil 9F67 et 9F68

L'**embout de seuil** ferme l'extrémité du seuil contre le montant du dormant et reçoit son
vissage.

| Embout | Seuil | Pour les dormants |
| --- | --- | --- |
| 9F57 | 9F67 | 6100, 6102 |
| 9F58 | 9F67 | 6105, 6155 |
| 9F59 | 9F67 | 6106 |
| 9F60 | 9F67 | 6107 |
| J087 | 9F67 | 6156 |
| 9F65 | 9F67 | 6101, 6104, 6108, 6109, 6110, 6111, 2501, 2502 |
| M833 | 9F67 | 6159 |
| 9F61 | 9F68 | 6100, 6102 |
| 9F62 | 9F68 | 6105, 6155 |
| 9F63 | 9F68 | 6106 |
| 9F64 | 9F68 | 6107 |
| J088 | 9F68 | 6156 |
| 9F66 | 9F68 | 6101, 6104, 6108, 6109, 6110, 6111, 2501, 2502 |
| M834 | 9F68 | 6159 |

(schéma: raw/poster-kommerling-70-principaux-2025-03.pdf, p. 1, colonne de droite)

![Embouts pour seuil 9F67, angle d'assemblage 9326](/assets/profiles/systeme70/assemblages/embouts-seuil-9f67.png)

![Embouts pour seuil 9F68, angle d'assemblage 9669](/assets/profiles/systeme70/assemblages/embouts-seuil-9f68.png)

Chaque bloc montre à gauche un angle de dormant sur seuil en perspective, repéré **9326** (seuil
9F67) et **9669** (seuil 9F68), et à droite l'embout en perspective, avec son trou oblong de
vissage. Le manuel de mise en œuvre Système 70 Plateforme désigne le 9326 comme « Support de cale de vitrage » (PDF p. 126) : la lecture de ce repère est à vérifier (**VER-87**).

## Set d'assemblage mécanique dormant / seuil

| Set | Pour les dormants |
| --- | --- |
| 9F71 | 6101, 6104, 6108, 6109, 6110, 6111 |
| 9F72 | 6100, 6102, 6105, 6106, 6107, 6155, 6156 |
| J077 | 2502 |

« Pour une bonne applique de l'aile de 60 mm du dormant 6107, sur le seuil, utilisez l'insert
9F78 ! » [2 p. 1]

![Set d'assemblage mécanique dormant / seuil, avec l'insert 9F78](/assets/profiles/systeme70/assemblages/set-dormant-seuil-9f71-9f72-j077.png)

Le dessin éclaté montre, de haut en bas, la pièce d'assemblage avec sa vis, sa rondelle et sa
goupille, le montant du dormant marqué « Avec le dormant 6107 », la pièce d'ancrage basse, l'insert
**9F78** et le seuil avec ses deux vis de fixation.

## Sets d'assemblage mécanique des meneaux et traverses

| Set | Type sur la planche | Pour les profilés |
| --- | --- | --- |
| 9F73 | set d'assemblage mécanique en T, vissage dans la feuillure | 2425 |
| 9F76 | set d'assemblage mécanique en T, vissage dans la feuillure | 6127, 6157 |
| 9F75 | set d'assemblage mécanique en T, vissage dans la feuillure | 6126 |
| 9316 | set d'assemblage mécanique en T | 2425 |
| 9B51 | set d'assemblage mécanique en T | 2427 |
| 9B52 | set d'assemblage mécanique en croix | 2427 |
| 9F39 | assemblage mécanique pour angle variable | 2425 |

(schéma: raw/poster-kommerling-70-principaux-2025-03.pdf, p. 1, colonne de droite et cinquième bande)

![Set d'assemblage mécanique en T, vissage dans la feuillure](/assets/profiles/systeme70/assemblages/set-t-vissage-feuillure.png)

Le set en T à vissage dans la feuillure est une platine à deux ailes percées, qui se visse en fond
de feuillure, avec sa goupille et ses bouchons.

![Set d'assemblage mécanique en T 9316 et 9B51](/assets/profiles/systeme70/assemblages/set-t-9316-9b51.png)

![Set d'assemblage mécanique en croix 9B52](/assets/profiles/systeme70/assemblages/set-croix-9b52.png)

Les sets en T et en croix sont des pièces d'ancrage logées dans la chambre de renfort, tenues par
une goupille et serrées par une vis à tête cylindrique ; le set en croix ajoute une seconde pièce
d'ancrage traversée par une vis plus longue.

![Assemblage mécanique pour angle variable 9F39](/assets/profiles/systeme70/meneaux/assemblage-9f39-2425.png)

## Patins d'étanchéité et équerres

| Pièce | Désignation sur la planche | Pour les profilés |
| --- | --- | --- |
| 9718 | patin d'étanchéité | 6127, 6157 |
| 9719 | patin d'étanchéité | 2425 |
| 9B89 | patin d'étanchéité | 2427 |
| 9B56 | patin d'étanchéité | 6126 |
| 9714 | équerres G/D | - |

![Patin d'étanchéité et équerres G/D 9714](/assets/profiles/systeme70/assemblages/patins-etancheite-equerres-9714.png)

(schéma: raw/poster-kommerling-70-principaux-2025-03.pdf, p. 1, colonne de droite)

Le **patin d'étanchéité** est la plaque souple posée entre l'about de la traverse et le profilé
qui la reçoit. Les équerres 9714 sont dessinées par paire, gauche et droite ; le DTD les emploie
sur une partie fixe à seuil et en complément de la soudure à plat du 6127 (voir plus bas).

Le seuil aluminium est exclu sur un **oscillo-coulissant** [1 p. 4].

# Nomenclature des accessoires du DTD

La planche « Accessoires » de l'annexe du DTD dessine en perspective, sans cote sauf pour les
cales, les pièces d'accessoire du système 70 : embouts, sets d'assemblage, pièces de seuil,
patins d'étanchéité, cales. Elle ne donne pas leur fonction ; les fonctions ci-dessous sont celles
que le texte du DTD attribue à la même référence, au paragraphe cité [1 p. 24].

| Rangée sur la planche | Références telles qu'écrites | Fonction donnée par le texte du DTD |
| --- | --- | --- |
| 1 | 9414.1, 9A81, 9F30, 9F31, 9A82, 9F33, 9F35, 9F29, 9F28, M278, M375, M664 | embouts collés des battements (§ 2.2.3.2.1), sans attribution par référence |
| 2 | M626, M628, M629, et deux pièces sans légende | - |
| 3 | 9B51 / 9B52, 9C69, 9316.2, 9312, J077, M375, 9714 L+R | J077 : set dormant / seuil (tableau 4) ; 9714 : équerres (§ 2.2.3.1.1, § 2.2.3.1.4) |
| 4 | 9F71, 9F72, 9F73, 9F76, 9F75 | 9F71, 9F72 : sets dormant / seuil (tableau 4, § 2.2.3.1.4) |
| 5 | 9718, 9719, 9B56, 9714, 9F13, M771 | 9718 : patin d'étanchéité serré par l'assemblage par alvéovis (§ 2.2.3.3) |
| 6 | 9F66, 9F65, 9F57, 9F61, 9F58, 9F62, 9F59 | pièces d'assemblage dormant / seuil (tableau 4, § 2.2.3.1.4) |
| 7 | 9F63, 9F60, 9F64, J087, J088, M299, A272, M002 | 9F60, 9F63, 9F64, J087, J088 : pièces d'assemblage dormant / seuil ; M299 : patin d'étanchéité (§ 2.2.3.3) ; A272 : support de cale ; M002 : complément du J077 sur Z043 (tableau 4) |
| 8 | M298, M329, 9F97, EMBJ701 | M298 : étanchéité pièce d'appui / tapée (§ 2.2.3.1.3) |
| 9 | M325, 9D02 (25,5), 9D03 (33), 9D04 (58), 9D05 (88) | - |
| 10 | M613, M450, G008, G250, embout M643, embout M646, 5685, 5686 | M613 : étanchéité pièce d'appui / tapée ; M450, G008 : busette et manchon de drainage du capot complet ; G250 : mousse des angles bas ; M643, M646 : embouts d'appui et de tapée alu (§ 2.2.3.1.3, § 2.2.3.5) |

(schéma: raw/dtd-6-16-2335-v5-e-volution.pdf, p. 24)

Les cotes entre parenthèses sont les hauteurs, en mm, portées à droite des cales 9D02 à 9D05.
La référence **M375** est écrite deux fois sur la planche : sur un embout de la rangée 1 et sur la
pièce dessinée à côté des équerres 9714 L+R, rangée 3.

![Accessoires du DTD, rangées 1 à 3 : embouts de battement et sets d'assemblage](/assets/certifications/dtd-6-16-2335/accessoires-1-embouts-et-sets.png)

![Accessoires du DTD, rangées 4 à 8 : sets, patins, pièces de seuil](/assets/certifications/dtd-6-16-2335/accessoires-2-seuils-et-patins.png)

![Accessoires du DTD, rangées 9 et 10 : cales, patins, busette, manchon, embouts](/assets/certifications/dtd-6-16-2335/accessoires-3-cales-et-embouts.png)

Chaque pièce est dessinée en perspective avec sa référence sous le dessin ; les vis, goupilles
et bouchons dessinés à côté des sets appartiennent au set voisin.

# Pose du seuil aluminium, assemblages mécaniques et soudure à plat

Le mode opératoire de pose du seuil aluminium (cas des assemblages 9F57 à 9F66, J087 et J088, et
cas des 9F71, 9F72, J077 + M002, partie fixe sur seuil 9F67 ou Z043) est transcrit au § 2.2.3.1.4,
et les quatre types d'assemblage mécanique des meneaux et traverses (alvéovis, pièces d'ancrage et
goupille, équerres, ancrage avec fixation en feuillure) au § 2.2.3.3, de la page
[DTD n° DBV-24-6/16-2335_V5](/certifications/dtd-6-16-2335.md) [1 p. 4-5]. Les planches
d'assemblage, la soudure à plat du meneau 6127 (zone soudée d'inertie X 8,43 cm⁴ et Y 5,39 cm⁴,
équerres 9714 L+R dont les plots de centrage ont été supprimés) et l'usinage de la traverse avant
soudure sont sur [Fabrication et assemblage du système 70](/procedures/fabrication-systeme-70.md)
[1 p. 26-27].

Le manuel de mise en œuvre du système 70 donne, pour les mêmes seuils (et le 9F69), les débits de
seuil et de montant par dormant, les contours de fraisage, le gabarit de perçage 9918, le
drainage et le rejet d'eau A465 : voir
[Mise en œuvre du seuil aluminium du système 70](/procedures/mise-en-oeuvre-seuil-systeme-70.md),
[Porte-fenêtre avec fixe latéral du système 70](/procedures/porte-fenetre-fixe-lateral-systeme-70.md)
et [Porte d'entrée du système 70](/procedures/porte-d-entree-systeme-70.md) ; les accouplements
d'éléments et poteaux d'angle sont sur
[Accouplement d'éléments du système 70](/procedures/accouplement-elements-systeme-70.md).

# Accessoires et sets d'assemblage des plans e.VOLUTION de 2008

Les planches « Accessoires » 2 et 3 du classeur KÖMMERLING e.VOLUTION d'août 2008 (registre 1.6,
p. 46 et 47, échelle 1:2) dessinent les cales de transport, le recouvrement de seuil, le profilé
1023, les ponts de vitrage et l'équerre d'angle, puis trois tableaux : les sets d'assemblage de
chaque meneau, les sets d'assemblage des dormants et le set d'assemblage du meneau 2425. Les
joints de la première planche « Accessoires » (p. 45) sont sur
[Joints et garnitures des systèmes profine](/profiles/joints-et-garnitures-profine.md) [3 p. 46-47].

## Cales, recouvrement de seuil, profilé 1023, ponts de vitrage et équerre (p. 46)

Une ligne par pièce, avec la désignation et la légende écrites sur la planche. Les pièces sans
cote sont dessinées en perspective ; la [cale de transport](/reference/glossaire.md) est
dessinée seule puis en place dans la gorge d'un profilé coupé ; les ponts de vitrage et l'équerre
sont dessinés en place dans un angle d'ouvrant [3 p. 46].

| Pièce | Désignation complète | Légende sur la planche | Cotes portées (mm) | Dessin |
| --- | --- | --- | --- | ---: |
| 9A39 | F90-75- 9A39 | Cale de transport | - | ![Cale de transport 9A39](/assets/profiles/systeme70/accessoires/cale-transport-9a39-evo2008.png) |
| 9856 | F00-75- 9856 | Cale de transport | - | ![Cale de transport 9856](/assets/profiles/systeme70/accessoires/cale-transport-9856-evo2008.png) |
| 9G13 | F00-57- 9G13 | Recouvrement de seuil | 18,3 de large ; 27,5 et 22,6 de haut | ![Recouvrement de seuil 9G13](/assets/profiles/systeme70/accessoires/recouvrement-seuil-9g13-evo2008.png) |
| 1023 | F00-62- 1023 | F00-75- 9A23 Embout | 13 de large ; 34,5 de haut | ![Profilé 1023 et embout 9A23](/assets/profiles/systeme70/accessoires/profil-1023-embout-9a23-evo2008.png) |
| 9646 | F40-75- 9646 | Pont de vitrage ouvrant semi à fleur | - | ![Pont de vitrage 9646](/assets/profiles/systeme70/accessoires/pont-vitrage-9646-evo2008.png) |
| 9326 | F90-75- 9326 | Pont de vitrage ouvrant à déport | - | ![Pont de vitrage 9326](/assets/profiles/systeme70/accessoires/pont-vitrage-9326-evo2008.png) |
| 9669 | F90-67- 9669 | Equerre pour renforcement d'angle | - | ![Équerre 9669](/assets/profiles/systeme70/accessoires/equerre-9669-evo2008.png) |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 46)

Le 9G13 est dessiné clippé en tête de l'aile d'un profilé coupé, sans référence, avec deux
cotes verticales portées à gauche, 27,5 mm et 22,6 mm, et sa largeur de 18,3 mm en haut. Le 1023
est dessiné une seconde fois engagé entre deux profilés coupés dos à dos, avec une pièce conique
sous l'assemblage ; ni les deux profilés ni la pièce conique ne portent de référence. Le 9326 est
dessiné comme une grille posée en fond de feuillure, le long des deux côtés de l'angle ; le 9646
comme une pièce mince au fond de la feuillure ; l'équerre 9669 est vissée dans l'angle intérieur
du cadre [3 p. 46].

Le manuel de mise en œuvre de 2023 désigne le 9G13 « protège feuillure extérieure » et le 9326
« Support de cales de vitrage » (voir
[Renforts et accessoires par profilé du système 70](/profiles/systeme-70-accessoires-par-profile.md)) :
entrée **CTR-75** du registre [Contradictions entre sources](/anomalies/contradictions-entre-sources.md)
pour le 9G13 ; la légende de 2008 du 9326 est versée à **VER-87**.

## Sets d'assemblage des meneaux (p. 47)

Le tableau du haut de la p. 47 donne, pour chaque meneau dessiné en première colonne, la pièce
de trois assemblages : le [set d'assemblage mécanique](/reference/glossaire.md) en T (le meneau
aboutit sur un profilé), le set d'assemblage mécanique en croix (deux traverses se croisent) et
le set d'équerre de feuillure ; sous chaque pièce, son [gabarit de perçage](/reference/glossaire.md).
Une ligne par meneau et par assemblage ; « — » est écrit tel quel sur la planche quand le meneau
n'a pas cet assemblage [3 p. 47].

| Meneau | Assemblage | Pièce | Gabarit de perçage |
| --- | --- | --- | --- |
| 6127 (F91-15- 6127) | set d'assemblage mécanique en T | — | - |
| 6127 (F91-15- 6127) | set d'assemblage mécanique en croix | — | - |
| 6127 (F91-15- 6127) | set d'équerre de feuillure | F90-75- 9718.3 | F90-70- 9918 |
| 2425 (F95-15- 2425) | set d'assemblage mécanique en T | F90-84- 9316 | F90-70- 9918 |
| 2425 (F95-15- 2425) | set d'assemblage mécanique en croix | F90-84- 9317 | F90-70- 9918 |
| 2425 (F95-15- 2425) | set d'équerre de feuillure | F90-75- 9719 | F90-70- 9918 |
| 2427 (F95-15- 2427) | set d'assemblage mécanique en T | F90-84- 9B51 | F95-70- 9B44 |
| 2427 (F95-15- 2427) | set d'assemblage mécanique en croix | F90-75- 9B52 | F95-70- 9B44 |
| 2427 (F95-15- 2427) | set d'équerre de feuillure | F90-75- 9B89 | F90-70- 9918 |
| 2469 (F95-15- 2469) | set d'assemblage mécanique en T | F90-84- 9C69 | F90-70- 9918 |
| 2469 (F95-15- 2469) | set d'assemblage mécanique en croix | — | - |
| 2469 (F95-15- 2469) | set d'équerre de feuillure | F90-75- 9B56 | F90-70- 9918 |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 47)

L'en-tête du tableau dessine chaque type de set en perspective éclatée :

![Set d'assemblage mécanique en T, 2008](/assets/profiles/systeme70/assemblages/set-assemblage-t-evo2008.png)

Le set en T : une pièce d'ancrage à logement carré, une vis à tête cylindrique avec sa rondelle,
une goupille longue et deux bouchons.

![Set d'assemblage mécanique en croix, 2008](/assets/profiles/systeme70/assemblages/set-assemblage-croix-evo2008.png)

Le set en croix : deux pièces d'ancrage, une vis plus longue, une goupille et des bouchons.

![Set d'équerre de feuillure, 2008](/assets/profiles/systeme70/assemblages/set-equerre-feuillure-evo2008.png)

Le set d'équerre de feuillure : un patin et deux équerres percées. Les coupes des quatre meneaux
sont sur [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md),
*Meneaux et traverses des plans e.VOLUTION de 2008*.

Ce tableau ne donne pas les mêmes pièces que la liste écrite sous chaque meneau aux p. 22 et 23
du même classeur : pour le 6127, l'équerre de feuillure y est écrite 9714 (9718.3 à la p. 47) ;
pour le 2427, l'assemblage en T y est écrit 9B54 (9B51 à la p. 47) et le seul gabarit écrit est le
9918 (9B44 pour le T et la croix à la p. 47) — entrée **INC-213** du registre
[Incohérences internes](/anomalies/incoherences-internes.md). Le poster de 2025 donne comme les
p. 47 le 9B51 au 2427 et le 9316 au 2425 (*Sets d'assemblage mécanique des meneaux et traverses*
ci-dessus) ; il donne le patin 9B56 à la traverse d'ouvrant 6126, là où la p. 47 le donne au
meneau 2469 [3 p. 22-23, 47].

## Sets d'assemblage des dormants et du meneau 2425 (p. 47)

Le tableau « Set d'assemblage dormant » associe à chaque dormant son set d'assemblage, une ligne
par dormant ; l'encadré « Set d'assemblage meneaux » donne le set du meneau 2425 [3 p. 47].

| Dormant | Set d'assemblage |
| --- | --- |
| 2502 (F95-01- 2502) | F95-94- 9C79 |
| 6101 (F91-01- 6101) | F91-84- 9F40 |
| 6104 (F91-01- 6104) | F91-84- 9F41 |
| 6108 (F91-01- 6108) | F91-84- 9F41 |
| 6109 (F91-01- 6109) | F91-84- 9F41 |
| 6110 (F91-01- 6110) | F91-84- 9F41 |
| 6111 (F91-01- 6111) | F91-84- 9F41 |
| 6102 (F91-01- 6102) | F91-84- 9F43 |
| 6105 (F91-01- 6105) | F91-84- 9F43 |
| 6106 (F91-01- 6106) | F91-84- 9F43 |
| 6107 (F91-01- 6107) | F91-84- 9F43 |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 47)

![Set d'assemblage dormant, 2008 : tableau et perspective](/assets/profiles/systeme70/assemblages/sets-assemblage-dormant-evo2008.png)

À gauche du tableau, trois perspectives de haut en bas : le montant du dormant, la pièce
d'assemblage à deux flasques percées et un profilé plat strié dessiné en pied. Le dormant 6100
n'est pas dans ce tableau.

![Set d'assemblage meneaux 9C84 pour le meneau 2425](/assets/profiles/systeme70/assemblages/set-assemblage-meneau-9c84-evo2008.png)

L'encadré « Set d'assemblage meneaux » dessine de la même façon le meneau « F95-15- 2425 », le
« Set d'assemblage F95-94- 9C84 » et le profilé plat strié en pied. Aucun de ces deux tableaux ne
nomme le profilé strié ni ne dit sur quoi le set assemble le dormant ou le meneau. Les sets 9C79,
9F40, 9F41, 9F43 et 9C84 ne figurent dans aucune des tables du poster et du DTD reprises
ci-dessus, qui donnent pour les mêmes dormants les sets dormant / seuil 9F71, 9F72 et J077
(*Set d'assemblage dormant / seuil*) [3 p. 47].

# Citations

[1] [DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION](raw/dtd-6-16-2335-v5-e-volution.pdf)

[2] [Poster Kömmerling Gamme 70, profilés principaux, mars 2025](raw/poster-kommerling-70-principaux-2025-03.pdf), p. 1

[3] [Système e.VOLUTION, plan des profilés, système F 91, édition août 2008](raw/profine-plans-profiles-e-volution-2008-08.pdf), p. 12, 14, 22-23 et 46-47

# Voir aussi

- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Posters Gamme 70 KÖMMERLING](/sources/posters-kommerling-70.md)
- [Joints et garnitures des systèmes profine](/profiles/joints-et-garnitures-profine.md)
- [DTD n° DBV-24-6/16-2335_V5](/certifications/dtd-6-16-2335.md)
- [Fabrication et assemblage du système 70](/procedures/fabrication-systeme-70.md)
- [Mise en œuvre du seuil aluminium du système 70](/procedures/mise-en-oeuvre-seuil-systeme-70.md)
- [Accouplement d'éléments du système 70](/procedures/accouplement-elements-systeme-70.md)
- [profine](/fournisseurs/profine.md)
- [KÖMMERLING](/fournisseurs/kommerling.md)
