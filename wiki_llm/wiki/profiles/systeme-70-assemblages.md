---
type: Profilé
title: 'Assemblages du système 70 : méthodes dormant / ouvrant / traverse, seuils aluminium'
description: Méthodes d'assemblage admises du système 70 Plateforme entre chaque dormant ou ouvrant et les traverses 6126, 6127, 2425, 2427, 6157, pièce d'assemblage dormant / seuil, et seuils aluminium 9F67, 9F68, Z043, 9C42 avec leurs cotes.
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
  at: 2026-10-21T10:40:30Z
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

Autres parties de ce sujet :

- [Pièces d'assemblage du système 70 : embouts de seuil, sets, patins d'étanchéité, équerres](/profiles/systeme-70-assemblages-embouts-sets-patins-nomenclature-dtd.md)
- [Accessoires et sets d'assemblage du système 70 : meneaux et dormants, plans de 2008](/profiles/systeme-70-assemblages-accessoires-et-sets-plans-2008.md)
