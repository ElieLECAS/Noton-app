---
type: Profilé
title: Meneaux PERFORM76
description: Les quatre meneaux PERFORM76 — 76372, 76373 de dormant, 76301, 76303 d'ouvrant — et les alignements de traverse de soubassement.
tags: [perform76, meneau, traverse, soubassement, profile]
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
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
source_pages:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    pages: 12-15, 18, 20, 21
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
---

# Les quatre meneaux

Un **meneau** est un profil qui divise une fenêtre en plusieurs parties : vertical, il sépare
deux vantaux ; horizontal, il fait office de traverse. Un **meneau de dormant** divise le cadre
fixe (par exemple entre un fixe et un ouvrant), un **meneau d'ouvrant** divise un ouvrant. La
gamme PERFORM76 compte deux meneaux de dormant et deux meneaux d'ouvrant. La distinction est
stricte : **un meneau d'ouvrant ne se monte jamais sur un dormant**, mention portée sur les deux
planches concernées.

Les quatre forment deux couples, appariés par leur clair intérieur :

| Clair intérieur (mm) | Meneau de dormant | Meneau d'ouvrant |
| --- | --- | --- |
| 42 | 76372, 98 mm | 76301, 84 mm |
| 68 | 76373, 124 mm | 76303, 110 mm |

# Cotes

Largeurs et décomposition des quatre meneaux PERFORM76, en mm. La décomposition se lit de gauche
à droite : aile, clair intérieur (l'espace libre au centre du meneau), aile.

| Meneau | Emplacement | Largeur (mm) | Décomposition (mm) | Coupe |
| --- | --- | --- | --- | ---: |
| 76372 | dormant | 98 | 28 / 42 / 28 | ![Meneau 76372](/assets/profiles/perform76/meneaux/meneau-76372.png) |
| 76373 | dormant | 124 | 28 / 68 / 28 | ![Meneau 76373](/assets/profiles/perform76/meneaux/meneau-76373.png) |
| 76301 | ouvrant | 84 | 21 / 42 / 21 | ![Meneau 76301](/assets/profiles/perform76/meneaux/meneau-76301.png) |
| 76303 | ouvrant | 110 | 21 / 68 / 21 | ![Meneau 76303](/assets/profiles/perform76/meneaux/meneau-76303.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

Les meneaux de dormant portent des ailes de 28 mm, ceux d'ouvrant des ailes de 21 mm : d'où les
14 mm d'écart de largeur à clair intérieur égal.

Le sommaire des profilés du manuel profine donne 110 mm au 76373, contre 124 mm sur sa propre
planche de détail et au cahier PERFORM76 — entrée **INC-09** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

# Cotes des élargissements du meneau 76372

Sur la planche des meneaux, chaque meneau de dormant est aussi dessiné avec un ouvrant fermé
contre lui, d'un côté ou des deux. Au-dessus de la coupe, la largeur hors tout et sa
décomposition : **34** est la part d'un ouvrant de la paire basse, **74** celle d'un ouvrant de
la paire haute — les mêmes cotes que les bandes « ouvrant 76281 » et « ouvrant 76272 » des
planches de dormant. Sous la coupe, la **vue**, la largeur visible une fois la fenêtre fermée.
Cotes en mm [1 p. 13] :

| Composition | Largeur hors tout (mm) | Vue (mm) |
| --- | --- | --- |
| 98 | 98 | 42 |
| 34 + 98 | 132 | 49 / 34, soit 83 |
| 74 + 98 | 172 | 89 / 34, soit 123 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

## Meneau 76372 avec un ouvrant de la paire basse, 132 mm

![Élargissement du meneau 76372 à 132 mm](/assets/profiles/perform76/meneaux/elargissement-76372-132.png)

L'ouvrant, à gauche, recouvre 34 mm ; le meneau de 98 mm est à droite, vitrage fixe contre lui.
Vue : 49 + 34 = 83 mm.

## Meneau 76372 avec un ouvrant de la paire haute, 172 mm

![Élargissement du meneau 76372 à 172 mm](/assets/profiles/perform76/meneaux/elargissement-76372-172.png)

L'ouvrant, à gauche, avec son renfort de section carrée, recouvre 74 mm. Vue : 89 + 34 = 123 mm.

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

# Cotes des élargissements du meneau 76373

Mêmes compositions autour du meneau de dormant de 124 mm, qui admet en plus un ouvrant de chaque
côté. Cotes en mm [1 p. 13] :

| Composition | Largeur hors tout (mm) | Vue (mm) |
| --- | --- | --- |
| 124 | 124 | 68 |
| 34 + 124 | 158 | 49 / 60, soit 109 |
| 34 + 124 + 34 | 192 | 49 / 52 / 49, soit 150 |
| 74 + 124 | 198 | 89 / 60, soit 149 |
| 74 + 124 + 74 | 272 | 89 / 52 / 89, soit 230 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

## Meneau 76373 avec un ouvrant de la paire basse, 158 mm

![Élargissement du meneau 76373 à 158 mm](/assets/profiles/perform76/meneaux/elargissement-76373-158.png)

Un ouvrant à gauche (34 mm), vitrage fixe à droite. Vue : 49 + 60 = 109 mm.

## Meneau 76373 entre deux ouvrants de la paire basse, 192 mm

![Élargissement du meneau 76373 à 192 mm](/assets/profiles/perform76/meneaux/elargissement-76373-192.png)

Un ouvrant de chaque côté (34 + 124 + 34). Vue : 49 + 52 + 49 = 150 mm.

## Meneau 76373 avec un ouvrant de la paire haute, 198 mm

![Élargissement du meneau 76373 à 198 mm](/assets/profiles/perform76/meneaux/elargissement-76373-198.png)

Un ouvrant à gauche (74 mm), vitrage fixe à droite. Vue : 89 + 60 = 149 mm.

## Meneau 76373 entre deux ouvrants de la paire haute, 272 mm

![Élargissement du meneau 76373 à 272 mm](/assets/profiles/perform76/meneaux/elargissement-76373-272.png)

Un ouvrant de chaque côté (74 + 124 + 74). Vue : 89 + 52 + 89 = 230 mm.

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

Le 76373 offre cinq compositions contre trois au 76372, dont deux symétriques à trois éléments.

# Compatibilités

| Meneau | Se monte sur | Ne se monte pas sur |
| --- | --- | --- |
| 76372 | les cinq dormants 76171, 76172, 76177, 76180, 76185 | les ouvrants |
| 76373 | les cinq dormants | les ouvrants |
| 76301 | les ouvrants uniquement | **tout dormant** |
| 76303 | les ouvrants uniquement | **tout dormant** |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13, 14, 15, 18, 20 et 21)

Les combinaisons dormant et meneau donnent les mêmes cotes sur les cinq dormants : le choix du
meneau est indépendant du dormant. Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

Les cotes de débit des deux meneaux de dormant sont dans
[Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md). Les capots aluminium des
meneaux et des traverses, la traverse 76300 et la traverse complémentaire 76299 sont dans
[Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md) ; les pièces
d'assemblage en T et en X dans
[Assemblages du système 76](/profiles/systeme-76-assemblages.md).

# Alignement de la traverse de soubassement

Une fenêtre PERFORM76 à soubassement porte, à mi-hauteur, une **traverse** : la barre
horizontale qui sépare le vitrage du haut du **soubassement** du bas. Quand l'ensemble associe une
partie fixe et un ouvrant (le vantail qui s'ouvre), la traverse du fixe et celle de l'ouvrant ne
sont pas faites du même profilé : le fixe reçoit un meneau de dormant, l'ouvrant un meneau
d'ouvrant, plus étroit. Deux principes permettent de les aligner [1 p. 12].

**L'alignement des vitrages sur le dessus des traverses se demande à la commande** [1 p. 12] :
c'est un choix de fabrication, pas un réglage de chantier.

## Alignement standard à l'axe de traverse, entre fixe et ouvrant

Le fixe reçoit le meneau de dormant **76372** (98 mm), l'ouvrant le meneau d'ouvrant **76301**
(84 mm). Les deux traverses sont centrées sur la même ligne, l'**axe des traverses** (trait
d'axe rouge sur le schéma) : leurs milieux sont alignés, pas leurs bords.

![Alignement standard à l'axe traverse entre fixe et ouvrant](/assets/profiles/perform76/pose/alignement-axe-traverse.png)

Le schéma se lit en deux parties. À gauche, l'élévation de l'ensemble à trois vantaux, vu de
face : le FIXE à gauche, deux ouvrants à droite (la croix dessinée sur un vantail indique qu'il
s'ouvre), le soubassement sous la ligne rouge. À droite, deux coupes verticales, faites en
tranchant la menuiserie de haut en bas : la **coupe sur fixe** (repère Détail 1) et la **coupe
sur ouvrant** (repère Détail 2). La partie hachurée, sous chaque traverse, est le soubassement.
Cotes en mm :

| Coupe verticale | Traverse haute (mm) | Traverse intermédiaire, largeur (mm) | Décomposition de la traverse intermédiaire (mm) | Traverse basse (mm) |
| --- | --- | --- | --- | --- |
| Sur fixe, meneau 76372 | 74, soit 46 + 28 | 98 | 28 / 42 / 28 | 74, soit 28 + 46 |
| Sur ouvrant, meneau 76301 | 74 + 34, soit 38 + 49 + 21 | 84 | 21 / 42 / 21 | 34 + 74, soit 21 + 49 + 38 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 12)

Les deux traverses intermédiaires ont le même clair central de **42 mm**, et c'est lui qui est
aligné. Le meneau d'ouvrant de 84 mm est donc en retrait de **7 mm** en haut et en bas par rapport
au meneau de dormant de 98 mm — les deux cotes rouges de 7 de la coupe sur ouvrant.

## Alignement total avec faux ouvrant

Pour aligner à la fois **les vitrages et les soubassements**, la partie fixe est réalisée en
**faux ouvrant** : elle est construite comme un ouvrant, avec les mêmes profilés, mais ne s'ouvre
pas. Fixe et ouvrant ayant alors exactement la même section, tout s'aligne : le dessus et le
dessous des traverses, les vitrages, les soubassements. **Un seul meneau est employé, le
76301**, sur toutes les parties.

![Alignement total avec faux ouvrant](/assets/profiles/perform76/pose/alignement-total-faux-ouvrant.png)

À gauche, l'élévation : le faux ouvrant à gauche, dessiné comme les deux ouvrants, avec la
bande rouge « alignement total des vitrages et soubassements ». À droite, une seule coupe
verticale, sur ouvrant (repère Détail 2), valable pour les trois vantaux ; les traits d'axe
rouges marquent les lignes alignées sur tout l'ensemble. Cotes en mm :

| Coupe verticale | Traverse haute (mm) | Traverse intermédiaire, largeur (mm) | Décomposition de la traverse intermédiaire (mm) | Traverse basse (mm) |
| --- | --- | --- | --- | --- |
| Sur ouvrant et faux ouvrant, meneau 76301 | 74 + 34, soit 38 + 49 + 21 | 84 | 21 / 42 / 21 | 34 + 74, soit 21 + 49 + 38 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 12)

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, pages du PDF 12 à 15, 18, 20 et 21

# Voir aussi

- [Assemblage mécanique du meneau et de la traverse du système 76 Advanced](/procedures/assemblage-meneau-traverse-systeme-76.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
