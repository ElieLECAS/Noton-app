---
type: Profilé
title: Assemblages du système 70
description: Les méthodes d'assemblage autorisées entre dormants, ouvrants et traverses du système 70 Plateforme, et les sets d'assemblage entre chaque dormant et chaque seuil aluminium.
tags: [systeme-70, e-volution, profine, assemblage, soudure, traverse, meneau, seuil, dormant, ouvrant]
status: stable
sources:
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    id: dtd-6-16-2335-v5
    title: DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION
    last_modified: 2025-04-15
generated:
  by: process:claude-code
  at: 2026-09-19T09:00:00Z
---

# Trois méthodes d'assemblage, et elles ne sont pas interchangeables

Dans le **système 70 Plateforme** de [profine](/fournisseurs/profine.md), l'assemblage d'une
traverse ou d'un meneau sur un dormant ou sur un ouvrant se fait selon l'une de trois méthodes,
et **le couple profilé / traverse détermine lesquelles sont admises**.

| Code | Méthode |
| --- | --- |
| M | assemblage mécanique |
| S | soudure en V |
| SP | soudure à plat |
| SP\* | soudure à plat **avec équerres** |

Les équerres du code SP\* ne concernent que les dormants. Sur les ouvrants, la soudure à plat se
fait sans équerre et le code s'écrit SP.

# Assemblage dormant / traverse

Méthodes admises entre chaque dormant du système 70 et chacune des quatre traverses. Une cellule
énonce toutes les méthodes possibles, séparées par `/`.

| Dormant | Famille | Traverse 6127 | Traverse 2427 | Traverse 2425 | Traverse 6157 |
| --- | --- | --- | --- | --- | --- |
| 6100 | standard | M/S/SP\* | M | M | M/S |
| 6101 | standard | M/S/SP\* | M | M | M/S |
| 2501 | standard | M/SP\* | M | M | M |
| 2502 | standard | M/SP\* | M | M | M |
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

**Les dormants 2501 et 2502 sont les seuls à exclure la soudure en V** sur la traverse 6127, et
les seuls à n'admettre que le mécanique sur la traverse 6157. Ce sont aussi les deux dormants de
coulissant — voir [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md).

# Assemblage ouvrant / traverse

Méthodes admises entre chaque ouvrant du système 70 et chacune des cinq traverses.

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

**Les ouvrants 6121 à 6124 et 6150 à 6153 ne se montent pas sur la traverse 6126** : le tableau
porte un tiret, qui est une exclusion et non une donnée manquante. La soudure en V n'est admise
que sur la traverse 6127, et seulement pour les ouvrants 6112C, 6113, 6115, 6116 et 2416.

# Set d'assemblage dormant / seuil

Pièce d'assemblage à employer entre chaque dormant et chacun des trois seuils aluminium. Un `ou`
dans une cellule signale deux pièces également admises.

| Dormant | Famille | Seuil 9F67, 20 mm | Seuil 9F68, 36 mm | Seuil Z043, 20 mm |
| --- | --- | --- | --- | --- |
| 6100 | standard | 9F57 ou 9F72 | 9F61 ou 9F72 | 9F72 |
| 6101 | standard | 9F65 ou 9F71 | 9F66 ou 9F71 | 9F71 |
| 2501 | standard | 9F65 | 9F66 | - |
| 2502 | standard | 9F65 ou J077 | 9F66 ou J077 | J077 + M002 |
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

**Le dormant 2501 ne se monte pas sur le seuil Z043.** Le 2502 y demande la pièce J077 complétée
de l'embout M002, seul cas de la table où deux pièces sont cumulées.

Le seuil aluminium est exclu sur un **oscillo-coulissant** [1 p. 4].

# Pose du seuil aluminium

Pour les assemblages 9F57 à 9F66, J087 et J088 : un mastic polyuréthane est déposé à l'arrière de
la pièce d'assemblage, puis le seuil aluminium est vissé sur les patins d'étanchéité des pièces
d'assemblage à l'aide de **2 vis Ø 4 × 50** [1 p. 4].

Sur une partie fixe équipée du seuil 9F67 ou Z043, deux montages sont admis :

- un profilé de dormant monté sur le seuil pour permettre la prise en feuillure du vitrage ; les
  fonds de feuillure des montants sont alors étanchés au **mastic élastomère mono-composant** en
  partie basse, et la traverse PVC montée à l'aide d'**équerres 9714**
- ou la parclose extérieure **A271** montée sur le support de cale **A272**

(schéma: raw/dtd-6-16-2335-v5-e-volution.pdf, p. 30)

# Ce que la source ne donne pas

Le DTD ne donne ni la longueur de coupe des traverses, ni le couple de serrage des vis, ni le
nombre de points d'assemblage mécanique par traverse.

# Citations

[1] DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION —
`raw/dtd-6-16-2335-v5-e-volution.pdf`, p. 4, 12, 13 et 30

# Voir aussi

- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Joints et garnitures des systèmes profine](/profiles/joints-et-garnitures-profine.md)
- [DTD n° DBV-24-6/16-2335_V5](/sources/dtd-6-16-2335.md)
- [profine](/fournisseurs/profine.md)
