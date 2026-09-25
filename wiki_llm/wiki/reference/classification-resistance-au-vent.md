---
type: Référence
title: Classification de la résistance au vent, EN 12211
description: Les classes d'essai de pression de vent et de flèche relative normale selon l'EN 12211, et leur combinaison en une classification globale de résistance au vent (A1 à C5, ou Exxxx).
tags: [norme, en-12211, vent, fleche, classification, statique, reference]
usage: chiffrage
status: stable
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
generated:
  by: process:claude-code
  at: 2026-09-19T15:00:00Z
---

# Trois pressions d'essai pour une classe

L'EN 12211 fixe la méthode d'essai de résistance au vent pour une fenêtre ou une porte
complètement assemblée, quel que soit son matériau. Un essai porte sur trois pressions liées
entre elles : **P2 = 0,5 × P1** et **P3 = 1,5 × P1** [1 registre 1.3.3 p. 23].

# Classe de pression de vent

| Classe | P1 (Pa) | P2 (Pa) | P3 (Pa) |
| --- | --- | --- | --- |
| 0 | pas d'essai | pas d'essai | pas d'essai |
| 1 | 400 | 200 | 600 |
| 2 | 800 | 400 | 1 200 |
| 3 | 1 200 | 600 | 1 800 |
| 4 | 1 600 | 800 | 2 400 |
| 5 | 2 000 | 1 000 | 3 000 |
| Exxxx | xxxx | - | - |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.3, p. 23)

**Au-delà de la classe 5, la classe s'écrit Exxxx**, où `xxxx` est la pression d'essai P1 réelle
en Pa — par exemple E2350. La pression P1 de la classe testée est répétée 50 fois avant l'essai
de classification.

# Classe de flèche relative normale

Mesurée sur l'élément de dormant le plus déformé du corps d'épreuve, à la pression P1.

| Classe | Flèche relative normale |
| --- | --- |
| A | < 1/150 |
| B | < 1/200 |
| C | < 1/300 |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.3, p. 24)

**Sous P1 et P2**, aucun défaut visible n'est toléré à 1 m de distance sous lumière naturelle, et
l'accroissement de perméabilité à l'air ne doit pas dépasser 20 % de la perméabilité admissible
pour la classe revendiquée (EN 12207). **Sous P3**, un gauchissement ou une fissuration sont
admis à condition qu'aucune pièce ne se détache et que le corps d'épreuve reste fermé ; un
vitrage cassé peut être remplacé pour reprendre l'essai une fois [1 registre 1.3.3 p. 24].

# Classification globale de résistance au vent

Combine la classe de pression de vent (chiffre) et la classe de flèche (lettre).

| Classe de pression de vent | Flèche A | Flèche B | Flèche C |
| --- | --- | --- | --- |
| 1 | A1 | B1 | C1 |
| 2 | A2 | B2 | C2 |
| 3 | A3 | B3 | C3 |
| 4 | A4 | B4 | C4 |
| 5 | A5 | B5 | C5 |
| Exxxx | AExxxx | BExxxx | CExxxx |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.3, p. 24)

**Cette classification est indépendante de la classification A\*E\*V\* des labels commerciaux** —
voir [Labels et certifications](/certifications/labels-et-certifications.md) — bien que les deux
mesurent la tenue au vent. L'EN 12211 date la déformation admissible du dormant lui-même ;
l'AEV commercial reprend une classe de vent proche mais dans un cadre de certification différent
(NF, CSTBat). Aucune source du wiki ne fait explicitement la conversion d'une classification à
l'autre — à vérifier avant de citer l'une pour l'autre.

# Ce que la source ne donne pas

**Le tableau A.1 des régions climatiques par département, et le tableau A.2 de leur découpage
par canton** (registre 1.3.3, p. 16 à 18) ne sont pas transcrits ici : ce sont des tableaux
d'image sans couche texte, portant l'intégralité des départements français. Utiles pour
déterminer la région de vent d'un chantier, ils sont génériques au bâtiment et non spécifiques à
un produit PROFERM — à transcrire si un besoin précis de dimensionnement par département se
présente.

# Citations

[1] Directives générales profine, version janvier 2023 —
`raw/profine-directives-generales-2023-01.pdf`, registre 1.3.3, p. 23 et 24

# Voir aussi

- [Labels et certifications](/certifications/labels-et-certifications.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [profine](/fournisseurs/profine.md)
