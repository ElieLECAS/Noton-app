---
type: Référence
title: Classification de la résistance au vent, essai EN 12211
description: Les classes de pression de vent (P1, P2, P3) et de flèche relative normale d'une fenêtre ou d'une porte essayée selon l'EN 12211, les exigences sous P1, P2 et P3, et leur combinaison en une classification globale de résistance au vent (A1 à C5, ou Exxxx).
tags: [norme, en-12211, en-12210, vent, fleche, classification, statique, reference]
fournisseur: KÖMMERLING
usage: chiffrage
status: stable
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
source_pages:
  - resource: raw/profine-directives-generales-2023-01.pdf
    pages: 85-86
generated:
  by: process:claude-code
  at: 2026-09-28T19:00:00Z
---

# Domaine d'application

La classification de la résistance au vent range une fenêtre ou une porte selon deux résultats
d'essai : la pression de vent qu'elle a tenue (un chiffre, de 1 à 5, ou Exxxx) et la flèche
relative normale de son dormant (une lettre, A, B ou C). La norme de classification — la
NF EN 12210, nommée au registre 1.3.3, PDF p. 65 — définit la classification des résultats d'essai
pour les fenêtres et les portes complètement assemblées, quels que soient leurs matériaux, après
essai réalisé selon l'EN 12211 [1 p. 85].

Les classes à préconiser selon le site (région, terrain, hauteur) sont sur
[Classification A\*E\*V\* par site](/reference/classification-aev-par-site.md).

# Classes de pression de vent

L'EN 12211 décrit une méthode d'essai pour déterminer les limites P1, P2 et P3 pour le corps
d'épreuve (la fenêtre essayée). Ces limites sont exprimées en pascals (Pa). Les relations entre les
limites sont **P2 = 0,5 P1** et **P3 = 1,5 P1**. La classification doit être établie selon les
résultats des essais de résistance au vent aux pressions d'essai positives et négatives ; les
pressions d'essai sont données dans le tableau 1 [1 p. 85].

**Note** : la présente classification peut être employée avec d'autres normes ou règles de mise en
œuvre appropriées, et il est donc possible de s'en servir pour corrélation avec des prescriptions
réelles d'exposition [1 p. 85].

Tableau 1, classification des pressions de vent : une ligne par classe, pressions d'essai en Pa.

| Classe | P1 (Pa) | P2 (Pa), pression répétée 50 fois | P3 (Pa) |
| --- | --- | --- | --- |
| 0 | pas d'essai | pas d'essai | pas d'essai |
| 1 | 400 | 200 | 600 |
| 2 | 800 | 400 | 1 200 |
| 3 | 1 200 | 600 | 1 800 |
| 4 | 1 600 | 800 | 2 400 |
| 5 | 2 000 | 1 000 | 3 000 |
| E xxxx | xxxx | - | - |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 85)

Note 1) du tableau : la pression P2 est répétée 50 fois. Note 2) : un corps d'épreuve essayé avec
une pression de vent supérieure à la classe 5 est classé Exxxx, où xxxx est la pression d'essai
réelle P1 (par exemple : 2 350, etc.) ; le tableau ne donne alors ni P2 ni P3 [1 p. 85].

# Classes de flèche relative normale

La flèche relative normale de l'élément de dormant le plus déformé du corps d'épreuve, mesurée à la
pression d'essai P1, doit être classée comme indiqué dans le tableau 2 (la flèche relative est la
déformation rapportée à la portée de l'élément) [1 p. 86].

| Classe | Flèche relative normale |
| --- | --- |
| A | < 1/150 |
| B | < 1/200 |
| C | < 1/300 |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 86)

# Exigences

Les exigences suivantes doivent aussi être respectées pour que le produit puisse être classé
[1 p. 86].

**Sous pression de vent P1 et P2** : aucun défaut visible lors d'un examen avec une vision normale
ou corrigée à une distance de 1 m sous une lumière naturelle. Le corps d'épreuve doit rester en bon
état de fonctionnement, et l'accroissement maximal de la perméabilité à l'air résultant des essais
de résistance au vent à P1 et P2 ne doit pas dépasser 20 % de la perméabilité à l'air maximale
admissible pour la classe de perméabilité à l'air revendiquée, spécifiée dans la norme EN 12207
[1 p. 86].

**Note** : la classification revendiquée peut être déterminée à l'aide de l'essai de résistance au
vent. Si le fabricant souhaite se prévaloir d'une classification inférieure, il peut le faire
ainsi [1 p. 86].

**Sous pression de vent P3** : des défauts tels qu'un gauchissement et/ou cintrage d'un élément de
quincaillerie ou une fissuration d'éléments du dormant doivent être admis, à condition qu'aucune
pièce ne se détache et que le corps d'épreuve reste fermé. Toutefois, si le verre casse, il est
admis de le remplacer et de recommencer l'essai une fois [1 p. 86].

# Classification globale de résistance au vent

Les forces de vent et la flèche relative normale doivent être combinées dans une classification
globale, comme indiqué dans le tableau 3 : une ligne par classe de pression de vent, une colonne par
classe de flèche relative normale [1 p. 86].

| Classe de pression de vent | Flèche relative normale A | Flèche relative normale B | Flèche relative normale C |
| --- | --- | --- | --- |
| 1 | A1 | B1 | C1 |
| 2 | A2 | B2 | C2 |
| 3 | A3 | B3 | C3 |
| 4 | A4 | B4 | C4 |
| 5 | A5 | B5 | C5 |
| Exxxx | AExxxx | BExxxx | CExxxx |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 86)

**Note** : dans la classification de résistance au vent, le chiffre concerne la classe de pression
de vent — voir tableau 1 — et la lettre correspond à la flèche relative normale — voir tableau 2
[1 p. 86].

# Citations

[1] [Directives générales profine, version janvier 2023](raw/profine-directives-generales-2023-01.pdf),
registre 1.3.3, p. 23 et 24 imprimées (PDF p. 85 et 86, version janvier 2016)

# Voir aussi

- [Classification A\*E\*V\* par site](/reference/classification-aev-par-site.md)
- [Labels et certifications](/certifications/labels-et-certifications.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [profine](/fournisseurs/profine.md)
