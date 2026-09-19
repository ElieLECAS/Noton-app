---
type: Référence
title: Classification A*E*V* à préconiser par site
description: La classe A*E*V* minimale à préconiser pour une fenêtre ou une porte extérieure, en France métropolitaine et dans les quatre départements d'outre-mer, selon la région de vent, la catégorie de terrain et la hauteur du bâtiment, avec la résistance mécanique et la réduction pour ouvrage protégé.
tags: [aev, classification, vent, region-climatique, categorie-de-terrain, dom, resistance-mecanique, nf-en-12210, nf-en-12207, nf-en-12208, reference]
status: stable
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
source_pages:
  - resource: raw/profine-directives-generales-2023-01.pdf
    pages: 63-74
generated:
  by: process:claude-code
  at: 2026-09-19T17:30:00Z
---

# Trois paramètres de site déterminent la classe

La classe A\*E\*V\* qu'il faut préconiser pour une fenêtre ne dépend pas du produit : elle dépend
du site. Trois paramètres suffisent [1 registre 1.3.3 p. 1 à 4] :

1. **La région climatique** — 4 en France métropolitaine, définies par la vitesse de référence
   du vent, plus 4 départements d'outre-mer.
2. **La catégorie de terrain d'environnement** — 5 catégories, de la mer à la ville dense.
3. **La hauteur du bâtiment H**, mesurée au faîtage ou à l'accrotère — c'est la hauteur du
   bâtiment entier qui compte, pas la hauteur d'implantation de la fenêtre.

# Régions climatiques de France métropolitaine

| Région | Vitesse de référence du vent vb,0 (m/s) |
| --- | --- |
| 1 | 22 |
| 2 | 24 |
| 3 | 26 |
| 4 | 28 |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.3, annexe A, p. 15)

Ces régions sont celles de l'annexe nationale à la NF EN 1991-1-4, qui remplace les règles NV 65
en période transitoire. Pour trouver la région d'un département donné, voir
[Régions climatiques par département](/reference/regions-climatiques-par-departement.md).

# Catégories de terrain

| Catégorie | Définition |
| --- | --- |
| 0 | Mer ou zone côtière exposée aux vents de mer, lacs et plans d'eau parcourus sur au moins 5 km |
| II | Rase campagne, obstacles isolés séparés de plus de 40 fois leur hauteur |
| IIIa | Campagne avec haies, vignobles, bocages, habitat dispersé |
| IIIb | Zone urbanisée ou industrielle, bocage dense, vergers |
| IV | Zone urbaine (au moins 15 % de bâtiments de plus de 15 m), forêts |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.3, p. 2)

**La catégorie 0 n'est retenue qu'à moins de 20 fois la hauteur du bâtiment du rivage.** Sur le
littoral méditerranéen des régions 2 et 3 (hors Corse), où les vents forts viennent le plus
souvent de l'intérieur des terres, la catégorie retenue est II, pas 0 [1 registre 1.3.3 p. 2].

# Classification A\*E\*V\* à préconiser, France métropolitaine

Tableau récapitulatif : la classe minimale par région, catégorie de terrain et hauteur de
bâtiment. C'est la synthèse des tableaux de pression de vent (NF EN 12211), de perméabilité à
l'air (NF EN 12207) et d'étanchéité à l'eau (NF EN 12208) du même registre.

| Région | Terrain | H ≤ 9 m | 9 < H ≤ 18 m | 18 < H ≤ 28 m | 28 < H ≤ 50 m | 50 < H ≤ 100 m |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | IV | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 |
| 1 | IIIb | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 |
| 1 | IIIa | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 |
| 1 | II | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A2 | A\*3 E\*5 V\*A3 |
| 1 | 0 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 |
| 2 | IV | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 |
| 2 | IIIb | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 |
| 2 | IIIa | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*5 V\*A3 |
| 2 | II | A\*3 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 |
| 2 | 0 | A\*3 E\*5 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*6 V\*A4 |
| 3 | IV | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 |
| 3 | IIIb | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 |
| 3 | IIIa | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 |
| 3 | II | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 |
| 3 | 0 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*7 V\*A4 |
| 4 | IV | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A2 | A\*3 E\*6 V\*A3 |
| 4 | IIIb | A\*2 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 |
| 4 | IIIa | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 |
| 4 | II | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A4 |
| 4 | 0 | A\*3 E\*6 V\*A3 | A\*3 E\*6 V\*A4 | A\*3 E\*7 V\*A4 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A5 |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.3, p. 12 et 13)

**Ces classes sont des minimums**, valables pour des fenêtres et pour des portes de logement sur
coursive hors locaux non chauffés ; une classe supérieure reste toujours admissible. Pour une
porte extérieure de maison individuelle, **le niveau d'étanchéité à l'eau (E) est abaissé d'une
classe** par rapport à ce tableau, sans pouvoir descendre sous E\*0 [1 registre 1.3.3 p. 9 et 11].
**Une fenêtre de toit ne descend jamais sous E\*8A.**

**Un ouvrage protégé de la pluie admet une classe d'étanchéité réduite**, selon le facteur de
protection L/H de son auvent, en partant de la classe E\*n lue dans le tableau ci-dessus [1
registre 1.3.3 p. 12] :

| Facteur de protection L/H ≥ | Classe d'étanchéité réduite |
| --- | --- |
| 0,40 | E\*n-1 |
| 0,60 | E\*n-2 |
| 0,70 | E\*n-3 |
| 0,80 | E\*n-4 |
| 0,90 | E\*n-5 |
| 1,00 | E\*n-6 |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.3, p. 12)

Sans pouvoir descendre sous **E\*1 pour une fenêtre** ni **E\*0 pour une porte extérieure**, quel
que soit le nombre de crans que le calcul retire [1 registre 1.3.3 p. 12].

# Classification A\*E\*V\* à préconiser, départements d'outre-mer

Même tableau récapitulatif, quatre départements. Les classes y sont nettement plus sévères qu'en
métropole, et une hauteur de bâtiment supérieure à 100 m sort du domaine d'application du
document dans les deux cas [1 registre 1.3.3 p. 13].

| Département | Terrain | H ≤ 9 m | 9 < H ≤ 18 m | 18 < H ≤ 28 m | 28 < H ≤ 50 m | 50 < H ≤ 100 m |
| --- | --- | --- | --- | --- | --- | --- |
| Guadeloupe | IV | A\*3 E\*5 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A5 |
| Guadeloupe | IIIb | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A4 | A\*3 E\*8 V\*A5 |
| Guadeloupe | IIIa | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A4 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*AE2250 |
| Guadeloupe | II | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*AE2200 | A\*3 E\*9 V\*AE2550 |
| Guadeloupe | 0 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*AE2100 | A\*3 E\*8 V\*AE2250 | A\*3 E\*9 V\*AE2500 | A\*3 E\*9 V\*AE2800 |
| Guyane | IV | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 |
| Guyane | IIIb | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 |
| Guyane | IIIa | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 |
| Guyane | II | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 |
| Guyane | 0 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 | A\*2 E\*4 V\*A2 |
| Martinique | IV | A\*3 E\*4 V\*A2 | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 |
| Martinique | IIIb | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A4 |
| Martinique | IIIa | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A5 |
| Martinique | II | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A4 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*A5 |
| Martinique | 0 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*AE2200 |
| Réunion | IV | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 |
| Réunion | IIIb | A\*3 E\*4 V\*A2 | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A3 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A5 |
| Réunion | IIIa | A\*3 E\*5 V\*A3 | A\*3 E\*6 V\*A4 | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*A5 |
| Réunion | II | A\*3 E\*7 V\*A4 | A\*3 E\*8 V\*A4 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*A5 | A\*3 E\*9 V\*AE2300 |
| Réunion | 0 | A\*3 E\*8 V\*A4 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*A5 | A\*3 E\*8 V\*AE2200 | A\*3 E\*9 V\*AE2500 |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.3, p. 12 et 13)

**La Guyane ne se distingue pas de la métropole la plus clémente** : ses cinq lignes sont
identiques à A\*2 E\*4 V\*A2, quelle que soit la catégorie de terrain ou la hauteur. **La
Guadeloupe est le département le plus exigeant du corpus**, jusqu'à A\*3 E\*9 V\*AE2800 en bord de
mer au-delà de 50 m.

# Résistance mécanique

Exigences de résistance mécanique, indépendantes du site, relevées sur le registre (p. 14).
Elles se réfèrent à des normes d'essai et de classement distinctes de celles du vent, de la
perméabilité et de l'étanchéité.

**Fenêtres, hors portes extérieures :**

| Caractéristique | Norme d'essai | Norme de classement | Classe minimale |
| --- | --- | --- | --- |
| Effort de manœuvre, avant et après essais | NF EN 12046-1 | NF EN 13115 | 1 |
| Contreventement | NF EN 14608 | NF EN 13115 | 2 |
| Torsion statique | NF EN 14609 | NF EN 13115 | 2 |
| Endurance à l'ouverture-fermeture, mécanisme à un seul mouvement (à la française, soufflet, coulissant à translation) | NF EN 1191 | NF EN 12400 | 2 |
| Endurance à l'ouverture-fermeture, mécanisme à plusieurs mouvements (oscillo-battant, chaque mouvement testé sur le même corps d'épreuve) | NF EN 1191 | NF EN 12400 | 1 |

**Portes extérieures :**

| Caractéristique | Norme d'essai | Norme de classement | Classe minimale |
| --- | --- | --- | --- |
| Effort de manœuvre, avant et après essais | NF EN 12406-2 | NF EN 12217 | 1 |
| Contreventement | NF EN 974 | NF EN 1192 | 1 |
| Torsion statique | NF EN 948 | NF EN 1192 | 1 |
| Choc de corps dur | NF EN 950 | NF EN 1192 | 1 |
| Endurance à l'ouverture-fermeture | NF EN 1191 | NF EN 12400 | 3 |

**Un maître d'ouvrage peut exiger un classement plus sévère selon l'usage prévu** — ces classes
sont des minimums réglementaires, pas des cibles [1 registre 1.3.3 p. 14].

# Ce que la source ne donne pas ici

- **Les tableaux intermédiaires** (pressions P1 et P3 en Pa des tableaux 1 et 2, classes de
  résistance au vent seules du tableau 3, de perméabilité seules du tableau 4, d'étanchéité
  seules des tableaux 5 et 6) ne sont pas repris : les tableaux récapitulatifs ci-dessus, tirés du
  tableau 8 du registre, les ont déjà combinés
- **La carte de la Figure A.1** (fond de carte, p. 15) n'est pas reproduite : sa donnée est le
  tableau [Régions climatiques par département](/reference/regions-climatiques-par-departement.md),
  déjà transcrit sous forme de tableau
- **Les illustrations photographiques des catégories de terrain** (Figures A.2 et A.3, p. 19-20) :
  images d'exemple, sans donnée chiffrée propre au-delà de la définition déjà donnée plus haut

# Citations

[1] Directives générales profine, version janvier 2023 —
`raw/profine-directives-generales-2023-01.pdf`, registre 1.3.3, p. 1 à 14

# Voir aussi

- [Régions climatiques par département](/reference/regions-climatiques-par-departement.md)
- [Classification de la résistance au vent](/reference/classification-resistance-au-vent.md)
- [Labels et certifications](/certifications/labels-et-certifications.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [profine](/fournisseurs/profine.md)
