---
type: Procédure
title: Montage au bâtiment, directives générales profine
description: Les tolérances de fixation à la maçonnerie, les largeurs de joint de raccordement et la dilatation admissible pour le montage d'une menuiserie profine, avec les entrées d'air autoréglables.
tags: [profine, montage, fixation, joint, tolerance, dilatation, entree-air, atelier]
fournisseur: KÖMMERLING
usage: pose
status: stable
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
generated:
  by: process:claude-code
  at: 2026-09-19T17:45:00Z
---

# Ce que fait cette procédure

Elle donne les tolérances et les cotes de fixation d'une menuiserie profine au corps du
bâtiment (registre 1.3.7) et les principes d'intégration d'une entrée d'air autoréglable
(registre 1.3.8). Ce sont des règles génériques, indépendantes du système de profilés.

# Conditions et interdictions

- **Jamais de mortier pour remplir le joint de raccordement** : il se détache par les mouvements
  de la fenêtre et rend le raccordement fixe, ce qui empêche l'étanchéité durable [1 registre
  1.3.7 p. 2].
- **Jamais d'aluminium fixé de manière rigide dans le crépi ou le corps de construction** : sa
  dilatation propre l'exige mobile.
- **Jamais de plomb, cuivre ou alliage cuivreux (laiton) au contact de l'aluminium**, même à
  l'état fluide. L'acier galvanisé, l'inox et le zinc s'utilisent sans problème.
- **Toujours poser des cales résistant à la pression entre dormant et maçonnerie, à chaque point
  de verrouillage**, pour une menuiserie anti-effraction.

# Cotes

## Fixation à la maçonnerie

| Paramètre | Valeur |
| --- | --- |
| Écartement maximal entre points de fixation | 700 mm |
| Distance d'un point de fixation au coin intérieur | 100 à 150 mm |
| Distance au coin intérieur, meneaux et traverses, profilé blanc | environ 150 mm |
| Distance au coin intérieur, profilé non blanc | environ 250 mm |
| Joint de construction minimal | 10 mm |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.7, p. 3 et 5)

## Tolérances dimensionnelles de l'ouverture du bâtiment

| Surface de l'élément | Maçonnerie brute | Maçonnerie finie |
| --- | --- | --- |
| jusqu'à 2,5 m | ± 10 mm | ± 5 mm |
| de 2,5 à 5 m | ± 15 mm | ± 10 mm |
| au-delà de 5 m | ± 20 mm | ± 15 mm |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.7, p. 4)

**L'écart de verticalité admissible est de 1,5 mm/m, sans dépasser 3 mm au total** — un écart
supérieur dégrade la valeur Uw déclarée, pas seulement l'aspect [1 registre 1.3.7 p. 4].

## Largeur du joint de raccordement à la maçonnerie (mastic silicone)

| Surface du profilé | Pose entre mur, jusqu'à 1,5 m | jusqu'à 2,5 m | jusqu'à 3,5 m | jusqu'à 4,5 m | Pose en applique, jusqu'à 2,5 m | jusqu'à 3,5 m | jusqu'à 4,5 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Blanc | 10 mm | 15 mm | 20 mm | 25 mm | 10 mm | 10 mm | 15 mm |
| Non blanc | 15 mm | 20 mm | 25 mm | 30 mm | 10 mm | 15 mm | 20 mm |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.7, p. 6)

**L'épaisseur du mastic d'étanchéité correspond à la moitié de la largeur du joint**, sauf
indication contraire du fabricant.

## Dilatation de l'aluminium

**Dilatation d'environ 1,2 mm par mètre pour un écart de température de 50 °C.** Ne jamais monter
une longueur d'aluminium supérieure à 3 m sans joint de dilatation bout à bout [1 registre 1.3.7
p. 7].

# Entrées d'air autoréglables

| Paramètre | Valeur |
| --- | --- |
| Modules admis | 20 et 30 m³/h, selon NF E51-732 |
| Module 45 m³/h | non admissible tel quel — obtenu par deux modules de 22 m³/h |
| Passage direct dormant/ouvrant si entailles alignées | 12 mm |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.8, p. 1 et 2)

L'usinage de la mortaise se fait selon le cahier CSTB 3376 V3, en une, deux ou quatre lumières
oblongues pour limiter l'affaiblissement structurel du profilé. Ce registre ne traite pas la
performance acoustique du couple menuiserie/entrée d'air, seulement l'aspect aéraulique.

# Ce que le document ne dit pas

- La check-list de contrôle intermédiaire et final (registre 1.3.6) énumère des points à
  vérifier sans cote associée — c'est une liste de contrôle, pas une donnée technique.
- Les principes d'utilisation, de nettoyage et d'entretien pour l'usager final (registre 1.3.9)
  restent hors de cette page : ce sont des consignes grand public, sans prescription de
  fabrication ou de montage.

# Citations

[1] Directives générales profine, version janvier 2023 —
`raw/profine-directives-generales-2023-01.pdf`, registres 1.3.7 et 1.3.8

# Voir aussi

- [profine](/fournisseurs/profine.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [Couplages et contreventements d'éléments](/procedures/couplages-elements.md)
- [Pose de la PERFORM76](/procedures/pose-perform76.md)
