---
type: Quincaillerie
title: Poignée et pivot PERFORM76
description: Positions de poignée par hauteur d'ouvrant sur PERFORM76, position et réglage du pivot bas par dormant, et charge admissible par ouvrant.
tags: [perform76, poignee, pivot, charge, reglage, quincaillerie]
status: draft
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
    last_modified: 2026-09-02
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
verified:
  by: process:claude-code
  at: 2026-09-18T22:00:00Z
---

# Charge admissible sur le pivot bas

La charge admissible sur le pivot bas d'une PERFORM76 est de **100 kg par ouvrant** [1 p. 3].

Le catalogue général annonce 130 kg pour « le poids d'une fenêtre ». Les deux énoncés ne portent
peut-être pas sur le même objet, et l'écart est de 30 % : **retenir 100 kg pour un
dimensionnement d'atelier** — entrée **CTR-01** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

# Cotes de position de poignée

Position de l'axe de poignée selon la hauteur de l'ouvrant, en mm. **FFO** est la hauteur d'axe de
poignée au fond de la feuillure quincaillerie.

| Hauteur ouvrant mini (mm) | Hauteur ouvrant maxi (mm) | Axe poignée FFO (mm) | Axe poignée depuis bas ouvrant (mm) |
| --- | --- | --- | --- |
| 300 | 600 | 120 | 140 |
| 300 | 600 | 170 | 190 |
| 601 | 900 | 220 | 240 |
| 601 | 900 | 263 | 283 |
| 801 | 1 000 | 413 | 433 |
| 1 001 | 1 200 | 513 | 533 |
| 1 201 | 1 800 | 563 | 583 |
| 1 601 | 1 800 | 763 | 783 |
| 1 801 | 2 400 | 1 000 | 1 020 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 3)

La cote depuis le bas de l'ouvrant vaut toujours la cote FFO **plus 20 mm**, sur les neuf lignes.

## Positions admissibles par hauteur d'ouvrant

Les plages du tableau se recouvrent : plusieurs hauteurs d'ouvrant relèvent de deux lignes. Le
tableau ci-dessous découpe les plages en bandes disjointes et donne, pour chaque bande, **toutes
les positions FFO que le tableau source y autorise**. C'est une lecture des plages, pas une
prescription : le document ne dit pas laquelle retenir quand il y en a deux.

| Hauteur d'ouvrant (mm) | Positions FFO possibles (mm) |
| --- | --- |
| 300 à 600 | 120 ou 170 |
| 601 à 800 | 220 ou 263 |
| 801 à 900 | 220, 263 ou 413 |
| 901 à 1 000 | 413 |
| 1 001 à 1 200 | 513 |
| 1 201 à 1 600 | 563 |
| 1 601 à 1 800 | 563 ou 763 |
| 1 801 à 2 400 | 1 000 |

Le cas le plus large est celui d'un ouvrant de **801 à 900 mm**, où le tableau source autorise
trois positions distantes de 193 mm. **Faire valider ce tableau par le service technique avant
d'alimenter un configurateur** — entrée **INC-05** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

# Cotes de position du pivot bas

Position du pivot bas selon le dormant, en mm.

| Dormant | Position pivot (mm) | Retrait (mm) | Hauteur (mm) |
| --- | --- | --- | --- |
| 76171 | 20 | 19,5 | 11 |
| 76172 | 38 | 19,5 | 11 |
| 76177 | 37 | 19,5 | 11 |
| 76180 | 40 | 19,5 | 11 |
| 76185 | 57 | 19,5 | 11 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 3)

Le retrait de 19,5 mm et la hauteur de 11 mm sont communs aux cinq dormants : **seule la position
varie**, de 20 mm sur le 76171 à 57 mm sur le 76185.

# Réglage en hauteur

Le pivot bas se règle à la **clé 6 pans de 4 mm**, sur une plage de **± 2 mm** [1 p. 3]. La
planche du dormant 76177 porte en détail les cotes 8, 19,5, 12 et 8.

Une plage de ± 2 mm corrige un jeu de fabrication, pas un défaut de pose : le calage du dormant
doit être juste avant de compter sur ce réglage.

# Ce que la source ne donne pas

Le cahier technique PERFORM76 ne traite aucune autre pièce de quincaillerie : ni crémone, ni
gâche, ni paumelle haute, ni compas d'oscillo-battant, ni abaque de charge par type d'ouverture.

Les abaques de charge de la ferrure sont dans
[Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md), et les poignées
commerciales dans [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md).

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 3

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md)
- [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)
- [Glossaire des sigles et des cotes](/reference/glossaire.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
