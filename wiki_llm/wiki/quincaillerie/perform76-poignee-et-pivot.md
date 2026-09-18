---
type: Quincaillerie
title: Poignée et pivot PERFORM76
description: Positions de poignée par hauteur d'ouvrant sur PERFORM76, réglage du pivot bas et charge admissible par ouvrant.
tags: [perform76, poignee, pivot, charge, reglage, quincaillerie]
status: draft
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
stale_after: 2027-09-02
---

# Charge admissible : 100 kg, et non 130

Le cahier technique annonce **« Charge 100 Kg par ouvrant sur pivot bas »** (cahier technique,
p. 3), là où le [catalogue général](/sources/catalogue-general-2026.md) annonce un « pivot pouvant
supporter le poids d'une fenêtre jusqu'à 130 kg » (p. 6).

| Source | Valeur | Formulation |
| --- | --- | --- |
| Cahier technique PERFORM76, p. 3 | 100 kg | « par ouvrant sur pivot bas » |
| Catalogue général, p. 6 | 130 kg | « le poids d'une fenêtre » |

Les deux formulations ne portent peut-être pas sur le même objet, mais l'écart est de 30 %.
**Retenir 100 kg pour un dimensionnement d'atelier** — c'est le document technique — et faire
trancher le bureau d'études avant de s'engager sur 130 kg auprès d'un client. C'est la raison du
`status: draft` de cette page.

Entrée **CTR-01** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

# Cotes de position de poignée

Position de l'axe de poignée selon la hauteur de l'ouvrant, en mm, relevée sur le cahier
technique (p. 3). FFO désigne la hauteur d'axe de poignée au fond de la feuillure quincaillerie.

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

L'écart entre les deux dernières colonnes est constant : **la cote depuis le bas de l'ouvrant
vaut toujours la cote FFO + 20 mm**. C'est le contrôle de cohérence à faire si une valeur est
relevée à la main sur le schéma.

# Les plages de hauteur se chevauchent

Le tableau tel qu'il est imprimé n'est pas utilisable mécaniquement : plusieurs plages de hauteur
d'ouvrant se recouvrent.

| Chevauchement | Plages concernées |
| --- | --- |
| Plage identique, deux positions | 300-600 (120 FFO et 170 FFO) |
| Plage identique, deux positions | 601-900 (220 FFO et 263 FFO) |
| Recouvrement partiel | 601-900 et 801-1 000 |
| Recouvrement partiel | 1 201-1 800 et 1 601-1 800 |

Deux lectures sont possibles et le cahier ne tranche pas : soit les lignes donnent **plusieurs
positions admissibles** pour une même hauteur, au choix, soit certaines bornes sont erronées. Un
ouvrant de 850 mm relève de deux lignes qui donnent 263 et 413 mm — l'écart est trop grand pour
être indifférent.

**À faire valider par le service technique avant d'automatiser ce tableau**, notamment si le but
est d'alimenter un configurateur. Entrée **INC-05** du registre
[Incohérences internes](/anomalies/incoherences-internes.md), qui détaille les quatre
chevauchements.

# Cotes de réglage et de position du pivot bas

Positionnement du pivot bas selon le dormant, en mm, relevé sur le cahier technique (p. 3).

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

Le pivot bas se règle **par clé 6 pans de 4 mm**, sur une plage de **± 2 mm** (cahier technique,
p. 3). Le schéma du dormant 76177 porte les cotes de détail 8, 19,5, 12 et 8.

Une plage de ± 2 mm est étroite : elle corrige un jeu de fabrication, pas un défaut de pose. Le
calage du dormant doit être juste avant de compter sur ce réglage.

# À documenter

Le cahier technique ne traite du reste de la quincaillerie PERFORM76 ni de près ni de loin :
crémones, gâches, paumelles hautes, compas d'oscillo-battant, abaques de charge par type
d'ouverture. Rien de tout cela n'y figure. À compléter quand une documentation
[ROTO](/fournisseurs/roto.md) ou un abaque de quincaillerie sera versé dans `raw/`.

Les poignées elles-mêmes — Sécustik® ATLANTA, TOULON — sont décrites côté commercial dans
[Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md).

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 3
[2] [Catalogue menuiseries PROFERM, édition janvier 2026](/sources/catalogue-general-2026.md),
p. 6

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md)
- [ROTO](/fournisseurs/roto.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
