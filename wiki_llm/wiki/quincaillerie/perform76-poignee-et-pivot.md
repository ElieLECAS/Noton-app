---
type: Quincaillerie
title: Poignée et pivot PERFORM76
description: Positions de poignée par hauteur d'ouvrant sur PERFORM76, position et réglage du pivot bas par dormant, et charge admissible par ouvrant.
tags: [perform76, poignee, pivot, charge, reglage, quincaillerie]
gamme: PERFORM
systeme: 76
usage: [atelier, sav]
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
    last_modified: 2026-09-02
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine, édition décembre 2023
    last_modified: 2023-12-14
source_pages:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    pages: 3
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    pages: 100
generated:
  by: process:multimodal-direct
  at: 2026-09-19T21:15:00Z
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

| Schéma | Coupe |
| --- | ---: |
| Hauteur poignée suivant hauteur ouvrant, FFO et bas ouvrant | ![Hauteur poignée suivant hauteur ouvrant](/assets/profiles/perform76/pivot/poignee-hauteur.png) |

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

| Dormant | Position pivot (mm) | Retrait (mm) | Hauteur (mm) | Coupe |
| --- | --- | --- | --- | ---: |
| 76171 | 20 | 19,5 | 11 | ![Dormant 76171](/assets/profiles/perform76/pivot/pivot-76171.png) |
| 76172 | 38 | 19,5 | 11 | ![Dormant 76172](/assets/profiles/perform76/pivot/pivot-76172.png) |
| 76177 | 37 | 19,5 | 11 | ![Dormant 76177](/assets/profiles/perform76/pivot/pivot-76177.png) |
| 76180 | 40 | 19,5 | 11 | ![Dormant 76180](/assets/profiles/perform76/pivot/pivot-76180.png) |
| 76185 | 57 | 19,5 | 11 | ![Dormant 76185](/assets/profiles/perform76/pivot/pivot-76185.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 3)

Le réglage en hauteur du pivot bas se fait à la clé 6 pans de 4 mm, à ± 2 mm, pour une charge de
100 kg par ouvrant.

| Schéma | Coupe |
| --- | ---: |
| Pivot bas, réglage et positionnement | ![Pivot bas réglage et positionnement](/assets/profiles/perform76/pivot/pivot-bas-reglage.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 3)

Le retrait de 19,5 mm et la hauteur de 11 mm sont communs aux cinq dormants : **seule la position
varie**, de 20 mm sur le 76171 à 57 mm sur le 76185.

# Nombre et implantation des paumelles par hauteur d'ouvrant (registre 2.3.3)

Relevé sur le graphique de distribution du manuel Système 76 Advanced (registre 2.3.3, p. 100) :

| Hauteur fond de feuillure HFF (mm) | Nombre de paumelles / paliers | Position des points de reprise de charge | Accessoire de maintien médian requis |
| --- | --- | --- | --- |
| 300 à 800 | 2 paumelles | 1 palier d'angle bas + 1 palier compas haut | — |
| 801 à 1 400 | 2 paumelles | 1 palier d'angle bas + 1 palier compas haut | 1 verrouilleur médian vertical (G1) |
| 1 401 à 1 800 | 2 paumelles renforcées | 1 palier d'angle 130 kg + 1 palier compas | 1 verrouilleur vertical (G1 + G2) |
| 1 801 à 2 200 (porte-fenêtre) | 3 paumelles ou 2 paumelles + 2 verrouilleurs | Palier bas + compas haut + 1 paumelle intermédiaire à mi-hauteur | Verrouilleur vertical arrière continu |
| 2 201 à 2 400 | 3 paumelles renforcées (150 kg) | Palier bas + compas haut + 1 paumelle intermédiaire à 350 mm sous l'angle haut | Verrouilleur vertical arrière continu + allonge |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.3, p. 100)

Pour les portes d'entrée de la gamme PERFORM (ouvrants 76272 et 76279), 3 paumelles en applique **Roto Solid B** sont montées de série, portées à 4 paumelles pour vantail de hauteur $> 2\,200\text{ mm}$ ou masse $> 120\text{ kg}$ — voir [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md).

# Ce que la source ne donne pas

Le cahier technique PERFORM76 ne traite aucune autre pièce de quincaillerie : ni crémone, ni
gâche, ni paumelle haute, ni compas d'oscillo-battant, ni abaque de charge par type d'ouverture.

Les abaques de charge de la ferrure sont dans
[Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md), et les poignées
commerciales dans [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md).

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 3

[2] Mise en œuvre Système 76 Advanced, profine, édition décembre 2023 —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registre 2.3.3, p. 100

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md)
- [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)
- [Glossaire des sigles et des cotes](/reference/glossaire.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
