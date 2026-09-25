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
    pages: 6
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    pages: 100
generated:
  by: process:multimodal-direct
  at: 2026-09-19T21:15:00Z
---

La poignée d'une fenêtre PERFORM76 se place à une hauteur qui dépend de la hauteur de l'ouvrant
(le cadre qui s'ouvre), et l'ouvrant tourne sur un **pivot bas**, la ferrure du bas qui porte son
poids. Cette page donne où placer l'une et l'autre, et comment régler le pivot.

# Charge admissible sur le pivot bas

La charge admissible sur le pivot bas d'une PERFORM76 est de **100 kg par ouvrant** [1 p. 6].

Le catalogue général annonce 130 kg pour « le poids d'une fenêtre ». Les deux énoncés ne portent
peut-être pas sur le même objet, et l'écart est de 30 % : **retenir 100 kg pour un
dimensionnement d'atelier** — entrée **CTR-01** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

# Cotes de position de poignée

Position de l'axe de poignée selon la hauteur de l'ouvrant, en mm. Deux cotes donnent la même
position, mesurée depuis deux repères différents : **FFO**, la hauteur de l'axe de poignée
mesurée depuis le **fond de feuillure quincaillerie** (le creux du profilé où se loge la
ferrure), et la hauteur mesurée depuis le **bas de l'ouvrant**. Une ligne du tableau se lit :
pour un ouvrant dont la hauteur est comprise entre le minimum et le maximum, l'axe de poignée se
place à la cote FFO, soit à la cote « bas ouvrant ».

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

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 6)

Le schéma montre, en coupe verticale, le bas d'un ouvrant avec sa poignée : les deux flèches
de cote partent l'une du fond de feuillure quincaillerie (hauteur FFO), l'autre du bas de
l'ouvrant, et montent jusqu'à l'axe de la poignée.

![Hauteur poignée suivant hauteur ouvrant](/assets/profiles/perform76/pivot/poignee-hauteur.png)

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 6)

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

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 6)

Le retrait se lit sur les coupes de la colonne *Coupe* : la cote de 19,5 mm part du bord de
l'ouvrant, la position du pivot est la seconde cote horizontale, et la hauteur de 11 mm est
cotée à droite.

## Réglage du pivot bas

Le pivot bas se règle à la **clé 6 pans de 4 mm**, avec une course de **± 2 mm**. La charge
admissible est de **100 kg par ouvrant** [1 p. 6].

![Pivot bas réglage et positionnement](/assets/profiles/perform76/pivot/pivot-bas-reglage.png)

Le schéma montre l'angle bas d'un ouvrant sur un dormant 76177, vu de face : le vitrage en bleu,
le pivot dessiné dans l'angle, les traits rouges marquant le contour de l'ouvrant. Quatre cotes
sont portées autour du pivot : **8 mm** et **19,5 mm** dans le sens horizontal, **12 mm** et
**8 mm** dans le sens vertical ; la planche ne légende pas leur point de départ. Les flèches rouges verticales indiquent le réglage en hauteur,
les flèches horizontales, sous le pivot, un réglage latéral.

Les deux vis de réglage sont libellées « Réglage hauteur » sur la planche, y compris celle qui
porte les flèches horizontales — entrée **INC-17** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 6)

Le retrait de 19,5 mm et la hauteur de 11 mm sont communs aux cinq dormants : **seule la position
varie**, de 20 mm sur le 76171 à 57 mm sur le 76185.

# Nombre de paumelles par hauteur d'ouvrant

Le système 76 Advanced prévoit **2 paumelles de 50 à 90 cm de hauteur d'ouvrant, 3 de 100 à
160 cm, 4 de 170 à 210 cm et 5 de 220 à 250 cm** ; le tableau et ses conditions sont dans
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md#nombre-de-paumelles)
[2 p. 100].

Pour les portes d'entrée de la gamme PERFORM (ouvrants 76272 et 76279), 3 paumelles en applique **Roto Solid B** sont montées de série, portées à 4 paumelles pour vantail de hauteur $> 2\,200\text{ mm}$ ou masse $> 120\text{ kg}$ — voir [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md).

# Ce que la source ne donne pas

Le cahier technique PERFORM76 ne traite aucune autre pièce de quincaillerie : ni crémone, ni
gâche, ni paumelle haute, ni compas d'oscillo-battant, ni abaque de charge par type d'ouverture.

Les abaques de charge de la ferrure sont dans
[Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md), et les poignées
commerciales dans [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md).

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, page du PDF 6

[2] Mise en œuvre Système 76 Advanced, profine, édition décembre 2023 —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, PDF p. 100 (registre 2.3.3, p. 3)

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md)
- [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)
- [Glossaire des sigles et des cotes](/reference/glossaire.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
