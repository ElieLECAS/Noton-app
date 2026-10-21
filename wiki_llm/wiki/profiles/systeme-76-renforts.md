---
type: Profilé
title: 'Renforts acier du système 76 par profilé : IW, IG, dormants, ouvrants, meneaux'
description: Ce que désignent les inerties IW (vent) et IG (poids) d'un renfort acier du système 76 Advanced, puis les renforts admis, avec épaisseur d'acier, IW et IG, dans chaque dormant, ouvrant, meneau, traverse, battement et profilé complémentaire.
tags: [systeme-76-advanced, renfort, acier, inertie, statique, atelier]
systeme: 76
fournisseur: KÖMMERLING
usage: atelier
status: draft
sources:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
    last_modified: 2025-06-19
  - resource: raw/poster-systeme-76-advanced-principaux-2022.pdf
    id: poster-76-advanced-principaux
    title: Poster Système 76 Advanced, profilés principaux, 2022
    last_modified: 2022-12-31
  - resource: raw/poster-systeme-76-advanced-complementaires-2022.pdf
    id: poster-76-advanced-complementaires
    title: Poster Système 76 Advanced, profilés complémentaires, 2022
    last_modified: 2022-12-31
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
    last_modified: 2025-06-19
source_pages:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    pages: 5-13, 19-35, 38, 40-53, 118-126
  - resource: raw/poster-systeme-76-advanced-principaux-2022.pdf
    pages: 1
  - resource: raw/poster-systeme-76-advanced-complementaires-2022.pdf
    pages: 1
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    pages: 21
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    pages: 6, 20
generated:
  by: process:claude-code
  at: 2026-10-21T10:43:10Z
---

# Ce que désignent IW et IG

Chaque renfort acier du **système 76 Advanced à joint central** de
[profine](/fournisseurs/profine.md) porte deux valeurs d'inertie, et elles ne servent pas à la
même chose [1 registre 2.3.3 p. 2] :

| Valeur | Ce qu'elle borne | Ce qu'on en fait |
| --- | --- | --- |
| **IW** | la reprise des efforts dans la direction du vent | détermine la dimension réalisable sous une charge de vent donnée |
| **IG** | la reprise des efforts de poids | détermine l'**épaisseur de vitrage** admissible, donc le poids suspendu |

Un renfort à forte IW et faible IG tient le vent mais pas le triple vitrage. C'est la distinction
qui explique que deux renforts de même épaisseur d'acier ne donnent pas les mêmes limites.

Les inerties sont exprimées en **cm⁴**, l'épaisseur d'acier en **mm**. Les limites dimensionnelles
qui en découlent sont dans
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Renforts des dormants

Renforts admis par dormant, relevés sur les planches de profilés du registre 2.1.2 (p. 19 à 25 du
PDF, pages imprimées 3 à 9, version octobre 2021). Une ligne par couple dormant et renfort.

| Dormant | Renfort | Épaisseur d'acier (mm) | IW (cm⁴) | IG (cm⁴) |
| --- | --- | --- | --- | --- |
| 76171 | V306.Z | 1,5 | 2,3 | 1,3 |
| 76171 | V307.Z | 2,0 | 2,9 | 1,6 |
| 76171 | V308 | 2,5 | 3,4 | 1,9 |
| 76171 | V309.Z | 1,5 | 2,5 | 2,0 |
| 76171 | V310 | 2,0 | 3,2 | 2,5 |
| 76172 | V314.Z | 2,0 | 5,7 | 8,4 |
| 76172 | V325 | 2,0 | 4,2 | 7,5 |
| 76172 | V353 | 1,5 | 4,4 | 6,5 |
| 76173 | V306.Z | 1,5 | 2,3 | 1,3 |
| 76173 | V307.Z | 2,0 | 2,9 | 1,6 |
| 76173 | V308 | 2,5 | 3,4 | 1,9 |
| 76173 | V309.Z | 1,5 | 2,5 | 2,0 |
| 76173 | V310 | 2,0 | 3,2 | 2,5 |
| 76177 | V291.Z | 1,5 | 2,3 | 0,7 |
| 76178 | V291.Z | 1,5 | 2,3 | 0,7 |
| 76180 | V306.Z | 1,5 | 2,3 | 1,3 |
| 76180 | V307.Z | 2,0 | 2,9 | 1,6 |
| 76180 | V308 | 2,5 | 3,4 | 1,9 |
| 76180 | V309.Z | 1,5 | 2,5 | 2,0 |
| 76180 | V310 | 2,0 | 3,2 | 2,5 |
| 76185 | V291.Z | 1,5 | 2,3 | 0,7 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 19 à 25)

**Les trois dormants rénovation 76177, 76178 et 76185 n'ont qu'un seul renfort possible**, le
V291.Z, et c'est celui de plus faible IG de tout le système : 0,7 cm⁴. Les trois dormants neufs
76171, 76173 et 76180 en acceptent cinq, le **76172** trois : V314.Z et V353, désignés « soudé »,
et V325. Le **V314.Z** est le renfort de dormant le plus raide en poids de ce tableau (IG 8,4 cm⁴) ;
les trois renforts du 76172 équipent aussi le dormant de porte 76102 (voir
[Porte d'entrée du système 76 Advanced](/portes/systeme-76-advanced-porte-d-entree.md)).

À épaisseur d'acier égale, **le V309.Z apporte plus d'IG que le V306.Z** (2,0 contre 1,3 cm⁴)
pour une IW comparable : c'est le renfort à demander quand le vitrage est lourd et que le vent ne
l'est pas.

**Un huitième dormant, le 76179, apparaît sur le DTD** du système, groupé avec les 76171, 76172 et
76180 dans le tableau d'assignation du drainage — ni ses cotes ni son renfort ne sont donnés par
une source du wiki [2 p. 47]. Le même tableau porte la référence **78275** parmi les ouvrants, une
coquille probable pour 76275 — entrée **INC-14** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

# Renforts des ouvrants

Relevés sur les planches du registre 2.1.2 (p. 27 à 31 du PDF, pages imprimées 11 à 15, version
octobre 2021) et sur les abaques du registre 2.3.3 (p. 6 à 11).

| Ouvrant | Renfort | Épaisseur d'acier (mm) | IW (cm⁴) | IG (cm⁴) |
| --- | --- | --- | --- | --- |
| 76271 | V306.Z | 1,5 | 2,3 | 1,3 |
| 76271 | V307.Z | 2,0 | 2,9 | 1,6 |
| 76271 | V308 | 2,5 | 3,4 | 1,9 |
| 76271 | capot alu A072, sans renfort acier | — | 0,8 | 1,5 |
| 76272 | V326.Z | 2,0 | 5,0 | 5,4 |
| 76272 | V314.Z | 2,0 | 5,7 | 8,4 |
| 76272 | V337 R/L | 2,0 | 5,7 | 8,4 |
| 76272 | V339 R/L | 2,0 | 5,7 | 8,4 |
| 76272 | V353 | 1,5 | 4,4 | 6,5 |
| 76274 | V266.Z | 2,0 | 2,5 | 0,5 |
| 76275 | V266.Z | 2,0 | 2,5 | 0,5 |
| 76276 | V266.Z | 2,0 | 2,5 | 0,5 |
| 76279 | V326.Z | 2,0 | 5,0 | 5,4 |
| 76279 | V314.Z | 2,0 | 5,7 | 8,4 |
| 76279 | V337 R/L | 2,0 | 5,7 | 8,4 |
| 76279 | V339 R/L | 2,0 | 5,7 | 8,4 |
| 76279 | V353 | 1,5 | 4,4 | 6,5 |
| 76281 | V266.Z | 2,0 | 2,5 | 0,5 |
| 76283 | V314.Z | 2,0 | 5,7 | 8,4 |
| 76283 | V326.Z | 2,0 | 5,0 | 5,4 |
| 76283 | V337 R/L | 2,0 | 5,7 | 8,4 |
| 76283 | V339 R/L | 2,0 | 5,7 | 8,4 |
| 76283 | V353 | 1,5 | 4,4 | 6,5 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 27 à 31)

Sur les trois ouvrants de 110 mm, les V314.Z, V337 R/L, V339 R/L et V353 sont désignés « soudé »,
le V326.Z non [1 p. 29-30].

**L'écart entre les ouvrants bas et les ouvrants hauts est un facteur 10 en IG** : 0,5 cm⁴ pour le
V266.Z des 76275 et 76281, 5,4 à 8,4 cm⁴ pour les renforts des 76272 et 76279. C'est ce qui
détermine l'épaisseur de vitrage admissible, pas la hauteur du profil elle-même.

Le **76271 équipé du capot aluminium A072 se passe de renfort acier** — c'est la variante AluClip
Pro. Ses inerties sont celles du capot, et elles sont les plus faibles du système en vent (0,8 cm⁴).
Une **équerre de feuillure J079 dans les quatre angles devient obligatoire à partir de 40 kg de
poids d'ouvrant** [1 registre 2.6.3 p. 1]. Les battements, eux, restent
systématiquement renforcés.

Voir [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md).

# Renforts des meneaux, traverses et battements

Relevés sur les planches du registre 2.1.2 (p. 32 à 35 du PDF, pages imprimées 16 à 19, version
octobre 2021).

| Profilé | Famille | Renfort | Épaisseur d'acier (mm) | IW (cm⁴) | IG (cm⁴) |
| --- | --- | --- | --- | --- | --- |
| 76300 | croisillon d'ouvrant | V312.Z | 1,5 | 1,5 | 0,3 |
| 76301 | traverse d'ouvrant | V320.Z | 1,5 | 3,5 | 2,4 |
| 76303 | traverse d'ouvrant | V323.Z | 1,5 | 9,2 | 13,0 |
| 76303 | traverse d'ouvrant | V322 | 2,5 | 16,8 | 22,0 |
| 76303 | traverse d'ouvrant | V324 | 2,5 | - | - |
| 76372 | meneau de dormant | V318.Z | 2,0 | 7,0 | 3,4 |
| 76372 | meneau de dormant | V319 | 2,5 | 9,6 | 4,2 |
| 76373 | meneau de dormant | V323.Z | 1,5 | 9,2 | 13,0 |
| 76373 | meneau de dormant | V322 | 2,5 | 16,8 | 22,0 |
| 76373 | meneau de dormant | V324 | 2,5 | 13,3 | 17,1 |
| 76471 | battement | V316 | 1,5 | 2,4 | 0,4 |
| 76472 | battement | V317 | 2,0 | 4,8 | 2,3 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 32 à 35)

Relu en image, le cartouche du meneau 76373 porte bien ses propres références, V323.Z, V322 et
V324. Le **V324** y est désigné « Capot Alu », mais son dessin est celui d'un renfort en L de
44 × 55 mm, en tôle de **2,5 mm**, avec IG 17,1 et IW 13,3 cm⁴ [1 p. 33] ; sur la planche du
76303, le même V324 est désigné « Renfort 2,5 mm » mais sa case « Valeurs » porte les valeurs E et
S du capot A070 au lieu de ses inerties [1 p. 34] — entrée **INC-30** du registre
[Incohérences internes](/anomalies/incoherences-internes.md). L'inertie du V324 sur le 76303 reste
donc sans valeur ici. **À confirmer avant tout calcul statique** — entrée **VER-23** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Sur ces planches, le V318.Z est coté **45 × 28,6 mm**, le V320.Z **37 × 28,6 mm** et le V319
44 × 29 mm [1 p. 33-34], là où le poster des profilés principaux donne 29 mm de haut au V318.Z et
au V320.Z (voir *Sections et inerties des renforts*) — entrée **CTR-23** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

Le **battement 76473 ne reçoit aucun renfort interne** dans le manuel de mise en œuvre, alors que les 76471 et 76472 en reçoivent un chacun. En revanche, le **DTD n° DBV-25-6/16-2334_V5** (p. 6, 20 et 43) prévoit pour le 76473 la possibilité d'un **renfort inox extérieur VSF01** ($21,4 \times 11\text{ mm}$, épaisseur $2,5\text{ mm}$, inertie $I_x = 1,08\text{ cm}^4$), vissé sur la face extérieure du battement tous les $250\text{ mm}$ à l'aide de vis inox A2 de $4,1 \times 40\text{ mm}$.

Les **dormants pour cadres fixes exclusifs 76101 et 76102** (inconnus du cahier PERFORM76) reçoivent quant à eux :
- 76101 : V306, V307, V309 ou **V329** ($32,5 \times 28\text{ mm}$, épaisseur $1,25\text{ mm}$, $I_x = 2,15\text{ cm}^4$)
- 76102 : V314 ($I_x = 5,7\text{ cm}^4$) ou V326 ($I_x = 5,0\text{ cm}^4$)
- Les renforts soudables avec insert **V337L/V337R** et **V339L/V339R** ($35 \times 45\text{ mm}$, épaisseur $2\text{ mm}$) équipent également les profils 76102/76172/76272/76279.

Voir [Meneaux PERFORM76](/profiles/perform76-meneaux.md) et
[Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md).

# Renforts des profilés complémentaires

Chaque planche du registre 2.1.3 « Plan des Profilés » du manuel de mise en œuvre dessine, dans
son tableau « Accessoires », le ou les renforts admis dans le profilé complémentaire, avec leurs
cotes, l'épaisseur de la tôle et les deux inerties IG (poids) et IW (vent) en cm⁴, indices écrits.
Une ligne par profilé et par renfort, telle que la planche la porte ; la largeur est la cote
horizontale du dessin du renfort, la hauteur sa cote verticale.

| Profilé | Titre de la planche | Renfort | Désignation | Largeur (mm) | Hauteur (mm) | Épaisseur d'acier (mm) | IG (cm⁴) | IW (cm⁴) | Page PDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 76299 | 76299 Compensateur | V323.Z | Renfort 1,5 mm | 45 | 55 | 1,5 | 13,0 | 9,2 | 38 |
| 76299 | 76299 Compensateur | V322 | Renfort 2,5 mm | 44 | 55 | 2,5 | 22,0 | 16,8 | 38 |
| 76701 | Elargisseur 30 mm | V312.Z | Renfort 1,5 mm | 33 | 13 | 1,5 | 0,3 | 1,5 | 40 |
| 76702 | Elargisseur 60 mm | V314.Z | Renfort 2,0 mm, soudé | 35 | 45 | 2 | 8,4 | 5,7 | 40 |
| 76703 | Elargisseur 120 mm | V314.Z | Renfort 2,0 mm, soudé | 35 | 45 | 2 | 8,4 | 5,7 | 41 |
| 76704 | Réhausse 45 mm | V114 | Renfort 1,5 mm | 25 | 30 | 1,5 | 1,9 | 1,4 | 41 |
| 76706 | Réhausse 45 mm | V114 | Renfort 1,5 mm | 25 | 30 | 1,5 | 1,9 | 1,4 | 41 |
| 76705 | Réhausse 150 mm | V317 | Renfort 2,0 mm | 40 | 25 | 2 | 4,8 | 2,3 | 42 |
| 76708 | Réhausse 50 mm | V407 | Renfort1,5 mm | 20 | 25 | 1,5 | 1,0 | 0,7 | 42 |
| 76709 | Réhausse 100 mm | V075 | Renfort1,5 mm | 45 | 18 | 1,5 | 4,1 | 1,0 | 42 |
| 76709 | Réhausse 100 mm | V299 | Renfort1,5 mm | 33 | 17 | 1,5 | 1,8 | 0,6 | 42 |
| 76715 | Réhausse 50 mm, complément pour 76708 | V407 | Renfort 1,5 mm | 20 | 25 | 1,5 | 1,0 | 0,7 | 43 |
| 76722 | Réhausse 35 mm | V167 | Renfort 1,5 mm | 30 | 20 | 1,5 | 1,61 | 1,07 | 43 |
| 76777 | Réhausse 55 mm | V114 | Renfort 1,5 mm | 25 | 30 | 1,5 | 1,9 | 1,4 | 43 |
| 76605 | Profilé de liaison-H | V330 | Renfort 2,5 mm | - | - | 2,5 | 0,8 | 5,5 | 50 |
| 76605 | Profilé de liaison-H | V331 | Renfort 2,5 mm | - | - | 2,5 | 1,0 | 10,0 | 50 |
| 76608 | Profilé de liaison-H | V288 | Renfort 2,0 mm | - | - | 2 | 0,7 | 20,4 | 51 |
| A250 | Profilé de liaison | V264 | Renfort | - | - | - | - | 17,9 | 52 |
| 8340 | Poteau d'angle variable | V265 | Renfort 2,0 mm, soudé | - | - | 2 | 8,7 | 8,7 | 52 |
| 8355 | Poteau d'angle 90° | V263 | Renfort 2,0 mm, soudé | - | - | 2 | 14,4 | 14,4 | 53 |
| 8356 | Poteau d'angle 135° | V262 | Renfort 2,0 mm | - | - | 2 | 7,6 | 7,6 | 53 |
| 76802 | Profilé de jonction CVR | V332 | - | - | - | - | 0,1 | 5,7 | 53 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 38 à 53, registre 2.1.3, p. 1 à 15)

Les renforts des profilés de liaison, des poteaux et du 76802 sont dessinés sans cote au manuel :
leurs sections sont celles du tableau *Renforts dessinés sur le poster des profilés
complémentaires* plus bas. Le profilé de guidage de tablier **76800** n'a pas de tableau
d'accessoires, donc pas de renfort ; une lecture antérieure lui prêtait le V262, qui est celui du
poteau 8356. L'**A250**, profilé aluminium, porte sous sa coupe « A250 Alu IW = 79 cm⁴ » et
« Valeur de remplacement acier IW = 26 cm⁴ » [1 p. 52].

L'élargisseur **76700** n'a pas de tableau d'accessoires : aucun renfort. Les quatre renforts que
les planches du poster ne dessinent pas :

| Renfort | Coupe |
| --- | ---: |
| V407 | ![Renfort V407](/assets/profiles/systeme76/renforts/renfort-v407.png) |
| V299 | ![Renfort V299](/assets/profiles/systeme76/renforts/renfort-v299.png) |
| V167 | ![Renfort V167](/assets/profiles/systeme76/renforts/renfort-v167.png) |
| V288 | ![Renfort V288](/assets/profiles/systeme76/renforts/renfort-v288.png) |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 42, 43 et 51)

Le **V075** porte au manuel **IG 4,1 et IW 1,0**, indices écrits ; sur le poster des profilés
complémentaires, ses deux inerties sont écrites « I = » sans indice (entrée **INC-20**). Le
**V317** est dessiné couché, 40 × 25 mm, comme sur le poster des profilés principaux, mais avec
les inerties du poster des profilés complémentaires, IG 4,8 et IW 2,3 — entrée **VER-24** du
registre [Informations à vérifier](/anomalies/informations-a-verifier.md). Le **V264** porte au
manuel une seule inertie, IW 17,9 cm⁴, contre deux valeurs sans indice de 0,5 et 1,8 cm⁴ sur le
poster (entrée **INC-20**) ; le **V263** 14,4 cm⁴ contre 14,5 (entrée **CTR-23**).

Voir [Élargisseurs et assemblage PERFORM76](/profiles/perform76-elargisseurs-et-assemblage.md).

# Citations

[1] Mise en œuvre Système 76 Advanced, profine —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registres 2.1.1 (porte d'entrée, p. 5 à 13
du PDF), 2.1.2 (p. 19 à 35), 2.1.3, 2.3.3, 2.4.1 et 2.6.3

[2] DTD n° DBV-25-6/16-2334_V5, système 76 Advanced —
`raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf`, p. 47

[3] [Poster Système 76 Advanced, profilés principaux, 2022](raw/poster-systeme-76-advanced-principaux-2022.pdf), p. 1

[4] [Poster Système 76 Advanced, profilés complémentaires, 2022](raw/poster-systeme-76-advanced-complementaires-2022.pdf), p. 1

[5] [DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED](raw/dta-trocal-76-advanced-6-16-2334-v5.pdf)

# Voir aussi

- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
- [Renforts et accessoires par profilé du système 76](/profiles/systeme-76-accessoires-par-profile.md)
- [Porte d'entrée du système 76 Advanced](/portes/systeme-76-advanced-porte-d-entree.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md)
- [Profilés complémentaires du système 76](/profiles/systeme-76-profiles-complementaires.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)

Autres parties de ce sujet :

- [Sections et inerties des renforts du système 76 : largeur, hauteur, épaisseur, IG, IW](/profiles/systeme-76-renforts-sections-et-inerties.md)
- [Renforts du système 76 à inertie unique et renforts écrits dans la coupe de chaque profilé](/profiles/systeme-76-renforts-profiles-equipes.md)
- [Vissage des renforts dans les profilés du système 76 Advanced : vis, cote de l'axe, paroi](/profiles/systeme-76-renforts-vissage.md)
