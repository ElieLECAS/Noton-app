---
type: Profilé
title: Renforts du système 76
description: Les renforts acier du système 76 Advanced à joint central, leur épaisseur, leurs inerties IW et IG, et le profilé que chacun équipe.
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
    pages: 5-13, 19-35
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
  at: 2026-09-18T09:00:00Z
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

Relevés sur les planches du registre 2.1.3 (p. 1 à 16, versions mars 2021 et septembre 2023).

| Profilé | Famille | Renfort | Épaisseur d'acier (mm) | IW (cm⁴) | IG (cm⁴) |
| --- | --- | --- | --- | --- | --- |
| 76700 | élargisseur 15 mm | V312.Z | 1,5 | 1,5 | 0,3 |
| 76701 | élargisseur 30 mm | V312.Z | 1,5 | 1,5 | 0,3 |
| 76702 | élargisseur 60 mm | V312.Z | 1,5 | 1,5 | 0,3 |
| 76703 | élargisseur 120 mm | V114 | 1,5 | 1,4 | 1,9 |
| 76704 | réhausse 45 mm | V114 | 1,5 | 1,4 | 1,9 |
| 76705 | réhausse 150 mm | V317 | 2,0 | 2,3 | 4,8 |
| 76706 | réhausse 45 mm | V114 | 1,5 | 1,4 | 1,9 |
| 76708 | réhausse 50 mm | V075 | 1,5 | 1,0 | 4,1 |
| 76709 | réhausse 100 mm | V299 | 1,5 | 0,6 | 1,8 |
| 76715 | réhausse 50 mm, complément du 76708 | V407 | 1,5 | 0,7 | 1,0 |
| 76722 | réhausse 35 mm | V167 | 1,5 | 1,07 | 1,61 |
| 76777 | réhausse 55 mm | V114 | 1,5 | 1,4 | 1,9 |
| 76605 | profilé de liaison en H | V330 | 2,5 | 5,5 | 0,8 |
| 76608 | profilé de liaison en H | V288 | 2,0 | 20,4 | 0,7 |
| 76206 | profilé complémentaire | V323.Z | 1,5 | 9,2 | 13,0 |
| 76206 | profilé complémentaire | V322 | 2,5 | 16,8 | 22,0 |
| 76800 | profilé de guidage de tablier | V262 | 2,0 | 7,6 | 7,6 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.3, p. 1 à 16)

Le **V288 du profilé de liaison 76608 est le renfort le plus raide au vent de tout le système** :
20,4 cm⁴, pour une IG quasi nulle de 0,7. C'est un profil de liaison, pas un porteur de vitrage.

**Le renfort V317 apparaît avec des inerties permutées selon la planche** : IW 4,8 / IG 2,3 sur le
battement 76472, IW 2,3 / IG 4,8 sur la réhausse 76705. C'est cohérent avec un montage tourné de
90°, l'axe fort passant du vent au poids, sans que le document l'écrive. Entrée **VER-24** du
registre [Informations à vérifier](/anomalies/informations-a-verifier.md).

Voir [Élargisseurs et assemblage PERFORM76](/profiles/perform76-elargisseurs-et-assemblage.md).

# Sections et inerties des renforts

Chaque renfort acier du système 76 Advanced est dessiné en coupe, à l'échelle, avec ses cotes
extérieures, l'épaisseur de sa tôle et ses deux inerties, **IG** (poids) et **IW** (vent), en
cm⁴. Une ligne par renfort ; la largeur est la cote horizontale du dessin, la hauteur la cote
verticale.

| Renfort | Largeur (mm) | Hauteur (mm) | Épaisseur d'acier (mm) | IG (cm⁴) | IW (cm⁴) | Coupe |
| --- | --- | --- | --- | --- | --- | ---: |
| V306.Z | 32,5 | 28 | 1,5 | 1,3 | 2,3 | ![Renfort V306.Z](/assets/profiles/systeme76/renforts/renfort-v306z.png) |
| V307.Z | 32,5 | 28 | 2 | 1,6 | 2,9 | ![Renfort V307.Z](/assets/profiles/systeme76/renforts/renfort-v307z.png) |
| V308 | 32,5 | 28 | 2,5 | 1,9 | 3,4 | ![Renfort V308](/assets/profiles/systeme76/renforts/renfort-v308.png) |
| V309.Z | 32,5 | 28 | 1,5 | 2,0 | 2,5 | ![Renfort V309.Z](/assets/profiles/systeme76/renforts/renfort-v309z.png) |
| V310 | 32,5 | 28 | 2 | 2,5 | 3,2 | ![Renfort V310](/assets/profiles/systeme76/renforts/renfort-v310.png) |
| V266.Z | 36,5 | 20 | 2 | 0,52 | 2,52 | ![Renfort V266.Z](/assets/profiles/systeme76/renforts/renfort-v266z.png) |
| V291.Z | 43 | 29 | 1,5 | 0,74 | 2,26 | ![Renfort V291.Z](/assets/profiles/systeme76/renforts/renfort-v291z.png) |
| V316 | 40 | 17,5 | 1,5 | 0,4 | 2,4 | ![Renfort V316](/assets/profiles/systeme76/renforts/renfort-v316.png) |
| V317 | 40 | 25 | 2 | 2,3 | 4,8 | ![Renfort V317](/assets/profiles/systeme76/renforts/renfort-v317.png) |
| V318.Z | 45 | 29 | 2 | 3,4 | 7,0 | ![Renfort V318.Z](/assets/profiles/systeme76/renforts/renfort-v318z.png) |
| V319 | 44 | 29 | 2,5 | 4,2 | 9,6 | ![Renfort V319](/assets/profiles/systeme76/renforts/renfort-v319.png) |
| V320.Z | 37 | 29 | 1,5 | 2,4 | 3,5 | ![Renfort V320.Z](/assets/profiles/systeme76/renforts/renfort-v320z.png) |
| V312.Z | 33 | 13 | 1,5 | 0,3 | 1,5 | ![Renfort V312.Z](/assets/profiles/systeme76/renforts/renfort-v312z.png) |
| V314.Z | 35 | 45 | 2 | 8,4 | 5,7 | ![Renfort V314.Z](/assets/profiles/systeme76/renforts/renfort-v314z.png) |
| V353 | 35 | 45 | 1,5 | 6,5 | 4,4 | ![Renfort V353](/assets/profiles/systeme76/renforts/renfort-v353.png) |
| V326.Z | 35 | 45 | 2 | 5,4 | 5,0 | ![Renfort V326.Z](/assets/profiles/systeme76/renforts/renfort-v326z.png) |
| V325 | 35 | 45 | 2 | 7,5 | 4,2 | ![Renfort V325](/assets/profiles/systeme76/renforts/renfort-v325.png) |
| V337/V339 R/L | 35 | 45 | 2 | 8,4 | 5,7 | ![Renfort V337/V339 R/L](/assets/profiles/systeme76/renforts/renfort-v337-v339.png) |
| V322 | 44 | 55 | 2,5 | 22,0 | 16,8 | ![Renfort V322](/assets/profiles/systeme76/renforts/renfort-v322.png) |
| V323.Z | 45 | 55 | 1,5 | 13,0 | 9,2 | ![Renfort V323.Z](/assets/profiles/systeme76/renforts/renfort-v323z.png) |
| V260 | 40 | 60 | 2,5 | 22,8 | 12,1 | ![Renfort V260](/assets/profiles/systeme76/renforts/renfort-v260.png) |
| V333/V335 R/L | 40 | 60 | 2,5 | 22,8 | 12,1 | ![Renfort V333/V335 R/L](/assets/profiles/systeme76/renforts/renfort-v333-v335.png) |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1, bandes inférieures)

Les inerties se lisent sur la planche avec deux décimales pour deux renforts : **V266.Z** (0,52 et
2,52), **V291.Z** (0,74 et 2,26) ; les tableaux plus haut les arrondissent à une décimale. Le
V291.Z est légendé « V291.Z 1 » sous son dessin — entrée **INC-19** du registre
[Incohérences internes](/anomalies/incoherences-internes.md). Le **V317** est dessiné avec les
inerties du battement 76472, IG 2,3 et IW 4,8 (voir **VER-24**).

Les cotes de détail de quelques renforts, portées sur leur dessin : le V325 a un décrochement de
7,5 × 28 mm ; le V326.Z est ouvert, avec un retour de 37,6 mm et une lèvre de 11,3 mm ; les
V337/V339 R/L et V333/V335 R/L ont une fenêtre de 22 mm à 3 mm du bord, avec des retours de 27 et
7,5 mm (V337/V339) ou de 32 et 15,8 mm (V333/V335) ; le V266.Z porte une lèvre de 8,5 mm et des
retours de 2 et 6,7 mm ; le V316 une lèvre de 15,9 mm et un retour de 7 mm.

Le **V324** et le **V329** ne sont pas dessinés sur la planche : le V324 n'y apparaît que dans la
liste du meneau 76373.

## Renforts dessinés sur le poster des profilés complémentaires

Douze renforts sont dessinés sur la planche des profilés complémentaires, avec les mêmes
indications. Deux figurent déjà sur la planche des profilés principaux (V314.Z, V312.Z) avec les
mêmes valeurs ; leur coupe est celle du tableau précédent.

| Renfort | Largeur (mm) | Hauteur (mm) | Épaisseur d'acier (mm) | IG (cm⁴) | IW (cm⁴) | Coupe |
| --- | --- | --- | --- | --- | --- | ---: |
| V114 | 25 | 30 | 1,5 | 1,9 | 1,4 | ![Renfort V114](/assets/profiles/systeme76/renforts/renfort-v114.png) |
| V314.Z | 35 | 45 | 2 | 8,4 | 5,7 | ![Renfort V314.Z](/assets/profiles/systeme76/renforts/renfort-v314z.png) |
| V317 | 24 | 40 | 2 | 4,8 | 2,3 | ![Renfort V317](/assets/profiles/systeme76/renforts/renfort-v317-tourne.png) |
| V312.Z | 33 | 13 | 1,5 | 0,3 | 1,5 | ![Renfort V312.Z](/assets/profiles/systeme76/renforts/renfort-v312z.png) |
| V330 | 45 | 15 | 2,5 | 0,8 | 5,5 | ![Renfort V330](/assets/profiles/systeme76/renforts/renfort-v330.png) |
| V331 | 56,5 | 15 | 2,5 | 1,0 | 10,0 | ![Renfort V331](/assets/profiles/systeme76/renforts/renfort-v331.png) |
| V075 | 45 | 18 | 1,5 | 1,0 (« I ») | 4,1 (« I ») | ![Renfort V075](/assets/profiles/systeme76/renforts/renfort-v075.png) |
| V265 | Ø 50 | - | 2 | 8,7 | 8,7 | ![Renfort V265](/assets/profiles/systeme76/renforts/renfort-v265.png) |
| V263 | 50 | 50 | 2 | 14,5 | 14,5 | ![Renfort V263](/assets/profiles/systeme76/renforts/renfort-v263.png) |
| V262 | 26,2 | 49 | 2 | 7,6 | 7,6 | ![Renfort V262](/assets/profiles/systeme76/renforts/renfort-v262.png) |
| V332 | 58,3 | 10,6 | 2 | 0,1 | 5,7 | ![Renfort V332](/assets/profiles/systeme76/renforts/renfort-v332.png) |
| V264 | 60 | 10 | plat plein | 0,5 (« I ») | 1,8 (« I ») | ![Renfort V264](/assets/profiles/systeme76/renforts/renfort-v264.png) |

(schéma: raw/poster-systeme-76-advanced-complementaires-2022.pdf, p. 1, coin inférieur droit)

Pour le **V075** et le **V264**, les deux inerties sont écrites « I = » sans l'indice G ou W :
la première est reprise dans la colonne IG, la seconde dans la colonne IW, dans l'ordre des
autres renforts de la planche — entrée **INC-20** du registre
[Incohérences internes](/anomalies/incoherences-internes.md). Le V264 est dessiné en plat plein
de 60 × 10 mm.

Le **V317** est dessiné debout sur cette planche, 24 × 40 mm, avec IG 4,8 et IW 2,3 : l'inverse
du dessin de la planche des profilés principaux, couché, 40 × 25 mm, IG 2,3 et IW 4,8. Les
inerties permutées vont avec le dessin tourné (voir **VER-24**) ; la largeur, 24 ou 25 mm, ne
concorde pas — entrée **CTR-23** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md). Le **V263** porte
14,5 cm⁴ sur la planche, contre 14,4 sur la page du poteau 8355.

## Renforts à inertie unique et profilés d'affectation

Dix-neuf renforts sont dessinés sous le titre « Renforts », chacun avec ses cotes extérieures,
l'épaisseur de sa tôle, **une seule inertie** (« Inertie … Cm4 ») et la liste des profilés qu'il
équipe (« Pour profilé(s) »). Les références y sont écrites sans le suffixe .Z. Les coupes sont
celles des tableaux précédents quand le renfort y est dessiné de même ; celles du V291, du VSF01
et du V329 sont propres à cette planche.

| Renfort | Largeur (mm) | Hauteur (mm) | Épaisseur d'acier (mm) | Inertie portée (cm⁴) | Coupe |
| --- | --- | --- | --- | --- | ---: |
| V306 | 32,5 | 28 | 1,5 | 2,3 | ![Renfort V306](/assets/profiles/systeme76/renforts/renfort-v306z.png) |
| V307 | 32,5 | 28 | 2 | 2,9 | ![Renfort V307](/assets/profiles/systeme76/renforts/renfort-v307z.png) |
| V309 | 32,5 | 28 | 1,5 | 2,5 | ![Renfort V309](/assets/profiles/systeme76/renforts/renfort-v309z.png) |
| V314 | 35 | 45 | 2 | 5,7 | ![Renfort V314](/assets/profiles/systeme76/renforts/renfort-v314z.png) |
| V326 | 35 | 45 | 2 | 5,0 | ![Renfort V326](/assets/profiles/systeme76/renforts/renfort-v326z.png) |
| V291 | 42 | 30 | - | 2,3 | ![Renfort V291](/assets/profiles/systeme76/renforts/renfort-v291-dta.png) |
| V266 | 36,5 | 20 | 2 | 2,5 | ![Renfort V266](/assets/profiles/systeme76/renforts/renfort-v266z.png) |
| V312 | 33 | 13 | 1,5 | 1,5 | ![Renfort V312](/assets/profiles/systeme76/renforts/renfort-v312z.png) |
| V316 | 40 | 17,5 | 1,5 | 2,4 | ![Renfort V316](/assets/profiles/systeme76/renforts/renfort-v316.png) |
| V317 | 25 | 40 | 2 | 4,8 | ![Renfort V317](/assets/profiles/systeme76/renforts/renfort-v317.png) |
| V320 | 37 | 29 | 1,5 | 3,5 | ![Renfort V320](/assets/profiles/systeme76/renforts/renfort-v320z.png) |
| V318 | 45 | 29 | 2 | 7,0 | ![Renfort V318](/assets/profiles/systeme76/renforts/renfort-v318z.png) |
| V323 | 45 | 55 | 1,5 | 9,2 | ![Renfort V323](/assets/profiles/systeme76/renforts/renfort-v323z.png) |
| VSF01 | 11 | 21,4 | 2,5 | 1,08 | ![Renfort VSF01](/assets/profiles/systeme76/renforts/renfort-vsf01.png) |
| V337L | 35 | 45 | 2 | - | ![Renfort V337L](/assets/profiles/systeme76/renforts/renfort-v337-v339.png) |
| V337R | 35 | 45 | 2 | - | ![Renfort V337R](/assets/profiles/systeme76/renforts/renfort-v337-v339.png) |
| V339L | 35 | 45 | 2 | - | ![Renfort V339L](/assets/profiles/systeme76/renforts/renfort-v337-v339.png) |
| V339R | 35 | 45 | 2 | - | ![Renfort V339R](/assets/profiles/systeme76/renforts/renfort-v337-v339.png) |
| V329 | 32,5 | 28 | 1,25 | 2,15 | ![Renfort V329](/assets/profiles/systeme76/renforts/renfort-v329.png) |

(schéma: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf, p. 21 ; même planche, mêmes inerties et
mêmes affectations : raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 20)

Pour chacun des treize renforts qui figurent aussi dans les tableaux par profilé ci-dessus,
l'inertie portée est la valeur de la colonne IW. Les V337L, V337R, V339L et V339R sont dessinés
sans inertie ni profilé. Le VSF01 est une cornière de 21,4 mm de haut à retour de 11 mm, en tôle de
2,5 mm ; il se visse à l'extérieur du battement 76473 — prescription dans
[DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md), section 2.2.3.4. Le **V291** y est coté
**42 × 30 mm**, contre 43 × 29 mm sur la planche des profilés principaux, et le **V317**
25 × 40 mm, comme sur la planche des profilés principaux et non 24 — entrée **CTR-23**.

Affectation des renforts aux profilés, une ligne par couple renfort et profilé :

| Renfort | Profilé désigné « pour profilé(s) » |
| --- | --- |
| V306 | 76101 |
| V306 | 76171 |
| V306 | 76173 |
| V306 | 76180 |
| V306 | 76271 |
| V307 | 76101 |
| V307 | 76171 |
| V307 | 76173 |
| V307 | 76180 |
| V307 | 76271 |
| V309 | 76101 |
| V309 | 76171 |
| V309 | 76173 |
| V309 | 76180 |
| V314 | 76102 |
| V314 | 76172 |
| V314 | 76272 |
| V314 | 76279 |
| V326 | 76102 |
| V326 | 76172 |
| V326 | 76272 |
| V326 | 76279 |
| V291 | 76177 |
| V291 | 76178 |
| V291 | 76185 |
| V266 | 76281 |
| V266 | 76274 |
| V266 | 76275 |
| V266 | 76276 |
| V312 | 76300 |
| V316 | 76471 |
| V317 | 76472 |
| V320 | 76301 |
| V318 | 76372 |
| V323 | 76303 |
| V323 | 76373 |
| VSF01 | 76473 |
| V329 | 76101 |
| V329 | 76171 |
| V329 | 76173 |
| V329 | 76180 |

(schéma: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf, p. 21 ; même planche, mêmes inerties et
mêmes affectations : raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 20)

Cette affectation ne concorde pas partout avec celle du manuel de mise en œuvre : le **76172**
reçoit ici le V314 ou le V326 (V314.Z, V325 et V353 dans le manuel, pas de V326), le **V329**
équipe les dormants 76101, 76171, 76173 et 76180 (aucun V329 dans le manuel), et les V308, V310,
V319, V322, V324, V325 et V353 n'y sont pas dessinés [5 p. 21] — entrée **CTR-21**. Les V306 et
V307 équipent le 76101 et l'ouvrant 76271 dans les deux documents.

# Renforts dessinés dans chaque profilé

Sur les deux planches, chaque coupe porte dans sa chambre centrale la liste des renforts qu'elle
admet. Une ligne par profilé ; les renforts sont écrits dans l'ordre de la
planche. Les cotes des profilés sont dans
[Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md).

| Profilé | Famille | Renforts écrits dans la coupe |
| --- | --- | --- |
| 76171 | dormant | V306.Z, V307.Z, V308, V309.Z, V310 |
| 76172 | dormant | V314.Z, V325, V353 |
| 76173 | dormant, deux chambres | V306.Z, V307.Z, V308, V309.Z, V310, dans chacune des deux chambres |
| 76180 | dormant | V306.Z, V307.Z, V308, V309.Z, V310 |
| 76177 | dormant | V291.Z |
| 76185 | dormant | V291.Z |
| 76178 | dormant | V291.Z |
| 76101 | cadre fixe | V306.Z, V307.Z, V308, V309.Z, V310 |
| 76102 | cadre fixe | V314.Z, V325, V353 |
| 76281 | ouvrant | V266.Z |
| 76275 | ouvrant | V266.Z |
| 76274 | ouvrant | V266.Z |
| 76276 | ouvrant | V266.Z |
| 76271 | ouvrant | V306.Z, V307.Z, V308 |
| 76272 | ouvrant | V314.Z, V326.Z, V337 R/L, V339 R/L, V353 |
| 76279 | ouvrant | V314.Z, V326.Z, V337 R/L, V339 R/L, V353 |
| 76283 | ouvrant | V314.Z, V326.Z, V337 R/L, V339 R/L, V353 |
| 76372 | meneau de dormant | V318.Z, V319 |
| 76373 | meneau de dormant | V323.Z, V322, V324 |
| 76300 | traverse d'ouvrant | V312.Z |
| 76301 | traverse d'ouvrant | V320.Z |
| 76303 | traverse d'ouvrant | V323.Z, V322 |
| 76299 | traverse d'ouvrant | V323.Z, V322 |
| 76471 | battement | V316 |
| 76472 | battement | V317 |
| 76473 | battement | aucun |
| 76201 | profilé | V306.Z, V307.Z, V308 |
| 76206 | profilé | V260, V333R/L, V335R/L |
| 76207 | profilé | V260, V333R/L, V335R/L |
| 76401 | profilé | V316 |
| 76402, 76404 | profilé | V310 |

| 76701 | élargisseur | V312.Z |
| 76702 | élargisseur | V314.Z |
| 76703 | élargisseur | V314.Z, dans chacune des deux chambres |
| 76704 | réhausse | V114 |
| 76706 | réhausse | V114 |
| 76714 | profilé de 150 mm | V115, dans chacune des deux chambres |
| 76705 | réhausse | V317, dans chacune des trois chambres |
| 76802 | profilé | V332 |
| 76605 | liaison en H | V330 / V331 |
| 76608 | liaison en H | V288 |
| A250 | liaison aluminium | V264, deux fois |
| 8355 | poteau d'angle | V263 |
| 8356 | poteau d'angle | V262 |
| 8340 | poteau d'angle | V265 |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1 ; raw/poster-systeme-76-advanced-complementaires-2022.pdf, p. 1)

Les lignes des élargisseurs aux poteaux viennent de la planche des profilés complémentaires ; les
76700, 76708, 76709 et 76713 y sont dessinés sans renfort. Pour les élargisseurs, la planche et le
cahier PERFORM76 donnent V312.Z au 76701 et V314.Z aux 76702 et 76703, là où le tableau des
profilés complémentaires plus haut donne V312.Z aux 76700 à 76702 et V114 au 76703 — entrée
**CTR-22**. Les renforts V115, V288, V299, V407 et V167 ne sont dessinés sur aucune des deux
planches.

Cette liste ne concorde pas avec le tableau des renforts à inertie unique pour cinq profilés :
76206 (V260, V333, V335 ici, V323.Z et V322 dans les tableaux des profilés complémentaires),
76102 (V325 ici, V326 dans le tableau du DTA), 76101 (V308 et V310 ici, V329 dans le tableau du
DTA), 76303 (pas de V324 ici), 76172 (V325 et V353 ici). Pour les dormants, les ouvrants, les
battements et les cadres, elle concorde avec les planches du manuel de mise en œuvre, qui
confirment en plus le V324 du 76303 [1 p. 19-35] et donnent au 76206 de porte d'entrée les V260,
V333R/L et V335R/L [1 p. 9]. **Aucune des
deux listes n'est retenue contre l'autre** — entrée **CTR-21** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md). La planche confirme
en revanche les trois renforts du meneau 76373, V323.Z, V322 et V324, que **VER-23** mettait en
doute.

# Vissage des renforts

La fixation des renforts acier dans les chambres des profilés PVC s'effectue au moyen de **vis auto-perforeuses à tête fraisée** (norme DIN 7504 P) [1 registre 2.4.1 p. 1 à 9].

### Dimensions des vis selon renfort

| Renfort | Type et dimension de vis | Usage profilé |
| --- | --- | --- |
| **V312.Z** | **3,9 × 13 mm** auto-perforeuse à tête fraisée | Croisillons 76300, élargisseurs 76700-76702 |
| **Tous les autres renforts** (V306.Z, V307.Z, V308, V309.Z, V310, V314.Z, V326.Z, V266.Z, V316, V317, V318.Z, V319, V322, V323.Z, V324, V325, etc.) | **3,9 × 16 mm** auto-perforeuse à tête fraisée | Dormants, ouvrants, meneaux, battements et profilés complémentaires |

### Règles et entraxes de fixation en atelier

* **Pas de vissage maximal (entraxe entre vis)** :
  * **Profilés blancs** : entraxe maximal de **300 mm**.
  * **Profilés couleur ou plaxés** (1 ou 2 faces) : entraxe maximal ramené à **250 mm** en raison des contraintes thermiques et des gradients d'échauffement accrus.
* **Distance aux extrémités coupées** :
  * La première vis à chaque extrémité doit impérativement être positionnée entre **20 mm et 50 mm** de l'extrémité coupée de la barre d'acier ou du fond de feuillure soudé.
  * *Règle critique* : interdiction de visser à moins de 20 mm du bord (risque d'éclatement de la chambre PVC et interférence mécanique avec le miroir de soudage ou les ébavureuses d'angle).
* **Nombre minimal de vis** :
  * Tout tronçon de renfort doit comporter **au minimum 3 vis** de fixation, y compris sur les profilés courts ou impostes.
* **Axe et guidage de perçage** :
  * Le positionnement s'effectue dans la gorge de centrage coextrudée en fond de rainure de ferrure ou en fond de feuillure du profilé PVC, assurant la prise directe de la vis auto-foreuse dans l'épaisseur d'acier sans avant-trou.
* **Profilés de couleur et AluClip** :
  * Sur tout profilé filmé, laqué ou capoté aluminium, le vissage du renfort s'accompagne obligatoirement de la **ventilation des préchambres extérieures** (évacuation des calories emmagasinées) — voir [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

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
