---
type: Procédure
title: Drainage, décompression et ventilation du système 76 Advanced
description: Les perçages de drainage, de décompression et de ventilation des profilés sombres du système 76 Advanced à joint central, profilé par profilé, avec leurs angles, leurs cotes et leurs dimensions d'usinage.
tags: [systeme-76-advanced, profine, joint-central, drainage, decompression, ventilation, couleur, usinage, atelier]
systeme: 76
fournisseur: KÖMMERLING
usage: atelier
famille: drainage
status: stable
sources:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
    last_modified: 2025-06-19
source_pages:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    pages: 127-134
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    pages: 4, 6, 7
generated:
  by: process:claude-code
  at: 2026-09-25T12:00:00Z
---

# Ce que fait cette procédure

Elle situe, profilé par profilé, les usinages qui laissent sortir l'eau et l'air d'une fenêtre du
**système 76 Advanced à joint central** de [profine](/fournisseurs/profine.md) :

- le **drainage** est le trou oblong qui évacue vers l'extérieur l'eau entrée dans la feuillure
  (le logement du vitrage ou de l'ouvrant) ;
- la **décompression** est l'ouverture qui équilibre la pression d'air entre la feuillure et
  l'extérieur, pour que l'eau ne soit pas retenue ;
- la **ventilation** est le perçage des préchambres extérieures (les chambres du profilé côté
  extérieur) des profilés sombres, pour qu'elles n'accumulent pas la chaleur du soleil.

Les positions des usinages le long des barres (distances aux coins, entraxes, nombre de trous)
sont celles du registre 1.3.1 des directives générales, reprises dans
[Drainage, décompression et vitrage, directives générales profine](/procedures/drainage-et-vitrage-generaux.md).
Cette page donne ce que le système 76 ajoute : les angles de perçage, les cotes qui placent chaque
trou sur la coupe et les chambres à ne pas percer [1 PDF p. 127, registre 2.4.2, p. 1]. Les
schémas du DTA (couleurs sombres, capotage) sont dans
[Fabrication et assemblage du système 76 Advanced](/procedures/fabrication-systeme-76-advanced.md).

# Conditions et interdictions

« Les positions des usinages figurent dans le registre 1.3.1. Sur la gamme 76, veillez aux
détails suivants » [1 PDF p. 127, registre 2.4.2, p. 1] :

- **Sur le dormant, l'ouvrant et le meneau, la chambre repérée en gris sur chaque coupe « ne doit
  pas être endommagée par l'usinage du trou de drainage ».** C'est la chambre qui borde en biais
  le fond de feuillure, que le perçage incliné longe sans l'entamer.
- **« Pour éviter toute accumulation de chaleur, les préchambres extérieures des profilés de
  couleur, laqués ou filmés 2 faces ou 1 face extérieure doivent impérativement être
  ventilées. »** Les préchambres concernées sont marquées d'un astérisque sur les coupes
  [1 PDF p. 128 et 131, registre 2.4.2, p. 2 et 5].
- **Alternative au trou oblong** : « Pour des profondeurs de perçages supérieures à 50 mm les trous
  oblongs peuvent être remplacés par 3 trous de Ø 6 », trois trous alignés sur 25 mm de long et
  6 mm de haut [1 PDF p. 128, registre 2.4.2, p. 2].

Le [DTD n° DBV-25-6/16-2334_V5](/sources/dtd-6-16-2334.md) ne raisonne pas en « couleurs » mais
en **clarté colorimétrique L\*** (valeur mesurée de la clarté d'une teinte, de 0 pour le noir à
100 pour le blanc), avec un seuil unique, **L\* inférieur à 82** :

| Prescription du DTD | Ce qu'elle impose |
| --- | --- |
| Profilé PVC revêtu d'un film ou d'une laque à L\* < 82 | renfort obligatoire |
| Profilé PVC revêtu d'un capotage aluminium à L\* < 82 | renfort obligatoire |
| Chambres des profilés à L\* < 82 communiquant avec l'extérieur | décompression par orifices de **Ø 5 mm minimum** |
| Habillage monoparoi à L\* < 82 ou non défini | **interdit en traverse basse**, quelle que soit la technologie de coloration |

(prescriptions : raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 4, 6 et 7 ; texte complet au
§ 2.2.3, 2.2.3.4 et 2.4 de [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md))

# Étapes

## Chambre à ne pas percer

Trois coupes, un dormant, un ouvrant et un meneau à joint central, dessinées sans référence : sur
chacune, la chambre grisée, en biais contre le fond de feuillure, « ne doit pas être endommagée
par l'usinage du trou de drainage ». Le trait fin qui traverse la coupe est l'axe du perçage de
drainage.

![Chambre à ne pas endommager, dormant](/assets/procedures/moe-76-advanced/drainage-chambre-a-preserver-1.png)

![Chambre à ne pas endommager, ouvrant](/assets/procedures/moe-76-advanced/drainage-chambre-a-preserver-2.png)

![Chambre à ne pas endommager, meneau](/assets/procedures/moe-76-advanced/drainage-chambre-a-preserver-3.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 127, registre 2.4.2, p. 1, version janvier 2016)

## Drainage du dormant

La traverse basse du dormant se draine de deux façons, dessinées côte à côte : **par l'avant**,
un trou oblong horizontal débouchant sur la face extérieure, ou **par le bas**, un trou oblong
vertical débouchant sous le dormant. Dans les deux cas, un premier trou oblong de 5 × 25 mm est
percé en biais, à **40°**, depuis le fond de feuillure. Le pictogramme de fenêtre en haut à droite
situe la coupe : traverse basse du dormant, deux flèches.

Sur chaque coupe, les trous sont en gris foncé ; la cote de 44 mm (38 mm sur les dormants
rénovation) est portée entre le haut du profilé et le trou avant ; la cote de 6 mm est portée
entre la face extérieure et le trou par le bas ; la cote verticale (29, 46 ou 58 mm) est portée le
long du trou par le bas. Les astérisques repèrent les préchambres à ventiler sur un profilé de
couleur.

| Dormant | Perçage en biais | Drainage par l'avant : cote (mm) | Drainage par l'avant : trou | Drainage par le bas : cote depuis la face (mm) | Drainage par le bas : cote le long du trou (mm) | Drainage par le bas : trou | Préchambres marquées * | Page PDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 76171 | 40°, 5 × 25 mm | 44 | 5 × 25 mm | 6 | 29 | 5 × 25 mm | 1 | 128 |
| 76172 | 40°, 5 × 25 mm | 44 | 5 × 25 mm | 6 | 46 | 5 × 25 mm ou 3 × Ø 6 | 2 | 128 |
| 76173 | 40°, 5 × 25 mm | 44 | 5 × 25 mm | 6 | 58 | 5 × 25 mm ou 3 × Ø 6 | 3 | 129 |
| 76180 | 40°, 5 × 25 mm | 44 | 5 × 25 mm | - | - | - | aucune | 129 |
| 76177, 76178, 76185 | 40°, 5 × 25 mm | 38 | 5 × 25 mm | - | - | - | aucune | 129 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 128-129, registre 2.4.2, p. 2 et 3, version janvier 2016)

Le dormant 76180 et les dormants rénovation 76177, 76178 et 76185 ne sont dessinés qu'avec le
drainage par l'avant. Le 76172 porte l'alternative « ou 3 × Ø6 » alors que sa cote le long du
trou est de 46 mm, sous le seuil de 50 mm de l'alternative — entrée **VER-57** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

![Drainage du dormant 76171, par l'avant et par le bas](/assets/procedures/moe-76-advanced/drainage-dormant-76171.png)

![Drainage du dormant 76172, par l'avant et par le bas, alternative 3 × Ø 6](/assets/procedures/moe-76-advanced/drainage-dormant-76172.png)

![Drainage du dormant 76173, par l'avant et par le bas](/assets/procedures/moe-76-advanced/drainage-dormant-76173.png)

![Drainage par l'avant des dormants 76180, 76177, 76178 et 76185](/assets/procedures/moe-76-advanced/drainage-dormant-76180-76177-76178-76185.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 128-129)

## Drainage de l'ouvrant

La traverse basse de l'ouvrant se draine par un trou oblong de 5 × 25 mm percé en biais à
**50°** depuis la feuillure, qui rejoint un trou oblong vertical de 5 × 25 mm débouchant sous
l'ouvrant, à 6 mm de la face extérieure. La cote verticale est portée le long du trou vertical
[1 PDF p. 130, registre 2.4.2, p. 4].

| Ouvrant | Perçage en biais | Cote depuis la face (mm) | Cote le long du trou vertical (mm) | Trou vertical | Page PDF |
| --- | --- | --- | --- | --- | --- |
| 76281 / 76275 / 76274 / 76276 | 50°, 5 × 25 mm | 6 | 7 | 5 × 25 mm | 130 |
| 76271 | 50°, 5 × 25 mm | 6 | 8 | 5 × 25 mm | 130 |
| 76272 / 76279 | 50°, 5 × 25 mm | 6 | 43 | 5 × 25 mm | 130 |
| 76283 | 50°, 5 × 25 mm | 6 | 71 | 5 × 25 mm ou 3 × Ø 6 | 130 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 130, registre 2.4.2, p. 4, version janvier 2016)

![Drainage des ouvrants 76281, 76275, 76274, 76276, 76271, 76272, 76279 et 76283](/assets/procedures/moe-76-advanced/drainage-ouvrants.png)

Les coupes sont regroupées comme sur la planche : en haut, les ouvrants 76281 / 76275 / 76274 /
76276 ; au milieu, le 76271 ; en bas, les 76272 / 76279 et, à droite, le 76283 à ouverture
extérieure avec l'alternative des trois trous de Ø 6.

## Drainage de la traverse de dormant (meneau 76372)

La traverse intermédiaire de dormant se draine comme le dormant : perçage en biais à **40°**,
5 × 25 mm, puis **par l'avant** (trou oblong de 5 × 25 mm, cote de 44 mm depuis le haut du
profilé) ou **par le bas** (trou oblong de 5 × 25 mm à 6 mm de la face extérieure, cote de 50 mm le
long du trou). Le profilé dessiné, sous le titre « Traverse », est le **76372** ; ses deux
préchambres extérieures portent l'astérisque de ventilation des profilés de couleur
[1 PDF p. 131, registre 2.4.2, p. 5].

![Drainage de la traverse 76372, par l'avant et par le bas](/assets/procedures/moe-76-advanced/drainage-meneau-76372.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 131, registre 2.4.2, p. 5, version janvier 2016)

## Drainage sur fixe dormant ou traverse

Sur un vitrage fixe posé directement dans le dormant ou dans la traverse, l'eau de la feuillure
descend par les trous grisés, suivant les flèches : dans le dormant, par la chambre avant puis
sous le profilé ; dans la traverse, par la chambre avant puis par le bas de la traverse.

![Drainage sur fixe dormant ou traverse](/assets/procedures/moe-76-advanced/drainage-fixe-dormant-ou-traverse.png)

**« Réaliser un fraisage de 25 mm à 100 mm de chaque extrémité »** : sur la barre basse d'un fixe,
vue de face, un fraisage de 25 mm de long est fait à 100 mm de chaque extrémité intérieure, de
part et d'autre ; les fraisages sont les deux rectangles gris [1 PDF p. 131, registre 2.4.2, p. 5].

![Fraisage de 25 mm à 100 mm de chaque extrémité](/assets/procedures/moe-76-advanced/drainage-fraisage-joint-fixe.png)

| Usinage | Longueur du fraisage (mm) | Distance à l'extrémité (mm) | Nombre |
| --- | --- | --- | --- |
| Fraisage sur fixe dormant ou traverse | 25 | 100 | 1 à chaque extrémité |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 131)

## Ventilation du meneau sombre

Sur un meneau de couleur (profilé sombre), les deux préchambres extérieures sont ventilées par
des perçages de **Ø 5 mm** inclinés à **40°**, un de chaque côté de la coupe du **76372**. Le long
du meneau, vu de face, les perçages sont faits par paires, **à 200 mm de chaque extrémité** ; le
pictogramme situe les perçages en haut du meneau d'une fenêtre à deux vantaux
[1 PDF p. 132, registre 2.4.2, p. 6].

![Ventilation du meneau 76372, profilé sombre](/assets/procedures/moe-76-advanced/ventilation-meneau-76372.png)

| Profilé | Perçage | Angle | Distance à chaque extrémité (mm) | Perçages par position |
| --- | --- | --- | --- | --- |
| 76372 | Ø 5 | 40° | 200 | 2 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 132, registre 2.4.2, p. 6, version janvier 2016)

## Drainage de la traverse d'ouvrant

La traverse d'ouvrant, qui recoupe un ouvrant en deux champs vitrés, se draine par un trou
oblong de 5 × 25 mm percé en biais à **50°**, puis par un trou vertical de 5 × 25 mm à 6 mm de la
face extérieure [1 PDF p. 133, registre 2.4.2, p. 7].

| Traverse d'ouvrant | Perçage en biais | Cote depuis la face (mm) | Cote le long du trou vertical (mm) | Trou vertical | Page PDF |
| --- | --- | --- | --- | --- | --- |
| 76300 | 50°, 5 × 25 mm | 6 | 33 | 5 × 25 mm | 133 |
| 76301 | 50°, 5 × 25 mm | 6 | 44 | 5 × 25 mm | 133 |
| 76303 | 50°, 5 × 25 mm | 6 | 70 | 5 × 25 mm ou 3 × Ø 6 | 133 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 133, registre 2.4.2, p. 7, version janvier 2016)

![Drainage des traverses d'ouvrant 76300, 76301 et 76303](/assets/procedures/moe-76-advanced/drainage-traverses-ouvrant.png)

## Ventilation de la traverse d'ouvrant sombre

Sur une traverse d'ouvrant de couleur, les deux préchambres extérieures sont ventilées par des
perçages de **Ø 5 mm** inclinés à **50°**, un de chaque côté de la coupe, sur les **76300** et
**76301**. Le long de la traverse, les perçages sont faits par paires, **à 200 mm de chaque
extrémité** [1 PDF p. 134, registre 2.4.2, p. 8].

![Ventilation des traverses d'ouvrant 76300 et 76301, profilé sombre](/assets/procedures/moe-76-advanced/ventilation-traverses-ouvrant.png)

| Profilé | Perçage | Angle | Distance à chaque extrémité (mm) | Perçages par position |
| --- | --- | --- | --- | --- |
| 76300 | Ø 5 | 50° | 200 | 2 |
| 76301 | Ø 5 | 50° | 200 | 2 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 134, registre 2.4.2, p. 8, version janvier 2016)

## Ventilation de la traverse d'ouvrant 76303 sombre

La traverse d'ouvrant **76303** de couleur se ventile, comme les 76300 et 76301, par des perçages
de **Ø 5 mm** inclinés à **50°** dans ses deux préchambres extérieures, par paires, **à 200 mm de
chaque extrémité**. Elle demande en plus une mise à l'air de ses chambres en bout, selon l'une de
deux variantes [1 PDF p. 135, registre 2.4.2, p. 9, version janvier 2016].

**Variante 1 — usinage en bout.** Un **fraisage de 5 × 5 × ~75 mm** est fait en bout de la
traverse, sur la face coupée, en travers des chambres (bande grise de la vue en bout). La coupe
B-B, légendée « Usinage traverse d'ouvrant 76303 », cote ce fraisage : 5 mm de profondeur, 5 mm de
largeur, à 4 mm du bord. Sur la vue de face, le fraisage est dessiné en pointillé aux deux
extrémités de la traverse.

![Ventilation de la traverse 76303, variante 1, usinage en bout](/assets/procedures/moe-76-advanced/ventilation-traverse-76303-variante-1-usinage-en-bout.png)

**Variante 2 — perçage en bout.** « Perçage Ø 5 mm sous 40° jusqu'au milieu du profilé » : depuis
l'extrémité de la traverse, deux perçages de Ø 5 mm inclinés à **40°** remontent dans le profilé,
sur **50 mm** de long, jusqu'à se rejoindre au milieu (coupe A-A). Sur la vue de face, ils sont
dessinés en biais aux deux extrémités.

![Ventilation de la traverse 76303, variante 2, perçage en bout](/assets/procedures/moe-76-advanced/ventilation-traverse-76303-variante-2-percage-en-bout.png)

| Traverse 76303, profilé sombre | Usinage | Cotes (mm) | Angle |
| --- | --- | --- | --- |
| Perçages des préchambres | Ø 5, par paires | 200 de chaque extrémité | 50° |
| Variante 1, usinage en bout | fraisage | 5 × 5 × ~75 ; à 4 du bord | - |
| Variante 2, perçage en bout | Ø 5, jusqu'au milieu du profilé | 50 de long | 40° |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 135, registre 2.4.2, p. 9)

## Ventilation du battement 76473 sombre

Sur le battement **76473** de couleur, la ventilation passe par l'embout **M462.R** (la pièce
d'extrémité du battement) : le carré marqué d'une croix sur l'embout, à **22,5 mm** de son bord
droit et à **6,5 mm** de son bord haut, est le point de ventilation dessiné par la planche
[1 PDF p. 136, registre 2.4.2, p. 10, version janvier 2022].

![Ventilation du battement 76473 par l'embout M462.R](/assets/procedures/moe-76-advanced/ventilation-battement-76473-embout-m462r.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 136)

La planche ne dit pas si ce point est un perçage à faire en atelier ou une ouverture venue de
moulage sur l'embout.

## Décompression du dormant et du joint de frappe

**« La décompression du dormant se fait au milieu de chaque ouvrant ! »** Elle s'obtient en
coupant le joint de frappe : le détail « Décompression (Couper le joint) » montre la gorge du
joint vide, à côté du détail « Joint de frappe » où le joint est en place. Le pictogramme situe
les deux coupes sur une fenêtre à deux ouvrants superposés : **coupe A-A** en traverse haute du
dormant, **coupe B-B** au droit de la traverse intermédiaire entre les deux ouvrants. Les
astérisques repèrent les préchambres à ventiler sur un profilé de couleur
[1 PDF p. 137, registre 2.4.2, p. 11, version janvier 2016].

![Décompression du dormant, coupes A-A et B-B](/assets/procedures/moe-76-advanced/decompression-dormant-coupes-a-a-b-b.png)

Sur un **vitrage fixe**, la décompression est une **« décompression fixe »** : un perçage de
**Ø 6 mm** au travers du profilé, sous le joint de vitrage, côté extérieur. Sur un ouvrant, elle
reste la découpe du joint de frappe. La planche le montre sur deux fenêtres, un ouvrant au-dessus
d'un fixe (à gauche) et un fixe au-dessus d'un ouvrant (à droite)
[1 PDF p. 138, registre 2.4.2, p. 12, version janvier 2016].

![Décompression fixe Ø 6 et découpe du joint de frappe](/assets/procedures/moe-76-advanced/decompression-fixe-et-ouvrant.png)

| Élément | Décompression | Cote (mm) | Position |
| --- | --- | --- | --- |
| Dormant, face à un ouvrant | joint de frappe coupé | - | au milieu de chaque ouvrant |
| Vitrage fixe | décompression fixe, perçage | Ø 6 | côté extérieur, sous le joint de vitrage |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 137-138)

## Drainage des profilés du « Système 76 AD »

Les pages 139 à 142 portent l'en-tête « Système 76 AD » ou « Système 76 Advanced AD », et non
« Système 76 Advanced Joint Central » ; elles dessinent les dormants **76101** et **76102**, les
ouvrants **76201**, **76206** et **76207** (les profilés de la
[porte d'entrée du système 76 Advanced](/portes/systeme-76-advanced-porte-d-entree.md)) et les
traverses **76301** et **76303**. « Les positions des ouvertures figurent dans le registre 1.3.1.
Attention aux détails suivants du Système Advanced 76. » [1 PDF p. 139, registre 2.4.2, p. 13,
version janvier 2016]

**Chambre à ne pas percer.** Comme sur le joint central, la chambre grisée d'un dormant, d'un
ouvrant et d'un meneau « ne doit pas être endommagée par l'usinage du trou de drainage ».

![Système 76 AD : chambre à ne pas endommager](/assets/procedures/moe-76-advanced/drainage-ad-chambre-a-preserver.png)

**Ventilation ou drainage des préchambres.** Sur un dormant à **ouverture intérieure**, la
préchambre se ventile ou se draine par un trou oblong de **5 × 25 mm** percé à **50°** ; sur un
dormant à **ouverture extérieure**, les deux préchambres marquées d'un astérisque sont
ventilées, sans perçage coté sur la planche. « Pour éviter toute accumulation de chaleur, les
préchambres extérieures des profilés de couleur, laqués ou filmés 2 faces ou 1 face extérieure
doivent impérativement être ventilées. »

![Système 76 AD : ventilation ou drainage des préchambres, ouverture intérieure et extérieure](/assets/procedures/moe-76-advanced/drainage-ad-ventilation-prechambres-ouverture-interieure-exterieure.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 139)

**Dormants, ouvrants et traverses.** Le drainage est percé en biais à **50°** (5 × 25 mm), puis
par l'avant (trou oblong horizontal, cote de 39 mm depuis le haut du profilé) ou par le bas (trou
oblong vertical à 6 mm de la face extérieure, cote portée le long du trou)
[1 PDF p. 140-142, registre 2.4.2, p. 14 à 16, version janvier 2016].

| Profilé (Système 76 AD) | Famille | Perçage en biais | Par l'avant : cote (mm) | Par le bas : cote depuis la face (mm) | Par le bas : cote le long du trou (mm) | Par le bas : trou | Préchambres marquées * | Page PDF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 76101 | dormant | 50°, 5 × 25 mm | 39 | 6 | 29 | 5 × 25 mm | 1 | 140 |
| 76102 | dormant | 50°, 5 × 25 mm | 39 | 6 | 46 | 5 × 25 mm ou 3 × Ø 6 | 2 | 140 |
| 76201 | ouvrant | 50°, 5 × 25 mm | - | 6 | 8 | 5 × 25 mm | aucune | 141 |
| 76206 | ouvrant | 50°, 5 × 25 mm | - | 6 | 63 | 5 × 25 mm ou 3 × Ø 6 | aucune | 141 |
| 76207 | ouvrant | 50°, 5 × 25 mm | - | 6 | 88 | 5 × 25 mm ou 3 × Ø 6 | aucune | 141 |
| 76301 | traverse | 50°, 5 × 25 mm | 39 | 6 | 44 | 5 × 25 mm | 2 | 142 |
| 76303 | traverse | 50°, 5 × 25 mm | 39 | 6 | 70 | 5 × 25 mm ou 3 × Ø 6 | 3 | 142 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 140-142)

Les ouvrants 76201, 76206 et 76207 ne sont dessinés qu'avec le drainage par le bas. Le 76102
porte, comme le 76172, l'alternative « ou 3 × Ø6 » avec une cote de 46 mm — entrée **VER-57**.

![Système 76 AD : drainage des dormants 76101 et 76102](/assets/procedures/moe-76-advanced/drainage-ad-dormants-76101-76102.png)

![Système 76 AD : drainage des ouvrants 76201, 76206 et 76207](/assets/procedures/moe-76-advanced/drainage-ad-ouvrants-76201-76206-76207.png)

![Système 76 AD : drainage des traverses 76301 et 76303](/assets/procedures/moe-76-advanced/drainage-ad-traverses-76301-76303.png)

## Ventilation des meneaux et traverses du « Système 76 AD » sombres

**Meneau 76301.** Sur un meneau de couleur, les deux préchambres extérieures du **76301** sont
ventilées par des perçages de **Ø 5 mm** inclinés à **50°**, par paires, **à 200 mm de chaque
extrémité** ; le pictogramme situe les perçages en haut du meneau d'une fenêtre à deux vantaux
[1 PDF p. 143, registre 2.4.2, p. 17, version janvier 2016].

![Système 76 AD : ventilation du meneau 76301 sombre](/assets/procedures/moe-76-advanced/ventilation-ad-meneau-76301.png)

**Meneau 76303.** « Ventilation pour meneau 76303 » : mêmes perçages de Ø 5 mm à 50°, par paires
à 200 mm de chaque extrémité, et, comme pour la traverse d'ouvrant 76303 du joint central, une
mise à l'air en bout selon deux variantes [1 PDF p. 144, registre 2.4.2, p. 18, version janvier
2016] :

- **Variante 1 — usinage en bout** : fraisage de **5 × 5 × ~75 mm** en bout (coupe A-A), coté sur
  le détail « Usinage traverse 76303 » à 5 mm de profondeur, 5 mm de largeur, 4 mm du bord ;
- **Variante 2 — perçage en bout** : « Perçage Ø 5 mm sous 40° jusqu'au milieu du profilé », sur
  **50 mm** de long (coupe A-A).

![Système 76 AD : ventilation du meneau 76303, variante 1, usinage en bout](/assets/procedures/moe-76-advanced/ventilation-ad-meneau-76303-variante-1-usinage-en-bout.png)

![Système 76 AD : ventilation du meneau 76303, variante 2, perçage en bout](/assets/procedures/moe-76-advanced/ventilation-ad-meneau-76303-variante-2-percage-en-bout.png)

Sur cette page, la coupe de la variante 1 est légendée « 76303 » mais la coupe A-A porte
« 76300 », et la coupe de la variante 2, du même dessin, est légendée « **76300** » — entrée
**INC-57** du registre [Incohérences internes](/anomalies/incoherences-internes.md).

**Traverse d'ouvrant 76300.** Drainage : perçage en biais à **50°**, 5 × 25 mm, puis trou
vertical de 5 × 25 mm à **6 mm** de la face extérieure, cote de **33 mm** le long du trou
(coupe A-A) ; « Idem pour profilés 76301, 76303 en temps que traverse d'ouvrant ». Ventilation du
profilé sombre (coupe B-B) : perçages de **Ø 5 mm** à **50°**, par paires, à **200 mm de chaque
extrémité** [1 PDF p. 145, registre 2.4.2, p. 19, version janvier 2016].

![Système 76 AD : drainage de la traverse d'ouvrant 76300](/assets/procedures/moe-76-advanced/drainage-ad-traverse-ouvrant-76300.png)

![Système 76 AD : ventilation de la traverse d'ouvrant 76300 sombre](/assets/procedures/moe-76-advanced/ventilation-ad-traverse-ouvrant-76300.png)

| Profilé (Système 76 AD), profilé sombre | Perçage | Angle | Distance à chaque extrémité (mm) | Mise à l'air en bout | Page PDF |
| --- | --- | --- | --- | --- | --- |
| 76301, meneau | Ø 5, par paires | 50° | 200 | - | 143 |
| 76303, meneau | Ø 5, par paires | 50° | 200 | fraisage 5 × 5 × ~75 mm, ou perçage Ø 5 à 40° sur 50 mm | 144 |
| 76300, traverse d'ouvrant | Ø 5, par paires | 50° | 200 | - | 145 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 143-145)

## Décompression du « Système 76 AD »

Les pages 146 et 147 reprennent, pour le « Système 76 Advanced AD », les planches de décompression
du joint central : **« La décompression du dormant se fait au milieu de chaque ouvrant ! »**, par
découpe du joint de frappe (coupes A-A et B-B d'une fenêtre à deux ouvrants superposés), et
**décompression fixe** par un perçage de **Ø 6 mm** sur un vitrage fixe, sur les deux fenêtres
ouvrant au-dessus d'un fixe et fixe au-dessus d'un ouvrant [1 PDF p. 146-147, registre 2.4.2,
p. 20 et 21, version janvier 2016].

![Système 76 AD : décompression du dormant, coupes A-A et B-B](/assets/procedures/moe-76-advanced/decompression-ad-dormant-coupes-a-a-b-b.png)

![Système 76 AD : décompression fixe Ø 6 et découpe du joint de frappe](/assets/procedures/moe-76-advanced/decompression-ad-fixe-et-ouvrant.png)

# Ce que le document ne dit pas

- La cote de drainage par le bas des dormants 76180, 76177, 76178 et 76185, dessinés seulement
  avec le drainage par l'avant
- Le repère exact à partir duquel se mesurent les cotes verticales du drainage par le bas : elles
  sont portées sur la coupe, sans légende

# Citations

[1] Mise en œuvre Système 76 Advanced, profine —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registre 2.4.2 « Drainage, décompression,
ventilation profilé sombre »

[2] [DTD n° DBV-25-6/16-2334_V5, système 76 Advanced](raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf), p. 4, 6 et 7

# Voir aussi

- [Drainage, décompression et vitrage, directives générales profine](/procedures/drainage-et-vitrage-generaux.md)
- [Fabrication et assemblage du système 76 Advanced](/procedures/fabrication-systeme-76-advanced.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md)
- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
