---
type: Profilé
title: 'Vissage des renforts dans les profilés du système 76 Advanced : vis, cote de l''axe, paroi'
description: 'Fixation du renfort acier par vis auto-perforeuse à tête fraisée dans chaque profilé : cote de l''axe de la vis, paroi vissée et dimension de la vis, profilé par profilé, et schémas de vissage des dormants, ouvrants, battements, traverses d''ouvrant et meneaux.'
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

Cette page reprend, du manuel de mise en œuvre Système 76 Advanced de profine et des posters des profilés, le vissage des renforts dans les profilés. Les renforts admis par profilé, avec leurs inerties IW et IG, sont dans [Renforts acier du système 76 par profilé : IW, IG, dormants, ouvrants, meneaux](/profiles/systeme-76-renforts.md).

# Vissage des renforts

Le renfort acier glissé dans la chambre d'un profilé PVC y est fixé par des **vis
auto-perforeuses à tête fraisée** : la vis perce elle-même le PVC et l'acier, et sa tête conique
s'encastre dans la paroi. Pour chaque profilé du système 76 Advanced à joint central, la planche
de mise en œuvre dessine la coupe du profilé avec son renfort en place, la vis à sa position, la
cote qui situe l'axe de la vis, la dimension de la vis et les renforts qui se vissent de la même
façon [1 PDF p. 118-126, registre 2.4.1, p. 1 à 9].

Toutes les vis sont des 3,9 × 16 mm, sauf sur la traverse d'ouvrant 76300 (renfort V312.Z), où
la vis est une **3,9 × 13 mm** [1 PDF p. 125, registre 2.4.1, p. 8].

L'entraxe maximal entre vis (300 mm en blanc, 250 mm en couleur), la distance maximale du
vissage à l'onglet et le nombre minimal de trois vis par renfort sont des règles générales de
toutes les fenêtres profine : elles sont portées par
[Fabrication des profilés PVC, directives profine](/procedures/fabrication-profiles-pvc.md), et
la planche du système 76 n'en donne pas d'autre.

## Position de la vis, profilé par profilé

Chaque tableau donne, pour une famille de profilés, une ligne par dessin de la planche. La
**cote** est la distance, en mm, entre l'axe de la vis et le bord du profilé repéré sur le
dessin (bord droit ou bord gauche de la coupe, telle qu'elle est imprimée) ; la **paroi vissée**
est la paroi de la chambre, en haut ou en bas du dessin, que traverse la vis ; la colonne des
renforts reprend les légendes imprimées à côté de la coupe, dans l'ordre de la planche. Un
profilé dessiné deux ou trois fois correspond à deux ou trois façons de loger le renfort :
ouverture du renfort en U tournée vers le haut ou vers le bas du dessin, chambre inférieure ou
supérieure. Échelle des planches : 1:2.

**Dormants.** Le dormant est le cadre fixe scellé dans la maçonnerie.

| Profilé | Cote de la vis (mm) | Bord de référence de la cote | Paroi vissée | Position du renfort sur le dessin | Renforts légendés | Vis (mm) | Page PDF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 76171 | 25 | droit | bas | U ouvert vers le haut | V306.Z, V307.Z, V308, V309.Z, V310 | 3,9 × 16 | 118 |
| 76171 | 46 | droit | bas | U ouvert vers le bas | V306.Z, V307.Z, V308 | 3,9 × 16 | 118 |
| 76172 | 25 | droit | bas | tube fermé | V314.Z, V353, V325 | 3,9 × 16 | 118 |
| 76173 | 25 | droit | bas, chambre inférieure | U ouvert vers le haut | V306.Z, V307.Z, V308, V309.Z, V310 | 3,9 × 16 | 119 |
| 76173 | 46 | droit | bas, chambre inférieure | U ouvert vers le bas | V306.Z, V307.Z, V308 | 3,9 × 16 | 119 |
| 76173 | 25 | droit | haut, chambre supérieure | U ouvert vers le bas | V306.Z, V307.Z, V308, V309.Z, V310 | 3,9 × 16 | 119 |
| 76180 | 30 | droit | bas | U ouvert vers le haut | V306.Z, V307.Z, V308, V309.Z, V310 | 3,9 × 16 | 120 |
| 76180 | 45 | droit | bas | U ouvert vers le bas | V306.Z, V307.Z, V308 | 3,9 × 16 | 120 |
| 76177 | 46 | droit | bas | renfort coudé | V291.Z | 3,9 × 16 | 120 |
| 76178 | 46 | droit | bas | renfort coudé | V291.Z | 3,9 × 16 | 121 |
| 76185 | 46 | droit | bas | renfort coudé | V266.Z | 3,9 × 16 | 121 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 118-121)

Le dormant rénovation 76185 est légendé **V266.Z** sur la planche de vissage, alors que les
planches des profilés et le tableau des renforts plus haut lui donnent le V291.Z — entrée
**INC-55** du registre [Incohérences internes](/anomalies/incoherences-internes.md). Les deux
valeurs restent affichées, aucune n'est retenue.

**Ouvrants.** L'ouvrant est le cadre mobile qui porte le vitrage.

| Profilé | Cote de la vis (mm) | Bord de référence de la cote | Paroi vissée | Position du renfort sur le dessin | Renforts légendés | Vis (mm) | Page PDF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sans référence (1er dessin) | 32 | gauche | haut | renfort en L | aucun | 3,9 × 16 | 122 |
| sans référence (2e dessin) | 39 | gauche | haut | renfort en L | aucun | 3,9 × 16 | 122 |
| 76271 | 32 | gauche | haut | renfort en L | V306.Z, V307.Z, V308 | 3,9 × 16 | 123 |
| 76274 | 39 | gauche | haut | renfort en L | V266.Z | 3,9 × 16 | 123 |
| 76276 | 25 | droit | haut | trois U ouverts vers le bas | V266.Z | 3,9 × 16 | 123 |
| 76272 | 25 | droit | haut | tube fermé | V314.Z, V353, V337/V339 D/G | 3,9 × 16 | 124 |
| 76279 | 25 | droit | haut | tube fermé | V314.Z, V353, V337/V339 D/G | 3,9 × 16 | 124 |
| 76283 | 28 | droit | haut | tube fermé | V314.Z, V353, V337/V339 D/G | 3,9 × 16 | 124 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 122-124)

La page 122 (registre 2.4.1, p. 5) dessine deux ouvrants sans aucune référence de profilé ni de
renfort ; leurs coupes et leurs cotes (32 et 39 mm) sont celles des deux premiers dessins de la
page 123. Sur la page 123, les légendes et les dessins ne concordent pas : le 76271 porte les
légendes V306.Z, V307.Z et V308 sous un seul renfort dessiné en L, et le 76276 porte la légende
V266.Z sous trois renforts dessinés en U — entrée **INC-56**. Les renforts par profilé restent
ceux du tableau *Renforts à inertie unique* plus haut (76271 : V306.Z, V307.Z, V308 ; 76276 :
V266.Z) ; la planche de vissage ne suffit pas à les confirmer.

**Battements, traverses d'ouvrant et meneaux.** Le battement est le profilé rapporté sur
l'ouvrant d'une fenêtre à deux vantaux, qui ferme le jeu entre les deux vantaux ; la traverse
d'ouvrant et le meneau sont les profilés intermédiaires qui recoupent un ouvrant ou un dormant.

| Profilé | Cote de la vis (mm) | Bord de référence de la cote | Paroi vissée | Position du renfort sur le dessin | Renforts légendés | Vis (mm) | Page PDF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 76471 | 22 | gauche | haut | renfort en C | V316 | 3,9 × 16 | 125 |
| 76472 | 22 | gauche | haut | tube fermé | V317 | 3,9 × 16 | 125 |
| 76300 | 25 | droit | bas | tube fermé plat | V312.Z | 3,9 × 13 | 125 |
| 76303 | 25 | droit | bas | tube fermé | V323.Z, V322 | 3,9 × 16 | 126 |
| 76372 | 25 | droit | bas | tube fermé | V318.Z, V319 | 3,9 × 16 | 126 |
| 76373 | 25 | droit | bas | tube fermé | V323.Z, V322 | 3,9 × 16 | 126 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 125-126)

Le battement 76473, qui ne reçoit pas de renfort, n'est pas dessiné sur la planche de vissage. Le
renfort V324 du meneau 76373 n'y figure pas non plus : la planche ne dessine que les V323.Z et
V322 pour ce meneau.

## Schémas de vissage

Chaque schéma montre la coupe du profilé, le renfort acier hachuré dans sa chambre, la vis
dessinée à sa place avec sa tête fraisée côté paroi, la cote qui situe l'axe de la vis, et, à
droite, le dessin de chaque renfort qui se visse de la même façon avec sa légende. La référence du
profilé et la dimension de la vis sont écrites sous la coupe.

### Dormants

Vissage du dormant 76171, renfort en U ouvert vers le haut, vis à 25 mm du bord droit :

![Vissage du renfort dans le dormant 76171, vis à 25 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76171-25.png)

Vissage du dormant 76171, renfort en U ouvert vers le bas, vis à 46 mm du bord droit :

![Vissage du renfort dans le dormant 76171, vis à 46 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76171-46.png)

Vissage du dormant 76172, renfort tubulaire, vis à 25 mm :

![Vissage du renfort dans le dormant 76172, vis à 25 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76172-25.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 118)

Vissage du dormant 76173 dans sa chambre inférieure, renfort en U ouvert vers le haut, vis à
25 mm :

![Vissage du renfort dans le dormant 76173, chambre inférieure, vis à 25 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76173-25-bas.png)

Vissage du dormant 76173 dans sa chambre inférieure, renfort en U ouvert vers le bas, vis à
46 mm :

![Vissage du renfort dans le dormant 76173, vis à 46 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76173-46.png)

Vissage du dormant 76173 dans sa chambre supérieure, vis dans la paroi du haut à 25 mm :

![Vissage du renfort dans le dormant 76173, chambre supérieure, vis à 25 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76173-25-haut.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 119)

Vissage du dormant 76180, renfort en U ouvert vers le haut, vis à 30 mm ; puis renfort en U
ouvert vers le bas, vis à 45 mm :

![Vissage du renfort dans le dormant 76180, vis à 30 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76180-30.png)

![Vissage du renfort dans le dormant 76180, vis à 45 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76180-45.png)

Vissage des dormants rénovation 76177, 76178 et 76185, renfort coudé, vis à 46 mm :

![Vissage du renfort V291.Z dans le dormant 76177](/assets/procedures/moe-76-advanced/vissage-renfort-76177-46.png)

![Vissage du renfort V291.Z dans le dormant 76178](/assets/procedures/moe-76-advanced/vissage-renfort-76178-46.png)

![Vissage du renfort légendé V266.Z dans le dormant 76185](/assets/procedures/moe-76-advanced/vissage-renfort-76185-46.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 120-121)

### Ouvrants

Les deux ouvrants sans référence de la page 122, vis à 32 et 39 mm du bord gauche :

![Vissage d'un renfort en L, ouvrant sans référence, vis à 32 mm](/assets/procedures/moe-76-advanced/vissage-renfort-sans-reference-32.png)

![Vissage d'un renfort en L, ouvrant sans référence, vis à 39 mm](/assets/procedures/moe-76-advanced/vissage-renfort-sans-reference-39.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 122)

Vissage des ouvrants 76271 (32 mm), 76274 (39 mm) et 76276 (25 mm) ; légendes en désaccord avec
les dessins, entrée **INC-56** :

![Vissage du renfort dans l'ouvrant 76271, vis à 32 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76271-32.png)

![Vissage du renfort V266.Z dans l'ouvrant 76274, vis à 39 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76274-39.png)

![Vissage du renfort dans l'ouvrant 76276, vis à 25 mm](/assets/procedures/moe-76-advanced/vissage-renfort-76276-25.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 123)

Vissage des ouvrants de 110 mm 76272, 76279 et 76283, renfort tubulaire, vis dans la paroi du
haut à 25, 25 et 28 mm :

![Vissage du renfort dans l'ouvrant 76272](/assets/procedures/moe-76-advanced/vissage-renfort-76272-25.png)

![Vissage du renfort dans l'ouvrant 76279](/assets/procedures/moe-76-advanced/vissage-renfort-76279-25.png)

![Vissage du renfort dans l'ouvrant 76283](/assets/procedures/moe-76-advanced/vissage-renfort-76283-28.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 124)

### Battements, traverses d'ouvrant et meneaux

Vissage des battements 76471 (V316) et 76472 (V317), vis à 22 mm du bord gauche, et de la
traverse d'ouvrant 76300 (V312.Z), vis de 3,9 × 13 mm à 25 mm :

![Vissage du renfort V316 dans le battement 76471](/assets/procedures/moe-76-advanced/vissage-renfort-76471-22.png)

![Vissage du renfort V317 dans le battement 76472](/assets/procedures/moe-76-advanced/vissage-renfort-76472-22.png)

![Vissage du renfort V312.Z dans la traverse 76300](/assets/procedures/moe-76-advanced/vissage-renfort-76300-25.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 125)

Vissage de la traverse d'ouvrant 76303 et des meneaux 76372 et 76373, vis à 25 mm :

![Vissage du renfort dans la traverse 76303](/assets/procedures/moe-76-advanced/vissage-renfort-76303-25.png)

![Vissage du renfort dans le meneau 76372](/assets/procedures/moe-76-advanced/vissage-renfort-76372-25.png)

![Vissage du renfort dans le meneau 76373](/assets/procedures/moe-76-advanced/vissage-renfort-76373-25.png)

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 126)

Les planches du registre 2.4.1 portent des dates de révision différentes : janvier 2016 (p. 1, 2,
5 à 8), juillet 2017 (p. 3), mars 2021 (p. 4 et 9).

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

Cette page est une partie d'un sujet qui en compte 4 ; les autres :

- [Renforts acier du système 76 par profilé : IW, IG, dormants, ouvrants, meneaux](/profiles/systeme-76-renforts.md)
- [Sections et inerties des renforts du système 76 : largeur, hauteur, épaisseur, IG, IW](/profiles/systeme-76-renforts-sections-et-inerties.md)
- [Renforts du système 76 à inertie unique et renforts écrits dans la coupe de chaque profilé](/profiles/systeme-76-renforts-profiles-equipes.md)
