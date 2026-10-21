---
type: Profilé
title: Valeurs statiques Iz des meneaux, traverses et assemblages, système 70 (e.VOLUTION 2008)
description: Moments d'inertie Iz et Iy des renforts de meneaux et traverses, des assemblages de dormants, des assemblages par contreventements et avec élargisseurs du système 70, registre 4.2 « Statique » de 2008.
tags: [profine, systeme-70, statique, inertie, iw, ig, vent, fleche, renfort, meneau, traverse, plan-de-charge, allege, accouplement, contreventement]
systeme: 70
fournisseur: KÖMMERLING
usage: [atelier, chiffrage]
famille: statique
status: draft
sources:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008
    last_modified: 2008-08-31
source_pages:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    pages: 147-165
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 140, 157-192
generated:
  by: process:claude-code
  at: 2026-10-21T10:41:54Z
---

Cette page reprend, du manuel Mise en œuvre Système 70 Plateforme et du classeur e.VOLUTION d'août 2008, les valeurs statiques Iz et Iy du registre 4.2 « Statique » des plans de 2008. Le contexte commun est dans [Calcul statique du système 70 : méthode, formule générale, plans de charge et exemples](/profiles/systeme-70-statique-et-inerties.md).

# Valeurs statiques et exemple de calcul des plans e.VOLUTION de 2008

Le registre 4.2 « Statique » du classeur e.VOLUTION d'août 2008 (système F 91) donne, après le
texte sur l'exposition, les valeurs statiques des meneaux et des assemblages, puis un exemple de
calcul. Les valeurs y sont des Iz : moment d'inertie du renfort autour de l'axe qui travaille
quand le vent pousse sur la fenêtre, en cm⁴ ; l'Iy est l'inertie autour de l'autre axe (voir le
[glossaire](/reference/glossaire.md)). Planches à l'échelle 1:2, sauf l'exemple (dessins non à
l'échelle) [2 p. 157-170].

## Meneaux et traverses (2008)

Une ligne par meneau / traverse dessiné avec son renfort : code complet du profilé et du renfort
tel qu'imprimé, Iz et Iy du renfort, cotes portées sur la planche [2 p. 157-158].

| Profilé | Code imprimé du profilé | Renfort | Code imprimé du renfort | Iz (cm⁴) | Iy (cm⁴) | Cotes du profilé (mm) | Cotes du renfort (mm) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| 6127 | F91-15- 6127 | V603 | F00-40- V603 | 3,00 | 1,10 | 70 × 80, chambre 40 | 36 × 24, épaisseur 2 | ![Meneau 6127 et renfort V603](/assets/profiles/systeme70/statique/valeurs-statiques-meneau-6127-v603-evo2008.png) |
| 2427 | F95-15- 2427 | 9119 | F00-40- 9119 | 8,70 | 12,30 | 70 × 115, chambre 75 | 40 × 50, épaisseur 2 | ![Meneau 2427 et renfort 9119](/assets/profiles/systeme70/statique/valeurs-statiques-meneau-2427-9119-evo2008.png) |
| 2425 | F95-15- 2425 | 9132 | F00-40- 9132 | 9,10 | 3,30 | 70 × 90, chambre 50 | 48 × 25, épaisseur 2,5 | ![Meneau 2425 et renfort 9132](/assets/profiles/systeme70/statique/valeurs-statiques-meneau-2425-9132-evo2008.png) |
| 2469 | F95-15- 2469 | V081 | F00-40- V081 | 1,30 | 0,10 | 70 × 68, chambre 28 | 35 × 8, épaisseur 1,5 | ![Traverse 2469 et renfort V081](/assets/profiles/systeme70/statique/valeurs-statiques-meneau-2469-v081-evo2008.png) |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 157-158)

Les mêmes meneaux, avec leur Iz, figurent au registre 1.4 du même classeur : voir *Meneaux et
traverses des plans e.VOLUTION de 2008* sur [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md).

## Assemblages de dormants (2008)

Deux dormants identiques accouplés dos à dos, un isolant entre eux ; leurs renforts
s'additionnent en une « valeur statique ». Sur chaque coupe, les repères 1, 2 et 3 renvoient aux
remarques, (D) désigne le dormant [2 p. 159].

| Assemblage | Profilé | Renfort | Iz de chaque renfort (cm⁴) | Valeur statique (cm⁴) | Hauteur du dormant portée (mm) | Coupe |
| --- | --- | --- | --- | --- | --- | ---: |
| 2 × 6100 | 6100 | V600 | 2,10 + 2,10 | 4,20 | 110 | ![Assemblage de deux dormants 6100](/assets/profiles/systeme70/statique/assemblage-dormants-6100-v600-evo2008.png) |
| 2 × 6101 | 6101 | V601 | 3,20 + 3,20 | 6,40 | 128 | ![Assemblage de deux dormants 6101](/assets/profiles/systeme70/statique/assemblage-dormants-6101-v601-evo2008.png) |
| 2 × 2502 | 2502 | V030 | 4,50 + 4,50 | 9,00 | 170 | ![Assemblage de deux dormants 2502](/assets/profiles/systeme70/statique/assemblage-dormants-2502-v030-evo2008.png) |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 159)

Remarques de la planche, dans ses termes (titre imprimé « Remaques ») [2 p. 159] :

1. Assemblage des profilés : étancher côté intérieur et extérieur avec du silicone.
2. Partie centrale creuse à combler avec du compriband.
3. Assemblage par vis autoforeuses galvanisées diam mini 5,5 mm, entr'axe 6101/2502 = 400 mm,
   entr'axe 6100 = 300 mm (assemblage par boulon traversant diam 6 mm autorisé).
   **Attention** : la première vis doit être placée à 100 - 150 mm de l'intérieur dormant D.

Le schéma ci-dessous montre, en élévation, les deux dormants accouplés posés sur la traverse
basse (D) : la première vis à 100-150 mm, les suivantes à 400 mm (dessin non à l'échelle).

![Entraxe des vis d'assemblage de deux dormants](/assets/profiles/systeme70/statique/assemblage-dormants-entraxe-vis-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 159)

## Assemblages par contreventements (2008)

Un contreventement est une pièce rigide (fer plat ou profilé renforcé) placée entre deux dormants
6101 accouplés ; son inertie s'ajoute à celle des deux renforts V601 [2 p. 160-162].

| Assemblage | Contreventement | Renfort du contreventement | Iz V601 + V601 (cm⁴) | Iz du contreventement (cm⁴) | Valeur statique (cm⁴) | Cotes portées (mm) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| Assemblage 1 | 0481 | fer plat 10 × 80 | 3,20 + 3,20 | 42,70 | 49,10 | - | ![Assemblage 1 : 6101, 6101 et fer plat 10 × 80](/assets/profiles/systeme70/statique/contreventement-1-6101-0481-fer-plat-evo2008.png) |
| Assemblage 2 | 1184 | 9109 | 3,20 + 3,20 | 98,70 | 105,10 | - | ![Assemblage 2 : 6101, 6101 et 1184 / 9109](/assets/profiles/systeme70/statique/contreventement-2-6101-1184-9109-evo2008.png) |
| Assemblage 3 | K352 | K208 | 3,20 + 3,20 | 20,30 | 26,70 | - | ![Assemblage 3 : 6101, 6101 et K352 / K208](/assets/profiles/systeme70/statique/contreventement-3-6101-k352-k208-evo2008.png) |
| Assemblage 4 | 1338 | V250 | 3,20 + 3,20 | 71,60 | 78,00 | 14 ; 122 (70 + 52) ; 64,5 + 25 + 64,5 = 154 | ![Assemblage 4 : 6101, 6101 et 1338 / V250](/assets/profiles/systeme70/statique/contreventement-4-6101-1338-v250-evo2008.png) |
| Assemblage 5 | 1114 | V250 | 3,20 + 3,20 | 71,60 | 78,00 | 14 ; 126,5 (70 + 56,5) ; 64,5 + 25 + 64,5 = 154 | ![Assemblage 5 : 6101, 6101 et 1114 / V250](/assets/profiles/systeme70/statique/contreventement-5-6101-1114-v250-evo2008.png) |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 160-162)

Remarques des planches des assemblages 1 (p. 160) et 2 et 3 (p. 161), dans leurs termes ; elles
ne diffèrent que par le produit d'étanchéité du repère 1 [2 p. 160-161] :

1. Assemblage des profilés : étancher côte intérieur et extérieur avec du silicone (p. 160) ;
   avec du Ködisil BA-W (p. 161).
2. Partie centrale creuse à combler avec du compriband.
3. Assemblage par boulons galvas diam sup à 6 mm, entr'axe inf ou = à 400 mm.
   **Attention** : la première vis doit être placée à 100 - 150 mm de l'intérieur dormant D.
4. Liaison du raidisseur principal. Attention à l'interprétation du moment d'inertie total de ce
   genre de constructions. Ce moment d'inertie est optimal si : \*a) la liaison haute et basse du
   raidisseur principal au GO au moyen de pattes est assurée ; \*b) ou si la liaison des dormants
   au GO est placée à 100-150 mm du bord intérieur du dormant D.

GO désigne le gros œuvre (la maçonnerie qui reçoit la menuiserie). Le schéma ci-dessous montre
en élévation l'assemblage posé sur la traverse basse (D) : entraxe 400 mm, première fixation à
100-150 mm, repère \*a sous le raidisseur, distances \*b de part et d'autre (dessin non à
l'échelle). La planche de la p. 161 porte le même schéma.

![Liaison au gros œuvre d'un assemblage par contreventement](/assets/profiles/systeme70/statique/contreventement-1-liaison-go-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 160-161)

La planche des assemblages 4 et 5 (p. 162) ne porte pas de remarques.

## Assemblages avec élargisseurs (2008)

Un élargisseur est un profilé rapporté sur un dormant ou un meneau ; renforcé, il ajoute son
inertie à celle de l'élément [2 p. 163].

| Assemblage | Profilés et renforts | Iz (cm⁴) | Valeur statique imprimée (cm⁴) | Cotes portées (mm) | Coupe |
| --- | --- | --- | --- | --- | ---: |
| Assemblage 6 | 6101 / V601 ; 6101 / V601 ; 0205 / fer plat 10 × 80 | 3,20 ; 3,20 ; 42,70 | 49,10 | 16 ; 70 + 35 ; 18,5 et 24 | ![Assemblage 6 : 6101, 6101 et élargisseur 0205](/assets/profiles/systeme70/statique/elargisseur-6-6101-0205-fer-plat-evo2008.png) |
| Assemblage 7 | 2427 / 9119 ; 154 / 40 × 60 | 8,70 ; 19,00 | 26,70 | 52 ; 64,7 + 70 ; 20 + 75 + 20 | ![Assemblage 7 : meneau 2427 et élargisseur 154](/assets/profiles/systeme70/statique/elargisseur-7-2427-154-evo2008.png) |
| Assemblage 8 | 2427 / 9119 ; 1338 / 9120 | 8,70 ; 10,00 | 18,70 | 23,5 ; 52,5 + 70 ; 20 + 75 + 20 | ![Assemblage 8 : meneau 2427 et 1338](/assets/profiles/systeme70/statique/elargisseur-8-2427-1338-9120-evo2008.png) |
| Assemblage 9 | 2427 / 9119 ; 1114 / 9120 | 8,70 ; 10,00 | 18,70 | 25 ; 57 + 70 ; 20 + 75 + 20 | ![Assemblage 9 : meneau 2427 et 1114](/assets/profiles/systeme70/statique/elargisseur-9-2427-1114-9120-evo2008.png) |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 163)

**La valeur statique de l'assemblage 7 est imprimée 26,70 cm⁴ alors que 8,70 + 19,00 font
27,70 (INC-223).** Le manuel de 2023 dessine la même géométrie (52 ; 64,7 + 70) avec le profilé
93002 et le renfort V260 à 22,8 cm⁴, pour 31,5 cm⁴ (**CTR-85**) : voir *Assemblages de meneau
avec contreventement* plus haut.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 147 à 180 du
PDF (pages imprimées 1 à 34)

[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 —
`raw/profine-plans-profiles-e-volution-2008-08.pdf`, registre 4.2 « Statique », PDF p. 140 et 157 à 192 (pages imprimées 3 et 20 à 55)

# Voir aussi

Cette page est une partie d'un sujet qui en compte 5 ; les autres :

- [Calcul statique du système 70 : méthode, formule générale, plans de charge et exemples](/profiles/systeme-70-statique-et-inerties.md)
- [Moments d'inertie des fers plats et tubes acier, valeurs statiques des meneaux, système 70](/profiles/systeme-70-statique-et-inerties-inerties-fers-plats-tubes-et-valeurs-statiques.md)
- [Exemple de calcul statique par les tables, formulaire et exigences de flèche, système 70](/profiles/systeme-70-statique-et-inerties-exemple-de-calcul-par-les-tables.md)
- [Exemple de calcul statique, fers plats et tubes acier, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-statique-et-inerties-exemple-de-calcul-evo2008.md)
