---
type: Profilé
title: 'Calcul statique du système 70 : méthode, formule générale, plans de charge et exemples'
description: 'Moment d''inertie minimal du renfort d''acier d''un meneau, d''une traverse ou d''un ouvrant du système 70 : formule générale (I, Pr, S, E, f), plans de charge des cas 1 à 5 et deux exemples d''application.'
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

# Ce que règle le calcul statique

Le calcul statique fixe le moment d'inertie minimal du renfort d'acier d'un élément de
menuiserie PVC du système 70 Plateforme. Le moment d'inertie (I, en cm⁴) mesure la rigidité
d'une section à la flexion : plus il est grand, moins l'élément fléchit sous la pression du vent.
Le renfort est le profilé d'acier glissé dans la chambre du profilé PVC
([glossaire](/reference/glossaire.md)). Section 2.3.4 du manuel
[Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md), PDF p. 147 à 180.

Les quinze tables des moments d'inertie requis, une par classement au vent, sont sur leurs pages :

| Classement | Pression (Pa) | Flèche admissible | Page PDF | Page du wiki |
| --- | --- | --- | --- | --- |
| V\*A1 | 400 | 1/150 | 166 | [Moments d'inertie requis, V\*A1](/profiles/systeme-70-inerties-va1.md) |
| V\*A2 | 800 | 1/150 | 167 | [Moments d'inertie requis, V\*A2](/profiles/systeme-70-inerties-va2.md) |
| V\*A3 | 1200 | 1/150 | 168 | [Moments d'inertie requis, V\*A3](/profiles/systeme-70-inerties-va3.md) |
| V\*A4 | 1600 | 1/150 | 169 | [Moments d'inertie requis, V\*A4](/profiles/systeme-70-inerties-va4.md) |
| V\*A5 | 2000 | 1/150 | 170 | [Moments d'inertie requis, V\*A5](/profiles/systeme-70-inerties-va5.md) |
| V\*B1 | 400 | 1/200 | 171 | [Moments d'inertie requis, V\*B1](/profiles/systeme-70-inerties-vb1.md) |
| V\*B2 | 800 | 1/200 | 172 | [Moments d'inertie requis, V\*B2](/profiles/systeme-70-inerties-vb2.md) |
| V\*B3 | 1200 | 1/200 | 173 | [Moments d'inertie requis, V\*B3](/profiles/systeme-70-inerties-vb3.md) |
| V\*B4 | 1600 | 1/200 | 174 | [Moments d'inertie requis, V\*B4](/profiles/systeme-70-inerties-vb4.md) |
| V\*B5 | 2000 | 1/200 | 175 | [Moments d'inertie requis, V\*B5](/profiles/systeme-70-inerties-vb5.md) |
| V\*C1 | 400 | 1/300 | 176 | [Moments d'inertie requis, V\*C1](/profiles/systeme-70-inerties-vc1.md) |
| V\*C2 | 800 | 1/300 | 177 | [Moments d'inertie requis, V\*C2](/profiles/systeme-70-inerties-vc2.md) |
| V\*C3 | 1200 | 1/300 | 178 | [Moments d'inertie requis, V\*C3](/profiles/systeme-70-inerties-vc3.md) |
| V\*C4 | 1600 | 1/300 | 179 | [Moments d'inertie requis, V\*C4](/profiles/systeme-70-inerties-vc4.md) |
| V\*C5 | 2000 | 1/300 | 180 | [Moments d'inertie requis, V\*C5](/profiles/systeme-70-inerties-vc5.md) |

Le classeur e.VOLUTION de 2008 imprime les mêmes quinze tables au registre 4.2 (PDF p. 171 à 185) : relues case par case contre celles de 2023, elles sont identiques, et chacune le dit sur sa page [2 p. 171-185].

# Méthode de calcul

## 1. Généralités

Le renforcement des menuiseries est destiné à assurer la rigidité des éléments PVC, en fonction
de la résistance nécessaire recherchée, suivant les critères ci-dessous [1 p. 147] :

- résistance à la déformation sous une pression reproduisant les effets du vent : cas des parties
  ouvrantes, cas des éléments de l'ossature tels que meneaux, traverses d'impostes ;
- résistance aux chocs dits « chocs de sécurité » : cas de traverses d'allège ;
- résistance aux déformations dues au mode de pose : cas des dormants posés sur doublages.

## 2.1 Formule générale

I = (5 × Pr × S × L³ × 10) / (384 × E × f) [1 p. 147]

| Symbole | Grandeur | Unité |
| --- | --- | --- |
| I | moment d'inertie minimum admissible du renforcement | cm⁴ |
| Pr | pression à appliquer | Pa |
| S | surface du plan de charge | m² |
| L³ | longueur de l'élément à renforcer élevée au cube | m³ |
| E | module d'élasticité à la flexion du renforcement | kgf/mm² |
| f | flèche maximale admissible | m |

Nota : la fraction 5/384 correspond à un coefficient relatif à une charge uniformément répartie.
La valeur 10 au numérateur sert à balancer la formule compte tenu des unités prises [1 p. 147].

## 2.2.1 Valeur de I

Le renforcement proposé devra avoir un moment d'inertie égal ou supérieur à la valeur de I
calculée selon la formule ci-avant. Pour les cas de renforcement d'un même élément avec plusieurs
renforts, les moments d'inertie de ces renforts s'additionnent [1 p. 147].

## 2.2.2 Valeur de Pr

La pression Pr est à considérer appliquée depuis l'extérieur de la menuiserie et
perpendiculairement à celle-ci. Lorsqu'il est demandé un classement « V » précis (selon la norme
EN 12211), Pr se lit sur le tableau suivant ; P1, P2 et P3 sont les trois pressions d'essai de la
classe, en Pa [1 p. 148].

| Classe | P1 (Pa) | P2 (Pa) | P3 (Pa) |
| --- | --- | --- | --- |
| 0 | pas d'essai | pas d'essai | pas d'essai |
| 1 | 400 | 200 | 600 |
| 2 | 800 | 400 | 1200 |
| 3 | 1200 | 600 | 1800 |
| 4 | 1600 | 800 | 2400 |
| 5 | 2000 | 1000 | 3000 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 148)

Lorsqu'il n'est pas demandé de classement « V », il convient de se référer à la FD P 20-201
« choix des fenêtres en fonction de leur exposition » (mémento), en prenant les valeurs indiquées
au tableau 2 de sa page 9 après en avoir déterminé les paramètres : hauteur des fenêtres au-dessus
du sol, zone et situation [1 p. 148]. Le classement A\*E\*V est expliqué sur
[Labels et certifications](/certifications/labels-et-certifications.md).

Cas particulier des allèges devant assurer la sécurité des personnes. L'allège est la
partie basse, pleine ou vitrée, d'une baie. Pour ce cas, la pression Pr est remplacée par la
valeur Q correspondant à la charge [1 p. 148] :

| Cas | Q (kg/m) |
| --- | --- |
| allège devant assurer totalement la sécurité des personnes | 100 |
| allège avec une autre protection jumelée, telle que garde-corps ou allège maçonnée | 75 |

La formule initiale devient alors : I = (5 × Q × L⁴ × 100) / (384 × E × f) [1 p. 148].

## 2.2.3 Valeur de S

La valeur de S est variable, en fonction des dimensions de la menuiserie et du positionnement,
dans la menuiserie, de l'élément dont on cherche le renforcement [1 p. 148]. Les plans de charge
sont dessinés plus bas.

## 2.2.4 Valeur de E

Le module d'élasticité à la flexion est variable en fonction de la matière propre au renfort
[1 p. 148] :

| Matière du renfort | E (N/mm²) | E (kgf/mm²) |
| --- | --- | --- |
| acier | 210 000 | 21 000 |
| aluminium | 70 000 | 7 000 |
| bois massif | 10 000 | 1 000 |

Nota : pour le PVC, la valeur de E étant relativement peu élevée (2 500 N/mm²), il n'est pas
introduit dans les calculs des moments d'inertie propres aux profils PVC. Seuls sont considérés
ceux des renforts [1 p. 148].

## 2.2.5 Valeur de f

La flèche f est variable en fonction du positionnement, dans la menuiserie, de l'élément dont on
cherche le renforcement [1 p. 150] :

| Élément | Flèche admissible f |
| --- | --- |
| parties ouvrantes (montant de rive, montant milieu, traverse d'ouvrant) | L/150 |
| éléments du dormant (meneau, traverse d'imposte ou intermédiaire) | L/150 |
| cas particulier des traverses d'allège | L/300 |

Les tables des moments d'inertie requis existent aussi pour des flèches de 1/200 (classements V\*B)
et 1/300 (classements V\*C) ; aucun texte de la section ne dit à quels éléments elles s'appliquent.

# Plans de charge

Le plan de charge est la part de la surface de la menuiserie dont la pression du vent est
reprise par un élément donné. Sur chaque schéma, la surface hachurée est le plan de charge de
l'élément ; L et H sont la largeur et la hauteur, L¹, L², H¹, H² les largeurs et hauteurs des
parties [1 p. 149]. Légende : S 1 = plan de charge d'un montant de rive (cas 1), d'un meneau
(cas 2) ; S 2 = plan de charge d'une traverse ; S 3 = plan de charge des montants milieu.

## Cas 1 : châssis, croisées et portes-croisées à 1 ou 2 vantaux

![Plans de charge, cas 1 : châssis, croisées et portes-croisées à 1 ou 2 vantaux](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-1-chassis.png)

À un vantail (à gauche), le montant de rive reprend une bande de largeur L/2 bornée à L/2 des
angles (S 1), la traverse un triangle (S 2) ; à deux vantaux (à droite), les montants de rive et
les montants milieu (S 3) reprennent chacun L/4.

## Cas 2 : meneaux

![Plans de charge, cas 2 : un seul meneau](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-2-un-meneau.png)

Cas avec un seul meneau : le plan de charge S 1 du meneau est tracé à 45° depuis les angles de
chaque partie. Sur le schéma de gauche, il s'arrête à L¹/2 du côté de la partie L¹ et à H/2 du
côté de la partie L² ; sur celui de droite, il forme un losange arrêté à H/2 de part et d'autre du
meneau.

![Plans de charge, cas 2 : deux meneaux ou plus](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-2-deux-meneaux.png)

Cas avec deux meneaux ou plus : chaque meneau reprend la moitié des parties voisines (L¹/2, L²/2) ;
le second plan de charge est repéré « S 1 bis ».

## Cas 3 : traverse intermédiaire (imposte ou allège)

![Plans de charge, cas 3 : traverse intermédiaire](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-3-traverse-intermediaire.png)

La traverse reprend H¹/2 et H²/2 de part et d'autre (S 2), la charge étant arrêtée à 45° aux
extrémités.

## Cas 4 : meneau avec traverse intermédiaire (composé)

![Plans de charge, cas 4 : meneau avec traverse intermédiaire](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-4-meneau-compose.png)

Le meneau reprend un rectangle de largeur L¹/2 + L²/2 sur toute la hauteur H (S 1).

## Cas 5 : traverse intermédiaire avec meneaux composés

![Plans de charge, cas 5 : traverse intermédiaire avec meneaux composés](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-5-traverse-meneaux-composes.png)

La traverse, arrêtée sur les meneaux, reprend H¹/2 et H²/2 sur la seule largeur L¹ (S 2).

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 149)

## Exemples de plans de charge pour traverse et meneau

Remarque : a ou b < L/2 [1 p. 165]. Les traits épais sont les éléments filants ou courts, les
cercles leurs appuis, les tiretés les bissectrices à 45° qui délimitent les plans de charge.

![Meneau filant et traverse](/assets/procedures/moe-systeme-70/statique/plans-de-charge-meneau-filant-et-traverse.png)

Un meneau filant (qui va d'un bout à l'autre du dormant) et une traverse arrêtée sur lui : le
meneau reprend les largeurs de charge a et b tracées à 45° ; la part repérée par un astérisque
« Ce plan de charge n'est pas pris en compte ».

![Deux meneaux filants et traverses courtes](/assets/procedures/moe-systeme-70/statique/plans-de-charge-meneaux-filants-traverses-courtes.png)

Deux meneaux filants et trois traverses courtes : les plans de charge des meneaux sont hachurés.

![Traverses courtes entre meneaux filants](/assets/procedures/moe-systeme-70/statique/plans-de-charge-traverses-courtes-entre-meneaux-filants.png)

Même composition, plans de charge des traverses courtes hachurés.

![Traverse filante sous coffre de volet roulant](/assets/procedures/moe-systeme-70/statique/plans-de-charge-traverse-filante-sous-cvr.png)

Traverse filante sous CVR (coffre de volet roulant), deux meneaux et trois traverses : la
traverse filante reprend le trapèze hachuré.

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 165)

# Exemples d'application de la formule générale

## Exemple n° 1 : porte-croisée à 3 vantaux

![Porte-croisée à 3 vantaux de l'exemple n° 1](/assets/procedures/moe-systeme-70/statique/exemple-1-porte-croisee-3-vantaux.png)

Porte-croisée de 2,1 m de large (0,7 m de fixe F et 1,4 m d'ouvrant) sur 2,15 m de haut ;
classement demandé V 2. Calcul de renforcement du meneau [1 p. 150] :

| Grandeur | Valeur imprimée |
| --- | --- |
| Pr | 800 Pa |
| S 1 | 1,645 m² |
| L³ | 2,15³ = 9,94 |
| E | acier 21 000 kgf/mm² |
| f | L/150 = 2,15/150 = 0,014 m |
| I | (5 × 300 × 1,645 × 9,94 × 10) / (384 × 21 000 × 0,014) = 5,79 cm⁴ |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 150)

**Le calcul imprimé porte 300 au numérateur alors que Pr vaut 800 Pa ; le résultat 5,79 cm⁴ est
celui obtenu avec 800 — entrée INC-139**.

## Exemple n° 2 : croisée à 3 vantaux sur allège

![Croisée à 3 vantaux sur allège de l'exemple n° 2](/assets/procedures/moe-systeme-70/statique/exemple-2-croisee-3-vantaux-sur-allege.png)

Croisée de 2,2 m de large, 1,25 m de vitrage sur 0,9 m d'allège pleine (2,15 m au total) ; pas de
classement V demandé ; hauteur au-dessus du sol 28 à 50 m, région A, situation b (périphérie d'un
grand centre urbain). Calcul de renforcement de la traverse d'allège [1 p. 150] :

| Grandeur | Valeur imprimée |
| --- | --- |
| Q | 100 kg/m |
| L⁴ | 2,20⁴ = 23,42 |
| E | acier 21 000 kgf/mm² |
| f | L/300 = 2,20/300 = 0,073 m, limité à 0,010 m |
| I | (5 × 100 × 23,43 × 100) / (384 × 21 000 × 0,010) = 14,52 cm⁴ |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 150)

**2,20/300 vaut 0,0073 m et non 0,073 m ; le calcul emploie 0,010 m, et L⁴ est écrit 23,42 puis
23,43 — entrée INC-140**.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 147 à 180 du
PDF (pages imprimées 1 à 34)

[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 —
`raw/profine-plans-profiles-e-volution-2008-08.pdf`, registre 4.2 « Statique », PDF p. 140 et 157 à 192 (pages imprimées 3 et 20 à 55)

# Voir aussi

- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Abaques dimensionnels de renforcement du système 70](/profiles/systeme-70-abaques-dimensionnels.md)
- [Mise en œuvre des renforts du système 70](/procedures/mise-en-oeuvre-renforts-systeme-70.md)
- [Accouplement des éléments du système 70](/procedures/accouplement-elements-systeme-70.md)
- [Classification de la résistance au vent](/reference/classification-resistance-au-vent.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)

Autres parties de ce sujet :

- [Moments d'inertie des fers plats et tubes acier, valeurs statiques des meneaux, système 70](/profiles/systeme-70-statique-et-inerties-inerties-fers-plats-tubes-et-valeurs-statiques.md)
- [Exemple de calcul statique par les tables, formulaire et exigences de flèche, système 70](/profiles/systeme-70-statique-et-inerties-exemple-de-calcul-par-les-tables.md)
- [Valeurs statiques Iz des meneaux, traverses et assemblages, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-statique-et-inerties-valeurs-statiques-evo2008.md)
- [Exemple de calcul statique, fers plats et tubes acier, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-statique-et-inerties-exemple-de-calcul-evo2008.md)
