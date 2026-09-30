---
type: Référence
title: Terminologie, légendes et calcul d'une cote d'élément, directives profine
description: Les planches de terminologie des directives générales profine (repères A à R, DHT, CCD, CCO, CCV, détails X, Y, Z) pour les fenêtres à joint de frappe et à joint central et pour les coulissants, la règle qui transforme une cote extérieure d'ouvrant maximale en cote d'élément, et les tailles d'ouvrants minimales en oscillo-battant et en soufflet.
tags: [profine, terminologie, legende, reperes, dht, ccd, cco, ccv, cote-element, cote-exterieure-ouvrant, taille-ouvrant-minimale, oscillo-battant, soufflet, coulissant]
systeme: [70, 76]
fournisseur: KÖMMERLING
usage: [atelier, chiffrage]
status: stable
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
source_pages:
  - resource: raw/profine-directives-generales-2023-01.pdf
    pages: 9-16
generated:
  by: process:claude-code
  at: 2026-09-28T12:30:00Z
---

Le registre 1.1.2 « Terminologie et légendes » des directives générales profine fixe les noms des
cotes et des parties d'une menuiserie PVC employés dans tous les manuels profine (systèmes 70 et
76), et la manière de passer d'une cote d'ouvrant à une cote de fenêtre. Les sigles seuls sont
aussi dans le [glossaire](/reference/glossaire.md) ; cette page porte les planches qui les
définissent [1 p. 9-16]. Chez PROFERM, le système 76 est celui de la fenêtre PERFORM76 ([PERFORM](/gammes/perform.md)) ; aucun document PROFERM ne rattache le système 70 à une gamme (**VER-28**).

# Tailles d'ouvrants maximales et cote d'élément

**Les cotes maximales indiquées dans les registres respectifs correspondent seulement aux cotes
extérieures d'ouvrant maximales.** Pour déterminer les dimensions maximales des éléments (cote
finale de fenêtre), il faut ajouter les cotes des profilés adjacents à tous les côtés [1 p. 9].

Trois cotes s'emboîtent : la **cote d'élément** (la fenêtre finie, profilés d'élargissement et
coffre compris), la **cote extérieure dormant** (le cadre fixe, ou dormant, mesuré à l'extérieur)
et la **cote extérieure ouvrant** (la partie mobile, ou ouvrant, mesurée à l'extérieur). Le
schéma ci-dessous montre les trois cotes sur une coupe : dormant à gauche avec un
profilé adjacent, ouvrant à droite.

![Cote d'élément, cote extérieure dormant et cote extérieure ouvrant](/assets/reference/directives-profine/cote-element-cote-dormant-cote-ouvrant.png)

Chaque flèche part du bord qui borne la cote et court vers la droite : la cote d'élément part du
bord extérieur du profilé adjacent, la cote extérieure dormant du bord extérieur du dormant, la
cote extérieure ouvrant du bord extérieur de l'ouvrant.

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 9)

## Exemple de calcul d'une hauteur d'élément

La coupe verticale de l'exemple montre, de haut en bas, un coffre de volet roulant de 205 mm, le
dormant haut (vue intérieure de 42 mm), l'ouvrant, le dormant bas (42 mm) et des profilés
d'élargissement de 120 mm sous le dormant. La cote extérieure ouvrant de l'exemple est de
2 100 mm ; la cote extérieure dormant (2 184 mm) est comptée entre le haut du dormant haut et le
bas du dormant bas, la hauteur d'élément (2 509 mm) du haut du coffre au bas des élargisseurs.

![Exemple de calcul de la hauteur d'un élément, coupe verticale](/assets/reference/directives-profine/cote-element-exemple-coupe-verticale.png)

Sur la coupe, les cotes de droite (205, 42, 2 100, 42, 120) s'additionnent de haut en bas ; les
cotes de gauche donnent la cote extérieure dormant (2 184 mm) et la hauteur élément (2 509 mm).

| Poste de l'exemple | Cote (mm) |
| --- | --- |
| Cote extérieure ouvrant | 2 100 |
| Vue intérieure de dormant 2 × 42 | 84 |
| Coffre de volet roulant | 205 |
| Profilés d'élargissement | 120 |
| **Hauteur totale élément** | 2 509 |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 9)

# Tailles d'ouvrants minimales

La taille d'ouvrant minimale dépend de la butée, c'est-à-dire du type d'ouverture : un
oscillo-battant s'ouvre à la française et en soufflet, un soufflet (imposte) s'ouvre en
soufflet seulement. Largeur et hauteur en mm [1 p. 9].

| Butée | Taille d'ouvrant min. — largeur (mm) | Taille d'ouvrant min. — hauteur (mm) |
| --- | --- | --- |
| Oscillo-battante | 340 | 660 |
| Soufflet (impostes) | 660 | 340 |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 9)

La largeur minimale de l'oscillo-battant est la hauteur minimale du soufflet, et inversement.

# Repères A à R d'une coupe dormant-ouvrant

La planche ci-dessous nomme chaque partie et chaque cote d'une coupe verticale d'ouvrant sur
dormant, côté extérieur à gauche, côté intérieur à droite ; les trois directions sont la hauteur,
la largeur et la profondeur (de l'extérieur vers l'intérieur). La vignette en haut à gauche porte
les cotes de baie O, P, Q et R sur une fenêtre vue de face ; le trièdre en haut à droite oriente
les axes (haut / bas, arrière / intérieur, extérieur / devant).

![Repères A à R d'une coupe ouvrant sur dormant](/assets/reference/directives-profine/reperes-a-r-coupe-ouvrant-dormant.png)

Les lettres cerclées désignent les parties et les cotes courtes ; les autres cotes sont écrites en
toutes lettres le long des flèches.

| Repère | Désignation |
| --- | --- |
| A | Dormant |
| B | Ouvrant |
| C | Parclose |
| D | Rainure crémone |
| E | Dos de dormant |
| F | Chambre des renforts |
| G | Rainure parclose |
| H | Feuillure vitrage |
| I | Feuillure quincaillerie |
| J | hauteur joint comprimé |
| K | Recouvrement ouvrant |
| L | Hauteur de calage vitrage |
| M | Hauteur pré-cale |
| N | Jeu de fonctionnement |
| O | Dimension extérieure dormant |
| P | Clair de jour vitrage |
| Q | Clair de jour ouvrant |
| R | Clair de jour dormant |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 10)

Les cotes écrites en toutes lettres sur la même planche sont : en profondeur, la profondeur
profilé, la hauteur recouvrement, la feuillure vitrage, le vitrage, le retrait parclose, la
profondeur parclose, la feuillure dormant, l'axe quincaillerie, la feuillure ouvrant et la
profondeur élément ; en hauteur, la hauteur/largeur de vue globale, la hauteur de vue ouvrant
(extérieure), la hauteur de vue dormant (extérieure), la hauteur feuillure vitrage ouvrant, la
hauteur feuillure vitrage, la hauteur parclose, la hauteur feuillure ouvrant, la hauteur totale
ouvrant, la hauteur de vue intérieure ouvrant, la hauteur feuillure dormant, le recouvrement
feuillure ouvrant, la hauteur intérieure dormant et la hauteur de vue intérieure dormant
[1 p. 10].

# Cotes de baie DHT, CCD, CCO, CCV

Quatre sigles désignent les cotes d'une fenêtre, de la plus grande à la plus petite [1 p. 11] :

| Sigle | Désignation |
| --- | --- |
| DHT | Dimension Hors Tout |
| CCD | Cote clair de dormant |
| CCO | Cote clair d'ouvrant |
| CCV | Cote clair de vitrage |

Sur les planches, trois largeurs de profilé séparent ces cotes : la largeur dormant est portée
entre la CCO et la DHT, la largeur de vue dormant entre la CCO et la CCD, la largeur de vue
ouvrant entre la CCV et la CCO. Les planches suivantes les portent sur chaque
type de construction.

## Fenêtre à joint de frappe

Coupe d'un ouvrant sur un dormant posé contre une maçonnerie isolée, vitrage à
gauche : la fenêtre ferme par deux joints de frappe (7), l'un sur le dormant, l'autre sur
l'ouvrant.

![Terminologie d'une fenêtre à joint de frappe](/assets/reference/directives-profine/terminologie-fenetre-joint-de-frappe.png)

Les chiffres cerclés sont les composants ; les lettres cerclées X, Y, Z entourent les détails ;
l'astérisque marque le jeu de fonctionnement d'un joint. Cotes nommées sur la planche : DHT, CCD,
CCO, CCV, largeur dormant, largeur de vue ouvrant, largeur de vue dormant, hauteur de feuillure,
prise en feuillure, feuillure dormant, jeu périphérique, largeur feuillure, cote extérieure
ouvrant, hauteur recouvrement, profondeur dormant, profondeur feuillure de vitrage ouvrant ; la
rainure crémone, le calage et le dos du dormant sont nommés en place [1 p. 11].

| Composition menuiserie | Désignation |
| --- | --- |
| 1 | Dormant |
| 2 | Ouvrant |
| 3 | Parclose avec joint |
| 4 | Renfort (dans la chambre de renfort) |
| 5 | Support cale de vitrage |
| 6 | Joint de vitrage |
| 7 | Joint de frappe |

| Détail | Désignation |
| --- | --- |
| X | Prise en feuillure |
| Y | Recouvrement de dormant |
| Z | Recouvrement d'ouvrant |
| \* | Jeu de fonctionnement joint |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 11)

## Fenêtre à joint central

Même coupe pour une fenêtre à joint central : un troisième joint (9), porté par le dormant au
milieu de la profondeur, s'ajoute aux joints de frappe, et l'ouvrant porte un joint de feuillure
(8).

![Terminologie d'une fenêtre à joint central](/assets/reference/directives-profine/terminologie-fenetre-joint-central.png)

Mêmes conventions que la planche à joint de frappe ; les cotes nommées sont les mêmes, la
feuillure de dormant et la profondeur étant portées au lieu de la feuillure dormant et de la
profondeur dormant [1 p. 12].

| Composition menuiserie | Désignation |
| --- | --- |
| 1 | Dormant |
| 2 | Ouvrant |
| 3 | Parclose avec joint |
| 4 | Renfort (dans la chambre de renfort) |
| 5 | Support cale de vitrage |
| 6 | Joint de vitrage |
| 7 | Joint de frappe |
| 8 | Joint de feuillure d'ouvrant |
| 9 | Joint central |

| Détail | Désignation |
| --- | --- |
| X | Prise en feuillure |
| Y | Feuillure dormant |
| Z | Feuillure ouvrant |
| \* | jeu de fonctionnement joint |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 12)

Sur la fenêtre à joint de frappe, Y et Z désignent les recouvrements ; sur la fenêtre à joint
central, les mêmes lettres désignent les feuillures [1 p. 11-12].

## Coulissant : dormant et ouvrant

Coupe d'un coulissant à deux ouvrants sur un dormant posé sur maçonnerie : les ouvrants
portent sur des rails de guidage (8) et le dormant porte des joints brosse (7). La vignette montre
la DHT, la CCD, la CCO et la CCV sur un coulissant vu de face.

![Terminologie d'un coulissant, dormant et ouvrant](/assets/reference/directives-profine/terminologie-coulissant-dormant-ouvrant.png)

Mêmes conventions ; le double astérisque marque la prise en feuillure ouvrant. Cotes nommées :
CCV, hauteur feuillure, hauteur ouvrant, profondeur de feuillure de vitrage, CCO, CCD, DHT,
largeur de vue de dormant, hauteur dormant, largeur dormant ; le dos de dormant est nommé en place
[1 p. 13].

| Composition menuiserie | Désignation |
| --- | --- |
| 1 | Dormant |
| 2 | Ouvrant |
| 3 | Parclose avec joint |
| 4 | Renfort (dans la chambre de renfort) |
| 5 | Support cale de vitrage |
| 6 | Joint de vitrage |
| 7 | Joint brosse dormant |
| 8 | Rail de guidage |

| Détail | Désignation |
| --- | --- |
| X | Prise en feuillure |
| Y | Feuillure ouvrant avec parclose |
| Z | Feuillure ouvrant |
| \* | jeu de fonctionnement joint |
| \*\* | Prise en feuillure ouvrant |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 13)

## Coulissant : partie centrale à chicane et pièce d'étanchéité médiane

Coupe de la partie centrale d'un coulissant, là où les deux ouvrants se rejoignent : une chicane
(2) vissée (5), un joint brosse (6) et une pièce d'étanchéité médiane (4) dessinée en contour
autour des deux ouvrants, avec ses deux points de vissage.

![Terminologie de la partie centrale d'un coulissant, chicane et pièce d'étanchéité médiane](/assets/reference/directives-profine/terminologie-coulissant-partie-centrale-chicane.png)

Cotes nommées : entraxe de vissage pièce d'étanchéité médiane, partie centrale, axe partie
centrale ; l'astérisque marque ici l'entraxe de vissage [1 p. 14].

| Composition menuiserie | Désignation |
| --- | --- |
| 1 | Ouvrant |
| 2 | Chicane |
| 3 | Parclose avec joint |
| 4 | Pièce d'étanchéité médiane |
| 5 | Vis |
| 6 | Joint brosse |

| Détail | Désignation |
| --- | --- |
| X | Prise en feuillure |
| Y | Joint brosse chicane |
| \* | Entraxe de vissage |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 14)

## Coulissant : profilé de liaison

Coupe d'un ouvrant de coulissant contre le dormant posé sur maçonnerie : un profilé de liaison (7)
est placé entre l'ouvrant et le dormant, au droit de la rainure quincaillerie.

![Terminologie d'un coulissant avec profilé de liaison](/assets/reference/directives-profine/terminologie-coulissant-profile-de-liaison.png)

Cotes nommées : DHT, CCD, CCO, CCV, largeur dormant, largeur de vue ouvrant, largeur de vue
dormant, hauteur de feuillure, profondeur de feuillure, cote extérieure ouvrant ; le calage, la
feuillure, la rainure quincaillerie et le dos de dormant sont nommés en place [1 p. 15].

| Composition menuiserie | Désignation |
| --- | --- |
| 1 | Dormant |
| 2 | Ouvrant |
| 3 | Parclose avec joint |
| 4 | Renfort (dans la chambre de renfort) |
| 5 | Support de cale de vitrage |
| 6 | Joint de vitrage |
| 7 | Profilé de liaison |

| Détail | Désignation |
| --- | --- |
| X | Prise en feuillure |
| \*\* | Hauteur de prise en feuillure ouvrant |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 15)

La légende de la planche numérote le profilé de liaison « ⑦ ⑧ » (voir **INC-88**) ; sur le
dessin, il porte le repère 7.

## Coulissant : verrouillage central

Coupe de la partie centrale d'un coulissant à verrouillage central : entre les deux ouvrants,
un verrouillage central (4) vissé (5) et un joint (6) ; le détail Y désigne le joint Q-Lon de la
chicane.

![Terminologie d'un coulissant à verrouillage central](/assets/reference/directives-profine/terminologie-coulissant-verrouillage-central.png)

Cotes nommées : partie centrale, axe partie centrale [1 p. 16].

| Composition menuiserie | Désignation |
| --- | --- |
| 1 | Ouvrant |
| 2 | Chicane |
| 3 | Parclose avec joint |
| 4 | Verrouillage central |
| 5 | Vis |
| 6 | Joint |

| Détail | Désignation |
| --- | --- |
| X | Prise en feuillure |
| Y | Joint Q-Lon chicane |
| Z | Joint verrouillage central |

(schéma: raw/profine-directives-generales-2023-01.pdf, p. 16)

Le détail Z figure dans la légende mais n'est pas repéré sur le dessin (voir **INC-88**).

# Citations

[1] [Directives générales profine, version janvier 2023](raw/profine-directives-generales-2023-01.pdf),
registre 1.1.2 « Terminologie et légendes » (pages de version janvier 2023), PDF p. 9 à 16
(imprimées 1.1.2 p. 1 à 8)

# Voir aussi

- [Glossaire des sigles et des cotes](/reference/glossaire.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Conditions d'utilisation des directives profine](/normes/directives-profine-conditions-d-utilisation.md)
