---
type: Profilé
title: Cotes des battements, des traverses d'ouvrant et élargisseur d'ouvrant EO20545, système 76
description: Cotes en coupe des battements 76471 et 76472 et des traverses d'ouvrant 76300, 76301 et 76303 avec leurs capots, et élargisseur d'ouvrant EO20545 à rupture de pont thermique.
tags: [systeme-76-advanced, profine, dormant, ouvrant, meneau, traverse, battement, capot, aluclip, profile]
systeme: 76
fournisseur: KÖMMERLING
usage: atelier
famille: profiles-principaux
status: draft
sources:
  - resource: raw/poster-systeme-76-advanced-principaux-2022.pdf
    id: poster-76-advanced-principaux
    title: Poster Système 76 Advanced, profilés principaux, 2022
    last_modified: 2022-12-31
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
    last_modified: 2025-06-19
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
    last_modified: 2025-06-19
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
source_pages:
  - resource: raw/poster-systeme-76-advanced-principaux-2022.pdf
    pages: 1
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    pages: 5, 13-16, 20
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    pages: 3-5, 12-15, 21, 33
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    pages: 4-13, 15-35, 38, 50, 52-53, 305-311
generated:
  by: process:claude-code
  at: 2026-10-21T10:43:10Z
---

Cette page reprend, du poster Système 76 Advanced (profilés principaux, 2022) et du manuel de mise en œuvre de profine, les cotes des battements, des traverses d'ouvrant et l'élargisseur d'ouvrant EO20545. Les caractéristiques du système et les cotes des dormants sont dans [Profilés principaux du système 76 : caractéristiques, cotes des dormants et capots](/profiles/systeme-76-profiles-principaux.md).

# Cotes des battements

Le **battement** est le profilé fixé sur l'un des deux ouvrants d'une fenêtre à deux vantaux, qui
recouvre la jonction avec l'autre ouvrant. Chaque coupe porte en haut la largeur, à gauche la
hauteur, à droite sa décomposition, et en bas une cote de 17,5 mm sur les 76471 et 76472.

| Battement | Largeur (mm) | Hauteur (mm) | Décomposition droite, de haut en bas (mm) | Cote basse (mm) | Capot | Coupe |
| --- | --- | --- | --- | --- | --- | ---: |
| 76471 | 94,5 | 62 | 29 + 16 + 17 | 17,5 | A051 | ![Battement 76471](/assets/profiles/systeme76/battements/battement-76471.png) |
| 76472 | 75 | 80 | 29 + 34 + 17 | 17,5 | A052 | ![Battement 76472](/assets/profiles/systeme76/battements/battement-76472.png) |
| 76473 | 31 | 46 | - | - | A318 | ![Battement 76473](/assets/profiles/systeme76/battements/battement-76473.png) |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1, troisième bande)

**Les planches du manuel de mise en œuvre ne cotent pas les battements comme le poster** : le
**76471** y fait **94,1 mm** de large pour 62 mm de haut, décomposés 22 + 21 + 19, sans cote basse ;
le **76472** 75 mm pour 80 mm, décomposés 22 + 41 + 17 ; le **76473** 31 mm pour 46 mm, comme
ci-dessus [4 p. 32] — entrée **CTR-26** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md). Le tableau garde les
valeurs du poster, les deux jeux de cotes étant portés avec leur source.

| Capot | Battement habillé | Largeur du retour (mm) | Hauteur (mm) | Coupe |
| --- | --- | --- | --- | ---: |
| A051 | 76471 | 19,9 | 67,9 | ![Capot A051](/assets/profiles/systeme76/battements/capot-a051.png) |
| A052 | 76472 | 19,9 | 85,7 | ![Capot A052](/assets/profiles/systeme76/battements/capot-a052.png) |
| A318 | 76473 | 10,8 | 50,8 | ![Capot A318](/assets/profiles/systeme76/battements/capot-a318.png) |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1, troisième bande)

Le battement **76473** est dessiné avec deux embouts au-dessus de lui : **M462** et **M501**, ce
dernier marqué « AluClip ». Le M462 est l'embout bas dont dépend la formule de débit du 76473 —
voir [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md#débit-du-battement-lui-même-et-de-son-renfort).

| Référence | Légende sur la planche | Dessin |
| --- | --- | ---: |
| M462 | - | ![M462](/assets/profiles/systeme76/battements/embout-m462.png) |
| M501 | AluClip | ![M501](/assets/profiles/systeme76/battements/embout-m501.png) |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1, deuxième bande)

Les assemblages de battement retenus en PERFORM76 sont dans
[Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md).

# Cotes des traverses d'ouvrant

La **traverse d'ouvrant** (appelée aussi meneau d'ouvrant) divise un ouvrant ; elle ne se monte
pas sur un dormant — voir [Meneaux PERFORM76](/profiles/perform76-meneaux.md). Chaque coupe porte
l'épaisseur de 76 mm en haut, la largeur totale à gauche et, à droite, sa décomposition en
aile + clair + aile, ailes de 21 mm.

| Traverse | Épaisseur (mm) | Largeur totale (mm) | Décomposition droite (mm) | Capot | Coupe |
| --- | --- | --- | --- | --- | ---: |
| 76300 | 76 | 68 | 21 + 26 + 21 | A045 | ![Traverse 76300](/assets/profiles/systeme76/meneaux/traverse-76300.png) |
| 76301 | 76 | 84 | 21 + 42 + 21 | A069 | ![Traverse 76301](/assets/profiles/systeme76/meneaux/traverse-76301.png) |
| 76303 | 76 | 110 | 21 + 68 + 21 | A070 | ![Traverse 76303](/assets/profiles/systeme76/meneaux/traverse-76303.png) |
| 76299 | 76 | 99,4 | 21 + 97 | A055 | ![Traverse 76299](/assets/profiles/systeme76/meneaux/traverse-76299.png) |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1, quatrième et cinquième bandes, à gauche)

Les planches du manuel de mise en œuvre donnent les mêmes cotes aux 76300 (68 = 21 + 26 + 21),
76301 (84 = 21 + 42 + 21) et 76303 (110 = 21 + 68 + 21), titrés « Traverse d'ouvrant » ; le titre
du 76303 annonce 119 mm — entrée **INC-09** [4 p. 34-35].

Le **76299** n'est pas symétrique : il n'a qu'une aile de 21 mm, et sa décomposition droite
(21 + 97 = 118 mm) dépasse la cote gauche de 99,4 mm, le profilé descendant sous la face
cotée à gauche. Le 76299 est la traverse complémentaire d'ouvrant, de 97 mm [2 p. 15].

Les quatre traverses sont redessinées avec leur capot sur les planches « Traverse complémentaire
d'ouvrant » et « Meneau / traverse intermédiaire ouvrant », avec les mêmes
capots (A055, A045, A069, A070) et les mêmes clairs (97, 26, 42 et 68 mm). La coupe du **76300**
y porte en plus une cote verticale intérieure de **25 mm** prise depuis le haut de la traverse
[3 p. 16] [2 p. 15].

## Élargisseur d'ouvrant EO20545

L'**élargisseur d'ouvrant EO20545** est un profilé aluminium à rupture de pont thermique : deux
coques aluminium reliées par des barrettes en polyamide 6.6 (légende « Alu - PA66 »), qui se
rapporte sur un cadre ouvrant pour l'élargir. La faible conductivité du polyamide lui confère une
isolation thermique limitant la condensation superficielle et les déperditions ; ses prescriptions
de qualité et de marquage (marque QB 49) sont dans
[DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) [3 p. 5, 16].

| Référence | Matière | Cotes | Coupe |
| --- | --- | --- | ---: |
| EO20545 | aluminium et polyamide PA66 | - | ![Élargisseur ouvrant EO20545](/assets/profiles/systeme76/elargisseurs/elargisseur-ouvrant-eo20545.png) |

(schéma: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf, p. 16)

La coupe n'est pas cotée. La référence est écrite **E020545** (chiffre zéro) sur la planche et
**EO20545** (lettre O) dans le texte, au DTA comme au DTD [2 p. 3, 5, 15] — entrée **INC-21** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

| Capot | Traverse habillée | Largeur du retour (mm) | Hauteur (mm) | Coupe |
| --- | --- | --- | --- | ---: |
| A045 | 76300 | 16,2 | 74 | ![Capot A045](/assets/profiles/systeme76/meneaux/capot-a045.png) |
| A069 | 76301 | 16,3 | 89,8 | ![Capot A069](/assets/profiles/systeme76/meneaux/capot-a069.png) |
| A046 | - | 19,7 | 90 | ![Capot A046](/assets/profiles/systeme76/meneaux/capot-a046.png) |
| A070 | 76303 | 16,3 | 116 | ![Capot A070](/assets/profiles/systeme76/meneaux/capot-a070.png) |
| A047 | - | 19,7 | 116 | ![Capot A047](/assets/profiles/systeme76/meneaux/capot-a047.png) |
| A055 | 76299 | 16,3 | 100,5 | ![Capot A055](/assets/profiles/systeme76/meneaux/capot-a055.png) |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1)

Les capots **A046** (19,7 × 90 mm) et **A047** (19,7 × 116 mm) sont dessinés à gauche des
capots A069 et A070, avec la largeur de retour de 19,7 mm des capots de meneau de dormant ; la
planche ne dit pas quel profilé ils habillent — entrée **VER-43**.

Sous les capots A047 et A070 sont dessinées trois pièces pleines, dessinées hachurées, **M527**, **M528** et
**M529**, sans cote ni légende.

| Référence | Dessin |
| --- | ---: |
| M527 | ![M527](/assets/profiles/systeme76/accessoires/m527.png) |
| M528 | ![M528](/assets/profiles/systeme76/accessoires/m528.png) |
| M529 | ![M529](/assets/profiles/systeme76/accessoires/m529.png) |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1, coin inférieur gauche)

# Citations

[1] [Poster Système 76 Advanced, profilés principaux, 2022](raw/poster-systeme-76-advanced-principaux-2022.pdf), p. 1

[2] [DTD n° DBV-25-6/16-2334_V5, système 76 Advanced](raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf)

[3] [DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED](raw/dta-trocal-76-advanced-6-16-2334-v5.pdf)

[4] Mise en œuvre Système 76 Advanced, profine — `raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`,
registre 2.1.1 (p. 4 à 16 du PDF), registre 2.1.2 (p. 17 et suivantes) et registre 2.6.2
« AluClip, Profilés principaux et accessoires » (p. 305 à 311 du PDF, pages imprimées 1 à 7)

# Voir aussi

Cette page est une partie d'un sujet qui en compte 5 ; les autres :

- [Profilés principaux du système 76 : caractéristiques, cotes des dormants et capots](/profiles/systeme-76-profiles-principaux.md)
- [Cotes des cadres fixes 76101, 76102, des ouvrants et des meneaux de dormant, système 76](/profiles/systeme-76-profiles-principaux-cadres-fixes-ouvrants-meneaux.md)
- [Capots de la variante AluClip du système 76 Advanced : planches et valeurs E et S](/profiles/systeme-76-profiles-principaux-capots-aluclip-valeurs-e-et-s.md)
- [Profilés de la porte d'entrée du système 76 Advanced : ouvrants 76201, 76206, 76207](/profiles/systeme-76-profiles-principaux-profiles-de-la-porte-d-entree.md)
