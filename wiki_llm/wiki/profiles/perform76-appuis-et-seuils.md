---
type: Profilé
title: Appuis et seuils PERFORM76
description: Les sept appuis PERFORM76, le nez d'appui 4319, les quatre seuils aluminium A075 à A343 et les deux compensateurs de rénovation, avec leur affectation par dormant quand elle est connue.
tags: [perform76, appui, seuil, nez-d-appui, compensateur, rejet-d-eau]
gamme: PERFORM
systeme: 76
fournisseur: KÖMMERLING
usage: [atelier, pose]
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
    last_modified: 2026-09-02
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
    last_modified: 2025-06-19
  - resource: raw/poster-systeme-76-advanced-principaux-2022.pdf
    id: poster-76-advanced-principaux
    title: Poster Système 76 Advanced, profilés principaux, 2022
    last_modified: 2022-12-31
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
    last_modified: 2025-06-19
source_pages:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    pages: 14-22, 24
  - resource: raw/poster-systeme-76-advanced-principaux-2022.pdf
    pages: 1
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    pages: 8-10, 14, 17-18, 42
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    pages: 5, 13, 17, 20
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
---

# Deux familles d'appuis, selon le dormant

L'**appui** est le profil posé sous le dormant bas, penté vers l'extérieur, qui rejette l'eau
de pluie loin du mur ; le **seuil** est son équivalent au bas d'une porte-fenêtre, sur lequel on
passe. Les appuis PERFORM76 forment deux groupes qui ne se mélangent pas.

| Groupe | Appuis | Dormants |
| --- | --- | --- |
| Rénovation et neuf avec aile | 6136, 6137, 76768 | 76177, 76180, 76185 |
| Neuf sans aile | 76751, 76752, 76753, 76758 + 76719 | 76171, 76172 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 16, 20, 21 et 22)

# Cotes des appuis 6136, 6137 et 76768

Appuis du premier groupe, cotes en mm. Les trois sont pentés à **3°**.

| Appui | Largeur (mm) | Longueur (mm) | Épaisseur (mm) | Retombée (mm) | Nez d'appui 4319 | Coupe |
| --- | --- | --- | --- | --- | --- | ---: |
| 6136 | 67 | 127 | 14 | 3,5 | oui | ![Appui 6136](/assets/profiles/perform76/appuis/appui-6136.png) |
| 6137 | 97 | 157 | 14 | 3,5 | oui | ![Appui 6137](/assets/profiles/perform76/appuis/appui-6137.png) |
| 76768 | 136 | 196 | 14 | 22 | non | ![Appui 76768](/assets/profiles/perform76/appuis/appui-76768.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 16)

La **hauteur d'about** de ces trois appuis dépend du dormant :

| Dormant | Aile (mm) | Hauteur d'about (mm) |
| --- | --- | --- |
| 76177 | 40 | 26 |
| 76185 | 60 | 46 |
| 76180 | 20 | 6 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 16, 18, 19 et 24)

Sur le dormant neuf 76180, les mêmes appuis sont cotés **14** au-dessus et **6** en dessous, sans
la retombée de 3,5 mm portée sur les dormants rénovation [1 p. 19]. La même cote de 6 mm figure
sous les appuis 6137 et 76768 de la planche du 76171, où le dormant bas devient un 76180
[1 p. 24].

Les 20 mm d'écart de hauteur d'about entre 76177 et 76185 reprennent exactement les 20 mm d'écart
d'aile.

# Cotes des appuis 76751, 76752, 76753 et 76758

Appuis du second groupe, cotes en mm.

| Appui | Hauteur (mm) | Décomposition (mm) | Pente | Particularité | Coupe |
| --- | --- | --- | --- | --- | ---: |
| 76751 | 30 | 20 / 56 | - | livré non monté | ![Appui 76751](/assets/profiles/perform76/appuis/appui-76751.png) |
| 76752 | 50 | 20 / 56 | - | livré non monté | ![Appui 76752](/assets/profiles/perform76/appuis/appui-76752.png) |
| 76753 | 35 | 21 / 46, sur 76 | - | - | ![Appui 76753](/assets/profiles/perform76/appuis/appui-76753.png) |
| 76758 + 76719 | 15 à 20 | 80 en saillie, dénivelé 3, retombée 5,5, largeur totale 156 | **5°** | ensemble de deux profils | ![Appui 76758 + 76719](/assets/profiles/perform76/appuis/appui-76758.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 22 et 24)

**L'ensemble 76758 + 76719 est le seul appui penté à 5°** ; les six autres appuis de la gamme sont
à 3°.

Les appuis **76751 et 76752 sont livrés non montés**, à intégrer au temps d'atelier. La mention du
cahier technique cite un « 76152 » qui n'existe nulle part ailleurs — entrée **INC-04** du
registre [Incohérences internes](/anomalies/incoherences-internes.md).

L'appui **76758 ne se commande pas seul** : il forme un ensemble avec le **76719**. C'est l'appui
du dormant 76171 en pose isolée jusqu'à 155 mm d'isolant — voir
[Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md).

# Cotes du nez d'appui et du seuil

| Référence | Type | Cotes (mm) | Compatibilité | Coupe |
| --- | --- | --- | --- | ---: |
| 4319 | nez d'appui | 18 / 8 | appuis 6136 et 6137 uniquement | ![Nez d'appui 4319](/assets/profiles/perform76/appuis/nez-appui-4319.png) |
| A076 | seuil aluminium | 76 de large, 10 et 10, about 20 | les cinq dormants | ![Seuil A076 + Rejet d'eau A062](/assets/profiles/perform76/appuis/seuil-a076.png) |
| A062 | rejet d'eau | - | s'associe au seuil A076 | - |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 16 et 22)

Le nez d'appui **4319 ne se monte pas sur le 76768** : il n'est représenté que sur les appuis
6136 et 6137.

Le seuil **A076 et le rejet d'eau A062 vont toujours ensemble** ; ils ne sont jamais figurés
séparément sur les cinq planches dormant. Aucune cote n'est donnée pour le A062.

# Pièces d'appui et tapées aluminium

Les dormants capotés (habillés d'un capot aluminium) peuvent recevoir des tapées et des appuis
aluminium, distincts des appuis et tapées PVC ci-dessus. Une ligne par référence ; la cote est
celle que la coupe porte : la largeur de l'appui (cote horizontale du haut), la hauteur de la
tapée (cote verticale de droite).

| Pièce | Fonction | Cote portée sur la coupe (mm) | Coupe |
| --- | --- | --- | ---: |
| A475 | pièce d'appui alu | 97 | ![Appui A475](/assets/profiles/systeme76/tapees-alu/appui-a475.png) |
| A476 | pièce d'appui alu | 137 | ![Appui A476](/assets/profiles/systeme76/tapees-alu/appui-a476.png) |
| A477 | pièce d'appui alu, réservée à la rénovation | 77 | ![Appui A477](/assets/profiles/systeme76/tapees-alu/appui-a477.png) |
| A491 | pièce d'appui alu | 57 | ![Appui A491](/assets/profiles/systeme76/tapees-alu/appui-a491.png) |
| DT100 | bavette alu, dessinée parmi les pièces d'appui alu | 100 | ![Appui DT100](/assets/profiles/systeme76/tapees-alu/appui-dt100.png) |
| A469 | tapée alu | 30 | ![Tapée A469](/assets/profiles/systeme76/tapees-alu/tapee-a469.png) |
| A470 | tapée alu | 50 | ![Tapée A470](/assets/profiles/systeme76/tapees-alu/tapee-a470.png) |
| A471 | tapée alu | 70 | ![Tapée A471](/assets/profiles/systeme76/tapees-alu/tapee-a471.png) |
| A472 | tapée alu | 90 | ![Tapée A472](/assets/profiles/systeme76/tapees-alu/tapee-a472.png) |
| A473 | tapée alu | 110 | ![Tapée A473](/assets/profiles/systeme76/tapees-alu/tapee-a473.png) |

(schéma: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf, p. 18 ; même planche :
raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 17)

Le DT100 est dessiné avec les pièces d'appui aluminium A475 à A491 [2 p. 18] ; son schéma de montage le nomme « bavette DT100 » — voir [Fabrication et assemblage du système 76 Advanced](/procedures/fabrication-systeme-76-advanced.md#montage-de-la-bavette-dt100) [2 p. 42].
La hauteur de chacune de ces pièces (31,5 et 35 mm pour les tapées, 19,7 mm pour les appuis), leur
mode opératoire pas à pas et leurs cotes de débit sont dans
[Capotage AluClip du système 76 Advanced](/procedures/capotage-aluclip-systeme-76.md#tapées-et-appuis-aluminium-sur-dormant-capoté).
Les tapées et appuis aluminium se vissent sur le dormant capoté avec un entraxe maximum de 400 mm,
avec les embouts M646 et M643 et la mousse G251 — prescription complète dans
[DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md), section 2.2.3.3.5.

Les deux embouts des tapées et appuis aluminium sont dessinés sous le titre « Embouts de pièces
d'appui et de tapées aluminium » : l'embout **M643**, en vue de face, qui obture l'extrémité de la
pièce d'appui aluminium, et l'embout de tapée **M646**, vu de dessus (en haut, avec ses tétons et
ses trois logements de vis) et vu de face (en bas, marqué « M646-R »). Aucun des deux n'est coté
[3 p. 20].

| Embout | Fonction | Coupe |
| --- | --- | ---: |
| M643 | embout d'extrémité de pièce d'appui aluminium | ![Embout M643](/assets/profiles/systeme76/tapees-alu/embout-m643.png) |
| M646 | embout de tapée aluminium, avec sa plaquette en mousse repliée qui reçoit la pièce d'appui | ![Embout M646](/assets/profiles/systeme76/tapees-alu/embout-m646.png) |

(schéma: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 20)

**L'appui A476 ne peut pas être assemblé avec la tapée A469** ; il s'assemble avec les tapées
A470, A471, A472 ou A473 [2 p. 8-9]. L'appui A477 est réservé aux mises en œuvre en rénovation
sur dormant existant.

**Ces mêmes références A469 à A473 désignent des « embouts d'extrémité de pièce d'appui » sur le
poster Gamme 70 KÖMMERLING**, une fonction différente de la tapée qu'elles nomment ici. Système
distinct, plan illisible à confirmer : à vérifier avant toute commande croisée entre les deux
systèmes — entrée **VER-40** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Le système 76 Advanced compte trois seuils supplémentaires, absents du cahier technique
PERFORM76. Largeur et hauteur en mm ; les coupes des A075 et A077 sont celles de la planche des
profilés principaux du système, où chaque seuil est dessiné avec l'extrémité du dormant posée
dessus.

| Seuil | Largeur (mm) | Hauteur (mm) | Coupe |
| --- | --- | --- | ---: |
| A075 | 76 | 26 | ![Seuil A075](/assets/profiles/systeme76/seuils/seuil-a075.png) |
| A077 | 123 | 20 | ![Seuil A077](/assets/profiles/systeme76/seuils/seuil-a077.png) |
| A343 | 135 | 20 | ![Seuil A343](/assets/profiles/systeme76/seuils/seuil-a343.png) |

(schéma: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf, p. 14 ; raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 13 ; raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1)

Le **A343** n'est pas dessiné sur la planche des profilés principaux ; sa coupe le montre seul,
sans dormant, avec sa largeur de 135 mm en haut et sa hauteur de 20 mm à droite [2 p. 14]. Les
seuils A075, A076, A077 et A343 sont des seuils mixtes aluminium – PVC, dont la fabrication est décrite
dans [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md), section 2.8.6.1.

## Seuil A076 avec A065 et Z151

Sur la planche des profilés principaux, le seuil **A076** (76 mm de large, 20 mm de haut) est
dessiné sous deux pièces : le profilé aluminium **A065**, de 76 mm, et le profilé **Z151**, dessiné en trait
fin entre les deux. La planche ne légende pas leur
fonction.

![Seuil A076 avec A065 et Z151](/assets/profiles/systeme76/seuils/seuil-a076-a065-z151.png)

De haut en bas : le A065, la Z151, puis le A076 avec l'extrémité du dormant posée dessus, à
droite. La cote de 76 mm, en haut, est portée au-dessus du A065 ; la cote de 20 mm, à droite, est
la hauteur du seuil.

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1, troisième bande)

## Pièces dessinées avec les seuils

| Pièce | Position sur la planche | Dessin |
| --- | --- | ---: |
| G067 | à gauche du seuil A075 | ![G067](/assets/profiles/systeme76/seuils/bouchon-g067.png) |
| G255 | à gauche du seuil A077 | ![G255](/assets/profiles/systeme76/seuils/profil-g255.png) |

(schéma: raw/poster-systeme-76-advanced-principaux-2022.pdf, p. 1)

Le **G067** est le bouchon d'about propre au seuil A075. Les sets d'assemblage de chaque dormant
sur les seuils A076, A077 et A075 sont dans
[Assemblages du système 76](/profiles/systeme-76-assemblages.md).

**Le A075 est plus haut que le A076** — 26 contre 20 mm — pour la même largeur de 76 mm. Il se
monte **uniquement avec contre-profilage du montant** (équerre M150 et pièce d'étanchéité J064),
son extrémité est obturée par le bouchon G067 et il est drainé à 100 mm des montants puis tous les
600 mm maximum par un orifice de 5 × 25 mm environ ; les seuils A076, A077 et A343 se montent
sans contre-profilage (équerre M154 et pièce de compensation) ou avec (M150 et J064) [3 p. 5].
Le montage complet est au § 2.2.3.1.4 de [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md). Les sets, les schémas de perçage, les
gabarits, les contours de fraisage et les étapes de montage de chaque dormant et meneau sur ces
seuils, ainsi que les rejets d'eau A062 et A064, sont dans
[Mise en œuvre du seuil du système 76 Advanced](/procedures/mise-en-oeuvre-seuil-systeme-76.md) ;
la porte-fenêtre avec fixe latéral sur seuil filant dans
[Porte-fenêtre avec fixe latéral du système 76 Advanced](/procedures/porte-fenetre-fixe-lateral-systeme-76.md).
Le A077 (123 mm) et le A343 (135 mm) sont plus larges ;
aucune source n'attribue chaque seuil à un dormant précis — entrée **VER-39** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

# Cotes des compensateurs

Les deux compensateurs sont réservés aux dormants rénovation 76177 et 76185. Chacun se monte dans
deux orientations.

| Compensateur | Cotes (mm) | Usage | Coupe |
| --- | --- | --- | ---: |
| 6143 | 19 × 29 | profil rénovation | ![Compensateur 6143](/assets/profiles/perform76/compensateurs/compensateur-6143.png) |
| 6144 | 12 × 16 | profil rénovation | ![Compensateur 6144](/assets/profiles/perform76/compensateurs/compensateur-6144.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 16)

## Montage des compensateurs

Le **compensateur** est un petit profil tubulaire clipsé sous l'aile du dormant rénovation, qui
comble l'écart avec l'ancien bâti. Chacun se monte dans deux orientations : debout ou couché, ce
qui donne deux épaisseurs de compensation avec la même pièce.

![Dormant rénovation 76177 + compensateur 6143](/assets/profiles/perform76/compensateurs/montage-76177-6143.png)

![Dormant rénovation 76185 + compensateur 6143](/assets/profiles/perform76/compensateurs/montage-76185-6143.png)

Le compensateur **6143** se monte debout (19 de large, 29 de haut) ou couché (29 de large,
19 de haut), sur le 76177 comme sur le 76185.

![Dormant rénovation 76177 + compensateur 6144](/assets/profiles/perform76/compensateurs/montage-76177-6144.png)

![Dormant rénovation 76185 + compensateur 6144](/assets/profiles/perform76/compensateurs/montage-76185-6144.png)

Le compensateur **6144** se monte de même, en 12 × 16 ou en 16 × 12.

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 16)

# Compatibilités

| Référence | 76171 | 76172 | 76177 | 76180 | 76185 |
| --- | --- | --- | --- | --- | --- |
| Appui 6136 | non | non | oui | oui | oui |
| Appui 6137 | non | non | oui | oui | oui |
| Appui 76768 | non | non | oui | oui | oui |
| Appui 76751 | oui | oui | non | non | non |
| Appui 76752 | oui | oui | non | non | non |
| Appui 76753 | oui | oui | non | non | non |
| Appui 76758 + 76719 | oui | oui | non | non | non |
| Nez d'appui 4319 | non | non | oui | oui | oui |
| Seuil A076 + rejet d'eau A062 | oui | oui | oui | oui | oui |
| Compensateur 6143 | non | non | oui | non | oui |
| Compensateur 6144 | non | non | oui | non | oui |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 14, 15, 16, 18, 20, 21 et 22)

Le seuil aluminium est le seul profil complémentaire commun aux cinq dormants.

Les appuis 6137 et 76768 apparaissent malgré tout sur un dormant 76171 à partir de 175 mm
d'isolant, parce que le dormant bas devient alors un 76180 — voir
[Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md).

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, pages du PDF 14 à 22 et 24

[2] DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED —
`raw/dta-trocal-76-advanced-6-16-2334-v5.pdf`, p. 8, 9, 14 et 18

[3] [DTD n° DBV-25-6/16-2334_V5, système 76 Advanced](raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf)

# Voir aussi

- [Profilés complémentaires du système 76](/profiles/systeme-76-profiles-complementaires.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md)
- [Élargisseurs et assemblage PERFORM76](/profiles/perform76-elargisseurs-et-assemblage.md)
- [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md)
- [Assemblages du système 76](/profiles/systeme-76-assemblages.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
