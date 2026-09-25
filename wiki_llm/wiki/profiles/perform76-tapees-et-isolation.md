---
type: Profilé
title: Tapées et isolation PERFORM76
description: Les sept tapées de pose PERFORM76 et l'épaisseur d'isolant qu'elles permettent, qui change selon le dormant, avec les appuis et pattes de pose associés.
tags: [perform76, tapee, isolation, patte-de-pose, appui, clameau]
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
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
    last_modified: 2025-06-19
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
    last_modified: 2025-06-19
source_pages:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    pages: 17, 19, 24
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    pages: 17
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    pages: 4, 16, 23
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
---

# Une tapée ne donne pas la même isolation selon le dormant

Quand le mur est doublé d'un isolant à l'intérieur, la fenêtre doit arriver jusqu'au nu de cet
isolant. La **tapée** est le profil PVC rapporté sur le dormant qui prolonge la menuiserie de
l'épaisseur nécessaire : plus l'isolant est épais, plus la tapée est haute. Cette page donne, pour
chaque dormant et chaque épaisseur d'isolation, la tapée, la patte de pose et l'appui à monter,
avec les coupes du cahier technique.

Montée sur un dormant **76171**, une tapée PERFORM76 donne **15 mm d'isolant de plus** que le même
profil monté sur un 76177, un 76185 ou un 76180. L'épaisseur d'isolation se lit donc dans la
colonne du dormant réellement employé.

Trois familles de pièces se lisent dans les colonnes voisines d'un même tableau et ne se
substituent pas :

| Famille | Rôle | Références |
| --- | --- | --- |
| Tapée | donne l'épaisseur d'isolation | 6138 à 6142, 76772, 76769 |
| Appui | reçoit le rejet d'eau sous le dormant | 6136, 6137, 76758, 76768 |
| Patte de pose | fixe le dormant dans le gros œuvre | NT1939 à NT1953 |

# Cotes des tapées par dormant

Épaisseur d'isolation permise par chaque tapée PERFORM76, en mm, selon le dormant support.

| Tapée | Cote propre (mm) | Iso sur 76177 et 76185 (mm) | Iso sur 76180 (mm) | Iso sur 76171 (mm) | Coupe |
| --- | --- | --- | --- | --- | ---: |
| sans tapée | - | 65 | 65 | 80 | - |
| 6138 | 15 | 80 | 80 | 95 | ![Tapée 6138](/assets/profiles/perform76/tapees/tapee-6138.png) |
| 6139 | 35 | 100 | 100 | 115 | ![Tapée 6139](/assets/profiles/perform76/tapees/tapee-6139.png) |
| 6140 | 55 | 120 | 120 | 135 | ![Tapée 6140](/assets/profiles/perform76/tapees/tapee-6140.png) |
| 6141 | 75 | 140 | 140 | 155 | ![Tapée 6141](/assets/profiles/perform76/tapees/tapee-6141.png) |
| 6142 | 95 | 160 | 160 | 175 | ![Tapée 6142](/assets/profiles/perform76/tapees/tapee-6142.png) |
| 76772 | 115 | 180 | 180 | 195 | ![Tapée 76772](/assets/profiles/perform76/tapees/tapee-76772.png) |
| 76769 | 135 | 200 | 200 | 215 | ![Tapée 76769](/assets/profiles/perform76/tapees/tapee-76769.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 17, 19 et 24)

**Sur un dormant 76171, les seules épaisseurs d'isolation qui existent sont 80, 95, 115, 135, 155,
175, 195 et 215 mm.** Une demande à 140 mm sur ce dormant se traite en 135 ou en 155 mm : le
140 mm appartient au 76180 et aux dormants rénovation.

Dans le DTD et le DTA du système, quatre de ces tapées portent le suffixe **.1** et une épaisseur
propre supérieure de 0,5 mm : 6139.1, 6140.1, 6141.1 et 6142.1 pour 35,5, 55,5, 75,5 et 95,5 mm,
76772 et 76769 pour 115,5 et 135,5 mm, contre 35, 55, 75, 95, 115 et 135 mm au cahier technique
[2 p. 4]. Les sept tapées y sont dessinées sous le nom de **fourrures d'épaisseur PVC** [3 p. 17] [2 p. 16].
L'isolation obtenue par dormant est celle du tableau ci-dessus, relevée sur le cahier technique.

# Correspondance avec les pièces d'appui

Une **pièce d'appui** PVC se visse dans les alvéovis (logements de vis) du nez de la tapée ; ses
chambres doivent correspondre au nez de la tapée, et toutes les tapées n'ont pas de
correspondance avec les quatre pièces d'appui du système. Le tableau des correspondances des
chambres des pièces d'appui avec les nez des fourrures d'épaisseur se lit par ligne : une tapée,
son épaisseur, puis un `X` sous chaque pièce d'appui qui lui correspond ; une case vide (`-`)
marque l'absence de correspondance.

| Tapée | Épaisseur tapée (mm) | Pièce d'appui 6137 | Pièce d'appui 6136 | Pièce d'appui 76758 | Pièce d'appui 76768 |
| --- | --- | --- | --- | --- | --- |
| 6138 | 15 | X | X | X | X |
| 6139.1 | 35,5 | X | X | X | X |
| 6140.1 | 55,5 | X | X | X | X |
| 6141.1 | 75,5 | X | - | X | X |
| 6142.1 | 95,5 | X | - | - | X |
| 76772 | 115,5 | - | - | - | X |
| 76769 | 135,5 | - | - | - | X |

(schéma: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 4)

**Les tapées 76772 et 76769 ne correspondent qu'à la pièce d'appui 76768.** La 6141.1 ne
correspond pas à la 6136 ; la 6142.1 ne correspond qu'aux 6137 et 76768 [2 p. 4].

L'étanchéité pièce d'appui / tapée est assurée par la pièce M298 ou M613 comprimée lors du vissage
de la pièce d'appui dans les alvéovis de la fourrure d'épaisseur ; les fourrures sont clippées et
vissées avec un entraxe maximum de 300 mm (DTD § 2.2.3.1.3, sur
[DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md)) [2 p. 4].

# Embouts des pièces d'appui

L'obturation des chambres de pièces d'appui, quelle que soit la correspondance entre les parois de
la pièce d'appui et le nez de la fourrure d'épaisseur, est réalisée par des embouts ajustés en PVC
expansé [2 p. 4]. Chaque embout est une cale pentée qui suit la pente de la pièce d'appui : il se
glisse dans les chambres de la pièce d'appui, au droit du nez de la fourrure d'épaisseur. Une ligne
par pièce d'embout, de gauche à droite sur la planche : sa largeur (cote horizontale), sa longueur
de 60 mm (profondeur d'insertion), son épaisseur à chaque extrémité de la pente, et, sur les 9F55.1
et 9F56.1, la feuillure de 4,5 × 9,5 mm taillée sous l'extrémité mince [2 p. 23].

| Pièce d'appui | Embout | Nombre de pièces | Pièce | Largeur (mm) | Longueur (mm) | Épaisseur, extrémité mince (mm) | Épaisseur, extrémité épaisse (mm) | Feuillure (mm) | Coupe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| 6136 | 9F55.1 | 1 | unique | 60,8 | 60 | 11,3 | 14,5 | 4,5 × 9,5 | ![Embout 9F55.1 pour 6136](/assets/profiles/systeme76/appuis/embout-9f55-1.png) |
| 6137 | 9F56.1 | 2 | première | 46 | 60 | 9,7 | 12,1 | 4,5 × 9,5 | ![Embout 9F56.1 pour 6137](/assets/profiles/systeme76/appuis/embout-9f56-1.png) |
| 6137 | 9F56.1 | 2 | seconde | 42,5 | 60 | 12,3 | 14,5 | - | ![Embout 9F56.1 pour 6137](/assets/profiles/systeme76/appuis/embout-9f56-1.png) |
| 76758 | AC011 | 3 | première | 23 | 60 | 1,5 | 3,5 | - | ![Embout AC011 pour 76758](/assets/profiles/systeme76/appuis/embout-ac011.png) |
| 76758 | AC011 | 3 | deuxième | 18,5 | 60 | 3,5 | 5,5 | - | ![Embout AC011 pour 76758](/assets/profiles/systeme76/appuis/embout-ac011.png) |
| 76758 | AC011 | 3 | troisième | 18,5 | 60 | 5,5 | 7 | - | ![Embout AC011 pour 76758](/assets/profiles/systeme76/appuis/embout-ac011.png) |
| 76768 | M780 | - | M780 | - | 60 | - | - | - | ![Embouts M780, M781, M782 pour 76768](/assets/profiles/systeme76/appuis/embouts-m780-m781-m782-76768.png) |
| 76768 | M781 | - | M781 | - | 60 | - | - | - | ![Embouts M780, M781, M782 pour 76768](/assets/profiles/systeme76/appuis/embouts-m780-m781-m782-76768.png) |
| 76768 | M782 | - | M782 | - | 60 | - | - | - | ![Embouts M780, M781, M782 pour 76768](/assets/profiles/systeme76/appuis/embouts-m780-m781-m782-76768.png) |

(schéma: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 23)

Chaque coupe montre à gauche le dormant, la pièce d'appui vissée sous lui et, hachurées dans ses
chambres, les pièces d'embout ; à droite, chaque pièce d'embout de profil (la pente) puis de face
(le rectangle de 60 mm). Les embouts M780, M781 et M782 de la pièce d'appui 76768 sont dessinés en
trois références distinctes, en PVC expansé (hachure pointillée), avec leur seule longueur de
60 mm : leur largeur et leurs épaisseurs ne sont pas cotées, et la planche ne porte pas de nombre
de pièces pour eux. La planche est titrée « Embouts de pièces d'appui (PVC expansé) » ; les légendes
sont « EMBOUT 9F55.1 POUR 6136 (1 PIECE) », « EMBOUT 9F56.1 POUR 6137 (2 PIECES) », « EMBOUT AC011
POUR 76758 (3 PIECES) », « M780, M781, M782 EMBOUTS POUR 76768 » [2 p. 23]. Le texte du § 2.2.3.1.3
ne cite que les 9F55.1, 9F56.1 et AC011 [2 p. 4]. La coupe « Assemblage fourrure et pièce d'appui PVC » écrit l'embout du 6137
« 9F56 », sans suffixe — entrée **INC-25**.

La cote propre est la hauteur de la tapée, constante quel que soit le dormant. Les cotes de
montage varient : **35 / 16** sur les dormants rénovation, **35 / 39** sur le 76180, **35 / 45**
sur le 76171, à l'exception de la tapée 76769 qui y est cotée 35 / 39.

# La contrainte du 76171 au-delà de 155 mm

Sur un dormant 76171, **au-delà de 155 mm d'isolant le dormant bas devient un 76180 à aile de
20 mm**.

| Iso sur 76171 (mm) | Appui | Dormant bas |
| --- | --- | --- |
| 80 | 76758 | 76171 |
| 95 | 76758 | 76171 |
| 115 | 76758 | 76171 |
| 135 | 76758 | 76171 |
| 155 | 76758 | 76171 |
| 175 | 6137 | **76180, aile de 20 mm** |
| 195 | 76768 | **76180, aile de 20 mm** |
| 215 | 76768 | **76180, aile de 20 mm** |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 24)

C'est une contrainte de conception : elle change la nomenclature du châssis, et se vérifie au
chiffrage d'un projet en isolation renforcée.

# Appuis sur dormants 76177 et 76185

Sur les deux dormants rénovation, l'épaisseur d'isolation choisit la tapée et l'appui (le profil
penté posé sous le dormant bas, qui rejette l'eau). Une ligne se lit : pour cette épaisseur
d'isolation, monter cette tapée et cet appui.

| Épaisseur d'isolation (mm) | Tapée | Appui |
| --- | --- | --- |
| 60 | - | 6136 |
| 80 | 6138 | 6136 |
| 100 | 6139 | 6136 |
| 120 | 6140 | 6136 |
| 140 | 6141 | 6137 |
| 160 | 6142 | 6137 |
| 180 | 76772 | 76768 |
| 200 | 76769 | 76768 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 17)

La colonne *Tapée* est relevée sur la planche des tapées de la même page, où chaque coupe porte
son épaisseur d'isolation (« Iso de 80 » pour la 6138, et ainsi de suite).

Aucune patte de pose n'est affectée à ces deux dormants rénovation : la fixation suit les
principes de pose rénovation, décrits dans
[Pose de la PERFORM76](/procedures/pose-perform76.md).

Ce tableau part de 60 mm alors que la planche de tapées dessine « Iso de 65 » sans tapée — entrée
**INC-06** du registre [Incohérences internes](/anomalies/incoherences-internes.md).

## Coupes des tapées sur dormants 76177 et 76185

Chaque coupe montre le dormant rénovation vu en tranche et la tapée posée au-dessus. Trois cotes
sont portées : à gauche une hauteur verticale, en haut la tapée elle-même (35 de large et 16 de
retour), à droite sa hauteur propre ; l'étiquette verticale « Iso de … » donne l'épaisseur
d'isolation. Cotes en mm :

| Tapée | Hauteur cotée à gauche (mm) | Largeur de la tapée / retour (mm) | Hauteur propre de la tapée (mm) | Iso de (mm) |
| --- | --- | --- | --- | --- |
| sans tapée | 60 | largeur de dormant 51 | - | 65 |
| 6138 | 75 | 35 / 16 | 15 | 80 |
| 6139 | 95 | 35 / 16 | 35 | 100 |
| 6140 | 115 | 35 / 16 | 55 | 120 |
| 6141 | 135 | 35 / 16 | 75 | 140 |
| 6142 | 155 | 35 / 16 | 95 | 160 |
| 76772 | 175 | 35 / 16 | 115 | 180 |
| 76769 | 195 | 35 / 16 | 135 | 200 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 17)

### Isolation 65 mm, sans tapée

![Isolation 65 mm, sans tapée](/assets/profiles/perform76/isolation/76177-76185-sans-tapee.png)

Tapée : sans tapée ; épaisseur d'isolation : 65 mm.

### Isolation 80 mm, tapée 6138

![Isolation 80 mm, tapée 6138](/assets/profiles/perform76/isolation/76177-76185-6138.png)

Tapée : 6138 ; épaisseur d'isolation : 80 mm.

### Isolation 100 mm, tapée 6139

![Isolation 100 mm, tapée 6139](/assets/profiles/perform76/isolation/76177-76185-6139.png)

Tapée : 6139 ; épaisseur d'isolation : 100 mm.

### Isolation 120 mm, tapée 6140

![Isolation 120 mm, tapée 6140](/assets/profiles/perform76/isolation/76177-76185-6140.png)

Tapée : 6140 ; épaisseur d'isolation : 120 mm.

### Isolation 140 mm, tapée 6141

![Isolation 140 mm, tapée 6141](/assets/profiles/perform76/isolation/76177-76185-6141.png)

Tapée : 6141 ; épaisseur d'isolation : 140 mm.

### Isolation 160 mm, tapée 6142

![Isolation 160 mm, tapée 6142](/assets/profiles/perform76/isolation/76177-76185-6142.png)

Tapée : 6142 ; épaisseur d'isolation : 160 mm.

### Isolation 180 mm, tapée 76772

![Isolation 180 mm, tapée 76772](/assets/profiles/perform76/isolation/76177-76185-76772.png)

Tapée : 76772 ; épaisseur d'isolation : 180 mm.

### Isolation 200 mm, tapée 76769

![Isolation 200 mm, tapée 76769](/assets/profiles/perform76/isolation/76177-76185-76769.png)

Tapée : 76769 ; épaisseur d'isolation : 200 mm.

# Appuis et pattes de pose sur dormant 76180

Sur le dormant neuf 76180, l'épaisseur d'isolation choisit la tapée, la **patte de pose** (la
pièce métallique qui fixe le dormant au mur à travers l'isolant) et l'appui. La patte s'accroche
au dormant par le clameau réf. **CP14GGOM0012**, **sans cale**.

| Épaisseur d'isolation (mm) | Tapée | Patte de pose | Appui |
| --- | --- | --- | --- |
| 60 | - | NT1939 | 6136 |
| 80 | 6138 | NT1939 | 6136 |
| 100 | 6139 | NT1943 | 6136 |
| 120 | 6140 | NT1945 | 6136 |
| 140 | 6141 | NT1947 | 6137 |
| 160 | 6142 | NT1949 | 6137 |
| 180 | 76772 | NT1951 | 76768 |
| 200 | 76769 | NT1953 | 76768 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 19)

Comme pour les dormants rénovation, le tableau part de 60 mm alors que la coupe sans tapée porte
« Iso de 65 » (**INC-06**).

## Coupes des tapées et pattes sur dormant 76180

Chaque coupe montre le dormant 76180, la tapée au-dessus et la patte de pose en gris, qui part
vers la gauche le long de l'isolant ; la patte porte sa référence gravée. En haut, les cotes
horizontales : la patte, puis la tapée (35), puis le dormant (39, ou 74 sans tapée). À gauche,
la hauteur, égale à l'épaisseur d'isolation. Cotes en mm :

| Épaisseur d'isolation (mm) | Tapée | Cotes horizontales en haut (mm) | Hauteur propre de la tapée (mm) |
| --- | --- | --- | --- |
| 65 | sans tapée | 70 / 74 | - |
| 80 | 6138 | 55 / 35 / 39 | 15 |
| 100 | 6139 | 80 / 35 / 39 | 35 |
| 120 | 6140 | 80 / 35 / 39 | 55 |
| 140 | 6141 | 80 / 35 / 39 | 75 |
| 160 | 6142 | 80 / 35 / 39 | 95 |
| 180 | 76772 | 80 / 35 / 39 | 115 |
| 200 | 76769 | 80 / 35 / 39 | 135 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 19)

### Isolation 65 mm, sans tapée

![Isolation 65 mm, sans tapée](/assets/profiles/perform76/isolation/76180-sans-tapee.png)

Tapée : sans tapée ; épaisseur d'isolation : 65 mm ; cotes en haut : 70 / 74 mm.

### Isolation 80 mm, tapée 6138

![Isolation 80 mm, tapée 6138](/assets/profiles/perform76/isolation/76180-6138.png)

Tapée : 6138 ; épaisseur d'isolation : 80 mm ; cotes en haut : 55 / 35 / 39 mm.

### Isolation 100 mm, tapée 6139

![Isolation 100 mm, tapée 6139](/assets/profiles/perform76/isolation/76180-6139.png)

Tapée : 6139 ; épaisseur d'isolation : 100 mm ; cotes en haut : 80 / 35 / 39 mm.

### Isolation 120 mm, tapée 6140

![Isolation 120 mm, tapée 6140](/assets/profiles/perform76/isolation/76180-6140.png)

Tapée : 6140 ; épaisseur d'isolation : 120 mm ; cotes en haut : 80 / 35 / 39 mm.

### Isolation 140 mm, tapée 6141

![Isolation 140 mm, tapée 6141](/assets/profiles/perform76/isolation/76180-6141.png)

Tapée : 6141 ; épaisseur d'isolation : 140 mm ; cotes en haut : 80 / 35 / 39 mm.

### Isolation 160 mm, tapée 6142

![Isolation 160 mm, tapée 6142](/assets/profiles/perform76/isolation/76180-6142.png)

Tapée : 6142 ; épaisseur d'isolation : 160 mm ; cotes en haut : 80 / 35 / 39 mm.

### Isolation 180 mm, tapée 76772

![Isolation 180 mm, tapée 76772](/assets/profiles/perform76/isolation/76180-76772.png)

Tapée : 76772 ; épaisseur d'isolation : 180 mm ; cotes en haut : 80 / 35 / 39 mm.

### Isolation 200 mm, tapée 76769

![Isolation 200 mm, tapée 76769](/assets/profiles/perform76/isolation/76180-76769.png)

Tapée : 76769 ; épaisseur d'isolation : 200 mm ; cotes en haut : 80 / 35 / 39 mm.

# Appuis et pattes de pose sur dormant 76171

Sur le dormant neuf sans aile 76171, même principe, avec en plus une **cale** réf.
**CTHNT0030**, sauf à 80 mm d'isolation. Clameau réf. **CP14GGOM0012**.

| Épaisseur d'isolation (mm) | Tapée | Patte de pose | Appui | Cale |
| --- | --- | --- | --- | --- |
| 80 | - | NT1939 | 76758 | **sans cale** |
| 95 | 6138 | NT1939 | 76758 | CTHNT0030 |
| 115 | 6139 | NT1943 | 76758 | CTHNT0030 |
| 135 | 6140 | NT1945 | 76758 | CTHNT0030 |
| 155 | 6141 | NT1947 | 76758 | CTHNT0030 |
| 175 | 6142 | NT1949 | 6137 | CTHNT0030 |
| 195 | 76772 | NT1951 | 76768 | CTHNT0030 |
| 215 | 76769 | NT1953 | 76768 | CTHNT0030 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 24)

Le cas à 80 mm est le seul **sans cale** du tableau. Les trois dernières lignes imposent le
dormant bas 76180 (astérisque de la planche : « Dormant en partie basse 76180 aile de 20 mm »).

## Coupes des tapées et pattes sur dormant 76171

Mêmes coupes que sur le 76180, avec la cote du dormant à 45 mm au lieu de 39 (74 sans tapée), et
une cote rouge de **6 mm** au pied de la patte sur toutes les coupes sauf celle sans tapée — qui
est aussi la seule configuration sans cale. Cotes en mm :

| Épaisseur d'isolation (mm) | Tapée | Cotes horizontales en haut (mm) | Hauteur propre de la tapée (mm) | Cote rouge au pied (mm) |
| --- | --- | --- | --- | --- |
| 80 | sans tapée | 70 / 74 | - | - |
| 95 | 6138 | 55 / 35 / 45 | 15 | 6 |
| 115 | 6139 | 80 / 35 / 45 | 35 | 6 |
| 135 | 6140 | 80 / 35 / 45 | 55 | 6 |
| 155 | 6141 | 80 / 35 / 45 | 75 | 6 |
| 175 | 6142 | 80 / 35 / 45 | 95 | 6 |
| 195 | 76772 | 80 / 35 / 45 | 115 | 6 |
| 215 | 76769 | 80 / 35 / 39 | 135 | 6 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 24)

### Isolation 80 mm, sans tapée

![Isolation 80 mm, sans tapée](/assets/profiles/perform76/isolation/76171-sans-tapee.png)

Tapée : sans tapée ; épaisseur d'isolation : 80 mm ; cotes en haut : 70 / 74 mm.

### Isolation 95 mm, tapée 6138

![Isolation 95 mm, tapée 6138](/assets/profiles/perform76/isolation/76171-6138.png)

Tapée : 6138 ; épaisseur d'isolation : 95 mm ; cotes en haut : 55 / 35 / 45 mm.

### Isolation 115 mm, tapée 6139

![Isolation 115 mm, tapée 6139](/assets/profiles/perform76/isolation/76171-6139.png)

Tapée : 6139 ; épaisseur d'isolation : 115 mm ; cotes en haut : 80 / 35 / 45 mm.

### Isolation 135 mm, tapée 6140

![Isolation 135 mm, tapée 6140](/assets/profiles/perform76/isolation/76171-6140.png)

Tapée : 6140 ; épaisseur d'isolation : 135 mm ; cotes en haut : 80 / 35 / 45 mm.

### Isolation 155 mm, tapée 6141

![Isolation 155 mm, tapée 6141](/assets/profiles/perform76/isolation/76171-6141.png)

Tapée : 6141 ; épaisseur d'isolation : 155 mm ; cotes en haut : 80 / 35 / 45 mm.

### Isolation 175 mm, tapée 6142

![Isolation 175 mm, tapée 6142](/assets/profiles/perform76/isolation/76171-6142.png)

Tapée : 6142 ; épaisseur d'isolation : 175 mm ; cotes en haut : 80 / 35 / 45 mm.

### Isolation 195 mm, tapée 76772

![Isolation 195 mm, tapée 76772](/assets/profiles/perform76/isolation/76171-76772.png)

Tapée : 76772 ; épaisseur d'isolation : 195 mm ; cotes en haut : 80 / 35 / 45 mm.

### Isolation 215 mm, tapée 76769

![Isolation 215 mm, tapée 76769](/assets/profiles/perform76/isolation/76171-76769.png)

Tapée : 76769 ; épaisseur d'isolation : 215 mm ; cotes en haut : 80 / 35 / 39 mm.

# Compatibilités des pattes de pose

Une ligne par couple patte et dormant : une même patte dessert les deux dormants neufs à des
épaisseurs d'isolation différentes.

| Patte | Dormant | Épaisseur d'isolation (mm) |
| --- | --- | --- |
| NT1939 | 76180 | 60 |
| NT1939 | 76180 | 80 |
| NT1939 | 76171 | 80 |
| NT1939 | 76171 | 95 |
| NT1943 | 76180 | 100 |
| NT1943 | 76171 | 115 |
| NT1945 | 76180 | 120 |
| NT1945 | 76171 | 135 |
| NT1947 | 76180 | 140 |
| NT1947 | 76171 | 155 |
| NT1949 | 76180 | 160 |
| NT1949 | 76171 | 175 |
| NT1951 | 76180 | 180 |
| NT1951 | 76171 | 195 |
| NT1953 | 76180 | 200 |
| NT1953 | 76171 | 215 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 19 et 24)

**Aucune épaisseur intermédiaire n'existe** : les valeurs de ce tableau sont les seules
documentées.

# Tapées aluminium sur dormant capoté

Un dormant habillé d'un capot aluminium (variante AluClip) reçoit des tapées **aluminium** A469 à
A473, et non les tapées PVC ci-dessus. Leurs cotes sont dans
[Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md#pièces-dappui-et-tapées-aluminium),
leur mise en œuvre et leurs cotes de débit dans
[Capotage AluClip du système 76 Advanced](/procedures/capotage-aluclip-systeme-76.md#tapées-et-appuis-aluminium-sur-dormant-capoté).

# Ce que la source ne donne pas

- Aucune **planche de tapées pour le dormant 76172**.
- Le **nom des cotes** portées sur les coupes (hauteur de gauche, cote de la patte, cote rouge
  de 6 mm) : elles sont relevées sans légende.
- Aucune patte de pose pour les dormants rénovation 76177 et 76185.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, pages du PDF 17, 19 et 24

[2] DTD n° DBV-25-6/16-2334_V5, système 76 Advanced —
`raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf`, p. 4, 16 et 23

[3] [DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED](raw/dta-trocal-76-advanced-6-16-2334-v5.pdf)

# Voir aussi

- [Profilés complémentaires du système 76](/profiles/systeme-76-profiles-complementaires.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md)
- [Pose de la PERFORM76](/procedures/pose-perform76.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
- [DTD n° DBV-25-6/16-2334_V5](/sources/dtd-6-16-2334.md)
