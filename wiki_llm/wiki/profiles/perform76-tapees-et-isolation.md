---
type: Profilé
title: Tapées et isolation PERFORM76
description: Les sept tapées de pose PERFORM76 et l'épaisseur d'isolant qu'elles permettent, qui change selon le dormant, avec les appuis et pattes de pose associés.
tags: [perform76, tapee, isolation, patte-de-pose, appui, clameau]
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
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
verified:
  by: process:claude-code
  at: 2026-09-18T22:00:00Z
---

# Une tapée ne donne pas la même isolation selon le dormant

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

| Tapée | Cote propre (mm) | Iso sur 76177 et 76185 (mm) | Iso sur 76180 (mm) | Iso sur 76171 (mm) |
| --- | --- | --- | --- | --- |
| sans tapée | - | 65 | 65 | 80 |
| 6138 | 15 | 80 | 80 | 95 |
| 6139 | 35 | 100 | 100 | 115 |
| 6140 | 55 | 120 | 120 | 135 |
| 6141 | 75 | 140 | 140 | 155 |
| 6142 | 95 | 160 | 160 | 175 |
| 76772 | 115 | 180 | 180 | 195 |
| 76769 | 135 | 200 | 200 | 215 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 14, 16 et 21)

**Sur un dormant 76171, les seules épaisseurs d'isolation qui existent sont 80, 95, 115, 135, 155,
175, 195 et 215 mm.** Une demande à 140 mm sur ce dormant se traite en 135 ou en 155 mm : le
140 mm appartient au 76180 et aux dormants rénovation.

Le DTD du système donne les mêmes tapées à 0,5 mm près et suffixées **.1** sur quatre d'entre
elles : 6139.1, 6140.1, 6141.1 et 6142.1 pour 35,5, 55,5, 75,5 et 95,5 mm d'épaisseur propre —
contre 35, 55, 75 et 95 mm au cahier technique. Écart mineur d'arrondi entre les deux sources,
sans incidence sur l'isolation obtenue [2 p. 4].

# Correspondance avec les pièces d'appui

Chaque tapée n'est pas compatible avec chacun des quatre appuis de pièce d'appui du système.
Correspondance relevée sur le DTD, `X` marquant une compatibilité.

| Tapée | Épaisseur (mm) | Appui 6137 | Appui 6136 | Appui 76758 | Appui 76768 |
| --- | --- | --- | --- | --- | --- |
| 6138 | 15 | X | X | X | X |
| 6139.1 | 35,5 | X | X | X | X |
| 6140.1 | 55,5 | X | X | X | X |
| 6141.1 | 75,5 | X | X | X | - |
| 6142.1 | 95,5 | X | X | - | - |
| 76772 | 115,5 | X | - | - | - |
| 76769 | 135,5 | X | - | - | - |

(schéma: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 4)

**Plus la tapée est épaisse, moins elle admet d'appuis** : les deux tapées les plus fines
(6138 et 6139.1) se posent sur les quatre appuis, les deux plus épaisses (76772 et 76769)
n'admettent plus que le 6137.

# Embouts des pièces d'appui

Chaque appui de pièce d'appui a son propre embout, en PVC expansé, qui obture ses chambres.

| Appui | Embout | Nombre de pièces |
| --- | --- | --- |
| 6136 | 9F55.1 | 1 |
| 6137 | 9F56.1 | 2 |
| 76758 | AC011 | 3 |
| 76768 | M780, M781, M782 | 3 |

(schéma: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf, p. 4 et 23)

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

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 21)

C'est une contrainte de conception : elle change la nomenclature du châssis, et se vérifie au
chiffrage d'un projet en isolation renforcée.

# Appuis sur dormants 76177 et 76185

| Épaisseur d'isolation (mm) | Appui |
| --- | --- |
| 60 | 6136 |
| 80 | 6136 |
| 100 | 6136 |
| 120 | 6136 |
| 140 | 6137 |
| 160 | 6137 |
| 180 | 76768 |
| 200 | 76768 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 14)

Aucune patte de pose n'est affectée à ces deux dormants rénovation : la fixation suit les
principes de pose rénovation, décrits dans
[Pose de la PERFORM76](/procedures/pose-perform76.md).

Ce tableau part de 60 mm alors que la planche de tapées annonce 65 mm sans tapée — entrée
**INC-06** du registre [Incohérences internes](/anomalies/incoherences-internes.md).

# Appuis et pattes de pose sur dormant 76180

Clameau réf. **CP14GGOM0012**, **sans cale**.

| Épaisseur d'isolation (mm) | Patte de pose | Appui |
| --- | --- | --- |
| 60 | NT1939 | 6136 |
| 80 | NT1939 | 6136 |
| 100 | NT1943 | 6136 |
| 120 | NT1945 | 6136 |
| 140 | NT1947 | 6137 |
| 160 | NT1949 | 6137 |
| 180 | NT1951 | 76768 |
| 200 | NT1953 | 76768 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 16)

# Appuis et pattes de pose sur dormant 76171

Clameau réf. **CP14GGOM0012**, cale réf. **CTHNT0030**.

| Épaisseur d'isolation (mm) | Patte de pose | Appui | Cale |
| --- | --- | --- | --- |
| 80 | NT1939 | 76758 | **sans cale** |
| 95 | NT1939 | 76758 | CTHNT0030 |
| 115 | NT1943 | 76758 | CTHNT0030 |
| 135 | NT1945 | 76758 | CTHNT0030 |
| 155 | NT1947 | 76758 | CTHNT0030 |
| 175 | NT1949 | 6137 | CTHNT0030 |
| 195 | NT1951 | 76768 | CTHNT0030 |
| 215 | NT1953 | 76768 | CTHNT0030 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 21)

Le cas à 80 mm est le seul **sans cale** du tableau. Les trois dernières lignes imposent le
dormant bas 76180.

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

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 16 et 21)

**Aucune épaisseur intermédiaire n'existe** : les valeurs de ce tableau sont les seules
documentées.

# Ce que la source ne donne pas

- Aucune **planche de tapées pour le dormant 76172**.
- Aucune **cote pour les pattes NT1939 à NT1953**, seulement leur affectation.
- Aucune patte de pose pour les dormants rénovation 76177 et 76185.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 14, 16 et 21

[2] DTD n° DBV-25-6/16-2334_V5, système 76 Advanced —
`raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf`, p. 4 et 23

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md)
- [Pose de la PERFORM76](/procedures/pose-perform76.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
- [DTD n° DBV-25-6/16-2334_V5](/sources/dtd-6-16-2334.md)
