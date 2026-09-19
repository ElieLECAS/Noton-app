---
type: Profilé
title: Appuis et seuils PERFORM76
description: Les sept appuis PERFORM76, le nez d'appui 4319, les quatre seuils aluminium A075 à A343 et les deux compensateurs de rénovation, avec leur affectation par dormant quand elle est connue.
tags: [perform76, appui, seuil, nez-d-appui, compensateur, rejet-d-eau]
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
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
verified:
  by: process:claude-code
  at: 2026-09-18T22:00:00Z
---

# Deux familles d'appuis, selon le dormant

Les appuis PERFORM76 forment deux groupes qui ne se mélangent pas.

| Groupe | Appuis | Dormants |
| --- | --- | --- |
| Rénovation et neuf avec aile | 6136, 6137, 76768 | 76177, 76180, 76185 |
| Neuf sans aile | 76751, 76752, 76753, 76758 + 76719 | 76171, 76172 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13, 17, 18 et 19)

# Cotes des appuis 6136, 6137 et 76768

Appuis du premier groupe, cotes en mm. Les trois sont pentés à **3°**.

| Appui | Largeur (mm) | Longueur (mm) | Épaisseur (mm) | Retombée (mm) | Nez d'appui 4319 |
| --- | --- | --- | --- | --- | --- |
| 6136 | 67 | 127 | 14 | 3,5 | oui |
| 6137 | 97 | 157 | 14 | 3,5 | oui |
| 76768 | 136 | 196 | 14 | 22 | non |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

La **hauteur d'about** de ces trois appuis dépend du dormant :

| Dormant | Aile (mm) | Hauteur d'about (mm) |
| --- | --- | --- |
| 76177 | 40 | 26 |
| 76185 | 60 | 46 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

Les 20 mm d'écart de hauteur d'about reprennent exactement les 20 mm d'écart d'aile.

# Cotes des appuis 76751, 76752, 76753 et 76758

Appuis du second groupe, cotes en mm.

| Appui | Hauteur (mm) | Décomposition (mm) | Pente | Particularité |
| --- | --- | --- | --- | --- |
| 76751 | 30 | 20 / 56 | - | livré non monté |
| 76752 | 50 | 20 / 56 | - | livré non monté |
| 76753 | 35 | 21 / 46, sur 76 | - | - |
| 76758 + 76719 | 15 à 20 | 80 en saillie, dénivelé 3, retombée 5,5, largeur totale 156 | **5°** | ensemble de deux profils |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 19 et 21)

**L'ensemble 76758 + 76719 est le seul appui penté à 5°** ; les six autres appuis de la gamme sont
à 3°.

Les appuis **76751 et 76752 sont livrés non montés**, à intégrer au temps d'atelier. La mention du
cahier technique cite un « 76152 » qui n'existe nulle part ailleurs — entrée **INC-04** du
registre [Incohérences internes](/anomalies/incoherences-internes.md).

L'appui **76758 ne se commande pas seul** : il forme un ensemble avec le **76719**. C'est l'appui
du dormant 76171 en pose isolée jusqu'à 155 mm d'isolant — voir
[Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md).

# Cotes du nez d'appui et du seuil

| Référence | Type | Cotes (mm) | Compatibilité |
| --- | --- | --- | --- |
| 4319 | nez d'appui | 18 / 8 | appuis 6136 et 6137 uniquement |
| A076 | seuil aluminium | 76 de large, 10 et 10, about 20 | les cinq dormants |
| A062 | rejet d'eau | - | s'associe au seuil A076 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13, 19 et 22)

Le nez d'appui **4319 ne se monte pas sur le 76768** : il n'est représenté que sur les appuis
6136 et 6137.

Le seuil **A076 et le rejet d'eau A062 vont toujours ensemble** ; ils ne sont jamais figurés
séparément sur les cinq planches dormant. Aucune cote n'est donnée pour le A062.

# Pièces d'appui et tapées aluminium

Pour les dormants capotés, le DTA donne une famille d'appuis et de tapées en aluminium, distincte
des appuis et tapées PVC ci-dessus.

| Pièce | Fonction | Largeur ou hauteur (mm) |
| --- | --- | --- |
| A475 | pièce d'appui alu | 97 |
| A476 | pièce d'appui alu | 137 |
| A477 | pièce d'appui alu, réservée à la rénovation | 77 |
| A491 | pièce d'appui alu | 57 |
| DT100 | bavette | 100 |
| A469 | tapée alu | 30 |
| A470 | tapée alu | 50 |
| A471 | tapée alu | 70 |
| A472 | tapée alu | 90 |
| A473 | tapée alu | 110 |

(schéma: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf, p. 18)

**L'appui A476 ne peut pas être assemblé avec la tapée A469** ; il s'assemble avec les tapées
A470, A471, A472 ou A473 [2 p. 8-9]. L'appui A477 est réservé aux mises en œuvre en rénovation
sur dormant existant.

**Ces mêmes références A469 à A473 désignent des « embouts d'extrémité de pièce d'appui » sur le
poster Gamme 70 KÖMMERLING**, une fonction différente de la tapée qu'elles nomment ici. Système
distinct, plan illisible à confirmer : à vérifier avant toute commande croisée entre les deux
systèmes — entrée **VER-40** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Le DTA du système donne trois seuils supplémentaires, absents du cahier technique PERFORM76.

| Seuil | Largeur (mm) | Hauteur (mm) |
| --- | --- | --- |
| A075 | 76 | 26 |
| A077 | 123 | 20 |
| A343 | 135 | 20 |

(schéma: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf, p. 14)

**Le A075 est plus haut que le A076** — 26 contre 20 mm — pour la même largeur de 76 mm ; c'est
le seuil que le DTD associe à un montage sans contre-profilage du montant, avec un bouchon
d'about G067 propre à cette référence. Le A077 et le A343 sont plus larges : ils desservent
vraisemblablement les dormants larges ou les configurations rénovation, sans que le document
attribue chaque seuil à un dormant précis — entrée **VER-39** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

# Cotes des compensateurs

Les deux compensateurs sont réservés aux dormants rénovation 76177 et 76185. Chacun se monte dans
deux orientations.

| Compensateur | Cotes (mm) | Usage |
| --- | --- | --- |
| 6143 | 19 × 29 | profil rénovation |
| 6144 | 12 × 16 | profil rénovation |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

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

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 11, 12, 13, 15, 17, 18 et 19)

Le seuil aluminium est le seul profil complémentaire commun aux cinq dormants.

Les appuis 6137 et 76768 apparaissent malgré tout sur un dormant 76171 à partir de 175 mm
d'isolant, parce que le dormant bas devient alors un 76180 — voir
[Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md).

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 11 à 19, 21 et 22

[2] DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED —
`raw/dta-trocal-76-advanced-6-16-2334-v5.pdf`, p. 8, 9, 14 et 18

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md)
- [Élargisseurs et assemblage PERFORM76](/profiles/perform76-elargisseurs-et-assemblage.md)
- [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
