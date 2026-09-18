---
type: Profilé
title: Appuis et seuils PERFORM76
description: Les sept appuis PERFORM76, le nez d'appui 4319, le seuil alu A076 et les deux compensateurs de rénovation, avec leur affectation par dormant.
tags: [perform76, appui, seuil, nez-d-appui, compensateur, rejet-d-eau]
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
stale_after: 2027-09-02
---

# Deux familles d'appuis, selon le dormant

Les appuis PERFORM76 se répartissent en deux groupes qui ne se mélangent pas (cahier technique,
p. 13, 17, 18 et 19) :

| Groupe | Appuis | Dormants |
| --- | --- | --- |
| Rénovation et neuf avec aile | 6136, 6137, 76768 | 76177, 76180, 76185 |
| Neuf sans aile | 76751, 76752, 76753, 76758 + 76719 | 76171, 76172 |

# Cotes des appuis 6136, 6137 et 76768

Cotes des appuis du premier groupe, en mm, relevées sur le cahier technique (p. 13 et 14). Tous
sont pentés à 3°.

| Appui | Largeur (mm) | Longueur (mm) | Épaisseur (mm) | Retombée (mm) | Nez d'appui 4319 |
| --- | --- | --- | --- | --- | --- |
| 6136 | 67 | 127 | 14 | 3,5 | oui |
| 6137 | 97 | 157 | 14 | 3,5 | oui |
| 76768 | 136 | 196 | 14 | 22 | non |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13)

**La hauteur d'about change selon le dormant** (cahier technique, p. 13) :

| Dormant | Hauteur d'about (mm) |
| --- | --- |
| 76177, aile de 40 mm | 26 |
| 76185, aile de 60 mm | 46 |

Les 20 mm d'écart correspondent exactement à l'écart d'aile entre les deux dormants.

# Cotes des appuis 76751, 76752, 76753 et 76758

Cotes des appuis du second groupe, en mm, relevées sur le cahier technique (p. 19).

| Appui | Hauteur (mm) | Décomposition (mm) | Particularité |
| --- | --- | --- | --- |
| 76751 | 30 | 20 / 56 | livré non monté |
| 76752 | 50 | 20 / 56 | livré non monté |
| 76753 | 35 | 21 / 46, sur 76 | - |
| 76758 + 76719 | - | 80, 15, 5,5, 20 | penté à 3°, ensemble de deux profils |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 19)

**Les appuis 76751 et 76752 sont livrés non montés** (cahier technique, p. 17) : à intégrer au
temps d'atelier. Le cahier écrit cette mention « Appui 76751 et 76152 livrés non montés » ; le
76152 n'existant nulle part ailleurs dans le document, il s'agit vraisemblablement du 76752 —
voir [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md).

L'appui **76758** ne se commande pas seul : il forme un ensemble avec le **76719** (cahier
technique, p. 19 et 21). C'est l'appui utilisé sur le dormant 76171 en pose isolée jusqu'à
155 mm d'isolant — voir
[Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md).

# Cotes du nez d'appui et du seuil

| Référence | Type | Cotes (mm) | Compatibilité |
| --- | --- | --- | --- |
| 4319 | nez d'appui | 18 / 8 | appuis 6136 et 6137 uniquement |
| A076 | seuil aluminium | 76, 10, 20 | les cinq dormants |
| A062 | rejet d'eau | - | s'associe au seuil A076 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 13 et 19)

Le nez d'appui **4319 n'est pas compatible avec le 76768** : le cahier ne le représente que sur
les appuis 6136 et 6137 (cahier technique, p. 13 et 14).

Le seuil **A076 et le rejet d'eau A062 vont toujours ensemble** : le cahier ne les figure jamais
séparément, sur aucune des cinq planches dormant. Le cahier ne donne aucune cote pour le A062.

# Cotes des compensateurs

Les deux compensateurs sont réservés aux profils rénovation, donc aux dormants 76177 et 76185
(cahier technique, p. 13). Chacun se monte dans deux orientations.

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

Relevé sur le cahier technique (p. 11, 12, 13, 15, 17, 18 et 19). Le seuil aluminium est le seul
profil complémentaire commun aux cinq dormants.

Les appuis 6137 et 76768 font exception à la séparation des deux groupes dans un cas précis :
sur un dormant 76171 en forte isolation, à partir de 175 mm, l'appui devient un 6137 ou un
76768 — mais parce que le dormant bas est alors remplacé par un 76180. Voir
[Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md).

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 11 à 19 et 21

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md)
- [Élargisseurs et assemblage PERFORM76](/profiles/perform76-elargisseurs-et-assemblage.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
