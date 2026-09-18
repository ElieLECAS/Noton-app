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
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
stale_after: 2027-09-02
---

# La règle à retenir

**Une même tapée ne donne pas la même épaisseur d'isolant selon le dormant sur lequel elle est
montée.** Sur un dormant 76171, chaque tapée gagne 15 mm d'isolant par rapport au même profil
monté sur un 76177, un 76185 ou un 76180.

Lire l'épaisseur d'isolant dans la colonne du dormant réellement utilisé, jamais dans celle d'à
côté. C'est l'erreur qui coûte une tapée refaite.

**Sur un dormant 76171, les seules épaisseurs d'isolation qui existent sont 80, 95, 115, 135,
155, 175, 195 et 215 mm.** Il n'y a pas de 140 mm sur ce dormant : 140 mm est une valeur du 76180
et du 76177/76185. Une demande à 140 mm sur un 76171 se traite en 135 ou en 155 mm, pas entre les
deux.

Ne pas confondre les familles : la **tapée** est le profil qui donne l'épaisseur d'isolation
(6138 à 6142, 76772, 76769), l'**appui** est une autre pièce (6136, 6137, 76758, 76768), la
**patte de pose** une troisième (NT1939 à NT1953). Une même épaisseur d'isolation se lit sur trois
tableaux différents, un par famille.

# Cotes des tapées par dormant

Épaisseur d'isolation permise par chaque tapée, en mm, relevée sur le cahier technique (p. 14,
16 et 21).

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

La cote propre est la hauteur de la tapée elle-même, constante quel que soit le dormant. Les
cotes de montage, elles, varient : 35/16 sur les dormants rénovation, 35/39 sur le 76180, 35/45
sur le 76171 (cahier technique, p. 14, 16 et 21).

Le cahier technique ne donne **aucune planche de tapées pour le dormant 76172** — compatibilité
à confirmer auprès du bureau d'études.

# La contrainte du 76171 en forte isolation

Sur un dormant 76171, **au-delà de 155 mm d'isolant, le dormant bas doit être remplacé par un
76180 à aile de 20 mm** (cahier technique, p. 21, note en bas de tableau).

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

C'est une contrainte de conception, pas un détail de pose : elle change la nomenclature du
châssis. À vérifier dès le chiffrage d'un projet en isolation renforcée.

# Appuis et pattes de pose sur dormants 76177 et 76185

Correspondance épaisseur d'isolation / référence d'appui, relevée sur le cahier technique (p. 14).

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

Aucune patte de pose n'est indiquée pour ces deux dormants rénovation : la fixation se fait par
les principes de pose rénovation. Voir
[Pose de la PERFORM76](/procedures/pose-perform76.md).

Le tableau d'appuis part de 60 mm alors que la planche de tapées annonce « Iso de 65 » sans
tapée. Le cahier technique ne raccorde pas les deux valeurs — écart mineur, à confirmer.

# Appuis et pattes de pose sur dormant 76180

Correspondance épaisseur d'isolation / patte / appui, relevée sur le cahier technique (p. 16).
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

Correspondance épaisseur d'isolation / patte / appui, relevée sur le cahier technique (p. 21).
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

Les trois dernières lignes imposent le dormant bas 76180 — voir la section précédente. Le premier
cas, à 80 mm, est le seul **sans cale** de tout le tableau.

# Compatibilités des pattes de pose

**Une ligne par couple patte et dormant.** Une même patte dessert les deux dormants neufs à des
épaisseurs d'isolation différentes : lire la ligne dont la colonne « Dormant » correspond au
dormant réellement utilisé.

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

Les sept pattes ne sont documentées que sur les dormants neufs 76171 et 76180. Le cahier ne
donne aucune cote pour ces pattes, seulement leur affectation.

**Aucune épaisseur intermédiaire n'existe.** Les valeurs ci-dessus sont les seules documentées :
140 mm est une épaisseur valable sur le 76180 mais **pas sur le 76171**, où les valeurs
encadrantes sont 135 et 155 mm.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 14, 16 et 21

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md)
- [Pose de la PERFORM76](/procedures/pose-perform76.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
