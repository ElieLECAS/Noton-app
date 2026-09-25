---
type: Profilé
title: Meneaux PERFORM76
description: Les quatre meneaux PERFORM76 — 76372, 76373 de dormant, 76301, 76303 d'ouvrant — et les alignements de traverse de soubassement.
tags: [perform76, meneau, traverse, soubassement, profile]
gamme: PERFORM
systeme: 76
fournisseur: KÖMMERLING
usage: atelier
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
    last_modified: 2026-09-02
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
verified:
  by: process:claude-code
  at: 2026-09-18T22:00:00Z
---

# Les quatre meneaux

La gamme PERFORM76 compte deux meneaux de dormant et deux meneaux d'ouvrant. La distinction est
stricte : **un meneau d'ouvrant ne se monte jamais sur un dormant**, mention portée sur les deux
planches concernées.

Les quatre forment deux couples, appariés par leur clair intérieur :

| Clair intérieur (mm) | Meneau de dormant | Meneau d'ouvrant |
| --- | --- | --- |
| 42 | 76372, 98 mm | 76301, 84 mm |
| 68 | 76373, 124 mm | 76303, 110 mm |

# Cotes

Largeurs et décomposition des quatre meneaux PERFORM76, en mm. La décomposition se lit de gauche
à droite : aile, clair intérieur, aile.

| Meneau | Emplacement | Largeur (mm) | Décomposition (mm) |
| --- | --- | --- | --- |
| 76372 | dormant | 98 | 28 / 42 / 28 |
| 76373 | dormant | 124 | 28 / 68 / 28 |
| 76301 | ouvrant | 84 | 21 / 42 / 21 |
| 76303 | ouvrant | 110 | 21 / 68 / 21 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 10)

Les meneaux de dormant portent des ailes de 28 mm, ceux d'ouvrant des ailes de 21 mm : d'où les
14 mm d'écart de largeur à clair intérieur égal.

Le sommaire des profilés du manuel profine donne 110 mm au 76373, contre 124 mm sur sa propre
planche de détail et au cahier PERFORM76 — entrée **INC-09** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

# Cotes des élargissements du meneau 76372

Largeurs hors tout obtenues en accolant des profils au meneau de dormant de 98 mm, en mm.

| Composition | Largeur hors tout (mm) | Vue (mm) |
| --- | --- | --- |
| 98 | 98 | 42 |
| 34 + 98 | 132 | 49 / 34, soit 83 |
| 74 + 98 | 172 | 89 / 34, soit 123 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 10)

# Cotes des élargissements du meneau 76373

Largeurs hors tout obtenues en accolant des profils au meneau de dormant de 124 mm, en mm.

| Composition | Largeur hors tout (mm) | Vue (mm) |
| --- | --- | --- |
| 124 | 124 | 68 |
| 34 + 124 | 158 | 49 / 60, soit 109 |
| 34 + 124 + 34 | 192 | 49 / 52 / 49, soit 150 |
| 74 + 124 | 198 | 89 / 60, soit 149 |
| 74 + 124 + 74 | 272 | 89 / 52 / 89, soit 230 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 10)

Le 76373 offre cinq compositions contre trois au 76372, dont deux symétriques à trois éléments.
C'est le meneau des ensembles à trois vantaux et plus.

# Compatibilités

| Meneau | Se monte sur | Ne se monte pas sur |
| --- | --- | --- |
| 76372 | les cinq dormants 76171, 76172, 76177, 76180, 76185 | les ouvrants |
| 76373 | les cinq dormants | les ouvrants |
| 76301 | les ouvrants uniquement | **tout dormant** |
| 76303 | les ouvrants uniquement | **tout dormant** |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 10, 11, 12, 15, 17 et 18)

Les combinaisons dormant et meneau donnent les mêmes cotes sur les cinq dormants : le choix du
meneau est indépendant du dormant. Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

Les cotes de débit des deux meneaux de dormant sont dans
[Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md).

# Alignement de la traverse de soubassement

| Principe | Meneaux utilisés | Résultat |
| --- | --- | --- |
| Alignement à l'axe de traverse, entre fixe et ouvrant | 76372 sur le fixe, 76301 sur l'ouvrant | axes de traverses alignés |
| Alignement total avec faux ouvrant | 76301 | vitrages **et** soubassements alignés |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 9)

**L'alignement des vitrages sur le dessus des traverses se demande à la commande** [1 p. 9]. Ce
n'est pas un réglage de chantier.

L'alignement total impose un **faux ouvrant** sur la partie fixe.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 9 à 12, 15, 17 et 18

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
