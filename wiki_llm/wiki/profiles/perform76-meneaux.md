---
type: Profilé
title: Meneaux PERFORM76
description: Les quatre meneaux PERFORM76 — 76372, 76373 de dormant, 76301, 76303 d'ouvrant — et les alignements de traverse de soubassement.
tags: [perform76, meneau, traverse, soubassement, profile]
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
stale_after: 2027-09-02
---

# Les quatre meneaux

Deux meneaux de dormant et deux meneaux d'ouvrant (cahier technique, p. 10). La distinction est
stricte : **les meneaux d'ouvrant portent la mention « montage uniquement compatible avec les
ouvrants »** et ne se montent jamais sur un dormant.

Les quatre meneaux forment deux couples, appariés par leur clair intérieur :

| Clair intérieur | Meneau de dormant | Meneau d'ouvrant |
| --- | --- | --- |
| 42 mm | 76372 (98 mm) | 76301 (84 mm) |
| 68 mm | 76373 (124 mm) | 76303 (110 mm) |

# Cotes

Largeurs et décomposition des quatre meneaux, en mm, relevées sur le cahier technique (p. 10).
La décomposition se lit de gauche à droite : aile, clair intérieur, aile.

| Meneau | Emplacement | Largeur (mm) | Décomposition (mm) |
| --- | --- | --- | --- |
| 76372 | dormant | 98 | 28 / 42 / 28 |
| 76373 | dormant | 124 | 28 / 68 / 28 |
| 76301 | ouvrant | 84 | 21 / 42 / 21 |
| 76303 | ouvrant | 110 | 21 / 68 / 21 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 10)

Les meneaux de dormant ont des ailes de 28 mm, ceux d'ouvrant des ailes de 21 mm — d'où les
14 mm d'écart de largeur à clair intérieur égal.

**Attention en croisant avec la documentation profine** : le sommaire des profilés du manuel de
mise en œuvre annonce « 76373 Meneau de 110 mm » et « 76303 Meneau de 110 mm » sur la même page,
alors que ses propres planches de détail donnent 124 mm pour le 76373 et 119 mm pour le 76303.
Les 124 mm du 76373 recoupent le cahier technique PERFORM76 — entrée **INC-09** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

Les cotes de débit des deux meneaux de dormant, elles, ne figurent que dans le manuel profine :
voir [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md).

# Cotes des élargissements du meneau 76372

Largeurs hors tout obtenues en accolant des profils au meneau de dormant 98 mm, en mm, relevées
sur le cahier technique (p. 10).

| Composition (mm) | Largeur hors tout (mm) | Vue (mm) |
| --- | --- | --- |
| 98 | 98 | 42 |
| 34 + 98 | 132 | 49 / 34, soit 83 |
| 74 + 98 | 172 | 89 / 34, soit 123 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 10)

# Cotes des élargissements du meneau 76373

Largeurs hors tout obtenues en accolant des profils au meneau de dormant 124 mm, en mm, relevées
sur le cahier technique (p. 10).

| Composition (mm) | Largeur hors tout (mm) | Vue (mm) |
| --- | --- | --- |
| 124 | 124 | 68 |
| 34 + 124 | 158 | 49 / 60, soit 109 |
| 34 + 124 + 34 | 192 | 49 / 52 / 49, soit 150 |
| 74 + 124 | 198 | 89 / 60, soit 149 |
| 74 + 124 + 74 | 272 | 89 / 52 / 89, soit 230 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 10)

Le 76373 offre cinq compositions contre trois pour le 76372, dont deux symétriques à trois
éléments. C'est le meneau à retenir pour un ensemble à trois vantaux ou plus.

# Compatibilités

| Meneau | Se monte sur | Ne se monte pas sur |
| --- | --- | --- |
| 76372 | les cinq dormants 76171, 76172, 76177, 76180, 76185 | les ouvrants |
| 76373 | les cinq dormants | les ouvrants |
| 76301 | les ouvrants uniquement | **tout dormant** |
| 76303 | les ouvrants uniquement | **tout dormant** |

Relevé sur le cahier technique (p. 10, 11, 12, 15, 17 et 18). Les combinaisons dormant + meneau
donnent les mêmes cotes sur les cinq dormants : **le choix du meneau est indépendant du
dormant**. Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

# Alignement de la traverse de soubassement

Deux principes d'alignement sont documentés (cahier technique, p. 9).

| Principe | Meneaux utilisés | Résultat |
| --- | --- | --- |
| Alignement standard à l'axe traverse, entre fixe et ouvrant | 76372 sur le fixe, 76301 sur l'ouvrant | axes de traverses alignés |
| Alignement total avec faux ouvrant | 76301 | vitrages **et** soubassements alignés |

**L'alignement des vitrages par rapport au dessus des traverses est possible à la demande du
client, mais doit être demandé à la commande** (cahier technique, p. 9). Ce n'est pas un réglage
de chantier : passé la commande, c'est perdu.

L'alignement total exige un **faux ouvrant** sur la partie fixe : c'est le prix esthétique à
payer pour que vitrages et soubassements soient sur la même ligne.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 9, 10, 11, 12, 15, 17 et 18

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
