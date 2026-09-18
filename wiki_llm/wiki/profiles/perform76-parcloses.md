---
type: Profilé
title: Parcloses PERFORM76
description: Les 27 parcloses PERFORM76 d'ouvrant et de dormant, classées par épaisseur de vitrage admissible de 16 à 50 mm.
tags: [perform76, parclose, vitrage, profile, feuillure]
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

# Comment choisir une parclose

La parclose se choisit par **l'épaisseur du vitrage à tenir**. Le cahier technique porte cette
épaisseur en bleu sur chaque schéma, la cote propre de la parclose en noir (cahier technique,
p. 5).

Deux familles séparées, non interchangeables : les parcloses d'**ouvrant** et celles de
**dormant**. Toutes les parcloses de la gamme sont arrondies (cahier technique, p. 1).

**Point à retenir avant de chiffrer** : l'ouvrant couvre de 16 à 50 mm de vitrage, le dormant
seulement de 28 à 48 mm. **Un vitrage de moins de 28 mm ne peut pas être tenu en dormant** —
aucune parclose de dormant ne descend sous cette épaisseur.

# Cotes des parcloses d'ouvrant

Les 17 parcloses d'ouvrant, classées par épaisseur de vitrage croissante, en mm, relevées sur le
cahier technique (p. 5).

| Parclose | Épaisseur de vitrage (mm) | Cote parclose (mm) |
| --- | --- | --- |
| 2452 | 16 | 41,5 |
| 2451 | 18 | 39,5 |
| 2453 | 20 | 37,5 |
| 76501 | 24 | 34 |
| 76527 | 26 | 31,5 |
| 76526 | 28 | 29,5 |
| 76516 | 30 | 28 |
| 2454 | 31 | 26,5 |
| 2433 | 33 | 23,5 |
| 76503 | 36 | 22 |
| 76504 | 38 | 20 |
| 76505 | 40 | 18 |
| 76506 | 42 | 16 |
| 76507 | 44 | 14 |
| 76508 | 48 | 12 |
| 76509 | 48 | 10,8 |
| 76515 | 50 | 9,5 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 5)

**Le 48 mm a deux parcloses** : la 76508 (cote 12) et la 76509 (cote 10,8). Le cahier ne dit pas
ce qui départage les deux — à trancher au bureau d'études.

Il n'y a **pas de parclose d'ouvrant pour 22 mm ni pour 34 mm** de vitrage : la série saute de 20
à 24, et de 33 à 36.

# Cotes des parcloses de dormant

Les 10 parcloses de dormant, classées par épaisseur de vitrage croissante, en mm, relevées sur le
cahier technique (p. 5).

| Parclose | Épaisseur de vitrage (mm) | Cote parclose (mm) |
| --- | --- | --- |
| 2634 | 28 | 29,5 |
| 2636 | 30 | 27 |
| 2638 | 31 | 26,5 |
| 2640 | 34 | 23,5 |
| 76573 | 36 | 22 |
| 76575 | 40 | 18 |
| 76576 | 42 | 16 |
| 76577 | 44 | 14 |
| 76578 | 46 | 12 |
| 76579 | 48 | 10,8 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 5)

La série de dormant offre une épaisseur que l'ouvrant n'a pas : **46 mm**, avec la 76578. À
l'inverse elle ne couvre ni 16, 18, 20, 24, 26, 33 ni 50 mm.

# Cotes de la feuillure

Épaisseur de remplissage valable pour une feuillure de 62 mm, avec un joint post-extrudé ou
d'épaisseur équivalente, en mm, relevée sur le cahier technique (p. 5).

| Emplacement | Feuillure (mm) | Épaisseur de remplissage (mm) |
| --- | --- | --- |
| Ouvrant | 62 | 21 |
| Dormant | 62 | 28 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 5)

# Compatibilités

| Parclose | Se monte sur | Ne se monte pas sur |
| --- | --- | --- |
| Série ouvrant (2451 à 76527) | les quatre ouvrants 76272, 76275, 76279, 76281 | les dormants |
| Série dormant (2634 à 76579) | les cinq dormants 76171, 76172, 76177, 76180, 76185 | les ouvrants |

Le cahier technique ne mentionne aucune restriction d'une parclose à un ouvrant ou à un dormant
particulier : à l'intérieur de sa famille, une parclose se monte partout, l'épaisseur de vitrage
étant le seul critère. Voir
[Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md) et
[Dormants PERFORM76](/profiles/perform76-dormants.md).

# Vitrage de série

Le double vitrage standard de la PERFORM76 fait **28 mm** — 6 mm, 18 mm de gaz argon, 4 mm — avec
intercalaire **TGI noir**, pour un **Ug de 1,1 W/m²K** (cahier technique, p. 1). Il correspond
donc à la parclose d'ouvrant **76526** et à la parclose de dormant **2634**.

Les compositions optionnelles du catalogue général se lisent avec ce tableau : un STADIP
44²/16/4 fait 64 mm, au-delà de la plus épaisse parclose de la gamme (50 mm) — entrée **CTR-02**
du registre [Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

**Le DTA du système tranche la question.** Le procédé 76 Advanced de
[profine](/fournisseurs/profine.md) admet un « vitrage isolant double ou triple **jusqu'à 50 mm
d'épaisseur** » (DTA n° 6/16-2334_V5, p. 9). La limite n'est donc pas une lacune de la gamme de
parcloses : **c'est le domaine d'emploi réglementaire du procédé qui s'arrête à 50 mm**.

**Le manuel de fabrication profine, lui, annonce « de 16 à 48 mm » pour le système à joint
central** (registre 2.1.1, p. 1), tout en donnant 36 à 50 mm pour la variante AluClip Zero
(registre 2.6.5). La borne opposable reste celle du DTA, 50 mm — entrée **CTR-17** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

Le tableau de vitrage profine, qui associe chaque épaisseur à sa parclose et à son joint, n'a pas
été transcrit : il est porté en graphique sur les neuf planches du registre 2.3.2. Voir
[Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md).

Un STADIP 44²/16/4 de 64 mm sur une PERFORM76 sortirait de l'Avis Technique — ce qui engage bien
au-delà d'un problème de parclose. Voir [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) et
[Performances des vitrages](/vitrages/performances-vitrages.md).

Le DTA ajoute une seconde limite que le cahier technique ne donne pas : **au-delà de 12 mm
d'épaisseur de verre ou de 60 kg de masse de vantail**, le fabricant doit démontrer par voie
expérimentale la conformité mécanique de la conception selon la norme NF P 20-302.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 1 et 5
[2] [Catalogue menuiseries PROFERM, édition janvier 2026](/sources/catalogue-general-2026.md),
p. 27

# Voir aussi

- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Performances des vitrages](/vitrages/performances-vitrages.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
