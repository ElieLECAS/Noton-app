---
type: Profilé
title: Cotes de débit du système 76
description: Les cotes à déduire de la dimension hors tout pour débiter dormants, meneaux, ouvrants, battements et seuils du système 76 Advanced à joint central.
tags: [systeme-76-advanced, cote-de-debit, dormant, ouvrant, meneau, battement, seuil, atelier]
systeme: 76
fournisseur: KÖMMERLING
usage: atelier
status: stable
sources:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
generated:
  by: process:claude-code
  at: 2026-09-18T09:00:00Z
---

# Ce que donnent ces tableaux

Les cotes de débit du **système 76 Advanced à joint central** de
[profine](/fournisseurs/profine.md) sont des **cotes à déduire**, pas des longueurs à couper.
On part de la dimension hors tout de l'élément et on retranche, coupe par coupe, la valeur du
profilé concerné (registre 2.3.1, p. 1, version octobre 2021).

Quatre dimensions se déduisent en cascade, avec les sigles fixés par les
[Directives générales profine](/sources/profine-directives-generales.md) :

| Sigle | Dimension |
| --- | --- |
| DHT | dimension hors tout, c'est-à-dire la dimension extérieure du dormant |
| DEO | dimension extérieure d'ouvrant |
| DFO | dimension de feuillure d'ouvrant |
| — | dimension de vitrage |

**Les valeurs de chaque tableau valent pour une seule coupe.** Un dormant a deux montants : une
largeur hors tout perd la valeur du montant gauche *et* celle du montant droit.

# L'exemple du manuel

Fenêtre à deux vantaux avec meneau, DHT 2 000 × 1 200 mm, dormant 76171, meneau 76372, ouvrant
76271 (registre 2.3.1, p. 1) :

| Étape | Calcul | Résultat (mm) |
| --- | --- | --- |
| Demi-largeur au meneau | X = 2 000 / 2 | 1 000 |
| Dimension extérieure d'ouvrant | DEO = X − (a + b) = 1 000 − (38 + 13) | 949 |
| Dimension du vitrage | 949 − 2 × 60 | 829 |

`a` est la cote du dormant 76171, `b` celle du meneau 76372, `60` celle du vitrage de l'ouvrant
76271 — les trois se lisent dans les tableaux ci-dessous.

# Cotes de débit des dormants

Cotes à déduire de la **dimension hors tout (DHT)**, en mm, pour une seule coupe, relevées sur le
registre 2.3.1 (p. 2, version octobre 2021).

| Dormant | DEO (mm) | DFO (mm) | Vitrage fixe (mm) | Renfort de dormant (mm) | Meneau/traverse (mm) | Renfort de meneau/traverse (mm) | Parclose (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 76171 | 38 | 58 | 59 | 45 | 40 | 71 | 46 |
| 76172 | 56 | 76 | 77 | 61 | 58 | 89 | 64 |
| 76173 | 68 | 88 | 89 | 75 en haut, 45 vers le bas | 70 | 101 | 76 |
| 76177 | 15 | 35 | 36 | 24 | 17 | 50 | 25 |
| 76178 | 15 | 35 | 36 | 24 | 17 | 50 | 25 |
| 76180 | 38 | 58 | 59 | 45 | 40 | 71 | 46 |
| 76185 | 15 | 35 | 36 | 24 | 17 | 50 | 25 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 2)

Le **76173 est le seul dormant dont le renfort ne se débite pas symétriquement** : 75 mm en haut,
45 mm vers le bas. C'est la seule cote du tableau qui dépende du sens de pose.

Les dormants **76177, 76178 et 76185** partagent les mêmes cotes de débit : ce sont les trois
dormants rénovation, qui ne diffèrent que par la largeur de leur aile. Les dormants **76171 et
76180** partagent également les leurs.

Trois de ces sept dormants ne sont pas proposés par PROFERM : **76173** et **76178** n'ont jamais
figuré au cahier technique PERFORM76. Voir
[Dormants PERFORM76](/profiles/perform76-dormants.md).

# Cotes de débit des meneaux de dormant

Cotes à déduire de la **dimension hors tout (DHT)**, en mm, pour une seule coupe, relevées sur le
registre 2.3.1 (p. 3, version octobre 2021). Ces valeurs se prennent **à l'axe du meneau**.

| Meneau | DEO (mm) | DFO (mm) | Vitrage fixe (mm) | Meneau dans meneau (mm) | Renfort de meneau/traverse (mm) | Parclose (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| 76372 | 13 | 33 | 32 | 19 | 46 | 21 |
| 76373 | 26 | 46 | 45 | 19 | 46 | 34 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 3)

Le **débit du renfort de meneau est le même pour les deux meneaux** — 46 mm — alors que tout le
reste change. Voir [Meneaux PERFORM76](/profiles/perform76-meneaux.md).

# Cotes de débit des ouvrants

Cotes à déduire de la **dimension extérieure d'ouvrant (DEO)**, et non de la DHT, en mm, pour une
seule coupe, relevées sur le registre 2.3.1 (p. 4, version octobre 2021).

| Ouvrant | DFO (mm) | Vitrage (mm) | Renfort d'ouvrant (mm) | Parclose (mm) | Meneau (mm) | Renfort de meneau (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| 76271 | 20 | 60 | 55 | 57 | 51 | 82 |
| 76272 | 20 | 92 | 87 | 89 | 83 | 116 |
| 76275 | 20 | 52 | 47 | 49 | 43 | 74 |
| 76279 | 20 | 92 | 87 | 89 | 83 | 116 |
| 76281 | 20 | 52 | 47 | 49 | 43 | 74 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 4)

**La feuillure se déduit de 20 mm sur les cinq ouvrants**, sans exception. Tout le reste suit la
hauteur du profil : la paire basse 76275 / 76281 partage ses cotes, la paire haute 76272 / 76279
aussi, et le 76271 est seul entre les deux.

L'ouvrant **76271 n'est pas au cahier technique PERFORM76**, qui n'en documente que quatre. Voir
[Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md).

# Cotes de débit d'un dormant recevant un battement

Ces cotes ne se déduisent pas de la DHT mais de **X, la distance de l'axe du châssis au bord
extérieur du dormant**. Elles changent avec le battement utilisé : une ligne par couple dormant et
battement (registre 2.3.1, p. 5, 6 et 7, version octobre 2021).

| Dormant | Battement | DEO (mm) | DFO (mm) |
| --- | --- | --- | --- |
| 76171 | 76471 | X − 32 | X − 72 |
| 76171 | 76472 | X − 41 | X − 81 |
| 76171 | 76473 | X − 24 | X − 64 |
| 76172 | 76471 | X − 50 | X − 90 |
| 76172 | 76472 | X − 59 | X − 99 |
| 76172 | 76473 | X − 42 | X − 82 |
| 76173 | 76471 | X − 62 | X − 102 |
| 76173 | 76472 | X − 71 | X − 111 |
| 76173 | 76473 | X − 54 | X − 94 |
| 76177 | 76471 | X − 9 | X − 49 |
| 76177 | 76472 | X − 18 | X − 58 |
| 76177 | 76473 | X − 1 | X − 41 |
| 76178 | 76471 | X − 9 | X − 49 |
| 76178 | 76472 | X − 18 | X − 58 |
| 76178 | 76473 | X − 1 | X − 41 |
| 76180 | 76471 | X − 32 | X − 72 |
| 76180 | 76472 | X − 41 | X − 81 |
| 76180 | 76473 | X − 24 | X − 64 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 5, 6 et 7)

**Le dormant 76185 n'apparaît sur aucune des trois planches de battement**, alors qu'il figure sur
la planche générale des dormants. Ni exclusion ni oubli ne sont énoncés —
entrée **VER-21** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

# Débit du battement lui-même et de son renfort

Formules relevées sur le registre 2.3.1 (p. 5, 6 et 7, version octobre 2021). `DEO` est la
dimension extérieure d'ouvrant obtenue au tableau précédent.

| Battement | Débit du battement | Renfort associé | Débit du renfort |
| --- | --- | --- | --- |
| 76471 | DEO − 2 × 47 mm | V316 | DEO − 2 × 59 mm |
| 76472 | DEO − 2 × 47 mm | V317 | DEO − 2 × 59 mm |
| 76473 avec embouts M462 | DEO − 2 × 47 mm | — | — |
| 76473 sans embout bas M462 | DEO − (47 mm + 33 mm) | — | — |

**Les trois battements se débitent à la même formule, DEO − 2 × 47 mm**, mais seul le 76473
change de formule selon la présence de l'embout bas **M462**. Aucun renfort n'est associé au
76473. Voir [Renforts du système 76](/profiles/systeme-76-renforts.md).

# Cotes de débit des seuils aluminium

Cotes à déduire de la **dimension hors tout (DHT)**, en mm, relevées sur le registre 2.3.1
(p. 8, version octobre 2021).

| Cas | Seuils concernés | DEO (mm) | DFO (mm) | Dormant (mm) |
| --- | --- | --- | --- | --- |
| a | A076, A077-A343 | 10 | 30 | 20 |
| b, usiné | A076, A077-A343 | 10 | 30 | 10,6 |
| c, dormant usiné | A075 | 10 | 30 | 12,1 |

Débit du renfort vertical de dormant, en mm, dans les trois mêmes cas :

| Dormant | Cas a (mm) | Cas b, usiné (mm) | Cas c, dormant usiné (mm) |
| --- | --- | --- | --- |
| 76171 | 100 | 91 | 92 |
| 76172 | 100 | 91 | 92 |
| 76173 | 100 | 91 | 92 |
| 76177 | 86 | 75 | 75 |
| 76178 | 86 | 75 | 75 |
| 76180 | 76 | 74 | 74 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.3.1, p. 8)

**Le libellé des trois colonnes est ambigu sur la planche** : l'en-tête se lit « a | b usiné | c
dormant usiné », les seuils A076 et A077-A343 étant portés sous a et sous b, et l'A075 sous c
seulement. La distinction entre les cas a et b — usinage du seuil ou du dormant — est à vérifier
sur le dessin avant d'engager un débit. Entrée **VER-22** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Le seuil **A076** est celui que PROFERM met en œuvre, avec son rejet d'eau **A062**. Voir
[Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md).

# Citations

[1] Mise en œuvre Système 76 Advanced, profine —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registre 2.3.1, p. 1 à 8

# Voir aussi

- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Meneaux PERFORM76](/profiles/perform76-meneaux.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
