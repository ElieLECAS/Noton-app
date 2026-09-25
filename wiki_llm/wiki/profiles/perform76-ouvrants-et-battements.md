---
type: Profilé
title: Ouvrants et battements PERFORM76
description: Les quatre ouvrants PERFORM76 — 76272, 76275, 76279, 76281 — droits ou galbés, et les battements associés.
tags: [perform76, ouvrant, battement, profile, galbe]
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
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    id: dtd-6-16-2335-v5
    title: DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION
    last_modified: 2025-04-15
source_pages:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    pages: 4, 14, 15, 18, 20, 21
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
---

# Les quatre ouvrants

L'**ouvrant** est le cadre mobile de la fenêtre, celui qui s'ouvre ; il porte le vitrage et la
quincaillerie, et vient se fermer contre le dormant, le cadre fixe. La gamme
[PERFORM76](/gammes/perform.md) compte quatre ouvrants, tous à **6 chambres**, avec joint de
feuillure sous vitrage et renfort acier galvanisé de **2 mm** [1 p. 4].

Deux axes les distinguent : le profil **droit à 76 mm** ou **galbé à 83 mm** d'épaisseur, et deux
hauteurs de profil, une paire basse et une paire haute.

Les quatre se montent sur les cinq dormants, sans restriction. Voir
[Dormants PERFORM76](/profiles/perform76-dormants.md).

# Cotes

Les quatre ouvrants PERFORM76, cotes en mm. Chaque coupe montre l'ouvrant en tranche, vitrage
en haut : l'épaisseur est la cote horizontale du bas (76 mm pour un ouvrant droit, 83 mm pour un
galbé, dont la face arrondie déborde de 7 mm), les trois hauteurs sont les cotes verticales
portées sur la planche — la première à gauche, les deux autres à droite.

| Ouvrant | Profil | Épaisseur (mm) | Hauteur 1 (mm) | Hauteur 2 (mm) | Hauteur 3 (mm) | Coupe |
| --- | --- | --- | --- | --- | --- | ---: |
| 76281 | droit | 76 | 39 | 49 | 70 | ![Ouvrant 76281](/assets/profiles/perform76/ouvrants/ouvrant-76281.png) |
| 76275 | galbé | 83 | 39 | 49 | 70 | ![Ouvrant 76275](/assets/profiles/perform76/ouvrants/ouvrant-76275.png) |
| 76272 | droit | 76 | 79 | 89 | 110 | ![Ouvrant 76272](/assets/profiles/perform76/ouvrants/ouvrant-76272.png) |
| 76279 | galbé | 83 | 79 | 89 | 110 | ![Ouvrant 76279](/assets/profiles/perform76/ouvrants/ouvrant-76279.png) |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 14, 15, 18, 20 et 21)

La finesse des ouvrants apporte un gain de lumière important [1 p. 4].

# Battements

Sur une fenêtre à deux vantaux, les deux ouvrants se rejoignent au centre sur le **battement** :
un ensemble de profils qui ferme le milieu de la fenêtre. Le cahier technique
donne les battements comme des **assemblages**, associés à un ouvrant, et non comme des profils
isolés. Largeur hors tout et clair intérieur, en mm :

| Assemblage de battement | Ouvrant associé | Largeur hors tout (mm) | Clair intérieur (mm) |
| --- | --- | --- | --- |
| 76274 / 76473 / 1547 / 76281 | 76281 | 112 | 60 |
| 76272 / 76833 / 76272 | 76272 | 226 | 48 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 14)

## Battement 76274 / 76473 / 1547 / 76281, avec l'ouvrant 76281

![Battement 76274 / 76473 / 1547 / 76281](/assets/profiles/perform76/battements/battement-76473.png)

Coupe horizontale du centre de la fenêtre : les deux ouvrants 76281 à gauche et à droite, le
battement au milieu. Au-dessus, la largeur hors tout de **112 mm**, décomposée en 33 / 46 / 33 ;
sous la coupe, 5 / 60 / 5, soit 70.

## Battement 76272 / 76833 / 76272, avec l'ouvrant 76272

![Battement 76272 / 76833 / 76272](/assets/profiles/perform76/battements/battement-76833.png)

Même coupe avec les ouvrants 76272 : **226 mm** hors tout, décomposés en 73 / 80 / 73 ; sous la
coupe, 84 / 48 / 52.

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 14)

Les largeurs de battement sont **indépendantes du dormant** : 112 mm avec l'ouvrant 76281 et
226 mm avec l'ouvrant 76272, sur les cinq dormants.

Le **battement central réduit de 112 mm** est la configuration qui porte l'argument de clarté de
la gamme, avec renfort acier de 2 mm dans l'ouvrant, battement intérieur et poignée centrée
[1 p. 4]. Le battement de l'ouvrant 76272, à 226 mm, en fait le double.

## Ce que sont les références des chaînes d'assemblage

| Référence | Nature | Cote propre (mm) |
| --- | --- | --- |
| 76274 | ouvrant réduit, version gauche | 50 |
| 76276 | ouvrant réduit, variante Design | 50 |
| 76473 | battement | 46 |
| 76833 | battement intérieur, clippé et collé | - |
| 1547 | battement intérieur, clippé et collé | - |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.2, p. 2, 15, 16 et 34)

Les battements **1547 et 76833** appartiennent à la famille des battements intérieurs clippés et
collés, aux côtés des 6129, 6131 et 6133 ; les battements extérieurs du même système sont les
1578, 6130, 6128, 6132, 6162 et A176 [3 § 2.2.3.2.1]. **Tous reçoivent des embouts collés.**

# Les trois battements du système 76

| Battement | Cote propre (mm) | Renfort associé | Au cahier PERFORM76 |
| --- | --- | --- | --- |
| 76471 | 62 | V316 | non |
| 76472 | 80 | V317 | non |
| 76473 | 46 | aucun | oui |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.2, p. 2 et 16)

PROFERM retient le plus étroit des trois, le **76473 à 46 mm**. Leurs cotes de débit sont dans
[Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md).

# Les ouvrants du système que PROFERM ne propose pas

Le système 76 Advanced compte **huit ouvrants**, dont quatre au cahier PERFORM76.

| Ouvrant | Cote propre (mm) | Au cahier PERFORM76 |
| --- | --- | --- |
| 76271 | 78 | non |
| 76272 | 110 | oui |
| 76274 | 50, ouvrant réduit | non, cité en chaîne d'assemblage |
| 76275 | 70 | oui |
| 76276 | 50, ouvrant réduit Design | non |
| 76279 | 110 | oui |
| 76281 | 70 | oui |
| 76283 | 110, ouvrant extérieur | non |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.2, p. 1 et 2)

Le **76271, à 78 mm, est le seul ouvrant intermédiaire du système** entre les 70 mm de la paire
basse et les 110 mm de la paire haute. C'est aussi le seul qui se réalise **sans renfort acier**,
avec le capot aluminium A072 de la variante AluClip Pro.

# Compatibilités

| Élément | 76281 | 76275 | 76272 | 76279 |
| --- | --- | --- | --- | --- |
| Les cinq dormants | oui | oui | oui | oui |
| Battement 76274 / 76473 / 1547 | oui | - | non | - |
| Battement 76272 / 76833 | non | - | oui | - |
| Meneau d'ouvrant 76301, 84 mm | oui | oui | oui | oui |
| Meneau d'ouvrant 76303, 110 mm | oui | oui | oui | oui |
| Parcloses d'ouvrant | oui | oui | oui | oui |

Les meneaux 76301 et 76303 **ne se montent jamais sur un dormant**. Voir
[Meneaux PERFORM76](/profiles/perform76-meneaux.md).

# Ce que la source ne donne pas

- **L'usage de chaque paire d'ouvrants.** La paire haute 76272 et 76279 porte un renfort tubulaire
  de section carrée et donne des assemblages nettement plus larges ; aucune destination n'est
  écrite — entrée **VER-13** du registre
  [Informations à vérifier](/anomalies/informations-a-verifier.md).
- **Le battement des ouvrants galbés** 76275 et 76279 n'est associé à aucune chaîne d'assemblage.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, pages du PDF 4, 14, 15, 18, 20 et 21

[2] Mise en œuvre Système 76 Advanced, profine —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registre 2.1.2

[3] DTD n° DBV-24-6/16-2335_V5 — `raw/dtd-6-16-2335-v5-e-volution.pdf`, § 2.2.3.2.1

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Parcloses PERFORM76](/profiles/perform76-parcloses.md)
- [Meneaux PERFORM76](/profiles/perform76-meneaux.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
