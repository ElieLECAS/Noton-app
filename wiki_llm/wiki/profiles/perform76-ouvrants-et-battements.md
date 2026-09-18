---
type: Profilé
title: Ouvrants et battements PERFORM76
description: Les quatre ouvrants PERFORM76 — 76272, 76275, 76279, 76281 — droits ou galbés, et les battements associés.
tags: [perform76, ouvrant, battement, profile, galbe]
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    id: dtd-6-16-2335-v5
    title: DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
stale_after: 2027-09-02
---

# Les quatre ouvrants

La gamme [PERFORM76](/gammes/perform.md) compte quatre ouvrants, tous à **6 chambres**, avec
joint de feuillure sous vitrage et renfort acier galvanisé de **2 mm** (cahier technique, p. 1).

Deux axes les distinguent :

- **droit ou galbé** : l'ouvrant droit fait 76 mm d'épaisseur, le galbé 83 mm (cahier technique,
  p. 1)
- **deux hauteurs de profil** : une paire basse (76281, 76275) et une paire haute (76272, 76279)

Les quatre ouvrants sont proposés sur les cinq dormants, sans restriction (cahier technique,
p. 11, 12, 15, 17 et 18). Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

# Cotes

Cotes des quatre ouvrants PERFORM76, en mm, relevées sur le cahier technique (p. 1, 11, 12, 15,
17 et 18). Les trois cotes de hauteur sont celles portées sur les planches, du dedans vers le
dehors.

| Ouvrant | Profil | Épaisseur (mm) | Hauteur 1 (mm) | Hauteur 2 (mm) | Hauteur 3 (mm) |
| --- | --- | --- | --- | --- | --- |
| 76281 | droit | 76 | 39 | 49 | 70 |
| 76275 | galbé | 83 | 39 | 49 | 70 |
| 76272 | droit | 76 | 79 | 89 | 110 |
| 76279 | galbé | 83 | 79 | 89 | 110 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 11)

Le cahier technique ne dit pas à quel usage correspond chaque paire. La paire haute (76272,
76279) est représentée avec un renfort tubulaire de section carrée et donne des assemblages
nettement plus larges, ce qui suggère un emploi en porte-fenêtre ou en grande dimension — mais le
document ne l'écrit pas. À confirmer auprès du bureau d'études.

L'argument mis en avant par le cahier est inverse : « la finesse des ouvrants apporte un gain de
lumière important » (cahier technique, p. 1).

# Battements

Le cahier technique ne donne pas les battements comme des profils isolés mais comme des
**assemblages**, associés à un ouvrant (cahier technique, p. 11, 12, 15, 17 et 18).

| Assemblage de battement | Ouvrant associé | Largeur hors tout (mm) | Clair intérieur (mm) |
| --- | --- | --- | --- |
| 76274 / 76473 / 1547 / 76281 | 76281 | 112 | 60 |
| 76272 / 76833 / 76272 | 76272 | 226 | 48 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 11)

Les références individuelles **76274**, **76473**, **1547** et **76833** n'apparaissent dans le
cahier technique PERFORM76 que dans ces chaînes, sans cote propre ni description. **Le manuel de
mise en œuvre profine en identifie trois sur quatre** (registre 2.1.2, p. 2, 15, 16 et 34) :

| Référence | Ce que c'est | Cote propre |
| --- | --- | --- |
| 76274 | ouvrant réduit, version gauche | 50 mm |
| 76473 | battement | 46 mm |
| 76833 | battement intérieur | non cotée |
| 1547 | battement intérieur, clippé et collé | non cotée |

Le **76276** est la variante Design du 76274, de même cote propre de 50 mm.

**1547 et 76833 sont identifiés par le [DTD n° DBV-24-6/16-2335_V5](/sources/dtd-6-16-2335.md)**
(§ 2.2.3.2.1), qui range les battements du système en deux familles : extérieurs — 1578, 6130,
6128, 6132, 6162, A176 — et **intérieurs, clippés et collés — 1547, 6129, 6131, 76833, 6133**.
Dans tous les cas, ces battements **reçoivent des embouts collés**. Ce DTD décrit le système 2335
et non le 76 Advanced : le partage de ces deux références entre les deux systèmes est un fait, pas
une équivalence de gamme.

Voir
[Renforts du système 76](/profiles/systeme-76-renforts.md) et
[Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md).

# Les trois battements du système 76

Le manuel profine donne trois battements là où le cahier PERFORM76 n'en nomme qu'un
(registre 2.1.2, p. 2 et 16, version octobre 2021).

| Battement | Cote propre (mm) | Renfort associé | Au cahier PERFORM76 |
| --- | --- | --- | --- |
| 76471 | 62 | V316 | non |
| 76472 | 80 | V317 | non |
| 76473 | 46 | aucun | oui |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.2, p. 2 et 16)

**PROFERM retient le plus étroit des trois**, le 76473 à 46 mm, celui qui porte l'argument du
battement central réduit. Les cotes de débit des trois battements, et celles du dormant qui les
reçoit, sont dans
[Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md).

Les largeurs de battement sont **indépendantes du dormant** : 112 mm avec l'ouvrant 76281 et
226 mm avec l'ouvrant 76272, sur les cinq dormants.

# Le battement central réduit

Le cahier technique annonce un **battement central réduit d'une largeur de 112 mm** pour un gain
de clarté, avec renfort acier dans l'ouvrant de 2 mm d'épaisseur, battement intérieur et poignée
centrée pour une symétrie parfaite (cahier technique, p. 1).

Les 112 mm correspondent donc à l'assemblage de l'ouvrant **76281** — le battement de l'ouvrant
76272, à 226 mm, est le double. C'est la configuration 76281 qui porte l'argument commercial.

# Compatibilités

| Élément | 76281 | 76275 | 76272 | 76279 |
| --- | --- | --- | --- | --- |
| Les cinq dormants | oui | oui | oui | oui |
| Battement 76274 / 76473 / 1547 | oui | - | non | - |
| Battement 76272 / 76833 | non | - | oui | - |
| Meneau ouvrant 76301 (84 mm) | oui | oui | oui | oui |
| Meneau ouvrant 76303 (110 mm) | oui | oui | oui | oui |
| Parcloses d'ouvrant | oui | oui | oui | oui |

Le cahier technique n'associe explicitement de battement qu'aux ouvrants droits 76281 et 76272.
Les ouvrants galbés 76275 et 76279 sont les variantes galbées des mêmes profils, mais leur
battement n'est pas nommé — à confirmer.

Les meneaux 76301 et 76303 portent la mention « montage uniquement compatible avec les
ouvrants » : ils ne se montent jamais sur un dormant. Voir
[Meneaux PERFORM76](/profiles/perform76-meneaux.md).

# Les ouvrants du système que PROFERM ne propose pas

Le système 76 Advanced compte **huit ouvrants**, dont quatre seulement figurent au cahier
technique PERFORM76 (registre 2.1.2, p. 1 et 2, version octobre 2021).

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
basse et les 110 mm de la paire haute — et c'est aussi le seul qui se réalise **sans renfort
acier**, avec le capot aluminium A072 de la variante AluClip Pro. PROFERM ne le propose pas.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 1, 10, 11, 12, 15, 17 et 18

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Parcloses PERFORM76](/profiles/perform76-parcloses.md)
- [Meneaux PERFORM76](/profiles/perform76-meneaux.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
