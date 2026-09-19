---
type: Profilé
title: Élargisseurs et assemblage PERFORM76
description: Les quatre élargisseurs PERFORM76, les profils d'assemblage et les trois poteaux d'angle, réservés aux dormants sans aile 76171 et 76172.
tags: [perform76, elargisseur, poteau-angle, assemblage, renfort, profile]
status: stable
sources:
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
    last_modified: 2026-09-02
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
verified:
  by: process:claude-code
  at: 2026-09-18T22:00:00Z
---

# Réservés aux dormants sans aile

Tous les profils de cette page se montent **uniquement sur les dormants 76171 et 76172**, les deux
dormants neufs sans aile. Aucun ne se monte sur un dormant rénovation ni sur le 76180 [1 p. 19 et
20]. Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

# Cotes des élargisseurs

Les quatre élargisseurs PERFORM76, cotes en mm.

| Élargisseur | Élargissement (mm) | Hauteur (mm) | Renfort acier |
| --- | --- | --- | --- |
| 76700 | 15 | 76 | non |
| 76701 | 30 | 76 | non |
| 76702 | 60 | 76 | **en option**, réf. V314.Z |
| 76703 | 120 | 76 | **en option**, réf. V314.Z × 2 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 19)

Le renfort acier n'existe que sur les **76702 et 76703**, et seulement en option : à partir de
60 mm d'élargissement, la question du renfort se pose au chiffrage. Le 76703 en reçoit **deux**.

# Cotes des profils d'assemblage

| Référence | Type | Cotes (mm) |
| --- | --- | --- |
| 76600 | profil d'assemblage | détail 4,8 / 17 |
| 76821 | adaptateur | 55,5 / 12 / 12 / 13,5, hors tout 75,5 |
| 76822 | clip cornière | 72, décomposé 21 / 7 / 21 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 20)

L'**adaptateur 76821 est requis par les trois poteaux d'angle** : il figure sur chacun des trois
schémas et ne se commande jamais seul dans un angle.

Le clip cornière 76822 s'interpose entre une cornière et le dormant.

# Cotes des poteaux d'angle

Les trois poteaux d'angle PERFORM76, cotes en mm.

| Poteau | Angle | Référence | Largeur (mm) | Hauteur (mm) | Reprise (mm) | Adaptateur requis |
| --- | --- | --- | --- | --- | --- | --- |
| Poteau d'angle à 90° | 90° | 8355 | 104 | 88 | 12 | 76821 |
| Poteau d'angle à 135° | 135° | 8356 | 54 | 88 | 12 | 76821 |
| Poteau d'angle variable | variable | 8340 + 8341 | 84 × 84 | 88 | 12 | 76821 |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 20)

Trois précisions portées sur les schémas :

- le poteau à **135°** porte une cote d'onglet à **45°** et une retombée de **31 mm** ;
- le poteau **variable** est construit sur un rayon de **R42,5** et se compose de **deux
  références**, 8340 et 8341, qui ne se commandent pas séparément ;
- les trois poteaux partagent la même hauteur de 88 mm et la même cote de reprise de 12 mm.

# Compatibilités

| Référence | 76171 | 76172 | 76177 | 76180 | 76185 |
| --- | --- | --- | --- | --- | --- |
| Élargisseurs 76700, 76701 | oui | oui | non | non | non |
| Élargisseurs 76702, 76703 | oui | oui | non | non | non |
| Renfort V314.Z | sur 76702 et 76703 uniquement | idem | non | non | non |
| Profil 76600 | oui | oui | non | non | non |
| Adaptateur 76821 | oui | oui | non | non | non |
| Clip cornière 76822 | oui | oui | non | non | non |
| Poteaux d'angle 8355, 8356, 8340 + 8341 | oui | oui | non | non | non |

(schéma: raw/cahier-technique-perform76-2026-09-02-cc03.pdf, p. 19 et 20)

# Ce que la source ne donne pas

Les références **8340, 8341, 8355 et 8356** sortent de la numérotation 76xxx de la gamme. Elles
appartiennent au catalogue [profine](/fournisseurs/profine.md), comme toutes les références du
cahier technique, sans que leur famille d'origine soit précisée.

# Citations

[1] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, p. 19 et 20

# Voir aussi

- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md)
- [KÖMMERLING](/fournisseurs/kommerling.md)
- [Cahier technique PERFORM76](/sources/cahier-technique-perform76.md)
