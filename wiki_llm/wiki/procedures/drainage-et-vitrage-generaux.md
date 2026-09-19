---
type: Procédure
title: Drainage, décompression et vitrage, directives générales profine
description: Les cotes de drainage, de décompression et de ventilation communes à tous les systèmes profine, et les règles de calage d'un vitrage isolant, avec la conversion poids de vitre / épaisseur.
tags: [profine, drainage, decompression, ventilation, vitrage, calage, atelier]
status: stable
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
generated:
  by: process:claude-code
  at: 2026-09-19T17:00:00Z
---

# Ce que fait cette procédure

Elle donne les cotes communes à tous les systèmes profine (registres 1.3.1 et 1.3.2) pour le
drainage et la décompression d'un cadre, et pour le calage d'un vitrage isolant. Ce sont des
règles génériques : un système ou une gamme précise peut porter ses propres cotes, qui priment
sur celles-ci — voir par exemple
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Drainage et décompression du dormant

| Paramètre | Valeur |
| --- | --- |
| Nombre minimal d'ouvertures de drainage, dormant bas | 2 |
| Position depuis le coin intérieur | 20 à 200 mm |
| Entraxe maximal entre ouvertures | 600 mm |
| Dimension minimale d'une ouverture | trou oblong 5 × 25 mm |
| Décalage feuillure → préchambre → extérieur | environ 50 mm |
| Décompression par découpe du joint de frappe | 100 mm, centrée, une fois par ouvrant |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.1, p. 2)

**Sur une fenêtre à deux vantaux, la découpe de décompression se fait au-dessus du vantail
semi-fixe, jamais au niveau du battement.**

# Décompression de la feuillure de vitrage

| Paramètre | Valeur |
| --- | --- |
| Ouvertures en partie basse transversale | au moins 2, entraxe max. 600 mm |
| Ouvertures en partie haute transversale | 1 par angle |
| Dimension minimale d'une ouverture | trou oblong 5 × 25 mm |
| Décalage feuillure → préchambre → extérieur | environ 50 mm |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.1, p. 3)

# Ventilation des profilés en couleur

**Toute préchambre extérieure fermée sur elle-même, sur un profilé en couleur, laqué ou filmé,
exposée au soleil, doit être ventilée** par un perçage Ø 5 mm minimum, une fois par extrémité
pour une barre simple (meneau, battement, élargisseur), une fois par angle haut pour un cadre
soudé [1 registre 1.3.1 p. 4].

# Vitrage : poids et calage

**1 mm d'épaisseur de vitre pèse 2,5 kg/m²** [1 registre 1.3.2 p. 1]. Les parcloses se posent
toujours côté intérieur.

| Paramètre de calage | Valeur |
| --- | --- |
| Tolérance de débit d'une parclose | + 0,5 mm par mètre |
| Longueur d'une cale de support | 100 mm |
| Largeur d'une cale de support | au moins 2 mm de plus que l'épaisseur du vitrage isolant |
| Distance minimale d'une cale de distance au coin intérieur | 150 mm |
| Cale de distance centrée supplémentaire | si le bord de vitrage dépasse 1 300 mm |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.2, p. 2, 3 et 4)

**Les cales en bois dur sont interdites.** Seul un matériau compatible avec la zone d'assemblage
du double vitrage est admis, et le calage se fait selon le type d'ouverture — la distance de cale
de support sur une partie fixe dépend de la longueur de la cale elle-même ; sur un châssis
oscillo-battant, elle dépend de la ferrure. Une fenêtre à croisillons cale chaque champ selon son
propre type d'ouverture [1 registre 1.3.2 p. 4 et 5].

# Ce que le document ne dit pas

Les cotes de drainage propres à un profilé nommé (par exemple les dormants PERFORM76) priment
sur ces règles génériques et sont documentées séparément — voir
[Pose de la PERFORM76](/procedures/pose-perform76.md). Le registre 1.3.1 ne donne pas de méthode
de calcul de la section de drainage nécessaire selon la surface vitrée ; c'est une cote fixe,
indépendante de la taille de l'élément.

# Citations

[1] Directives générales profine, version janvier 2023 —
`raw/profine-directives-generales-2023-01.pdf`, registres 1.3.1 et 1.3.2

# Voir aussi

- [profine](/fournisseurs/profine.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [Fabrication des profilés PVC](/procedures/fabrication-profiles-pvc.md)
- [Pose de la PERFORM76](/procedures/pose-perform76.md)
