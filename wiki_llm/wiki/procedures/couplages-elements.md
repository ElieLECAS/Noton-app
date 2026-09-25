---
type: Procédure
title: Couplages et contreventements d'éléments, directives profine
description: Les seuils de largeur au-delà desquels un couplage de fenêtres profine exige un jeu de dilatation, et les distances de vissage des couplages et contreventements.
tags: [profine, couplage, contreventement, dilatation, vissage, atelier]
fournisseur: KÖMMERLING
usage: [atelier, pose]
status: stable
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
generated:
  by: process:claude-code
  at: 2026-09-19T16:30:00Z
---

# Ce que fait cette procédure

Elle fixe les règles de conception et de fabrication des couplages entre éléments de fenêtre
(dos de dormant, profilé spécial, contreventement) et des contreventements, registre 1.3.5 des
Directives générales profine. Un couplage assure la stabilité dimensionnelle aux charges de vent
et d'exploitation ; il doit toujours être ancré sur le corps du bâtiment pour transmettre la
charge [1 registre 1.3.5 p. 1 et 3].

# Conditions et interdictions

- **Toujours visser les couplages et contreventements dans l'acier**, sauf exception explicite de
  la directive.
- **Ne jamais frapper au marteau dans la zone d'angle** d'un dormant déjà soudé pour l'assembler
  au couplage — danger de casse d'angle. Aligner et fixer au serre-joint à la place.
- **Toujours prolonger le plan d'étanchéité de la fenêtre sur le pourtour du couplage** : étancher
  entre le couplage et le dos de dormant, étancher les faces frontales des profilés, fermer les
  chambres en acier, aérer les préchambres.
- Les moments d'inertie nécessaires se déterminent selon la **DIN EN 1991-1-4/AN** (charges de
  vent) et, selon le matériau du renfort, la **DIN EN 1993-1-1/AN** ou **DIN EN 1993-1-4/AN**
  (acier), ou la **DIN EN 1999-1-1/AN** (aluminium) — ces normes remplacent depuis fin 2013 la
  DIN 1055 feuille 4.

# Cotes

## Jeu de dilatation

| Seuil | Largeur totale du couplage |
| --- | --- |
| Jeu de dilatation nécessaire, en blanc | à partir de 3,50 m |
| Jeu de dilatation nécessaire, en couleur | à partir de 2,50 m |
| Largeur minimale du joint de compensation | 5 mm |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.5, p. 2)

**Le seuil est 1 m plus bas en couleur qu'en blanc** : la dilatation thermique d'un profilé sombre
est plus forte, à surface égale.

## Vissage des couplages et contreventements

| Paramètre | Valeur |
| --- | --- |
| Première vis, depuis le coin intérieur | 150 mm (250 mm pour un élément en couleur) |
| Deuxième et troisième vis | 150 mm d'entraxe |
| Vis suivantes | 300 mm d'entraxe |
| Diamètre de vis minimal | 5 mm |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.3.5, p. 3)

Une cale de distance est insérée à chaque point de vissage pour répartir la pression. Les
profilés élargisseurs se clippent sur l'arrière du dormant avec une cale, jamais sans, sous peine
de déformation par charge ponctuelle.

# Ce que le document ne dit pas

Le choix entre les trois solutions de profilé — couplage de dos de dormant, couplage avec profilé
spécial, contreventement — n'est pas réduit à une règle chiffrée : il dépend de la position de
montage et du besoin de transmission de charge ou de compensation de dilatation, au cas par cas.
Aucun calcul type n'est donné pour dimensionner un renfort de couplage.

# Citations

[1] Directives générales profine, version janvier 2023 —
`raw/profine-directives-generales-2023-01.pdf`, registre 1.3.5, p. 1 à 4

# Voir aussi

- [profine](/fournisseurs/profine.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [Fabrication des profilés PVC](/procedures/fabrication-profiles-pvc.md)
