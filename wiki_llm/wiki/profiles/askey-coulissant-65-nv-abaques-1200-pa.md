---
type: Référence
title: "Abaques de dimensions admissibles 1 200 Pa du coulissant ASKEY 65 NV (L/150, L/200, L/300)"
description: "Moment d'inertie requis (cm⁴) selon la hauteur (1,00 à 2,50 m) et la largeur (0,50 à 2,00 m) d'un vantail du Coulissant 65 NV ASKEY sous 1 200 Pa, pour une flèche admissible L/150 (max. 15 mm), L/200 et L/300 ; zone bleue des dimensions maximales du DTA."
tags: [askey, coulissant, abaque, inertie, charge-de-vent, resistance-au-vent, fleche, dta, conception]
gamme: Coulissant 65 NV
fournisseur: ASKEY
usage: chiffrage
famille: abaques
status: draft
sources:
  - resource: raw/askey1.pdf
    title: Coulissant 65 NV ASKEY - Conception
    last_modified: 2022-12-13
source_pages:
  - resource: raw/askey1.pdf
    pages: 29-31
generated:
  by: process:claude-code
  at: 2026-10-10T19:22:07Z
---

# Lecture des abaques 1 200 Pa

Les trois abaques ci-dessous donnent, pour un vantail du Coulissant 65 NV ASKEY ([gamme](/gammes/askey-coulissant-65-nv.md)), le moment d'inertie requis sous une pression de vent de 1 200 Pa. L'exemple d'utilisation, fait sur l'abaque 800 Pa, est dans [Abaques de dimensions admissibles 800 Pa](/profiles/askey-coulissant-65-nv-abaques-800-pa.md). Légende commune : « En bleu les dimensions maximales (VANTAIL H X L) (DTA) ». Sous chaque abaque : « Poids max. vantail 80/160/220 kg * » et « * selon type de roulette ». La zone bleue des trois abaques couvre les hauteurs de 2,20 m à 1,00 m et les largeurs de 0,50 m à 1,50 m.

# Abaque 1 200 Pa, flèche admissible L/150, max. 15 mm

Pression de vent 1 200 Pa ; flèche admissible L/150, avec un maximum de 15 mm [1 p. 29]. Lecture : l'abaque est une matrice de 16 hauteurs de vantail (lignes, de 2,50 m en haut à 1,00 m en bas, de 0,10 m en 0,10 m) par 16 largeurs de vantail (colonnes, de 0,50 m à 2,00 m, de 0,10 m en 0,10 m) ; chaque case est le moment d'inertie requis, en cm⁴, de la combinaison de profilés du vantail. Pour un vantail de hauteur H et de largeur L, on lit la case à l'intersection de la ligne H et de la colonne L, puis on choisit une combinaison de profilés dont l'inertie Ix est au moins égale (voir [Combinaisons de profilés de vantaux du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-combinaisons-de-profiles-de-vantaux.md)). Les cases sont relevées une à une sur la planche ; aucune valeur n'est interpolée entre deux graduations.

| Hauteur de vantail (m) \ largeur de vantail (m) | 0,50 | 0,60 | 0,70 | 0,80 | 0,90 | 1,00 | 1,10 | 1,20 | 1,30 | 1,40 | 1,50 | 1,60 | 1,70 | 1,80 | 1,90 | 2,00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,50 | 28,6 | 34,1 | 39,4 | 44,6 | 49,6 | 54,5 | 59,1 | 63,5 | 67,6 | 71,5 | 75,1 | 78,4 | 81,4 | 84,1 | 86,4 | 88,4 |
| 2,40 | 24,3 | 28,9 | 33,4 | 37,8 | 42,0 | 46,0 | 49,8 | 53,5 | 56,9 | 60,0 | 62,9 | 65,6 | 67,9 | 70,0 | 71,8 | 73,2 |
| 2,30 | 20,4 | 24,3 | 28,1 | 31,7 | 35,2 | 38,6 | 41,7 | 44,7 | 47,4 | 50,0 | 52,3 | 54,4 | 56,2 | 57,7 | 59,0 | 60,0 |
| 2,20 | 17,5 | 20,8 | 24,0 | 27,0 | 30,0 | 32,8 | 35,4 | 37,8 | 40,1 | 42,2 | 44,0 | 45,6 | 47,0 | 48,1 | 49,0 | 49,7 |
| 2,10 | 15,2 | 18,0 | 20,8 | 23,4 | 25,9 | 28,3 | 30,5 | 32,5 | 34,4 | 36,0 | 37,5 | 38,8 | 39,8 | 40,6 | 41,2 | 41,6 |
| 2,00 | 13,1 | 15,5 | 17,8 | 20,1 | 22,2 | 24,2 | 26,0 | 27,7 | 29,2 | 30,5 | 31,6 | 32,6 | 33,3 | 33,9 | 34,2 | 34,3 |
| 1,90 | 11,2 | 13,2 | 15,2 | 17,1 | 18,9 | 20,5 | 22,0 | 23,3 | 24,5 | 25,5 | 26,4 | 27,1 | 27,5 | 27,8 | 27,9 | 27,9 |
| 1,80 | 9,5 | 11,2 | 12,9 | 14,4 | 15,9 | 17,2 | 18,4 | 19,5 | 20,4 | 21,1 | 21,7 | 22,1 | 22,4 | 22,5 | 22,5 | 22,5 |
| 1,70 | 7,9 | 9,4 | 10,7 | 12,0 | 13,2 | 14,3 | 15,2 | 16,0 | 16,7 | 17,2 | 17,6 | 17,8 | 17,9 | 17,9 | 17,9 | 17,9 |
| 1,60 | 6,6 | 7,8 | 8,9 | 9,9 | 10,8 | 11,7 | 12,4 | 13,0 | 13,4 | 13,8 | 14,0 | 14,0 | 14,0 | 14,0 | 14,0 | 14,0 |
| 1,50 | 5,4 | 6,4 | 7,2 | 8,0 | 8,8 | 9,4 | 9,9 | 10,3 | 10,6 | 10,8 | 10,8 | 10,8 | 10,8 | 10,8 | 10,8 | 10,8 |
| 1,40 | 4,4 | 5,1 | 5,8 | 6,4 | 7,0 | 7,4 | 7,8 | 8,0 | 8,2 | 8,2 | 8,2 | 8,2 | 8,2 | 8,2 | 8,2 | 8,2 |
| 1,30 | 3,5 | 4,0 | 4,6 | 5,0 | 5,4 | 5,7 | 5,9 | 6,1 | 6,1 | 6,1 | 6,1 | 6,1 | 6,1 | 6,1 | 6,1 | 6,1 |
| 1,20 | 2,7 | 3,1 | 3,5 | 3,8 | 4,1 | 4,3 | 4,4 | 4,4 | 4,4 | 4,4 | 4,4 | 4,4 | 4,4 | 4,4 | 4,4 | 4,4 |
| 1,10 | 2,0 | 2,4 | 2,6 | 2,9 | 3,0 | 3,1 | 3,1 | 3,1 | 3,1 | 3,1 | 3,1 | 3,1 | 3,1 | 3,1 | 3,1 | 3,1 |
| 1,00 | 1,5 | 1,7 | 1,9 | 2,0 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 |

(schéma: raw/askey1.pdf, p. 29)

# Abaque 1 200 Pa, flèche admissible L/200

Pression de vent 1 200 Pa ; flèche admissible L/200. Même lecture [1 p. 30]. Les 256 cases de cet abaque sont identiques une à une à celles de l'abaque 800 Pa, flèche admissible L/300 (**VER-728**).

| Hauteur de vantail (m) \ largeur de vantail (m) | 0,50 | 0,60 | 0,70 | 0,80 | 0,90 | 1,00 | 1,10 | 1,20 | 1,30 | 1,40 | 1,50 | 1,60 | 1,70 | 1,80 | 1,90 | 2,00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,50 | 34,3 | 40,9 | 47,3 | 53,5 | 59,6 | 65,4 | 70,9 | 76,2 | 81,1 | 85,8 | 90,1 | 94,1 | 97,7 | 100,9 | 103,7 | 106,1 |
| 2,40 | 30,3 | 36,1 | 41,7 | 47,2 | 52,5 | 57,5 | 62,3 | 66,8 | 71,1 | 75,0 | 78,7 | 82,0 | 84,9 | 87,5 | 89,7 | 91,5 |
| 2,30 | 26,6 | 31,7 | 36,6 | 41,4 | 45,9 | 50,3 | 54,4 | 58,3 | 61,9 | 65,2 | 68,2 | 70,9 | 73,3 | 75,3 | 77,0 | 78,3 |
| 2,20 | 23,3 | 27,7 | 31,9 | 36,0 | 40,0 | 43,7 | 47,2 | 50,5 | 53,5 | 56,2 | 58,7 | 60,8 | 62,7 | 64,2 | 65,4 | 66,2 |
| 2,10 | 20,2 | 24,0 | 27,7 | 31,2 | 34,5 | 37,7 | 40,6 | 43,3 | 45,8 | 48,0 | 50,0 | 51,7 | 53,1 | 54,2 | 54,9 | 55,4 |
| 2,00 | 17,4 | 20,7 | 23,8 | 26,8 | 29,6 | 32,2 | 34,7 | 36,9 | 38,9 | 40,7 | 42,2 | 43,5 | 44,4 | 45,1 | 45,6 | 45,7 |
| 1,90 | 14,9 | 17,6 | 20,3 | 22,8 | 25,1 | 27,3 | 29,3 | 31,1 | 32,7 | 34,1 | 35,2 | 36,1 | 36,7 | 37,1 | 37,2 | 37,2 |
| 1,80 | 12,6 | 14,9 | 17,1 | 19,2 | 21,1 | 22,9 | 24,5 | 25,9 | 27,2 | 28,2 | 29,0 | 29,5 | 29,9 | 30,0 | 30,0 | 30,0 |
| 1,70 | 10,6 | 12,5 | 14,3 | 16,0 | 17,6 | 19,0 | 20,3 | 21,3 | 22,2 | 22,9 | 23,5 | 23,8 | 23,9 | 23,9 | 23,9 | 23,9 |
| 1,60 | 8,8 | 10,4 | 11,8 | 13,2 | 14,4 | 15,5 | 16,5 | 17,3 | 17,9 | 18,4 | 18,6 | 18,7 | 18,7 | 18,7 | 18,7 | 18,7 |
| 1,50 | 7,2 | 8,5 | 9,6 | 10,7 | 11,7 | 12,5 | 13,2 | 13,7 | 14,1 | 14,4 | 14,5 | 14,5 | 14,5 | 14,5 | 14,5 | 14,5 |
| 1,40 | 5,8 | 6,8 | 7,7 | 8,6 | 9,3 | 9,9 | 10,4 | 10,7 | 10,9 | 11,0 | 11,0 | 11,0 | 11,0 | 11,0 | 11,0 | 11,0 |
| 1,30 | 4,6 | 5,4 | 6,1 | 6,7 | 7,2 | 7,6 | 7,9 | 8,1 | 8,2 | 8,2 | 8,2 | 8,2 | 8,2 | 8,2 | 8,2 | 8,2 |
| 1,20 | 3,6 | 4,2 | 4,7 | 5,1 | 5,5 | 5,7 | 5,9 | 5,9 | 5,9 | 5,9 | 5,9 | 5,9 | 5,9 | 5,9 | 5,9 | 5,9 |
| 1,10 | 2,7 | 3,2 | 3,5 | 3,8 | 4,0 | 4,1 | 4,2 | 4,2 | 4,2 | 4,2 | 4,2 | 4,2 | 4,2 | 4,2 | 4,2 | 4,2 |
| 1,00 | 2,0 | 2,3 | 2,5 | 2,7 | 2,8 | 2,9 | 2,9 | 2,9 | 2,9 | 2,9 | 2,9 | 2,9 | 2,9 | 2,9 | 2,9 | 2,9 |

(schéma: raw/askey1.pdf, p. 30)

# Abaque 1 200 Pa, flèche admissible L/300

Pression de vent 1 200 Pa ; flèche admissible L/300. Même lecture [1 p. 31].

| Hauteur de vantail (m) \ largeur de vantail (m) | 0,50 | 0,60 | 0,70 | 0,80 | 0,90 | 1,00 | 1,10 | 1,20 | 1,30 | 1,40 | 1,50 | 1,60 | 1,70 | 1,80 | 1,90 | 2,00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,50 | 51,5 | 61,3 | 71,0 | 80,3 | 89,4 | 98,0 | 106,4 | 114,3 | 121,7 | 128,7 | 135,2 | 141,1 | 146,5 | 151,3 | 155,5 | 159,1 |
| 2,40 | 45,5 | 54,2 | 62,6 | 70,8 | 78,7 | 86,3 | 93,5 | 100,3 | 106,6 | 112,6 | 118,0 | 123,0 | 127,4 | 131,2 | 134,6 | 137,3 |
| 2,30 | 40,0 | 47,6 | 54,9 | 62,1 | 68,9 | 75,4 | 81,6 | 87,4 | 92,8 | 97,8 | 102,3 | 106,3 | 109,9 | 112,9 | 115,4 | 117,4 |
| 2,20 | 34,9 | 41,5 | 47,9 | 54,1 | 59,9 | 65,5 | 70,8 | 75,7 | 80,2 | 84,3 | 88,0 | 91,2 | 94,0 | 96,3 | 98,1 | 99,4 |
| 2,10 | 30,3 | 36,0 | 41,5 | 46,8 | 51,8 | 56,5 | 60,9 | 65,0 | 68,7 | 72,1 | 75,0 | 77,5 | 79,6 | 81,2 | 82,4 | 83,1 |
| 2,00 | 26,1 | 31,0 | 35,7 | 40,2 | 44,4 | 48,3 | 52,0 | 55,4 | 58,4 | 61,0 | 63,3 | 65,2 | 66,7 | 67,7 | 68,4 | 68,6 |
| 1,90 | 22,3 | 26,5 | 30,4 | 34,2 | 37,7 | 41,0 | 44,0 | 46,7 | 49,1 | 51,1 | 52,8 | 54,1 | 55,1 | 55,7 | 55,9 | 55,9 |
| 1,80 | 18,9 | 22,4 | 25,7 | 28,8 | 31,7 | 34,4 | 36,8 | 38,9 | 40,7 | 42,2 | 43,4 | 44,3 | 44,8 | 45,0 | 45,0 | 45,0 |
| 1,70 | 15,9 | 18,8 | 21,5 | 24,0 | 26,4 | 28,5 | 30,4 | 32,0 | 33,4 | 34,4 | 35,2 | 35,6 | 35,8 | 35,8 | 35,8 | 35,8 |
| 1,60 | 13,2 | 15,5 | 17,8 | 19,8 | 21,7 | 23,3 | 24,7 | 25,9 | 26,9 | 27,5 | 27,9 | 28,1 | 28,1 | 28,1 | 28,1 | 28,1 |
| 1,50 | 10,8 | 12,7 | 14,5 | 16,1 | 17,5 | 18,8 | 19,8 | 20,6 | 21,2 | 21,6 | 21,7 | 21,7 | 21,7 | 21,7 | 21,7 | 21,7 |
| 1,40 | 8,7 | 10,2 | 11,6 | 12,8 | 13,9 | 14,8 | 15,5 | 16,0 | 16,4 | 16,5 | 16,5 | 16,5 | 16,5 | 16,5 | 16,5 | 16,5 |
| 1,30 | 6,9 | 8,1 | 9,1 | 10,1 | 10,8 | 11,4 | 11,9 | 12,2 | 12,2 | 12,2 | 12,2 | 12,2 | 12,2 | 12,2 | 12,2 | 12,2 |
| 1,20 | 5,4 | 6,3 | 7,0 | 7,7 | 8,2 | 8,6 | 8,8 | 8,9 | 8,9 | 8,9 | 8,9 | 8,9 | 8,9 | 8,9 | 8,9 | 8,9 |
| 1,10 | 4,1 | 4,7 | 5,3 | 5,7 | 6,0 | 6,2 | 6,3 | 6,3 | 6,3 | 6,3 | 6,3 | 6,3 | 6,3 | 6,3 | 6,3 | 6,3 |
| 1,00 | 3,0 | 3,5 | 3,8 | 4,1 | 4,2 | 4,3 | 4,3 | 4,3 | 4,3 | 4,3 | 4,3 | 4,3 | 4,3 | 4,3 | 4,3 | 4,3 |

(schéma: raw/askey1.pdf, p. 31)

# Citations

[1] [Coulissant 65 NV ASKEY - Conception](raw/askey1.pdf), p. 29-31

# Voir aussi

- [Abaques de dimensions admissibles 800 Pa du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-abaques-800-pa.md)
- [Combinaisons de profilés de vantaux du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-combinaisons-de-profiles-de-vantaux.md)
