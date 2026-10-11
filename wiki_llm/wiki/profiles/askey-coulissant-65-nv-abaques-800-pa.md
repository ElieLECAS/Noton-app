---
type: Référence
title: "Abaques de dimensions admissibles 800 Pa du coulissant ASKEY 65 NV (L/150, L/200, L/300)"
description: "Moment d'inertie requis (cm⁴) selon la hauteur (1,00 à 2,50 m) et la largeur (0,50 à 2,00 m) d'un vantail du Coulissant 65 NV ASKEY sous 800 Pa, pour une flèche admissible L/150 (max. 15 mm), L/200 et L/300 ; zone bleue des dimensions maximales du DTA."
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
    pages: 26-28
generated:
  by: process:claude-code
  at: 2026-10-10T19:22:07Z
---

# Exemple d'utilisation des tableaux

Un abaque statique donne, pour un vantail de la gamme Coulissant 65 NV ASKEY ([gamme](/gammes/askey-coulissant-65-nv.md)), le moment d'inertie que doivent atteindre les profilés du vantail pour une pression de vent et une flèche admissible données. Le moment d'inertie (cm⁴) mesure la résistance d'un profilé à la flexion. La planche « Exemple d'utilisation des tableaux » reprend l'abaque 800 Pa, flèche admissible L/150, max. 15 mm (tableau de la section suivante), et l'exemple en trois étapes [1 p. 26] :

1. Choisir la dimension de vantail (p.ex. L = 1,3 m / H = 2,25 m).
2. Déterminer le moment d'inertie requis avec l'abaque (Ix = 37,9 cm⁴).
3. Choisir la combinaison de profilés appropriée (p.ex. W3041250/W3041251).

Sur la planche, la hauteur 2,25 m est cerclée à gauche de la ligne 2,30, la largeur 1,30 m est cerclée sous la colonne 1,30, et la case 31,6 est cerclée à leur croisement ; la combinaison W3041250 et W3041251 est dessinée avec la mention « Ix = 39 cm4 ». Le texte de l'exemple donne 37,9 cm⁴, valeur de la case H = 2,40 m, L = 1,30 m, alors que la case cerclée porte 31,6 cm⁴ (**INC-741**).

![Combinaison W3041250 et W3041251 de l'exemple](/assets/profiles/askey-coulissant-65-nv/abaques/exemple-w3041250-w3041251.png)

La figure montre la combinaison W3041250 / W3041251 imprimée dans l'exemple, avec « W3041250, W3041251, Ix = 39 cm4 ».

(schéma: raw/askey1.pdf, p. 26)

Légende commune aux six abaques : « En bleu les dimensions maximales (VANTAIL H X L) (DTA) ». Sous chaque abaque : « Poids max. vantail 80/160/220 kg * » et « * selon type de roulette ».

# Abaque 800 Pa, flèche admissible L/150, max. 15 mm

Pression de vent 800 Pa ; flèche admissible L/150, avec un maximum de 15 mm. Lecture : l'abaque est une matrice de 16 hauteurs de vantail (lignes, de 2,50 m en haut à 1,00 m en bas, de 0,10 m en 0,10 m) par 16 largeurs de vantail (colonnes, de 0,50 m à 2,00 m, de 0,10 m en 0,10 m) ; chaque case est le moment d'inertie requis, en cm⁴, de la combinaison de profilés du vantail. Pour un vantail de hauteur H et de largeur L, on lit la case à l'intersection de la ligne H et de la colonne L, puis on choisit une combinaison de profilés dont l'inertie Ix est au moins égale (voir [Combinaisons de profilés de vantaux du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-combinaisons-de-profiles-de-vantaux.md)). Les cases sont relevées une à une sur la planche ; aucune valeur n'est interpolée entre deux graduations. La zone bleue couvre les hauteurs de 2,20 m à 1,00 m et les largeurs de 0,50 m à 1,30 m : H maximale 2,20 m (le bord supérieur du bleu est sous la ligne 2,30), L maximale 1,30 m [1 p. 26].

| Hauteur de vantail (m) \ largeur de vantail (m) | 0,50 | 0,60 | 0,70 | 0,80 | 0,90 | 1,00 | 1,10 | 1,20 | 1,30 | 1,40 | 1,50 | 1,60 | 1,70 | 1,80 | 1,90 | 2,00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,50 | 19,1 | 22,7 | 26,3 | 29,7 | 33,1 | 36,3 | 39,4 | 42,3 | 45,1 | 47,7 | 50,1 | 52,3 | 54,3 | 56,0 | 57,6 | 58,9 |
| 2,40 | 16,2 | 19,3 | 22,3 | 25,2 | 28,0 | 30,7 | 33,2 | 35,6 | 37,9 | 40,0 | 42,0 | 43,7 | 45,3 | 46,7 | 47,8 | 48,8 |
| 2,30 | 13,6 | 16,2 | 18,7 | 21,1 | 23,5 | 25,7 | 27,8 | 29,8 | 31,6 | 33,3 | 34,9 | 36,2 | 37,4 | 38,5 | 39,3 | 40,0 |
| 2,20 | 11,6 | 13,8 | 16,0 | 18,0 | 20,0 | 21,8 | 23,6 | 25,2 | 26,7 | 28,1 | 29,3 | 30,4 | 31,3 | 32,1 | 32,7 | 33,1 |
| 2,10 | 10,1 | 12,0 | 13,8 | 15,6 | 17,3 | 18,8 | 20,3 | 21,7 | 22,9 | 24,0 | 25,0 | 25,8 | 26,5 | 27,1 | 27,5 | 27,7 |
| 2,00 | 8,7 | 10,3 | 11,9 | 13,4 | 14,8 | 16,1 | 17,3 | 18,5 | 19,5 | 20,3 | 21,1 | 21,7 | 22,2 | 22,6 | 22,8 | 22,9 |
| 1,90 | 7,4 | 8,8 | 10,1 | 11,4 | 12,6 | 13,7 | 14,7 | 15,6 | 16,4 | 17,0 | 17,6 | 18,0 | 18,4 | 18,6 | 18,6 | 18,6 |
| 1,80 | 6,3 | 7,5 | 8,6 | 9,6 | 10,6 | 11,5 | 12,3 | 13,0 | 13,6 | 14,1 | 14,5 | 14,8 | 14,9 | 15,0 | 15,0 | 15,0 |
| 1,70 | 5,3 | 6,3 | 7,2 | 8,0 | 8,8 | 9,5 | 10,1 | 10,7 | 11,1 | 11,5 | 11,7 | 11,9 | 11,9 | 11,9 | 11,9 | 11,9 |
| 1,60 | 4,4 | 5,2 | 5,9 | 6,6 | 7,2 | 7,8 | 8,2 | 8,6 | 9,0 | 9,2 | 9,3 | 9,4 | 9,4 | 9,4 | 9,4 | 9,4 |
| 1,50 | 3,6 | 4,2 | 4,8 | 5,4 | 5,8 | 6,3 | 6,6 | 6,9 | 7,1 | 7,2 | 7,2 | 7,2 | 7,2 | 7,2 | 7,2 | 7,2 |
| 1,40 | 2,9 | 3,4 | 3,9 | 4,3 | 4,6 | 4,9 | 5,2 | 5,3 | 5,5 | 5,5 | 5,5 | 5,5 | 5,5 | 5,5 | 5,5 | 5,5 |
| 1,30 | 2,3 | 2,7 | 3,0 | 3,4 | 3,6 | 3,8 | 4,0 | 4,1 | 4,1 | 4,1 | 4,1 | 4,1 | 4,1 | 4,1 | 4,1 | 4,1 |
| 1,20 | 1,8 | 2,1 | 2,3 | 2,6 | 2,7 | 2,9 | 2,9 | 3,0 | 3,0 | 3,0 | 3,0 | 3,0 | 3,0 | 3,0 | 3,0 | 3,0 |
| 1,10 | 1,4 | 1,6 | 1,8 | 1,9 | 2,0 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 | 2,1 |
| 1,00 | 1,0 | 1,2 | 1,3 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 | 1,4 |

(schéma: raw/askey1.pdf, p. 26)

# Abaque 800 Pa, flèche admissible L/200

Pression de vent 800 Pa ; flèche admissible L/200. Même lecture. La zone bleue couvre les hauteurs de 2,20 m à 1,00 m et les largeurs de 0,50 m à 1,50 m [1 p. 27].

| Hauteur de vantail (m) \ largeur de vantail (m) | 0,50 | 0,60 | 0,70 | 0,80 | 0,90 | 1,00 | 1,10 | 1,20 | 1,30 | 1,40 | 1,50 | 1,60 | 1,70 | 1,80 | 1,90 | 2,00 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,50 | 22,9 | 27,3 | 31,5 | 35,7 | 39,7 | 43,6 | 47,3 | 50,8 | 54,1 | 57,2 | 60,1 | 62,7 | 65,1 | 67,2 | 69,1 | 70,7 |
| 2,40 | 20,2 | 24,1 | 27,8 | 31,5 | 35,0 | 38,3 | 41,5 | 44,6 | 47,4 | 50,0 | 52,4 | 54,6 | 56,6 | 58,3 | 59,8 | 61,0 |
| 2,30 | 17,8 | 21,1 | 24,4 | 27,6 | 30,6 | 33,5 | 36,3 | 38,9 | 41,3 | 43,5 | 45,5 | 47,3 | 48,8 | 50,2 | 51,3 | 52,2 |
| 2,20 | 15,5 | 18,5 | 21,3 | 24,0 | 26,6 | 29,1 | 31,5 | 33,6 | 35,6 | 37,5 | 39,1 | 40,5 | 41,8 | 42,8 | 43,6 | 44,2 |
| 2,10 | 13,5 | 16,0 | 18,4 | 20,8 | 23,0 | 25,1 | 27,1 | 28,9 | 30,5 | 32,0 | 33,3 | 34,5 | 35,4 | 36,1 | 36,6 | 36,9 |
| 2,00 | 11,6 | 13,8 | 15,9 | 17,8 | 19,7 | 21,5 | 23,1 | 24,6 | 25,9 | 27,1 | 28,1 | 29,0 | 29,6 | 30,1 | 30,4 | 30,5 |
| 1,90 | 9,9 | 11,8 | 13,5 | 15,2 | 16,8 | 18,2 | 19,5 | 20,7 | 21,8 | 22,7 | 23,5 | 24,1 | 24,5 | 24,7 | 24,8 | 24,8 |
| 1,80 | 8,4 | 10,0 | 11,4 | 12,8 | 14,1 | 15,3 | 16,3 | 17,3 | 18,1 | 18,8 | 19,3 | 19,7 | 19,9 | 20,0 | 20,0 | 20,0 |
| 1,70 | 7,1 | 8,3 | 9,6 | 10,7 | 11,7 | 12,7 | 13,5 | 14,2 | 14,8 | 15,3 | 15,6 | 15,8 | 15,9 | 15,9 | 15,9 | 15,9 |
| 1,60 | 5,9 | 6,9 | 7,9 | 8,8 | 9,6 | 10,4 | 11,0 | 11,5 | 11,9 | 12,2 | 12,4 | 12,5 | 12,5 | 12,5 | 12,5 | 12,5 |
| 1,50 | 4,8 | 5,6 | 6,4 | 7,1 | 7,8 | 8,3 | 8,8 | 9,2 | 9,4 | 9,6 | 9,6 | 9,6 | 9,6 | 9,6 | 9,6 | 9,6 |
| 1,40 | 3,9 | 4,5 | 5,2 | 5,7 | 6,2 | 6,6 | 6,9 | 7,1 | 7,3 | 7,3 | 7,3 | 7,3 | 7,3 | 7,3 | 7,3 | 7,3 |
| 1,30 | 3,1 | 3,6 | 4,1 | 4,5 | 4,8 | 5,1 | 5,3 | 5,4 | 5,4 | 5,4 | 5,4 | 5,4 | 5,4 | 5,4 | 5,4 | 5,4 |
| 1,20 | 2,4 | 2,8 | 3,1 | 3,4 | 3,6 | 3,8 | 3,9 | 3,9 | 3,9 | 3,9 | 3,9 | 3,9 | 3,9 | 3,9 | 3,9 | 3,9 |
| 1,10 | 1,8 | 2,1 | 2,3 | 2,5 | 2,7 | 2,8 | 2,8 | 2,8 | 2,8 | 2,8 | 2,8 | 2,8 | 2,8 | 2,8 | 2,8 | 2,8 |
| 1,00 | 1,3 | 1,5 | 1,7 | 1,8 | 1,9 | 1,9 | 1,9 | 1,9 | 1,9 | 1,9 | 1,9 | 1,9 | 1,9 | 1,9 | 1,9 | 1,9 |

(schéma: raw/askey1.pdf, p. 27)

# Abaque 800 Pa, flèche admissible L/300

Pression de vent 800 Pa ; flèche admissible L/300. Même lecture. La zone bleue couvre les hauteurs de 2,20 m à 1,00 m et les largeurs de 0,50 m à 1,50 m [1 p. 28].

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

(schéma: raw/askey1.pdf, p. 28)

# Citations

[1] [Coulissant 65 NV ASKEY - Conception](raw/askey1.pdf), p. 26-28

# Voir aussi

- [Abaques de dimensions admissibles 1 200 Pa du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-abaques-1200-pa.md)
- [Combinaisons de profilés de vantaux du coulissant ASKEY 65 NV](/profiles/askey-coulissant-65-nv-combinaisons-de-profiles-de-vantaux.md)
