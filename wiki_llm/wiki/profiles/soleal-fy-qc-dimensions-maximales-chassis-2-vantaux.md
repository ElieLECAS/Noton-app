---
type: Profilé
title: Dimensions maximales des châssis 2 vantaux Apparent et Minimal du SOLEAL FY 55 et 65
description: "Abaques de limite du châssis 2 vantaux selon l'inertie des montants, ouvrants Apparent et Minimal, SOLEAL FY 55 et FY 65 : hauteur H maximale lue pour chaque largeur W, flèche inférieure à 15 mm, montants T215180, T215186, TFY1206, TFY1207, TFY1338, TFY1232."
tags: [technal, soleal-fy, abaque, inertie, battement, ouvrant-apparent, ouvrant-minimal, 55mm, 65mm, conception]
gamme: LUMINE
systeme: SOLEAL FY
fournisseur: TECHNAL
usage: chiffrage
famille: dimensions-maximales
status: stable
sources:
  - resource: raw/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf
    title: SOLEAL FY 55 & 65, Évolution Quincaillerie Cachée, Conception, réf. 6319.003
    last_modified: 2021-07-29
source_pages:
  - resource: raw/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf
    pages: 22
generated:
  by: process:claude-code
  at: 2026-10-09T19:51:57Z
---

# Limite d'un châssis 2 vantaux par rapport à l'inertie

La LUMINE55 de PROFERM est fabriquée sur le système TECHNAL SOLEAL FY 55 ; le SOLEAL FY 65 est le module de 65 mm du même système (la correspondance avec la LUMINE65 n'est écrite dans aucune source, voir **VER-392**). Pour un châssis à deux vantaux, la hauteur H qu'on peut atteindre dépend de la largeur W et de l'inertie des deux montants qui se rejoignent au centre : plus le châssis est large, plus la hauteur maximale baisse. Chaque abaque (graphique) porte deux courbes, numérotées 1 pour le SOLEAL FY 55 et 2 pour le SOLEAL FY 65, avec la condition « Flèche inférieure à 15 mm » (la flèche est la déformation du montant sous charge) [1 p. 22].

| Version d'ouvrant | Courbe | Module | Montants du châssis 2 vantaux | Condition |
| --- | --- | --- | --- | --- |
| Apparent | 1 | FY55 | T215180 + T215186 | flèche inférieure à 15 mm |
| Apparent | 2 | FY65 | TFY1206 + TFY1207 | flèche inférieure à 15 mm |
| Minimal | 1 | FY55 | TFY1338 + T215186 | flèche inférieure à 15 mm |
| Minimal | 2 | FY65 | TFY1232 + TFY1207 | flèche inférieure à 15 mm |

(schéma: raw/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf, p. 22)

Les moments d'inertie de ces montants sont sur la page [Moments d'inertie Ix et Iy des profilés du SOLEAL FY 55 et 65](/profiles/soleal-fy-qc-inerties-des-profiles.md).

# Châssis 2 vantaux, version Apparent

![Coupe du châssis 2 vantaux version Apparent](/assets/profiles/technal-soleal-fy-qc/dimensions-maximales/coupe-battement-apparent-et-montants.png)

La coupe horizontale montre le croisement des deux vantaux Apparent au centre du châssis, avec la légende des deux courbes. Le graphique « Limite du châssis par rapport à l'inertie » porte W en abscisse (graduations imprimées 0, 0,5, 1, 1,5, 2, 2,5, 2,8 et 3,5 m, avec un trait de grille tous les 0,25 m et un trait renforcé à 2,8 m) et H en ordonnée (graduations imprimées 0,00, 2,30, 2,50, 2,70, 2,90, 3,10 et 3,30 m, traits intermédiaires tous les 0,1 m).

![Limite du châssis 2 vantaux version Apparent par rapport à l'inertie](/assets/profiles/technal-soleal-fy-qc/dimensions-maximales/abaque-chassis-2-vantaux-apparent.png)

Hauteur H maximale lue sur chaque courbe à chaque graduation imprimée de W, en m, précision de lecture ± 0,03 m. « - » signale une largeur où la courbe n'est pas tracée : aucune valeur n'est étendue au-delà du tracé.

| W (m) | H maxi, courbe 1, FY55, T215180 + T215186 (m) | H maxi, courbe 2, FY65, TFY1206 + TFY1207 (m) |
| --- | ---: | ---: |
| 0 | - | - |
| 0,5 | - | - |
| 1 | - | - |
| 1,5 | 2,95 | - |
| 2 | 2,76 | - |
| 2,5 | 2,63 | 2,88 |
| 2,8 | 2,57 | 2,82 |
| 3,5 | - | - |

La courbe 1 est tracée de W = 1,4 m (H = 3,00 m, premier point) à W = 2,8 m (H = 2,57 m, dernier point). La courbe 2 est tracée de W = 2,12 m (H = 3,00 m, premier point) à W = 2,8 m (H = 2,82 m, dernier point) [1 p. 22].

(schéma: raw/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf, p. 22)

# Châssis 2 vantaux, version Minimal

![Coupe du châssis 2 vantaux version Minimal](/assets/profiles/technal-soleal-fy-qc/dimensions-maximales/coupe-battement-minimal-et-montants.png)

La coupe horizontale montre le croisement des deux vantaux Minimal au centre du châssis, avec la légende des deux courbes. Le graphique porte les mêmes graduations de W que le précédent ; les graduations imprimées de H sont 0,00, 2,10, 2,30, 2,50, 2,70, 2,90, 3,10 et 3,30 m.

![Limite du châssis 2 vantaux version Minimal par rapport à l'inertie](/assets/profiles/technal-soleal-fy-qc/dimensions-maximales/abaque-chassis-2-vantaux-minimal.png)

Hauteur H maximale lue sur chaque courbe à chaque graduation imprimée de W, en m, précision de lecture ± 0,03 m.

| W (m) | H maxi, courbe 1, FY55, TFY1338 + T215186 (m) | H maxi, courbe 2, FY65, TFY1232 + TFY1207 (m) |
| --- | ---: | ---: |
| 0 | - | - |
| 0,5 | - | - |
| 1 | 2,85 | - |
| 1,5 | 2,56 | 2,75 |
| 2 | 2,40 | 2,57 |
| 2,5 | 2,31 | 2,46 |
| 2,8 | 2,26 | 2,41 |
| 3,5 | - | - |

La courbe 1 est tracée de W = 0,8 m (H = 3,00 m, premier point) à W = 2,8 m (H = 2,26 m, dernier point). La courbe 2 est tracée de W = 1,05 m (H = 3,00 m, premier point) à W = 2,8 m (H = 2,41 m, dernier point) [1 p. 22].

(schéma: raw/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf, p. 22)

# Ce que la source ne donne pas

Aucune courbe n'est tracée en dessous de H = 2,26 m ni au-delà de W = 2,8 m ; l'abaque ne dit pas si la limite de largeur de 2,8 m est une limite du châssis ou la fin du tracé.

# Citations

[1] [SOLEAL FY 55 & 65, Évolution Quincaillerie Cachée, Conception, réf. 6319.003](raw/SOLEAL-FY-55-65-evolution-QC-catalogue-conception-6319-003-072021-FR (1).pdf), p. 22

# Voir aussi

- [LUMINE](/gammes/lumine.md)
- [Dormants et ouvrants SOLEAL FY](/profiles/soleal-fy-dormants-et-ouvrants.md)
