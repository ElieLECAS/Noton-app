---
type: Profilé
title: "Courbes d'utilisation au vent du coulissant LUMEAL GA (EN 12211) : V*A2 et V*A3"
description: "Hauteur H maximale d'un ouvrant ou d'une partie fixe du LUMEAL GA selon sa largeur L (1000 à 4500 mm), pour sept associations de montants centraux, sous 800 Pa (V*A2) et 1200 Pa (V*A3), flèche 1/150, charge 300 kg, demi-périmètre 5 ml."
tags: [technal, lumeal-ga, abaque, vent, nf-en-12211, va2, va3, fleche, coulissant, resistance-au-vent]
gamme: LUMINE
systeme: LUMEAL GA
fournisseur: TECHNAL
usage: chiffrage
famille: courbes-d-utilisation
status: stable
sources:
  - resource: raw/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf
    id: technal-lumeal-ga-conception-5156-007
    title: LUMEAL GA — Catalogue de conception Le Coulissant Minimal (Réf. 5156.007)
    last_modified: 2021-01-21
source_pages:
  - resource: raw/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf
    pages: 16
generated:
  by: process:claude-code
  at: 2026-10-09T09:42:23Z
---

# Conditions d'emploi des courbes

Une courbe d'utilisation donne la hauteur maximale H qu'un ouvrant (le vantail mobile) ou une partie fixe du coulissant LUMEAL GA de [TECHNAL](/fournisseurs/technal.md) (la LUMÉAL55 de PROFERM, voir [Coulissants aluminium](/gammes/coulissants-aluminium.md)) peut atteindre pour une largeur L donnée, sans que la flèche (la déformation sous le vent) dépasse la limite. Les courbes sont établies selon la norme EN 12211 (septembre 2000) : « Courbe tracée pour 2 vantaux et 1 vantail + fixe : L et H hors tout » [1 p. 16].

Chaque abaque porte le titre « Double vitrage pour classification V1. Critère de flèche : maxi : 1/150 de la hauteur de l'ouvrant » sous une pression de vent de 800 Pascals (abaque « V*A2 », bandeau « Ga CLASSE A2, Vent 800 Pa, Flèche 1/150 ») ou de 1200 Pascals (abaque « V*A3 », bandeau « Ga CLASSE A3, Vent 1200 Pa, Flèche 1/150 ») [1 p. 16].

Deux limites s'appliquent aux deux abaques : « Charge maximum par ouvrant limitée à 300kg » et « Le demi périmètre de l'ouvrant ou de la partie fixe est limité à 5ml » (le demi-périmètre est la somme L + H, en mètres linéaires) [1 p. 16].

# Montants de chaque courbe

Sept courbes sont tracées sur chaque abaque. Chacune correspond à une association de montants centraux d'ouvrant, repérée par le numéro du cercle au bout de la courbe. Sur la légende, le montant de gauche est du côté EXT (extérieur) et celui de droite du côté INT (intérieur). Les références des profilés sont dans [Profilés dormants et ouvrants LUMEAL GA](/profiles/lumeal-ga-dormants-et-ouvrants.md) [1 p. 16].

![Légende des sept associations de montants des courbes d'utilisation](/assets/profiles/technal-lumeal-ga/abaques/legende-montants-courbes-1-a-7.png)

| Courbe | Montant côté EXT | Montant côté INT | Profilé central |
| --- | --- | --- | --- |
| 1 | T141037 | T141037 | - |
| 2 | T141037 | T141038 | - |
| 3 | T141037 | T141024 | - |
| 4 | T141038 | T141038 | - |
| 5 | T141038 | T141024 | - |
| 6 | T141024 | 141024 | - |
| 7 | T821003 et T341007 (côté gauche du dessin) | T341008 (côté droit du dessin) | T141033 |

La courbe 7 est repérée par le profilé T141033 avec T821003, T341007 et T341008 ; la note qui suit la légende (« pour la percussion centrale : ») donne la cote L. La référence de la courbe 6 côté INT est imprimée « 141024 », sans le T : **INC-444** (retenu : T141024) [1 p. 16].

(schéma: raw/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 16)

# Cotes

Hauteur H maximale en mm, relevée sur les deux abaques de la p. 16 à chaque graduation imprimée de L (1000, 1500, 2000, 2500, 3000, 3500, 4000 et 4500 mm), à la croisée de la courbe et de la ligne verticale de la graduation. Précision de lecture ±50 mm. Le cadre des abaques s'arrête à H = 3000 mm : un `-` signale une courbe tracée au-dessus de ce cadre à cette graduation (elle y entre à une valeur de L plus grande), ou une courbe qui s'arrête sous son cercle de repère avant la graduation 4500 mm. Rien n'est interpolé entre deux graduations [1 p. 16].

## Abaque V*A2 : vent 800 Pa, flèche 1/150

![Abaque V*A2 : H maximal selon L sous 800 Pa](/assets/profiles/technal-lumeal-ga/abaques/abaque-va2-800-pa.png)

Courbes 1 à 7 sous 800 Pa. La courbe 2 part du coin supérieur gauche du cadre (L = 1000 mm, H = 3000 mm). Les courbes 4, 5 et 6 s'arrêtent sous leur repère avant L = 4500 mm.

<table>
<thead>
<tr><th rowspan="2">L hors tout (mm)</th><th colspan="7">H hors tout maximal (mm), vent 800 Pa, courbe n</th></tr>
<tr><th>1 (T141037 + T141037)</th><th>2 (T141037 + T141038)</th><th>3 (T141037 + T141024)</th><th>4 (T141038 + T141038)</th><th>5 (T141038 + T141024)</th><th>6 (T141024 + 141024)</th><th>7 (percussion centrale)</th></tr>
</thead>
<tbody>
<tr><td>1000</td><td>2450</td><td>3000</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>1500</td><td>2185</td><td>2730</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>2000</td><td>2020</td><td>2550</td><td>2935</td><td>2845</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>2500</td><td>1925</td><td>2440</td><td>2795</td><td>2710</td><td>2985</td><td>-</td><td>-</td></tr>
<tr><td>3000</td><td>1870</td><td>2365</td><td>2705</td><td>2625</td><td>2890</td><td>-</td><td>2985</td></tr>
<tr><td>3500</td><td>1840</td><td>2315</td><td>2640</td><td>2560</td><td>2810</td><td>3000</td><td>2910</td></tr>
<tr><td>4000</td><td>1820</td><td>2285</td><td>2600</td><td>2535</td><td>2765</td><td>2940</td><td>2850</td></tr>
<tr><td>4500</td><td>1800</td><td>2250</td><td>2555</td><td>-</td><td>-</td><td>-</td><td>2820</td></tr>
</tbody>
</table>

(schéma: raw/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 16)

## Abaque V*A3 : vent 1200 Pa, flèche 1/150

![Abaque V*A3 : H maximal selon L sous 1200 Pa](/assets/profiles/technal-lumeal-ga/abaques/abaque-va3-1200-pa.png)

Courbes 1 à 7 sous 1200 Pa. La courbe 7 s'arrête sous son repère vers L = 4100 mm et la courbe 3 vers L = 4300 mm : la valeur à 4500 mm est `-` pour ces deux courbes.

<table>
<thead>
<tr><th rowspan="2">L hors tout (mm)</th><th colspan="7">H hors tout maximal (mm), vent 1200 Pa, courbe n</th></tr>
<tr><th>1 (T141037 + T141037)</th><th>2 (T141037 + T141038)</th><th>3 (T141037 + T141024)</th><th>4 (T141038 + T141038)</th><th>5 (T141038 + T141024)</th><th>6 (T141024 + 141024)</th><th>7 (percussion centrale)</th></tr>
</thead>
<tbody>
<tr><td>1000</td><td>2185</td><td>2735</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>1500</td><td>1920</td><td>2470</td><td>2855</td><td>2755</td><td>-</td><td>-</td><td>-</td></tr>
<tr><td>2000</td><td>1780</td><td>2320</td><td>2680</td><td>2595</td><td>2855</td><td>-</td><td>2965</td></tr>
<tr><td>2500</td><td>1710</td><td>2195</td><td>2545</td><td>2465</td><td>2710</td><td>2910</td><td>2815</td></tr>
<tr><td>3000</td><td>1690</td><td>2120</td><td>2465</td><td>2395</td><td>2635</td><td>2810</td><td>2720</td></tr>
<tr><td>3500</td><td>1675</td><td>2085</td><td>2415</td><td>2350</td><td>2570</td><td>2735</td><td>2660</td></tr>
<tr><td>4000</td><td>1660</td><td>2070</td><td>2380</td><td>2320</td><td>2540</td><td>2700</td><td>2610</td></tr>
<tr><td>4500</td><td>1650</td><td>2050</td><td>-</td><td>2295</td><td>2510</td><td>2670</td><td>-</td></tr>
</tbody>
</table>

(schéma: raw/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 16)

# Cote L de la percussion centrale

La légende se termine par la note « pour la percussion centrale : » suivie de trois schémas en plan des vantaux, chacun avec la cote L. Sur le schéma du haut à gauche (quatre vantaux), L va du bord gauche du deuxième vantail au bord droit du troisième ; sur celui du haut à droite (trois vantaux décalés), L va du bord gauche du premier vantail au bord droit du deuxième ; sur celui du bas (six vantaux), L va du bord gauche du troisième vantail au bord droit du quatrième [1 p. 16].

![Cote L de la percussion centrale sur quatre, trois et six vantaux](/assets/profiles/technal-lumeal-ga/abaques/nota-percussion-centrale-cote-l.png)

(schéma: raw/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf, p. 16)

# Ce que la source ne donne pas

Les abaques ne donnent aucune valeur au-delà de H = 3000 mm ni au-delà de L = 4500 mm, et aucune valeur au-dessous de H = 1000 mm ou de L = 1000 mm. Les courbes 4, 5 et 6 de l'abaque V*A2 et les courbes 3 et 7 de l'abaque V*A3 s'arrêtent sous leur cercle de repère : leur point d'arrêt exact n'est pas lisible.

# Citations

[1] LUMEAL GA — Catalogue de conception Le Coulissant Minimal (Réf. 5156.007) — [raw/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf](raw/LUMEAL-GA-catalogue-conception-5156-007-012021-FR.pdf), p. 16
