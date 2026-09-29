---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*B3
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*B3, pression de 1200 Pa, flèche admissible 1/200, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, vb3]
systeme: 70
fournisseur: KÖMMERLING
usage: [atelier, chiffrage]
famille: statique
status: draft
sources:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008
    last_modified: 2008-08-31
source_pages:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    pages: 173
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 178
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*B3 (1200 Pa, flèche 1/200)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*B3 : pression de 1200 Pa (1,2 kN/m²), flèche admissible 1/200**
de la portée [1 p. 173].

Une ligne est une portée L (distance entre appuis du meneau ou de la traverse, en cm), une
colonne une largeur de charge a (en cm), c'est-à-dire la largeur de la surface de vitrage que
l'élément reprend d'un côté ; la dernière colonne est la flèche admissible calculée pour cette
portée, en cm. La largeur de charge se détermine sur les plans de charge et ne dépasse pas la
moitié de la portée : les cases où a dépasse L/2 sont vides sur la table. Pour un élément
chargé des deux côtés, on lit une valeur pour a et une pour b et on les additionne — méthode,
formule et exemple sur [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md).

La table est titrée « Table des moments d'inerties Iz en cm4 » et l'en-tête de page « Tableau des
moments d'inerties Iw » — entrée **INC-141** du registre [Incohérences internes](/anomalies/incoherences-internes.md) ;
la méthode de la section 2.3.4 nomme cette valeur Iw (inertie au vent).

<table>
<thead>
<tr><th rowspan="2">L - portée (cm)</th><th colspan="19">a - largeur de charge (cm)</th><th rowspan="2">Flèche calculée (cm)</th></tr>
<tr><th>20</th><th>30</th><th>40</th><th>50</th><th>60</th><th>70</th><th>80</th><th>90</th><th>100</th><th>110</th><th>120</th><th>130</th><th>140</th><th>150</th><th>160</th><th>170</th><th>180</th><th>190</th><th>200</th></tr>
</thead>
<tbody>
<tr><th>100</th><td>0,3</td><td>0,4</td><td>0,5</td><td>0,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,50</td></tr>
<tr><th>110</th><td>0,4</td><td>0,5</td><td>0,6</td><td>0,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,55</td></tr>
<tr><th>120</th><td>0,5</td><td>0,7</td><td>0,9</td><td>1,0</td><td>1,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,60</td></tr>
<tr><th>130</th><td>0,6</td><td>0,9</td><td>1,1</td><td>1,3</td><td>1,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,65</td></tr>
<tr><th>140</th><td>0,8</td><td>1,1</td><td>1,4</td><td>1,6</td><td>1,8</td><td>1,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,70</td></tr>
<tr><th>150</th><td>1,0</td><td>1,4</td><td>1,8</td><td>2,1</td><td>2,3</td><td>2,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,75</td></tr>
<tr><th>160</th><td>1,2</td><td>1,7</td><td>2,2</td><td>2,6</td><td>2,9</td><td>3,1</td><td>3,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>170</th><td>1,4</td><td>2,1</td><td>2,7</td><td>3,2</td><td>3,6</td><td>3,8</td><td>4,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,85</td></tr>
<tr><th>180</th><td>1,7</td><td>2,5</td><td>3,2</td><td>3,8</td><td>4,3</td><td>4,7</td><td>4,9</td><td>5,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,90</td></tr>
<tr><th>190</th><td>2,0</td><td>2,9</td><td>3,8</td><td>4,6</td><td>5,2</td><td>5,7</td><td>6,0</td><td>6,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,95</td></tr>
<tr><th>200</th><td>2,3</td><td>3,4</td><td>4,5</td><td>5,4</td><td>6,2</td><td>6,8</td><td>7,2</td><td>7,5</td><td>7,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>210</th><td>2,7</td><td>4,0</td><td>5,2</td><td>6,3</td><td>7,2</td><td>8,0</td><td>8,6</td><td>9,0</td><td>9,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,05</td></tr>
<tr><th>220</th><td>3,1</td><td>4,6</td><td>6,0</td><td>7,3</td><td>8,4</td><td>9,4</td><td>10,1</td><td>10,7</td><td>11,0</td><td>11,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,10</td></tr>
<tr><th>230</th><td>3,6</td><td>5,3</td><td>6,9</td><td>8,4</td><td>9,7</td><td>10,9</td><td>11,8</td><td>12,5</td><td>13,0</td><td>13,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,15</td></tr>
<tr><th>240</th><td>4,1</td><td>6,0</td><td>7,9</td><td>9,6</td><td>11,1</td><td>12,5</td><td>13,7</td><td>14,6</td><td>15,3</td><td>15,7</td><td>15,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,20</td></tr>
<tr><th>250</th><td>4,6</td><td>6,8</td><td>8,9</td><td>10,9</td><td>12,7</td><td>14,3</td><td>15,7</td><td>16,8</td><td>17,7</td><td>18,3</td><td>18,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,25</td></tr>
<tr><th>260</th><td>5,2</td><td>7,7</td><td>10,1</td><td>12,3</td><td>14,4</td><td>16,2</td><td>17,9</td><td>19,2</td><td>20,3</td><td>21,1</td><td>21,6</td><td>21,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,30</td></tr>
<tr><th>270</th><td>5,8</td><td>8,6</td><td>11,3</td><td>13,9</td><td>16,2</td><td>18,4</td><td>20,3</td><td>21,9</td><td>23,2</td><td>24,2</td><td>24,9</td><td>25,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,35</td></tr>
<tr><th>280</th><td>6,5</td><td>9,6</td><td>12,6</td><td>15,5</td><td>18,2</td><td>20,6</td><td>22,8</td><td>24,7</td><td>26,3</td><td>27,6</td><td>28,5</td><td>29,1</td><td>29,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,40</td></tr>
<tr><th>290</th><td>7,2</td><td>10,7</td><td>14,1</td><td>17,3</td><td>20,3</td><td>23,1</td><td>25,6</td><td>27,8</td><td>29,7</td><td>31,3</td><td>32,4</td><td>33,2</td><td>33,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,45</td></tr>
<tr><th>300</th><td>8,0</td><td>11,9</td><td>15,6</td><td>19,2</td><td>22,6</td><td>25,7</td><td>28,6</td><td>31,1</td><td>33,4</td><td>35,2</td><td>36,7</td><td>37,7</td><td>38,4</td><td>38,6</td><td></td><td></td><td></td><td></td><td></td><td>1,50</td></tr>
<tr><th>310</th><td>8,8</td><td>13,1</td><td>17,3</td><td>21,3</td><td>25,0</td><td>28,6</td><td>31,8</td><td>34,7</td><td>37,3</td><td>39,4</td><td>41,2</td><td>42,6</td><td>43,5</td><td>43,9</td><td></td><td></td><td></td><td></td><td></td><td>1,55</td></tr>
<tr><th>320</th><td>9,7</td><td>14,4</td><td>19,0</td><td>23,4</td><td>27,6</td><td>31,6</td><td>35,2</td><td>38,5</td><td>41,4</td><td>44,0</td><td>46,1</td><td>47,8</td><td>49,0</td><td>49,7</td><td>49,9</td><td></td><td></td><td></td><td></td><td>1,60</td></tr>
<tr><th>330</th><td>10,6</td><td>15,8</td><td>20,9</td><td>25,8</td><td>30,4</td><td>34,8</td><td>38,9</td><td>42,6</td><td>45,9</td><td>48,8</td><td>51,3</td><td>53,3</td><td>54,9</td><td>55,9</td><td>56,4</td><td></td><td></td><td></td><td></td><td>1,65</td></tr>
<tr><th>340</th><td>11,6</td><td>17,3</td><td>22,9</td><td>28,2</td><td>33,4</td><td>38,2</td><td>42,7</td><td>46,9</td><td>50,7</td><td>54,0</td><td>56,9</td><td>59,3</td><td>61,2</td><td>62,5</td><td>63,4</td><td>63,6</td><td></td><td></td><td></td><td>1,70</td></tr>
<tr><th>350</th><td>12,7</td><td>18,9</td><td>25,0</td><td>30,9</td><td>36,5</td><td>41,8</td><td>46,9</td><td>51,5</td><td>55,7</td><td>59,5</td><td>62,8</td><td>65,6</td><td>67,9</td><td>69,6</td><td>70,8</td><td>71,4</td><td></td><td></td><td></td><td>1,75</td></tr>
<tr><th>360</th><td>13,8</td><td>20,6</td><td>27,2</td><td>33,7</td><td>39,8</td><td>45,7</td><td>51,2</td><td>56,4</td><td>61,1</td><td>65,4</td><td>69,2</td><td>72,4</td><td>75,1</td><td>77,2</td><td>78,8</td><td>79,7</td><td>80,0</td><td></td><td></td><td>1,80</td></tr>
<tr><th>370</th><td>15,0</td><td>22,4</td><td>29,6</td><td>36,6</td><td>43,3</td><td>49,8</td><td>55,9</td><td>61,6</td><td>66,8</td><td>71,6</td><td>75,9</td><td>79,6</td><td>82,7</td><td>85,3</td><td>87,2</td><td>88,5</td><td>89,2</td><td></td><td></td><td>1,85</td></tr>
<tr><th>380</th><td>16,3</td><td>24,3</td><td>32,1</td><td>39,7</td><td>47,1</td><td>54,1</td><td>60,8</td><td>67,0</td><td>72,9</td><td>78,2</td><td>83,0</td><td>87,2</td><td>90,8</td><td>93,8</td><td>96,2</td><td>97,9</td><td>98,9</td><td>99,3</td><td></td><td>1,90</td></tr>
<tr><th>390</th><td>17,6</td><td>26,2</td><td>34,7</td><td>43,0</td><td>51,0</td><td>58,6</td><td>65,9</td><td>72,8</td><td>79,2</td><td>85,1</td><td>90,5</td><td>95,3</td><td>99,4</td><td>102,9</td><td>105,8</td><td>107,9</td><td>109,4</td><td>110,1</td><td></td><td>1,95</td></tr>
<tr><th>400</th><td>19,0</td><td>28,3</td><td>37,5</td><td>46,4</td><td>55,1</td><td>63,4</td><td>71,4</td><td>78,9</td><td>86,0</td><td>92,5</td><td>98,4</td><td>103,8</td><td>108,5</td><td>112,5</td><td>115,9</td><td>118,5</td><td>120,4</td><td>121,5</td><td>121,9</td><td>2,00</td></tr>
<tr><th>450</th><td>27,0</td><td>40,4</td><td>53,6</td><td>66,5</td><td>79,1</td><td>91,3</td><td>103,1</td><td>114,4</td><td>125,1</td><td>135,2</td><td>144,7</td><td>153,5</td><td>161,6</td><td>168,9</td><td>175,3</td><td>180,9</td><td>185,6</td><td>189,4</td><td>192,3</td><td>2,25</td></tr>
<tr><th>500</th><td>37,1</td><td>55,5</td><td>73,6</td><td>91,5</td><td>109,1</td><td>126,2</td><td>142,8</td><td>158,8</td><td>174,3</td><td>189,1</td><td>203,1</td><td>216,4</td><td>228,8</td><td>240,3</td><td>250,9</td><td>260,4</td><td>269,0</td><td>276,5</td><td>282,9</td><td>2,50</td></tr>
<tr><th>550</th><td>49,4</td><td>73,9</td><td>98,2</td><td>122,2</td><td>145,7</td><td>168,8</td><td>191,4</td><td>213,4</td><td>234,7</td><td>255,2</td><td>274,9</td><td>293,7</td><td>311,6</td><td>328,5</td><td>344,3</td><td>359,0</td><td>372,5</td><td>384,9</td><td>395,9</td><td>2,75</td></tr>
<tr><th>600</th><td>64,2</td><td>96,0</td><td>127,7</td><td>158,9</td><td>189,8</td><td>220,1</td><td>249,9</td><td>279,0</td><td>307,3</td><td>334,8</td><td>361,4</td><td>387,1</td><td>411,7</td><td>435,1</td><td>457,4</td><td>478,5</td><td>498,3</td><td>516,7</td><td>533,7</td><td>3,00</td></tr>
<tr><th>650</th><td>81,6</td><td>122,2</td><td>162,5</td><td>202,4</td><td>241,9</td><td>280,8</td><td>319,1</td><td>356,6</td><td>393,3</td><td>429,2</td><td>464,0</td><td>497,8</td><td>530,5</td><td>561,9</td><td>592,0</td><td>620,8</td><td>648,1</td><td>673,9</td><td>698,2</td><td>3,25</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 173, registre 2.3.4, p. 27)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

Le produit pression × dénominateur de flèche est le même que pour [V*A4](/profiles/systeme-70-inerties-va4.md) (1600 Pa, flèche 1/150), [V*C2](/profiles/systeme-70-inerties-vc2.md) (800 Pa, flèche 1/300) : les deux tables portent, cellule par cellule, les mêmes moments d'inertie ; seule la colonne de flèche calculée diffère quand le dénominateur change.

# Même table dans les plans e.VOLUTION de 2008

Le registre 4.2 « Statique » du classeur e.VOLUTION d'août 2008 (système F 91) imprime la même
table, « Table des moments d'inerties Iz en cm4 », classement V\*B3, 1200 Pa, flèche 1/200,
portées 100 à 650 cm, largeurs de charge 20 à 200 cm et flèche calculée : relue case par case à
400 dpi contre la table ci-dessus, elle porte les mêmes valeurs, les mêmes cases vides et les
mêmes flèches calculées. Son en-tête de page écrit « Tableau des moments d'inerties Iz » [2 p. 178].

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 178)

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 173 du PDF
(page imprimée 27)

[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 —
`raw/profine-plans-profiles-e-volution-2008-08.pdf`, registre 4.2 « Statique », PDF p. 178 (page imprimée 41)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
