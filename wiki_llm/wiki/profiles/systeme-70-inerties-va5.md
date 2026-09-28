---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*A5
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*A5, pression de 2000 Pa, flèche admissible 1/150, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, va5]
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
source_pages:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    pages: 170
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*A5 (2000 Pa, flèche 1/150)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*A5 : pression de 2000 Pa (2,0 kN/m²), flèche admissible 1/150**
de la portée [1 p. 170].

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
<tr><th>100</th><td>0,3</td><td>0,5</td><td>0,6</td><td>0,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,67</td></tr>
<tr><th>110</th><td>0,5</td><td>0,7</td><td>0,8</td><td>0,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,73</td></tr>
<tr><th>120</th><td>0,6</td><td>0,9</td><td>1,1</td><td>1,2</td><td>1,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>130</th><td>0,8</td><td>1,1</td><td>1,4</td><td>1,6</td><td>1,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,87</td></tr>
<tr><th>140</th><td>1,0</td><td>1,4</td><td>1,8</td><td>2,1</td><td>2,2</td><td>2,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,93</td></tr>
<tr><th>150</th><td>1,2</td><td>1,8</td><td>2,2</td><td>2,6</td><td>2,9</td><td>3,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>160</th><td>1,5</td><td>2,2</td><td>2,8</td><td>3,2</td><td>3,6</td><td>3,8</td><td>3,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,07</td></tr>
<tr><th>170</th><td>1,8</td><td>2,6</td><td>3,3</td><td>4,0</td><td>4,4</td><td>4,8</td><td>5,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,13</td></tr>
<tr><th>180</th><td>2,1</td><td>3,1</td><td>4,0</td><td>4,8</td><td>5,4</td><td>5,9</td><td>6,2</td><td>6,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,20</td></tr>
<tr><th>190</th><td>2,5</td><td>3,7</td><td>4,7</td><td>5,7</td><td>6,5</td><td>7,1</td><td>7,5</td><td>7,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,27</td></tr>
<tr><th>200</th><td>2,9</td><td>4,3</td><td>5,6</td><td>6,7</td><td>7,7</td><td>8,5</td><td>9,1</td><td>9,4</td><td>9,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,33</td></tr>
<tr><th>210</th><td>3,4</td><td>5,0</td><td>6,5</td><td>7,8</td><td>9,0</td><td>10,0</td><td>10,8</td><td>11,3</td><td>11,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,40</td></tr>
<tr><th>220</th><td>3,9</td><td>5,8</td><td>7,5</td><td>9,1</td><td>10,5</td><td>11,7</td><td>12,7</td><td>13,4</td><td>13,8</td><td>13,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,47</td></tr>
<tr><th>230</th><td>4,5</td><td>6,6</td><td>8,6</td><td>10,5</td><td>12,1</td><td>13,6</td><td>14,8</td><td>15,7</td><td>16,3</td><td>16,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,53</td></tr>
<tr><th>240</th><td>5,1</td><td>7,5</td><td>9,8</td><td>12,0</td><td>13,9</td><td>15,6</td><td>17,1</td><td>18,2</td><td>19,1</td><td>19,6</td><td>19,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,60</td></tr>
<tr><th>250</th><td>5,8</td><td>8,5</td><td>11,2</td><td>13,6</td><td>15,9</td><td>17,9</td><td>19,6</td><td>21,0</td><td>22,1</td><td>22,8</td><td>23,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,67</td></tr>
<tr><th>260</th><td>6,5</td><td>9,6</td><td>12,6</td><td>15,4</td><td>18,0</td><td>20,3</td><td>22,3</td><td>24,1</td><td>25,4</td><td>26,4</td><td>27,0</td><td>27,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,73</td></tr>
<tr><th>270</th><td>7,3</td><td>10,8</td><td>14,1</td><td>17,3</td><td>20,3</td><td>22,9</td><td>25,3</td><td>27,4</td><td>29,0</td><td>30,3</td><td>31,1</td><td>31,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,80</td></tr>
<tr><th>280</th><td>8,1</td><td>12,0</td><td>15,8</td><td>19,4</td><td>22,7</td><td>25,8</td><td>28,5</td><td>30,9</td><td>32,9</td><td>34,5</td><td>35,7</td><td>36,4</td><td>36,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,87</td></tr>
<tr><th>290</th><td>9,0</td><td>13,4</td><td>17,6</td><td>21,6</td><td>25,4</td><td>28,9</td><td>32,0</td><td>34,8</td><td>37,1</td><td>39,1</td><td>40,5</td><td>41,5</td><td>42,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,93</td></tr>
<tr><th>300</th><td>10,0</td><td>14,8</td><td>19,5</td><td>24,0</td><td>28,2</td><td>32,2</td><td>35,7</td><td>38,9</td><td>41,7</td><td>44,0</td><td>45,8</td><td>47,1</td><td>47,9</td><td>48,2</td><td></td><td></td><td></td><td></td><td></td><td>2,00</td></tr>
<tr><th>310</th><td>11,0</td><td>16,4</td><td>21,6</td><td>26,6</td><td>31,3</td><td>35,7</td><td>39,7</td><td>43,4</td><td>46,6</td><td>49,3</td><td>51,5</td><td>53,2</td><td>54,3</td><td>54,9</td><td></td><td></td><td></td><td></td><td></td><td>2,07</td></tr>
<tr><th>320</th><td>12,1</td><td>18,0</td><td>23,8</td><td>29,3</td><td>34,5</td><td>39,5</td><td>44,0</td><td>48,1</td><td>51,8</td><td>55,0</td><td>57,6</td><td>59,7</td><td>61,2</td><td>62,1</td><td>62,4</td><td></td><td></td><td></td><td></td><td>2,13</td></tr>
<tr><th>330</th><td>13,3</td><td>19,8</td><td>26,1</td><td>32,2</td><td>38,0</td><td>43,5</td><td>48,6</td><td>53,2</td><td>57,4</td><td>61,0</td><td>64,1</td><td>66,7</td><td>68,6</td><td>69,9</td><td>70,5</td><td></td><td></td><td></td><td></td><td>2,20</td></tr>
<tr><th>340</th><td>14,5</td><td>21,7</td><td>28,6</td><td>35,3</td><td>41,7</td><td>47,8</td><td>53,4</td><td>58,6</td><td>63,3</td><td>67,5</td><td>71,1</td><td>74,1</td><td>76,5</td><td>78,2</td><td>79,2</td><td>79,5</td><td></td><td></td><td></td><td>2,27</td></tr>
<tr><th>350</th><td>15,9</td><td>23,6</td><td>31,2</td><td>38,6</td><td>45,6</td><td>52,3</td><td>58,6</td><td>64,4</td><td>69,7</td><td>74,4</td><td>78,5</td><td>82,1</td><td>84,9</td><td>87,1</td><td>88,5</td><td>89,2</td><td></td><td></td><td></td><td>2,33</td></tr>
<tr><th>360</th><td>17,3</td><td>25,7</td><td>34,0</td><td>42,1</td><td>49,8</td><td>57,1</td><td>64,1</td><td>70,5</td><td>76,4</td><td>81,7</td><td>86,5</td><td>90,5</td><td>93,9</td><td>96,5</td><td>98,4</td><td>99,6</td><td>100,0</td><td></td><td></td><td>2,40</td></tr>
<tr><th>370</th><td>18,8</td><td>28,0</td><td>37,0</td><td>45,7</td><td>54,2</td><td>62,2</td><td>69,8</td><td>77,0</td><td>83,5</td><td>89,5</td><td>94,8</td><td>99,5</td><td>103,4</td><td>106,6</td><td>109,0</td><td>110,6</td><td>111,5</td><td></td><td></td><td>2,47</td></tr>
<tr><th>380</th><td>20,3</td><td>30,3</td><td>40,1</td><td>49,6</td><td>58,8</td><td>67,6</td><td>76,0</td><td>83,8</td><td>91,1</td><td>97,7</td><td>103,7</td><td>109,0</td><td>113,5</td><td>117,3</td><td>120,3</td><td>122,4</td><td>123,7</td><td>124,1</td><td></td><td>2,53</td></tr>
<tr><th>390</th><td>22,0</td><td>32,8</td><td>43,4</td><td>53,7</td><td>63,7</td><td>73,3</td><td>82,4</td><td>91,0</td><td>99,0</td><td>106,4</td><td>113,1</td><td>119,1</td><td>124,3</td><td>128,7</td><td>132,2</td><td>134,9</td><td>136,7</td><td>137,6</td><td></td><td>2,60</td></tr>
<tr><th>400</th><td>23,7</td><td>35,4</td><td>46,9</td><td>58,0</td><td>68,9</td><td>79,3</td><td>89,2</td><td>98,6</td><td>107,4</td><td>115,6</td><td>123,0</td><td>129,7</td><td>135,6</td><td>140,7</td><td>144,8</td><td>148,1</td><td>150,5</td><td>151,9</td><td>152,4</td><td>2,67</td></tr>
<tr><th>450</th><td>33,8</td><td>50,5</td><td>66,9</td><td>83,1</td><td>98,8</td><td>114,1</td><td>128,8</td><td>142,9</td><td>156,4</td><td>169,1</td><td>180,9</td><td>191,9</td><td>202,0</td><td>211,1</td><td>219,1</td><td>226,1</td><td>232,0</td><td>236,7</td><td>240,3</td><td>3,00</td></tr>
<tr><th>500</th><td>46,4</td><td>69,4</td><td>92,1</td><td>114,4</td><td>136,3</td><td>157,7</td><td>178,5</td><td>198,6</td><td>217,9</td><td>236,3</td><td>253,9</td><td>270,5</td><td>286,0</td><td>300,4</td><td>313,6</td><td>325,5</td><td>336,2</td><td>345,6</td><td>353,6</td><td>3,33</td></tr>
<tr><th>550</th><td>61,8</td><td>92,4</td><td>122,7</td><td>152,7</td><td>182,2</td><td>211,1</td><td>239,3</td><td>266,7</td><td>293,3</td><td>319,0</td><td>343,6</td><td>367,2</td><td>389,5</td><td>410,6</td><td>430,4</td><td>448,8</td><td>465,7</td><td>481,1</td><td>494,9</td><td>3,67</td></tr>
<tr><th>600</th><td>80,2</td><td>120,1</td><td>159,6</td><td>198,7</td><td>237,2</td><td>275,2</td><td>312,4</td><td>348,7</td><td>384,1</td><td>418,5</td><td>451,8</td><td>483,8</td><td>514,6</td><td>543,9</td><td>571,8</td><td>598,1</td><td>622,8</td><td>645,8</td><td>667,1</td><td>4,00</td></tr>
<tr><th>650</th><td>102,0</td><td>152,7</td><td>203,1</td><td>253,0</td><td>302,3</td><td>351,0</td><td>398,8</td><td>445,8</td><td>491,7</td><td>536,5</td><td>580,0</td><td>622,3</td><td>663,1</td><td>702,4</td><td>740,0</td><td>776,0</td><td>810,1</td><td>842,4</td><td>872,8</td><td>4,33</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 170, registre 2.3.4, p. 24)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 170 du PDF
(page imprimée 24)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
