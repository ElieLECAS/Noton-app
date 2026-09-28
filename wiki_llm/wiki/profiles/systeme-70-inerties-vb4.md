---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*B4
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*B4, pression de 1600 Pa, flèche admissible 1/200, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, vb4]
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
    pages: 174
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*B4 (1600 Pa, flèche 1/200)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*B4 : pression de 1600 Pa (1,6 kN/m²), flèche admissible 1/200**
de la portée [1 p. 174].

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
<tr><th>100</th><td>0,4</td><td>0,5</td><td>0,6</td><td>0,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,50</td></tr>
<tr><th>110</th><td>0,5</td><td>0,7</td><td>0,8</td><td>0,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,55</td></tr>
<tr><th>120</th><td>0,7</td><td>0,9</td><td>1,1</td><td>1,3</td><td>1,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,60</td></tr>
<tr><th>130</th><td>0,8</td><td>1,2</td><td>1,5</td><td>1,7</td><td>1,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,65</td></tr>
<tr><th>140</th><td>1,1</td><td>1,5</td><td>1,9</td><td>2,2</td><td>2,4</td><td>2,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,70</td></tr>
<tr><th>150</th><td>1,3</td><td>1,9</td><td>2,4</td><td>2,8</td><td>3,1</td><td>3,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,75</td></tr>
<tr><th>160</th><td>1,6</td><td>2,3</td><td>2,9</td><td>3,5</td><td>3,8</td><td>4,1</td><td>4,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>170</th><td>1,9</td><td>2,8</td><td>3,6</td><td>4,2</td><td>4,7</td><td>5,1</td><td>5,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,85</td></tr>
<tr><th>180</th><td>2,3</td><td>3,3</td><td>4,3</td><td>5,1</td><td>5,8</td><td>6,3</td><td>6,6</td><td>6,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,90</td></tr>
<tr><th>190</th><td>2,7</td><td>3,9</td><td>5,1</td><td>6,1</td><td>6,9</td><td>7,6</td><td>8,0</td><td>8,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,95</td></tr>
<tr><th>200</th><td>3,1</td><td>4,6</td><td>5,9</td><td>7,2</td><td>8,2</td><td>9,0</td><td>9,7</td><td>10,0</td><td>10,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>210</th><td>3,6</td><td>5,3</td><td>6,9</td><td>8,4</td><td>9,6</td><td>10,7</td><td>11,5</td><td>12,0</td><td>12,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,05</td></tr>
<tr><th>220</th><td>4,2</td><td>6,2</td><td>8,0</td><td>9,7</td><td>11,2</td><td>12,5</td><td>13,5</td><td>14,3</td><td>14,7</td><td>14,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,10</td></tr>
<tr><th>230</th><td>4,8</td><td>7,0</td><td>9,2</td><td>11,2</td><td>13,0</td><td>14,5</td><td>15,8</td><td>16,7</td><td>17,4</td><td>17,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,15</td></tr>
<tr><th>240</th><td>5,4</td><td>8,0</td><td>10,5</td><td>12,8</td><td>14,9</td><td>16,7</td><td>18,2</td><td>19,4</td><td>20,3</td><td>20,9</td><td>21,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,20</td></tr>
<tr><th>250</th><td>6,1</td><td>9,1</td><td>11,9</td><td>14,5</td><td>16,9</td><td>19,1</td><td>20,9</td><td>22,4</td><td>23,6</td><td>24,4</td><td>24,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,25</td></tr>
<tr><th>260</th><td>6,9</td><td>10,2</td><td>13,4</td><td>16,4</td><td>19,2</td><td>21,7</td><td>23,8</td><td>25,7</td><td>27,1</td><td>28,2</td><td>28,8</td><td>29,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,30</td></tr>
<tr><th>270</th><td>7,7</td><td>11,5</td><td>15,1</td><td>18,5</td><td>21,6</td><td>24,5</td><td>27,0</td><td>29,2</td><td>31,0</td><td>32,3</td><td>33,2</td><td>33,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,35</td></tr>
<tr><th>280</th><td>8,6</td><td>12,8</td><td>16,9</td><td>20,7</td><td>24,2</td><td>27,5</td><td>30,4</td><td>33,0</td><td>35,1</td><td>36,8</td><td>38,0</td><td>38,8</td><td>39,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,40</td></tr>
<tr><th>290</th><td>9,6</td><td>14,3</td><td>18,8</td><td>23,1</td><td>27,1</td><td>30,8</td><td>34,1</td><td>37,1</td><td>39,6</td><td>41,7</td><td>43,3</td><td>44,3</td><td>44,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,45</td></tr>
<tr><th>300</th><td>10,6</td><td>15,8</td><td>20,8</td><td>25,6</td><td>30,1</td><td>34,3</td><td>38,1</td><td>41,5</td><td>44,5</td><td>46,9</td><td>48,9</td><td>50,3</td><td>51,1</td><td>51,4</td><td></td><td></td><td></td><td></td><td></td><td>1,50</td></tr>
<tr><th>310</th><td>11,7</td><td>17,5</td><td>23,0</td><td>28,3</td><td>33,4</td><td>38,1</td><td>42,4</td><td>46,3</td><td>49,7</td><td>52,6</td><td>54,9</td><td>56,7</td><td>58,0</td><td>58,6</td><td></td><td></td><td></td><td></td><td></td><td>1,55</td></tr>
<tr><th>320</th><td>12,9</td><td>19,2</td><td>25,4</td><td>31,3</td><td>36,8</td><td>42,1</td><td>46,9</td><td>51,3</td><td>55,3</td><td>58,6</td><td>61,5</td><td>63,7</td><td>65,3</td><td>66,3</td><td>66,6</td><td></td><td></td><td></td><td></td><td>1,60</td></tr>
<tr><th>330</th><td>14,2</td><td>21,1</td><td>27,9</td><td>34,4</td><td>40,5</td><td>46,4</td><td>51,8</td><td>56,8</td><td>61,2</td><td>65,1</td><td>68,4</td><td>71,1</td><td>73,1</td><td>74,5</td><td>75,2</td><td></td><td></td><td></td><td></td><td>1,65</td></tr>
<tr><th>340</th><td>15,5</td><td>23,1</td><td>30,5</td><td>37,7</td><td>44,5</td><td>50,9</td><td>57,0</td><td>62,5</td><td>67,6</td><td>72,0</td><td>75,9</td><td>79,1</td><td>81,6</td><td>83,4</td><td>84,5</td><td>84,8</td><td></td><td></td><td></td><td>1,70</td></tr>
<tr><th>350</th><td>16,9</td><td>25,2</td><td>33,3</td><td>41,2</td><td>48,7</td><td>55,8</td><td>62,5</td><td>68,7</td><td>74,3</td><td>79,4</td><td>83,8</td><td>87,5</td><td>90,6</td><td>92,9</td><td>94,4</td><td>95,2</td><td></td><td></td><td></td><td>1,75</td></tr>
<tr><th>360</th><td>18,4</td><td>27,5</td><td>36,3</td><td>44,9</td><td>53,1</td><td>60,9</td><td>68,3</td><td>75,2</td><td>81,5</td><td>87,2</td><td>92,2</td><td>96,5</td><td>100,1</td><td>103,0</td><td>105,0</td><td>106,2</td><td>106,6</td><td></td><td></td><td>1,80</td></tr>
<tr><th>370</th><td>20,0</td><td>29,8</td><td>39,5</td><td>48,8</td><td>57,8</td><td>66,4</td><td>74,5</td><td>82,1</td><td>89,1</td><td>95,5</td><td>101,2</td><td>106,1</td><td>110,3</td><td>113,7</td><td>116,3</td><td>118,0</td><td>118,9</td><td></td><td></td><td>1,85</td></tr>
<tr><th>380</th><td>21,7</td><td>32,3</td><td>42,8</td><td>52,9</td><td>62,7</td><td>72,1</td><td>81,0</td><td>89,4</td><td>97,1</td><td>104,2</td><td>110,6</td><td>116,3</td><td>121,1</td><td>125,1</td><td>128,3</td><td>130,6</td><td>131,9</td><td>132,4</td><td></td><td>1,90</td></tr>
<tr><th>390</th><td>23,4</td><td>35,0</td><td>46,3</td><td>57,3</td><td>68,0</td><td>78,2</td><td>87,9</td><td>97,1</td><td>105,6</td><td>113,5</td><td>120,7</td><td>127,0</td><td>132,6</td><td>137,2</td><td>141,0</td><td>143,9</td><td>145,8</td><td>146,8</td><td></td><td>1,95</td></tr>
<tr><th>400</th><td>25,3</td><td>37,8</td><td>50,0</td><td>61,9</td><td>73,5</td><td>84,6</td><td>95,2</td><td>105,2</td><td>114,6</td><td>123,3</td><td>131,2</td><td>138,4</td><td>144,6</td><td>150,0</td><td>154,5</td><td>158,0</td><td>160,5</td><td>162,0</td><td>162,5</td><td>2,00</td></tr>
<tr><th>450</th><td>36,0</td><td>53,9</td><td>71,4</td><td>88,6</td><td>105,4</td><td>121,7</td><td>137,4</td><td>152,5</td><td>166,8</td><td>180,3</td><td>193,0</td><td>204,7</td><td>215,4</td><td>225,1</td><td>233,7</td><td>241,2</td><td>247,5</td><td>252,5</td><td>256,4</td><td>2,25</td></tr>
<tr><th>500</th><td>49,5</td><td>74,0</td><td>98,2</td><td>122,0</td><td>145,4</td><td>168,2</td><td>190,4</td><td>211,8</td><td>232,4</td><td>252,1</td><td>270,8</td><td>288,5</td><td>305,0</td><td>320,4</td><td>334,5</td><td>347,2</td><td>358,7</td><td>368,6</td><td>377,2</td><td>2,50</td></tr>
<tr><th>550</th><td>65,9</td><td>98,6</td><td>130,9</td><td>162,9</td><td>194,3</td><td>225,1</td><td>255,2</td><td>284,5</td><td>312,9</td><td>340,3</td><td>366,5</td><td>391,6</td><td>415,5</td><td>438,0</td><td>459,1</td><td>478,7</td><td>496,7</td><td>513,2</td><td>527,9</td><td>2,75</td></tr>
<tr><th>600</th><td>85,6</td><td>128,1</td><td>170,2</td><td>211,9</td><td>253,0</td><td>293,5</td><td>333,2</td><td>372,0</td><td>409,7</td><td>446,4</td><td>481,9</td><td>516,1</td><td>548,9</td><td>580,2</td><td>609,9</td><td>638,0</td><td>664,3</td><td>688,9</td><td>711,5</td><td>3,00</td></tr>
<tr><th>650</th><td>108,8</td><td>162,9</td><td>216,6</td><td>269,9</td><td>322,5</td><td>374,4</td><td>425,4</td><td>475,5</td><td>524,5</td><td>572,2</td><td>618,7</td><td>663,7</td><td>707,3</td><td>749,2</td><td>789,4</td><td>827,7</td><td>864,2</td><td>898,6</td><td>931,0</td><td>3,25</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 174, registre 2.3.4, p. 28)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 174 du PDF
(page imprimée 28)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
