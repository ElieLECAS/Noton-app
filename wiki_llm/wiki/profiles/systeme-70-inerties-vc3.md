---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*C3
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*C3, pression de 1200 Pa, flèche admissible 1/300, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, vc3]
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
    pages: 178
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*C3 (1200 Pa, flèche 1/300)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*C3 : pression de 1200 Pa (1,2 kN/m²), flèche admissible 1/300**
de la portée [1 p. 178].

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
<tr><th>100</th><td>0,4</td><td>0,6</td><td>0,7</td><td>0,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,33</td></tr>
<tr><th>110</th><td>0,6</td><td>0,8</td><td>1,0</td><td>1,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,37</td></tr>
<tr><th>120</th><td>0,7</td><td>1,0</td><td>1,3</td><td>1,4</td><td>1,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,40</td></tr>
<tr><th>130</th><td>0,9</td><td>1,3</td><td>1,7</td><td>1,9</td><td>2,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,43</td></tr>
<tr><th>140</th><td>1,2</td><td>1,7</td><td>2,1</td><td>2,5</td><td>2,7</td><td>2,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,47</td></tr>
<tr><th>150</th><td>1,5</td><td>2,1</td><td>2,7</td><td>3,1</td><td>3,4</td><td>3,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,50</td></tr>
<tr><th>160</th><td>1,8</td><td>2,6</td><td>3,3</td><td>3,9</td><td>4,3</td><td>4,6</td><td>4,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,53</td></tr>
<tr><th>170</th><td>2,1</td><td>3,1</td><td>4,0</td><td>4,8</td><td>5,3</td><td>5,7</td><td>5,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,57</td></tr>
<tr><th>180</th><td>2,6</td><td>3,7</td><td>4,8</td><td>5,7</td><td>6,5</td><td>7,0</td><td>7,4</td><td>7,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,60</td></tr>
<tr><th>190</th><td>3,0</td><td>4,4</td><td>5,7</td><td>6,8</td><td>7,8</td><td>8,5</td><td>9,0</td><td>9,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,63</td></tr>
<tr><th>200</th><td>3,5</td><td>5,2</td><td>6,7</td><td>8,1</td><td>9,2</td><td>10,2</td><td>10,9</td><td>11,3</td><td>11,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,67</td></tr>
<tr><th>210</th><td>4,1</td><td>6,0</td><td>7,8</td><td>9,4</td><td>10,8</td><td>12,0</td><td>12,9</td><td>13,5</td><td>13,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,70</td></tr>
<tr><th>220</th><td>4,7</td><td>6,9</td><td>9,0</td><td>10,9</td><td>12,6</td><td>14,1</td><td>15,2</td><td>16,0</td><td>16,6</td><td>16,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,73</td></tr>
<tr><th>230</th><td>5,4</td><td>7,9</td><td>10,3</td><td>12,6</td><td>14,6</td><td>16,3</td><td>17,7</td><td>18,8</td><td>19,6</td><td>19,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,77</td></tr>
<tr><th>240</th><td>6,1</td><td>9,0</td><td>11,8</td><td>14,4</td><td>16,7</td><td>18,8</td><td>20,5</td><td>21,9</td><td>22,9</td><td>23,5</td><td>23,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>250</th><td>6,9</td><td>10,2</td><td>13,4</td><td>16,3</td><td>19,0</td><td>21,4</td><td>23,5</td><td>25,2</td><td>26,5</td><td>27,4</td><td>27,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,83</td></tr>
<tr><th>260</th><td>7,8</td><td>11,5</td><td>15,1</td><td>18,5</td><td>21,6</td><td>24,4</td><td>26,8</td><td>28,9</td><td>30,5</td><td>31,7</td><td>32,4</td><td>32,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,87</td></tr>
<tr><th>270</th><td>8,7</td><td>12,9</td><td>17,0</td><td>20,8</td><td>24,3</td><td>27,5</td><td>30,4</td><td>32,8</td><td>34,8</td><td>36,3</td><td>37,4</td><td>37,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,90</td></tr>
<tr><th>280</th><td>9,7</td><td>14,4</td><td>19,0</td><td>23,3</td><td>27,3</td><td>31,0</td><td>34,2</td><td>37,1</td><td>39,5</td><td>41,4</td><td>42,8</td><td>43,6</td><td>43,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,93</td></tr>
<tr><th>290</th><td>10,8</td><td>16,1</td><td>21,1</td><td>25,9</td><td>30,5</td><td>34,6</td><td>38,4</td><td>41,7</td><td>44,6</td><td>46,9</td><td>48,7</td><td>49,8</td><td>50,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,97</td></tr>
<tr><th>300</th><td>12,0</td><td>17,8</td><td>23,4</td><td>28,8</td><td>33,9</td><td>38,6</td><td>42,9</td><td>46,7</td><td>50,0</td><td>52,8</td><td>55,0</td><td>56,6</td><td>57,5</td><td>57,9</td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>310</th><td>13,2</td><td>19,7</td><td>25,9</td><td>31,9</td><td>37,5</td><td>42,8</td><td>47,7</td><td>52,0</td><td>55,9</td><td>59,2</td><td>61,8</td><td>63,8</td><td>65,2</td><td>65,9</td><td></td><td></td><td></td><td></td><td></td><td>1,03</td></tr>
<tr><th>320</th><td>14,5</td><td>21,6</td><td>28,5</td><td>35,2</td><td>41,5</td><td>47,4</td><td>52,8</td><td>57,8</td><td>62,2</td><td>66,0</td><td>69,1</td><td>71,6</td><td>73,4</td><td>74,5</td><td>74,9</td><td></td><td></td><td></td><td></td><td>1,07</td></tr>
<tr><th>330</th><td>15,9</td><td>23,7</td><td>31,3</td><td>38,6</td><td>45,6</td><td>52,2</td><td>58,3</td><td>63,9</td><td>68,9</td><td>73,2</td><td>77,0</td><td>80,0</td><td>82,3</td><td>83,8</td><td>84,6</td><td></td><td></td><td></td><td></td><td>1,10</td></tr>
<tr><th>340</th><td>17,4</td><td>26,0</td><td>34,3</td><td>42,4</td><td>50,0</td><td>57,3</td><td>64,1</td><td>70,4</td><td>76,0</td><td>81,0</td><td>85,3</td><td>88,9</td><td>91,8</td><td>93,8</td><td>95,0</td><td>95,5</td><td></td><td></td><td></td><td>1,13</td></tr>
<tr><th>350</th><td>19,0</td><td>28,4</td><td>37,5</td><td>46,3</td><td>54,8</td><td>62,8</td><td>70,3</td><td>77,3</td><td>83,6</td><td>89,3</td><td>94,3</td><td>98,5</td><td>101,9</td><td>104,5</td><td>106,2</td><td>107,1</td><td></td><td></td><td></td><td>1,17</td></tr>
<tr><th>360</th><td>20,7</td><td>30,9</td><td>40,8</td><td>50,5</td><td>59,7</td><td>68,6</td><td>76,9</td><td>84,6</td><td>91,7</td><td>98,1</td><td>103,7</td><td>108,6</td><td>112,7</td><td>115,8</td><td>118,1</td><td>119,5</td><td>120,0</td><td></td><td></td><td>1,20</td></tr>
<tr><th>370</th><td>22,5</td><td>33,6</td><td>44,4</td><td>54,9</td><td>65,0</td><td>74,7</td><td>83,8</td><td>92,4</td><td>100,2</td><td>107,4</td><td>113,8</td><td>119,4</td><td>124,1</td><td>127,9</td><td>130,8</td><td>132,8</td><td>133,7</td><td></td><td></td><td>1,23</td></tr>
<tr><th>380</th><td>24,4</td><td>36,4</td><td>48,1</td><td>59,6</td><td>70,6</td><td>81,1</td><td>91,2</td><td>100,6</td><td>109,3</td><td>117,3</td><td>124,5</td><td>130,8</td><td>136,3</td><td>140,8</td><td>144,3</td><td>146,9</td><td>148,4</td><td>148,9</td><td></td><td>1,27</td></tr>
<tr><th>390</th><td>26,4</td><td>39,3</td><td>52,1</td><td>64,5</td><td>76,5</td><td>88,0</td><td>98,9</td><td>109,2</td><td>118,8</td><td>127,7</td><td>135,7</td><td>142,9</td><td>149,1</td><td>154,4</td><td>158,6</td><td>161,9</td><td>164,0</td><td>165,1</td><td></td><td>1,30</td></tr>
<tr><th>400</th><td>28,5</td><td>42,5</td><td>56,2</td><td>69,7</td><td>82,7</td><td>95,2</td><td>107,1</td><td>118,4</td><td>128,9</td><td>138,7</td><td>147,6</td><td>155,7</td><td>162,7</td><td>168,8</td><td>173,8</td><td>177,7</td><td>180,6</td><td>182,3</td><td>182,9</td><td>1,33</td></tr>
<tr><th>450</th><td>40,6</td><td>60,6</td><td>80,3</td><td>99,7</td><td>118,6</td><td>136,9</td><td>154,6</td><td>171,5</td><td>187,7</td><td>202,9</td><td>217,1</td><td>230,3</td><td>242,4</td><td>253,3</td><td>262,9</td><td>271,3</td><td>278,4</td><td>284,1</td><td>288,4</td><td>1,50</td></tr>
<tr><th>500</th><td>55,7</td><td>83,2</td><td>110,5</td><td>137,3</td><td>163,6</td><td>189,2</td><td>214,2</td><td>238,3</td><td>261,4</td><td>283,6</td><td>304,7</td><td>324,6</td><td>343,2</td><td>360,4</td><td>376,3</td><td>390,7</td><td>403,5</td><td>414,7</td><td>424,3</td><td>1,67</td></tr>
<tr><th>550</th><td>74,1</td><td>110,9</td><td>147,3</td><td>183,2</td><td>218,6</td><td>253,3</td><td>287,1</td><td>320,1</td><td>352,0</td><td>382,8</td><td>412,4</td><td>440,6</td><td>467,4</td><td>492,7</td><td>516,5</td><td>538,5</td><td>558,8</td><td>577,3</td><td>593,9</td><td>1,83</td></tr>
<tr><th>600</th><td>96,3</td><td>144,1</td><td>191,5</td><td>238,4</td><td>284,7</td><td>330,2</td><td>374,8</td><td>418,4</td><td>461,0</td><td>502,2</td><td>542,1</td><td>580,6</td><td>617,5</td><td>652,7</td><td>686,2</td><td>717,7</td><td>747,4</td><td>775,0</td><td>800,5</td><td>2,00</td></tr>
<tr><th>650</th><td>122,4</td><td>183,3</td><td>243,7</td><td>303,6</td><td>362,8</td><td>421,2</td><td>478,6</td><td>534,9</td><td>590,0</td><td>643,8</td><td>696,0</td><td>746,7</td><td>795,7</td><td>842,8</td><td>888,0</td><td>931,2</td><td>972,2</td><td>1010,9</td><td>1047,3</td><td>2,17</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 178, registre 2.3.4, p. 32)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 178 du PDF
(page imprimée 32)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
