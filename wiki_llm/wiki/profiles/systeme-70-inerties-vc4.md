---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*C4
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*C4, pression de 1600 Pa, flèche admissible 1/300, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, vc4]
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
    pages: 179
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*C4 (1600 Pa, flèche 1/300)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*C4 : pression de 1600 Pa (1,6 kN/m²), flèche admissible 1/300**
de la portée [1 p. 179].

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
<tr><th>100</th><td>0,6</td><td>0,8</td><td>0,9</td><td>1,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,33</td></tr>
<tr><th>110</th><td>0,8</td><td>1,1</td><td>1,3</td><td>1,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,37</td></tr>
<tr><th>120</th><td>1,0</td><td>1,4</td><td>1,7</td><td>1,9</td><td>2,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,40</td></tr>
<tr><th>130</th><td>1,3</td><td>1,8</td><td>2,2</td><td>2,5</td><td>2,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,43</td></tr>
<tr><th>140</th><td>1,6</td><td>2,3</td><td>2,9</td><td>3,3</td><td>3,6</td><td>3,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,47</td></tr>
<tr><th>150</th><td>2,0</td><td>2,8</td><td>3,6</td><td>4,2</td><td>4,6</td><td>4,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,50</td></tr>
<tr><th>160</th><td>2,4</td><td>3,5</td><td>4,4</td><td>5,2</td><td>5,8</td><td>6,1</td><td>6,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,53</td></tr>
<tr><th>170</th><td>2,9</td><td>4,2</td><td>5,3</td><td>6,3</td><td>7,1</td><td>7,6</td><td>7,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,57</td></tr>
<tr><th>180</th><td>3,4</td><td>5,0</td><td>6,4</td><td>7,6</td><td>8,6</td><td>9,4</td><td>9,8</td><td>10,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,60</td></tr>
<tr><th>190</th><td>4,0</td><td>5,9</td><td>7,6</td><td>9,1</td><td>10,4</td><td>11,4</td><td>12,0</td><td>12,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,63</td></tr>
<tr><th>200</th><td>4,7</td><td>6,9</td><td>8,9</td><td>10,7</td><td>12,3</td><td>13,6</td><td>14,5</td><td>15,0</td><td>15,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,67</td></tr>
<tr><th>210</th><td>5,4</td><td>8,0</td><td>10,4</td><td>12,6</td><td>14,4</td><td>16,0</td><td>17,2</td><td>18,1</td><td>18,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,70</td></tr>
<tr><th>220</th><td>6,3</td><td>9,2</td><td>12,0</td><td>14,6</td><td>16,8</td><td>18,7</td><td>20,3</td><td>21,4</td><td>22,1</td><td>22,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,73</td></tr>
<tr><th>230</th><td>7,2</td><td>10,6</td><td>13,8</td><td>16,8</td><td>19,4</td><td>21,7</td><td>23,6</td><td>25,1</td><td>26,1</td><td>26,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,77</td></tr>
<tr><th>240</th><td>8,1</td><td>12,0</td><td>15,7</td><td>19,2</td><td>22,3</td><td>25,0</td><td>27,3</td><td>29,2</td><td>30,5</td><td>31,3</td><td>31,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>250</th><td>9,2</td><td>13,6</td><td>17,8</td><td>21,8</td><td>25,4</td><td>28,6</td><td>31,4</td><td>33,6</td><td>35,4</td><td>36,5</td><td>37,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,83</td></tr>
<tr><th>260</th><td>10,4</td><td>15,4</td><td>20,1</td><td>24,6</td><td>28,8</td><td>32,5</td><td>35,7</td><td>38,5</td><td>40,7</td><td>42,2</td><td>43,2</td><td>43,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,87</td></tr>
<tr><th>270</th><td>11,6</td><td>17,2</td><td>22,6</td><td>27,7</td><td>32,4</td><td>36,7</td><td>40,5</td><td>43,8</td><td>46,4</td><td>48,5</td><td>49,8</td><td>50,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,90</td></tr>
<tr><th>280</th><td>13,0</td><td>19,2</td><td>25,3</td><td>31,0</td><td>36,4</td><td>41,3</td><td>45,7</td><td>49,5</td><td>52,7</td><td>55,2</td><td>57,1</td><td>58,2</td><td>58,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,93</td></tr>
<tr><th>290</th><td>14,4</td><td>21,4</td><td>28,2</td><td>34,6</td><td>40,6</td><td>46,2</td><td>51,2</td><td>55,6</td><td>59,4</td><td>62,5</td><td>64,9</td><td>66,5</td><td>67,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,97</td></tr>
<tr><th>300</th><td>16,0</td><td>23,7</td><td>31,2</td><td>38,4</td><td>45,2</td><td>51,5</td><td>57,2</td><td>62,3</td><td>66,7</td><td>70,4</td><td>73,3</td><td>75,4</td><td>76,7</td><td>77,1</td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>310</th><td>17,6</td><td>26,2</td><td>34,5</td><td>42,5</td><td>50,1</td><td>57,1</td><td>63,6</td><td>69,4</td><td>74,5</td><td>78,9</td><td>82,4</td><td>85,1</td><td>86,9</td><td>87,8</td><td></td><td></td><td></td><td></td><td></td><td>1,03</td></tr>
<tr><th>320</th><td>19,4</td><td>28,8</td><td>38,0</td><td>46,9</td><td>55,3</td><td>63,1</td><td>70,4</td><td>77,0</td><td>82,9</td><td>88,0</td><td>92,2</td><td>95,5</td><td>97,9</td><td>99,4</td><td>99,9</td><td></td><td></td><td></td><td></td><td>1,07</td></tr>
<tr><th>330</th><td>21,3</td><td>31,7</td><td>41,8</td><td>51,5</td><td>60,8</td><td>69,6</td><td>77,7</td><td>85,1</td><td>91,8</td><td>97,7</td><td>102,6</td><td>106,7</td><td>109,7</td><td>111,8</td><td>112,8</td><td></td><td></td><td></td><td></td><td>1,10</td></tr>
<tr><th>340</th><td>23,3</td><td>34,7</td><td>45,8</td><td>56,5</td><td>66,7</td><td>76,4</td><td>85,5</td><td>93,8</td><td>101,3</td><td>108,0</td><td>113,8</td><td>118,6</td><td>122,4</td><td>125,1</td><td>126,7</td><td>127,3</td><td></td><td></td><td></td><td>1,13</td></tr>
<tr><th>350</th><td>25,4</td><td>37,8</td><td>50,0</td><td>61,7</td><td>73,0</td><td>83,7</td><td>93,7</td><td>103,0</td><td>111,5</td><td>119,1</td><td>125,7</td><td>131,3</td><td>135,8</td><td>139,3</td><td>141,6</td><td>142,8</td><td></td><td></td><td></td><td>1,17</td></tr>
<tr><th>360</th><td>27,6</td><td>41,2</td><td>54,5</td><td>67,3</td><td>79,7</td><td>91,4</td><td>102,5</td><td>112,8</td><td>122,2</td><td>130,8</td><td>138,3</td><td>144,8</td><td>150,2</td><td>154,4</td><td>157,5</td><td>159,3</td><td>160,0</td><td></td><td></td><td>1,20</td></tr>
<tr><th>370</th><td>30,0</td><td>44,8</td><td>59,2</td><td>73,2</td><td>86,7</td><td>99,6</td><td>111,8</td><td>123,1</td><td>133,6</td><td>143,2</td><td>151,7</td><td>159,2</td><td>165,5</td><td>170,6</td><td>174,4</td><td>177,0</td><td>178,3</td><td></td><td></td><td>1,23</td></tr>
<tr><th>380</th><td>32,5</td><td>48,5</td><td>64,2</td><td>79,4</td><td>94,1</td><td>108,2</td><td>121,5</td><td>134,1</td><td>145,7</td><td>156,4</td><td>166,0</td><td>174,4</td><td>181,7</td><td>187,7</td><td>192,4</td><td>195,8</td><td>197,9</td><td>198,6</td><td></td><td>1,27</td></tr>
<tr><th>390</th><td>35,2</td><td>52,5</td><td>69,4</td><td>86,0</td><td>102,0</td><td>117,3</td><td>131,9</td><td>145,6</td><td>158,5</td><td>170,3</td><td>181,0</td><td>190,5</td><td>198,8</td><td>205,8</td><td>211,5</td><td>215,8</td><td>218,7</td><td>220,1</td><td></td><td>1,30</td></tr>
<tr><th>400</th><td>37,9</td><td>56,6</td><td>75,0</td><td>92,9</td><td>110,2</td><td>126,9</td><td>142,8</td><td>157,8</td><td>171,9</td><td>184,9</td><td>196,8</td><td>207,5</td><td>217,0</td><td>225,0</td><td>231,7</td><td>237,0</td><td>240,8</td><td>243,0</td><td>243,8</td><td>1,33</td></tr>
<tr><th>450</th><td>54,1</td><td>80,8</td><td>107,1</td><td>132,9</td><td>158,1</td><td>182,6</td><td>206,1</td><td>228,7</td><td>250,2</td><td>270,5</td><td>289,5</td><td>307,1</td><td>323,2</td><td>337,7</td><td>350,6</td><td>361,8</td><td>371,2</td><td>378,8</td><td>384,5</td><td>1,50</td></tr>
<tr><th>500</th><td>74,2</td><td>111,0</td><td>147,3</td><td>183,0</td><td>218,1</td><td>252,3</td><td>285,6</td><td>317,7</td><td>348,6</td><td>378,1</td><td>406,2</td><td>432,7</td><td>457,5</td><td>480,6</td><td>501,7</td><td>520,9</td><td>538,0</td><td>553,0</td><td>565,8</td><td>1,67</td></tr>
<tr><th>550</th><td>98,8</td><td>147,8</td><td>196,4</td><td>244,3</td><td>291,5</td><td>337,7</td><td>382,8</td><td>426,8</td><td>469,3</td><td>510,4</td><td>549,8</td><td>587,5</td><td>623,2</td><td>657,0</td><td>688,6</td><td>718,0</td><td>745,1</td><td>769,7</td><td>791,9</td><td>1,83</td></tr>
<tr><th>600</th><td>128,3</td><td>192,1</td><td>255,3</td><td>317,9</td><td>379,6</td><td>440,3</td><td>499,8</td><td>557,9</td><td>614,6</td><td>669,6</td><td>722,8</td><td>774,1</td><td>823,3</td><td>870,3</td><td>914,9</td><td>957,0</td><td>996,5</td><td>1033,3</td><td>1067,3</td><td>2,00</td></tr>
<tr><th>650</th><td>163,2</td><td>244,4</td><td>325,0</td><td>404,8</td><td>483,7</td><td>561,6</td><td>638,1</td><td>713,2</td><td>786,7</td><td>858,3</td><td>928,0</td><td>995,6</td><td>1060,9</td><td>1123,8</td><td>1184,0</td><td>1241,6</td><td>1296,2</td><td>1347,9</td><td>1396,4</td><td>2,17</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 179, registre 2.3.4, p. 33)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 179 du PDF
(page imprimée 33)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
