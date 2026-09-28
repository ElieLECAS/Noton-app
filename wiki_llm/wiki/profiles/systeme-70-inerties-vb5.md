---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*B5
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*B5, pression de 2000 Pa, flèche admissible 1/200, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, vb5]
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
    pages: 175
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*B5 (2000 Pa, flèche 1/200)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*B5 : pression de 2000 Pa (2,0 kN/m²), flèche admissible 1/200**
de la portée [1 p. 175].

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
<tr><th>100</th><td>0,5</td><td>0,6</td><td>0,8</td><td>0,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,50</td></tr>
<tr><th>110</th><td>0,6</td><td>0,9</td><td>1,1</td><td>1,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,55</td></tr>
<tr><th>120</th><td>0,8</td><td>1,2</td><td>1,4</td><td>1,6</td><td>1,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,60</td></tr>
<tr><th>130</th><td>1,0</td><td>1,5</td><td>1,9</td><td>2,1</td><td>2,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,65</td></tr>
<tr><th>140</th><td>1,3</td><td>1,9</td><td>2,4</td><td>2,7</td><td>3,0</td><td>3,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,70</td></tr>
<tr><th>150</th><td>1,6</td><td>2,4</td><td>3,0</td><td>3,5</td><td>3,8</td><td>4,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,75</td></tr>
<tr><th>160</th><td>2,0</td><td>2,9</td><td>3,7</td><td>4,3</td><td>4,8</td><td>5,1</td><td>5,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>170</th><td>2,4</td><td>3,5</td><td>4,5</td><td>5,3</td><td>5,9</td><td>6,4</td><td>6,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,85</td></tr>
<tr><th>180</th><td>2,8</td><td>4,1</td><td>5,3</td><td>6,4</td><td>7,2</td><td>7,8</td><td>8,2</td><td>8,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,90</td></tr>
<tr><th>190</th><td>3,3</td><td>4,9</td><td>6,3</td><td>7,6</td><td>8,6</td><td>9,5</td><td>10,0</td><td>10,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,95</td></tr>
<tr><th>200</th><td>3,9</td><td>5,7</td><td>7,4</td><td>9,0</td><td>10,3</td><td>11,3</td><td>12,1</td><td>12,5</td><td>12,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>210</th><td>4,5</td><td>6,7</td><td>8,7</td><td>10,5</td><td>12,0</td><td>13,3</td><td>14,4</td><td>15,0</td><td>15,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,05</td></tr>
<tr><th>220</th><td>5,2</td><td>7,7</td><td>10,0</td><td>12,1</td><td>14,0</td><td>15,6</td><td>16,9</td><td>17,8</td><td>18,4</td><td>18,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,10</td></tr>
<tr><th>230</th><td>6,0</td><td>8,8</td><td>11,5</td><td>14,0</td><td>16,2</td><td>18,1</td><td>19,7</td><td>20,9</td><td>21,7</td><td>22,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,15</td></tr>
<tr><th>240</th><td>6,8</td><td>10,0</td><td>13,1</td><td>16,0</td><td>18,6</td><td>20,8</td><td>22,8</td><td>24,3</td><td>25,4</td><td>26,1</td><td>26,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,20</td></tr>
<tr><th>250</th><td>7,7</td><td>11,4</td><td>14,9</td><td>18,2</td><td>21,2</td><td>23,8</td><td>26,1</td><td>28,0</td><td>29,5</td><td>30,4</td><td>30,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,25</td></tr>
<tr><th>260</th><td>8,6</td><td>12,8</td><td>16,8</td><td>20,5</td><td>24,0</td><td>27,1</td><td>29,8</td><td>32,1</td><td>33,9</td><td>35,2</td><td>36,0</td><td>36,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,30</td></tr>
<tr><th>270</th><td>9,7</td><td>14,4</td><td>18,8</td><td>23,1</td><td>27,0</td><td>30,6</td><td>33,8</td><td>36,5</td><td>38,7</td><td>40,4</td><td>41,5</td><td>42,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,35</td></tr>
<tr><th>280</th><td>10,8</td><td>16,0</td><td>21,1</td><td>25,9</td><td>30,3</td><td>34,4</td><td>38,1</td><td>41,2</td><td>43,9</td><td>46,0</td><td>47,5</td><td>48,5</td><td>48,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,40</td></tr>
<tr><th>290</th><td>12,0</td><td>17,8</td><td>23,5</td><td>28,8</td><td>33,9</td><td>38,5</td><td>42,7</td><td>46,4</td><td>49,5</td><td>52,1</td><td>54,1</td><td>55,4</td><td>56,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,45</td></tr>
<tr><th>300</th><td>13,3</td><td>19,8</td><td>26,0</td><td>32,0</td><td>37,6</td><td>42,9</td><td>47,6</td><td>51,9</td><td>55,6</td><td>58,7</td><td>61,1</td><td>62,9</td><td>63,9</td><td>64,3</td><td></td><td></td><td></td><td></td><td></td><td>1,50</td></tr>
<tr><th>310</th><td>14,7</td><td>21,8</td><td>28,8</td><td>35,4</td><td>41,7</td><td>47,6</td><td>53,0</td><td>57,8</td><td>62,1</td><td>65,7</td><td>68,7</td><td>70,9</td><td>72,4</td><td>73,2</td><td></td><td></td><td></td><td></td><td></td><td>1,55</td></tr>
<tr><th>320</th><td>16,2</td><td>24,0</td><td>31,7</td><td>39,1</td><td>46,1</td><td>52,6</td><td>58,7</td><td>64,2</td><td>69,1</td><td>73,3</td><td>76,8</td><td>79,6</td><td>81,6</td><td>82,8</td><td>83,2</td><td></td><td></td><td></td><td></td><td>1,60</td></tr>
<tr><th>330</th><td>17,7</td><td>26,4</td><td>34,8</td><td>42,9</td><td>50,7</td><td>58,0</td><td>64,8</td><td>71,0</td><td>76,5</td><td>81,4</td><td>85,5</td><td>88,9</td><td>91,4</td><td>93,2</td><td>94,0</td><td></td><td></td><td></td><td></td><td>1,65</td></tr>
<tr><th>340</th><td>19,4</td><td>28,9</td><td>38,1</td><td>47,1</td><td>55,6</td><td>63,7</td><td>71,2</td><td>78,2</td><td>84,5</td><td>90,0</td><td>94,8</td><td>98,8</td><td>102,0</td><td>104,2</td><td>105,6</td><td>106,1</td><td></td><td></td><td></td><td>1,70</td></tr>
<tr><th>350</th><td>21,2</td><td>31,5</td><td>41,7</td><td>51,4</td><td>60,8</td><td>69,7</td><td>78,1</td><td>85,8</td><td>92,9</td><td>99,2</td><td>104,7</td><td>109,4</td><td>113,2</td><td>116,1</td><td>118,0</td><td>119,0</td><td></td><td></td><td></td><td>1,75</td></tr>
<tr><th>360</th><td>23,0</td><td>34,3</td><td>45,4</td><td>56,1</td><td>66,4</td><td>76,2</td><td>85,4</td><td>94,0</td><td>101,9</td><td>109,0</td><td>115,3</td><td>120,7</td><td>125,2</td><td>128,7</td><td>131,3</td><td>132,8</td><td>133,3</td><td></td><td></td><td>1,80</td></tr>
<tr><th>370</th><td>25,0</td><td>37,3</td><td>49,3</td><td>61,0</td><td>72,2</td><td>83,0</td><td>93,1</td><td>102,6</td><td>111,4</td><td>119,3</td><td>126,4</td><td>132,7</td><td>137,9</td><td>142,1</td><td>145,4</td><td>147,5</td><td>148,6</td><td></td><td></td><td>1,85</td></tr>
<tr><th>380</th><td>27,1</td><td>40,4</td><td>53,5</td><td>66,2</td><td>78,4</td><td>90,2</td><td>101,3</td><td>111,7</td><td>121,4</td><td>130,3</td><td>138,3</td><td>145,3</td><td>151,4</td><td>156,4</td><td>160,4</td><td>163,2</td><td>164,9</td><td>165,5</td><td></td><td>1,90</td></tr>
<tr><th>390</th><td>29,3</td><td>43,7</td><td>57,9</td><td>71,6</td><td>85,0</td><td>97,7</td><td>109,9</td><td>121,4</td><td>132,1</td><td>141,9</td><td>150,8</td><td>158,8</td><td>165,7</td><td>171,5</td><td>176,3</td><td>179,8</td><td>182,3</td><td>183,5</td><td></td><td>1,95</td></tr>
<tr><th>400</th><td>31,6</td><td>47,2</td><td>62,5</td><td>77,4</td><td>91,8</td><td>105,7</td><td>119,0</td><td>131,5</td><td>143,3</td><td>154,1</td><td>164,0</td><td>172,9</td><td>180,8</td><td>187,5</td><td>193,1</td><td>197,5</td><td>200,6</td><td>202,5</td><td>203,2</td><td>2,00</td></tr>
<tr><th>450</th><td>45,1</td><td>67,3</td><td>89,3</td><td>110,8</td><td>131,8</td><td>152,1</td><td>171,8</td><td>190,6</td><td>208,5</td><td>225,4</td><td>241,2</td><td>255,9</td><td>269,3</td><td>281,4</td><td>292,2</td><td>301,5</td><td>309,3</td><td>315,7</td><td>320,4</td><td>2,25</td></tr>
<tr><th>500</th><td>61,8</td><td>92,5</td><td>122,7</td><td>152,5</td><td>181,8</td><td>210,3</td><td>238,0</td><td>264,7</td><td>290,5</td><td>315,1</td><td>338,5</td><td>360,6</td><td>381,3</td><td>400,5</td><td>418,1</td><td>434,1</td><td>448,3</td><td>460,8</td><td>471,5</td><td>2,50</td></tr>
<tr><th>550</th><td>82,4</td><td>123,2</td><td>163,7</td><td>203,6</td><td>242,9</td><td>281,4</td><td>319,0</td><td>355,6</td><td>391,1</td><td>425,3</td><td>458,2</td><td>489,5</td><td>519,4</td><td>547,5</td><td>573,8</td><td>598,4</td><td>620,9</td><td>641,5</td><td>659,9</td><td>2,75</td></tr>
<tr><th>600</th><td>107,0</td><td>160,1</td><td>212,8</td><td>264,9</td><td>316,3</td><td>366,9</td><td>416,5</td><td>464,9</td><td>512,2</td><td>558,0</td><td>602,4</td><td>645,1</td><td>686,1</td><td>725,2</td><td>762,4</td><td>797,5</td><td>830,4</td><td>861,1</td><td>889,4</td><td>3,00</td></tr>
<tr><th>650</th><td>136,0</td><td>203,6</td><td>270,8</td><td>337,3</td><td>403,1</td><td>468,0</td><td>531,8</td><td>594,3</td><td>655,6</td><td>715,3</td><td>773,4</td><td>829,7</td><td>884,1</td><td>936,5</td><td>986,7</td><td>1034,6</td><td>1080,2</td><td>1123,2</td><td>1163,7</td><td>3,25</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 175, registre 2.3.4, p. 29)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 175 du PDF
(page imprimée 29)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
