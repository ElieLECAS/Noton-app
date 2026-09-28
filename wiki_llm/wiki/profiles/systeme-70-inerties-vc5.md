---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*C5
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*C5, pression de 2000 Pa, flèche admissible 1/300, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, vc5]
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
    pages: 180
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*C5 (2000 Pa, flèche 1/300)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*C5 : pression de 2000 Pa (2,0 kN/m²), flèche admissible 1/300**
de la portée [1 p. 180].

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
<tr><th>100</th><td>0,7</td><td>1,0</td><td>1,1</td><td>1,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,33</td></tr>
<tr><th>110</th><td>0,9</td><td>1,3</td><td>1,6</td><td>1,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,37</td></tr>
<tr><th>120</th><td>1,2</td><td>1,7</td><td>2,1</td><td>2,4</td><td>2,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,40</td></tr>
<tr><th>130</th><td>1,6</td><td>2,2</td><td>2,8</td><td>3,2</td><td>3,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,43</td></tr>
<tr><th>140</th><td>2,0</td><td>2,8</td><td>3,6</td><td>4,1</td><td>4,5</td><td>4,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,47</td></tr>
<tr><th>150</th><td>2,4</td><td>3,5</td><td>4,5</td><td>5,2</td><td>5,7</td><td>6,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,50</td></tr>
<tr><th>160</th><td>3,0</td><td>4,3</td><td>5,5</td><td>6,5</td><td>7,2</td><td>7,7</td><td>7,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,53</td></tr>
<tr><th>170</th><td>3,6</td><td>5,2</td><td>6,7</td><td>7,9</td><td>8,9</td><td>9,6</td><td>9,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,57</td></tr>
<tr><th>180</th><td>4,3</td><td>6,2</td><td>8,0</td><td>9,6</td><td>10,8</td><td>11,7</td><td>12,3</td><td>12,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,60</td></tr>
<tr><th>190</th><td>5,0</td><td>7,4</td><td>9,5</td><td>11,4</td><td>13,0</td><td>14,2</td><td>15,0</td><td>15,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,63</td></tr>
<tr><th>200</th><td>5,9</td><td>8,6</td><td>11,2</td><td>13,4</td><td>15,4</td><td>17,0</td><td>18,1</td><td>18,8</td><td>19,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,67</td></tr>
<tr><th>210</th><td>6,8</td><td>10,0</td><td>13,0</td><td>15,7</td><td>18,1</td><td>20,0</td><td>21,5</td><td>22,6</td><td>23,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,70</td></tr>
<tr><th>220</th><td>7,8</td><td>11,5</td><td>15,0</td><td>18,2</td><td>21,0</td><td>23,4</td><td>25,3</td><td>26,7</td><td>27,6</td><td>27,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,73</td></tr>
<tr><th>230</th><td>8,9</td><td>13,2</td><td>17,2</td><td>21,0</td><td>24,3</td><td>27,2</td><td>29,5</td><td>31,4</td><td>32,6</td><td>33,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,77</td></tr>
<tr><th>240</th><td>10,2</td><td>15,0</td><td>19,7</td><td>24,0</td><td>27,8</td><td>31,3</td><td>34,2</td><td>36,5</td><td>38,1</td><td>39,2</td><td>39,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>250</th><td>11,5</td><td>17,0</td><td>22,3</td><td>27,2</td><td>31,7</td><td>35,7</td><td>39,2</td><td>42,0</td><td>44,2</td><td>45,7</td><td>46,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,83</td></tr>
<tr><th>260</th><td>13,0</td><td>19,2</td><td>25,2</td><td>30,8</td><td>36,0</td><td>40,6</td><td>44,7</td><td>48,1</td><td>50,8</td><td>52,8</td><td>54,0</td><td>54,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,87</td></tr>
<tr><th>270</th><td>14,5</td><td>21,5</td><td>28,3</td><td>34,6</td><td>40,5</td><td>45,9</td><td>50,6</td><td>54,7</td><td>58,0</td><td>60,6</td><td>62,3</td><td>63,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,90</td></tr>
<tr><th>280</th><td>16,2</td><td>24,1</td><td>31,6</td><td>38,8</td><td>45,5</td><td>51,6</td><td>57,1</td><td>61,9</td><td>65,9</td><td>69,0</td><td>71,3</td><td>72,7</td><td>73,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,93</td></tr>
<tr><th>290</th><td>18,0</td><td>26,8</td><td>35,2</td><td>43,2</td><td>50,8</td><td>57,7</td><td>64,0</td><td>69,6</td><td>74,3</td><td>78,2</td><td>81,1</td><td>83,1</td><td>84,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,97</td></tr>
<tr><th>300</th><td>19,9</td><td>29,7</td><td>39,0</td><td>48,0</td><td>56,5</td><td>64,3</td><td>71,5</td><td>77,9</td><td>83,4</td><td>88,0</td><td>91,7</td><td>94,3</td><td>95,9</td><td>96,4</td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>310</th><td>22,0</td><td>32,8</td><td>43,2</td><td>53,1</td><td>62,6</td><td>71,4</td><td>79,5</td><td>86,7</td><td>93,1</td><td>98,6</td><td>103,0</td><td>106,4</td><td>108,7</td><td>109,8</td><td></td><td></td><td></td><td></td><td></td><td>1,03</td></tr>
<tr><th>320</th><td>24,2</td><td>36,1</td><td>47,6</td><td>58,6</td><td>69,1</td><td>78,9</td><td>88,0</td><td>96,3</td><td>103,6</td><td>109,9</td><td>115,2</td><td>119,4</td><td>122,4</td><td>124,2</td><td>124,8</td><td></td><td></td><td></td><td></td><td>1,07</td></tr>
<tr><th>330</th><td>26,6</td><td>39,6</td><td>52,2</td><td>64,4</td><td>76,0</td><td>87,0</td><td>97,1</td><td>106,4</td><td>114,8</td><td>122,1</td><td>128,3</td><td>133,3</td><td>137,2</td><td>139,7</td><td>141,0</td><td></td><td></td><td></td><td></td><td>1,10</td></tr>
<tr><th>340</th><td>29,1</td><td>43,3</td><td>57,2</td><td>70,6</td><td>83,4</td><td>95,5</td><td>106,8</td><td>117,3</td><td>126,7</td><td>135,0</td><td>142,2</td><td>148,2</td><td>152,9</td><td>156,3</td><td>158,4</td><td>159,1</td><td></td><td></td><td></td><td>1,13</td></tr>
<tr><th>350</th><td>31,7</td><td>47,3</td><td>62,5</td><td>77,2</td><td>91,3</td><td>104,6</td><td>117,2</td><td>128,8</td><td>139,4</td><td>148,8</td><td>157,1</td><td>164,1</td><td>169,8</td><td>174,1</td><td>177,0</td><td>178,5</td><td></td><td></td><td></td><td>1,17</td></tr>
<tr><th>360</th><td>34,5</td><td>51,5</td><td>68,1</td><td>84,1</td><td>99,6</td><td>114,3</td><td>128,1</td><td>141,0</td><td>152,8</td><td>163,5</td><td>172,9</td><td>181,0</td><td>187,8</td><td>193,1</td><td>196,9</td><td>199,2</td><td>200,0</td><td></td><td></td><td>1,20</td></tr>
<tr><th>370</th><td>37,5</td><td>55,9</td><td>74,0</td><td>91,5</td><td>108,4</td><td>124,5</td><td>139,7</td><td>153,9</td><td>167,1</td><td>179,0</td><td>189,7</td><td>199,0</td><td>206,8</td><td>213,2</td><td>218,0</td><td>221,3</td><td>222,9</td><td></td><td></td><td>1,23</td></tr>
<tr><th>380</th><td>40,6</td><td>60,6</td><td>80,2</td><td>99,3</td><td>117,6</td><td>135,2</td><td>151,9</td><td>167,6</td><td>182,1</td><td>195,5</td><td>207,4</td><td>218,0</td><td>227,1</td><td>234,6</td><td>240,5</td><td>244,8</td><td>247,4</td><td>248,2</td><td></td><td>1,27</td></tr>
<tr><th>390</th><td>44,0</td><td>65,6</td><td>86,8</td><td>107,5</td><td>127,4</td><td>146,6</td><td>164,9</td><td>182,1</td><td>198,1</td><td>212,8</td><td>226,2</td><td>238,2</td><td>248,5</td><td>257,3</td><td>264,4</td><td>269,8</td><td>273,4</td><td>275,2</td><td></td><td>1,30</td></tr>
<tr><th>400</th><td>47,4</td><td>70,8</td><td>93,7</td><td>116,1</td><td>137,8</td><td>158,6</td><td>178,5</td><td>197,3</td><td>214,9</td><td>231,2</td><td>246,1</td><td>259,4</td><td>271,2</td><td>281,3</td><td>289,7</td><td>296,2</td><td>301,0</td><td>303,8</td><td>304,8</td><td>1,33</td></tr>
<tr><th>450</th><td>67,6</td><td>101,0</td><td>133,9</td><td>166,2</td><td>197,7</td><td>228,2</td><td>257,7</td><td>285,9</td><td>312,8</td><td>338,1</td><td>361,8</td><td>383,8</td><td>404,0</td><td>422,1</td><td>438,2</td><td>452,2</td><td>464,0</td><td>473,5</td><td>480,7</td><td>1,50</td></tr>
<tr><th>500</th><td>92,8</td><td>138,7</td><td>184,1</td><td>228,8</td><td>272,6</td><td>315,4</td><td>356,9</td><td>397,1</td><td>435,7</td><td>472,7</td><td>507,8</td><td>540,9</td><td>571,9</td><td>600,7</td><td>627,1</td><td>651,1</td><td>672,5</td><td>691,2</td><td>707,2</td><td>1,67</td></tr>
<tr><th>550</th><td>123,5</td><td>184,8</td><td>245,5</td><td>305,4</td><td>364,3</td><td>422,1</td><td>478,5</td><td>533,4</td><td>586,6</td><td>638,0</td><td>687,3</td><td>734,3</td><td>779,0</td><td>821,2</td><td>860,8</td><td>897,5</td><td>931,4</td><td>962,2</td><td>989,9</td><td>1,83</td></tr>
<tr><th>600</th><td>160,4</td><td>240,1</td><td>319,1</td><td>397,3</td><td>474,5</td><td>550,3</td><td>624,7</td><td>697,4</td><td>768,3</td><td>837,0</td><td>903,6</td><td>967,7</td><td>1029,1</td><td>1087,8</td><td>1143,6</td><td>1196,2</td><td>1245,6</td><td>1291,6</td><td>1334,1</td><td>2,00</td></tr>
<tr><th>650</th><td>204,0</td><td>305,5</td><td>406,2</td><td>506,0</td><td>604,7</td><td>702,0</td><td>797,6</td><td>891,5</td><td>983,3</td><td>1072,9</td><td>1160,1</td><td>1244,5</td><td>1326,1</td><td>1404,7</td><td>1480,0</td><td>1552,0</td><td>1620,3</td><td>1684,9</td><td>1745,5</td><td>2,17</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 180, registre 2.3.4, p. 34)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 180 du PDF
(page imprimée 34)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
