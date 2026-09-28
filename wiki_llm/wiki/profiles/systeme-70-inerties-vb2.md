---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*B2
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*B2, pression de 800 Pa, flèche admissible 1/200, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, vb2]
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
    pages: 172
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*B2 (800 Pa, flèche 1/200)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*B2 : pression de 800 Pa (0,8 kN/m²), flèche admissible 1/200**
de la portée [1 p. 172].

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
<tr><th>100</th><td>0,2</td><td>0,3</td><td>0,3</td><td>0,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,50</td></tr>
<tr><th>110</th><td>0,3</td><td>0,4</td><td>0,4</td><td>0,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,55</td></tr>
<tr><th>120</th><td>0,3</td><td>0,5</td><td>0,6</td><td>0,6</td><td>0,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,60</td></tr>
<tr><th>130</th><td>0,4</td><td>0,6</td><td>0,7</td><td>0,8</td><td>0,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,65</td></tr>
<tr><th>140</th><td>0,5</td><td>0,8</td><td>1,0</td><td>1,1</td><td>1,2</td><td>1,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,70</td></tr>
<tr><th>150</th><td>0,7</td><td>0,9</td><td>1,2</td><td>1,4</td><td>1,5</td><td>1,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,75</td></tr>
<tr><th>160</th><td>0,8</td><td>1,2</td><td>1,5</td><td>1,7</td><td>1,9</td><td>2,0</td><td>2,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>170</th><td>1,0</td><td>1,4</td><td>1,8</td><td>2,1</td><td>2,4</td><td>2,5</td><td>2,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,85</td></tr>
<tr><th>180</th><td>1,1</td><td>1,7</td><td>2,1</td><td>2,5</td><td>2,9</td><td>3,1</td><td>3,3</td><td>3,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,90</td></tr>
<tr><th>190</th><td>1,3</td><td>2,0</td><td>2,5</td><td>3,0</td><td>3,5</td><td>3,8</td><td>4,0</td><td>4,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,95</td></tr>
<tr><th>200</th><td>1,6</td><td>2,3</td><td>3,0</td><td>3,6</td><td>4,1</td><td>4,5</td><td>4,8</td><td>5,0</td><td>5,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>210</th><td>1,8</td><td>2,7</td><td>3,5</td><td>4,2</td><td>4,8</td><td>5,3</td><td>5,7</td><td>6,0</td><td>6,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,05</td></tr>
<tr><th>220</th><td>2,1</td><td>3,1</td><td>4,0</td><td>4,9</td><td>5,6</td><td>6,2</td><td>6,8</td><td>7,1</td><td>7,4</td><td>7,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,10</td></tr>
<tr><th>230</th><td>2,4</td><td>3,5</td><td>4,6</td><td>5,6</td><td>6,5</td><td>7,2</td><td>7,9</td><td>8,4</td><td>8,7</td><td>8,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,15</td></tr>
<tr><th>240</th><td>2,7</td><td>4,0</td><td>5,2</td><td>6,4</td><td>7,4</td><td>8,3</td><td>9,1</td><td>9,7</td><td>10,2</td><td>10,4</td><td>10,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,20</td></tr>
<tr><th>250</th><td>3,1</td><td>4,5</td><td>5,9</td><td>7,3</td><td>8,5</td><td>9,5</td><td>10,5</td><td>11,2</td><td>11,8</td><td>12,2</td><td>12,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,25</td></tr>
<tr><th>260</th><td>3,5</td><td>5,1</td><td>6,7</td><td>8,2</td><td>9,6</td><td>10,8</td><td>11,9</td><td>12,8</td><td>13,6</td><td>14,1</td><td>14,4</td><td>14,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,30</td></tr>
<tr><th>270</th><td>3,9</td><td>5,7</td><td>7,5</td><td>9,2</td><td>10,8</td><td>12,2</td><td>13,5</td><td>14,6</td><td>15,5</td><td>16,2</td><td>16,6</td><td>16,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,35</td></tr>
<tr><th>280</th><td>4,3</td><td>6,4</td><td>8,4</td><td>10,3</td><td>12,1</td><td>13,8</td><td>15,2</td><td>16,5</td><td>17,6</td><td>18,4</td><td>19,0</td><td>19,4</td><td>19,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,40</td></tr>
<tr><th>290</th><td>4,8</td><td>7,1</td><td>9,4</td><td>11,5</td><td>13,5</td><td>15,4</td><td>17,1</td><td>18,5</td><td>19,8</td><td>20,8</td><td>21,6</td><td>22,2</td><td>22,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,45</td></tr>
<tr><th>300</th><td>5,3</td><td>7,9</td><td>10,4</td><td>12,8</td><td>15,1</td><td>17,2</td><td>19,1</td><td>20,8</td><td>22,2</td><td>23,5</td><td>24,4</td><td>25,1</td><td>25,6</td><td>25,7</td><td></td><td></td><td></td><td></td><td></td><td>1,50</td></tr>
<tr><th>310</th><td>5,9</td><td>8,7</td><td>11,5</td><td>14,2</td><td>16,7</td><td>19,0</td><td>21,2</td><td>23,1</td><td>24,8</td><td>26,3</td><td>27,5</td><td>28,4</td><td>29,0</td><td>29,3</td><td></td><td></td><td></td><td></td><td></td><td>1,55</td></tr>
<tr><th>320</th><td>6,5</td><td>9,6</td><td>12,7</td><td>15,6</td><td>18,4</td><td>21,0</td><td>23,5</td><td>25,7</td><td>27,6</td><td>29,3</td><td>30,7</td><td>31,8</td><td>32,6</td><td>33,1</td><td>33,3</td><td></td><td></td><td></td><td></td><td>1,60</td></tr>
<tr><th>330</th><td>7,1</td><td>10,6</td><td>13,9</td><td>17,2</td><td>20,3</td><td>23,2</td><td>25,9</td><td>28,4</td><td>30,6</td><td>32,6</td><td>34,2</td><td>35,6</td><td>36,6</td><td>37,3</td><td>37,6</td><td></td><td></td><td></td><td></td><td>1,65</td></tr>
<tr><th>340</th><td>7,8</td><td>11,6</td><td>15,3</td><td>18,8</td><td>22,2</td><td>25,5</td><td>28,5</td><td>31,3</td><td>33,8</td><td>36,0</td><td>37,9</td><td>39,5</td><td>40,8</td><td>41,7</td><td>42,2</td><td>42,4</td><td></td><td></td><td></td><td>1,70</td></tr>
<tr><th>350</th><td>8,5</td><td>12,6</td><td>16,7</td><td>20,6</td><td>24,3</td><td>27,9</td><td>31,2</td><td>34,3</td><td>37,2</td><td>39,7</td><td>41,9</td><td>43,8</td><td>45,3</td><td>46,4</td><td>47,2</td><td>47,6</td><td></td><td></td><td></td><td>1,75</td></tr>
<tr><th>360</th><td>9,2</td><td>13,7</td><td>18,2</td><td>22,4</td><td>26,6</td><td>30,5</td><td>34,2</td><td>37,6</td><td>40,7</td><td>43,6</td><td>46,1</td><td>48,3</td><td>50,1</td><td>51,5</td><td>52,5</td><td>53,1</td><td>53,3</td><td></td><td></td><td>1,80</td></tr>
<tr><th>370</th><td>10,0</td><td>14,9</td><td>19,7</td><td>24,4</td><td>28,9</td><td>33,2</td><td>37,3</td><td>41,0</td><td>44,5</td><td>47,7</td><td>50,6</td><td>53,1</td><td>55,2</td><td>56,9</td><td>58,1</td><td>59,0</td><td>59,4</td><td></td><td></td><td>1,85</td></tr>
<tr><th>380</th><td>10,8</td><td>16,2</td><td>21,4</td><td>26,5</td><td>31,4</td><td>36,1</td><td>40,5</td><td>44,7</td><td>48,6</td><td>52,1</td><td>55,3</td><td>58,1</td><td>60,6</td><td>62,6</td><td>64,1</td><td>65,3</td><td>66,0</td><td>66,2</td><td></td><td>1,90</td></tr>
<tr><th>390</th><td>11,7</td><td>17,5</td><td>23,1</td><td>28,7</td><td>34,0</td><td>39,1</td><td>44,0</td><td>48,5</td><td>52,8</td><td>56,8</td><td>60,3</td><td>63,5</td><td>66,3</td><td>68,6</td><td>70,5</td><td>71,9</td><td>72,9</td><td>73,4</td><td></td><td>1,95</td></tr>
<tr><th>400</th><td>12,6</td><td>18,9</td><td>25,0</td><td>31,0</td><td>36,7</td><td>42,3</td><td>47,6</td><td>52,6</td><td>57,3</td><td>61,6</td><td>65,6</td><td>69,2</td><td>72,3</td><td>75,0</td><td>77,2</td><td>79,0</td><td>80,3</td><td>81,0</td><td>81,3</td><td>2,00</td></tr>
<tr><th>450</th><td>18,0</td><td>26,9</td><td>35,7</td><td>44,3</td><td>52,7</td><td>60,9</td><td>68,7</td><td>76,2</td><td>83,4</td><td>90,2</td><td>96,5</td><td>102,4</td><td>107,7</td><td>112,6</td><td>116,9</td><td>120,6</td><td>123,7</td><td>126,3</td><td>128,2</td><td>2,25</td></tr>
<tr><th>500</th><td>24,7</td><td>37,0</td><td>49,1</td><td>61,0</td><td>72,7</td><td>84,1</td><td>95,2</td><td>105,9</td><td>116,2</td><td>126,0</td><td>135,4</td><td>144,2</td><td>152,5</td><td>160,2</td><td>167,2</td><td>173,6</td><td>179,3</td><td>184,3</td><td>188,6</td><td>2,50</td></tr>
<tr><th>550</th><td>32,9</td><td>49,3</td><td>65,5</td><td>81,4</td><td>97,2</td><td>112,6</td><td>127,6</td><td>142,3</td><td>156,4</td><td>170,1</td><td>183,3</td><td>195,8</td><td>207,7</td><td>219,0</td><td>229,5</td><td>239,3</td><td>248,4</td><td>256,6</td><td>264,0</td><td>2,75</td></tr>
<tr><th>600</th><td>42,8</td><td>64,0</td><td>85,1</td><td>106,0</td><td>126,5</td><td>146,8</td><td>166,6</td><td>186,0</td><td>204,9</td><td>223,2</td><td>240,9</td><td>258,0</td><td>274,4</td><td>290,1</td><td>305,0</td><td>319,0</td><td>332,2</td><td>344,4</td><td>355,8</td><td>3,00</td></tr>
<tr><th>650</th><td>54,4</td><td>81,5</td><td>108,3</td><td>134,9</td><td>161,2</td><td>187,2</td><td>212,7</td><td>237,7</td><td>262,2</td><td>286,1</td><td>309,3</td><td>331,9</td><td>353,6</td><td>374,6</td><td>394,7</td><td>413,9</td><td>432,1</td><td>449,3</td><td>465,5</td><td>3,25</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 172, registre 2.3.4, p. 26)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 172 du PDF
(page imprimée 26)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
