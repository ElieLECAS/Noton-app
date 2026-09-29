---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*A3
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*A3, pression de 1200 Pa, flèche admissible 1/150, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, va3]
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
    pages: 168
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 173
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*A3 (1200 Pa, flèche 1/150)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*A3 : pression de 1200 Pa (1,2 kN/m²), flèche admissible 1/150**
de la portée [1 p. 168].

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
<tr><th>100</th><td>0,2</td><td>0,3</td><td>0,3</td><td>0,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,67</td></tr>
<tr><th>110</th><td>0,3</td><td>0,4</td><td>0,5</td><td>0,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,73</td></tr>
<tr><th>120</th><td>0,4</td><td>0,5</td><td>0,6</td><td>0,7</td><td>0,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>130</th><td>0,5</td><td>0,7</td><td>0,8</td><td>1,0</td><td>1,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,87</td></tr>
<tr><th>140</th><td>0,6</td><td>0,9</td><td>1,1</td><td>1,2</td><td>1,3</td><td>1,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,93</td></tr>
<tr><th>150</th><td>0,7</td><td>1,1</td><td>1,3</td><td>1,6</td><td>1,7</td><td>1,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>160</th><td>0,9</td><td>1,3</td><td>1,7</td><td>1,9</td><td>2,2</td><td>2,3</td><td>2,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,07</td></tr>
<tr><th>170</th><td>1,1</td><td>1,6</td><td>2,0</td><td>2,4</td><td>2,7</td><td>2,9</td><td>3,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,13</td></tr>
<tr><th>180</th><td>1,3</td><td>1,9</td><td>2,4</td><td>2,9</td><td>3,2</td><td>3,5</td><td>3,7</td><td>3,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,20</td></tr>
<tr><th>190</th><td>1,5</td><td>2,2</td><td>2,8</td><td>3,4</td><td>3,9</td><td>4,3</td><td>4,5</td><td>4,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,27</td></tr>
<tr><th>200</th><td>1,8</td><td>2,6</td><td>3,3</td><td>4,0</td><td>4,6</td><td>5,1</td><td>5,4</td><td>5,6</td><td>5,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,33</td></tr>
<tr><th>210</th><td>2,0</td><td>3,0</td><td>3,9</td><td>4,7</td><td>5,4</td><td>6,0</td><td>6,5</td><td>6,8</td><td>6,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,40</td></tr>
<tr><th>220</th><td>2,3</td><td>3,5</td><td>4,5</td><td>5,5</td><td>6,3</td><td>7,0</td><td>7,6</td><td>8,0</td><td>8,3</td><td>8,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,47</td></tr>
<tr><th>230</th><td>2,7</td><td>4,0</td><td>5,2</td><td>6,3</td><td>7,3</td><td>8,1</td><td>8,9</td><td>9,4</td><td>9,8</td><td>10,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,53</td></tr>
<tr><th>240</th><td>3,1</td><td>4,5</td><td>5,9</td><td>7,2</td><td>8,4</td><td>9,4</td><td>10,2</td><td>10,9</td><td>11,4</td><td>11,7</td><td>11,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,60</td></tr>
<tr><th>250</th><td>3,5</td><td>5,1</td><td>6,7</td><td>8,2</td><td>9,5</td><td>10,7</td><td>11,8</td><td>12,6</td><td>13,3</td><td>13,7</td><td>13,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,67</td></tr>
<tr><th>260</th><td>3,9</td><td>5,8</td><td>7,6</td><td>9,2</td><td>10,8</td><td>12,2</td><td>13,4</td><td>14,4</td><td>15,2</td><td>15,8</td><td>16,2</td><td>16,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,73</td></tr>
<tr><th>270</th><td>4,4</td><td>6,5</td><td>8,5</td><td>10,4</td><td>12,2</td><td>13,8</td><td>15,2</td><td>16,4</td><td>17,4</td><td>18,2</td><td>18,7</td><td>18,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,80</td></tr>
<tr><th>280</th><td>4,9</td><td>7,2</td><td>9,5</td><td>11,6</td><td>13,6</td><td>15,5</td><td>17,1</td><td>18,6</td><td>19,8</td><td>20,7</td><td>21,4</td><td>21,8</td><td>22,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,87</td></tr>
<tr><th>290</th><td>5,4</td><td>8,0</td><td>10,6</td><td>13,0</td><td>15,2</td><td>17,3</td><td>19,2</td><td>20,9</td><td>22,3</td><td>23,4</td><td>24,3</td><td>24,9</td><td>25,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,93</td></tr>
<tr><th>300</th><td>6,0</td><td>8,9</td><td>11,7</td><td>14,4</td><td>16,9</td><td>19,3</td><td>21,4</td><td>23,4</td><td>25,0</td><td>26,4</td><td>27,5</td><td>28,3</td><td>28,8</td><td>28,9</td><td></td><td></td><td></td><td></td><td></td><td>2,00</td></tr>
<tr><th>310</th><td>6,6</td><td>9,8</td><td>12,9</td><td>15,9</td><td>18,8</td><td>21,4</td><td>23,8</td><td>26,0</td><td>27,9</td><td>29,6</td><td>30,9</td><td>31,9</td><td>32,6</td><td>32,9</td><td></td><td></td><td></td><td></td><td></td><td>2,07</td></tr>
<tr><th>320</th><td>7,3</td><td>10,8</td><td>14,3</td><td>17,6</td><td>20,7</td><td>23,7</td><td>26,4</td><td>28,9</td><td>31,1</td><td>33,0</td><td>34,6</td><td>35,8</td><td>36,7</td><td>37,3</td><td>37,4</td><td></td><td></td><td></td><td></td><td>2,13</td></tr>
<tr><th>330</th><td>8,0</td><td>11,9</td><td>15,7</td><td>19,3</td><td>22,8</td><td>26,1</td><td>29,1</td><td>31,9</td><td>34,4</td><td>36,6</td><td>38,5</td><td>40,0</td><td>41,1</td><td>41,9</td><td>42,3</td><td></td><td></td><td></td><td></td><td>2,20</td></tr>
<tr><th>340</th><td>8,7</td><td>13,0</td><td>17,2</td><td>21,2</td><td>25,0</td><td>28,7</td><td>32,1</td><td>35,2</td><td>38,0</td><td>40,5</td><td>42,7</td><td>44,5</td><td>45,9</td><td>46,9</td><td>47,5</td><td>47,7</td><td></td><td></td><td></td><td>2,27</td></tr>
<tr><th>350</th><td>9,5</td><td>14,2</td><td>18,7</td><td>23,2</td><td>27,4</td><td>31,4</td><td>35,1</td><td>38,6</td><td>41,8</td><td>44,6</td><td>47,1</td><td>49,2</td><td>50,9</td><td>52,2</td><td>53,1</td><td>53,5</td><td></td><td></td><td></td><td>2,33</td></tr>
<tr><th>360</th><td>10,4</td><td>15,4</td><td>20,4</td><td>25,2</td><td>29,9</td><td>34,3</td><td>38,4</td><td>42,3</td><td>45,8</td><td>49,0</td><td>51,9</td><td>54,3</td><td>56,3</td><td>57,9</td><td>59,1</td><td>59,8</td><td>60,0</td><td></td><td></td><td>2,40</td></tr>
<tr><th>370</th><td>11,3</td><td>16,8</td><td>22,2</td><td>27,4</td><td>32,5</td><td>37,3</td><td>41,9</td><td>46,2</td><td>50,1</td><td>53,7</td><td>56,9</td><td>59,7</td><td>62,1</td><td>64,0</td><td>65,4</td><td>66,4</td><td>66,9</td><td></td><td></td><td>2,47</td></tr>
<tr><th>380</th><td>12,2</td><td>18,2</td><td>24,1</td><td>29,8</td><td>35,3</td><td>40,6</td><td>45,6</td><td>50,3</td><td>54,6</td><td>58,6</td><td>62,2</td><td>65,4</td><td>68,1</td><td>70,4</td><td>72,2</td><td>73,4</td><td>74,2</td><td>74,5</td><td></td><td>2,53</td></tr>
<tr><th>390</th><td>13,2</td><td>19,7</td><td>26,0</td><td>32,2</td><td>38,2</td><td>44,0</td><td>49,5</td><td>54,6</td><td>59,4</td><td>63,9</td><td>67,9</td><td>71,4</td><td>74,6</td><td>77,2</td><td>79,3</td><td>80,9</td><td>82,0</td><td>82,6</td><td></td><td>2,60</td></tr>
<tr><th>400</th><td>14,2</td><td>21,2</td><td>28,1</td><td>34,8</td><td>41,3</td><td>47,6</td><td>53,5</td><td>59,2</td><td>64,5</td><td>69,4</td><td>73,8</td><td>77,8</td><td>81,4</td><td>84,4</td><td>86,9</td><td>88,9</td><td>90,3</td><td>91,1</td><td>91,4</td><td>2,67</td></tr>
<tr><th>450</th><td>20,3</td><td>30,3</td><td>40,2</td><td>49,9</td><td>59,3</td><td>68,5</td><td>77,3</td><td>85,8</td><td>93,8</td><td>101,4</td><td>108,6</td><td>115,1</td><td>121,2</td><td>126,6</td><td>131,5</td><td>135,7</td><td>139,2</td><td>142,0</td><td>144,2</td><td>3,00</td></tr>
<tr><th>500</th><td>27,8</td><td>41,6</td><td>55,2</td><td>68,6</td><td>81,8</td><td>94,6</td><td>107,1</td><td>119,1</td><td>130,7</td><td>141,8</td><td>152,3</td><td>162,3</td><td>171,6</td><td>180,2</td><td>188,1</td><td>195,3</td><td>201,7</td><td>207,4</td><td>212,2</td><td>3,33</td></tr>
<tr><th>550</th><td>37,1</td><td>55,4</td><td>73,6</td><td>91,6</td><td>109,3</td><td>126,6</td><td>143,6</td><td>160,0</td><td>176,0</td><td>191,4</td><td>206,2</td><td>220,3</td><td>233,7</td><td>246,4</td><td>258,2</td><td>269,3</td><td>279,4</td><td>288,7</td><td>297,0</td><td>3,67</td></tr>
<tr><th>600</th><td>48,1</td><td>72,0</td><td>95,7</td><td>119,2</td><td>142,3</td><td>165,1</td><td>187,4</td><td>209,2</td><td>230,5</td><td>251,1</td><td>271,1</td><td>290,3</td><td>308,7</td><td>326,4</td><td>343,1</td><td>358,9</td><td>373,7</td><td>387,5</td><td>400,2</td><td>4,00</td></tr>
<tr><th>650</th><td>61,2</td><td>91,6</td><td>121,9</td><td>151,8</td><td>181,4</td><td>210,6</td><td>239,3</td><td>267,5</td><td>295,0</td><td>321,9</td><td>348,0</td><td>373,4</td><td>397,8</td><td>421,4</td><td>444,0</td><td>465,6</td><td>486,1</td><td>505,5</td><td>523,7</td><td>4,33</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 168, registre 2.3.4, p. 22)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Même table dans les plans e.VOLUTION de 2008

Le registre 4.2 « Statique » du classeur e.VOLUTION d'août 2008 (système F 91) imprime la même
table, « Table des moments d'inerties Iz en cm4 », classement V\*A3, 1200 Pa, flèche 1/150,
portées 100 à 650 cm, largeurs de charge 20 à 200 cm et flèche calculée : relue case par case à
400 dpi contre la table ci-dessus, elle porte les mêmes valeurs, les mêmes cases vides et les
mêmes flèches calculées. Son en-tête de page écrit « Tableau des moments d'inerties Iz » [2 p. 173].

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 173)

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 168 du PDF
(page imprimée 22)

[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 —
`raw/profine-plans-profiles-e-volution-2008-08.pdf`, registre 4.2 « Statique », PDF p. 173 (page imprimée 36)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
