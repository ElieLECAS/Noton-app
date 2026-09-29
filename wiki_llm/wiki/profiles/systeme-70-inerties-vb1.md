---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*B1
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*B1, pression de 400 Pa, flèche admissible 1/200, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, vb1]
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
    pages: 171
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 176
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*B1 (400 Pa, flèche 1/200)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*B1 : pression de 400 Pa (0,4 kN/m²), flèche admissible 1/200**
de la portée [1 p. 171].

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
<tr><th>100</th><td>0,1</td><td>0,1</td><td>0,2</td><td>0,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,50</td></tr>
<tr><th>110</th><td>0,1</td><td>0,2</td><td>0,2</td><td>0,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,55</td></tr>
<tr><th>120</th><td>0,2</td><td>0,2</td><td>0,3</td><td>0,3</td><td>0,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,60</td></tr>
<tr><th>130</th><td>0,2</td><td>0,3</td><td>0,4</td><td>0,4</td><td>0,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,65</td></tr>
<tr><th>140</th><td>0,3</td><td>0,4</td><td>0,5</td><td>0,5</td><td>0,6</td><td>0,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,70</td></tr>
<tr><th>150</th><td>0,3</td><td>0,5</td><td>0,6</td><td>0,7</td><td>0,8</td><td>0,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,75</td></tr>
<tr><th>160</th><td>0,4</td><td>0,6</td><td>0,7</td><td>0,9</td><td>1,0</td><td>1,0</td><td>1,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>170</th><td>0,5</td><td>0,7</td><td>0,9</td><td>1,1</td><td>1,2</td><td>1,3</td><td>1,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,85</td></tr>
<tr><th>180</th><td>0,6</td><td>0,8</td><td>1,1</td><td>1,3</td><td>1,4</td><td>1,6</td><td>1,6</td><td>1,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,90</td></tr>
<tr><th>190</th><td>0,7</td><td>1,0</td><td>1,3</td><td>1,5</td><td>1,7</td><td>1,9</td><td>2,0</td><td>2,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,95</td></tr>
<tr><th>200</th><td>0,8</td><td>1,1</td><td>1,5</td><td>1,8</td><td>2,1</td><td>2,3</td><td>2,4</td><td>2,5</td><td>2,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>210</th><td>0,9</td><td>1,3</td><td>1,7</td><td>2,1</td><td>2,4</td><td>2,7</td><td>2,9</td><td>3,0</td><td>3,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,05</td></tr>
<tr><th>220</th><td>1,0</td><td>1,5</td><td>2,0</td><td>2,4</td><td>2,8</td><td>3,1</td><td>3,4</td><td>3,6</td><td>3,7</td><td>3,7</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,10</td></tr>
<tr><th>230</th><td>1,2</td><td>1,8</td><td>2,3</td><td>2,8</td><td>3,2</td><td>3,6</td><td>3,9</td><td>4,2</td><td>4,3</td><td>4,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,15</td></tr>
<tr><th>240</th><td>1,4</td><td>2,0</td><td>2,6</td><td>3,2</td><td>3,7</td><td>4,2</td><td>4,6</td><td>4,9</td><td>5,1</td><td>5,2</td><td>5,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,20</td></tr>
<tr><th>250</th><td>1,5</td><td>2,3</td><td>3,0</td><td>3,6</td><td>4,2</td><td>4,8</td><td>5,2</td><td>5,6</td><td>5,9</td><td>6,1</td><td>6,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,25</td></tr>
<tr><th>260</th><td>1,7</td><td>2,6</td><td>3,4</td><td>4,1</td><td>4,8</td><td>5,4</td><td>6,0</td><td>6,4</td><td>6,8</td><td>7,0</td><td>7,2</td><td>7,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,30</td></tr>
<tr><th>270</th><td>1,9</td><td>2,9</td><td>3,8</td><td>4,6</td><td>5,4</td><td>6,1</td><td>6,8</td><td>7,3</td><td>7,7</td><td>8,1</td><td>8,3</td><td>8,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,35</td></tr>
<tr><th>280</th><td>2,2</td><td>3,2</td><td>4,2</td><td>5,2</td><td>6,1</td><td>6,9</td><td>7,6</td><td>8,2</td><td>8,8</td><td>9,2</td><td>9,5</td><td>9,7</td><td>9,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,40</td></tr>
<tr><th>290</th><td>2,4</td><td>3,6</td><td>4,7</td><td>5,8</td><td>6,8</td><td>7,7</td><td>8,5</td><td>9,3</td><td>9,9</td><td>10,4</td><td>10,8</td><td>11,1</td><td>11,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,45</td></tr>
<tr><th>300</th><td>2,7</td><td>4,0</td><td>5,2</td><td>6,4</td><td>7,5</td><td>8,6</td><td>9,5</td><td>10,4</td><td>11,1</td><td>11,7</td><td>12,2</td><td>12,6</td><td>12,8</td><td>12,9</td><td></td><td></td><td></td><td></td><td></td><td>1,50</td></tr>
<tr><th>310</th><td>2,9</td><td>4,4</td><td>5,8</td><td>7,1</td><td>8,3</td><td>9,5</td><td>10,6</td><td>11,6</td><td>12,4</td><td>13,1</td><td>13,7</td><td>14,2</td><td>14,5</td><td>14,6</td><td></td><td></td><td></td><td></td><td></td><td>1,55</td></tr>
<tr><th>320</th><td>3,2</td><td>4,8</td><td>6,3</td><td>7,8</td><td>9,2</td><td>10,5</td><td>11,7</td><td>12,8</td><td>13,8</td><td>14,7</td><td>15,4</td><td>15,9</td><td>16,3</td><td>16,6</td><td>16,6</td><td></td><td></td><td></td><td></td><td>1,60</td></tr>
<tr><th>330</th><td>3,5</td><td>5,3</td><td>7,0</td><td>8,6</td><td>10,1</td><td>11,6</td><td>13,0</td><td>14,2</td><td>15,3</td><td>16,3</td><td>17,1</td><td>17,8</td><td>18,3</td><td>18,6</td><td>18,8</td><td></td><td></td><td></td><td></td><td>1,65</td></tr>
<tr><th>340</th><td>3,9</td><td>5,8</td><td>7,6</td><td>9,4</td><td>11,1</td><td>12,7</td><td>14,2</td><td>15,6</td><td>16,9</td><td>18,0</td><td>19,0</td><td>19,8</td><td>20,4</td><td>20,8</td><td>21,1</td><td>21,2</td><td></td><td></td><td></td><td>1,70</td></tr>
<tr><th>350</th><td>4,2</td><td>6,3</td><td>8,3</td><td>10,3</td><td>12,2</td><td>13,9</td><td>15,6</td><td>17,2</td><td>18,6</td><td>19,8</td><td>20,9</td><td>21,9</td><td>22,6</td><td>23,2</td><td>23,6</td><td>23,8</td><td></td><td></td><td></td><td>1,75</td></tr>
<tr><th>360</th><td>4,6</td><td>6,9</td><td>9,1</td><td>11,2</td><td>13,3</td><td>15,2</td><td>17,1</td><td>18,8</td><td>20,4</td><td>21,8</td><td>23,1</td><td>24,1</td><td>25,0</td><td>25,7</td><td>26,3</td><td>26,6</td><td>26,7</td><td></td><td></td><td>1,80</td></tr>
<tr><th>370</th><td>5,0</td><td>7,5</td><td>9,9</td><td>12,2</td><td>14,4</td><td>16,6</td><td>18,6</td><td>20,5</td><td>22,3</td><td>23,9</td><td>25,3</td><td>26,5</td><td>27,6</td><td>28,4</td><td>29,1</td><td>29,5</td><td>29,7</td><td></td><td></td><td>1,85</td></tr>
<tr><th>380</th><td>5,4</td><td>8,1</td><td>10,7</td><td>13,2</td><td>15,7</td><td>18,0</td><td>20,3</td><td>22,3</td><td>24,3</td><td>26,1</td><td>27,7</td><td>29,1</td><td>30,3</td><td>31,3</td><td>32,1</td><td>32,6</td><td>33,0</td><td>33,1</td><td></td><td>1,90</td></tr>
<tr><th>390</th><td>5,9</td><td>8,7</td><td>11,6</td><td>14,3</td><td>17,0</td><td>19,5</td><td>22,0</td><td>24,3</td><td>26,4</td><td>28,4</td><td>30,2</td><td>31,8</td><td>33,1</td><td>34,3</td><td>35,3</td><td>36,0</td><td>36,5</td><td>36,7</td><td></td><td>1,95</td></tr>
<tr><th>400</th><td>6,3</td><td>9,4</td><td>12,5</td><td>15,5</td><td>18,4</td><td>21,1</td><td>23,8</td><td>26,3</td><td>28,7</td><td>30,8</td><td>32,8</td><td>34,6</td><td>36,2</td><td>37,5</td><td>38,6</td><td>39,5</td><td>40,1</td><td>40,5</td><td>40,6</td><td>2,00</td></tr>
<tr><th>450</th><td>9,0</td><td>13,5</td><td>17,9</td><td>22,2</td><td>26,4</td><td>30,4</td><td>34,4</td><td>38,1</td><td>41,7</td><td>45,1</td><td>48,2</td><td>51,2</td><td>53,9</td><td>56,3</td><td>58,4</td><td>60,3</td><td>61,9</td><td>63,1</td><td>64,1</td><td>2,25</td></tr>
<tr><th>500</th><td>12,4</td><td>18,5</td><td>24,5</td><td>30,5</td><td>36,4</td><td>42,1</td><td>47,6</td><td>52,9</td><td>58,1</td><td>63,0</td><td>67,7</td><td>72,1</td><td>76,3</td><td>80,1</td><td>83,6</td><td>86,8</td><td>89,7</td><td>92,2</td><td>94,3</td><td>2,50</td></tr>
<tr><th>550</th><td>16,5</td><td>24,6</td><td>32,7</td><td>40,7</td><td>48,6</td><td>56,3</td><td>63,8</td><td>71,1</td><td>78,2</td><td>85,1</td><td>91,6</td><td>97,9</td><td>103,9</td><td>109,5</td><td>114,8</td><td>119,7</td><td>124,2</td><td>128,3</td><td>132,0</td><td>2,75</td></tr>
<tr><th>600</th><td>21,4</td><td>32,0</td><td>42,6</td><td>53,0</td><td>63,3</td><td>73,4</td><td>83,3</td><td>93,0</td><td>102,4</td><td>111,6</td><td>120,5</td><td>129,0</td><td>137,2</td><td>145,0</td><td>152,5</td><td>159,5</td><td>166,1</td><td>172,2</td><td>177,9</td><td>3,00</td></tr>
<tr><th>650</th><td>27,2</td><td>40,7</td><td>54,2</td><td>67,5</td><td>80,6</td><td>93,6</td><td>106,4</td><td>118,9</td><td>131,1</td><td>143,1</td><td>154,7</td><td>165,9</td><td>176,8</td><td>187,3</td><td>197,3</td><td>206,9</td><td>216,0</td><td>224,6</td><td>232,7</td><td>3,25</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 171, registre 2.3.4, p. 25)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Même table dans les plans e.VOLUTION de 2008

Le registre 4.2 « Statique » du classeur e.VOLUTION d'août 2008 (système F 91) imprime la même
table, « Table des moments d'inerties Iz en cm4 », classement V\*B1, 400 Pa, flèche 1/200,
portées 100 à 650 cm, largeurs de charge 20 à 200 cm et flèche calculée : relue case par case à
400 dpi contre la table ci-dessus, elle porte les mêmes valeurs, les mêmes cases vides et les
mêmes flèches calculées. Son en-tête de page écrit « Tableau des moments d'inerties Iz » [2 p. 176].

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 176)

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 171 du PDF
(page imprimée 25)

[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 —
`raw/profine-plans-profiles-e-volution-2008-08.pdf`, registre 4.2 « Statique », PDF p. 176 (page imprimée 39)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
