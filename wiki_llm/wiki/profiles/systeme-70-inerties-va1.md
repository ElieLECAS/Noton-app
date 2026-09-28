---
type: Profilé
title: Moments d'inertie requis du système 70, classement V*A1
description: Table des moments d'inertie Iw requis (cm⁴) d'un meneau ou d'une traverse du système 70 Plateforme pour le classement V*A1, pression de 400 Pa, flèche admissible 1/150, portée de 100 à 650 cm et largeur de charge de 20 à 200 cm.
tags: [profine, systeme-70, statique, inertie, iw, iz, meneau, traverse, vent, fleche, va1]
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
    pages: 166
generated:
  by: process:claude-code
  at: 2026-09-28T18:00:00Z
---

# Moments d'inertie requis, classement V\*A1 (400 Pa, flèche 1/150)

La table donne le moment d'inertie Iw minimal (en cm⁴) que doit offrir le renfort d'un
meneau ou d'une traverse du système 70 Plateforme — le meneau est le montant fixe qui sépare
deux vantaux, la traverse l'élément horizontal ([glossaire](/reference/glossaire.md)) — pour le
classement au vent **V\*A1 : pression de 400 Pa (0,4 kN/m²), flèche admissible 1/150**
de la portée [1 p. 166].

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
<tr><th>100</th><td>0,1</td><td>0,1</td><td>0,1</td><td>0,1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,67</td></tr>
<tr><th>110</th><td>0,1</td><td>0,1</td><td>0,2</td><td>0,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,73</td></tr>
<tr><th>120</th><td>0,1</td><td>0,2</td><td>0,2</td><td>0,2</td><td>0,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,80</td></tr>
<tr><th>130</th><td>0,2</td><td>0,2</td><td>0,3</td><td>0,3</td><td>0,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,87</td></tr>
<tr><th>140</th><td>0,2</td><td>0,3</td><td>0,4</td><td>0,4</td><td>0,4</td><td>0,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0,93</td></tr>
<tr><th>150</th><td>0,2</td><td>0,4</td><td>0,4</td><td>0,5</td><td>0,6</td><td>0,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,00</td></tr>
<tr><th>160</th><td>0,3</td><td>0,4</td><td>0,6</td><td>0,6</td><td>0,7</td><td>0,8</td><td>0,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,07</td></tr>
<tr><th>170</th><td>0,4</td><td>0,5</td><td>0,7</td><td>0,8</td><td>0,9</td><td>1,0</td><td>1,0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,13</td></tr>
<tr><th>180</th><td>0,4</td><td>0,6</td><td>0,8</td><td>1,0</td><td>1,1</td><td>1,2</td><td>1,2</td><td>1,2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,20</td></tr>
<tr><th>190</th><td>0,5</td><td>0,7</td><td>0,9</td><td>1,1</td><td>1,3</td><td>1,4</td><td>1,5</td><td>1,5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,27</td></tr>
<tr><th>200</th><td>0,6</td><td>0,9</td><td>1,1</td><td>1,3</td><td>1,5</td><td>1,7</td><td>1,8</td><td>1,9</td><td>1,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,33</td></tr>
<tr><th>210</th><td>0,7</td><td>1,0</td><td>1,3</td><td>1,6</td><td>1,8</td><td>2,0</td><td>2,2</td><td>2,3</td><td>2,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,40</td></tr>
<tr><th>220</th><td>0,8</td><td>1,2</td><td>1,5</td><td>1,8</td><td>2,1</td><td>2,3</td><td>2,5</td><td>2,7</td><td>2,8</td><td>2,8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,47</td></tr>
<tr><th>230</th><td>0,9</td><td>1,3</td><td>1,7</td><td>2,1</td><td>2,4</td><td>2,7</td><td>3,0</td><td>3,1</td><td>3,3</td><td>3,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,53</td></tr>
<tr><th>240</th><td>1,0</td><td>1,5</td><td>2,0</td><td>2,4</td><td>2,8</td><td>3,1</td><td>3,4</td><td>3,6</td><td>3,8</td><td>3,9</td><td>3,9</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,60</td></tr>
<tr><th>250</th><td>1,2</td><td>1,7</td><td>2,2</td><td>2,7</td><td>3,2</td><td>3,6</td><td>3,9</td><td>4,2</td><td>4,4</td><td>4,6</td><td>4,6</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,67</td></tr>
<tr><th>260</th><td>1,3</td><td>1,9</td><td>2,5</td><td>3,1</td><td>3,6</td><td>4,1</td><td>4,5</td><td>4,8</td><td>5,1</td><td>5,3</td><td>5,4</td><td>5,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,73</td></tr>
<tr><th>270</th><td>1,5</td><td>2,2</td><td>2,8</td><td>3,5</td><td>4,1</td><td>4,6</td><td>5,1</td><td>5,5</td><td>5,8</td><td>6,1</td><td>6,2</td><td>6,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,80</td></tr>
<tr><th>280</th><td>1,6</td><td>2,4</td><td>3,2</td><td>3,9</td><td>4,5</td><td>5,2</td><td>5,7</td><td>6,2</td><td>6,6</td><td>6,9</td><td>7,1</td><td>7,3</td><td>7,3</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,87</td></tr>
<tr><th>290</th><td>1,8</td><td>2,7</td><td>3,5</td><td>4,3</td><td>5,1</td><td>5,8</td><td>6,4</td><td>7,0</td><td>7,4</td><td>7,8</td><td>8,1</td><td>8,3</td><td>8,4</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1,93</td></tr>
<tr><th>300</th><td>2,0</td><td>3,0</td><td>3,9</td><td>4,8</td><td>5,6</td><td>6,4</td><td>7,1</td><td>7,8</td><td>8,3</td><td>8,8</td><td>9,2</td><td>9,4</td><td>9,6</td><td>9,6</td><td></td><td></td><td></td><td></td><td></td><td>2,00</td></tr>
<tr><th>310</th><td>2,2</td><td>3,3</td><td>4,3</td><td>5,3</td><td>6,3</td><td>7,1</td><td>7,9</td><td>8,7</td><td>9,3</td><td>9,9</td><td>10,3</td><td>10,6</td><td>10,9</td><td>11,0</td><td></td><td></td><td></td><td></td><td></td><td>2,07</td></tr>
<tr><th>320</th><td>2,4</td><td>3,6</td><td>4,8</td><td>5,9</td><td>6,9</td><td>7,9</td><td>8,8</td><td>9,6</td><td>10,4</td><td>11,0</td><td>11,5</td><td>11,9</td><td>12,2</td><td>12,4</td><td>12,5</td><td></td><td></td><td></td><td></td><td>2,13</td></tr>
<tr><th>330</th><td>2,7</td><td>4,0</td><td>5,2</td><td>6,4</td><td>7,6</td><td>8,7</td><td>9,7</td><td>10,6</td><td>11,5</td><td>12,2</td><td>12,8</td><td>13,3</td><td>13,7</td><td>14,0</td><td>14,1</td><td></td><td></td><td></td><td></td><td>2,20</td></tr>
<tr><th>340</th><td>2,9</td><td>4,3</td><td>5,7</td><td>7,1</td><td>8,3</td><td>9,6</td><td>10,7</td><td>11,7</td><td>12,7</td><td>13,5</td><td>14,2</td><td>14,8</td><td>15,3</td><td>15,6</td><td>15,8</td><td>15,9</td><td></td><td></td><td></td><td>2,27</td></tr>
<tr><th>350</th><td>3,2</td><td>4,7</td><td>6,2</td><td>7,7</td><td>9,1</td><td>10,5</td><td>11,7</td><td>12,9</td><td>13,9</td><td>14,9</td><td>15,7</td><td>16,4</td><td>17,0</td><td>17,4</td><td>17,7</td><td>17,8</td><td></td><td></td><td></td><td>2,33</td></tr>
<tr><th>360</th><td>3,5</td><td>5,1</td><td>6,8</td><td>8,4</td><td>10,0</td><td>11,4</td><td>12,8</td><td>14,1</td><td>15,3</td><td>16,3</td><td>17,3</td><td>18,1</td><td>18,8</td><td>19,3</td><td>19,7</td><td>19,9</td><td>20,0</td><td></td><td></td><td>2,40</td></tr>
<tr><th>370</th><td>3,8</td><td>5,6</td><td>7,4</td><td>9,1</td><td>10,8</td><td>12,4</td><td>14,0</td><td>15,4</td><td>16,7</td><td>17,9</td><td>19,0</td><td>19,9</td><td>20,7</td><td>21,3</td><td>21,8</td><td>22,1</td><td>22,3</td><td></td><td></td><td>2,47</td></tr>
<tr><th>380</th><td>4,1</td><td>6,1</td><td>8,0</td><td>9,9</td><td>11,8</td><td>13,5</td><td>15,2</td><td>16,8</td><td>18,2</td><td>19,5</td><td>20,7</td><td>21,8</td><td>22,7</td><td>23,5</td><td>24,1</td><td>24,5</td><td>24,7</td><td>24,8</td><td></td><td>2,53</td></tr>
<tr><th>390</th><td>4,4</td><td>6,6</td><td>8,7</td><td>10,7</td><td>12,7</td><td>14,7</td><td>16,5</td><td>18,2</td><td>19,8</td><td>21,3</td><td>22,6</td><td>23,8</td><td>24,9</td><td>25,7</td><td>26,4</td><td>27,0</td><td>27,3</td><td>27,5</td><td></td><td>2,60</td></tr>
<tr><th>400</th><td>4,7</td><td>7,1</td><td>9,4</td><td>11,6</td><td>13,8</td><td>15,9</td><td>17,8</td><td>19,7</td><td>21,5</td><td>23,1</td><td>24,6</td><td>25,9</td><td>27,1</td><td>28,1</td><td>29,0</td><td>29,6</td><td>30,1</td><td>30,4</td><td>30,5</td><td>2,67</td></tr>
<tr><th>450</th><td>6,8</td><td>10,1</td><td>13,4</td><td>16,6</td><td>19,8</td><td>22,8</td><td>25,8</td><td>28,6</td><td>31,3</td><td>33,8</td><td>36,2</td><td>38,4</td><td>40,4</td><td>42,2</td><td>43,8</td><td>45,2</td><td>46,4</td><td>47,3</td><td>48,1</td><td>3,00</td></tr>
<tr><th>500</th><td>9,3</td><td>13,9</td><td>18,4</td><td>22,9</td><td>27,3</td><td>31,5</td><td>35,7</td><td>39,7</td><td>43,6</td><td>47,3</td><td>50,8</td><td>54,1</td><td>57,2</td><td>60,1</td><td>62,7</td><td>65,1</td><td>67,2</td><td>69,1</td><td>70,7</td><td>3,33</td></tr>
<tr><th>550</th><td>12,4</td><td>18,5</td><td>24,5</td><td>30,5</td><td>36,4</td><td>42,2</td><td>47,9</td><td>53,3</td><td>58,7</td><td>63,8</td><td>68,7</td><td>73,4</td><td>77,9</td><td>82,1</td><td>86,1</td><td>89,8</td><td>93,1</td><td>96,2</td><td>99,0</td><td>3,67</td></tr>
<tr><th>600</th><td>16,0</td><td>24,0</td><td>31,9</td><td>39,7</td><td>47,4</td><td>55,0</td><td>62,5</td><td>69,7</td><td>76,8</td><td>83,7</td><td>90,4</td><td>96,8</td><td>102,9</td><td>108,8</td><td>114,4</td><td>119,6</td><td>124,6</td><td>129,2</td><td>133,4</td><td>4,00</td></tr>
<tr><th>650</th><td>20,4</td><td>30,5</td><td>40,6</td><td>50,6</td><td>60,5</td><td>70,2</td><td>79,8</td><td>89,2</td><td>98,3</td><td>107,3</td><td>116,0</td><td>124,5</td><td>132,6</td><td>140,5</td><td>148,0</td><td>155,2</td><td>162,0</td><td>168,5</td><td>174,6</td><td>4,33</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 166, registre 2.3.4, p. 20)

Valeurs relevées ligne par ligne sur la page rendue à 400 dpi ; elles concordent toutes, au
dixième près, avec la formule de la charge trapézoïdale de la section 2.3.4.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 166 du PDF
(page imprimée 20)

# Voir aussi

- [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
