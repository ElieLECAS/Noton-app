---
type: Profilé
title: Cotes de débit du système 70
description: Les cotes à déduire de la dimension hors tout pour débiter les dormants, meneaux et traverses, ouvrants et battements du système 70 Plateforme de profine, avec l'exemple de calcul DEO = X / Y – (a + b) et le débit des battements extérieurs et intérieurs.
tags: [profine, systeme-70, cote-de-debit, coupe, dormant, ouvrant, meneau, traverse, battement, renfort, atelier]
systeme: 70
fournisseur: KÖMMERLING
usage: atelier
famille: cotes-de-debit
status: draft
sources:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
source_pages:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    pages: 104-119
generated:
  by: process:claude-code
  at: 2026-09-28T12:00:00Z
---

# Ce que donnent ces tableaux

Les cotes de débit du système 70 Plateforme de [profine](/fournisseurs/profine.md) (gammes
e.VOLUTION, e.MOTION et e.XCLUSIVE) sont des cotes à déduire, pas des longueurs à couper. On
part de la dimension hors tout de l'élément et on retranche, coupe par coupe, la valeur du profilé
concerné. « Pour déterminer les cotes à déduire, il faut se reporter aux valeurs indiquées dans
les tableaux des pages suivantes. Ces valeurs sont à prendre sur les différentes coupes
représentées pour chaque cas. » [1 p. 104]

Sur chaque planche, la coupe cotée porte des repères numérotés ①, ②, ③, ④ sous le dessin, et
chaque ligne du tableau commence par le repère de la cote qu'elle donne ; une ligne sans repère
(les renforts) n'a pas de cote dessinée. Les dessins sont « non à l'échelle ». En tête de chaque
tableau : « Les dimensions indiquées se réfèrent uniquement à une seule coupe » — un dormant a deux
montants, une largeur hors tout perd la valeur du montant gauche *et* celle du montant droit.

Quatre dimensions se déduisent en cascade, avec les sigles du [glossaire](/reference/glossaire.md) :

| Sigle | Dimension |
| --- | --- |
| DHT | dimension hors tout, c'est-à-dire la dimension extérieure du dormant |
| DEO | dimension extérieure d'ouvrant |
| DFO | dimension de feuillure d'ouvrant (la feuillure est le logement du vitrage) |
| — | dimension de vitrage |

Le renfort est la barre d'acier glissée dans la chambre du profilé PVC pour le rigidifier ;
le meneau (vertical) et la traverse (horizontale) partagent un châssis ; le battement
est le profil qui couvre la jonction de deux ouvrants sans meneau. Les tableaux donnent deux débits
de renfort de meneau / traverse, selon que le meneau est assemblé au set d'assemblage mécanique
ou à l'équerre d'assemblage — voir
[Assemblages du système 70](/profiles/systeme-70-assemblages.md).

# L'exemple du manuel

« une fenêtre à deux vantaux avec meneau », « Dimension extérieure dormant = dimension hors tout
DHT = 2000 x 1200 mm (L x H) », coupée en B-B à mi-hauteur [1 p. 104] :

![Fenêtre deux vantaux de l'exemple, 2 000 × 1 200 mm, coupe B-B](/assets/procedures/moe-systeme-70/debit/debit-exemple-fenetre-deux-vantaux.png)

La coupe B-B montre, de gauche à droite, le dormant 6101, l'ouvrant 6112, le meneau
6127, l'ouvrant 6112 et le dormant 6101. La dimension hors tout est partagée à l'axe du
meneau en X et Y. Sous la coupe, a est la cote du dormant (du bord extérieur du dormant
au bord de l'ouvrant) et b celle du meneau (de l'axe au bord de l'ouvrant) ; au-dessus de a et
b, la dimension extérieure ouvrant, la dimension feuillure ouvrant et la dimension vitrage de
chaque vantail.

![Coupe B-B de l'exemple : dormants 6101, ouvrants 6112, meneau 6127](/assets/procedures/moe-systeme-70/debit/debit-exemple-coupe-b-b.png)

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 104)

« Cotes à soustraire » : 1) « Détermination des dimensions extérieures ouvrant (largeur) DEO »,
DEO = X / Y – (a + b) ; 2) « Détermination de la dimension du vitrage », **Dimension vitrage =
DEO – 2 x (56)**.

| Étape | Calcul imprimé | Résultat (mm) |
| --- | --- | --- |
| Demi-largeur au meneau | DHT = 2000 ; X = 1000 | 1 000 |
| Dimension extérieure d'ouvrant | DEO = 1000 – (36 + 12) | 952 |
| Dimension du vitrage | = 952 – 111 | 841 |

`a` = 36 est la cote DEO du dormant 6101 et `b` = 12 celle du meneau 6127, lues sur les
vignettes de la planche et dans les tableaux ci-dessous.

![Repères a = 36 (dormant 6101) et b = 12 (meneau 6127)](/assets/procedures/moe-systeme-70/debit/debit-exemple-reperes-a-b.png)

![Dimension vitrage = DEO – 2 x (56), vignette de l'ouvrant 6112 cotée 55,5](/assets/procedures/moe-systeme-70/debit/debit-exemple-dimension-vitrage.png)

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 104)

**La formule écrit 2 x (56), la vignette de l'ouvrant 6112 porte 55,5 et le calcul retranche 111
(2 × 55,5)** ; le tableau des ouvrants donne 56 pour le 6112. Avec 56, le vitrage de l'exemple
serait de 840 mm et non de 841 — entrée **INC-125** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

# Cotes de débit des dormants

Cotes à déduire de la dimension hors tout (DHT), en mm, pour une seule coupe, relevées sur les
deux planches des dormants (registre 2.3.1, p. 2 et 3, version septembre 2023). Les deux
pictogrammes de chaque planche montrent la coupe prise sur un châssis fixe à traverse et sur une
fenêtre oscillo-battante.

## Coupe du dormant 6100

![Coupes cotées du dormant 6100, repères ① à ④](/assets/procedures/moe-systeme-70/debit/debit-dormant-6100-reperes.png)

À droite, le dormant 6100 avec l'ouvrant : ① DEO et ② DFO, prises depuis le bord extérieur
du dormant, la DFO se trouvant 20 mm plus loin que la DEO ; un jeu de 12+1 mm est coté
entre le dormant et l'ouvrant. À gauche, le dormant seul : ④ jusqu'à la « Dimension
traverse » et ③ jusqu'à la « Dimension vitrage pour fixe ». Le renfort est dessiné hachuré
dans la chambre centrale.

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 105)

## Coupe des dormants 6101 et 6104

![Coupes cotées des dormants 6101 et 6104, repères ① à ④](/assets/procedures/moe-systeme-70/debit/debit-dormants-6101-6104-reperes.png)

À gauche, le dormant 6101 seul, avec ④ et ③. À droite, le dormant 6104 avec l'ouvrant : la
DHT part de l'aile de recouvrement du 6104, qui déborde de 20 mm ; ① (36) va de ce point au
bord de l'ouvrant, 56 du bord extérieur du dormant au bord de l'ouvrant, puis 20 mm jusqu'à
② DFO ; jeu de 12⁺¹ mm entre dormant et ouvrant et 8 mm entre le bord du dormant et
l'ouvrant.

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 106)

## Tableau

Les vignettes des dormants à aile (rénovation) portent la largeur de l'aile à côté de la flèche de
la DHT : 40 (6105, 6106, 6155), 60 (6107), 70 (6156), 20 (6104 et 6108 à 6111,
6158). Une colonne de la planche groupe plusieurs dormants sous une seule valeur : chaque dormant a
ici sa ligne.

<table>
<thead>
<tr><th rowspan="2">Dormant</th><th rowspan="2">① DEO (mm)</th><th rowspan="2">② DFO (mm)</th><th rowspan="2">③ Vitrage fixe (mm)</th><th rowspan="2">Renfort de dormant (mm)</th><th rowspan="2">④ Meneau / traverse (mm)</th><th colspan="2">Renfort de meneau / traverse (mm)</th><th rowspan="2">Page PDF</th></tr>
<tr><th>set d'assemblage mécanique</th><th>équerre d'assemblage</th></tr>
</thead>
<tbody>
<tr><td>6100</td><td>27</td><td>47</td><td>38</td><td>32</td><td>33</td><td>93</td><td>38</td><td>105</td></tr>
<tr><td>6106</td><td>27</td><td>47</td><td>38</td><td>32</td><td>33</td><td>93</td><td>38</td><td>105</td></tr>
<tr><td>6101</td><td>36</td><td>56</td><td>47</td><td>41</td><td>42</td><td>102</td><td>47</td><td>105</td></tr>
<tr><td>2502</td><td>57</td><td>77</td><td>68</td><td>62</td><td>63</td><td>123</td><td>68</td><td>105</td></tr>
<tr><td>6105</td><td>19</td><td>39</td><td>30</td><td>24</td><td>25</td><td>85</td><td>30</td><td>105</td></tr>
<tr><td>6107</td><td>19</td><td>39</td><td>30</td><td>24</td><td>25</td><td>85</td><td>30</td><td>105</td></tr>
<tr><td>6155</td><td>19</td><td>39</td><td>30</td><td>24</td><td>25</td><td>85</td><td>30</td><td>105</td></tr>
<tr><td>6156</td><td>19</td><td>39</td><td>30</td><td>24</td><td>25</td><td>85</td><td>30</td><td>105</td></tr>
<tr><td>6104</td><td>36</td><td>56</td><td>47</td><td>41</td><td>42</td><td>102</td><td>47</td><td>106</td></tr>
<tr><td>6108</td><td>36</td><td>56</td><td>47</td><td>41</td><td>42</td><td>102</td><td>47</td><td>106</td></tr>
<tr><td>6109</td><td>36</td><td>56</td><td>47</td><td>41</td><td>42</td><td>102</td><td>47</td><td>106</td></tr>
<tr><td>6110</td><td>36</td><td>56</td><td>47</td><td>41</td><td>42</td><td>102</td><td>47</td><td>106</td></tr>
<tr><td>6111</td><td>36</td><td>56</td><td>47</td><td>41</td><td>42</td><td>102</td><td>47</td><td>106</td></tr>
<tr><td>6158</td><td>36</td><td>56</td><td>47</td><td>41</td><td>42</td><td>102</td><td>47</td><td>106</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, registre 2.3.1, p. 2 et 3)

Le dormant rénovation 6102 (aile de 30 mm) figure dans les tableaux des battements ci-dessous
mais dans aucun des deux tableaux des dormants : ses cotes à déduire de la DHT (DEO, DFO, vitrage
fixe, renfort, meneau) ne sont pas données — entrée **VER-85** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Les cotes des profilés eux-mêmes (largeurs, ailes, chambres) sont dans
[Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md#cotes-des-dormants).

# Cotes de débit des meneaux et traverses

Cotes à déduire en mm, pour une seule coupe, relevées sur la planche des meneaux / traverses
(registre 2.3.1, p. 4). L'en-tête du tableau écrit « partant de la dimension hors tout = DHT » ;
sur la coupe, toutes les cotes partent de l'axe du meneau, comme la cote X ou Y de l'exemple.

![Coupe cotée du meneau 6127, repères ① à ④](/assets/procedures/moe-systeme-70/debit/debit-meneau-6127-reperes.png)

Le meneau 6127 est dessiné avec l'ouvrant à droite et le côté fixe à gauche. Depuis l'axe, vers
l'ouvrant : ① DEO puis ② DFO, avec un jeu de 12 mm entre meneau et ouvrant ; vers le
fixe : ④ jusqu'à la « Dimension traverse » (meneau ou traverse venant buter dans ce meneau) et
③ jusqu'à la « Dimension vitrage pour fixe ». Le renfort est dessiné hachuré dans la chambre
centrale.

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 107)

<table>
<thead>
<tr><th rowspan="2">Meneau / traverse</th><th rowspan="2">① DEO (mm)</th><th rowspan="2">② DFO (mm)</th><th rowspan="2">③ Vitrage fixe (mm)</th><th rowspan="2">④ Meneau / traverse dans meneau / traverse (mm)</th><th colspan="2">Renfort de meneau / traverse (mm)</th></tr>
<tr><th>set d'assemblage mécanique</th><th>équerre d'assemblage</th></tr>
</thead>
<tbody>
<tr><td>6127</td><td>12</td><td>32</td><td>23</td><td>17</td><td>77</td><td>22</td></tr>
<tr><td>6157</td><td>12</td><td>32</td><td>23</td><td>17</td><td>77</td><td>22</td></tr>
<tr><td>2425</td><td>17</td><td>37</td><td>28</td><td>22</td><td>82</td><td>27</td></tr>
<tr><td>2427</td><td>29,5</td><td>49,5</td><td>40,5</td><td>35,5</td><td>95,5</td><td>40,5</td></tr>
<tr><td>6126</td><td>—</td><td>—</td><td>17</td><td>11</td><td>—</td><td>16</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, registre 2.3.1, p. 4)

Les meneaux 6127 et 6157 partagent la même colonne. Le 6126 porte un tiret « — » en DEO,
en DFO et en renfort pour set d'assemblage mécanique : la planche ne donne pour lui que le vitrage
fixe (17), le meneau dans meneau (11) et le renfort pour équerre d'assemblage (16).

# Cotes de débit des ouvrants

Cotes à déduire de la dimension extérieure d'ouvrant (DEO), et non de la DHT, en mm, pour une
seule coupe, relevées sur la planche des ouvrants (registre 2.3.1, p. 5).

![Coupe cotée de l'ouvrant 6121, repères ② à ④](/assets/procedures/moe-systeme-70/debit/debit-ouvrant-6121-reperes.png)

L'ouvrant 6121 est dessiné dans son dormant, jeu de 12⁺¹ mm. Toutes les cotes partent du
bord extérieur de l'ouvrant (DEO) : ② jusqu'à la DFO, ④ jusqu'à la « Dimension
meneau/traverse » (traverse d'ouvrant) et ③ jusqu'à la « Dimension vitrage ».

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 108)

<table>
<thead>
<tr><th rowspan="2">Ouvrant</th><th rowspan="2">② DFO (mm)</th><th rowspan="2">③ Vitrage (mm)</th><th rowspan="2">Renfort d'ouvrant (mm)</th><th rowspan="2">④ Meneau / traverse (mm)</th><th colspan="2">Renfort de meneau / traverse (mm)</th></tr>
<tr><th>set d'assemblage mécanique</th><th>équerre d'assemblage</th></tr>
</thead>
<tbody>
<tr><td>6112</td><td>20</td><td>56</td><td>50</td><td>51</td><td>111</td><td>56</td></tr>
<tr><td>6121</td><td>20</td><td>56</td><td>50</td><td>51</td><td>111</td><td>56</td></tr>
<tr><td>6150</td><td>20</td><td>56</td><td>50</td><td>51</td><td>111</td><td>56</td></tr>
<tr><td>6117</td><td>20</td><td>56</td><td>50</td><td>51</td><td>111</td><td>56</td></tr>
<tr><td>6152</td><td>20</td><td>80</td><td>74</td><td>75</td><td>135</td><td>80</td></tr>
<tr><td>6119</td><td>20</td><td>80</td><td>74</td><td>75</td><td>135</td><td>80</td></tr>
<tr><td>6115</td><td>20</td><td>80</td><td>74</td><td>75</td><td>135</td><td>80</td></tr>
<tr><td>6123</td><td>20</td><td>80</td><td>74</td><td>75</td><td>135</td><td>80</td></tr>
<tr><td>2418</td><td>20</td><td>79</td><td>73</td><td>74</td><td>134</td><td>79</td></tr>
<tr><td>2416</td><td>20</td><td>101</td><td>95</td><td>96</td><td>156</td><td>101</td></tr>
<tr><td>2415</td><td>20</td><td>101</td><td>95</td><td>96</td><td>156</td><td>101</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, registre 2.3.1, p. 5)

La feuillure se déduit de 20 mm sur les onze ouvrants. Chaque colonne de la planche groupe deux
ouvrants dessinés l'un au-dessus de l'autre (6112 et 6121, 6150 et 6117, 6152 et 6119, 6115 et
6123, 2416 et 2415) ; l'ouvrant 2418 est seul dans sa colonne. La planche n'a pas de ligne ①.

# Cotes de débit d'un dormant recevant un battement

Sur une fenêtre à deux vantaux sans meneau, les cotes ne se déduisent pas de la DHT mais de X,
la distance de l'axe du châssis au bord extérieur du dormant (le pictogramme cote X du bord du
dormant à l'axe, Y de l'axe à l'autre bord). Elles changent avec le battement utilisé : le
battement 0140 d'une part, les battements 6130, 6128, 1578 et 6132 d'autre part, qui ont
tous les mêmes valeurs (registre 2.3.1, p. 6 à 16).

## Battement 0140

![Coupe cotée du battement 0140](/assets/procedures/moe-systeme-70/debit/debit-battement-0140-reperes.png)

Le battement tubulaire 0140, de 62 mm de large, est dessiné entre les deux ouvrants ; la
DFO de l'ouvrant de gauche est cotée jusqu'à son bord, jeu de 12⁺¹ mm entre les ouvrants, et,
sous la coupe, 6 mm et 23 mm cotés à partir de l'axe, X étant pris jusqu'à l'axe.

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 109)

## Battements extérieurs et intérieurs

Les autres battements se composent d'un battement extérieur (clipsé côté extérieur sur
l'ouvrant semi-fixe) et d'un battement intérieur (sous la jonction, côté intérieur). Sur chaque
coupe, la DEO de l'ouvrant de droite part à 14 mm au-delà de l'axe, sous le battement
intérieur ; les DFO sont cotées de part et d'autre de l'axe ; le jeu entre les ouvrants est coté
12⁺¹ mm (p. 110, 111, 116 à 119) ou 12 mm (p. 112 à 115).

![Battement 6130 et battement intérieur 6133, ouvrants 6113 et 6112](/assets/procedures/moe-systeme-70/debit/debit-battement-6130-6133-ouvrants-6112-6113.png)

![Battement 6130 et battement intérieur 6133, ouvrants 6116 et 6115](/assets/procedures/moe-systeme-70/debit/debit-battement-6130-6133-ouvrants-6115-6116.png)

![Battement 6128 et battement intérieur 6133 dessiné, ouvrants 6113 et 6112](/assets/procedures/moe-systeme-70/debit/debit-battement-6128-6133-ouvrants-6112-6113.png)

![Battement 6128 et battement intérieur 6133 dessiné, ouvrants 6116 et 6115](/assets/procedures/moe-systeme-70/debit/debit-battement-6128-6133-ouvrants-6115-6116.png)

![Battement 1578 et battement intérieur 1547 dessiné, ouvrants 6113 et 6112](/assets/procedures/moe-systeme-70/debit/debit-battement-1578-1547-ouvrants-6112-6113.png)

![Battement 1578 et battement intérieur 1547 dessiné, ouvrants 6116 et 6115](/assets/procedures/moe-systeme-70/debit/debit-battement-1578-1547-ouvrants-6115-6116.png)

![Battement 6132 et battement intérieur 6131, ouvrants 6122 et 6121](/assets/procedures/moe-systeme-70/debit/debit-battement-6132-6131-ouvrants-6121-6122.png)

![Battement 6132 (légendé 61362) et battement intérieur 6131, ouvrants 6124 et 6123](/assets/procedures/moe-systeme-70/debit/debit-battement-6132-6131-ouvrants-6123-6124.png)

![Battement 6162 dessiné et battement intérieur 6131, ouvrants 6122 et 6121](/assets/procedures/moe-systeme-70/debit/debit-battement-6162-6131-ouvrants-6121-6122.png)

![Battement 6162 dessiné et battement intérieur 6131, ouvrants 6124 et 6123](/assets/procedures/moe-systeme-70/debit/debit-battement-6162-6131-ouvrants-6123-6124.png)

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 110 à 119)

## Tableau

Cotes à déduire de X, en mm, une ligne par dormant, une colonne par battement extérieur.

<table>
<thead>
<tr><th rowspan="2">Dormant</th><th colspan="2">Battement 0140 (p. 109)</th><th colspan="2">Battements 6130, 6128, 1578, 6132 (p. 110 à 119)</th></tr>
<tr><th>DEO (mm)</th><th>DFO (mm)</th><th>DEO (mm)</th><th>DFO (mm)</th></tr>
</thead>
<tbody>
<tr><td>6100</td><td>X − 30</td><td>X − 70</td><td>X − 13</td><td>X − 53</td></tr>
<tr><td>6106</td><td>X − 30</td><td>X − 70</td><td>X − 13</td><td>X − 53</td></tr>
<tr><td>6102</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>6105</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>6107</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>6155</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>6156</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>6101</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6104</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6108</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6109</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6110</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6111</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6158</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>2502</td><td>X − 60</td><td>X − 100</td><td>X − 43</td><td>X − 83</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, registre 2.3.1, p. 6 à 16)

Les dormants 6109, 6110, 6111 et 6158 sont listés « (non représentés) » dans la colonne du 6101 et
du 6104. Chaque planche de battement ne liste pas tous les dormants ; une case vide ci-dessous
signifie que le dormant n'apparaît pas sur la planche :

| Page PDF | Battement extérieur | Battement intérieur | Ouvrants dessinés | Dormants absents de la planche |
| --- | --- | --- | --- | --- |
| 109 | 0140 | - | - | aucun |
| 110 | 6130 | 6133 | 6113, 6112 | aucun |
| 111 | 6130 | 6133 | 6116, 6115 | aucun |
| 112 | 6128 | 6129 au tableau, 6133 au dessin | 6113, 6112 | aucun |
| 113 | 6128 | 6129 au tableau, 6133 au dessin | 6116, 6115 | 6155 |
| 114 | 1578 | 6131 au tableau, 1547 au dessin | 6113, 6112 | aucun |
| 115 | 1578 | 6131 au tableau, 1547 au dessin | 6116, 6115 | 6155 |
| 116 | 6132 | 6131 | 6122, 6121 | 6156, 6158 |
| 117 | 6132 au tableau, 61362 au dessin | 6131 | 6124, 6123 | 6155, 6156, 6158 |
| 118 | 6132 au tableau, 6162 au dessin | 6131 | 6122, 6121 | 6156, 6158 |
| 119 | 6132 au tableau, 6162 au dessin | 6131 | 6124, 6123 | 6155, 6156, 6158 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 109 à 119)

Le dormant 6155 n'apparaît pas sur les planches des ouvrants 6115 / 6116 et 6123 / 6124 avec
les battements 6128, 1578 et 6132, et les dormants 6156 et 6158 sur aucune planche du
battement 6132 ; ni exclusion ni renvoi ne sont écrits — entrée **VER-85**.

Quatre écarts entre le tableau et le dessin d'une même planche : le battement intérieur est
écrit 6129 au tableau et dessiné 6133 (p. 112-113) — **INC-126** ; écrit 6131 et
dessiné 1547 (p. 114-115) — **INC-127** ; le battement extérieur est légendé 61362 au
dessin et 6132 au tableau (p. 117) — **INC-128** ; il est dessiné 6162, de profil
différent, et écrit 6132 au tableau (p. 118-119) — **INC-129**. Les valeurs de débit sont les
mêmes dans tous ces cas. Registre
[Incohérences internes](/anomalies/incoherences-internes.md).

# Débit des battements

Formules relevées sous les tableaux de battement (registre 2.3.1, p. 6 à 16). `DEO` est la
dimension extérieure d'ouvrant obtenue au tableau précédent.

| Battement extérieur | Battement intérieur | Débit du battement extérieur | Débit du battement intérieur | Pages PDF |
| --- | --- | --- | --- | --- |
| 0140 | - | DEO − 72 mm | - | 109 |
| 6130 | 6133 | DEO − 70 mm | DEO − 12 mm | 110-111 |
| 6128 | 6129 (tableau) / 6133 (dessin) | DEO − 70 mm | DEO − 12 mm | 112-113 |
| 1578 | 6131 (tableau) / 1547 (dessin) | DEO − 70 mm | DEO − 12 mm | 114-115 |
| 6132 | 6131 | DEO − 70 mm | DEO − 12 mm | 116-117 |
| 6132 (tableau) / 6162 (dessin) | 6131 | DEO − 70 mm | DEO − 12 mm | 118-119 |

Le battement 0140 se débite à DEO − 72 mm ; tous les battements extérieurs des autres planches à
DEO − 70 mm et tous les battements intérieurs à DEO − 12 mm. Les planches ne donnent ni le débit
d'un renfort de battement ni d'embout. Le registre 2.4.3 du même manuel donne d'autres débits (battement 0140 à DFO − 32 mm, 1578 à DFO − 35 mm, battement intérieur à DEO − 6 mm) : la contradiction n'est pas arbitrée (**INC-153**). Les profilés battements sont décrits dans
[Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md#cotes-des-battements),
leur mise en œuvre dans [Traitement du battement du système 70](/procedures/traitement-du-battement-systeme-70.md).

# Ce que la source ne donne pas

- Les cotes à déduire de la DHT du dormant 6102 (**VER-85**)
- La ligne ① (DEO) du tableau des ouvrants, et les cotes DEO / DFO du meneau 6126
- Le débit des seuils, des parcloses et des renforts de battement : aucune planche du registre
  2.3.1 ne les porte ; le débit des parcloses est dans
  [Tableau de vitrage du système 70](/profiles/systeme-70-tableau-de-vitrage.md)

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.1 « Cotes de débit/Coupes »,
p. 104 à 119 du PDF (pages imprimées 1 à 16)

# Voir aussi

- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Assemblages du système 70](/profiles/systeme-70-assemblages.md)
- [Plans de combinaison du système 70](/profiles/systeme-70-plans-de-combinaison.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
