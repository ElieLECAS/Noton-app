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
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008
    last_modified: 2008-08-31
source_pages:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    pages: 104-119
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 109-119
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

# Cotes de débit des plans e.VOLUTION de 2008

Le classeur KÖMMERLING « Système e.VOLUTION » d'août 2008 (système F 91) donne les mêmes cotes à
déduire dans son registre 3, « Côtes de débit » (orthographe de la planche). La page imprimée N
du registre 3 est la page N + 108 du PDF. Les planches des dormants sont à l'échelle 1:1 ; la
planche d'introduction porte « Dessins non à l'échelle ». Ces cotes décrivent les menuiseries
fabriquées d'après ce classeur ; les écarts avec le manuel de 2023 sont signalés sous chaque
table [2 p. 109-119].

## Indications et exemple de 2008 (p. 109)

« Indications concernant la détermination des côtes de débit : Pour déterminer les côtes à
déduire, il faut se reporter aux valeurs indiquées dans les tableaux des pages suivantes. Ces
valeurs sont à prendre sur les différentes coupes représentées pour chaque cas. » L'exemple est
« une fenêtre à deux vantaux avec meneau », « Dimension extérieur dormant = dimension hors tout
DHT = 2000 x 1200 mm (L x H) », dessinée avec deux vantaux marqués du symbole oscillo-battant et coupée en
B-B à mi-hauteur [2 p. 109].

![Fenêtre deux vantaux de l'exemple de 2008, 2 000 × 1 200 mm, coupe B-B](/assets/profiles/systeme70/debit/debit-exemple-fenetre-deux-vantaux-evo2008.png)

La coupe B-B montre, de gauche à droite, le dormant 6101, l'ouvrant 6112, le meneau 6127,
l'ouvrant 6112 et le dormant 6101. La dimension hors tout (DHT) est partagée à l'axe du meneau en
X et Y. Sous la coupe, de chaque côté : la « Dimension vitrage », la « Dimension feuillure
ouvrant » (DFO) et la « Dimension extérieure ouvrant » (DEO) de chaque vantail ; a est la cote du
dormant, du bord extérieur du dormant au bord de l'ouvrant, et b celle du meneau, de l'axe du
meneau au bord de l'ouvrant.

![Coupe B-B de l'exemple de 2008 : dormants 6101, ouvrants 6112, meneau 6127](/assets/profiles/systeme70/debit/debit-exemple-coupe-b-b-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 109)

« Cotes à soustraire » : 1) « Détermination des dimensions extérieures ouvrant (largeur) DEO »,
**DEO = X / Y – (a + b)** ; 2) « Détermination de la dimension du vitrage », **Dimension vitrage =
DEO – 2 x (55,5)**. Les vignettes donnent a = 36 sous le dormant 6101, b = 12 sous le meneau
6127 et 55,5 sous l'ouvrant 6112 [2 p. 109].

| Étape | Calcul imprimé | Résultat (mm) |
| --- | --- | --- |
| Données de l'exemple | « DHT = 2000; X = 1000; a = 36; b = 24 » | - |
| Dimension extérieure d'ouvrant | DEO = 1000 – (36 + 12) | 952 |
| Dimension du vitrage | = 952 – 111 | 841 |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 109)

La ligne des données écrit **b = 24**, alors que le calcul qui suit et la vignette du meneau 6127
donnent **b = 12** (**INC-217**). La formule du vitrage de 2008 écrit 2 x (55,5), la valeur de la
vignette et du calcul (2 × 55,5 = 111) ; celle du manuel de 2023 écrit 2 x (56) (**INC-125**).

![Repères a = 36 (dormant 6101) et b = 12 (meneau 6127), 2008](/assets/profiles/systeme70/debit/debit-exemple-reperes-a-b-evo2008.png)

![Dimension vitrage = DEO – 2 x (55,5), vignette de l'ouvrant 6112 cotée 55,5, 2008](/assets/profiles/systeme70/debit/debit-exemple-dimension-vitrage-evo2008.png)

## Cotes de débit des dormants de 2008 (p. 110 et 111)

Chaque planche porte deux pictogrammes : un châssis fixe à traverse, coupé sur le montant gauche
au-dessus de la traverse, et une fenêtre oscillo-battante coupée sur le montant gauche. Sous
eux, la coupe cotée du dormant, puis le tableau « Côtes de débit — Dormants », « Les dimensions
indiquées se réfèrent uniquement à une seule coupe », « Côtes à déduire en mm (partant de la
dimension hors tout = DHT) » [2 p. 110-111].

### Coupe du dormant 6100 (2008, p. 110)

À gauche, le dormant F91-01- 6100 seul : ④ va du bord extérieur du dormant à la « Dimension
traverse », ③ jusqu'à la « Dimension vitrage pour fixe ». À droite, le même dormant avec
l'ouvrant : ① DEO et ② DFO, prises depuis le bord extérieur du dormant, la DFO 20 mm plus loin que
la DEO ; un jeu de 12+1 mm est coté entre le dormant et l'ouvrant. Le renfort est dessiné hachuré
dans la chambre centrale.

![Coupes cotées du dormant 6100 de 2008, repères ① à ④](/assets/profiles/systeme70/debit/debit-dormant-6100-reperes-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 110)

### Coupe des dormants 6101 et 6104 (2008, p. 111)

À gauche, le dormant F91-01- 6101 seul, avec ④ et ③. À droite, le dormant F91-01- 6104 avec
l'ouvrant : la DHT part de l'aile de recouvrement du 6104 ; ① « (36) » va de ce point au bord de
l'ouvrant, 56 du bord gauche du dormant (en pied) au bord de l'ouvrant, puis 20 mm jusqu'à ② DFO ; jeu de
12⁺¹ mm entre dormant et ouvrant et 8 mm entre le bord du dormant et l'ouvrant.

![Coupes cotées des dormants 6101 et 6104 de 2008, repères ① à ④](/assets/profiles/systeme70/debit/debit-dormants-6101-6104-reperes-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 111)

### Tableau des dormants de 2008

Les vignettes du tableau dessinent chaque dormant et portent la largeur de l'aile à côté de la
flèche de la DHT : 40 (6106, 6105), 30 (6102), 60 (6107), 20 (6104, 6108 à 6111). Sur la planche,
une colonne groupe plusieurs dormants sous une seule valeur (6100 et 6106 ; 6102, 6105 et 6107 ;
6108 à 6111) : chaque dormant a ici sa ligne. Cotes à déduire de la DHT, en mm.

<table>
<thead>
<tr><th rowspan="2">Dormant</th><th rowspan="2">Aile portée sur la vignette (mm)</th><th rowspan="2">① DEO (mm)</th><th rowspan="2">② DFO (mm)</th><th rowspan="2">③ Vitrage fixe (mm)</th><th rowspan="2">Renfort de dormant (mm)</th><th rowspan="2">④ Meneau / traverse (mm)</th><th colspan="2">Renfort de meneau / traverse (mm)</th><th rowspan="2">Page PDF</th></tr>
<tr><th>set d'assemblage mécanique</th><th>équerre d'assemblage</th></tr>
</thead>
<tbody>
<tr><td>6100</td><td>-</td><td>27</td><td>47</td><td>37,5</td><td>32</td><td>33</td><td>93</td><td>38</td><td>110</td></tr>
<tr><td>6106</td><td>40</td><td>27</td><td>47</td><td>37,5</td><td>32</td><td>33</td><td>93</td><td>38</td><td>110</td></tr>
<tr><td>6101</td><td>-</td><td>36</td><td>56</td><td>46,5</td><td>41</td><td>42</td><td>102</td><td>47</td><td>110</td></tr>
<tr><td>2502</td><td>-</td><td>57</td><td>77</td><td>67,5</td><td>62</td><td>63</td><td>123</td><td>68</td><td>110</td></tr>
<tr><td>6102</td><td>30</td><td>19</td><td>39</td><td>29,5</td><td>24</td><td>25</td><td>85</td><td>30</td><td>110</td></tr>
<tr><td>6105</td><td>40</td><td>19</td><td>39</td><td>29,5</td><td>24</td><td>25</td><td>85</td><td>30</td><td>110</td></tr>
<tr><td>6107</td><td>60</td><td>19</td><td>39</td><td>29,5</td><td>24</td><td>25</td><td>85</td><td>30</td><td>110</td></tr>
<tr><td>2403</td><td>-</td><td>42</td><td>62</td><td>52,5</td><td>47</td><td>48</td><td>108</td><td>53</td><td>110</td></tr>
<tr><td>6104</td><td>20</td><td>36</td><td>56</td><td>46,5</td><td>41</td><td>42</td><td>102</td><td>47</td><td>111</td></tr>
<tr><td>6108</td><td>20</td><td>36</td><td>56</td><td>46,5</td><td>41</td><td>42</td><td>102</td><td>47</td><td>111</td></tr>
<tr><td>6109</td><td>20</td><td>36</td><td>56</td><td>46,5</td><td>41</td><td>42</td><td>102</td><td>47</td><td>111</td></tr>
<tr><td>6110</td><td>20</td><td>36</td><td>56</td><td>46,5</td><td>41</td><td>42</td><td>102</td><td>47</td><td>111</td></tr>
<tr><td>6111</td><td>20</td><td>36</td><td>56</td><td>46,5</td><td>41</td><td>42</td><td>102</td><td>47</td><td>111</td></tr>
</tbody>
</table>

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 110 et 111)

![En-tête du tableau des dormants de 2008, p. 110 : vignettes 6100, 6106, 6101, 2502, 6102, 6105, 6107, 2403](/assets/profiles/systeme70/debit/debit-dormants-tableau-1-vignettes-evo2008.png)

![En-tête du tableau des dormants de 2008, p. 111 : vignettes 6104, 6108, 6109, 6110, 6111](/assets/profiles/systeme70/debit/debit-dormants-tableau-2-vignettes-evo2008.png)

Sur chaque colonne, la DFO vaut la DEO plus 20 mm, le vitrage fixe la cote meneau / traverse plus
4,5 mm, le renfort de dormant la cote meneau / traverse moins 1 mm, le renfort de meneau la cote
meneau / traverse plus 60 mm (set d'assemblage mécanique) ou plus 5 mm (équerre d'assemblage).

Par rapport au tableau du manuel de 2023 (section *Cotes de débit des dormants* ci-dessus) : le
vitrage fixe est plus court de 0,5 mm pour chaque dormant (37,5 contre 38 ; 46,5 contre 47 ; 67,5
contre 68 ; 29,5 contre 30) (**CTR-78**) ; toutes les autres cotes sont les mêmes. Le tableau de
2008 donne les cotes du dormant rénovation 6102, que le manuel de 2023 ne donne pas (**VER-85**),
et celles du dormant 2403, absent du manuel ; il n'a pas les dormants 6155, 6156 et 6158
[2 p. 110-111].

## Cotes de débit des meneaux et traverses de 2008 (p. 112)

Le meneau est le profilé vertical, la traverse le profilé horizontal, qui partagent un châssis en
plusieurs parties. La planche du registre 3, page imprimée 4, porte un pictogramme de châssis :
à gauche un fixe à traverse, à droite un ouvrant oscillo-battant, la coupe étant prise sur le
meneau qui les sépare [2 p. 112].

![Coupe cotée du meneau 6127 de 2008, repères ① à ④, avec le pictogramme de la coupe](/assets/profiles/systeme70/debit/debit-meneau-6127-reperes-evo2008.png)

Le meneau F91-15- 6127 est dessiné avec son renfort hachuré dans la chambre centrale, l'ouvrant à
droite, le côté fixe à gauche. Toutes les cotes partent de l'axe du meneau (trait mixte vertical).
Vers l'ouvrant : ① jusqu'au bord de l'ouvrant, la « DEO » (dimension extérieure d'ouvrant), puis ②
jusqu'à la « DFO » (dimension de feuillure d'ouvrant) ; un jeu de 12⁺¹ mm est coté entre le meneau
et l'ouvrant. Vers le fixe : ④ jusqu'à la « Dimension traverse » (un meneau ou une traverse venant
buter dans celui-ci) et ③ jusqu'à la « Dimension vitrage pour fixe ».

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 112)

Tableau « Côtes de débit — Meneau/traverses », « Les dimensions indiquées se réferent uniquement à
une seule coupe », « Côtes à déduire en mm partant de la dimension hors tout = DHT », « Dessins
non à l'échelle ». Une ligne par meneau, cotes en mm ; « — » est le tiret imprimé sur la planche
(aucune valeur).

<table>
<thead>
<tr><th rowspan="2">Meneau / traverse</th><th rowspan="2">① DEO (mm)</th><th rowspan="2">② DFO (mm)</th><th rowspan="2">③ Vitrage fixe (mm)</th><th rowspan="2">④ Meneau / traverse dans meneau / traverse (mm)</th><th colspan="2">Renfort de meneau / traverse (mm)</th></tr>
<tr><th>set d'assemblage mécanique</th><th>équerre d'assemblage</th></tr>
</thead>
<tbody>
<tr><td>6127</td><td>12</td><td>32</td><td>22,5</td><td>17</td><td>77</td><td>22</td></tr>
<tr><td>2425</td><td>17</td><td>37</td><td>27,5</td><td>22</td><td>82</td><td>27</td></tr>
<tr><td>2427</td><td>29,5</td><td>49,5</td><td>40</td><td>35,5</td><td>95,5</td><td>40,5</td></tr>
<tr><td>2469</td><td>—</td><td>—</td><td>16,5</td><td>11</td><td>—</td><td>16</td></tr>
</tbody>
</table>

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 112)

![En-tête du tableau des meneaux de 2008 : vignettes 6127, 2425, 2427, 2469](/assets/profiles/systeme70/debit/debit-meneaux-tableau-vignettes-evo2008.png)

Le meneau 2469 n'a ni DEO, ni DFO, ni renfort pour set d'assemblage mécanique : la planche ne
donne pour lui que le vitrage fixe (16,5), le meneau dans meneau (11) et le renfort pour équerre
d'assemblage (16). Par rapport au tableau du manuel de 2023 (section *Cotes de débit des meneaux
et traverses* ci-dessus) : le vitrage fixe est plus court de 0,5 mm (22,5 contre 23 pour le 6127,
27,5 contre 28 pour le 2425, 40 contre 40,5 pour le 2427) (**CTR-78**), toutes les autres cotes
des 6127, 2425 et 2427 sont les mêmes ; le tableau de 2008 porte le 2469 et non les 6157 et 6126 du
manuel de 2023 [2 p. 112].

## Cotes de débit des ouvrants de 2008 (p. 113)

L'ouvrant est le cadre mobile qui porte le vitrage. Ses cotes se déduisent de la dimension
extérieure d'ouvrant (DEO) obtenue au tableau des dormants ou des meneaux, et non de la dimension
hors tout. Le pictogramme montre un ouvrant oscillo-battant coupé sur son montant gauche [2 p. 113].

![Coupe cotée de l'ouvrant 6121 de 2008, repères ② à ④, avec le pictogramme de la coupe](/assets/profiles/systeme70/debit/debit-ouvrant-6121-reperes-evo2008.png)

L'ouvrant F91-06- 6121 est dessiné dans son dormant, avec un jeu de 12⁺¹ mm, son renfort hachuré
dans la chambre. Toutes les cotes partent du bord extérieur de l'ouvrant, origine de la DEO : ②
jusqu'à la DFO, ④ jusqu'à la « Dimension meneau/traverse » (traverse d'ouvrant) et ③ jusqu'à la
« Dimension vitrage ».

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 113)

Tableau « Côtes de débit — Ouvrants », « Les dimensions indiquées se réfèrent uniquement à une
seule coupe », « Côtes à déduire en mm (partant de la dimension extérieure d'ouvrant = DEO) ». Sur
la planche, une colonne groupe deux ouvrants dessinés l'un au-dessus de l'autre (6112 et 6121,
6115 et 6123, 2416 et 2415) : chaque ouvrant a ici sa ligne. Cotes en mm ; la planche n'a pas de
ligne ①.

<table>
<thead>
<tr><th rowspan="2">Ouvrant</th><th rowspan="2">② DFO (mm)</th><th rowspan="2">③ Vitrage fixe (mm)</th><th rowspan="2">Renfort d'ouvrant (mm)</th><th rowspan="2">④ Meneau / traverse (mm)</th><th colspan="2">Renfort de meneau / traverse (mm)</th></tr>
<tr><th>set d'assemblage mécanique</th><th>équerre d'assemblage</th></tr>
</thead>
<tbody>
<tr><td>6112</td><td>20</td><td>55,5</td><td>50</td><td>51</td><td>111</td><td>56</td></tr>
<tr><td>6121</td><td>20</td><td>55,5</td><td>50</td><td>51</td><td>111</td><td>56</td></tr>
<tr><td>0112</td><td>20</td><td>56,5</td><td>51</td><td>52</td><td>112</td><td>57</td></tr>
<tr><td>0113</td><td>20</td><td>64,5</td><td>59</td><td>60</td><td>120</td><td>65</td></tr>
<tr><td>6115</td><td>20</td><td>79,5</td><td>74</td><td>75</td><td>135</td><td>80</td></tr>
<tr><td>6123</td><td>20</td><td>79,5</td><td>74</td><td>75</td><td>135</td><td>80</td></tr>
<tr><td>2418</td><td>20</td><td>78,5</td><td>73</td><td>74</td><td>134</td><td>79</td></tr>
<tr><td>2416</td><td>20</td><td>100,5</td><td>95</td><td>96</td><td>156</td><td>101</td></tr>
<tr><td>2415</td><td>20</td><td>100,5</td><td>95</td><td>96</td><td>156</td><td>101</td></tr>
</tbody>
</table>

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 113)

![En-tête du tableau des ouvrants de 2008 : vignettes 6112, 6121, 0112, 0113, 6115, 6123, 2418, 2416, 2415](/assets/profiles/systeme70/debit/debit-ouvrants-tableau-vignettes-evo2008.png)

La ligne ③ est écrite « Vitrage (fixe) » sur cette planche des ouvrants. La feuillure se déduit de
20 mm sur les neuf ouvrants. Le vitrage du 6112 vaut 55,5, la valeur de l'exemple de la p. 109
(« 2 x (55,5) ») ; le tableau du manuel de 2023 donne 56 (**INC-125**). Par rapport à ce tableau de
2023 (section *Cotes de débit des ouvrants* ci-dessus) : le vitrage est plus court de 0,5 mm pour
chaque ouvrant commun (55,5 contre 56 ; 79,5 contre 80 ; 78,5 contre 79 ; 100,5 contre 101)
(**CTR-78**), toutes les autres cotes sont les mêmes ; le tableau de 2008 porte les ouvrants 0112 et
0113, absents du manuel de 2023, et n'a pas les 6150, 6117, 6152 et 6119 [2 p. 113].

## Cotes de débit d'un dormant recevant un battement, 2008 (p. 114 à 119)

Sur une fenêtre à deux vantaux sans meneau, le battement est le profilé qui couvre la jonction des
deux ouvrants. Les cotes ne se déduisent pas de la DHT mais de X, la distance du bord extérieur du
dormant à l'axe du châssis (Y de l'axe à l'autre bord) ; le pictogramme de chaque planche montre
une fenêtre deux vantaux, vantail de droite oscillo-battant, coupée à la jonction. Chaque tableau
porte « Les dimensions indiquées se réfèrent uniquement à une seule coupe avec utilisation du
battement … » et « Côtes à déduire en mm » [2 p. 114-119].

### Battement 0141 (p. 114)

![Coupe cotée du battement 0141 de 2008 entre deux ouvrants](/assets/profiles/systeme70/debit/debit-battement-0141-reperes-evo2008.png)

Le battement F90-25- 0141, coté 44 mm de large, est dessiné entre les deux ouvrants avec son
renfort hachuré ; la DFO de l'ouvrant de gauche est cotée jusqu'à son bord, jeu de 12⁺¹ mm entre
les ouvrants ; sous la coupe, 6 mm et 14 mm sont cotés à partir de l'axe, X étant pris jusqu'à
l'axe.

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 114)

### Battement 0140 (p. 115)

![Coupe cotée du battement 0140 de 2008 entre deux ouvrants](/assets/profiles/systeme70/debit/debit-battement-0140-reperes-evo2008.png)

Le battement F90-25- 0140, coté 62 mm de large, est dessiné de la même façon, avec son renfort
hachuré ; jeu de 12⁺¹ mm entre les ouvrants ; sous la coupe, 6 mm et 23 mm cotés à partir de l'axe.
Un « +1 » isolé est imprimé contre l'ouvrant de gauche.

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 115)

### Battement extérieur 6130 et battement intérieur 6133 (p. 116)

![Battement 6130 et battement intérieur 6133 de 2008, ouvrants 6113 et 6112](/assets/profiles/systeme70/debit/debit-battement-6130-6133-ouvrants-6112-6113-evo2008.png)

Le battement extérieur F91-25- 6130 est clipsé sur l'ouvrant F91-08- 6113 (à gauche), le battement
intérieur F91-62- 6133 est posé sous la jonction, côté intérieur, et l'ouvrant F91-06- 6112 est à
droite. La DEO de l'ouvrant de droite part à 14 mm au-delà de l'axe, sous le battement intérieur ;
les DFO et les « Vitrage » sont cotés de part et d'autre de l'axe ; jeu de 12⁺¹ mm entre les
ouvrants.

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 116)

La p. 117 reprend la même coupe avec l'ouvrant F91-08- 6116 à gauche et l'ouvrant F91-06- 6115 à
droite, sous le même battement extérieur 6130 et le même battement intérieur 6133 ; cotes 14 et
12⁺¹ identiques, tableau identique à celui de la p. 116.

![Battement 6130 et battement intérieur 6133 de 2008, ouvrants 6116 et 6115](/assets/profiles/systeme70/debit/debit-battement-6130-6133-ouvrants-6115-6116-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 117)

### Battement extérieur 6132 et battement intérieur 6131 (p. 118 et 119)

Le battement extérieur F91-25- 6132, de dessus courbe comme les ouvrants dessinés, est clipsé sur
l'ouvrant de gauche ; le battement intérieur F91-62- 6131 est posé sous la jonction. P. 118 :
ouvrants F91-08- 6122 à gauche et F91-06- 6121 à droite ; p. 119 : ouvrants F91-08- 6124 à gauche
et F91-06- 6123 à droite. Sur les deux coupes, la DEO de l'ouvrant de droite part à 14 mm au-delà de
l'axe, jeu de 12⁺¹ mm entre les ouvrants ; les tableaux des deux planches sont identiques.

![Battement 6132 et battement intérieur 6131 de 2008, ouvrants 6122 et 6121](/assets/profiles/systeme70/debit/debit-battement-6132-6131-ouvrants-6121-6122-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 118)

![Battement 6132 et battement intérieur 6131 de 2008, ouvrants 6124 et 6123](/assets/profiles/systeme70/debit/debit-battement-6132-6131-ouvrants-6123-6124-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 119)

### Tableau des battements de 2008

Cotes à déduire de X, en mm, une ligne par dormant, une colonne par battement ; les planches des battements 6130 (p. 116, 117) et 6132 (p. 118, 119) portent les mêmes valeurs. Sur chaque planche
une colonne groupe plusieurs dormants sous une seule valeur (6100 et 6106 ; 6102, 6105 et 6107 ;
6101, 6104, 6108 et les 6109, 6110, 6111 « (non représenté) ») : chaque dormant a ici sa ligne. Les
vignettes portent la largeur de l'aile à côté de la flèche de la DHT : 30 (6102), 40 (6106,
6105), 60 (6107), 20 (6104, 6108). « - » : dormant absent de la planche.

<table>
<thead>
<tr><th rowspan="2">Dormant</th><th colspan="2">Battement 0141 (p. 114)</th><th colspan="2">Battement 0140 (p. 115)</th><th colspan="2">Battements 6130 et 6132 (p. 116 à 119)</th></tr>
<tr><th>DEO (mm)</th><th>DFO (mm)</th><th>DEO (mm)</th><th>DFO (mm)</th><th>DEO (mm)</th><th>DFO (mm)</th></tr>
</thead>
<tbody>
<tr><td>6100</td><td>X − 21</td><td>X − 61</td><td>X − 30</td><td>X − 70</td><td>X − 13</td><td>X − 53</td></tr>
<tr><td>6106</td><td>X − 21</td><td>X − 61</td><td>X − 30</td><td>X − 70</td><td>X − 13</td><td>X − 53</td></tr>
<tr><td>6102</td><td>X − 13</td><td>X − 53</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>6105</td><td>X − 13</td><td>X − 53</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>6107</td><td>X − 13</td><td>X − 53</td><td>X − 22</td><td>X − 62</td><td>X − 5</td><td>X − 45</td></tr>
<tr><td>6101</td><td>X − 30</td><td>X − 70</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6104</td><td>X − 30</td><td>X − 70</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6108</td><td>X − 30</td><td>X − 70</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6109</td><td>X − 30</td><td>X − 70</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6110</td><td>X − 30</td><td>X − 70</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>6111</td><td>X − 30</td><td>X − 70</td><td>X − 39</td><td>X − 79</td><td>X − 22</td><td>X − 62</td></tr>
<tr><td>2502</td><td>X − 51</td><td>X − 91</td><td>X − 60</td><td>X − 100</td><td>X − 43</td><td>X − 83</td></tr>
<tr><td>2403</td><td>-</td><td>-</td><td>X − 45</td><td>X − 85</td><td>-</td><td>-</td></tr>
</tbody>
</table>

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 114 à 119)

![En-tête du tableau du battement 0141 de 2008](/assets/profiles/systeme70/debit/debit-battement-0141-tableau-vignettes-evo2008.png)

![En-tête du tableau du battement 0140 de 2008, avec le dormant 2403](/assets/profiles/systeme70/debit/debit-battement-0140-tableau-vignettes-evo2008.png)

![En-tête du tableau du battement 6130 de 2008 (p. 116 et 117)](/assets/profiles/systeme70/debit/debit-battement-6130-tableau-vignettes-evo2008.png)

![En-tête du tableau du battement 6132 de 2008 (p. 118 et 119)](/assets/profiles/systeme70/debit/debit-battement-6132-tableau-vignettes-evo2008.png)

### Débit des battements de 2008

Formules imprimées sous chaque tableau ; `DEO` est la dimension extérieure d'ouvrant obtenue au
tableau.

| Battement | Débit imprimé | Page PDF |
| --- | --- | --- |
| 0141 | « Débit battement (0141) = DEO – 72 mm » | 114 |
| 0140 | « Débit battement (0141) = DEO – 72 mm » | 115 |
| 6130 (extérieur) | « Débit battement extérieur (6130) = DEO – 70 mm » | 116 |
| 6133 (intérieur) | « Débit battement intérieur (6133) = DEO – 6 mm » | 116 |
| 6130 (extérieur) | « Débit battement extérieur (6130) = DEO – 70 mm » | 117 |
| 6133 (intérieur) | « Débit battement intérieur (6133) = DEO – 6 mm » | 117 |
| 6132 (extérieur) | « Débit battement extérieur (6132) = DEO – 70 mm » | 118 |
| 6131 (intérieur) | « Débit battement intérieur (6131) = DEO – 6 mm » | 118 |
| 6132 (extérieur) | « Débit battement extérieur (6132) = DEO – 70 mm » | 119 |
| 6131 (intérieur) | « Débit battement intérieur (6131) = DEO – 6 mm » | 119 |

La planche du battement 0140 écrit « (0141) » dans sa ligne de débit (**INC-218**) ; le manuel de
2023 donne DEO − 72 mm pour le 0140. Les battements intérieurs 6133 et 6131 se débitent à DEO − 6 mm en 2008 et
à DEO − 12 mm au registre 2.3.1 du manuel de 2023 (**CTR-79**, voir aussi **INC-153**). Pour le
0140, le 6130 et le 6132, les cotes à déduire de X sont les mêmes qu'en 2023 ; le tableau de 2008 ajoute le
dormant 2403 sous le 0140 et n'a pas les dormants 6155, 6156 et 6158. Le battement 0141 n'a pas de
planche de débit dans le manuel de 2023 ; sur la planche du 6132 avec les ouvrants 6124 / 6123 (p. 119), le battement est légendé 6132, sans la légende « 61362 » ni le dessin du 6162 du manuel de 2023 (**INC-128**, **INC-129**) [2 p. 114-119].

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

[2] [Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008](raw/profine-plans-profiles-e-volution-2008-08.pdf), registre 3 « Côtes de débit », PDF p. 109-119 (pages imprimées 1 à 11)

# Voir aussi

- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
- [Plans des profilés e.VOLUTION, août 2008](/sources/plans-profiles-e-volution-2008.md)
- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Assemblages du système 70](/profiles/systeme-70-assemblages.md)
- [Plans de combinaison du système 70](/profiles/systeme-70-plans-de-combinaison.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
