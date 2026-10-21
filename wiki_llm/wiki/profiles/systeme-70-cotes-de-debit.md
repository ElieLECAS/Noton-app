---
type: Profilé
title: 'Cotes de débit du système 70 : dormants, meneaux, traverses et ouvrants'
description: Cotes de coupe à déduire pour débiter les dormants (6100, 6101, 6104), les meneaux, les traverses et les ouvrants du système 70, avec l'exemple de calcul du manuel.
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
  at: 2026-10-21T10:41:55Z
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
[Profilés et renforts du système 70](/profiles/systeme-70-dormants.md#cotes-des-dormants).

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

Autres parties de ce sujet :

- [Cotes de débit du système 70 : dormant recevant un battement et débit des battements](/profiles/systeme-70-cotes-de-debit-battements.md)
- [Cotes de débit des dormants, meneaux et traverses, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-cotes-de-debit-dormants-meneaux-evo2008.md)
- [Cotes de débit des ouvrants et des battements, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-cotes-de-debit-ouvrants-battements-evo2008.md)
