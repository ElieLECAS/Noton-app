---
type: Profilé
title: Cotes de débit des dormants, meneaux et traverses, système 70 (e.VOLUTION 2008)
description: 'Cotes de débit du classeur e.VOLUTION de 2008 pour le système 70 : indications et exemple, dormants 6100, 6101 et 6104, meneaux et traverses.'
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

Cette page reprend, du manuel du système 70 Plateforme et du classeur e.VOLUTION d'août 2008, les cotes de débit de 2008 : indications et exemple, dormants, meneaux et traverses. Le contexte commun est dans [Cotes de débit du système 70 : dormants, meneaux, traverses et ouvrants](/profiles/systeme-70-cotes-de-debit.md).

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

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.1 « Cotes de débit/Coupes »,
p. 104 à 119 du PDF (pages imprimées 1 à 16)

[2] [Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008](raw/profine-plans-profiles-e-volution-2008-08.pdf), registre 3 « Côtes de débit », PDF p. 109-119 (pages imprimées 1 à 11)

# Voir aussi

Cette page est une partie d'un sujet qui en compte 4 ; les autres :

- [Cotes de débit du système 70 : dormants, meneaux, traverses et ouvrants](/profiles/systeme-70-cotes-de-debit.md)
- [Cotes de débit du système 70 : dormant recevant un battement et débit des battements](/profiles/systeme-70-cotes-de-debit-battements.md)
- [Cotes de débit des ouvrants et des battements, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-cotes-de-debit-ouvrants-battements-evo2008.md)
