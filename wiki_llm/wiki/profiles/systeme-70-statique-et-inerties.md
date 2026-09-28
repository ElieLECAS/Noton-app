---
type: Profilé
title: Statique et moments d'inertie du système 70
description: La méthode de calcul statique du renforcement des menuiseries du système 70 Plateforme (formule générale, pressions des classes V selon l'EN 12211, allèges, modules d'élasticité, flèches admissibles, plans de charge), les moments d'inertie des fers plats et des tubes acier, les valeurs statiques Iw des meneaux et des assemblages de dormants, l'exemple d'application et les quinze tables des moments d'inertie requis.
tags: [profine, systeme-70, statique, inertie, iw, ig, vent, fleche, renfort, meneau, traverse, plan-de-charge, allege, accouplement, contreventement]
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
    pages: 147-165
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 138
generated:
  by: process:claude-code
  at: 2026-09-28T18:30:00Z
---

# Ce que règle le calcul statique

Le calcul statique fixe le moment d'inertie minimal du renfort d'acier d'un élément de
menuiserie PVC du système 70 Plateforme. Le moment d'inertie (I, en cm⁴) mesure la rigidité
d'une section à la flexion : plus il est grand, moins l'élément fléchit sous la pression du vent.
Le renfort est le profilé d'acier glissé dans la chambre du profilé PVC
([glossaire](/reference/glossaire.md)). Section 2.3.4 du manuel
[Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md), PDF p. 147 à 180.

Les quinze tables des moments d'inertie requis, une par classement au vent, sont sur leurs pages :

| Classement | Pression (Pa) | Flèche admissible | Page PDF | Page du wiki |
| --- | --- | --- | --- | --- |
| V\*A1 | 400 | 1/150 | 166 | [Moments d'inertie requis, V\*A1](/profiles/systeme-70-inerties-va1.md) |
| V\*A2 | 800 | 1/150 | 167 | [Moments d'inertie requis, V\*A2](/profiles/systeme-70-inerties-va2.md) |
| V\*A3 | 1200 | 1/150 | 168 | [Moments d'inertie requis, V\*A3](/profiles/systeme-70-inerties-va3.md) |
| V\*A4 | 1600 | 1/150 | 169 | [Moments d'inertie requis, V\*A4](/profiles/systeme-70-inerties-va4.md) |
| V\*A5 | 2000 | 1/150 | 170 | [Moments d'inertie requis, V\*A5](/profiles/systeme-70-inerties-va5.md) |
| V\*B1 | 400 | 1/200 | 171 | [Moments d'inertie requis, V\*B1](/profiles/systeme-70-inerties-vb1.md) |
| V\*B2 | 800 | 1/200 | 172 | [Moments d'inertie requis, V\*B2](/profiles/systeme-70-inerties-vb2.md) |
| V\*B3 | 1200 | 1/200 | 173 | [Moments d'inertie requis, V\*B3](/profiles/systeme-70-inerties-vb3.md) |
| V\*B4 | 1600 | 1/200 | 174 | [Moments d'inertie requis, V\*B4](/profiles/systeme-70-inerties-vb4.md) |
| V\*B5 | 2000 | 1/200 | 175 | [Moments d'inertie requis, V\*B5](/profiles/systeme-70-inerties-vb5.md) |
| V\*C1 | 400 | 1/300 | 176 | [Moments d'inertie requis, V\*C1](/profiles/systeme-70-inerties-vc1.md) |
| V\*C2 | 800 | 1/300 | 177 | [Moments d'inertie requis, V\*C2](/profiles/systeme-70-inerties-vc2.md) |
| V\*C3 | 1200 | 1/300 | 178 | [Moments d'inertie requis, V\*C3](/profiles/systeme-70-inerties-vc3.md) |
| V\*C4 | 1600 | 1/300 | 179 | [Moments d'inertie requis, V\*C4](/profiles/systeme-70-inerties-vc4.md) |
| V\*C5 | 2000 | 1/300 | 180 | [Moments d'inertie requis, V\*C5](/profiles/systeme-70-inerties-vc5.md) |

# Méthode de calcul

## 1. Généralités

Le renforcement des menuiseries est destiné à assurer la rigidité des éléments PVC, en fonction
de la résistance nécessaire recherchée, suivant les critères ci-dessous [1 p. 147] :

- résistance à la déformation sous une pression reproduisant les effets du vent : cas des parties
  ouvrantes, cas des éléments de l'ossature tels que meneaux, traverses d'impostes ;
- résistance aux chocs dits « chocs de sécurité » : cas de traverses d'allège ;
- résistance aux déformations dues au mode de pose : cas des dormants posés sur doublages.

## 2.1 Formule générale

I = (5 × Pr × S × L³ × 10) / (384 × E × f) [1 p. 147]

| Symbole | Grandeur | Unité |
| --- | --- | --- |
| I | moment d'inertie minimum admissible du renforcement | cm⁴ |
| Pr | pression à appliquer | Pa |
| S | surface du plan de charge | m² |
| L³ | longueur de l'élément à renforcer élevée au cube | m³ |
| E | module d'élasticité à la flexion du renforcement | kgf/mm² |
| f | flèche maximale admissible | m |

Nota : la fraction 5/384 correspond à un coefficient relatif à une charge uniformément répartie.
La valeur 10 au numérateur sert à balancer la formule compte tenu des unités prises [1 p. 147].

## 2.2.1 Valeur de I

Le renforcement proposé devra avoir un moment d'inertie égal ou supérieur à la valeur de I
calculée selon la formule ci-avant. Pour les cas de renforcement d'un même élément avec plusieurs
renforts, les moments d'inertie de ces renforts s'additionnent [1 p. 147].

## 2.2.2 Valeur de Pr

La pression Pr est à considérer appliquée depuis l'extérieur de la menuiserie et
perpendiculairement à celle-ci. Lorsqu'il est demandé un classement « V » précis (selon la norme
EN 12211), Pr se lit sur le tableau suivant ; P1, P2 et P3 sont les trois pressions d'essai de la
classe, en Pa [1 p. 148].

| Classe | P1 (Pa) | P2 (Pa) | P3 (Pa) |
| --- | --- | --- | --- |
| 0 | pas d'essai | pas d'essai | pas d'essai |
| 1 | 400 | 200 | 600 |
| 2 | 800 | 400 | 1200 |
| 3 | 1200 | 600 | 1800 |
| 4 | 1600 | 800 | 2400 |
| 5 | 2000 | 1000 | 3000 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 148)

Lorsqu'il n'est pas demandé de classement « V », il convient de se référer à la FD P 20-201
« choix des fenêtres en fonction de leur exposition » (mémento), en prenant les valeurs indiquées
au tableau 2 de sa page 9 après en avoir déterminé les paramètres : hauteur des fenêtres au-dessus
du sol, zone et situation [1 p. 148]. Le classement A\*E\*V est expliqué sur
[Labels et certifications](/certifications/labels-et-certifications.md).

Cas particulier des allèges devant assurer la sécurité des personnes. L'allège est la
partie basse, pleine ou vitrée, d'une baie. Pour ce cas, la pression Pr est remplacée par la
valeur Q correspondant à la charge [1 p. 148] :

| Cas | Q (kg/m) |
| --- | --- |
| allège devant assurer totalement la sécurité des personnes | 100 |
| allège avec une autre protection jumelée, telle que garde-corps ou allège maçonnée | 75 |

La formule initiale devient alors : I = (5 × Q × L⁴ × 100) / (384 × E × f) [1 p. 148].

## 2.2.3 Valeur de S

La valeur de S est variable, en fonction des dimensions de la menuiserie et du positionnement,
dans la menuiserie, de l'élément dont on cherche le renforcement [1 p. 148]. Les plans de charge
sont dessinés plus bas.

## 2.2.4 Valeur de E

Le module d'élasticité à la flexion est variable en fonction de la matière propre au renfort
[1 p. 148] :

| Matière du renfort | E (N/mm²) | E (kgf/mm²) |
| --- | --- | --- |
| acier | 210 000 | 21 000 |
| aluminium | 70 000 | 7 000 |
| bois massif | 10 000 | 1 000 |

Nota : pour le PVC, la valeur de E étant relativement peu élevée (2 500 N/mm²), il n'est pas
introduit dans les calculs des moments d'inertie propres aux profils PVC. Seuls sont considérés
ceux des renforts [1 p. 148].

## 2.2.5 Valeur de f

La flèche f est variable en fonction du positionnement, dans la menuiserie, de l'élément dont on
cherche le renforcement [1 p. 150] :

| Élément | Flèche admissible f |
| --- | --- |
| parties ouvrantes (montant de rive, montant milieu, traverse d'ouvrant) | L/150 |
| éléments du dormant (meneau, traverse d'imposte ou intermédiaire) | L/150 |
| cas particulier des traverses d'allège | L/300 |

Les tables des moments d'inertie requis existent aussi pour des flèches de 1/200 (classements V\*B)
et 1/300 (classements V\*C) ; aucun texte de la section ne dit à quels éléments elles s'appliquent.

# Plans de charge

Le plan de charge est la part de la surface de la menuiserie dont la pression du vent est
reprise par un élément donné. Sur chaque schéma, la surface hachurée est le plan de charge de
l'élément ; L et H sont la largeur et la hauteur, L¹, L², H¹, H² les largeurs et hauteurs des
parties [1 p. 149]. Légende : S 1 = plan de charge d'un montant de rive (cas 1), d'un meneau
(cas 2) ; S 2 = plan de charge d'une traverse ; S 3 = plan de charge des montants milieu.

## Cas 1 : châssis, croisées et portes-croisées à 1 ou 2 vantaux

![Plans de charge, cas 1 : châssis, croisées et portes-croisées à 1 ou 2 vantaux](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-1-chassis.png)

À un vantail (à gauche), le montant de rive reprend une bande de largeur L/2 bornée à L/2 des
angles (S 1), la traverse un triangle (S 2) ; à deux vantaux (à droite), les montants de rive et
les montants milieu (S 3) reprennent chacun L/4.

## Cas 2 : meneaux

![Plans de charge, cas 2 : un seul meneau](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-2-un-meneau.png)

Cas avec un seul meneau : le plan de charge S 1 du meneau est la moitié de chaque partie
(L¹/2 et L²/2 à gauche, H/2 de part et d'autre du milieu à droite quand les parties sont larges).

![Plans de charge, cas 2 : deux meneaux ou plus](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-2-deux-meneaux.png)

Cas avec deux meneaux ou plus : chaque meneau reprend la moitié des parties voisines (L¹/2, L²/2) ;
le second plan de charge est repéré « S 1 bis ».

## Cas 3 : traverse intermédiaire (imposte ou allège)

![Plans de charge, cas 3 : traverse intermédiaire](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-3-traverse-intermediaire.png)

La traverse reprend H¹/2 et H²/2 de part et d'autre (S 2), la charge étant arrêtée à 45° aux
extrémités.

## Cas 4 : meneau avec traverse intermédiaire (composé)

![Plans de charge, cas 4 : meneau avec traverse intermédiaire](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-4-meneau-compose.png)

Le meneau reprend un rectangle de largeur L¹/2 + L²/2 sur toute la hauteur H (S 1).

## Cas 5 : traverse intermédiaire avec meneaux composés

![Plans de charge, cas 5 : traverse intermédiaire avec meneaux composés](/assets/procedures/moe-systeme-70/statique/plan-de-charge-cas-5-traverse-meneaux-composes.png)

La traverse, arrêtée sur les meneaux, reprend H¹/2 et H²/2 sur la seule largeur L¹ (S 2).

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 149)

## Exemples de plans de charge pour traverse et meneau

Remarque : a ou b < L/2 [1 p. 165]. Les traits épais sont les éléments filants ou courts, les
cercles leurs appuis, les tiretés les bissectrices à 45° qui délimitent les plans de charge.

![Meneau filant et traverse](/assets/procedures/moe-systeme-70/statique/plans-de-charge-meneau-filant-et-traverse.png)

Un meneau filant (qui va d'un bout à l'autre du dormant) et une traverse arrêtée sur lui : le
meneau reprend les largeurs de charge a et b tracées à 45° ; la part repérée par un astérisque
« Ce plan de charge n'est pas pris en compte ».

![Deux meneaux filants et traverses courtes](/assets/procedures/moe-systeme-70/statique/plans-de-charge-meneaux-filants-traverses-courtes.png)

Deux meneaux filants et trois traverses courtes : les plans de charge des meneaux sont hachurés.

![Traverses courtes entre meneaux filants](/assets/procedures/moe-systeme-70/statique/plans-de-charge-traverses-courtes-entre-meneaux-filants.png)

Même composition, plans de charge des traverses courtes hachurés.

![Traverse filante sous coffre de volet roulant](/assets/procedures/moe-systeme-70/statique/plans-de-charge-traverse-filante-sous-cvr.png)

Traverse filante sous CVR (coffre de volet roulant), deux meneaux et trois traverses : la
traverse filante reprend le trapèze hachuré.

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 165)

# Exemples d'application de la formule générale

## Exemple n° 1 : porte-croisée à 3 vantaux

![Porte-croisée à 3 vantaux de l'exemple n° 1](/assets/procedures/moe-systeme-70/statique/exemple-1-porte-croisee-3-vantaux.png)

Porte-croisée de 2,1 m de large (0,7 m de fixe F et 1,4 m d'ouvrant) sur 2,15 m de haut ;
classement demandé V 2. Calcul de renforcement du meneau [1 p. 150] :

| Grandeur | Valeur imprimée |
| --- | --- |
| Pr | 800 Pa |
| S 1 | 1,645 m² |
| L³ | 2,15³ = 9,94 |
| E | acier 21 000 kgf/mm² |
| f | L/150 = 2,15/150 = 0,014 m |
| I | (5 × 300 × 1,645 × 9,94 × 10) / (384 × 21 000 × 0,014) = 5,79 cm⁴ |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 150)

**Le calcul imprimé porte 300 au numérateur alors que Pr vaut 800 Pa ; le résultat 5,79 cm⁴ est
celui obtenu avec 800 — entrée INC-139**.

## Exemple n° 2 : croisée à 3 vantaux sur allège

![Croisée à 3 vantaux sur allège de l'exemple n° 2](/assets/procedures/moe-systeme-70/statique/exemple-2-croisee-3-vantaux-sur-allege.png)

Croisée de 2,2 m de large, 1,25 m de vitrage sur 0,9 m d'allège pleine (2,15 m au total) ; pas de
classement V demandé ; hauteur au-dessus du sol 28 à 50 m, région A, situation b (périphérie d'un
grand centre urbain). Calcul de renforcement de la traverse d'allège [1 p. 150] :

| Grandeur | Valeur imprimée |
| --- | --- |
| Q | 100 kg/m |
| L⁴ | 2,20⁴ = 23,42 |
| E | acier 21 000 kgf/mm² |
| f | L/300 = 2,20/300 = 0,073 m, limité à 0,010 m |
| I | (5 × 100 × 23,43 × 100) / (384 × 21 000 × 0,010) = 14,52 cm⁴ |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 150)

**2,20/300 vaut 0,0073 m et non 0,073 m ; le calcul emploie 0,010 m, et L⁴ est écrit 23,42 puis
23,43 — entrée INC-140**.

# Moments d'inertie des fers plats

Un fer plat est une barre d'acier de section rectangulaire, de largeur B et de hauteur H (en
mm). Iw est son inertie autour de l'axe W-W, Ig autour de l'axe G-G [1 p. 151].

![Fer plat, axes W-W et G-G](/assets/procedures/moe-systeme-70/statique/fer-plat-iw-ig.png)

Formule : Iw = B · H³ / 12 ; Ig = H · B³ / 12. La table donne le moment d'inertie en cm⁴ ; une
ligne par hauteur H, une colonne par largeur B (en mm).

<table>
<thead>
<tr><th rowspan="2">H (mm)</th><th colspan="12">B (mm)</th></tr>
<tr><th>2</th><th>3</th><th>4</th><th>5</th><th>6</th><th>7</th><th>8</th><th>9</th><th>10</th><th>12</th><th>14</th><th>16</th></tr>
</thead>
<tbody>
<tr><th>20</th><td>0,13</td><td>0,20</td><td>0,27</td><td>0,33</td><td>0,40</td><td>0,47</td><td>0,53</td><td>0,60</td><td>0,67</td><td>0,80</td><td>0,93</td><td>1,07</td></tr>
<tr><th>25</th><td>0,26</td><td>0,39</td><td>0,52</td><td>0,65</td><td>0,78</td><td>0,91</td><td>1,04</td><td>1,17</td><td>1,30</td><td>1,56</td><td>1,82</td><td>2,08</td></tr>
<tr><th>30</th><td>0,45</td><td>0,68</td><td>0,90</td><td>1,13</td><td>1,35</td><td>1,58</td><td>1,80</td><td>2,03</td><td>2,25</td><td>2,70</td><td>3,15</td><td>3,60</td></tr>
<tr><th>35</th><td>0,71</td><td>1,07</td><td>1,43</td><td>1,79</td><td>2,14</td><td>2,50</td><td>2,86</td><td>3,22</td><td>3,57</td><td>4,29</td><td>5,00</td><td>5,72</td></tr>
<tr><th>40</th><td>1,07</td><td>1,60</td><td>2,13</td><td>2,67</td><td>3,20</td><td>3,73</td><td>4,27</td><td>4,80</td><td>5,33</td><td>6,40</td><td>7,47</td><td>8,53</td></tr>
<tr><th>45</th><td>1,52</td><td>2,28</td><td>3,04</td><td>3,80</td><td>4,56</td><td>5,32</td><td>6,08</td><td>6,83</td><td>7,59</td><td>9,11</td><td>10,63</td><td>12,15</td></tr>
<tr><th>50</th><td>2,08</td><td>3,13</td><td>4,17</td><td>5,21</td><td>6,25</td><td>7,29</td><td>8,33</td><td>9,38</td><td>10,42</td><td>12,50</td><td>14,58</td><td>16,67</td></tr>
<tr><th>55</th><td>2,77</td><td>4,16</td><td>5,55</td><td>6,93</td><td>8,32</td><td>9,71</td><td>11,09</td><td>12,48</td><td>13,86</td><td>16,64</td><td>19,41</td><td>22,18</td></tr>
<tr><th>60</th><td>3,60</td><td>5,40</td><td>7,20</td><td>9,00</td><td>10,80</td><td>12,60</td><td>14,40</td><td>16,20</td><td>18,00</td><td>21,60</td><td>25,20</td><td>28,80</td></tr>
<tr><th>65</th><td>4,58</td><td>6,87</td><td>9,15</td><td>11,44</td><td>13,73</td><td>16,02</td><td>18,31</td><td>20,60</td><td>22,89</td><td>27,46</td><td>32,04</td><td>36,62</td></tr>
<tr><th>70</th><td>5,72</td><td>8,57</td><td>11,43</td><td>14,29</td><td>17,15</td><td>20,01</td><td>22,87</td><td>25,73</td><td>28,58</td><td>34,30</td><td>40,02</td><td>45,73</td></tr>
<tr><th>75</th><td>7,03</td><td>10,55</td><td>14,06</td><td>17,58</td><td>21,09</td><td>24,61</td><td>28,13</td><td>31,64</td><td>35,16</td><td>42,19</td><td>49,22</td><td>56,25</td></tr>
<tr><th>80</th><td>8,53</td><td>12,80</td><td>17,07</td><td>21,33</td><td>25,60</td><td>29,87</td><td>34,13</td><td>38,40</td><td>42,67</td><td>51,20</td><td>59,73</td><td>68,27</td></tr>
<tr><th>85</th><td>10,24</td><td>15,35</td><td>20,47</td><td>25,59</td><td>30,71</td><td>35,82</td><td>40,94</td><td>46,06</td><td>51,18</td><td>61,41</td><td>71,65</td><td>81,88</td></tr>
<tr><th>90</th><td>12,15</td><td>18,23</td><td>24,30</td><td>30,38</td><td>36,45</td><td>42,53</td><td>48,60</td><td>54,68</td><td>60,75</td><td>72,90</td><td>85,05</td><td>97,20</td></tr>
<tr><th>95</th><td>14,29</td><td>21,43</td><td>28,58</td><td>35,72</td><td>42,87</td><td>50,01</td><td>57,16</td><td>64,30</td><td>71,45</td><td>85,74</td><td>100,03</td><td>114,32</td></tr>
<tr><th>100</th><td>16,67</td><td>25,00</td><td>33,33</td><td>41,67</td><td>50,00</td><td>58,33</td><td>66,67</td><td>75,00</td><td>83,33</td><td>100,00</td><td>116,67</td><td>133,33</td></tr>
</tbody>
</table>

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 151)

# Moments d'inertie des tubes en acier

« Valeurs pour profilés avec coins non arrondis. Les angles arrondis donnent des valeurs
légèrement plus faibles. » [1 p. 152-153]

## Tubes rectangulaires

![Tube rectangulaire en acier, cotes et axes](/assets/procedures/moe-systeme-70/statique/tube-acier-iw-ig.png)

Tube de dimensions extérieures a × b (H × B sur le schéma), intérieures h × b, d'épaisseur s.
Formule imprimée : Iw = H · B³/12 − h · b³/12 [cm⁴] ; Ig = B · H³/12 − b · h³/12 [cm⁴]. Une ligne
par tube et par épaisseur ; la source imprime les décimales avec un point, rendues ici avec une
virgule [1 p. 152].

| Tube a × b (mm) | Épaisseur s (mm) | Iw (cm⁴) | Ig (cm⁴) |
| --- | --- | --- | --- |
| 40 x 20 | 2 | 4,31 | 1,41 |
| 40 x 20 | 3 | 5,21 | 1,68 |
| 40 x 30 | 2 | 5,76 | 3,65 |
| 40 x 30 | 3 | 7,27 | 4,60 |
| 50 x 20 | 2 | 7,65 | 1,73 |
| 50 x 30 | 2 | 9,95 | 4,44 |
| 50 x 30 | 2,9 | 13,4 | 5,88 |
| 50 x 30 | 3 | 13,8 | 6,02 |
| 50 x 30 | 4 | 16,9 | 7,25 |
| 50 x 40 | 2 | 12,3 | 8,65 |
| 55 x 34 | 2 | 13,7 | 6,45 |
| 60 x 30 | 2 | 15,6 | 5,23 |
| 60 x 40 | 2 | 19,0 | 10,1 |
| 60 x 40 | 2,9 | 26,0 | 13,7 |
| 60 x 40 | 3 | 24,6 | 13,1 |
| 60 x 40 | 4 | 33,3 | 17,1 |
| 70 x 30 | 2,9 | 31,5 | 8,01 |
| 70 x 30 | 4 | 40,4 | 9,97 |
| 70 x 40 | 2 | 27,7 | 11,5 |
| 70 x 40 | 3 | 39,1 | 16,1 |
| 70 x 40 | 4 | 45,9 | 18,9 |
| 80 x 40 | 2 | 38,4 | 13,0 |
| 80 x 40 | 2,5 | 46,8 | 15,7 |
| 80 x 40 | 2,9 | 53,1 | 17,7 |
| 80 x 40 | 3 | 50,7 | 17,2 |
| 80 x 40 | 3,6 | 63,4 | 20,8 |
| 80 x 40 | 4 | 64,8 | 21,5 |
| 80 x 40 | 4,5 | 75,5 | 24,5 |
| 80 x 60 | 2,5 | 61,8 | 39,5 |
| 80 x 60 | 3,2 | 73,8 | 47,3 |
| 80 x 60 | 4 | 87,9 | 56,1 |
| 80 x 60 | 5 | 99,4 | 63,7 |
| 90 x 50 | 3,2 | 89,7 | 35,5 |
| 90 x 50 | 4 | 108 | 42,3 |
| 90 x 50 | 5,6 | 140 | 53,8 |
| 100 x 40 | 2 | 67,1 | 15,9 |
| 100 x 40 | 3 | 96,1 | 22,3 |
| 100 x 40 | 4 | 116 | 26,7 |
| 100 x 50 | 3 | 106 | 36,1 |
| 100 x 50 | 3,6 | 129 | 42,9 |
| 100 x 50 | 4 | 134 | 44,9 |
| 100 x 50 | 4,5 | 155 | 50,9 |
| 100 x 50 | 5 | 152 | 51,2 |
| 100 x 50 | 5,6 | 184 | 59,4 |

| Tube a × b (mm) | Épaisseur s (mm) | Iw (cm⁴) | Ig (cm⁴) |
| --- | --- | --- | --- |
| 100 x 60 | 3 | 124 | 56,0 |
| 100 x 60 | 3,6 | 146 | 65,2 |
| 100 x 60 | 4 | 159 | 71,0 |
| 100 x 60 | 4,5 | 176 | 77,9 |
| 100 x 60 | 5 | 174 | 78,9 |
| 100 x 60 | 6,3 | 228 | 99,6 |
| 110 x 60 | 3,6 | 182 | 70,2 |
| 110 x 60 | 4,5 | 219 | 83,7 |
| 120 x 60 | 3 | 189 | 64,4 |
| 120 x 60 | 4 | 247 | 82,7 |
| 120 x 60 | 5 | 296 | 98,2 |
| 120 x 60 | 6,3 | 354 | 116 |
| 120 x 80 | 3 | 230 | 123 |
| 120 x 80 | 4 | 300 | 160 |
| 120 x 80 | 5 | 362 | 192 |
| 120 x 80 | 6 | 393 | 210 |
| 120 x 80 | 6,3 | 435 | 229 |
| 140 x 70 | 4 | 379 | 130 |
| 140 x 70 | 5 | 450 | 153 |
| 140 x 70 | 6,3 | 529 | 179 |
| 140 x 80 | 4 | 438 | 183 |
| 140 x 80 | 5 | 529 | 220 |
| 140 x 80 | 5,6 | 582 | 241 |
| 140 x 80 | 6,3 | 639 | 263 |
| 140 x 80 | 7,1 | 702 | 287 |
| 160 x 80 | 4 | 614 | 207 |
| 160 x 80 | 5 | 735 | 247 |
| 160 x 90 | 5 | 782 | 319 |
| 160 x 90 | 6,3 | 943 | 383 |
| 160 x 90 | 8 | 1130 | 455 |
| 180 x 100 | 4 | 926 | 374 |
| 180 x 100 | 5 | 1120 | 451 |
| 180 x 100 | 5,6 | 1240 | 496 |
| 180 x 100 | 6,3 | 1310 | 527 |
| 180 x 100 | 7,1 | 1500 | 597 |
| 180 x 100 | 8,8 | 1760 | 696 |
| 200 x 120 | 6,3 | 2010 | 910 |
| 200 x 120 | 10 | 2890 | 1290 |
| 220 x 120 | 6,3 | 2540 | 992 |
| 220 x 120 | 10 | 3680 | 1410 |
| 260 x 140 | 8 | 5220 | 1990 |
| 260 x 140 | 10 | 6260 | 2370 |
| 260 x 180 | 8 | 6240 | 3540 |
| 260 x 180 | 10 | 7510 | 4240 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 152)

Quatre lignes donnent une inertie plus faible pour une épaisseur plus forte (60 × 40 :
26,0 à 2,9 mm, 24,6 à 3 mm ; 80 × 40 : 53,1 à 2,9 mm, 50,7 à 3 mm ; 100 × 50 : 155 à 4,5 mm,
152 à 5 mm ; 100 × 60 : 176 à 4,5 mm, 174 à 5 mm) — entrée **VER-90**.

## Tubes carrés

![Tube carré en acier, cotes et axes](/assets/procedures/moe-systeme-70/statique/tube-carre-acier-iw-ig.png)

Formule imprimée : Iw = B · H³/12 − b · h³/12 [cm⁴] ; Ig = H · B³/12 − h · b³/12 [cm⁴] ; pour un
tube carré, Iw et Ig sont égaux et la table n'en donne qu'une valeur [1 p. 153].

| Tube a × b (mm) | Épaisseur s (mm) | Iw = Ig (cm⁴) |
| --- | --- | --- |
| 30 x 30 | 1,5 | 2,32 |
| 30 x 30 | 2 | 2,94 |
| 30 x 30 | 2,5 | 3,49 |
| 30 x 30 | 3 | 3,98 |
| 40 x 40 | 1,5 | 5,64 |
| 40 x 40 | 2 | 7,21 |
| 40 x 40 | 2,5 | 8,63 |
| 40 x 40 | 2,9 | 9,66 |
| 40 x 40 | 3 | 9,91 |
| 40 x 40 | 4 | 12,1 |
| 45 x 45 | 2 | 10,6 |
| 45 x 45 | 3 | 13,4 |
| 50 x 50 | 2 | 14,6 |
| 50 x 50 | 2,5 | 17,6 |
| 50 x 50 | 2,9 | 19,8 |
| 50 x 50 | 3 | 20,4 |
| 50 x 50 | 4 | 25,4 |
| 60 x 60 | 2 | 25,7 |
| 60 x 60 | 2,5 | 31,3 |
| 60 x 60 | 2,9 | 35,5 |
| 60 x 60 | 3 | 36,5 |
| 60 x 60 | 4 | 45,9 |
| 60 x 60 | 5 | 54,1 |
| 65 x 65 | 5 | 63,7 |
| 70 x 70 | 3 | 59,4 |
| 70 x 70 | 3,2 | 62,7 |
| 70 x 70 | 4 | 75,3 |
| 70 x 70 | 5 | 89,6 |
| 70 x 70 | 6 | 91,4 |
| 80 x 80 | 3 | 90,2 |
| 80 x 80 | 3,6 | 106 |
| 80 x 80 | 4 | 111 |
| 80 x 80 | 4,5 | 127 |
| 80 x 80 | 5,6 | 151 |
| 90 x 90 | 3 | 127 |
| 90 x 90 | 3,6 | 153 |
| 90 x 90 | 4 | 162 |
| 90 x 90 | 4,5 | 185 |
| 90 x 90 | 5,6 | 220 |
| 100 x 100 | 3 | 175 |
| 100 x 100 | 4 | 233 |
| 100 x 100 | 5 | 281 |
| 100 x 100 | 6,3 | 339 |

| Tube a × b (mm) | Épaisseur s (mm) | Iw = Ig (cm⁴) |
| --- | --- | --- |
| 110 x 110 | 3 | 241 |
| 110 x 110 | 4 | 311 |
| 110 x 110 | 5 | 368 |
| 110 x 110 | 6,3 | 453 |
| 120 x 120 | 4 | 411 |
| 120 x 120 | 4,5 | 452 |
| 120 x 120 | 5 | 493 |
| 120 x 120 | 5,6 | 544 |
| 120 x 120 | 6,3 | 598 |
| 125 x 125 | 4 | 457 |
| 125 x 125 | 5 | 552 |
| 125 x 125 | 6 | 641 |
| 140 x 140 | 4 | 651 |
| 140 x 140 | 5 | 789 |
| 140 x 140 | 5,6 | 885 |
| 140 x 140 | 7,1 | 1080 |
| 140 x 140 | 8,8 | 1280 |
| 150 x 150 | 4 | 808 |
| 150 x 150 | 5 | 981 |
| 150 x 150 | 6 | 1150 |
| 160 x 160 | 4 | 991 |
| 160 x 160 | 5 | 1200 |
| 160 x 160 | 5,6 | 1330 |
| 160 x 160 | 6,3 | 1460 |
| 160 x 160 | 7,1 | 1610 |
| 160 x 160 | 8,8 | 1910 |
| 160 x 160 | 10 | 2100 |
| 180 x 180 | 5 | 1700 |
| 180 x 180 | 6,3 | 2120 |
| 180 x 180 | 7,1 | 2280 |
| 180 x 180 | 8,8 | 2800 |
| 180 x 180 | 10 | 3090 |
| 200 x 200 | 8 | 3620 |
| 200 x 200 | 10 | 4340 |
| 220 x 220 | 8 | 4890 |
| 220 x 220 | 10 | 5890 |
| 260 x 260 | 8,8 | 8980 |
| 260 x 260 | 11 | 10830 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 153)

Les formules des tubes carrés échangent Iw et Ig par rapport à celles des tubes rectangulaires —
entrée **INC-143**.

# Valeurs statiques des meneaux et traverses

Valeur statique Iw (inertie au vent, en cm⁴) du renfort de chaque meneau/traverse, telle
qu'imprimée sur la planche [1 p. 154-155]. Les cotes des meneaux et de leurs renforts sont sur
[Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md).

| Profilé | Renfort | Iw (cm⁴) | Cotes portées sur la planche (mm) | Coupe |
| --- | --- | --- | --- | ---: |
| 6127 | V603 | 3,2 | meneau 70 × 80, chambre 40 ; renfort 36 × 24, épaisseur 2 | ![Meneau 6127 et renfort V603](/assets/procedures/moe-systeme-70/statique/meneau-6127-v603.png) |
| 6157 | V010 | 7,0 | meneau 70 × 80 (20 + 40 + 20) ; renfort 47,5 × 19,8, épaisseur 2,5 | ![Meneau 6157 et renfort V010](/assets/procedures/moe-systeme-70/statique/meneau-6157-v010.png) |
| 6157 (titre de la planche) | V290 | 8,7 | profilé dessiné 70 × 115, chambre 75 ; renfort 40 × 50, épaisseur 2 | ![Profilé de 115 mm et renfort V290, titré 6157](/assets/procedures/moe-systeme-70/statique/meneau-6157-v290.png) |
| 2425 | 9132 | 9,1 | meneau 70 × 90, chambre 50 ; renfort 48 × 25, épaisseur 2,5 | ![Meneau 2425 et renfort 9132](/assets/procedures/moe-systeme-70/statique/meneau-2425-9132.png) |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 154-155)

**La troisième planche est titrée 6157 alors que le profilé dessiné mesure 115 mm, comme le 2427
qui porte le V290 (8,7 cm⁴) à la PDF p. 158 — entrée INC-142. Le V603 du 6127 vaut
3,2 cm⁴ ici et 3,0 cm⁴ dans l'exemple d'application (PDF p. 161) — entrée INC-144**.

# Valeurs statiques des assemblages

Quand deux profilés sont accouplés, les inerties de leurs renforts s'additionnent : la « valeur
statique » est la somme imprimée sous chaque planche [1 p. 156-158].

## Assemblages de dormants

Deux dormants accouplés dos à dos, un isolant entre eux, les cercles repérant les liaisons
[1 p. 156].

| Assemblage | Profilé | Renfort | Iw (cm⁴) | Valeur statique (cm⁴) | Coupe |
| --- | --- | --- | --- | --- | ---: |
| dormants 6100, hauteur 110 | 6100 (× 2) | V600 (× 2) | 2,1 + 2,1 | 4,2 | ![Accouplement de deux dormants 6100 avec V600](/assets/procedures/moe-systeme-70/statique/accouplement-dormants-6100-v600.png) |
| dormants 6101, hauteur 128 | 6101 (× 2) | V601 (× 2) | 3,2 + 3,2 | 6,4 | ![Accouplement de deux dormants 6101 avec V601](/assets/procedures/moe-systeme-70/statique/accouplement-dormants-6101-v601.png) |
| dormants 2502, hauteur 170 | 2502 (× 2) | V030 (× 2) | 4,5 + 4,5 | 9,0 | ![Accouplement de deux dormants 2502 avec V030](/assets/procedures/moe-systeme-70/statique/accouplement-dormants-2502-v030.png) |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 156)

## Assemblages de dormants avec contreventement

Le contreventement est un profilé renforcé placé entre deux dormants accouplés pour en
augmenter la rigidité [1 p. 157].

| Assemblage | Profilés et renforts (Iw, cm⁴) | Valeur statique (cm⁴) | Cotes portées (mm) | Coupe |
| --- | --- | --- | --- | ---: |
| 2 × 6101 + 70602 | 6101 V601 3,2 ; 6101 V601 3,2 ; 70602 V288 20,3 | 26,7 | 98 (70 + 20 sous le dormant) ; 31 + 46 + 31 = 108 | ![Contreventement 70602 et V288 entre deux dormants 6101](/assets/procedures/moe-systeme-70/statique/contreventement-6101-70602-v288.png) |
| 2 × 6101 + 93000 | 6101 V601 3,2 ; 6101 V601 3,2 ; 93000 V250 71,6 | 78,0 | 14 ; 126,5 (70 + 56,5) ; 64,5 + 25 + 64,5 = 154 | ![Contreventement 93000 et V250 entre deux dormants 6101](/assets/procedures/moe-systeme-70/statique/contreventement-6101-93000-v250.png) |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 157)

## Assemblages de meneau avec contreventement

| Assemblage | Profilés et renforts (Iw, cm⁴) | Valeur statique (cm⁴) | Cotes portées (mm) | Coupe |
| --- | --- | --- | --- | ---: |
| 2427 + 93002 | 2427 V290 8,7 ; 93002 V260 22,8 | 31,5 | 52 ; 64,7 + 70 ; 20 + 75 + 20 | ![Contreventement 93002 et V260 sur meneau 2427](/assets/procedures/moe-systeme-70/statique/contreventement-meneau-2427-93002-v260.png) |
| 2427 + 93000 | 2427 V290 8,7 ; 93000 V261 10,0 | 18,7 | 25 ; 57 + 70 ; 20 + 75 + 20 | ![Contreventement 93000 et V261 sur meneau 2427](/assets/procedures/moe-systeme-70/statique/contreventement-meneau-2427-93000-v261.png) |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 158)

Le profilé 93000 reçoit l'un des deux renforts V250 (71,6 cm⁴, entre deux dormants) ou V261
(10,0 cm⁴, sur le meneau 2427) : voir [Accessoires par profilé du système 70](/profiles/systeme-70-accessoires-par-profile.md).

# Exemple de calcul par les tables

## Données

Soit un bâtiment situé en zone 1, situation b (ville), ayant une hauteur de 18 à 28 m.
Détermination de la pression conventionnelle, tableau NF EN 12211, pour H de 18 à 28 m, zone 1
situation b : **PV = 800 Pa = 0,80 kN/m² (V\*A2)** [1 p. 159].

## Méthode

a) Faire un croquis à l'échelle.
b) Déterminer les montants ou traverses filants.
c) Déterminer les axes : horizontal 0, 1, 2, 3, etc. ; vertical A, B, C, D, etc.
d) Tracer les plans de charge à gauche et à droite du meneau ou de la traverse. Par
   simplification, les charges dues à la pression du vent se répartissent de part et d'autre du
   meneau comme une charge trapézoïdale.
e) Déterminer les largeurs de charge a et b en cm.
f) Se reporter à l'abaque pour déterminer en lecture directe le moment d'inertie Iw exprimé en cm⁴.
g) Reporter les valeurs dans le formulaire récapitulatif [1 p. 159].

## Châssis de l'exemple

![Châssis de l'exemple, 240 × 200 cm, axes et détails repérés](/assets/procedures/moe-systeme-70/statique/exemple-chassis-240x200-axes.png)

Châssis de 240 cm de large (140 + 100) sur 200 cm de haut ; le meneau (axe 1) sépare une partie
gauche recoupée par une traverse (axe B) à 80 cm du bas (80 + 120) et une partie droite ; les
cercles numérotés 1, 2 et 3 renvoient aux détails ci-dessous ; coupe verticale à droite, coupe
horizontale en bas [1 p. 159].

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 159)

| Détail | Profilés et renforts | Cotes portées (mm) | Coupe |
| --- | --- | --- | ---: |
| 1 | ouvrant 6112 + V158 sur dormant 6101 + V601 | 45 + 64 ; profondeur 70 | ![Détail 1 : ouvrant 6112 et dormant 6101](/assets/procedures/moe-systeme-70/statique/exemple-detail-1-6112-6101.png) |
| 2 | 6112 + V158, meneau 6127 + V603, 6112 + V158 | 45 + 80 + 45 | ![Détail 2 : ouvrants 6112 et meneau 6127](/assets/procedures/moe-systeme-70/statique/exemple-detail-2-6112-6127.png) |
| 3 | 6112 + V158, meneau 2425 + 9132, 6112 + V158 | 45 + 90 + 45 | ![Détail 3 : ouvrants 6112 et meneau 2425](/assets/procedures/moe-systeme-70/statique/exemple-detail-3-6112-2425.png) |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 160)

## Plans de charge et résultats

![Plans de charge de l'axe 1 et de l'axe B](/assets/procedures/moe-systeme-70/statique/exemple-plans-de-charge-axes-1-et-b.png)

Axe 1 (meneau) : A₁ = 140 cm, B₁ = 100 cm, portée L₁ = 200 cm, largeurs de charge a₁ = 70 cm et
b₁ = 50 cm. Axe B (traverse) : A₂ = 80 cm, B₂ = 120 cm, portée L₂ = 140 cm, a₂ = 40 cm, b₂ = 60 cm.
Les plans de charge sont tracés à 45° depuis les angles. Remarque : a et b peuvent avoir la valeur
maxi de L/2. En utilisant la formule ou l'abaque, on obtient les résultats suivants [1 p. 161] :

| Grandeur | Axe 1 | Axe B |
| --- | --- | --- |
| Pression (kN/m²) | 0,80 | 0,80 |
| Pression (Pa) | 800,00 | 800,00 |
| Portée L (cm) | 200,00 | 140,00 |
| Largeur de charge a (cm) | 70,00 | 40,00 |
| Largeur de charge b (cm) | 50,00 | 60,00 |
| Iw (a) (cm⁴) | 3,39 | 0,71 |
| Iw (b) (cm⁴) | 2,69 | 0,89 |
| Iw (a + b) (cm⁴) | 6,08 | 1,60 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 161)

Possibilité de construction système 70 : dormant 6101, ouvrant 6112 ; axe 1 : meneau 2425,
renfort 9132, Iw disponible 9,1 cm⁴ pour un Iw besoin de 6,08 cm⁴ ; axe B : traverse 6127, renfort
V603, Iw disponible 3,0 cm⁴ pour un Iw besoin de 1,6 cm⁴. « Compte tenu du fait que les valeurs
statiques des renforts utilisés sont supérieures à celles exigées, la limite de la flexion
maximale imposée par la norme n'est pas dépassée. » [1 p. 161]

Les valeurs 3,39 et 2,69 se lisent sur la table [V\*A2](/profiles/systeme-70-inerties-va2.md)
(portée 200, largeurs de charge 70 et 50) ; 0,71 et 0,89 sur la même table (portée 140, largeurs
40 et 60).

# Formule de la charge trapézoïdale

Les tables et les fiches de calcul utilisent la formule de la charge trapézoïdale, où la pression
w est répartie sur la largeur de charge a de part et d'autre de la poutre de portée L posée sur
deux appuis A et B [1 p. 162-163] :

![Formule de la charge trapézoïdale](/assets/procedures/moe-systeme-70/statique/formule-charge-trapezoidale.png)

Iw = w × L⁴ × a / (1920 × 10³ × E × f) × [25 − 40 × (a/L)² + 16 × (a/L)⁴] [cm⁴]

| Symbole | Grandeur | Unité |
| --- | --- | --- |
| w | pression | kN/m² |
| L | portée | cm |
| a | largeur de charge | cm |
| f | flèche maxi | cm |
| E | module d'élasticité | N/mm² |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 162-163)

**Sur le formulaire exemple (PDF p. 162), le dénominateur s'arrête à « E · » sans la flèche f ; le
formulaire vierge (PDF p. 164) écrit « f zul. » (flèche admissible) — entrée INC-145**.

# Formulaire de calcul

Le formulaire récapitulatif porte : Mo (maître d'ouvrage), chantier, fabricant ; système
(système 70, blanc, structure, couleur) ; 1. Situation construction (zones 1 à 5, situation a,
b, c, d, hauteur en m, pression en kN/m²) ; 2. Valeurs des éléments par axe (portée L, largeurs
de l'élément A et B, largeurs de charge a et b, « a/b ≤ max. L/2 ! », en cm) ; 3. Formule ;
4. Résultats (moment d'inertie I (a), I (b), I total) ; 5. Solution : profilés proposés ;
encadrés « Module d'élasticité » (acier 210 000 N/mm², alu 70 000 N/mm²) et « Flèche MAXI » (L/150,
L/300) [1 p. 162, 164].

![Schéma des plans de charge du formulaire](/assets/procedures/moe-systeme-70/statique/formulaire-schema-plans-de-charge.png)

Formulaire exemple rempli (PDF p. 162) : système 70, blanc ; zone 1, situation b, hauteur 18/28 m,
pression 0,8 kN/m² ; acier, flèche L/150.

| Ligne du formulaire | Axe 1 | Axe b |
| --- | --- | --- |
| Portée = L (cm) | 200 | 140 |
| Largeur de l'élément = A (cm) | 140 | 80 |
| Largeur de l'élément = B (cm) | 100 | 120 |
| Largeur de charge = a (cm) | 70 | 40 |
| Largeur de charge = b (cm) | 50 | 60 |
| Moment d'inertie I (a) (cm⁴) | 3,39 | 0,71 |
| Moment d'inertie I (b) (cm⁴) | 2,69 | 0,89 |
| I total (cm⁴) | 6,08 | 1,60 |
| Solution | profilé 2425, renfort 9132 | profilé 6127, renfort V603 |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 162)

Le formulaire vierge (PDF p. 164) reprend la même mise en page, sans aucune valeur.

## Fiches de calcul Iw sur meneau et sur traverse

Deux fiches (client, chantier, date) rappellent les pressions w et les flèches f des classements
V\*A2, V\*A3 et V\*A4, et donnent l'exemple rempli [1 p. 163].

| Fiche | Classement | w (kN/m²) | f (cm) |
| --- | --- | --- | --- |
| meneau (H = 200 cm) | V\*A2 | 0,8 | 1,33 |
| meneau (H = 200 cm) | V\*A3 | 1,2 | 1,33 |
| meneau (H = 200 cm) | V\*A4 | 1,6 | 1,33 |
| traverse (L = 140 cm) | V\*A2 | 0,8 | 0,933 |
| traverse (L = 140 cm) | V\*A3 | 1,2 | 0,933 |
| traverse (L = 140 cm) | V\*A4 | 1,6 | 0,933 |

![Fiche de calcul Iw sur meneau, plan de charge](/assets/procedures/moe-systeme-70/statique/fiche-meneau-plan-de-charge.png)

Fiche meneau : H = 200 cm, L1/2 = 70 cm, L2/2 = 50 cm, E = 210 000 N/mm², classement V\*A2,
I 1 = 3,39 cm⁴, I 2 = 2,69 cm⁴, Iw = 6,08 cm⁴.

![Fiche de calcul Iw sur traverse, plan de charge](/assets/procedures/moe-systeme-70/statique/fiche-traverse-plan-de-charge.png)

Fiche traverse : L = 140 cm, H1/2 = 40 cm, H2/2 = 60 cm, E = 210 000 N/mm², classement V\*A2,
I 1 = 0,71 cm⁴, I 2 = 0,89 cm⁴, Iw = 1,60 cm⁴.

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 163)

# Données du plan des profilés e.VOLUTION 2008

Ces deux phrases viennent du plan des profilés e.VOLUTION 2008 et n'ont pas été recontrôlées lors
de la relecture du manuel 2023 :

- Flèche standard f ≤ L/200 : règle générale des menuiseries avec vitrage isolant double, avec un
  maximum absolu de 15 mm pour L > 3 m [2 p. 138].
- Flèche de confort f ≤ L/300 : exigée pour les grands ensembles vitrés, triples vitrages lourds
  ou vitrages feuilletés de sécurité, avec un maximum absolu de 8 mm sur la hauteur de vitrage
  [2 p. 138].

# Ce que la source ne donne pas

- Les éléments auxquels s'appliquent les flèches de 1/200 et 1/300 des tables V\*B et V\*C, hors
  la traverse d'allège (1/300).
- Les inerties Ig des meneaux et des assemblages (seules les Iw sont imprimées).
- Le tableau 2 de la FD P 20-201, auquel la méthode renvoie.

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 147 à 180 du
PDF (pages imprimées 1 à 34)

[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 —
`raw/profine-plans-profiles-e-volution-2008-08.pdf`, p. 138

# Voir aussi

- [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)
- [Abaques dimensionnels de renforcement du système 70](/profiles/systeme-70-abaques-dimensionnels.md)
- [Mise en œuvre des renforts du système 70](/procedures/mise-en-oeuvre-renforts-systeme-70.md)
- [Accouplement des éléments du système 70](/procedures/accouplement-elements-systeme-70.md)
- [Classification de la résistance au vent](/reference/classification-resistance-au-vent.md)
- [Mise en œuvre Système 70 Plateforme](/sources/profine-mise-en-oeuvre-systeme-70.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
