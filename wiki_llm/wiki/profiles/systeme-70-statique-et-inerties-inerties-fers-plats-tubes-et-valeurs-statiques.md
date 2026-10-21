---
type: Profilé
title: Moments d'inertie des fers plats et tubes acier, valeurs statiques des meneaux, système 70
description: Tables des moments d'inertie Iw et Ig (cm⁴) des fers plats et des tubes acier rectangulaires et carrés, valeurs statiques des meneaux et traverses et des assemblages de dormants et de meneaux avec contreventement, système 70.
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
    pages: 140, 157-192
generated:
  by: process:claude-code
  at: 2026-10-21T10:41:54Z
---

Cette page reprend, du manuel Mise en œuvre Système 70 Plateforme et du classeur e.VOLUTION d'août 2008, les tables des moments d'inertie des fers plats et des tubes acier et les valeurs statiques des meneaux, traverses et assemblages. Le contexte commun est dans [Calcul statique du système 70 : méthode, formule générale, plans de charge et exemples](/profiles/systeme-70-statique-et-inerties.md).

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

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 147 à 180 du
PDF (pages imprimées 1 à 34)

[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 —
`raw/profine-plans-profiles-e-volution-2008-08.pdf`, registre 4.2 « Statique », PDF p. 140 et 157 à 192 (pages imprimées 3 et 20 à 55)

# Voir aussi

Cette page est une partie d'un sujet qui en compte 5 ; les autres :

- [Calcul statique du système 70 : méthode, formule générale, plans de charge et exemples](/profiles/systeme-70-statique-et-inerties.md)
- [Exemple de calcul statique par les tables, formulaire et exigences de flèche, système 70](/profiles/systeme-70-statique-et-inerties-exemple-de-calcul-par-les-tables.md)
- [Valeurs statiques Iz des meneaux, traverses et assemblages, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-statique-et-inerties-valeurs-statiques-evo2008.md)
- [Exemple de calcul statique, fers plats et tubes acier, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-statique-et-inerties-exemple-de-calcul-evo2008.md)
