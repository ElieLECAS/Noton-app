---
type: Profilé
title: Parcloses à pelle T591001 à T591012, pose de face et calage de la porte SOLEAL PY 55
description: Matrice parclose x joint intérieur (TAS0018 à T710002) donnant l'épaisseur de remplissage A de 4 à 42 mm, zone recommandée, parcloses pose de face T591221 à T591223 avec clip T770070 (5/ml), cales de 2 et 4 mm, support T740012, emplacement des cales.
tags: [technal, soleal-py, porte, parclose, prise-de-volume, joint, cale, vitrage, epaisseur-de-vitrage, 55mm]
gamme: LUMINE
systeme: SOLEAL PY
fournisseur: TECHNAL
usage: atelier
famille: parcloses
status: stable
sources:
  - resource: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf
    title: SOLEAL PY 55, Conception, réf. 4944.006
    last_modified: 2018-09-01
source_pages:
  - resource: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf
    pages: 30-31
generated:
  by: process:claude-code
  at: 2026-10-10T11:14:07Z
---

# Prises de volume et parcloses à pelle

La prise de volume est l'épaisseur de remplissage (vitrage ou panneau) que le profilé reçoit ; la parclose est la baguette qui maintient le vitrage dans son logement ; le joint intérieur est le joint placé entre le vitrage et la parclose. La porte SOLEAL PY 55 de [TECHNAL](/fournisseurs/technal.md) reçoit des parcloses à pelle associées à 7 joints tournants et des prises de volume de 4 à 42 mm ([Concept de la porte SOLEAL PY 55](/portes/soleal-py-concept-du-produit.md)). Le joint extérieur est le T410010 ; la cale de 2 mm n'est pas fournie ; le support de cale est le T740012 [1 p. 30].

Dans la matrice, trois cotes se lisent : A est l'épaisseur du remplissage, B l'épaisseur du joint intérieur (repère couleur et cote en mm imprimés au-dessus de chaque colonne), C la hauteur de la parclose. Pour un remplissage donné, on cherche la cellule qui vaut A : sa ligne donne la parclose, sa colonne le joint intérieur. La zone grisée de la planche est la « prise de volume recommandée » ; la dernière colonne la reprend. La planche imprime les références de joint sur deux lignes décalées sous les colonnes ; elles sont rattachées ici aux colonnes par leur position : Blanc TAS0018, Vert TAS0017, Rouge TAS0016, Bleu TAS0015, Jaune T710004, Marron T710002 (**VER-566**).

<table>
<thead>
<tr><th rowspan="3">Parclose</th><th rowspan="3">Hauteur de parclose C (mm)</th><th colspan="6">Épaisseur de remplissage A (mm) obtenue avec le joint intérieur</th><th rowspan="3">Prise de volume recommandée (zone grisée), A (mm)</th></tr>
<tr><th>Blanc TAS0018</th><th>Vert TAS0017</th><th>Rouge TAS0016</th><th>Bleu TAS0015</th><th>Jaune T710004</th><th>Marron T710002</th></tr>
<tr><th>B = 8 mm</th><th>B = 7 mm</th><th>B = 6 mm</th><th>B = 5 mm</th><th>B = 4 mm</th><th>B = 2 mm</th></tr>
</thead>
<tbody>
<tr><td>T591001</td><td>2</td><td>36</td><td>37</td><td>38</td><td>39</td><td>-</td><td>-</td><td>37 à 39</td></tr>
<tr><td>T591002</td><td>5</td><td>33</td><td>34</td><td>35</td><td>36</td><td>37</td><td>39</td><td>34 à 36</td></tr>
<tr><td>T591003</td><td>9</td><td>29</td><td>30</td><td>31</td><td>32</td><td>33</td><td>35</td><td>30 à 33</td></tr>
<tr><td>T591004</td><td>12</td><td>26</td><td>27</td><td>28</td><td>29</td><td>30</td><td>32</td><td>27 à 29</td></tr>
<tr><td>T591005</td><td>15</td><td>23</td><td>24</td><td>25</td><td>26</td><td>27</td><td>29</td><td>24 à 26</td></tr>
<tr><td>T591006</td><td>19</td><td>19</td><td>20</td><td>21</td><td>22</td><td>23</td><td>25</td><td>20 à 23</td></tr>
<tr><td>T591007</td><td>22</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>22</td><td>17 à 19</td></tr>
<tr><td>T591008</td><td>24</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>20</td><td>15 à 16</td></tr>
<tr><td>T591009</td><td>26</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>18</td><td>13 à 14</td></tr>
<tr><td>T591010</td><td>29</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>15</td><td>10 à 12</td></tr>
<tr><td>T591011</td><td>32</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>12</td><td>7 à 9</td></tr>
<tr><td>T591012</td><td>34</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>10</td><td>4 à 6</td></tr>
</tbody>
</table>

(schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 30)

La parclose T591001 **ne se monte pas** avec les joints jaune et marron : ces deux cellules sont vides sur la planche. Une valeur n'est donnée qu'en une cellule de la matrice ; aucune valeur intermédiaire n'est interpolée. Les valeurs de cette matrice sont les mêmes que celles de la matrice du SOLEAL FY 55 ([Prises de volume et parcloses SOLEAL FY 55](/profiles/soleal-fy-parcloses-et-vitrage.md)).

## Coupes des parcloses à pelle

Les coupes montrent les cotes A, B et C, le joint extérieur T410010, la cale de 2 mm non fournie et le support de cale T740012 ; deux coupes par famille de profilés (haut et bas de la planche). Pour les profilés laqués, la pièce plastique T197112 est optionnelle (pour tenir la parclose en place pendant le transport) ; pour les profilés anodisés, le clip TAT0049 est obligatoire (1 par extrémité de parclose) [1 p. 30].

![Parcloses à pelle, profilés laqués : coupes A, B, C, joint T410010, support T740012, pièce T197112](/assets/profiles/soleal-py/prises-de-volume/parcloses-a-pelle-profiles-laques.png)

Profilés laqués (schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 30).

![Parcloses à pelle, profilés anodisés : coupes A, B, C, joint T410010, support T740012, clip TAT0049](/assets/profiles/soleal-py/prises-de-volume/parcloses-a-pelle-profiles-anodises.png)

Profilés anodisés (schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 30).

# Parcloses pose de face

Les parcloses pose de face sont T591221, T591222 et T591223 (profilés laqués et anodisés). Elles se posent avec le clip T770070, obligatoire (5 par mètre linéaire) ; la cale est de 4 mm, non fournie ; le joint extérieur est le T410010 et le support de cale le T740012 [1 p. 31]. Même lecture que la matrice précédente (A, B, C, zone grisée).

| Parclose | Hauteur de parclose C (mm) | A avec Blanc TAS0018 (B = 8) | A avec Vert TAS0017 (B = 7) | A avec Rouge TAS0016 (B = 6) | A avec Bleu TAS0015 (B = 5) | A avec Jaune T710004 (B = 4) | A avec Marron T710002 (B = 2) | Zone grisée, A (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T591221 | 2 | 36 | 37 | 38 | 39 | 40 | 42 | 40 à 42 |
| T591222 | 15 | 23 | 24 | 25 | 26 | 27 | 29 | 24 à 27 |
| T591223 | 15 | 23 | 24 | 25 | 26 | 27 | 29 | 24 à 27 |

(schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 31)

![Parcloses pose de face, profilés laqués et anodisés : coupes A, B, C, cale de 4 mm, clip T770070](/assets/profiles/soleal-py/prises-de-volume/parcloses-pose-de-face-coupes.png)

Les deux coupes montrent la parclose pose de face avec la cale de 4 mm non fournie, le support T740012, le joint T410010 et le clip T770070 (schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 31).

## Position du clip T770070

Le clip T770070 est obligatoire, à raison de 5 par mètre linéaire. Le dessin montre le cadre vu de face avec les clips en noir : le premier clip de chaque côté est à 150 mm de l'angle (cotes 150), puis 5 clips par mètre linéaire (cote « 5/ml ») entre ces deux repères, sur les quatre côtés [1 p. 31].

![Position du clip T770070 sur le cadre : cotes 150 mm et 5/ml](/assets/profiles/soleal-py/prises-de-volume/pose-de-face-position-du-clip-t770070.png)

(schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 31)

## Avertissements

Lors de l'utilisation de ce type de parcloses, prévoir la largeur et la hauteur des remplissages 4 mm plus petites. Ce type de montage de parclose **n'est pas autorisé** dans le cas de justification de choc de sécurité intérieur ou extérieur suivant la norme EN13049 [1 p. 31].

# Mise en œuvre et emplacement des cales

Le support de cale est le T740012 (profilé plastique perforé représenté en perspective). La planche montre l'emplacement des cales, en noir, sur trois ouvrages : porte 1 vantail, porte 2 vantaux, porte 1 vantail en ensemble menuisé [1 p. 31]. Aucune cote d'emplacement n'est imprimée.

![Emplacement des cales : support T740012, porte 1 vantail, porte 2 vantaux, porte 1 vantail en ensemble menuisé](/assets/profiles/soleal-py/prises-de-volume/emplacement-des-cales-t740012.png)

(schéma: raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf, p. 31)

# Citations

[1] [SOLEAL PY 55, Conception, réf. 4944.006](raw/SOLEAL-PY-55-catalogue-conception-4944-006-092018-FR (1).pdf), p. 30-31
