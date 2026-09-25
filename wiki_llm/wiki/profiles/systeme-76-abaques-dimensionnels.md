---
type: Profilé
title: Abaques dimensionnels du système 76
description: Les limites de dimension de dormant et d'ouvrant du système 76 Advanced selon le renfort, la couleur, l'épaisseur de verre, la charge de vent et le battement, relevées sur les abaques du manuel profine, avec les poids d'ouvrant admissibles et les règles de renforcement.
tags: [systeme-76-advanced, abaque, dimension, renfort, couleur, vitrage, paumelle, poids-ouvrant, charge-de-vent, battement]
systeme: 76
fournisseur: KÖMMERLING
usage: [atelier, chiffrage]
famille: abaques
status: draft
sources:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
source_pages:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    pages: 98-117
generated:
  by: process:claude-code
  at: 2026-09-26T01:30:00Z
---

# Ce qu'est un abaque dimensionnel

Un **abaque dimensionnel** est un graphique qui donne les dimensions maximales réalisables : on
place le point (largeur ; hauteur) de la menuiserie et on regarde s'il tombe à l'intérieur de la
limite tracée. Le registre 2.3.3 du manuel
[Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md) en donne un
pour le dormant, puis un par renfort d'ouvrant [1 p. 98-117].

« La dimension des ouvrants dépend des valeurs statiques (IW, IG) du renfort utilisé, de la
conception du châssis, de sa couleur ainsi que des charges agissantes et des ferrures utilisées.
Les tailles d'ouvrant calculées constituent la base des abaques dimensionnels d'ouvrant présentés
dans les pages suivantes. » [1 p. 99] **IW** est l'inertie du renfort au vent, **IG** son inertie
au poids, en cm⁴ — voir [Renforts du système 76](/profiles/systeme-76-renforts.md).

Les abaques d'ouvrant indiquent « les cotes extérieures ouvrants maximales réalisables pour :
Châssis oscillo-battants et ouvrant à la française ; Châssis basculants (mêmes diagrammes de taille
en inversant la hauteur et la largeur) ; combinaisons ouvrant battement » [1 p. 98]. **Ce sont des
cotes d'ouvrant, pas des cotes de fenêtre.** Les dimensions maximales de baie relèvent du
[DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) ; les deux jeux de limites s'appliquent
tous les deux.

Référentiel applicable, en gras sur la planche : « Respecter les directives de renforcement
générales (reg. 1.2.3) et les exigences statiques (reg. 1.3.3). Respecter les directives de
fabrication générales pour les profilés blancs et couleurs (reg. 1.2 + 1.3). » Ces registres sont
ceux des [Directives générales profine](/sources/profine-directives-generales.md) [1 p. 99].

# Dimensions maximales de dormant

![Dimensions maximales de dormant par couleur](/assets/procedures/moe-76-advanced/abaque-dimensions-maximales-dormant.png)

L'abaque porte la largeur du dormant en m en abscisse et sa hauteur en m en ordonnée, graduées
tous les 0,5 m ; chaque couleur est une ligne brisée, marquée d'un symbole à chaque sommet :
losange pour le blanc, carré pour la couleur standard, triangle pour la couleur IR-Reflex, tireté
pour le capot aluminium. Le tableau donne les sommets marqués, lus sur la planche à ±0,05 m ; entre
deux sommets, la limite est le segment qui les joint [1 p. 98].

| Couleur | Sommets de la limite, (largeur ; hauteur) en m |
| --- | --- |
| Blanc | (0 ; 4,5), (2,7 ; 4,5), (3,5 ; 3,5), (4,5 ; 2,7), (4,5 ; 0) |
| Couleur IR-Reflex | (0 ; 3,5), (3,5 ; 3,5), (3,5 ; 0) |
| Capot aluminium | (0 ; 3), (3 ; 3), (3 ; 0) |
| Couleur standard | (0 ; 2,5), (2,5 ; 2,5), (2,5 ; 0) |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 98, registre 2.3.3, p. 1, version janvier 2016)

Remarques de la planche : « Dans le cas d'éléments couplés, les dimensions maximales du dormant
dépendent de la performance de l'accouplement : Des jeux de dilatation sont nécessaires pour les
éléments couplés : 1. à partir d'une largeur > 3,50 m pour le blanc, capots aluminium et film
IR-Reflex 2. à partir d'une largeur > 2,50 pour la couleur » [1 p. 98].

| Finition | Largeur à partir de laquelle un jeu de dilatation est nécessaire (m) |
| --- | --- |
| Blanc | 3,50 |
| Capot aluminium | 3,50 |
| Film IR-Reflex | 3,50 |
| Couleur | 2,50 |

Sur cet abaque, la couleur standard est marquée d'un carré et l'IR-Reflex d'un triangle ; sur les
abaques d'ouvrant, l'IR-Reflex est marquée d'un rond et la couleur standard d'un triangle.

# Comment lire les abaques d'ouvrant

## Charge du vent et charge du vitrage

« Valeur IW = valeur de stabilité pour la reprise des efforts dans la direction du vent. La valeur
IW du renfort utilisé est déterminante pour le calcul de la résistance aux charges de vent. Dans
les diagrammes des ouvrants simples, les efforts sont déjà pris en compte selon la couleur. Pour
tous les châssis à deux vantaux, les charges de vent sont indiquées par les courbes
caractéristiques. La courbe caractéristique indique la taille réalisable de l'élément pour la
charge de vent correspondante. » La légende dessine des courbes de 0,6, 0,8, 1,0, 1,2, 1,4, 1,6,
1,8, 2,0 et 2,5 kN/m² ; les abaques à deux vantaux n'en portent que deux, 0,8 et 1,2 kN/m²
[1 p. 99].

« Valeur IG = valeur de stabilité pour la reprise des efforts de poids. A partir de la valeur IG
du renfort utilisé, les dimensions d'ouvrant réalisables sont limitées en terme d'épaisseurs de
vitrage et de l'effort résultant de leur poids. Les abaques des ouvrants comportent les courbes
caractéristiques […] pour les diverses épaisseurs de vitrage » : **limitation 12, 16, 20, 24 et
28 mm**. « Important : Cette limitation compte également pour sur les éléments à deux ouvrants. »
[1 p. 99]

## Équerre de feuillure J079

« L'utilisation de l'equerre de feuillure J079 dans les 4 coins de l'ouvrant augmente
considérablement sa stabilité pour la reprise des poids de vitrage. Si l'equerre de feuillure J079
est montée, les limitations de vitrage peuvent être décalées de 2 courbes caractéristiques
(niveaux). Pour ex. dans le cas de mise en place d'un vitrage de 16mm d'épaisseur (de verre) et des
equerres de feuillure, les dimensions maximales à respecter sont celles d'un vitrage de 8 mm (2
courbes caractéristiques) » [1 p. 99]. Sur la variante **AluClip Pro**, l'équerre J079 devient
obligatoire à partir de **40 kg** de poids d'ouvrant (voir *AluClip Pro* plus bas).

## Épaisseur de verre

« A partir d'une épaisseur de verre de 12 mm et par rapport au poids de vitrage élevé obtenu, les
dimensions maxi des ouvrants sont restreintes. Cette restriction concerne aussi bien les
dimensions du blanc comme celles de la couleur. Pour les épaisseurs intermédiaires (ex : 13mm) la
limite supérieure s'applique (16mm). Ces restrictions sont également à prendre en considération
pour les vantaux recevant le battement. Pour déterminer l'épaisseur du verre : additionner les
différentes couches de verre, sans les intercalaires : ex. un vitrage 4-12-4-12-4 donne une
épaisseur totale de 4+4+4 = 12 mm. » [1 p. 103]

**Au-delà de 12 mm d'épaisseur de verre, le renforcement total est obligatoire et les limites
correspondantes respectées** [1 p. 100, 103].

## La règle des 25 %

« Les dimensions des ouvrants donnés ont été établies par rapport à la quincaillerie et du poids
total. La largeur ouvrant ne peut dépasser la hauteur ouvrant de plus 25 %. » Exemples portés sur
les planches : pour une largeur de 130 cm, la hauteur doit être au minimum de 103 cm ; pour
150 cm, de 120 cm [1 p. 103-104]. Sur chaque abaque, cette règle est la limite oblique basse,
tracée de (62,5 ; 50) à la largeur maximale.

## Catégories de couleur

« Afin de faciliter les procédés de fabrication, différentes techniques et couleurs sont
regroupées dans des catégories. Ces catégories sont prises en compte pour les diagrammes de
dimension. Cependant, les prescriptions pour le traitement des fenêtres en couleur ou de la
réalisation technique correspondante restent inchangées et valables quelle que soit la catégorie
de groupe assignée. » [1 p. 99]

![Légende des catégories et des marquages](/assets/procedures/moe-76-advanced/abaque-legende-categories.png)

**1. Blanc** (losange) [1 p. 100] :

- « Tous les ouvrants en couleur blanc et blanc crème, quel que soit le traitement de surface
  (couleur de base, plaxés, etc...), renforcés avec le renfort préconisé. »
- « Éléments avec vitrage collé de couleur blanc. Sans renfort en acier. »
- **Capots en aluminium** : « Les abaques dimensionnels d'ouvrant contiennent le marquage
  ci-contre pour chaque combinaison ouvrant-renfort » (pictogramme « Capot aluminium autorisé » ou
  « Capot aluminium non autorisé »). « Si les capots aluminium sont autorisés, ils peuvent être
  réalisés comme un ouvrant blanc du point de vue dimensionnel. Cependant, le traitement doit être
  réalisé comme pour les profilés en couleur (aération, vissage de renfort). » Tous les abaques
  d'ouvrant du registre portent le pictogramme « Capot aluminium autorisé ».
- **Renforcement selon abaques** : « Il ne concerne que les ouvrants en couleur blanc et ton pierre
  (cccc), quel que soit le traitement de surface (couleur de base, plaxés, etc...) avec le renfort
  préconisé. Les calculs d'inertie ont été réalisés pour une flèche de 1/150ème. (0,8kN/m² = 800Pa
  et 1.2kN/m² = 1200Pa). La position du renfort est indiqué sur les croquis ci-contre » (A sans
  renfort, H renforcement total).
- **Utilisation de paumelles (OF)** : « En cas d'utilisation de paumelles, le renforcement côté
  paumelle n'est pas nécessaire pour le blanc et ton pierre (cccc) quel que soit le traitement. »

**2. Couleurs IR-Reflex** (rond) : « Seules les couleurs indiquées dans le tableau suivant sont
autorisées. La catégorie des couleurs IR-Reflex comporte des couleurs capables de réfléchir le
rayonnement thermique du soleil. Renforcement systématique. » La liste est « celles en vigueur,
lors de l'édition du document. Elles peuvent évoluer avec l'offre couleur. Veuillez vous référer
au tarif en vigueur pour connaître l'offre du moment. » [1 p. 101]

| Désignation | Couleur profine | Référence fournisseur |
| --- | --- | --- |
| Chêne doré | 32 | 9.2178.301 - 116700 |
| Noyer | 52 | 9.2178.307 - 116700 |
| Acajou | 24 | 9.2178.321 - 116700 |
| Anthracite veiné | 16 | 02.20.71.000001 - 116700 |
| Gris argent | 21 | 02.20.71.000007 - 116700 |
| Gris signal | UJ | 02.20.71.000009 - 504700 |
| Chêne irlandais | IO | 9.3211.305 - 114800 |
| Métal brossé silver | AR | F436-1002 |
| Métal brossé platine | MP | F436-1004 |
| Alux DB 703 | AL | F436-1014 |
| Anthracite mat | AU | F436-6003 |
| AnTeak | AN | 9.3241.302 - 119500 |
| Rouge foncé | 46 | 02.11.31.000013 - 116700 |
| Vert mousse | 53 | 02.20.61.000002 - 116700 |
| Bleu acier | 88 | 02.20.51.000001 - 116700 |
| Brun noir | 80 | 02.20.81.000010 - 116700 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 101, registre 2.3.3, p. 4, version janvier 2022)

**3. Couleurs standard** (triangle) : « Tous les produits couleurs de notre programme de livraison
qui ne relèvent pas des catégories 1 ou 2. Renforcement systématique. » [1 p. 101]

| Désignation | Relief | Semblable à la référence fournisseur |
| --- | --- | --- |
| Brun Chocolat | 27 | 02.20.81.000018 - 116700 |
| Honey Oak super mat | HB | 3.0078.007 - 102201 |
| Siena PR | SC | 4.0131.006 - 114800 |
| Métal brossé quartz | MQ | F436-1005 |
| Gris basalte mat | BQ | F436 6048 |
| Brun sepia ulti mat | PF | 02.20.81.000030 - 504700 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 101)

## Nombre de paumelles

« Tableau indiquant le nombre de paumelles par hauteur d'ouvrant » [1 p. 100] :

| Hauteur d'ouvrant (cm) | Nombre de paumelles |
| --- | --- |
| 50 à 90 | 2 |
| 100 à 160 | 3 |
| 170 à 210 | 4 |
| 220 à 250 | 5 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 100, registre 2.3.3, p. 3)

Le tableau est gradué tous les 10 cm. Le même nombre est porté en marge gauche de chaque abaque
d'ouvrant, par une accolade sur l'axe des hauteurs (2, 3, 4, 5 de bas en haut, bornes à 90, 160 et
210 cm).

## Zones de renforcement

Chaque abaque est découpé en zones grisées, repérées par une lettre, qui disent quels profilés
renforcer :

![Zones A à D des abaques d'ouvrant simple](/assets/procedures/moe-76-advanced/abaque-legende-zones-a-d.png)

![Zones A à H des abaques à deux vantaux](/assets/procedures/moe-76-advanced/abaque-legende-zones-a-h.png)

| Zone | Ouvrant simple | Deux vantaux |
| --- | --- | --- |
| A | sans renfort | sans renfort |
| B | renforcement horizontal | renforcement horizontal |
| C | renforcement vertical | renforcement vertical 1 Ouvrant |
| D | renforcement total | renforcement vertical 1 Ouvrant et renforcement horizontal |
| E | - | renforcement vertical 2 ouvrants |
| F | - | renforcement total, sauf battement |
| G | - | renforcement vertical, avec battement |
| H | - | renforcement total, avec battement |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 103 et 109)

Sur l'abaque des ouvrants 76272 / 76279 avec battement 76473 (PDF p. 110), les zones C et D sont
légendées « renforcement vertical 2 ouvrants » et « renforcement total, sauf battement ».

# Abaques d'ouvrant simple

Chaque abaque porte la largeur d'ouvrant en abscisse et la hauteur en ordonnée, en cm, graduées
tous les 10 cm de 50 à 150 et de 50 à 250. Trois tableaux par renfort : la **limite de chaque
couleur** (ses coins), les **zones**, et les **courbes d'épaisseur de verre**, dont la hauteur est
relevée à chaque graduation de largeur où la courbe est légendée, précision de lecture ±3 cm ;
« - » là où la courbe passe au-dessus de 250 cm ou n'est pas légendée à cette graduation. Un
point sous la courbe de son épaisseur de verre est réalisable ; une courbe au-dessus de la limite
de couleur ne limite pas.

## Ouvrant 76281 / 76275 avec renfort V266.Z

Abaque « Ouvrant avec renfort V266.Z », ouvrant 76281 / 76275, renfort V266.Z, 2 mm, IW 2,5 cm⁴, IG 0,5 cm⁴ ; hauteur d'ouvrant
« max. 235 cm », largeur d'ouvrant « max. 130 cm » [1 p. 103].

![Abaque de l'ouvrant avec renfort V266.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v266z.png)

![Ouvrant 76281 / 76275 et renfort V266.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v266z-profil.png)

Limites de couleur, en cm : la hauteur maxi vaut jusqu'à la largeur indiquée, puis la limite
descend en oblique jusqu'au coin, puis suit la verticale de la largeur maxi jusqu'à la règle des
25 %.

| Couleur | Hauteur maxi (cm) | Largeur maxi à cette hauteur (cm) | Coin bas de l'oblique, largeur (cm) | Coin bas de l'oblique, hauteur (cm) | Largeur maxi (cm) | Hauteur mini à la largeur maxi (cm) |
| --- | --- | --- | --- | --- | --- | --- |
| blanc | 235 | 95 | 130 | 130 | 130 | 104 |
| IR-Reflex | 225 | 98 | 130 | 130 | 130 | 104 |
| standard | 195 | 100 | 120 | 130 | 120 | 96 |

Zones : séparation entre A / C et B / D à 75 cm de largeur et 130 cm de hauteur.

Courbes d'épaisseur de verre (hauteur maxi d'ouvrant, en cm) :

| Largeur d'ouvrant (cm) | Verre 12 mm (cm) | Verre 16 mm (cm) | Verre 20 mm (cm) | Verre 24 mm (cm) | Verre 28 mm (cm) |
| --- | --- | --- | --- | --- | --- |
| 50 | - | - | - | - | - |
| 60 | - | - | - | - | - |
| 70 | - | - | - | 220 | 190 |
| 80 | - | 239 | 193 | 163 | 141 |
| 90 | 242 | 184 | 149 | 126 | 109 |
| 100 | 193 | 147 | 120 | 101 | 88 |
| 110 | 158 | 120 | 98 | 84 | 73 |
| 120 | 132 | 101 | 83 | 71 | 62 |
| 130 | 112 | 86 | 71 | 61 | 54 |
| 140 | 97 | 75 | 62 | 53 | - |
| 150 | 85 | 66 | 55 | - | - |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 103, registre 2.3.3, p. 6, version janvier 2016)

Par rajout du renfort de feuillure **V280** (2 mm, IW 3 cm⁴) sur les montants, les couleurs standards peuvent être réalisées aux dimensions des couleurs IR-Reflex.

## Ouvrant 76271 avec renfort V306.Z

Abaque « Ouvrant avec renfort V306.Z », ouvrant 76271, renfort V306.Z, 1,5 mm, IW 2,3 cm⁴, IG 1,3 cm⁴ ; hauteur d'ouvrant
« max. 235 cm », largeur d'ouvrant « max. 150 cm » [1 p. 104].

![Abaque de l'ouvrant avec renfort V306.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v306z.png)

![Ouvrant 76271 et renfort V306.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v306z-profil.png)

Limites de couleur, en cm : la hauteur maxi vaut jusqu'à la largeur indiquée, puis la limite
descend en oblique jusqu'au coin, puis suit la verticale de la largeur maxi jusqu'à la règle des
25 %.

| Couleur | Hauteur maxi (cm) | Largeur maxi à cette hauteur (cm) | Coin bas de l'oblique, largeur (cm) | Coin bas de l'oblique, hauteur (cm) | Largeur maxi (cm) | Hauteur mini à la largeur maxi (cm) |
| --- | --- | --- | --- | --- | --- | --- |
| blanc | 235 | 105 | 150 | 150 | 150 | 120 |
| IR-Reflex | 215 | 116 | 150 | 150 | 150 | 120 |
| standard | 180 | 100 | 130 | 130 | 130 | 104 |

Zones : séparation entre A / C et B / D à 75 cm de largeur et 130 cm de hauteur.

Courbes d'épaisseur de verre (hauteur maxi d'ouvrant, en cm) :

| Largeur d'ouvrant (cm) | Verre 12 mm (cm) | Verre 16 mm (cm) | Verre 20 mm (cm) | Verre 24 mm (cm) | Verre 28 mm (cm) |
| --- | --- | --- | --- | --- | --- |
| 50 | - | - | - | - | - |
| 60 | - | - | - | - | - |
| 70 | - | - | - | - | 248 |
| 80 | - | - | - | 212 | 183 |
| 90 | - | 240 | 194 | 164 | 142 |
| 100 | - | 191 | 155 | 131 | 113 |
| 110 | 205 | 156 | 127 | 107 | 94 |
| 120 | 171 | 131 | 106 | 90 | 79 |
| 130 | 145 | 111 | 91 | 77 | 68 |
| 140 | 125 | 96 | 79 | 67 | 59 |
| 150 | 109 | 84 | 69 | 60 | 52 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 104, registre 2.3.3, p. 7, version janvier 2016)

## Ouvrant 76271 avec renfort V307.Z

Abaque « Ouvrant avec renfort V307.Z », ouvrant 76271, renfort V307.Z, 2 mm, IW 2,9 cm⁴, IG 1,6 cm⁴ ; hauteur d'ouvrant
« max. 250 cm », largeur d'ouvrant « max. 150 cm » [1 p. 105].

![Abaque de l'ouvrant avec renfort V307.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v307z.png)

![Ouvrant 76271 et renfort V307.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v307z-profil.png)

Limites de couleur, en cm : la hauteur maxi vaut jusqu'à la largeur indiquée, puis la limite
descend en oblique jusqu'au coin, puis suit la verticale de la largeur maxi jusqu'à la règle des
25 %.

| Couleur | Hauteur maxi (cm) | Largeur maxi à cette hauteur (cm) | Coin bas de l'oblique, largeur (cm) | Coin bas de l'oblique, hauteur (cm) | Largeur maxi (cm) | Hauteur mini à la largeur maxi (cm) |
| --- | --- | --- | --- | --- | --- | --- |
| blanc | 250 | 100 | 150 | 150 | 150 | 120 |
| IR-Reflex | 235 | 108 | 150 | 150 | 150 | 120 |
| standard | 200 | 100 | 130 | 130 | 130 | 104 |

Zones : séparation entre A / C et B / D à 75 cm de largeur et 130 cm de hauteur.

Courbes d'épaisseur de verre (hauteur maxi d'ouvrant, en cm) :

| Largeur d'ouvrant (cm) | Verre 12 mm (cm) | Verre 16 mm (cm) | Verre 20 mm (cm) | Verre 24 mm (cm) | Verre 28 mm (cm) |
| --- | --- | --- | --- | --- | --- |
| 50 | - | - | - | - | - |
| 60 | - | - | - | - | - |
| 70 | - | - | - | - | - |
| 80 | - | - | - | - | 229 |
| 90 | - | - | 243 | 204 | 176 |
| 100 | - | 239 | 193 | 163 | 141 |
| 110 | - | 195 | 158 | 133 | 116 |
| 120 | 213 | 162 | 132 | 112 | 97 |
| 130 | 180 | 138 | 112 | 95 | 83 |
| 140 | 155 | 119 | 97 | 82 | 72 |
| 150 | 135 | 104 | 85 | 72 | 63 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 105, registre 2.3.3, p. 8, version janvier 2016)

## Ouvrant 76271 avec renfort V308

Abaque « Ouvrant avec renfort V308 », ouvrant 76271, renfort V308, 2,5 mm, IW 3,4 cm⁴, IG 1,9 cm⁴ ; hauteur d'ouvrant
« max. 250 cm », largeur d'ouvrant « max. 150 cm » [1 p. 106].

![Abaque de l'ouvrant avec renfort V308](/assets/procedures/moe-76-advanced/abaque-ouvrant-v308.png)

![Ouvrant 76271 et renfort V308](/assets/procedures/moe-76-advanced/abaque-ouvrant-v308-profil.png)

Limites de couleur, en cm : la hauteur maxi vaut jusqu'à la largeur indiquée, puis la limite
descend en oblique jusqu'au coin, puis suit la verticale de la largeur maxi jusqu'à la règle des
25 %.

| Couleur | Hauteur maxi (cm) | Largeur maxi à cette hauteur (cm) | Coin bas de l'oblique, largeur (cm) | Coin bas de l'oblique, hauteur (cm) | Largeur maxi (cm) | Hauteur mini à la largeur maxi (cm) |
| --- | --- | --- | --- | --- | --- | --- |
| blanc et IR-Reflex | 250 | 100 | 150 | 150 | 150 | 120 |
| standard | 215 | 100 | 140 | 150 | 140 | 112 |

Zones : séparation entre A / C et B / D à 75 cm de largeur et 130 cm de hauteur.

Courbes d'épaisseur de verre (hauteur maxi d'ouvrant, en cm) :

| Largeur d'ouvrant (cm) | Verre 12 mm (cm) | Verre 16 mm (cm) | Verre 20 mm (cm) | Verre 24 mm (cm) | Verre 28 mm (cm) |
| --- | --- | --- | --- | --- | --- |
| 50 | - | - | - | - | - |
| 60 | - | - | - | - | - |
| 70 | - | - | - | - | - |
| 80 | - | - | - | - | - |
| 90 | - | - | - | 240 | 207 |
| 100 | - | - | 227 | 191 | 165 |
| 110 | - | - | 185 | 156 | 135 |
| 120 | - | - | 154 | 130 | 113 |
| 130 | - | 161 | 131 | 111 | 96 |
| 140 | - | 139 | 113 | 96 | 84 |
| 150 | - | 121 | 99 | 84 | 73 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 106, registre 2.3.3, p. 9, version janvier 2016)

Sur cet abaque, la courbe la plus à droite n'est pas légendée : elle n'est pas reprise dans le tableau.

## Ouvrant 76272, 76279 avec renfort V326.Z

Abaque « Ouvrant avec renfort V326.Z », ouvrant 76272, 76279, renfort V326.Z, 2 mm, IW 5,0 cm⁴, IG 5,4 cm⁴ ; hauteur d'ouvrant
« max. 250 cm », largeur d'ouvrant « max. 150 cm » [1 p. 107].

![Abaque de l'ouvrant avec renfort V326.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v326z.png)

![Ouvrant 76272, 76279 et renfort V326.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v326z-profil.png)

Limites de couleur, en cm : la hauteur maxi vaut jusqu'à la largeur indiquée, puis la limite
descend en oblique jusqu'au coin, puis suit la verticale de la largeur maxi jusqu'à la règle des
25 %.

| Couleur | Hauteur maxi (cm) | Largeur maxi à cette hauteur (cm) | Coin bas de l'oblique, largeur (cm) | Coin bas de l'oblique, hauteur (cm) | Largeur maxi (cm) | Hauteur mini à la largeur maxi (cm) |
| --- | --- | --- | --- | --- | --- | --- |
| blanc et IR-Reflex | 250 | 110 | 150 | 150 | 150 | 120 |
| standard | 220 | 110 | 140 | 140 | 140 | 112 |

Zones : séparation entre A / C et B / D à 80 cm de largeur et 130 cm de hauteur.

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 107, registre 2.3.3, p. 10, version janvier 2016)

L'abaque ne porte aucune courbe d'épaisseur de verre.

## Ouvrant 76272, 76279 avec renfort V314.Z

Abaque « Ouvrant avec renfort V314.Z », ouvrant 76272, 76279, renfort V314.Z, 2 mm, IW 5,7 cm⁴, IG 8,4 cm⁴ ; hauteur d'ouvrant
« max. 250 cm », largeur d'ouvrant « max. 150 cm » [1 p. 108].

![Abaque de l'ouvrant avec renfort V314.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v314z.png)

![Ouvrant 76272, 76279 et renfort V314.Z](/assets/procedures/moe-76-advanced/abaque-ouvrant-v314z-profil.png)

Limites de couleur, en cm : la hauteur maxi vaut jusqu'à la largeur indiquée, puis la limite
descend en oblique jusqu'au coin, puis suit la verticale de la largeur maxi jusqu'à la règle des
25 %.

| Couleur | Hauteur maxi (cm) | Largeur maxi à cette hauteur (cm) | Coin bas de l'oblique, largeur (cm) | Coin bas de l'oblique, hauteur (cm) | Largeur maxi (cm) | Hauteur mini à la largeur maxi (cm) |
| --- | --- | --- | --- | --- | --- | --- |
| blanc et IR-Reflex | 250 | 110 | 150 | 150 | 150 | 120 |
| standard | 220 | 110 | 140 | 140 | 140 | 112 |

Zones : séparation entre A / C et B / D à 80 cm de largeur et 130 cm de hauteur.

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 108, registre 2.3.3, p. 11, version janvier 2016)

L'abaque ne porte aucune courbe d'épaisseur de verre.

Au-delà de la limite de couleur, les courbes d'épaisseur de verre sont prolongées en tireté
jusqu'à 150 cm de largeur ; le tableau reprend les valeurs lues sur ces prolongements.

# Abaques à deux vantaux avec battement

Pour les fenêtres à deux vantaux sans meneau, chaque page porte deux abaques, pour **0,8 kN/m²**
et pour **1,2 kN/m²** de charge de vent. La limite de chaque couleur et les zones sont tracées
comme pour l'ouvrant simple ; en plus, des **courbes caractéristiques** donnent, pour chaque
combinaison dessinée en vignette à droite de l'abaque, la hauteur maxi réalisable. Les vignettes
sont numérotées ici de haut en bas : **1**, battement 76472 avec renfort V317 ; **2**, battement
76471 avec renfort V316 ; **3**, battement sans renfort, les deux ouvrants dessinés avec leur
renfort ; **4**, battement sans renfort, un seul ouvrant dessiné avec son renfort. Deux vignettes
portent en plus leur référence de renfort (« V317 », « V316 »), reprise dans les tableaux. Chaque
courbe est rattachée à la vignette où aboutit son trait ; hauteurs relevées à chaque graduation de
10 cm, précision ±3 cm, « - » là où la courbe est au-dessus de 250 cm.

Sur toutes ces pages, l'axe des hauteurs est légendé « Hauteur ouvrant **max. 235 cm** » et l'axe
des largeurs « Largeur ouvrant **max. 130 cm** », y compris quand la limite de couleur tracée
atteint 250 cm de haut ou 135 cm de large — entrée **INC-53** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

## Ouvrant 76281 / 76275 avec V266.Z, battement 76471, 76472, 76473

Limites de couleur, identiques sur les deux abaques : blanc 235 cm jusqu'à 95 cm de large, IR-Reflex 225 cm jusqu'à 98 cm, standard 195 cm jusqu'à 100 cm ; limite oblique jusqu'à (130 ; 130) pour le blanc et l'IR-Reflex, (120 ; 130) pour le standard ; largeur maxi 130 cm (blanc, IR-Reflex, hauteur mini 104 cm) et 120 cm (standard, hauteur mini 96 cm) ; zones séparées à 75 cm de largeur et 130 cm de hauteur [1 p. 109].

![Combinaisons ouvrant-battement de l'abaque 76281 / 76275 avec V266.Z](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76281-v266z-combinaisons.png)

### 0,8 kN/m²

![Abaque 76281 / 76275 avec V266.Z, 0,8 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76281-v266z-0-8.png)

| Largeur d'ouvrant (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- |
| 50 | 250 | 208 |
| 60 | 243 | 196 |
| 70 | 234 | 188 |
| 80 | 225 | 181 |
| 90 | 219 | 176 |
| 100 | 214 | 172 |
| 110 | 208 | 168 |
| 120 | 204 | 166 |
| 130 | 200 | 164 |
| 140 | 198 | 163 |
| 150 | 196 | 162 |

### 1,2 kN/m²

![Abaque 76281 / 76275 avec V266.Z, 1,2 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76281-v266z-1-2.png)

| Largeur d'ouvrant (cm) | V317 (cm) | V316 (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- | --- | --- |
| 50 | - | 251 | 226 | 182 |
| 60 | - | 242 | 218 | 172 |
| 70 | 249 | 232 | 209 | 165 |
| 80 | 241 | 226 | 203 | 160 |
| 90 | 236 | 218 | 196 | 155 |
| 100 | 230 | 212 | 190 | 152 |
| 110 | 225 | 206 | 185 | 149 |
| 120 | 220 | 202 | 180 | 148 |
| 130 | 217 | 199 | 178 | 147 |
| 140 | 214 | 197 | 176 | 146 |
| 150 | 211 | 195 | 175 | 146 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 109, registre 2.3.3, p. 12, version janvier 2016)

À 0,8 kN/m², les vignettes 1 (V317) et 2 (V316) n'ont pas de courbe dans le cadre de l'abaque. À 1,2 kN/m², la courbe V317 commence à 250 cm pour 68 cm de large.

## Ouvrant 76272 / 76279 avec V326.Z ou V314.Z, et 76281 / 76275 avec V266.Z, battement 76473 sans renfort

Limites de couleur, identiques sur les deux abaques : mêmes limites de couleur et mêmes coins que l'abaque précédent ; zones A, B, C (renforcement vertical 2 ouvrants) et D (renforcement total, sauf battement) seulement [1 p. 110].

![Combinaisons ouvrant-battement de l'abaque 76272 / 76279 avec V326.Z ou V314.Z, et 76281 / 76275 avec V266.Z](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v326z-v314z-76473-combinaisons.png)

### 0,8 kN/m²

![Abaque 76272 / 76279 avec V326.Z ou V314.Z, et 76281 / 76275 avec V266.Z, 0,8 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v326z-v314z-76473-0-8.png)

| Largeur d'ouvrant (cm) | V326Z (cm) |
| --- | --- |
| 50 | 250 |
| 60 | 241 |
| 70 | 233 |
| 80 | 226 |
| 90 | 218 |
| 100 | 213 |
| 110 | 208 |
| 120 | 204 |
| 130 | 200 |
| 140 | 198 |
| 150 | 196 |

### 1,2 kN/m²

![Abaque 76272 / 76279 avec V326.Z ou V314.Z, et 76281 / 76275 avec V266.Z, 1,2 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v326z-v314z-76473-1-2.png)

| Largeur d'ouvrant (cm) | V314Z (cm) | V326Z (cm) |
| --- | --- | --- |
| 50 | 251 | 226 |
| 60 | 241 | 215 |
| 70 | 232 | 206 |
| 80 | 226 | 198 |
| 90 | 218 | 192 |
| 100 | 212 | 187 |
| 110 | 207 | 183 |
| 120 | 203 | 180 |
| 130 | 200 | 178 |
| 140 | 198 | 176 |
| 150 | 196 | 175 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 110, registre 2.3.3, p. 13, version janvier 2022)

À 0,8 kN/m², la vignette V314Z n'a pas de courbe dans le cadre de l'abaque.

## Ouvrant 76271 avec V306.Z, battement 76471, 76472

Limites de couleur, identiques sur les deux abaques : blanc 235 cm jusqu'à 105 cm de large, IR-Reflex 215 cm jusqu'à 116 cm, standard 180 cm jusqu'à 100 cm ; limite oblique jusqu'à (135 ; 180) pour le blanc et l'IR-Reflex, largeur maxi 135 cm, hauteur mini 108 cm ; pour le standard, coin à 120 cm de large vers 145 à 147 cm de haut, largeur maxi 120 cm, hauteur mini 96 cm ; zones séparées à 75 cm et 130 cm [1 p. 111].

![Combinaisons ouvrant-battement de l'abaque 76271 avec V306.Z](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v306z-combinaisons.png)

### 0,8 kN/m²

![Abaque 76271 avec V306.Z, 0,8 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v306z-0-8.png)

| Largeur d'ouvrant (cm) | V316 (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- | --- |
| 50 | - | 246 | 200 |
| 60 | - | 238 | 191 |
| 70 | - | 231 | 183 |
| 80 | 247 | 223 | 176 |
| 90 | 242 | 217 | 171 |
| 100 | 236 | 211 | 167 |
| 110 | 231 | 206 | 164 |
| 120 | 227 | 201 | 162 |
| 130 | 224 | 197 | 161 |
| 140 | 219 | 194 | 160 |
| 150 | 216 | 191 | 159 |

### 1,2 kN/m²

![Abaque 76271 avec V306.Z, 1,2 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v306z-1-2.png)

| Largeur d'ouvrant (cm) | V317 (cm) | V316 (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- | --- | --- |
| 50 | - | 247 | 222 | 176 |
| 60 | - | 238 | 210 | 167 |
| 70 | 246 | 230 | 201 | 160 |
| 80 | 239 | 222 | 193 | 155 |
| 90 | 233 | 214 | 187 | 151 |
| 100 | 228 | 208 | 182 | 148 |
| 110 | 223 | 203 | 178 | 146 |
| 120 | 218 | 199 | 176 | 144 |
| 130 | 214 | 196 | 173 | 143 |
| 140 | 212 | 193 | 172 | 143 |
| 150 | 209 | 192 | 171 | 143 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 111, registre 2.3.3, p. 14, version janvier 2016)

À 0,8 kN/m², la vignette 1 (V317) n'a pas de courbe dans le cadre ; la courbe V316 commence à 250 cm pour 73 cm de large. Le coin standard se lit à 147 cm sur l'abaque 0,8 et à 144 cm sur l'abaque 1,2, écart inférieur à la précision de lecture.

## Ouvrant 76271 avec V307.Z, battement 76471, 76472

Limites de couleur, identiques sur les deux abaques : blanc 250 cm jusqu'à 100 cm de large, IR-Reflex 235 cm jusqu'à 108 cm, standard 200 cm jusqu'à 100 cm ; limite oblique jusqu'à (135 ; 180) pour le blanc et l'IR-Reflex, largeur maxi 135 cm, hauteur mini 108 cm ; standard jusqu'à (120 ; 160), largeur maxi 120 cm, hauteur mini 96 cm ; zones séparées à 75 cm et 130 cm [1 p. 112].

![Combinaisons ouvrant-battement de l'abaque 76271 avec V307.Z](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v307z-combinaisons.png)

### 0,8 kN/m²

![Abaque 76271 avec V307.Z, 0,8 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v307z-0-8.png)

| Largeur d'ouvrant (cm) | V316 (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- | --- |
| 50 | - | - | 218 |
| 60 | - | 249 | 206 |
| 70 | - | 243 | 198 |
| 80 | - | 236 | 190 |
| 90 | 248 | 229 | 184 |
| 100 | 242 | 222 | 180 |
| 110 | 237 | 217 | 176 |
| 120 | 234 | 213 | 173 |
| 130 | 231 | 209 | 171 |
| 140 | 228 | 206 | 169 |
| 150 | 226 | 204 | 168 |

### 1,2 kN/m²

![Abaque 76271 avec V307.Z, 1,2 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v307z-1-2.png)

| Largeur d'ouvrant (cm) | V317 (cm) | V316 (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- | --- | --- |
| 50 | - | - | 234 | 191 |
| 60 | - | 247 | 226 | 181 |
| 70 | - | 239 | 217 | 173 |
| 80 | 246 | 230 | 209 | 167 |
| 90 | 241 | 225 | 202 | 162 |
| 100 | 235 | 219 | 197 | 159 |
| 110 | 231 | 214 | 192 | 156 |
| 120 | 226 | 209 | 189 | 154 |
| 130 | 223 | 206 | 186 | 153 |
| 140 | 219 | 203 | 184 | 152 |
| 150 | 216 | 201 | 183 | 152 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 112, registre 2.3.3, p. 15, version janvier 2016)

À 0,8 kN/m², la vignette 1 (V317) n'a pas de courbe dans le cadre.

## Ouvrant 76271 avec V308, battement 76471, 76472

Limites de couleur, identiques sur les deux abaques : blanc et IR-Reflex 250 cm jusqu'à 100 cm de large, standard 215 cm jusqu'à 100 cm ; limite oblique jusqu'à (135 ; 180) pour le blanc et l'IR-Reflex, largeur maxi 135 cm, hauteur mini 108 cm ; standard jusqu'à 120 cm de large vers 182 cm de haut, largeur maxi 120 cm, hauteur mini 96 cm ; zones séparées à 75 cm et 130 cm [1 p. 113].

![Combinaisons ouvrant-battement de l'abaque 76271 avec V308](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v308-combinaisons.png)

### 0,8 kN/m²

![Abaque 76271 avec V308, 0,8 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v308-0-8.png)

| Largeur d'ouvrant (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- |
| 50 | - | 229 |
| 60 | - | 218 |
| 70 | - | 208 |
| 80 | 244 | 200 |
| 90 | 238 | 194 |
| 100 | 232 | 188 |
| 110 | 227 | 184 |
| 120 | 223 | 181 |
| 130 | 220 | 179 |
| 140 | 217 | 177 |
| 150 | 214 | 176 |

### 1,2 kN/m²

![Abaque 76271 avec V308, 1,2 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76271-v308-1-2.png)

| Largeur d'ouvrant (cm) | V317 (cm) | V316 (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- | --- | --- |
| 50 | - | - | 245 | 201 |
| 60 | - | - | 235 | 190 |
| 70 | - | 245 | 227 | 181 |
| 80 | - | 237 | 219 | 175 |
| 90 | 245 | 231 | 212 | 170 |
| 100 | 239 | 226 | 206 | 166 |
| 110 | 234 | 221 | 201 | 163 |
| 120 | 231 | 216 | 197 | 161 |
| 130 | 228 | 213 | 194 | 159 |
| 140 | 225 | 210 | 192 | 158 |
| 150 | 222 | 207 | 190 | 158 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 113, registre 2.3.3, p. 16, version janvier 2016)

À 0,8 kN/m², les vignettes 1 (V317) et 2 (V316) n'ont pas de courbe dans le cadre ; la courbe de la vignette 3 commence à 250 cm pour 75 cm de large.

## Ouvrant 76272 / 76279 avec V326.Z, battement 76471, 76472

Limites de couleur, identiques sur les deux abaques : blanc et IR-Reflex 250 cm jusqu'à 110 cm de large, standard 220 cm jusqu'à 110 cm ; limite oblique jusqu'à 135 cm de large vers 188 cm de haut pour le blanc et l'IR-Reflex, largeur maxi 135 cm, hauteur mini 108 cm ; standard jusqu'à 120 cm de large vers 194 cm, largeur maxi 120 cm, hauteur mini 96 cm ; zones séparées à 75 cm et 130 cm [1 p. 114].

![Combinaisons ouvrant-battement de l'abaque 76272 / 76279 avec V326.Z](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v326z-combinaisons.png)

### 0,8 kN/m²

![Abaque 76272 / 76279 avec V326.Z, 0,8 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v326z-0-8.png)

| Largeur d'ouvrant (cm) | vignette 4 (cm) |
| --- | --- |
| 50 | 250 |
| 60 | 241 |
| 70 | 233 |
| 80 | 226 |
| 90 | 220 |
| 100 | 213 |
| 110 | 208 |
| 120 | 204 |
| 130 | 200 |
| 140 | 198 |
| 150 | 196 |

### 1,2 kN/m²

![Abaque 76272 / 76279 avec V326.Z, 1,2 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v326z-1-2.png)

| Largeur d'ouvrant (cm) | V316 (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- | --- |
| 50 | - | - | 227 |
| 60 | - | - | 216 |
| 70 | - | 250 | 206 |
| 80 | - | 244 | 198 |
| 90 | 250 | 236 | 192 |
| 100 | 243 | 231 | 187 |
| 110 | 238 | 226 | 183 |
| 120 | 235 | 222 | 180 |
| 130 | 231 | 218 | 178 |
| 140 | 228 | 215 | 176 |
| 150 | 226 | 213 | 175 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 114, registre 2.3.3, p. 17, version janvier 2016)

À 0,8 kN/m², seule la vignette 4 a une courbe dans le cadre.

## Ouvrant 76272 / 76279 avec V314.Z, battement 76471, 76472

Limites de couleur, identiques sur les deux abaques : mêmes limites de couleur et mêmes coins que l'abaque du V326.Z ; zones séparées à 75 cm et 130 cm [1 p. 115].

![Combinaisons ouvrant-battement de l'abaque 76272 / 76279 avec V314.Z](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v314z-combinaisons.png)

### 0,8 kN/m²

![Abaque 76272 / 76279 avec V314.Z, 0,8 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v314z-0-8.png)

| Largeur d'ouvrant (cm) | vignette 4 (cm) |
| --- | --- |
| 50 | - |
| 60 | 248 |
| 70 | 240 |
| 80 | 233 |
| 90 | 228 |
| 100 | 222 |
| 110 | 217 |
| 120 | 212 |
| 130 | 208 |
| 140 | 205 |
| 150 | 203 |

### 1,2 kN/m²

![Abaque 76272 / 76279 avec V314.Z, 1,2 kN/m²](/assets/procedures/moe-76-advanced/abaque-deux-vantaux-76272-v314z-1-2.png)

| Largeur d'ouvrant (cm) | V316 (cm) | vignette 3 (cm) | vignette 4 (cm) |
| --- | --- | --- | --- |
| 50 | - | - | 234 |
| 60 | - | - | 226 |
| 70 | - | - | 214 |
| 80 | - | 249 | 206 |
| 90 | - | 244 | 199 |
| 100 | 249 | 238 | 194 |
| 110 | 245 | 234 | 190 |
| 120 | 241 | 230 | 186 |
| 130 | 238 | 227 | 184 |
| 140 | 235 | 224 | 182 |
| 150 | 232 | 221 | 180 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 115, registre 2.3.3, p. 18, version janvier 2016)

À 0,8 kN/m², seule la vignette 4 a une courbe dans le cadre ; elle commence à 250 cm pour 59 cm de large.

# AluClip Pro avec capot A072

Sur la variante **AluClip Pro**, l'ouvrant 76271 est capoté par l'aluminium **A072** (IW 0,8 cm⁴,
IG 1,5 cm⁴) sans renfort acier. « Remarque : Le renforcement des ouvrants n'est fondamentalement
pas necéssaire néanmoins à partir d'un poids d'ouvrant de 40 kg il faut impérativement rajouter
l'équerre J079 dans les 4 angles. » [1 p. 116]

![Abaque AluClip Pro avec capot A072](/assets/procedures/moe-76-advanced/abaque-aluclip-pro-a072.png)

Une seule limite, « Dimension maxi » : 250 cm de haut jusqu'à 100 cm de large, oblique jusqu'à
(150 ; 150), verticale à 150 cm de large jusqu'à 120 cm de haut, puis la règle des 25 %.

| Largeur d'ouvrant (cm) | Verre 12 mm (cm) | Verre 16 mm (cm) | Verre 20 mm (cm) | Verre 24 mm (cm) | Verre 28 mm (cm) |
| --- | --- | --- | --- | --- | --- |
| 50 | - | - | - | - | - |
| 60 | - | - | - | - | - |
| 70 | - | - | - | - | - |
| 80 | - | - | - | - | 229 |
| 90 | - | - | 243 | 204 | 176 |
| 100 | - | 239 | 193 | 163 | 141 |
| 110 | - | 195 | 158 | 133 | 116 |
| 120 | 214 | 163 | 132 | 112 | 97 |
| 130 | 181 | 138 | 112 | 95 | 83 |
| 140 | 155 | 119 | 97 | 83 | 72 |
| 150 | 135 | 104 | 85 | 72 | 63 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 116, registre 2.3.3, p. 19, version juillet 2017)

## AluClip Pro à deux vantaux

Deux abaques, un par battement, avec les courbes 0,8 et 1,2 kN/m² ; l'ouvrant 76271 est capoté
A072 des deux côtés. Limite « dimension maxi » commune : 250 cm de haut jusqu'à 100 cm de large,
oblique jusqu'à (135 ; 180), verticale à 135 cm jusqu'à 108 cm, puis la règle des 25 % [1 p. 117].

![AluClip Pro, battement 76472 avec V317](/assets/procedures/moe-76-advanced/abaque-aluclip-pro-a072-battement-76472-v317.png)

| Largeur d'ouvrant (cm) | 0,8 kN/m² (cm) | 1,2 kN/m² (cm) |
| --- | --- | --- |
| 50 | - | 250 |
| 60 | - | 240 |
| 70 | - | 231 |
| 80 | 249 | 222 |
| 90 | 241 | 216 |
| 100 | 236 | 209 |
| 110 | 230 | 204 |
| 120 | 225 | 199 |
| 130 | 221 | 195 |
| 140 | 217 | 192 |
| 150 | 214 | 190 |

**Le titre de cet abaque écrit « Battement 76471 avec renfort V316 »**, alors que la coupe et sa
légende dessinent le battement **76472** avec le renfort **V317** (IW 4,8 cm⁴, IG 2,3 cm⁴) —
entrée **INC-54**. La courbe 0,8 kN/m² commence à 250 cm pour 80 cm de large.

![AluClip Pro, battement 76471 avec V316](/assets/procedures/moe-76-advanced/abaque-aluclip-pro-a072-battement-76471-v316.png)

| Largeur d'ouvrant (cm) | 0,8 kN/m² (cm) | 1,2 kN/m² (cm) |
| --- | --- | --- |
| 50 | - | 244 |
| 60 | - | 232 |
| 70 | - | 221 |
| 80 | 243 | 212 |
| 90 | 235 | 205 |
| 100 | 228 | 200 |
| 110 | 222 | 195 |
| 120 | 217 | 192 |
| 130 | 214 | 189 |
| 140 | 210 | 187 |
| 150 | 208 | 185 |

Battement **76471** avec renfort **V316** (IW 2,4 cm⁴, IG 0,4 cm⁴). La courbe 0,8 kN/m² commence
à 250 cm pour 70 cm de large.

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 117, registre 2.3.3, p. 20, version avril 2018)

# Couleurs et ventilation

Les profilés de couleur, laqués ou filmés ont leurs propres limites dans les abaques ci-dessus (catégories de couleur). Leur fabrication demande en plus la ventilation des préchambres extérieures, et le DTD fixe le seuil de clarté L\* < 82 qui rend le renfort et la décompression obligatoires : ces règles sont portées par [Drainage, décompression et ventilation du système 76 Advanced](/procedures/drainage-decompression-ventilation-systeme-76.md).

# Points de verrouillage

**Deux distances maximales différentes entre points de verrouillage** figurent dans le même
registre :

| Page | Formulation du manuel | Valeur |
| --- | --- | --- |
| PDF p. 100, registre 2.3.3, p. 3 | « Les points de verrouillage (paumelle ) ne doivent pas être distants de plus de 70 cm. » | 70 cm |
| PDF p. 102, registre 2.3.3, p. 5 | « Les points de verrouillage (galets etc.) ne doivent pas être distants de plus de 80 cm. » | 80 cm |

Entrée **INC-10** du registre [Incohérences internes](/anomalies/incoherences-internes.md).

# Poids d'ouvrant admissible selon la ferrure

« Action de la ferrure : Il faut utiliser des ferrures adaptées aux poids d'ouvrants
correspondants. - Respecter la notice du fabricant ! - Respecter les remarques générales concernant
la ferrure (reg.1.3.4) ! » [1 p. 102]

« Les poids d'ouvrants admissibles indiqués dans le tableau suivant ont été déterminés à l'ift
Rosenheim selon la directive TBDK. Ces valeurs sont valables uniquement pour les composants
indiqués dans les rapports d'essais (compas, vis, acier) et le traitement décrit et elles peuvent
être considérées comme des valeurs indicatives. » Une ligne par configuration ; l'acier est celui
du dormant ou du meneau, les vis celles de la ferrure.

| Ferrure | Acier dans le dormant/meneau | Vis dans l'acier | Vis dans le PVC | Poids d'ouvrant admissible (kg) | Rapport d'essais |
| --- | --- | --- | --- | --- | --- |
| 100 kg | sans | - | 4 | 80 | 12-002529-PR01 PB-K20-09-de-01 |
| 100 kg | court (55 mm) | 1 | 3 | 90 | 12-002529-PR02 PB-K20-09-de-01 |
| 100 kg | long (5 mm) | 3 | 1 | 100 | 12-002529-PR03 PB-K20-09-de-01 |
| 130 kg | sans | - | 6 | 100 | 12-002529-PR04 PB-K20-09-de-01 |
| 130 kg | court (55 mm) | 2 | 4 | 110 | 12-002529-PR05 PB-K20-09-de-01 |
| 130 kg | long (5 mm) | 5 | 1 | 130 | 12-002529-PR06 PB-K20-09-de-01 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, p. 102, registre 2.3.3, p. 5, version janvier 2016)

La longueur « long (5 mm) » de l'acier ne peut pas être celle d'un acier plus long que le court de
55 mm — entrée **VER-25** du registre [Informations à vérifier](/anomalies/informations-a-verifier.md).

« En votre qualité de fabricant, vous êtes responsable du contrôle des composants que vous utilisez
et vous devez assurer dans votre production que les éléments possèdent les caractéristiques
correspondantes aux prescriptions de contrôle. » Encadré « Attention ! » : « Composants utilisés
ferrure, acier, vis. **Les poids d'ouvrants maximaux doivent être contrôlés et garantis par le
fabricant de fenêtres !** Faites déterminer les poids d'ouvrants admissibles pour les composants
que vous utilisez selon la directive TBDK en vigueur. Assurez dans votre production que les
éléments répondent aux caractéristiques de performance contrôlées (WPK). » [1 p. 102] Voir
[Roto NX](/quincaillerie/roto-nx.md).

# Ce que la source ne donne pas

- Les valeurs des courbes entre deux graduations de 10 cm, et au-delà de 250 cm de hauteur
- La largeur exacte des coins de limite qui ne tombent pas sur une graduation (95, 98, 105, 108,
  116 cm : lecture à ±2 cm)
- Le nom des combinaisons des vignettes 3 et 4 des abaques à deux vantaux

# Citations

[1] Mise en œuvre Système 76 Advanced, profine —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registre 2.3.3 « Abaques dimensionnels
d'ouvrant », p. 98 à 117 du PDF (pages imprimées 1 à 20), versions janvier 2016 à janvier 2022


# Voir aussi

- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Types d'ouverture et plans de combinaison du système 76](/profiles/systeme-76-plans-de-combinaison.md)
- [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md)
- [Directives générales profine](/sources/profine-directives-generales.md)
- [Incohérences internes](/anomalies/incoherences-internes.md)
