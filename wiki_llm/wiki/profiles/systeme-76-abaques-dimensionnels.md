---
type: Profilé
title: 'Abaques dimensionnels du système 76 : dormant, lecture, couleur, règle des 25 %'
description: 'Dimensions maximales de dormant par couleur, lecture des abaques d''ouvrant : charge du vent, équerre J079, épaisseur de verre, règle des 25 %, catégories de couleur, nombre de paumelles, zones de renforcement.'
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
  at: 2026-10-21T11:04:22Z
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

Autres parties de ce sujet :

- [Abaques d'ouvrant simple du système 76 : hauteur maximale par renfort et verre](/profiles/systeme-76-abaques-dimensionnels-ouvrant-simple.md)
- [Abaques du système 76 à deux vantaux avec battement 76471, 76472 et 76473](/profiles/systeme-76-abaques-dimensionnels-deux-vantaux-battement.md)
- [Abaques AluClip Pro du système 76, verrouillage et poids d'ouvrant selon la ferrure](/profiles/systeme-76-abaques-dimensionnels-aluclip-pro-ferrure.md)
