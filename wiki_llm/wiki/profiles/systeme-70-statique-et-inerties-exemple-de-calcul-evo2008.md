---
type: Profilé
title: Exemple de calcul statique, fers plats et tubes acier, système 70 (e.VOLUTION 2008)
description: Exemple de calcul du registre 4.2 de 2008 pour le système 70 (châssis 240 × 200 cm, meneau 2425, traverse 6127), détails, fiches de calcul Iz, formulaire récapitulatif vierge et méthode des fers plats et tubes acier.
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

Cette page reprend, du manuel Mise en œuvre Système 70 Plateforme et du classeur e.VOLUTION d'août 2008, l'exemple de calcul du registre 4.2 de 2008, ses fiches de calcul Iz et la méthode des fers plats et des tubes acier. Le contexte commun est dans [Calcul statique du système 70 : méthode, formule générale, plans de charge et exemples](/profiles/systeme-70-statique-et-inerties.md).

## Exemple de calcul (2008)

Le registre 4.2 porte le même exemple que le manuel de 2023, transcrit plus haut sous *Exemple de
calcul par les tables* : bâtiment en zone 1, situation b, hauteur 18 à 28 m, PV = 800 Pa =
0,80 kN/m² (V\*A2) ; méthode a) à g) ; châssis 240 × 200 cm ; axe 1 : L = 200 cm, a = 70, b = 50,
3,39 + 2,69 = 6,08 cm⁴ ; axe B : L = 140 cm, a = 40, b = 60, 0,71 + 0,89 = 1,60 cm⁴ ; meneau
2425 / 9132, Iz dispo 9,1 cm⁴ pour 6,08 ; traverse 6127 / V603, Iz dispo 3,0 cm⁴ pour 1,6 ;
même remarque finale sur la flexion maximale [2 p. 164-165]. Il en diffère sur ces points :

| Point | Plans e.VOLUTION de 2008 | Manuel de 2023 |
| --- | --- | --- |
| Grandeur lue sur l'abaque (étape f) et résultats | Iz | Iw |
| Ouvrant de la « possibilité de construction » | F91-06- 6121 | 6112 |
| Dormant | F91-01- 6101 | 6101 |
| Code imprimé du meneau 2425 | F40-15- 2425 p. 165 (F95-15- 2425 aux p. 158 et 168, **INC-224**) | - |
| Détails 1, 2, 3 du châssis (p. 168) | ouvrant 6121 + V058 sur dormant 6101 + V601 ; 6121 + V058, meneau 6127 + V603, 6121 + V058 ; 6121 + V058, meneau 2425 + 9132, 6121 + V058 | 6112 + V158 au lieu de 6121 + V058, mêmes dormant et meneaux |

![Châssis de l'exemple, 240 × 200 cm, axes et coupes (2008)](/assets/profiles/systeme70/statique/exemple-chassis-240x200-evo2008.png)

Le châssis mesure 240 cm (140 + 100) sur 200 cm (80 + 120) ; les cercles 1, 2, 3 repèrent les
liaisons, B l'axe de la traverse, 1 l'axe du meneau ; la coupe verticale est à droite, la coupe
horizontale en bas [2 p. 164].

![Plans de charge de l'axe 1 et de l'axe B (2008)](/assets/profiles/systeme70/statique/exemple-plans-de-charge-axes-1-et-b-evo2008.png)

Les surfaces hachurées sont les plans de charge, tracés à 45° depuis les angles : hachures
obliques pour le meneau (axe 1), horizontales pour la traverse (axe B). Remarque : a et b peuvent
avoir la valeur MAXI de L/2 [2 p. 165].

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 164-165)

Le formulaire récapitulatif rempli (« EXEMPLE ») porte : système e.VOLUTION, blanc ; zone 1,
situation b, hauteur 18/28 m, pression 0,8 kN/m² ; axes 1 et b : portées 200 et 140, largeurs de
l'élément A 140 et 80, B 100 et 120, largeurs de charge a 70 et 40, b 50 et 60 ; moments
d'inertie I (a) 3,39 et 0,71, I (b) 2,69 et 0,89, I total 6,08 et 1,60 ; solution : profilé 2425,
renfort 9132, axe 1 ; profilé 6127, renfort V603, axe B ; acier coché, flèche MAXI L/150 cochée.
La formule y est écrite Iz = w · L⁴ · a / (1920 · 10³ · E · f zul.) · [25 − 40 (a/L)² +
16 (a/L)⁴] [cm⁴]. Sous le formulaire : « REMARQUE — Pour le calcul se reporter ou registre
statique du CD Technique KÖMMERLING. » [2 p. 166]

![Formulaire récapitulatif rempli, exemple (2008)](/assets/profiles/systeme70/statique/formulaire-exemple-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 166)

**Les modules d'élasticité y sont imprimés « Acier 210 00 N/mm² » et « Alu 70 00 N/mm² », et la
ligne des largeurs de charge « Largeur de cahsrge » (INC-225)** ; les modules retenus sont ceux de
*2.2.4 Valeur de E* plus haut.

### Détails de l'exemple (2008)

Les trois détails repérés sur le châssis, dessinés à l'échelle 1:2 avec le code complet de
chaque profilé et de son renfort [2 p. 168] :

| Détail | Profilés et renforts, codes imprimés | Cotes portées (mm) | Coupe |
| --- | --- | --- | ---: |
| 1 | ouvrant F91-06- 6121 + F00-40- V058 sur dormant F91-01- 6101 + F00-40- V601 | 45 + 64 ; profondeur 70 | ![Détail 1 : ouvrant 6121 et dormant 6101 (2008)](/assets/profiles/systeme70/statique/exemple-detail-1-6121-6101-evo2008.png) |
| 2 | F91-06- 6121 + F00-40- V058, meneau F91-15- 6127 + F00-40- V603, F91-06- 6121 + F00-40- V058 | 45 + 80 + 45 | ![Détail 2 : ouvrants 6121 et meneau 6127 (2008)](/assets/profiles/systeme70/statique/exemple-detail-2-6121-6127-evo2008.png) |
| 3 | F91-06- 6121 + F00-40- V058, meneau F95-15- 2425 + F00-40- 9132, F91-06- 6121 + F00-40- V058 | 45 + 90 + 45 | ![Détail 3 : ouvrants 6121 et meneau 2425 (2008)](/assets/profiles/systeme70/statique/exemple-detail-3-6121-2425-evo2008.png) |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 168)

### Fiches de calcul Iz sur meneau et sur traverse (2008)

Deux fiches (client, chantier, date) donnent l'exemple rempli et rappellent, pour trois
classements, la pression w (kN/m²) et la flèche f (cm) ; la formule est écrite
Iz = w\*L⁴\*a/(1920.10³\*E\*f)\*[25-40\*(a/L)²+16\*(a/L)⁴], avec w pression en kN/m², L portée
en cm, a largeur de charge, f flèche maxi, E module d'élasticité en N/mm² [2 p. 167].

| Fiche | Classement imprimé | w (kN/m²) | f (cm) |
| --- | --- | --- | --- |
| meneau (H = 200 cm) | V\* A2 | 0,8 | 1,33 |
| meneau (H = 200 cm) | V\* A3 | 1,2 | 1,33 |
| meneau (H = 200 cm) | V\* A3 | 1,6 | 1,33 |
| traverse (L = 140 cm) | V\* A2 | 0,8 | 0,933 |
| traverse (L = 140 cm) | V\* A3 | 1,2 | 0,933 |
| traverse (L = 140 cm) | V\* A3 | 1,6 | 0,933 |

**Le troisième classement de chaque fiche est imprimé « V\* A3 » avec w = 1,6, pression de la
classe 4 (INC-226)** ; le manuel de 2023 écrit V\*A4 à cette ligne (tableau *Fiches de calcul Iw*
plus haut).

Fiche meneau : H = 200 cm, L1/2 = 70 cm, L2/2 = 50 cm, E = 210000 N/mm², classement V\*A2,
I 1 = 3,39 cm⁴, I 2 = 2,69 cm⁴, Iz = 6,08 cm⁴. Fiche traverse : L = 140 cm, H1/2 = 40 cm,
H2/2 = 60 cm, E = 210000 N/mm², V\*A2, I 1 = 0,71 cm⁴, I 2 = 0,89 cm⁴, Iz = 1,60 cm⁴ [2 p. 167].

![Fiche meneau : plan de charge (2008)](/assets/profiles/systeme70/statique/fiche-meneau-plan-de-charge-evo2008.png)

Le rectangle est la partie de menuiserie de hauteur H et de largeur L = L1 + L2 ; le meneau est
l'axe vertical, son plan de charge l'hexagone hachuré de largeur L1/2 + L2/2.

![Fiche traverse : plan de charge (2008)](/assets/profiles/systeme70/statique/fiche-traverse-plan-de-charge-evo2008.png)

La traverse est l'axe horizontal de portée L entre deux parties de hauteurs H1 et H2 ; son plan de
charge est l'hexagone hachuré de hauteur H1/2 + H2/2.

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 167)

### Formulaire récapitulatif vierge et plans de charge (2008)

Le formulaire vierge reprend la mise en page de l'exemple, sans aucune valeur, avec les mêmes
impressions (« 210 00 », « 70 00 », « cahsrge ») [2 p. 169].

Les quatre exemples de plans de charge pour traverse et meneau sont ceux du manuel de 2023
(*Exemples de plans de charge pour traverse et meneau* plus haut), avec la même remarque « a ou
b < L/2 » ; la page de l'exemple (p. 165) et le formulaire écrivent « a et b peuvent avoir la
valeur MAXI de L/2 » et « a/b ≤ max. L/2 ! » [2 p. 165-166, 170].

![Meneau filant et traverse (2008)](/assets/profiles/systeme70/statique/plan-de-charge-traverse-et-meneau-filant-evo2008.png)

Meneau filant et traverse arrêtée sur lui : largeurs a et b tracées à 45° ; la part repérée
par l'astérisque « Ce plan de charge n'est pas pris en compte ».

![Deux meneaux filants, plans de charge des meneaux (2008)](/assets/profiles/systeme70/statique/plan-de-charge-meneaux-filants-traverses-courtes-hautes-evo2008.png)

Deux meneaux filants et trois traverses courtes : plans de charge des meneaux hachurés.

![Traverses courtes entre meneaux filants (2008)](/assets/profiles/systeme70/statique/plan-de-charge-meneaux-filants-traverses-courtes-basses-evo2008.png)

Même composition, plans de charge des traverses courtes hachurés.

![Traverse filante sous CVR (2008)](/assets/profiles/systeme70/statique/plan-de-charge-traverse-filante-sous-cvr-evo2008.png)

Traverse filante sous CVR (coffre de volet roulant), deux meneaux et trois traverses : la
traverse filante reprend le trapèze hachuré.

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 170)

### Méthode de calcul, fers plats et tubes acier (2008)

Le registre 4.2 se termine, après les quinze tables d'inertie, sur la méthode de calcul et les
inerties des aciers du commerce, pages imprimées 49 à 55 [2 p. 186-192]. Relues en image contre
les sections *Méthode de calcul* à *Moments d'inertie des tubes en acier* plus haut :

- **généralités, formule générale, valeurs de I, Pr, S, E et f** (p. 186, 187 et haut de la
  p. 189) : même texte, même tableau des classes 0 à 5 (P1 400 à 2000 Pa, P2 200 à 1000 Pa, P3 600
  à 3000 Pa ; en-tête « P 1 en PA »), mêmes valeurs Q de 100 et 75 kg/m pour les allèges, mêmes
  modules d'élasticité (acier 210.000 N/mm² = 21.000 kgf/mm², aluminium 70.000 = 7.000, bois
  massif 10.000 = 1.000 ; PVC 2500 N/mm² non introduit), mêmes flèches L/150, L/150 et L/300. Le
  paragraphe des flèches est numéroté « 2.25 » (**INC-227**) ;
- **plans de charge, cas 1 à 5** (p. 188, « Dessins non à l'échelle ») : mêmes schémas et même
  légende S 1, S 2, S 3 que ceux reproduits plus haut ;
- **exemples n° 1 et n° 2** (p. 189) : mêmes données, mêmes schémas et mêmes calculs imprimés, y
  compris le « 5 x 300 » de l'exemple 1 (**INC-139**) et le « 0,073 m limité à 0,010 m » de
  l'exemple 2 (**INC-140**) ;
- **fers plats** (p. 190) : même table H 20 à 100 mm × B 2 à 16 mm, valeurs identiques case par
  case ;
- **tubes rectangulaires et carrés** (p. 191-192) : mêmes tubes, mêmes épaisseurs et mêmes
  valeurs (décimales imprimées avec un point), y compris les quatre inerties qui décroissent
  quand l'épaisseur croît (**VER-90**), et la même phrase « Valeurs pour profilés avec coins non
  arrondis. Les angles arrondis donnent des valeurs légèrement plus faibles. »

Les axes y sont nommés autrement : Z-Z et Y-Y, avec les inerties Iz et Iy, là où le manuel de 2023
écrit W-W et G-G, Iw et Ig. La colonne Iz de 2008 porte les valeurs de la colonne Iw de 2023, la
colonne Iy celles de la colonne Ig.

![Fer plat, axes Z-Z et Y-Y (2008)](/assets/profiles/systeme70/statique/fer-plat-iz-iy-evo2008.png)

Fer plat de largeur B et de hauteur H (mm), hachuré ; l'axe Z-Z est horizontal, l'axe Y-Y
vertical. Formule imprimée : Iz = B · H³ / 12 ; Iy = H · B³ / 12 [2 p. 190].

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 190)

![Tube rectangulaire en acier, axes Z-Z et Y-Y (2008)](/assets/profiles/systeme70/statique/tube-rectangulaire-iz-iy-evo2008.png)

Tube rectangulaire : dimension extérieure a (cotée H en bas du schéma) sur B, dimensions
intérieures h sur b, épaisseur s. Formule imprimée : Iz = B · H³/12 − b · h³/12 [cm⁴] ;
Iy = H · B³/12 − h · b³/12 [cm⁴] [2 p. 191]. Le manuel de 2023 imprime pour le tube rectangulaire
Iw = H · B³/12 − h · b³/12 : **CTR-86**.

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 191)

![Tube carré en acier, axes Z-Z et Y-Y (2008)](/assets/profiles/systeme70/statique/tube-carre-iz-iy-evo2008.png)

Tube carré de côté H sur B, épaisseur s ; même formule que le tube rectangulaire, et une seule
colonne « Iz / Iy » [2 p. 192].

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 192)

# Ce que la source ne donne pas

- Les éléments auxquels s'appliquent les flèches de 1/200 et 1/300 des tables V\*B et V\*C, hors
  la traverse d'allège (1/300).
- Les inerties Ig des meneaux et des assemblages (seules les Iw sont imprimées).
- Le tableau 2 de la FD P 20-201, auquel la méthode renvoie, n'est pas dans le manuel de 2023 ;
  le registre 4.2 des plans de 2008 reprend la FD P 20-201 avec un « Tableau 2 » des pressions P1
  (PDF p. 142) : voir [Choix des fenêtres en fonction de leur exposition,
  2008](/reference/choix-des-fenetres-exposition-au-vent-2008.md).

# Citations

[1] Mise en œuvre Système 70 Plateforme, profine, version septembre 2023 —
`raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf`, registre 2.3.4 « Statique », p. 147 à 180 du
PDF (pages imprimées 1 à 34)

[2] Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008 —
`raw/profine-plans-profiles-e-volution-2008-08.pdf`, registre 4.2 « Statique », PDF p. 140 et 157 à 192 (pages imprimées 3 et 20 à 55)

# Voir aussi

Cette page est une partie d'un sujet qui en compte 5 ; les autres :

- [Calcul statique du système 70 : méthode, formule générale, plans de charge et exemples](/profiles/systeme-70-statique-et-inerties.md)
- [Moments d'inertie des fers plats et tubes acier, valeurs statiques des meneaux, système 70](/profiles/systeme-70-statique-et-inerties-inerties-fers-plats-tubes-et-valeurs-statiques.md)
- [Exemple de calcul statique par les tables, formulaire et exigences de flèche, système 70](/profiles/systeme-70-statique-et-inerties-exemple-de-calcul-par-les-tables.md)
- [Valeurs statiques Iz des meneaux, traverses et assemblages, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-statique-et-inerties-valeurs-statiques-evo2008.md)
