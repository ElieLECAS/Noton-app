---
type: Profilé
title: Exemple de calcul statique par les tables, formulaire et exigences de flèche, système 70
description: Exemple chiffré de dimensionnement du renfort d'un meneau et d'une traverse par les tables du système 70 (zone 1, situation b, pression 800 Pa), formule de la charge trapézoïdale, formulaire de calcul et exigences de flèche.
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

Cette page reprend, du manuel Mise en œuvre Système 70 Plateforme et du classeur e.VOLUTION d'août 2008, l'exemple de calcul par les tables, la formule de la charge trapézoïdale, le formulaire de calcul et les exigences de flèche du registre 4.2 de 2008. Le contexte commun est dans [Calcul statique du système 70 : méthode, formule générale, plans de charge et exemples](/profiles/systeme-70-statique-et-inerties.md).

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

# Exigences de flèche du registre 4.2 des plans e.VOLUTION de 2008

Le registre 4.2 « Statique » du classeur e.VOLUTION d'août 2008 s'ouvre sur le choix des classes
A\*E\*V\* selon l'exposition (zones NV 65, situation, hauteur) et sur les exigences de rigidité et
de flèche sous la pression P1 [2 p. 140] ; ce texte et ses tableaux sont sur [Choix des fenêtres
en fonction de leur exposition, 2008](/reference/choix-des-fenetres-exposition-au-vent-2008.md).

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
- [Valeurs statiques Iz des meneaux, traverses et assemblages, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-statique-et-inerties-valeurs-statiques-evo2008.md)
- [Exemple de calcul statique, fers plats et tubes acier, système 70 (e.VOLUTION 2008)](/profiles/systeme-70-statique-et-inerties-exemple-de-calcul-evo2008.md)
