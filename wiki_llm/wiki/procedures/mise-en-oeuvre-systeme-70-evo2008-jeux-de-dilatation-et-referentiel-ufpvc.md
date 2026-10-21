---
type: Procédure
title: Jeux de dilatation du PVC et référentiel de pose UFPVC, système 70 (e.VOLUTION 2008)
description: Calcul des jeux de dilatation du PVC blanc (coefficient 0,1 mm par mètre et par degré) et référentiel UFPVC de pose en applique intérieure et en tunnel des menuiseries PVC à frappe sous avis technique, repères 1 à 4.
tags: [profine, kommerling, e-volution, systeme-70, mise-en-oeuvre, pose, fixation, joint-de-raccordement, calfeutrement, tolerance, gros-oeuvre, rejingot, reservation, ufpvc, coupe-de-principe, doublage, volet-roulant, porte-d-entree]
systeme: 70
fournisseur: KÖMMERLING
usage: [pose]
famille: mise-en-oeuvre
status: draft
sources:
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008
    last_modified: 2008-08-31
source_pages:
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 329-373
generated:
  by: process:claude-code
  at: 2026-10-21T10:41:54Z
---

Cette page reprend, du classeur KÖMMERLING e.VOLUTION d'août 2008 (registre 7.1, système 70), le calcul des jeux de dilatation du PVC et le référentiel de mise en œuvre de la source UFPVC. Le contexte commun est dans [Pose du système 70 : conditions, interdictions, étapes, points de fixation (2008)](/procedures/mise-en-oeuvre-systeme-70-evo2008.md).

# Calcul des jeux de dilatation

Le PVC s'allonge quand il chauffe et se raccourcit quand il refroidit : le jeu de dilatation
est l'espace laissé autour du châssis pour que ce mouvement ne force ni la fenêtre ni la maçonnerie.
Pour ces calculs, prendre un coefficient de dilatation du PVC blanc de 0,1 mm par mètre de profilé
et par degré de variation de température (valeur maximum). En blanc, une variation de
température de 10 °C donne une dilatation de 1 mm par mètre de profilé (10 °C × 0,1) [1 p. 335].

Le classeur donne d'autres coefficients de dilatation du PVC dans ses directives générales
(**INC-273**).

Formule à appliquer : dilatation (mm) = 0,1 × L × ΔT [1 p. 335].

| Terme | Définition | Unité |
| --- | --- | --- |
| 0,1 | constante de dilatation par mètre et par °C | mm/(m·°C) |
| L | longueur du châssis | m |
| ΔT | écart de température entre l'été et l'hiver | °C |

Exemple : un châssis de 3,0 m de largeur exposé à des températures moyennes de −10 °C en hiver et
de +30 °C en été aura une dilatation globale de 0,1 × 3,0 × 40 = 12 mm [1 p. 335].

Remarque concernant la pose : cette dilatation de 12 mm se répartit de chaque côté du châssis
(6 mm de chaque côté) et correspond à une variation entre le mini l'hiver et le maxi l'été. Si le
châssis est posé à une température ambiante d'environ 15 °C, la dilatation sera [1 p. 335] :

| Saison | Écart de température (°C) | Calcul | Dilatation (mm) | Effet de chaque côté du châssis |
| --- | --- | --- | --- | --- |
| été | de 15 à 30 °C = 15 | 0,1 × 3,0 × 15 | 4,5 | « ce qui fera 2 mm de chaque côté » (allongement) |
| hiver | de −10 °C à +15 °C = 25 | 0,1 × 3,0 × 25 | 7,5 | « réduction de la largeur du châssis d'environ 4,0 mm de chaque côté » |

Les moitiés exactes seraient 2,25 et 3,75 mm (**INC-274**).

Les variations dimensionnelles des profilés PVC ne sont donc pas négligeables, et il est très
important de bien estimer ces variations afin de poser correctement les châssis PVC. Ces variations
dimensionnelles s'adressent également aux profilés accessoires (pièce d'appui, cornière
d'habillage, lame de volet roulant…). **Attention : laisser les jeux nécessaires pour garantir le bon
fonctionnement** [1 p. 335].

# Référentiel de mise en œuvre (source UFPVC)

Les pages imprimées 8 et suivantes, intitulées « Référentiel de mise en oeuvre », sont marquées
« * source UFPVC » (le registre 5.9 du même classeur nomme l'« UNION FENETRE PVC »). Leurs dessins ne sont pas à l'échelle
[1 p. 336-338].

## Pose en applique intérieure et en tunnel de menuiseries PVC à frappe faisant l'objet d'un avis technique

La pose en applique intérieure fixe le dormant contre la face intérieure du mur, en recouvrant
le bord de la baie ; la pose en tunnel place le dormant à l'intérieur de l'épaisseur de la baie,
entre les tableaux ([glossaire](/reference/glossaire.md)).

« Ce document volontairement simplifié ne se substitue pas aux travaux de référence réalisés sous
l'égide du CSTB et auxquels il conviendra de se reporter pour toutes informations complémentaires.
Les méthodes de pose figurant ci-dessous sont données à titre indicatif et ne peuvent engager la
responsabilité de KÖMMERLING. » [1 p. 336]

### Repère 1 : travaux préparatoires

La perspective montre la baie vue de l'intérieur, avec ses deux flèches de mesure (hauteur et
largeur), le niveau posé sur l'appui et les deux fils à plomb le long du tableau et du mur.

![Baie : contrôle des dimensions, du niveau de l'appui et de l'aplomb](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-1-baie-controle-niveau-aplomb-evo2008.png)

1. Vérifier les dimensions de la baie en hauteur et en largeur (respect des tolérances de
   maçonnerie selon le DTU 36.1 / 37.1 de février 1985).
2. Contrôler le niveau de l'appui ainsi que l'aplomb des tableaux et du mur.
3. Installer, dans le cas d'un appui reconstitué, la lisse filante (profilé posé sur toute
   la largeur de l'appui, qui reçoit le dormant — voir la variante ci-dessous).
4. Mettre en place (si le joint d'appui est inaccessible après pose) une bande de mousse
   précomprimée ou non, imprégnée de butyl ou d'acrylique (**bitume interdit**), en la retournant de
   100 mm sur la hauteur des tableaux.

[1 p. 336]

La seconde perspective montre l'angle de l'appui et du tableau : la bande de mousse suit l'appui et
remonte de 100 mm sur le tableau.

![Bande de mousse sur l'appui, retournée de 100 mm sur le tableau](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-1-bande-mousse-appui-retour-100-mm-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 336)

### Repère 2 : pose en applique intérieure

Deux coupes : à gauche la coupe verticale sur l'appui (le dormant posé sur l'appui maçonné, la
pièce d'appui et le rejet d'eau à l'extérieur), à droite la coupe horizontale sur le tableau (le
dormant appliqué contre la face intérieure du mur). Les cercles repèrent les zones commentées. Les
annotations de la planche [1 p. 337] :

- « Etanchéité mise en place avant pose de la menuiserie » (sous le dormant, sur l'appui) ;
- « Dans cette zone, habillage selon principe arrêté au devis » (côté intérieur, sur l'appui et sur
  le tableau) ;
- « Mastic sur fond de joint » (entre le dormant et le tableau) ;
- « Cote d'appui minimum 13 mm », cotée sur la coupe du tableau entre le nu du tableau et le bord
  du dormant.

![Pose en applique intérieure : coupe sur appui et coupe sur tableau](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-2-pose-applique-interieure-evo2008.png)

#### Variante : appui reconstitué

Quand l'appui est reconstitué, le dormant repose sur une lisse filante. À gauche la coupe verticale,
à droite la perspective de la lisse dans la baie. Les annotations de la planche [1 p. 337] :

- « Mousse imprégnée ou mastic sur fond de joint » (entre la lisse et le dormant, et le long de la
  lisse sur la perspective) ;
- « Obturation de la rainure aux extrémités » ;
- « Dans cette zone, habillage selon principe arrêté au devis » ;
- pente de l'appui « ≥ 35 % » ; « Pour les angles inférieurs à 35 %, mettre en place une bavette de
  recouvrement » (tôle ou profilé qui recouvre l'appui) ;
- « Pénétration de la lisse derrière les tableaux. (Longueur lisse = Largeur fenêtre) » ;
- « Lisse filante (si acier, galvanisation Z 275) ».

![Variante appui reconstitué : coupe et lisse filante](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-2-variante-appui-reconstitue-lisse-filante-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 337)

### Pose en tunnel et fixation (planche sans repère)

Deux coupes : à gauche la coupe verticale sur l'appui (le dormant fixé par une vis dans l'appui), à
droite la coupe horizontale sur le tableau (le dormant fixé latéralement dans le tableau). Les
annotations de la planche [1 p. 338] :

- « Etanchéité réalisé après pose de la menuiserie » (sur l'appui et sur le tableau) ;
- « Dans cette zone, remplissage ou habillage selon principe arrêté au devis » ;
- « Fixation par vis + cheville métallique ou vis spéciale sans cheville » ;
- « Centrage par vérins ou calage temporaire ou définitif selon mode de fixation ».

![Pose en tunnel : coupe sur appui et coupe sur tableau](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-3-pose-en-tunnel-evo2008.png)

La vue de face montre une fenêtre à trois vantaux dans son dormant ; les petits rectangles autour
du dormant sont les éléments de fixation ou de calage (pattes, vérins), placés au droit des axes de
rotation et des points de condamnation.

![Éléments de fixation et de calage autour du dormant](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-3-elements-de-fixation-et-calage-evo2008.png)

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 338)

1. Mettre en place, si nécessaire, les éléments de fixation ou de calage (pattes, vérins) en
   les positionnant au droit des axes de rotation et des points de condamnation (les points de
   verrouillage), et en respectant une distance d'environ 100 mm par rapport à l'angle intérieur
   des assemblages soudés ainsi que l'entraxe maximum de 800 mm entre deux fixations (**700 mm
   admissible pour les dormants renforcés**).
2. Placer la fenêtre dans la baie.
3. Effectuer le calage pour la centrer et assurer le jeu minimum de 5 mm nécessaire à la
   réalisation du cordon d'étanchéité. En traverse basse, les cales d'assise (cales sur
   lesquelles repose le dormant) seront positionnées au droit des montants latéraux et
   intermédiaires s'ils existent.
4. Solidariser la fenêtre au gros œuvre à l'aide de fixations de dimensions et de nature
   appropriées.
5. Vérifier, comme stipulé au DTU 36.1, que les 2 mm de tolérance de pose sont respectés
   horizontalement (3 mm au-delà de 1,5 m) et verticalement dans les deux plans.

[1 p. 338]

L'entraxe de 800 mm (700 mm pour les dormants renforcés) et le jeu de 5 mm ne sont pas ceux des
généralités du même registre (70 cm, 10 mm au minimum) : **INC-275**.

### Repère 3 : contrôle, calfeutrement, finition et nettoyage

La vue de face montre la fenêtre à deux vantaux posée dans la baie, les éléments de fixation autour
du dormant et, en pointillé, les diagonales qui servent à contrôler l'équerrage des vantaux.

![Contrôle de l'alignement et du fonctionnement des vantaux](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-3-controle-alignement-vantaux-evo2008.png)

1. Contrôler l'alignement, le recouvrement sur dormant (la hauteur dont l'ouvrant chevauche
   le dormant, cotée sur le petit croquis ci-dessous) et le bon fonctionnement des vantaux.
2. Si nécessaire, procéder au réglage en agissant sur les excentriques de la ferrure (galets de
   crémone).

![Recouvrement de l'ouvrant sur le dormant](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-3-recouvrement-sur-dormant-evo2008.png)

3. Insérer dans l'interstice périphérique entre le gros œuvre et la menuiserie un fond de joint
   en mousse de section appropriée, puis réaliser le cordon d'étanchéité à l'aide d'un mastic
   élastomère (section minimum 5 mm × 5 mm) ou plastique 1ʳᵉ catégorie (section minimum
   5 mm × 8 mm) ayant fait préalablement l'objet d'un essai d'adhésivité cohésion (se référer à
   l'avis technique de la fenêtre ou au procès-verbal).
4. Serrer le mastic pour assurer son bon contact et sa continuité, et parfaire l'esthétique du
   cordon en le lissant à l'aide d'une spatule ou d'une pomme de terre tranchée et d'eau légèrement
   savonneuse.

La coupe horizontale sur le tableau montre le dormant et, dans le cercle, la zone où l'habillage
intérieur est posé : « Dans cette zone, habillage selon principe arrêté au devis ».

![Coupe sur tableau : zone d'habillage selon devis](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-3-coupe-tableau-habillage-evo2008.png)

| Mastic du cordon d'étanchéité | Section minimum (mm) |
| --- | --- |
| élastomère | 5 × 5 |
| plastique 1ʳᵉ catégorie | 5 × 8 |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 339)

Finition : réaliser les travaux de finition intérieure selon le principe arrêté au devis ;
poser la bavette de recouvrement, si nécessaire, en cas d'appui reconstitué.

Nettoyage : débarrasser la fenêtre des copeaux qui pourraient subsister et faire disparaître
les taches à l'aide d'eau savonneuse ou de produits recommandés ; laisser les lieux dans l'état de
propreté où on les a trouvés.

[1 p. 339]

### Repère 4 : réception et entretien

Le dessin montre le poseur qui présente la fenêtre ouverte à la cliente.

![Réception des travaux par le client](/assets/procedures/mise-en-oeuvre-evo2008/ufpvc-4-reception-evo2008.png)

Réception : faire réceptionner les travaux par le client et lui expliquer comment manœuvrer
convenablement la fenêtre.

Entretien : informer le client sur l'entretien de la fenêtre (« Voir fiche d'entretien » ; les
conseils d'entretien du classeur sont sur
[Usage, nettoyage et entretien d'une menuiserie profine](/entretien/usage-nettoyage-entretien-menuiseries-profine.md)).
[1 p. 339]

# Citations

[1] [Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008](raw/profine-plans-profiles-e-volution-2008-08.pdf)

# Voir aussi

Cette page est une partie d'un sujet qui en compte 7 ; les autres :

- [Pose du système 70 : conditions, interdictions, étapes, points de fixation (2008)](/procedures/mise-en-oeuvre-systeme-70-evo2008.md)
- [Symboles et instructions de pose en neuf d'une menuiserie du système 70 (e.VOLUTION 2008)](/procedures/mise-en-oeuvre-systeme-70-evo2008-symboles-et-instructions-de-pose-neuf.md)
- [Gros œuvre, appui, rejingot et types de pose en coupe horizontale, système 70 (2008)](/procedures/mise-en-oeuvre-systeme-70-evo2008-gros-oeuvre-appui-rejingot-et-types-de-pose.md)
- [Réservations maçonnerie selon le doublage, pose en feuillure et en applique, système 70](/procedures/mise-en-oeuvre-systeme-70-evo2008-reservations-maconnerie-et-doublages.md)
- [Coupes de principe de pose du système 70 : feuillure, doublages, porte d'entrée (2008)](/procedures/mise-en-oeuvre-systeme-70-evo2008-coupes-de-principe.md)
- [Exemples de pose en neuf et pose avec volet roulant, système 70 (e.VOLUTION 2008)](/procedures/mise-en-oeuvre-systeme-70-evo2008-exemples-et-volet-roulant.md)
