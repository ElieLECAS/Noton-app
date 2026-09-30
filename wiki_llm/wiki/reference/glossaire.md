---
type: Référence
title: Glossaire des sigles et des cotes
description: Les sigles, coefficients et repères de cote employés dans la documentation PROFERM et chez ses fournisseurs, avec leur sens et la page qui les utilise.
tags: [glossaire, sigle, abreviation, cote, coefficient, vocabulaire]
status: stable
sources:
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
  - resource: raw/roto-safe-e-jonction-de-cable-2024-11.pdf
    id: roto-safe-e-jonction-de-cable
    title: Roto Safe E, jonction de câble, réf. SUG_28_FR_v3, novembre 2024
    last_modified: 2024-11-30
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
    last_modified: 2023-01-31
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
    last_modified: 2025-06-19
  - resource: raw/catalogue-general-2026-01.pdf
    id: catalogue-general-2026
    title: Catalogue menuiseries PROFERM, édition janvier 2026
    last_modified: 2026-01-31
  - resource: raw/catalogue-portes-entree-2024-03.pdf
    id: catalogue-portes-entree-2024-03
    title: Catalogue portes d'entrée PROFERM, édition mars 2024
    last_modified: 2024-03-31
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés et manuel technique, système F 91, édition août 2008
    last_modified: 2008-08-31
source_pages:
  - resource: raw/profine-directives-generales-2023-01.pdf
    pages: 4, 8-16, 47
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 120-121, 138-166, 215-216, 227-228, 243-245, 253, 257, 259-260, 307, 309, 311, 317, 321
generated:
  by: process:claude-code
  at: 2026-09-18T22:00:00Z
---

# Coefficients thermiques

Quatre coefficients circulent dans le corpus et ne se comparent pas entre eux : chacun porte sur
un objet différent. Plus la valeur est basse, plus l'objet est isolant.

| Sigle | Porte sur | Unité |
| --- | --- | --- |
| Ug | le **vitrage** seul | W/m²K |
| Uf | le **profilé** seul | W/m²K |
| Uw | la **fenêtre complète**, vitrage et menuiserie | W/m²K |
| Up | le **panneau** de porte | W/m²K |

Deux coefficients accompagnent le Uw sur les fiches produit :

| Sigle | Sens |
| --- | --- |
| Sw | facteur solaire de la fenêtre : part de l'énergie solaire transmise |
| TLw | transmission lumineuse de la fenêtre |
| Gtot(i) | facteur solaire de l'ensemble vitrage et store, selon NF EN 14501 |

Une valeur annoncée sans son sigle ne se reprend pas : un 1,0 de Uf n'est pas un 1,0 de Uw. Voir
[Performances des vitrages](/vitrages/performances-vitrages.md).

# Pièces et gestes de la menuiserie

Les mots de base d'une fenêtre PVC, employés sur toutes les pages de profilés et de pose. Une
fenêtre se compose d'un cadre fixe scellé dans le mur, le dormant, et d'un ou plusieurs cadres
mobiles, les ouvrants, qui portent le vitrage.

| Terme | Sens | Où il s'emploie |
| --- | --- | --- |
| Dormant | cadre fixe de la fenêtre, fixé dans la maçonnerie ; il reçoit l'ouvrant | [Dormants PERFORM76](/profiles/perform76-dormants.md) |
| Store intégré | store vénitien ou plissé monté dans la menuiserie, entre les parcloses, et manœuvré de l'intérieur | [Stores intégrés](/equipements/stores-integres.md) |
| Clair de parclose | ouverture visible du vitrage entre les parcloses ; c'est la cote qui borne un store intégré | [Stores intégrés](/equipements/stores-integres.md) |
| Petits bois collés | baguettes collées sur le verre pour dessiner des carreaux ; incompatibles avec un store intégré | [Stores intégrés](/equipements/stores-integres.md) |
| Ouvrant | cadre mobile qui s'ouvre, aussi appelé vantail ; il porte le vitrage et la quincaillerie | [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md) |
| Faux ouvrant | partie fixe construite avec les profilés d'un ouvrant, pour que fixe et ouvrant aient la même allure | [Meneaux PERFORM76](/profiles/perform76-meneaux.md) |
| Ouvrant droit, ouvrant galbé | ouvrant à face plane ou à face arrondie ; sur la PERFORM76, 76 mm et 83 mm d'épaisseur | [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md) |
| Aile | rebord du dormant qui vient recouvrir le mur ou l'ancien bâti ; un dormant sans aile se pose dans l'embrasure | [Dormants PERFORM76](/profiles/perform76-dormants.md) |
| Neuf, rénovation | pose dans une maçonnerie neuve, ou pose sur l'ancien dormant bois conservé | [Pose de la PERFORM76](/procedures/pose-perform76.md) |
| Délignage | recoupe de l'aile d'un dormant pour la raccourcir, faite sur le chantier | [Dormants PERFORM76](/profiles/perform76-dormants.md) |
| Battement | profil vertical où se rejoignent les deux ouvrants d'une fenêtre à deux vantaux | [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md) |
| Joint de butée | joint contre lequel l'ouvrant vient s'appuyer en fermeture ; sur l'HYBRIDE de 72 mm, périphérique dans le dormant, l'ouvrant et entre les parcloses et le vitrage | [HYBRIDE](/gammes/hybride.md) |
| Cintrage, cintré | mise en forme courbe d'un profilé ; une menuiserie cintrée a une partie haute arrondie | [Coloris HYBRIDE](/coloris/coloris-hybride.md) |
| Ouvrant caché | ouvrant dissimulé derrière le dormant, vu de l'extérieur | [PERFORM+](/gammes/perform-plus.md), [LUMINE](/gammes/lumine.md) |
| Gâche, galet | la gâche est fixée sur le dormant ; le galet, porté par l'ouvrant et entraîné par la crémone, s'y engage pour verrouiller | [Roto NX](/quincaillerie/roto-nx.md) |
| Fiche | organe de rotation de l'ouvrant ; la PERFORM+ et l'HYBRIDE+ ne se réalisent pas avec des fiches | [PERFORM+](/gammes/perform-plus.md) |
| Meneau | profil qui divise un dormant (meneau de dormant) ou un ouvrant (meneau d'ouvrant) en plusieurs parties | [Meneaux PERFORM76](/profiles/perform76-meneaux.md) |
| Béquille double | poignée de porte ou de porte-fenêtre, présente des deux côtés, montée sur plaque ou sur rosace avec l'entrée de clé | [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md) |
| Rosace | petite pièce ronde ou ovale qui entoure la base d'une poignée ou l'entrée de clé, à la place d'une plaque | [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md) |
| Fausse crémone | tige décorative verticale apparente sur l'ouvrant | [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md) |
| Croisillon | barrette qui divise visuellement le vitrage en carreaux ; « sans croix » : les barrettes ne se croisent pas | [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md) |
| Soubassement | panneau plein en partie basse d'une porte-fenêtre, à la place du vitrage | [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md) |
| Semi-fixe | vantail d'une fenêtre à deux vantaux qui ne s'ouvre qu'après l'ouvrant principal | [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md) |
| Galandage | coulissant dont les vantaux s'effacent dans l'épaisseur du mur ou de la cloison | [Coulissants aluminium](/gammes/coulissants-aluminium.md) |
| Monobloc (porte) | porte d'entrée dont l'ouvrant et le panneau ne font qu'un | [Sélection Hexa](/portes/selection-hexa.md) |
| Imposte, tierce | partie fixe ajoutée au-dessus (imposte) ou à côté (tierce) d'une porte ou d'une fenêtre | [Collection Authentique](/portes/collection-authentique.md) |
| Grille de défense | grille en fer forgé posée devant le vitrage d'une porte, en applique sur le vitrage ou intégrée à celui-ci | [Collection Authentique](/portes/collection-authentique.md) |
| Insert inox | pièce d'acier inoxydable incrustée dans un usinage du panneau de porte ou collée sur celui-ci, sur une face (extérieure) ou deux faces | [Collection Contemporain, modèles déclinés 0, 1 et 2](/portes/collection-contemporain-declinaisons-inox.md) |
| Rainurage | décor de rainures dans la surface d'un panneau de porte | [Collection Contemporain](/portes/collection-contemporain.md) |
| Vitrage dépoli acide | verre rendu translucide par attaque à l'acide | [Collection Contemporain](/portes/collection-contemporain.md) |
| Poignée encastrée | poignée de tirage logée dans la face extérieure de l'ouvrant d'une porte, au lieu d'être posée en applique | [Collection Contemporain](/portes/collection-contemporain.md), [Collection Graphite](/portes/collection-graphite.md) |
| Dépoli sablé | verre rendu translucide par projection d'un abrasif (corindon) sous pression ; sur les modèles « PS », le fond est sablé et le motif reste transparent | [Collection Lumière](/portes/collection-lumiere.md) |
| Panneau verrier | panneau de porte entièrement vitré, posé dans l'ouvrant | [Collection Lumière](/portes/collection-lumiere.md) |
| Warm Edge | intercalaire isolant qui sépare les verres d'un vitrage isolant | [Collection Lumière](/portes/collection-lumiere.md) |
| Plaxé, plaxage | profilé ou panneau PVC revêtu d'un film décor ; s'oppose au teinté dans la masse et au laqué | [DTA 6/16-2334](/certifications/dta-6-16-2334.md), [Collection Contemporain](/portes/collection-contemporain.md) |
| Traverse | profil horizontal ; la traverse de soubassement sépare le vitrage du soubassement | [Meneaux PERFORM76](/profiles/perform76-meneaux.md) |
| Soubassement | partie basse d'une fenêtre ou d'une porte-fenêtre, sous la traverse | [Meneaux PERFORM76](/profiles/perform76-meneaux.md) |
| Parclose | baguette clipsée qui maintient le vitrage dans son logement ; elle se choisit par l'épaisseur du vitrage | [Parcloses PERFORM76](/profiles/perform76-parcloses.md) |
| Feuillure | logement en creux du profilé qui reçoit le vitrage ou l'ouvrant | [Parcloses PERFORM76](/profiles/perform76-parcloses.md) |
| Renfort | profil acier glissé dans une chambre du PVC pour le rigidifier | [Renforts du système 76](/profiles/systeme-76-renforts.md) |
| Chambre | chacun des compartiments creux du profilé PVC ; plus il y en a, plus le profilé isole | [PERFORM](/gammes/perform.md) |
| Tapée | profil rapporté sur le dormant pour épaissir la menuiserie jusqu'au nu de l'isolant intérieur | [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md) |
| Appui | profil sous le dormant bas, penté vers l'extérieur, qui rejette l'eau de pluie | [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md) |
| Nez d'appui | petit profil clipsé au bord de l'appui | [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md) |
| Seuil | profil bas d'une porte-fenêtre, sur lequel on passe | [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md) |
| Rejet d'eau | profil qui écarte l'eau du bas de l'ouvrant | [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md) |
| Compensateur | profil qui comble l'écart entre le dormant rénovation et l'ancien bâti | [Appuis et seuils PERFORM76](/profiles/perform76-appuis-et-seuils.md) |
| Capot aluminium, AluClip | profil aluminium clippé sur la face extérieure d'un profilé PVC, qui donne un aspect aluminium à l'extérieur de la menuiserie | [Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md) |
| Cadre fixe | dormant qui reçoit directement le vitrage, sans ouvrant | [Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md) |
| Coupe droite, grugeage | deux préparations du montant qui s'assemble sur un seuil : coupé d'équerre, ou entaillé pour épouser le profil du seuil | [Assemblages du système 76](/profiles/systeme-76-assemblages.md) |
| Assemblage en T, en X | traverse qui aboutit sur un profilé (T) ou qui en croise un autre (X) | [Assemblages du système 76](/profiles/systeme-76-assemblages.md) |
| Élargisseur | profil accolé au dormant pour élargir la menuiserie | [Élargisseurs et assemblage PERFORM76](/profiles/perform76-elargisseurs-et-assemblage.md) |
| Patte de pose, équerre de fixation | pièce métallique qui fixe le dormant au mur à travers l'isolant | [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md) |
| Clameau | pièce d'accrochage de la patte de pose sur le dormant | [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md) |
| Cale latérale | cale posée entre le dormant et la maçonnerie pour le positionner avant fixation | [Pose de la PERFORM76](/procedures/pose-perform76.md) |
| Compribande | bande de mousse imprégnée précomprimée qui gonfle dans le joint entre dormant et maçonnerie | [Pose de la PERFORM76](/procedures/pose-perform76.md) |
| Fond de joint | cordon placé au fond du joint avant le silicone, pour en régler la profondeur | [Pose de la PERFORM76](/procedures/pose-perform76.md) |
| Recouvrement | largeur sur laquelle une pièce en recouvre une autre : l'ouvrant sur le dormant, le dormant sur le mur | [Pose de la PERFORM76](/procedures/pose-perform76.md) |
| Drainage, décompression | usinages qui évacuent l'eau entrée en feuillure et équilibrent la pression d'air | [Pose de la PERFORM76](/procedures/pose-perform76.md) |
| Pivot bas | ferrure du bas de l'ouvrant autour de laquelle il tourne et qui porte son poids | [Poignée et pivot PERFORM76](/quincaillerie/perform76-poignee-et-pivot.md) |
| Pivot d'angle, palier d'angle | sur ferrure Roto NX, ferrure de l'angle bas de l'ouvrant côté paumelles (pivot) et ferrure de l'angle bas du dormant sur laquelle se pose le vantail (palier) | [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md) |
| Report de charge | sur ferrure Roto NX NT Designo II, ensemble d'une pièce vissée sur l'ouvrant, d'une pièce dormant vissée sur le palier d'angle et d'une tringle de soutien du vantail, réglé par la tension d'un ressort | [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md) |
| Clé Allen, clé six pans | clé mâle coudée à section hexagonale, désignée par sa cote sur plats (4 mm) ; écrite « clé alén » sur la notice du report de charge Roto | [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md) |
| Compas OF, compas OB | ferrure de l'angle haut de l'ouvrant ; le compas OF équipe un ouvrant à la française, le compas OB un oscillo-battant, dont il retient le vantail basculé en soufflet | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Équerre de compas | pièce coudée en L de l'angle haut de l'ouvrant, déposée avec le compas OF lors d'une transformation en OB | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Têtière (de compas) | longue ferrure plate vissée le long du haut de l'ouvrant, sur laquelle se monte le compas OB | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Coulisseau, plot | sur le compas OB Roto NX, le coulisseau est la pièce du bout du bras de compas qui se pose sur la têtière ; le plot est le téton du petit bras à lumière qui se relie au bras de compas | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Platine anti-rabattement | position finale de la platine du compas Roto NX, nommée « anti-rabattement » sur les schémas ; voir VER-109 | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Obturateur de manœuvre | petite pièce en plastique logée dans une lumière de la ferrure d'un ouvrant à la française, retirée « afin de libérer la manœuvre OB » | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Lumière | fente oblongue percée dans une ferrure ou un bras | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Soufflet, ouverture à soufflet | ouverture par basculement du vantail : le haut s'écarte vers l'intérieur, le bas reste tenu ; sur un oscillo-battant, elle s'ajoute à l'ouverture à la française | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Gâche OB | gâche vissée en traverse basse du dormant lors d'une transformation OF en OB, en version droite ou gauche ; voir VER-110 | [Transformation OF en OB ROTO NX](/procedures/transformation-of-en-ob-roto-nx.md) |
| Coupe verticale, coupe horizontale | dessin de la menuiserie comme tranchée de haut en bas, ou de gauche à droite, vu en bout | toutes les planches |
| Élévation | dessin de la menuiserie vue de face | [Meneaux PERFORM76](/profiles/perform76-meneaux.md) |
| Pose en applique, en tableau, en tunnel | le dormant fixé contre une face du mur, dans l'épaisseur de la baie contre un épaulement, ou dans l'épaisseur de la baie sans épaulement | [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) |
| Rejingot | ressaut de la maçonnerie sous l'appui, qui arrête l'eau | [Fabrication et assemblage du système 76 Advanced](/procedures/fabrication-systeme-76-advanced.md) |
| Pose en feuillure, en ébrasement, au nu intérieur | le dormant logé dans la feuillure (ressaut) des tableaux ; dans la feuillure avec des tableaux intérieurs évasés (ébrasement) ; aligné sur le plan de la face intérieure du mur | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| Calfeutrement | remplissage et étanchement du joint entre le dormant et le gros œuvre | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| Précadre | cadre de montage posé d'abord dans la baie, sur lequel la fenêtre est ensuite fixée | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| Réservation (maçonnerie) | ouverture laissée par le maçon pour recevoir la menuiserie, cotée en hauteur et largeur tableau fini | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| Face dressée | face du mur rendue plane pour recevoir le dormant en pose en applique | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| Lisse filante | profilé posé sur toute la largeur d'un appui reconstitué, qui reçoit le dormant ; « si acier, galvanisation Z 275 » | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| Coupe sur montant, coupe sur appui | coupe horizontale au droit d'un côté vertical du dormant ; coupe verticale au droit de sa traverse basse posée sur l'appui | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| Joint comprimé | bande serrée entre le dormant et la maçonnerie, dessinée derrière le joint de mastic sur les coupes de principe de 2008 | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| SNJF | sigle du label des mastics cité avec les joints élastomère « 1ère catégorie SNJF » ; il n'est pas développé dans le classeur e.VOLUTION | [Mise en œuvre du système 70, 2008](/procedures/mise-en-oeuvre-systeme-70-evo2008.md) |
| Menuiserie à frappe | fenêtre dont l'ouvrant vient battre contre le dormant (ouvrant à la française, oscillo-battant) | [Mise en œuvre en rénovation du système 70, 2008](/procedures/mise-en-oeuvre-renovation-systeme-70-evo2008.md) |
| Fourrure bois (rénovation) | pièce de bois traité logée dans la feuillure de l'ancien dormant conservé, pour créer une surface plane sous la nouvelle menuiserie | [Mise en œuvre en rénovation du système 70, 2008](/procedures/mise-en-oeuvre-renovation-systeme-70-evo2008.md) |
| Arasement (de la contre-feuillure) | coupe à ras de la partie saillante de l'ancien dormant bois | [Mise en œuvre en rénovation du système 70, 2008](/procedures/mise-en-oeuvre-renovation-systeme-70-evo2008.md) |
| Tapée de persienne | pièce de bois de l'ancien dormant qui porte les persiennes, conservée ou non en rénovation | [Mise en œuvre en rénovation du système 70, 2008](/procedures/mise-en-oeuvre-renovation-systeme-70-evo2008.md) |
| Bride en équerre | patte métallique en équerre qui fixe le dormant sur le rejingot | [Mise en œuvre en rénovation du système 70, 2008](/procedures/mise-en-oeuvre-renovation-systeme-70-evo2008.md) |
| Châssis à l'italienne | nom donné en 2008 à la menuiserie à ouverture extérieure | [Mise en œuvre en rénovation du système 70, 2008](/procedures/mise-en-oeuvre-renovation-systeme-70-evo2008.md) |
| Monomur | mur en blocs isolants porteurs, posé sans doublage | [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) |
| Bavette | tôle d'aluminium sous la traverse basse qui rejette l'eau au-delà du nu du mur | [Fabrication et assemblage du système 76 Advanced](/procedures/fabrication-systeme-76-advanced.md) |
| Habillage | cornière rapportée qui recouvre le raccord entre menuiserie et mur | [Profilés complémentaires du système 76](/profiles/systeme-76-profiles-complementaires.md) |
| Fourrure d'épaisseur | autre nom de la tapée PVC dans les documents profine | [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md) |
| Joint à la pompe | mastic injecté au pistolet dans un angle ou une jonction | [Fabrication et assemblage du système 76 Advanced](/procedures/fabrication-systeme-76-advanced.md) |
| Contre-profilage | usinage de l'extrémité d'un meneau, d'une traverse ou d'un montant à la forme du profilé qui le reçoit (dormant, seuil) | [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md) |
| Entretoise | tube (S048, S049, S050 sur le système 76) logé dans le profilé qui reçoit un meneau, pour que la vis d'assemblage n'écrase pas ses chambres ; il remplace le renfort | [Fabrication et assemblage du système 76 Advanced](/procedures/fabrication-systeme-76-advanced.md) |
| Alvéovis | logement de vis extrudé dans le nez d'une fourrure d'épaisseur, dans lequel se visse la pièce d'appui | [Tapées et isolation PERFORM76](/profiles/perform76-tapees-et-isolation.md) |
| Paumelle | charnière d'une porte ou d'une fenêtre ; en porte d'entrée, « drapeau » (fixée en applique sur la face de l'ouvrant) ou « tube » (cylindre vertical) | [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md) |
| Pêne dormant | pêne de serrure manœuvré par la clé seule | [Ouvrants de porte d'entrée](/portes/ouvrants-de-porte.md) |
| Galet | roulette de verrouillage qui se loge dans une gâche | [Ouvrants de porte d'entrée](/portes/ouvrants-de-porte.md) |
| Gâche | pièce fixée sur le dormant qui reçoit le pêne, le galet ou le crochet de la serrure | [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md) |
| Fouillot | pièce de la serrure qui reçoit le carré de la béquille | [Ouvrants de porte d'entrée](/portes/ouvrants-de-porte.md) |
| Béquille | poignée qui actionne la serrure d'une porte | [Accessoires de porte d'entrée](/quincaillerie/accessoires-portes-entree.md) |
| Bâton de tirage | poignée fixe, sans mécanisme, que l'on tire pour ouvrir ou fermer la porte | [Accessoires de porte d'entrée](/quincaillerie/accessoires-portes-entree.md) |
| Heurtoir | anneau ou pièce articulée fixée sur la porte pour y frapper | [Accessoires de porte d'entrée](/quincaillerie/accessoires-portes-entree.md) |
| Cimaise | moulure en relief rapportée horizontalement sur le panneau de porte | [Collection Classique](/portes/collection-classique.md) |
| Panneau (porte à panneau) | porte dont le panneau est rapporté dans un cadre ouvrant, avec un effet « escalier » au raccord | [Panneaux et monoblocs](/portes/panneaux-et-monoblocs.md) |
| Double frappe | deux battues d'étanchéité entre l'ouvrant et le seuil ou le dormant | [Ouvrants de porte d'entrée](/portes/ouvrants-de-porte.md) |
| Têtière (de serrure) | longue plaque métallique de la serrure de porte, qui affleure sur le chant du vantail et d'où sortent les pênes ; elle se loge dans une rainure fraisée à sa largeur | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Chant (du vantail) | tranche du vantail, côté serrure, où se fraise le logement de la têtière | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Pêne | pièce de la serrure qui sort de la têtière pour s'engager dans la gâche | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Entraxe E (E92) | sur une serrure de porte, distance entre l'axe du fouillot et l'axe du cylindre : E92 = 92 mm | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Cylindre (de serrure) | barillet dans lequel se tourne la clé | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Axe de fraisage | ligne verticale, sur le dormant, sur laquelle se centrent les fraisages des gâches ; elle dépend du profil utilisé | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Passage de câble | paire de pièces, l'une dans le dormant, l'autre dans l'ouvrant, par laquelle les fils électriques passent du dormant à la serrure motorisée | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Boîte noire (Eneo) | unité intérieure du contrôle d'accès 4 en 1, placée à l'intérieur, qui porte le bouton Reset | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Contrôle d'accès 4 en 1 (4in1) | boîtier extérieur qui commande l'ouverture d'une serrure motorisée par code PIN, empreinte digitale, smartphone Bluetooth ou support RFID | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| RFID, eKey | RFID : identification par radiofréquence, lecture sans contact d'un badge ou d'un porte-clés ; eKey : clé virtuelle attribuée à un smartphone | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Contact libre de potentiel (contact sec) | contact de relais qui ouvre ou ferme un circuit sans fournir lui-même de tension | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Contact reed | interrupteur magnétique, qui se ferme en présence d'un aimant | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| IN1, IN2, GND | sur un plan de câblage, entrées de commande 1 et 2 et masse (0 V) | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Transformateur, entrée primaire / secondaire | le transformateur Eneo reçoit le secteur (100 à 240 V AC) sur son primaire et rend du 24 V DC sur son secondaire | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Mode jour, mode nuit (Eneo) | modes de fonctionnement de la serrure Eneo CC, commutés par l'entrée IN2 ; en mode jour, la serrure ne se verrouille pas automatiquement | [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md) |
| Jonction de câble (Roto Safe E) | passage de câble Roto : une pièce dormant, une pièce d'ouvrant et une connexion enfichable démontable à 6 broches entre les deux | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| Connexion enfichable, douille, connecteur | prise démontable entre dormant et ouvrant ; la douille porte les six contacts femelles côté ouvrant, le connecteur les six broches côté dormant | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| Jeu en feuillure | espace libre entre le dormant et l'ouvrant fermé, dans la feuillure ; il décide de la pièce d'ouvrant de la jonction de câble (12 / 16 mm ou 4 / 12 mm) | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| Coffret de réception | boîtier de la pièce d'ouvrant 820194, encastré dans l'ouvrant, qui porte le circuit imprimé et les borniers | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| Spirale, ressort métallique | gaine en ressort qui protège le câble entre dormant et ouvrant ; il ne faut ni la tordre ni tirer dessus | [Montage de la jonction de câble Roto Safe E](/procedures/montage-jonction-de-cable-roto-safe-e.md) |
| Bloc d'alimentation intégré | alimentation secteur 230 V → 24 V logée dans la pièce dormant de la jonction de câble (2045681, 2045682) | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| Connecteur JST | petite prise du câble qui relie le lecteur extérieur (empreinte, 4 en 1) à la boîte noire et à la serrure | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| K1a, K1b | les deux bornes du contact libre de potentiel de la serrure E610 / E611 (fils gris et rose) | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| IP67 | indice de protection : totalement protégé contre la poussière (6) et contre l'immersion temporaire (7) | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| DEL, LED | diode électroluminescente, petit voyant lumineux | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| AC, DC | courant alternatif (le secteur) et courant continu (ce que reçoit la serrure) | [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) |
| DIN 107 | norme allemande qui définit la main (droite ou gauche) d'une porte ou d'une fenêtre ; les dessins Roto sont en version à droite | [Montage de la jonction de câble Roto Safe E](/procedures/montage-jonction-de-cable-roto-safe-e.md) |

# Cotes de fabrication

DHT, CCD, CCO et CCV sont définis par les planches du registre 1.1.2 des directives générales
profine, où chacun est porté sur une fenêtre et sur un coulissant — voir
[Terminologie et légendes profine](/reference/terminologie-et-legendes-profine.md) [1 p. 11-16].
DEO et DFO sont les cotes des tableaux de cotes de débit des manuels de système.

| Sigle | Sens |
| --- | --- |
| DHT | Dimension Hors Tout, la dimension extérieure du dormant |
| DEO | dimension extérieure d'ouvrant |
| DFO | dimension de feuillure d'ouvrant |
| CCD | Cote clair de dormant |
| CCO | Cote clair d'ouvrant |
| CCV | Cote clair de vitrage |

Les quatre premières se déduisent en cascade, chaque tableau de cotes de débit donnant la valeur
à retrancher pour **une seule coupe**. Voir
[Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md).

**Les cotes maximales des registres profine sont des cotes extérieures d'ouvrant** : la cote
d'élément (cote finale de fenêtre) s'obtient en ajoutant les cotes des profilés adjacents à tous
les côtés [1 p. 9] — exemple chiffré sur
[Terminologie et légendes profine](/reference/terminologie-et-legendes-profine.md).

Repères de cote de la planche de terminologie profine [1 p. 10] :

| Repère | Notion | Repère | Notion |
| --- | --- | --- | --- |
| A | Dormant | J | Hauteur joint comprimé |
| B | Ouvrant | K | Recouvrement ouvrant |
| C | Parclose | L | Hauteur de calage vitrage |
| D | Rainure crémone | M | Hauteur pré-cale |
| E | Dos de dormant | N | Jeu de fonctionnement |
| F | Chambre des renforts | O | Dimension extérieure dormant |
| G | Rainure parclose | P | Clair de jour vitrage |
| H | Feuillure vitrage | Q | Clair de jour ouvrant |
| I | Feuillure quincaillerie | R | Clair de jour dormant |

# Cotes de quincaillerie

Vocabulaire des catalogues et manuels [ROTO](/fournisseurs/roto.md).

| Sigle | Sens |
| --- | --- |
| LFF | largeur de fond de feuillure d'ouvrant |
| HFF | hauteur de fond de feuillure d'ouvrant |
| LFf, HFf | largeur et hauteur de fond de feuillure, sur un châssis fixe comme sur un ouvrant, dans la prise de mesures des [stores intégrés](/equipements/stores-integres.md) |
| EV | encombrement pris par le store replié sur le vitrage, sur les [stores intégrés](/equipements/stores-integres.md) |
| PV | poids de vantail |
| FFO | hauteur d'axe de poignée au fond de la feuillure quincaillerie |
| SDB | sécurité de base, le champ d'application sans classe d'effraction |
| KSR | basculement vertical, désignation de la famille de ferrure Roto NX KSR |
| GH | hauteur de poignée, dans les instructions de montage Roto NX KSR |
| GDS | gâche de sécurité |
| AF | axe de ferrure, dans le catalogue Roto NX PVC ; voir [Légende des tableaux du catalogue Roto NX](/quincaillerie/roto-nx-legende-catalogue.md) |
| PO | poids d'ouvrant, dans le catalogue Roto NX PVC (PV dans le manuel KSR) |
| TiltSafe | position de basculement (soufflet) avec retard d'effraction de la ferrure Roto NX, classification CDR 2 / CDR 2 N, voir [Roto NX](/quincaillerie/roto-nx.md) |
| Position [n] | numéro encadré d'une pièce sur les vues d'ensemble du chapitre « Aperçu des ferrures » du catalogue Roto NX PVC, repris d'une configuration à l'autre mais pas toujours d'un côté paumelles à l'autre ([50] : compression de feuillure côté P, report de charge côté Designo), voir [Aperçu des ferrures Roto NX, côté paumelles P](/quincaillerie/roto-nx-apercu-ferrures-cote-p.md) et [côté paumelles Designo (BA 13)](/quincaillerie/roto-nx-apercu-ferrures-designo.md) |
| Designo (BA 13), NT Designo II | côté paumelles Roto NX traité dans la section 3.2 du catalogue Roto NX PVC (« Côté paumelles Designo (BA 13) ») et appelé NT Designo II dans le manuel de montage KSR ; ses champs d'application se donnent sans et avec report de charge ; voir [Aperçu des ferrures Roto NX, côté paumelles Designo (BA 13)](/quincaillerie/roto-nx-apercu-ferrures-designo.md), [Configurations Roto NX KSR — Confort et NT Designo II](/quincaillerie/roto-nx-ksr-confort-designo.md) |
| O / N | oui / non, dans les tableaux Roto |
| J | imprimé dans les colonnes oui / non des tableaux de crémones Roto (anti-fausse manœuvre, loqueteau, renvoi d'angle intégré) à la place de O ; non défini par la liste des abréviations (entrée **INC-288**), voir [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md) |
| Axe de fouillot (catalogue Roto) | cote [B] (« Fouillotss ») du schéma de perçage et de fraisage des crémones Roto NX : 8, 15, ou 25 à 50 mm ; elle classe les crémones du catalogue, voir [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md) |
| Têtière de crémone, boîtier de crémone, boîtier de serrure | crémones Roto NX à fouillot de 25 à 50 mm : le boîtier de crémone ou le boîtier de serrure (à cylindre profilé ou rond) se monte dans la têtière de crémone ; le boîtier de serrure demande une têtière « Loqueteau J », voir [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md) |
| Raccord de crémone | pièce Roto NX courte (110 mm) listée par type de fenêtre (KSR, sortie de tringle, ouvrant basculant, oscillo-battant latéral, plein cintre, semi-fixe) ; certaines références sont appelées « prolongateur de 110 mm » dans les nomenclatures KSR, voir [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md) |
| Hauteur de levier | pour une crémone de semi-fixe Roto NX, l'équivalent de la hauteur de poignée : colonne à pictogramme « hauteur de levier fixe » ou « milieu/variable » des tableaux, voir [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md) |
| Levier séparé | pièce Roto NX 291743, nécessaire pour chaque crémone de semi-fixe avec levier séparé mobile, voir [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md) |
| Renvoi d'angle spécial court | renvoi d'angle Roto NX à longueur « 110 / 10 » (260280, 260282, 281288), appelé par les premières plages de HFF des tableaux de combinaisons des crémones de semi-fixe, voir [Renvois d'angle et verrouilleurs Roto NX](/quincaillerie/roto-nx-renvois-angle-et-verrouilleurs.md) |
| Verrou d'arête | pièce courte de la section 4.7 du catalogue Roto NX PVC, pour rainure de battement (standard, KSR) ou feuillure Euro ; 305638, 633419 et 618666 sont appelés « verrou pour semi-fixe » dans le manuel KSR, voir [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md) |
| Bras de compas | pièce du compas Roto NX, dessinée avec une paumelle à son extrémité ; côté paumelles P elle se commande par système Roto, LFF, taille, finition et sens DIN, côté Designo (BA 13) par système de profil, voir [Compas, paliers et pivots Roto NX](/quincaillerie/roto-nx-compas-et-paliers.md) |
| Système de profil (catalogue Roto) | colonne des tableaux Roto NX côté Designo (BA 13) qui nomme les profilés du marché auxquels la pièce convient (« Kömmerling 76 », « Trocal 76 », « Veka Softline 70 AD »…), voir [Compas, paliers et pivots Roto NX](/quincaillerie/roto-nx-compas-et-paliers.md) |
| Verrouilleur (catalogue Roto) | ferrure Roto NX dessinée en tringle plate, portant selon la référence aucun, un ou deux galets, une gâche soudée ou un crochet ; chapitre 7 du catalogue Roto NX PVC (en plusieurs pièces, têtière, opposé, crochet, plein cintre, confort), abrégé VM, voir [Renvois d'angle et verrouilleurs Roto NX](/quincaillerie/roto-nx-renvois-angle-et-verrouilleurs.md) |
| Palier compas, broche de palier de compas | palier côté paumelles P désigné par sa version (P 3/130, P 6/130, P 6/150, P 3/100, P 6/100) ; une broche 834705 est nécessaire pour chaque palier de compas, voir [Compas, paliers et pivots Roto NX](/quincaillerie/roto-nx-compas-et-paliers.md) |
| Gâche de basculement (catalogue Roto) | gâche de la section 9.1 du catalogue Roto NX PVC, appelée « Gâche OB » [28] dans les vues d'ensemble des ferrures ; standard (zinc, acier), TiltFirst, oscillo-battant latéral, commandée par système de profil, voir [Pièces de fermeture et gâches Roto NX](/quincaillerie/roto-nx-pieces-fermeture-gaches.md) |
| Sol (catalogue Roto) | colonne « J » / « N » des tableaux de gâches Roto NX ; les gâches zinc « avec sol » sont données pour CDR 1 N, CDR 2 / CDR 2 N, « sans sol » pour la sécurité de base ; sens non défini, entrée **VER-124** |
| 2ème compas | compas supplémentaire du chapitre 10.1 du catalogue Roto NX PVC, qui comprend une pièce dormant et une pièce d'ouvrant (255237, TiltFirst 292022, plein cintre 245764), voir [Deuxième compas, compas d'arrêt, limiteurs et compas d'aération Roto NX](/quincaillerie/roto-nx-compas-complementaires.md) |
| Limiteur d'ouverture à blocage indexé | élément de confort Roto NX qui limite l'ouverture du vantail (ouverture de 90° sur le schéma) ; « absence d'élément de sécurité selon la DIN EN 13126-5 » ; appelé « limiteur d'ouverture à positions indexées » dans le manuel KSR |
| Releveur d'ouvrant | pièce Roto NX 795925, « en combinaison avec compas d'aération ou limiteur d'ouverture à blocage indexé » |
| Compas d'aération | compas du chapitre 10.6 du catalogue Roto NX PVC ; son ouverture (80 ou 140) et ses éléments d'encliquetage n° 1 à 4 se choisissent par la HFF |
| Loqueteau (catalogue Roto) | pièce qui retient le vantail fermé sans le verrouiller (manuel KSR) ; le catalogue Roto NX PVC le décline en standard, aimant et NTi, chacun en pièces de dormant, têtière et pièces d'ouvrant (chapitre 11.1), désignations différentes du manuel (**CTR-121**), voir [Accessoires et gabarits d'atelier Roto NX](/quincaillerie/roto-nx-accessoires-et-gabarits.md) |
| Support, sous-cale (catalogue Roto) | pièce de dormant de la section 11.10 « Supports » du catalogue Roto NX PVC, commandée par système de profil, à laquelle renvoient les mentions « Support adapté » et « Sous-cale adaptée, voir → à partir de la page 417 », voir [Accessoires et gabarits d'atelier Roto NX](/quincaillerie/roto-nx-accessoires-et-gabarits.md) |
| Réhausse, compression de feuillure (catalogue Roto) | pièces de la section 11.9 « Réhausses » du catalogue Roto NX PVC, en position ouvrant ou dormant, données par jeu (4, 10, 12, 13 mm), voir [Accessoires et gabarits d'atelier Roto NX](/quincaillerie/roto-nx-accessoires-et-gabarits.md) |
| Limiteur d'ouverture par pivotement | élément de confort Roto NX de la section 11.5 (191, 335 / 355, A, 198) : « absence d'élément de sécurité selon la DIN EN 13126-5 », voir [Accessoires et gabarits d'atelier Roto NX](/quincaillerie/roto-nx-accessoires-et-gabarits.md) |
| Gabarit d'insertion (catalogue Roto) | gabarit de la section 12.2 du catalogue Roto NX PVC, un par famille de crémone et par plage de HFF ou de LFF, numéroté N° 1 à N° 55, voir [Accessoires et gabarits d'atelier Roto NX](/quincaillerie/roto-nx-accessoires-et-gabarits.md) |
| SEC | sécurité |
| cf. ill. | écrit « sans illustration » dans la légende du catalogue Roto NX PVC, entrée **INC-279** |
| Raccordable | se dit d'une pièce Roto qui s'accouple à une autre (prolongateur, verrouilleur) |
| Zone de recoupe | longueur dont une pièce Roto (crémone, tringle) peut être raccourcie |
| Ergot, perçage ergot | colonnes des tableaux Roto : l'ergot est dessiné comme un plot carré en saillie, le perçage ergot comme ce plot logé dans un trou dont le diamètre est coté |
| Système 12/20-13 | désignation Roto d'un système de profilé : jeu de feuillure 12 mm / largeur de recouvrement 20 mm - axe de ferrure 13 mm, voir [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md) |
| Roto Sil, Roto Sil Level 6 | traitement de surface des ferrures Roto NX (argent mat, sans chrome VI) et son complément pour rivets, goujons et éléments coulissants, voir [Roto NX](/quincaillerie/roto-nx.md) |
| Code couleur Roto (R01.1 …) | code de commande de la teinte d'une pièce apparente Roto, voir [Finitions Roto NX](/quincaillerie/roto-nx-finitions.md) |
| QM 328 | programme de certification des ferrures de l'ift Rosenheim, voir [Certificats Roto NX](/certifications/roto-nx-certificats.md) |
| AFM | anti-fausse manœuvre : dispositif qui empêche de basculer le vantail en soufflet quand il est ouvert à la française |
| KU | accouplable (verrouilleur médian « 600 KU ») |
| VM | verrouilleur médian |
| S | loqueteau |
| F (F8, F-6, F15) | fouillot ; le nombre est l'axe du fouillot en mm, négatif pour le fouillot -6 |
| Galet E, P, V, G | galets de verrouillage Roto NX : E excentrique réglable en pression d'appui, P excentrique de sécurité, V excentrique de sécurité réglable aussi en hauteur ; G, sur la crémone de semi-fixe, n'est pas défini par le manuel ; « 2 E » = deux galets E |
| Crémone à sortie de tringle | crémone d'ouvrant à la française dont les tringles sortent en haut et en bas du boîtier et se prolongent par un prolongateur, voir [Crémones Roto NX](/quincaillerie/roto-nx-cremones.md) |
| Crémone de semi-fixe | crémone du vantail secondaire d'une fenêtre à deux vantaux, manœuvrée par un levier |
| Ferrage symétrique | pièces côté paumelles posées à l'identique sur chacun des deux vantaux (équerre et compas OF, paliers, pivots) |
| Fichier gamme | document Roto par profilé, hors du corpus, auquel renvoient les gâches et verrouilleurs invisibles des nomenclatures |
| Compas d'arrêt | compas qui retient un vantail soufflet à son ouverture maximale ; latéral ou en haut |
| Compas soufflet | nom que le catalogue Roto NX PVC de 2023 donne au compas d'arrêt de la ferrure soufflet (« 2 compas soufflet latéralement », « compas soufflet en haut ») |
| LFO, HFO | largeur et hauteur de feuillure d'ouvrant, dans le catalogue Roto NX PVC (paumelle à soufflet pour recouvrement d'ouvrant, ouvrants pivotants) |
| FFB, FFH, FG | Flügelfalzbreite, Flügelfalzhöhe, Flügelgewicht : largeur et hauteur de fond de feuillure du vantail et poids du vantail, sur les pages imprimées en allemand du catalogue Roto NX PVC (entrée **INC-13**) |
| Côté crémone, côté axe | repères [A] et [B] des diagrammes de la fenêtre inclinée du catalogue Roto NX PVC ; leur lecture n'est pas expliquée (entrée **VER-121**), voir [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md) |
| Compas d'entrebâillement et de nettoyage | compas qui limite l'ouverture d'un soufflet et permet de le rabattre pour le nettoyage |
| Tolérance de châssis fixe | encombrement de la paumelle côté paumelles P, caches compris, voir [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md) |
| Bloc d'écartement | cale posée entre la maçonnerie et le dormant d'une fenêtre de sécurité, au droit des vissages de gâche de sécurité |
| Condamnation au cylindre | verrouillage d'une porte-fenêtre par une serrure à cylindre à clé (serrure H100 Roto NX) |

**Les cotes LFF et HFF sont des cotes de feuillure de vantail**, ni des cotes de baie ni des
cotes extérieures d'ouvrant : les trois séries de limites s'appliquent en même temps et chacune
peut être la contraignante. Voir
[Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md).

# Types d'ouverture

| Sigle | Sens |
| --- | --- |
| OF | ouvrant à la française |
| OB | ouvrant oscillo-battant |

**OF ne signifie pas « ouvrant fixe ».** Tout le corpus, DTA compris, emploie OF pour l'ouvrant à
la française, y compris dans les libellés de dimensions maximales du type « 2 vantaux OF ».

# Statique et renforts

| Sigle | Sens | Unité |
| --- | --- | --- |
| IW | inertie du renfort dans la direction du vent ; borne la dimension réalisable sous une charge de vent donnée | cm⁴ |
| IG | inertie du renfort vis-à-vis du poids ; borne l'épaisseur de vitrage admissible | cm⁴ |
| Iz | notation du moment d'inertie employée par les classeurs profine antérieurs, équivalente à IW sur les planches relevées | cm⁴ |
| Iy | second moment d'inertie imprimé à côté de Iz sous le renfort des diagrammes de renforcement du classeur e.VOLUTION de 2008 (« Iy = 0.47 », « Iz = 3.19 » pour le V058) ; l'axe et l'unité ne sont pas écrits sur la planche | - |
| Pression de vent (P1, P2, P3) | force exercée par le vent sur chaque mètre carré de la fenêtre ; P1 sert à mesurer la flèche, P2 (= P1/2) est répétée, P3 (= 1,5 P1) est la pression de sécurité ; voir [Classification de la résistance au vent](/reference/classification-resistance-au-vent.md) | Pa |
| E (module d'élasticité) | rigidité propre d'une matière en flexion : acier 210 000 N/mm², aluminium 70 000 N/mm² ; entre au dénominateur de la formule du moment d'inertie requis | N/mm² |
| Valeur statique | somme des moments d'inertie des renforts d'un assemblage (deux dormants accouplés, contreventement, élargisseur) ; voir [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) | cm⁴ |
| Zone de vent (1 à 5) | découpage de la France par département et par canton selon l'exposition au vent, défini par les règles NV 65 (modificatif n° 2) ; la zone 5 regroupe les DOM ; voir [Choix des fenêtres en fonction de leur exposition, 2008](/reference/choix-des-fenetres-exposition-au-vent-2008.md) | - |
| Situation a, b, c, d | environnement de la construction pour le choix des classes A\*E\*V\* : a grands centres urbains, b villes petites et moyennes, zones industrielles ou forestières, c rase campagne, d bord de mer ou de lac | - |
| GO | gros œuvre : la maçonnerie ou la structure du bâtiment qui reçoit la menuiserie | - |
| TBDK | directive allemande de la Gütegemeinschaft Schlösser und Beschläge, qui fixe les forces de traction à certifier selon le poids de vantail | - |
| WPK | contrôle de production en usine, au titre duquel le fabricant garantit le poids d'ouvrant | - |

Un renfort à forte IW et faible IG tient le vent mais pas le triple vitrage. Voir
[Renforts du système 76](/profiles/systeme-76-renforts.md).

# Classements de performance

Le classement **A\*E\*V** du CSTB mesure trois résistances, chacune notée séparément : A pour
l'air, E pour l'eau, V pour le vent.

| Notation | Sens |
| --- | --- |
| A\*n | perméabilité à l'air, de A\*1 à A\*4, A\*4 étant la plus étanche |
| E\*nA | étanchéité à l'eau, version A ; le nombre croît avec la performance |
| V\*Ln | résistance au vent : la lettre A, B ou C donne la classe de déformation, le chiffre la pression |

Voir [Labels et certifications](/certifications/labels-et-certifications.md) pour le classement
de chaque produit.

# Résistance à l'effraction

| Notation | Sens |
| --- | --- |
| RC 1 à RC 6 | classes de résistance à l'effraction de la série NF EN 1627 à 1630 |
| CDR 1 à CDR 6 | même classification, notation employée par les documents ROTO en français |

**RC et CDR désignent la même chose.** Le catalogue Roto NX emploie les deux, en français et en
allemand, dans le même document — entrée **INC-13** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

Le vitrage feuilleté a sa propre échelle, **P1 A à P5 A** selon la norme EN 356, mesurée par la
hauteur de chute et le nombre de billes. Voir
[Panneaux et monoblocs](/portes/panneaux-et-monoblocs.md).

# Matières et traitements

| Terme | Sens |
| --- | --- |
| PVB | PolyVinylButyral, film intercalaire d'un vitrage feuilleté |
| STADIP | désignation commerciale d'un vitrage feuilleté de sécurité |
| TGI | type d'intercalaire à bord chaud d'un vitrage isolant ; le vitrage de série PERFORM76 en emploie un de coloris noir |
| EPDM | élastomère des joints d'étanchéité |
| TPE | élastomère thermoplastique, matière de joints coextrudés |
| PCE | second élastomère de joint employé par le système 70 |
| RPT | rupture de pont thermique |
| Plaxage | application d'un film décor sur le profilé PVC |
| Teinté dans la masse | PVC coloré dans toute son épaisseur à l'extrusion, sans couche rapportée ; s'oppose au plaxé et au laqué |
| Satiné, granité | les deux aspects de laque proposés sur l'aluminium ; le granité a un grain visible |
| Anodisé, anodisé laqué | l'anodisation traite la surface de l'aluminium ; l'« anodisé laqué contretypé » est une laque qui en reproduit l'aspect |
| Texture (TEXTURAL) | décor à effet de matière (bois, cuir, carbone, métal) posé sur le profilé |
| RAL | nuancier normalisé de couleurs, chaque teinte désignée par un code à quatre chiffres (9016 blanc, 7016 gris anthracite) |
| Face (1 face, 2 faces) | côté de la menuiserie qui reçoit la teinte : 1 face = l'extérieur seul, 2 faces = extérieur et intérieur |
| Composition vinylique | recette de PVC dont est extrudé un profilé, certifiée sous la marque QB 34 et désignée par un code CSTB |
| Mousse PE | mousse de polyéthylène, matière des patins G251 et G067 |
| Laquage | application d'une laque, classée par le label QUALICOAT |
| Contretypé | teinte réalisée sur mesure pour s'approcher d'une texture donnée |
| Grain d'orge | finition de soudure d'angle, par opposition à la soudure ébavurée |
| L\* | clarté colorimétrique ; le seuil **L\* inférieur à 82** déclenche le renforcement et la décompression des profilés |
| Bevel | verre biseauté serti dans un cordon de plomb, sur un vitrage décoratif |
| Low-e, faible émissivité | couche du vitrage qui renvoie le rayonnement de chaleur ; associée à un gaz argon dans le vitrage isolant |
| Dépoli acide | verre rendu translucide par attaque à l'acide ; vitrage par défaut des portes d'entrée |
| PVD | Physical Vapor Deposition, dépôt de métal sous vide qui colore une pièce (heurtoirs, boutons) |
| Fer cémenté | fer durci en surface, finition de heurtoirs et de boutons |
| AEROLAME | âme de panneau de porte aluminium, renforcée de composites alvéolaires, contre l'effet bilame |
| Effet bilame | déformation d'un panneau dont les deux faces se dilatent différemment |

# Labels de traitement de surface

| Label | Porte sur |
| --- | --- |
| QUALICOAT | laquage de l'aluminium ; la classe 2 garantit une tenue supérieure à la classe 1 |
| QUALANOD | anodisation de l'aluminium |
| GSB International | label de qualité du thermolaquage par poudre, cité avec Qualicoat pour les profilés aluminium laqués livrés par profine ([Surfaces des profilés profine](/procedures/surfaces-collage-nettoyage-profine.md)) |
| QUALIMARINE | préparation de surface de l'aluminium laqué en ambiance marine |
| CEKAL | certification des vitrages isolants |

# Documents et organismes

| Sigle | Sens |
| --- | --- |
| DTA | Document Technique d'Application, avis du CSTB sur un procédé |
| VHBE, VHBH, FPKF | directives du Groupement Qualité Serrures et Ferrures (Gütegemeinschaft Schlösser und Beschläge) : recommandations aux utilisateurs finaux, maniement des ferrures en traitement ultérieur, compas d'entrebâillement et de nettoyage, voir [Roto NX KSR — conventions et consignes de sécurité](/procedures/roto-nx-ksr-consignes-generales.md) |
| VFF | Syndicat des fabricants de fenêtres et de façades (Allemagne), auteur des directives TLE.01 et WP.01 à WP.03 |
| DTD | Dossier Technique Détaillé, pièce jointe au DTA qui porte les prescriptions de fabrication |
| GS | Groupe Spécialisé du CSTB ; le n° 6 traite les menuiseries |
| CSTB | Centre Scientifique et Technique du Bâtiment |
| NF CSTBat | marque de certification du CSTB des fenêtres ; en 2008, condition pour pratiquer la soudure à plat |
| APSAD | Assemblée Plénière des Sociétés d'Assurances Dommages, qui préconise des niveaux de protection (vitrage SP510 de la collection Lumière) |
| FFCP | Fédération Française de Construction Passive |
| UFME | Union des Fabricants de Menuiseries Extérieures |
| SNEP | Syndicat National de l'Extrusion Plastique |
| PMR | personne à mobilité réduite ; qualifie un seuil surbaissé |
| ITE | isolation thermique par l'extérieur (enduit sur isolant et/ou bardage) |
| ETICS | système d'isolation thermique extérieure par enduit sur isolant |
| FDS | Fiche de Données de Sécurité |
| DdP | déclaration des performances, exigée pour le marquage CE |
| CPU | contrôle de production en usine |
| ITT | essais de type initiaux, réalisés par des organismes notifiés sur les systèmes profine ; voir [Conditions d'utilisation des directives profine](/normes/directives-profine-conditions-d-utilisation.md) |
| Cascading ITT | essais de type initiaux en cascade (EN 14351-1) : reprise, après autorisation de profine, des résultats des ITT profine par le fabricant de fenêtres |
| profine certified | marque des composants de fournisseurs homologués par profine |

Les références internes des documents ROTO suivent trois préfixes : **IMO** pour une instruction
de montage, **CTL** pour un catalogue, **SUG** pour une notice d'emploi.

# Normes citées dans le corpus

| Norme | Objet |
| --- | --- |
| NF DTU 36.5 | mise en œuvre des fenêtres et portes extérieures |
| NF EN 12207 | classement de la perméabilité à l'air |
| NF EN 14351-1+A2 | norme produit des fenêtres, dont le contrôle de production en usine (§ 7.3) |
| NF EN ISO 11600 | classification des mastics (25 E élastomère, 12.5 P plastique) |
| FD DTU 36.5 P3 | choix des fenêtres en fonction de leur exposition |
| NF DTU 39 | mise en œuvre des vitrages |
| NF EN 1627 à 1630 | résistance à l'effraction des fenêtres et portes |
| EN 356 | résistance du vitrage feuilleté au choc |
| NF EN 14501 | classement de la protection solaire des stores |
| DIN EN 13126/8 | ferrures de fenêtre oscillo-battante, dont la protection anticorrosion |
| EN 1670 | résistance à la corrosion de la quincaillerie |
| NF P 20-302 | caractéristiques des fenêtres, conformité mécanique |
| NF P20-650-1 | mise en œuvre des vitrages |
| NF P24-351 | protection contre la corrosion des menuiseries métalliques |
| Règles NV 65 | charges de neige et de vent, référentiel antérieur à l'Eurocode NF EN 1991-1-4 |

# Pièces, usinages et montages du système 70

Le vocabulaire du manuel de mise en œuvre Système 70 Plateforme de profine [4]. Chaque terme
renvoie à la page où il est employé.

| Terme | Sens | Où il s'emploie |
| --- | --- | --- |
| Plan de combinaison | coupe cotée d'un assemblage réel — profilés, renfort, joints, vitrage — avec le cartouche des profilés et renforts admis et leur Iw | [Types d'ouverture et plans de combinaison du système 70](/profiles/systeme-70-plans-de-combinaison.md) |
| Vitrage à sec | pose du vitrage entre des joints souples (EPDM ou PCE), sans mastic | [Tableau de vitrage du système 70](/profiles/systeme-70-tableau-de-vitrage.md) |
| Cales C1, C2, C3, C4 | cale d'assise, cale périphérique ajustée au jeu, cale de solidarisation collée, cale de sécurité libre (XP P 20-650-1) | [Tableau de vitrage du système 70](/profiles/systeme-70-tableau-de-vitrage.md) |
| OF PC | ouvrant à la française plein cintre | [Tableau de vitrage du système 70](/profiles/systeme-70-tableau-de-vitrage.md) |
| Support de cales de vitrage | pièce posée en fond de feuillure qui porte les cales du vitrage (9326 sur le système 70) | [Tableau de vitrage du système 70](/profiles/systeme-70-tableau-de-vitrage.md) |
| Set d'assemblage mécanique, équerre d'assemblage | deux façons de fixer un meneau ou une traverse sans soudure, chacune avec son propre débit de renfort ; le set se pose en T ou en croix (C) | [Assemblage mécanique du meneau et de la traverse du système 70](/procedures/assemblage-meneau-traverse-systeme-70.md) |
| Équerre de fond de feuillure | équerre vissée dans les deux profilés à assembler (9714 sur le système 70) | [Assemblage mécanique du meneau et de la traverse du système 70](/procedures/assemblage-meneau-traverse-systeme-70.md) |
| Alvéovis | rainure du meneau qui reçoit la vis d'assemblage | [Assemblage mécanique du meneau et de la traverse du système 70](/procedures/assemblage-meneau-traverse-systeme-70.md) |
| Cote X | décalage du perçage d'un assemblage à angle variable, donné par une table selon l'angle | [Assemblage mécanique du meneau et de la traverse du système 70](/procedures/assemblage-meneau-traverse-systeme-70.md) |
| Préchambre | chambre extérieure d'un profilé ; sur les profilés de couleur, elle est ventilée pour éviter l'accumulation de chaleur | [Drainage, décompression et ventilation du système 70](/procedures/drainage-decompression-ventilation-systeme-70.md) |
| Soudure à plat | assemblage d'un meneau ou d'une traverse soudé en bout, à plat, sur la face du profilé qui le reçoit, au lieu d'un assemblage mécanique | [Directives générales de fabrication du système 70, 2008](/procedures/directives-generales-systeme-70-evo2008.md) |
| Miroir (de soudeuse) | plaque chauffante de la soudeuse contre laquelle fondent les bouts de profilés à souder | [Directives générales de fabrication du système 70, 2008](/procedures/directives-generales-systeme-70-evo2008.md) |
| Ragréage du cordon de soudure | reprise à l'outil du bourrelet de matière sorti de la soudure, pour rendre la surface lisse | [Directives générales de fabrication du système 70, 2008](/procedures/directives-generales-systeme-70-evo2008.md) |
| Contre-profilage | usinage du bout d'un meneau ou d'une traverse à la forme de la feuillure qui le reçoit, avant soudure ou assemblage | [Directives générales de fabrication du système 70, 2008](/procedures/directives-generales-systeme-70-evo2008.md) |
| Assemblage en T, assemblage en croix | un meneau aboutit sur un profilé (T) ; deux meneaux se rencontrent de part et d'autre d'un meneau traversant (croix) | [Assemblages mécaniques des meneaux et traverses du système 70, 2008](/procedures/assemblage-mecanique-meneau-traverse-systeme-70-evo2008.md) |
| Contour de fraisage | découpe faite en bout d'un meneau pour qu'il épouse le profil du profilé qui le reçoit | [Assemblages mécaniques des meneaux et traverses du système 70, 2008](/procedures/assemblage-mecanique-meneau-traverse-systeme-70-evo2008.md) |
| Ouvrant à déport, ouvrant à fleurant | ouvrant dont la face déborde du dormant (déport) ; ouvrant dont la face affleure celle du dormant (fleurant) | [Tableau de vitrage du système 70](/profiles/systeme-70-tableau-de-vitrage.md) |
| Profilé filmé | profilé PVC recouvert d'un film décor collé sur une ou deux faces | [Directives générales de fabrication du système 70, 2008](/procedures/directives-generales-systeme-70-evo2008.md) |
| Cale C3S | en 2008, cale de solidarisation ajoutée aux cales C1 à C3 du DTU 39, facultative, posée avec un jeu de l'ordre du millimètre | [Tableau de vitrage du système 70](/profiles/systeme-70-tableau-de-vitrage.md) |
| Battement central réduit, ouvrant réduit | montage à deux vantaux avec un battement rapporté étroit ; l'ouvrant vertical étroit qui le reçoit est dit réduit | [Traitement du battement du système 70](/procedures/traitement-du-battement-systeme-70.md) |
| Embout d'épointage | embout qui ferme la pointe délignée de l'ouvrant au battement central réduit (9F13, M771) | [Traitement du battement du système 70](/procedures/traitement-du-battement-systeme-70.md) |
| Embout de frappe | nom donné en 2008 à l'embout collé sur la pointe délignée de l'ouvrant au battement central réduit (9F13) | [Traitement du battement du système 70, 2008](/procedures/traitement-du-battement-systeme-70-evo2008.md) |
| Renvoi de manœuvre, renvoi de fouillot | pièce de quincaillerie qui déporte la commande de la poignée pour la centrer sur le battement central réduit | [Traitement du battement du système 70, 2008](/procedures/traitement-du-battement-systeme-70-evo2008.md) |
| Fenêtre basculante, basculant | fenêtre dont l'ouvrant tourne autour de deux pivots placés dans ses montants, sur un axe horizontal ; en 2008, le système 70 la construit avec l'ouvrant 2418 et des battements tubulaires 0140 | [Systèmes spéciaux du système 70, 2008](/procedures/systemes-speciaux-systeme-70-evo2008.md) |
| Coulissant à déport, coulissante à déport | châssis dont l'ouvrant coulisse le long de la partie fixe ; en 2008, ouvrants 6112, 6115, 6121, 6123 et 2416 sur meneau 2425 ou 2427 | [Systèmes spéciaux du système 70, 2008](/procedures/systemes-speciaux-systeme-70-evo2008.md) |
| Insert de soudure, insert soudable | pièce enfoncée dans le renfort en bout de profilé avant la soudure d'angle des ouvrants de porte 2415 / 2416, pour la résistance au flambage (9287) | [Systèmes spéciaux du système 70, 2008](/procedures/systemes-speciaux-systeme-70-evo2008.md) |
| Rénovation sur dormant existant | pose d'un dormant PVC contre l'ancien cadre (bois ou acier) laissé en place, recouvert côté extérieur par un profilé d'habillage clippé | [Systèmes spéciaux du système 70, 2008](/procedures/systemes-speciaux-systeme-70-evo2008.md) |
| Vantail semi-fixe | vantail d'une fenêtre à deux vantaux qui s'ouvre en second ; il porte le battement | [Traitement du battement du système 70, 2008](/procedures/traitement-du-battement-systeme-70-evo2008.md) |
| Joint brosse | bande de poils qui ferme le jeu entre le rejet d'eau d'une porte et son seuil (9C44) | [Seuil aluminium de porte 9C42 du système 70, 2008](/procedures/seuil-alu-9c42-systeme-70-evo2008.md) |
| Entrée d'air autoréglable, mortaise, lumière | bouche de ventilation posée sur la menuiserie, dont le débit reste dans une plage fixée quand la pression change ; la mortaise est l'usinage qui la reçoit, fait d'une ou plusieurs lumières (fentes oblongues) | [Intégration des entrées d'air autoréglables du système 70, 2008](/procedures/entrees-d-air-systeme-70-evo2008.md) |
| Vis plot | vis de clippage du battement intérieur (S073) | [Traitement du battement du système 70](/procedures/traitement-du-battement-systeme-70.md) |
| Renfort pré-usiné | renfort acier livré à longueur et déjà percé (V069 en 2 m, V154 en 2,25 m) | [Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md) |
| Cale de transport | pièce qui maintient l'élément pendant le transport (9A39, 9856) | [Renforts et accessoires par profilé du système 70](/profiles/systeme-70-accessoires-par-profile.md) |
| Médiant | jonction centrale de deux vantaux ; sets M628, M629 | [Renforts et accessoires par profilé du système 70](/profiles/systeme-70-accessoires-par-profile.md) |
| Ouvrant de service | vantail qui s'ouvre en premier dans une fenêtre à deux vantaux | [Renforts et accessoires par profilé du système 70](/profiles/systeme-70-accessoires-par-profile.md) |
| G/D | gauche / droite, pour les embouts et équerres livrés par paire | [Renforts et accessoires par profilé du système 70](/profiles/systeme-70-accessoires-par-profile.md) |
| Réhausse | profilé PVC assemblé au dormant pour en augmenter la hauteur (0374, 0379, 0302 …) | [Mise en œuvre des profilés complémentaires du système 70](/procedures/mise-en-oeuvre-profiles-complementaires-systeme-70.md) |
| Olive de liaison | profilé de liaison qui accouple deux dormants dos à dos (1248) | [Accouplement d'éléments du système 70](/procedures/accouplement-elements-systeme-70.md) |
| Petit bois | baguette qui partage visuellement un vitrage en carreaux ; le petit bois collé se colle sur la face du vitrage (92005 sur le système 70) | [Mise en œuvre des profilés complémentaires du système 70](/procedures/mise-en-oeuvre-profiles-complementaires-systeme-70.md) |
| Joint de frappe | joint contre lequel l'ouvrant vient battre à la fermeture ; sur le système 70, un joint de frappe dormant (9C32.T) et un joint de frappe ouvrant (9C31.T) | [Joints et garnitures des systèmes profine](/profiles/joints-et-garnitures-profine.md) |
| Gabarit de perçage | outil posé sur le profilé pour percer les trous d'un assemblage à la bonne place (9918, 9B44, 9905 sur le système 70) | [Assemblages du système 70](/profiles/systeme-70-assemblages.md) |
| Compensation réno, aile de recouvrement | l'aile de recouvrement est l'aile du dormant de rénovation qui recouvre l'ancien dormant ; la compensation (6143, 6144) rattrape l'écart sous cette aile | [Mise en œuvre des profilés complémentaires du système 70](/procedures/mise-en-oeuvre-profiles-complementaires-systeme-70.md) |
| Prolongateur | pièce qui prolonge une pièce d'appui (4319) | [Profilés complémentaires du système 70](/profiles/systeme-70-profiles-complementaires.md) |
| Sécable | se coupe à longueur suivant des amorces moulées (embouts 9F97, 9F08, 9F10) | [Mise en œuvre des profilés complémentaires du système 70](/procedures/mise-en-oeuvre-profiles-complementaires-systeme-70.md) |
| Coulisse, tulipe | guide vertical du tablier de volet roulant ; pièce d'évasement posée en tête de coulisse | [Mise en œuvre des profilés complémentaires du système 70](/procedures/mise-en-oeuvre-profiles-complementaires-systeme-70.md) |
| Grugé, gruger | usiné au contour du profilé rencontré, pour qu'il vienne s'y emboîter | [Mise en œuvre du seuil aluminium du système 70](/procedures/mise-en-oeuvre-seuil-systeme-70.md) |
| Noyau, goupille | pièce d'ancrage d'un set d'assemblage logée dans la chambre de renfort ; cheville qui la bloque dans le montant | [Mise en œuvre du seuil aluminium du système 70](/procedures/mise-en-oeuvre-seuil-systeme-70.md) |
| CHC | vis à tête cylindrique à six pans creux | [Mise en œuvre du seuil aluminium du système 70](/procedures/mise-en-oeuvre-seuil-systeme-70.md) |
| Flambage | déformation d'un profilé comprimé | [Porte d'entrée du système 70](/procedures/porte-d-entree-systeme-70.md) |
| Contreventement, habillage de contreventement | renforcement d'un meneau ou d'un couplage contre le vent ; profilé renforcé rapporté sur le meneau (93000, 93002) | [Accouplement d'éléments du système 70](/procedures/accouplement-elements-systeme-70.md) |
| Valeur statique | inertie totale d'un couplage, somme des inerties de ses renforts, en cm⁴ | [Accouplement d'éléments du système 70](/procedures/accouplement-elements-systeme-70.md) |
| Poinçonnage (d'un capot) | découpe en bout de capot aluminium qui laisse passer le capot voisin à la jonction | [Capotage AluClip du système 70](/procedures/capotage-aluclip-systeme-70.md) |
| Busette | pièce qui habille l'orifice de drainage en façade (M450, 697010) | [Capotage AluClip du système 70](/procedures/capotage-aluclip-systeme-70.md) |
| Angle de pointe | angle aigu entre deux côtés d'une menuiserie oblique (trapèze, triangle) | [Châssis cintrés et trapézoïdaux du système 70](/procedures/chassis-cintres-trapezoidaux-systeme-70.md) |
| Plan de charge | part de la surface de la menuiserie dont la pression du vent est reprise par un élément | [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) |
| Largeur de charge (a, b) | largeur du plan de charge de part et d'autre de l'élément, en cm ; entrée des tables d'inertie | [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) |
| Portée (L) | distance entre appuis d'un meneau ou d'une traverse | [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) |
| Flèche (f) | déformation maximale admissible d'un élément sous le vent : L/150, L/200 ou L/300 | [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) |
| Fer plat | barre d'acier de section rectangulaire | [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) |
| Allège, traverse d'allège | partie basse d'une baie ; sa traverse est calculée pour la sécurité des personnes | [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) |
| Meneau filant, traverse courte | élément qui va d'un bout à l'autre du dormant ; élément arrêté sur un autre | [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) |
| Mo | maître d'ouvrage | [Statique et moments d'inertie du système 70](/profiles/systeme-70-statique-et-inerties.md) |
| AC3, AC4 | classes d'exigence acoustique ; le renforcement systématique y est conseillé | [Abaques dimensionnels de renforcement du système 70](/profiles/systeme-70-abaques-dimensionnels.md) |
| Renvoi d'angle, sortie de tringle | deux modes de transmission du verrouillage de la ferrure | [Abaques dimensionnels de renforcement du système 70](/profiles/systeme-70-abaques-dimensionnels.md) |
| Diagramme de renforcement | nom donné aux abaques de renforcement d'ouvrant par le classeur e.VOLUTION de 2008 | [Abaques de renforcement du système 70, plans e.VOLUTION de 2008](/profiles/systeme-70-abaques-evo2008.md) |
| Lames V.R. | lames du tablier d'un volet roulant (V.R.) ; la coulisse se choisit par leur épaisseur nominale (8, 12 ou 14 mm) | [Montage des profilés complémentaires du système 70, 2008](/procedures/montage-profiles-complementaires-systeme-70-evo2008.md) |
| Clip de coulisse (9447), gabarit de perçage (9905) | pièce enfoncée dans un trou Ø 7,5 du dormant ou de l'ouvrant sur laquelle se clippe la coulisse ou le rejet d'eau ; gabarit qui positionne ces trous | [Montage des profilés complémentaires du système 70, 2008](/procedures/montage-profiles-complementaires-systeme-70-evo2008.md) |
| Angle variable, adaptateur | profilés K340 et K341 qui réunissent deux dormants sous un angle de 90° à 180° | [Montage des profilés complémentaires du système 70, 2008](/procedures/montage-profiles-complementaires-systeme-70-evo2008.md) |
| Intercalaire de frappe | cale en bois interposée entre le marteau et le profilé pour répartir le choc lors de la mise en place d'un élément de jonction | [Montage des profilés complémentaires du système 70, 2008](/procedures/montage-profiles-complementaires-systeme-70-evo2008.md) |
| Petit bois rapporté collé | petit bois collé au double face sur la face du vitrage, d'un côté ou des deux | [Montage des profilés complémentaires du système 70, 2008](/procedures/montage-profiles-complementaires-systeme-70-evo2008.md) |

# Sigles non élucidés

| Sigle | Où | Entrée |
| --- | --- | --- |
| DV | Catalogue général, p. 17, qualifie un Uw de fenêtre | **VER-35** |
| LVFF | Catalogue Roto NX PVC, p. 282 et 298, encadrés INFO de la têtière de compas du vantail pivotant et du compas confort ; glosé « largeur de feuillure d'ouvrant » au seul repère [D] de la p. 403 | **INC-293** |
| DK | Catalogue Roto NX PVC, p. 403, note [7] du limiteur d'ouverture par pivotement 191 (« renvoi d'angle DK ») | **INC-302** |
| SKG | Catalogue Roto NX PVC, p. 426, « Clip d'information SKG** » 331459 | **INC-303** |
| FOF | Catalogue Roto NX PVC, p. 439, 442, 444, en-tête des tableaux d'affectation des gabarits d'insertion | **INC-306** |
| CV | Catalogue général, p. 17, qualifie un Uw de coulissant | **VER-35** |
| CVR, CRV, DL | Mise en œuvre Système 70 Plateforme, PDF p. 8-45 : « Profilé de liaison dormant CVR », « Embout haut G/D sous CRV », « Patin d'étanchéité pour DL/seuil » | **VER-73** |

Aucun document du corpus ne définit ces sigles. Voir
[Informations à vérifier](/anomalies/informations-a-verifier.md).

# Citations

[1] [Directives générales profine, version janvier 2023](raw/profine-directives-generales-2023-01.pdf),
registres 1.1.1 et 1.1.2, PDF p. 2 à 16

[2] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 —
`raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 10 à 13

[3] DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED —
`raw/dta-trocal-76-advanced-6-16-2334-v5.pdf`, p. 4 à 9

[4] [Mise en œuvre Système 70 Plateforme, profine, version septembre 2023](raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf),
PDF p. 1 à 371

# Voir aussi

- [Directives générales profine](/sources/profine-directives-generales.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Renforts du système 76](/profiles/systeme-76-renforts.md)
- [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)
- [Labels et certifications](/certifications/labels-et-certifications.md)
- [Performances des vitrages](/vitrages/performances-vitrages.md)
