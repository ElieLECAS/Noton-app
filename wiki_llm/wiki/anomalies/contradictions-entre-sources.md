---
type: Anomalie
title: Contradictions entre sources
description: Registre des points sur lesquels deux documents PROFERM affirment des choses différentes, avec la valeur à retenir en attendant l'arbitrage.
tags: [anomalie, contradiction, garantie, a-corriger]
status: stable
sources:
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
    last_modified: 2025-06-19
  - resource: raw/poster-systeme-76-advanced-complementaires-2022.pdf
    id: poster-76-advanced-complementaires
    title: Poster Système 76 Advanced, profilés complémentaires, 2022
    last_modified: 2022-12-31
  - resource: raw/poster-systeme-76-advanced-principaux-2022.pdf
    id: poster-76-advanced-principaux
    title: Poster Système 76 Advanced, profilés principaux, 2022
    last_modified: 2022-12-31
  - resource: raw/catalogue-general-2026-01.pdf
    id: catalogue-general-2026
    title: Catalogue menuiseries PROFERM, édition janvier 2026
    last_modified: 2026-01-31
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
    last_modified: 2026-09-02
  - resource: raw/brochure-perform-plus-hybride-plus-2023-05.pdf
    id: brochure-perform-plus-hybride-plus-2023-05
    title: Brochure Nouveautés PERFORM+ et HYBRIDE+, édition mai 2023
    last_modified: 2023-05-31
  - resource: raw/brochure-hybride-2025-03.pdf
    id: brochure-hybride-2025-03
    title: Brochure HYBRIDE, édition mars 2025
    last_modified: 2025-03-31
  - resource: raw/depliant-innoslide-2024-01.pdf
    id: depliant-innoslide-2024-01
    title: Dépliant INNOSLIDE, édition janvier 2024
    last_modified: 2024-01-31
  - resource: raw/depliant-lumeal-2026-04.pdf
    id: depliant-lumeal-2026-04
    title: Dépliant LUMÉAL, édition avril 2026
    last_modified: 2026-04-21
  - resource: raw/brochure-lumine65-2025-02.pdf
    id: brochure-lumine65-2025-02
    title: Brochure LUMINE65, édition février 2025
    last_modified: 2025-02-28
  - resource: raw/catalogue-portes-entree-2024-03.pdf
    id: catalogue-portes-entree-2024-03
    title: Catalogue portes d'entrée PROFERM, édition mars 2024
    last_modified: 2024-03-31
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
    last_modified: 2025-06-19
  - resource: raw/depliant-general-2023-06.pdf
    id: depliant-general-2023-06
    title: Dépliant général PROFERM, édition juin 2023
    last_modified: 2023-06-30
  - resource: raw/depliant-hybride-2023-06.pdf
    id: depliant-hybride-2023-06
    title: Dépliant HYBRIDE, édition juin 2023
    last_modified: 2023-06-30
  - resource: raw/depliant-lumeal-2023-06.pdf
    id: depliant-lumeal-2023-06
    title: Dépliant LUMÉAL, édition juin 2023
    last_modified: 2023-06-30
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
  - resource: raw/nuancier-vitrages-decoratifs.pdf
    id: nuancier-vitrages-decoratifs
    title: Nuancier des vitrages décoratifs PROFERM
generated:
  by: process:claude-code
  at: 2026-09-17T22:30:00Z
---

# Ce que contient ce registre

Les cas où **deux documents PROFERM disent des choses différentes** sur le même sujet. Les
contradictions internes à un seul document sont ailleurs :
[Incohérences internes](/anomalies/incoherences-internes.md).

Chaque entrée porte un identifiant stable en `CTR-`, et propose une valeur à retenir en
attendant l'arbitrage. **Cette valeur est un pis-aller, pas une décision** : seul le service
technique tranche.

# Registre

| ID | Sujet | Source A | Source B | Valeur à retenir en attendant | Impact |
| --- | --- | --- | --- | --- | --- |
| CTR-01 | Charge du pivot bas | Cahier technique PERFORM76, p. 3 (PDF 6) : 100 kg par ouvrant sur pivot bas | Catalogue général, p. 6 : jusqu'à 130 kg | **100 kg** — c'est le document technique | Menuiserie surdimensionnée acceptée en commande, risque de rupture |
| CTR-02 | Épaisseur de vitrage de sécurité STADIP 44² | Catalogue général, p. 27 : STADIP 44²/16/4 sur PERFORM | Lecture multimodale erronée initiale : la notation vitrière 44² désigne le feuilleté 44.2 (8,76 mm) et non 44 mm. L'épaisseur totale réelle est de 28,76 mm (8,76 + 16 + 4 mm). | **Levée (erreur d'interprétation initiale)** : l'épaisseur réelle de 28,76 mm s'insère directement dans les parcloses standard de 28 mm (parclose 76526) et respecte le domaine d'emploi du DTA plafonné à 50 mm. | Aucune anomalie réelle sur le vitrage ni sur les parcloses. |
| CTR-03 | Garantie structure | Catalogue général, p. 35 : 15 ans sur la structure de la fenêtre ; **brochure HYBRIDE, mars 2025, p. 3** : macaron « Garantie 15 ans sur la structure de la fenêtre » | Brochure PERFORM+/HYBRIDE+, p. 3 **et brochure LUMINE65, p. 5** : 20 ans sur la structure | **aucune** — la valeur ne suit ni la date ni le produit | Engagement contractuel erroné de 5 ans |
| CTR-04 | Garantie volet roulant | Catalogue général, p. 35 **et brochure LUMINE65** : 5 ans | **Six documents** : brochure PERFORM+/HYBRIDE+, les trois dépliants de juin 2023, dépliant INNOSLIDE, dépliant LUMÉAL 2026 : 7 ans | **aucune** — six sources contre deux, mais rien ne les sépare | Engagement contractuel erroné de 2 ans |
| CTR-05 | Garantie laquage | Catalogue général, p. 35 : 25 ans LUMINE65, 10 ans standards, 7 ans hors standards | Brochure PERFORM+/HYBRIDE+, p. 3 : 7 ans forfaitaires ; **dépliant HYBRIDE, juin 2023, p. 3** : « Laquage : 7 ans », sans distinction de couleur ; **dépliant LUMÉAL, juin 2023, p. 3** : « Laquage : 7 ans », de même, pour une HYBRIDE dont les neuf couleurs laquées sont dites « à prix préférentiel » | **La grille du catalogue** hors gammes + | Garantie sous-annoncée ou sur-annoncée selon la gamme |
| CTR-06 | Épaisseur du profilé PVC de l'HYBRIDE | Brochure HYBRIDE, mars 2025, p. 2 : « Robustesse de la fenêtre — Profilé PVC épaisseur 72mm », gamme unique (relu en image le 25/09/2026) ; **dépliant HYBRIDE, juin 2023, p. 2** : même phrase, même valeur | Catalogue général, p. 10 : 70 à 76 mm, deux déclinaisons | **70 à 76 mm**, source la plus récente | Réponse fausse sur l'épaisseur d'une HYBRIDE livrée avant 2026 |
| CTR-07 | Uw de l'HYBRIDE | Brochure HYBRIDE, mars 2025, p. 2 : « Performance thermique optimale. 5 chambres d'isolation. Uw jusqu'à 1.2 W/m²K » (relu en image le 25/09/2026) ; dépliant HYBRIDE, juin 2023, p. 2 : même phrase, même valeur | Catalogue général, p. 10 et 11 : jusqu'à 0,8 W/m²K | **0,8 W/m²K**, source la plus récente | Étude thermique fausse dans les deux sens, écart de 0,4 W/m²K |
| CTR-08 | Classement AEV du LUMÉAL | Dépliant LUMÉAL, avril 2026, p. 2 (p. 1 avant le 25/09/2026, erreur de locator) : A\*4 / E\*7A / V\*B3, sur 2 vantaux H 2,5 × L 3 m ; même classement et même dimension d'essai au dépliant de juin 2023, p. 2 | Catalogue général, p. 17 : A\*4 / E\*6A / V\*B2 pour tous les coulissants | **A\*4 / E\*7A / V\*B3**, source la plus récente et seule à donner la dimension d'essai | LUMÉAL disqualifié à tort sur chantier exposé |
| CTR-09 | Garantie de la ferrure Technal | Dépliants LUMÉAL de **juin 2023 et avril 2026** : ferrure Technal 10 ans, autre ferrure 2 ans | Catalogue général, p. 35 : ferrure ROTO 10 ans, « autre ferrure » 2 ans | **10 ans pour Technal**, valeur constante sur trois ans | Garantie sous-annoncée de 8 ans sur toute la gamme aluminium, depuis au moins 2023 |
| CTR-10 | Coloris du LUMÉAL | Dépliant LUMÉAL, p. 2 : 3 standards + 7 à prix préférentiel, pas de bicoloration, pas de Chêne doré. **Complément du 25/09/2026** : le dépliant de juin 2023, p. 3, relu en image, porte les mêmes 3 « couleurs extérieures standards » et 7 « couleurs extérieures à prix préférentiel », sans chêne doré ni anodisé, et ne dit rien d'une bicoloration ; le dépliant d'avril 2026, p. 2, relu en image, porte les mêmes dix teintes, ajoute « *Pas de bicoloration possible » en renvoi des deux groupes et ne donne plus de faces pour les sept couleurs à prix préférentiel (« 1 face » en 2023) | Catalogue général, p. 18 : Blanc 9016 brillant, Chêne doré 2 faces, laquage toutes teintes RAL | **La liste du dépliant**, source la plus récente | Coloris vendu puis indisponible, ou inversement |
| CTR-11 | Classement AEV du coulissant LUMINE65 | Brochure LUMINE65, p. 3 : A\*4 / E\*6A / V\*A3, sur 2 vantaux | Catalogue général, p. 17 : A\*4 / E\*6A / V\*B2 pour tous les coulissants | **A\*4 / E\*6A / V\*A3**, seule source qui essaie ce produit précis | Classement produit faux en appel d'offres |
| CTR-12 | Nom de la poignée encastrée de coulissant | Brochure LUMINE65, p. 3 : « Poignée encastrée **MLINI** — En option. Disponible en inox, blanc ou noir », sur le coulissant LUMINE65 ; la photo montre une poignée argent à levier incliné | Catalogue général, p. 17 : « **DEHLI** — En option. Poignée encastrée. Disponible en inox, blanc ou noir », pour tous les coulissants ; la photo montre une cuvette blanche aux bords arrondis. Relu en image le 25/09/2026 : les deux photos ne représentent pas le même objet | **Aucune** — même fonction, même finitions, même position dans la liste, mais deux noms et deux dessins : il peut s'agir de deux poignées distinctes. Les deux sont portées sur [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md) | Référence introuvable à la commande, ou poignée livrée qui n'est pas celle montrée au client |
| CTR-13 | Uw de l'HYBRIDE en juin 2023 | Dépliant HYBRIDE, p. 2 : 1,2 W/m²K | Dépliant général **du même mois**, p. 4 : 1,3 W/m²K | **1,2 W/m²K**, valeur du document produit | Étude thermique fausse sur de l'existant posé avant 2026 |
| CTR-14 | Liste des vitrages décoratifs | Nuancier vitrages, **sans date** : 7 verres dont un **Imprimé 200** inconnu ailleurs | Catalogue général, p. 26 : 9 verres dont **Listral** et **Mimosa**, absents du nuancier | **Largement levée** : le catalogue portes, p. 150, révèle que l'imprimé 200 est le verre de face intérieure des panneaux classiques, pas un décoratif au choix | Reste le Listral et le Mimosa vendables sans échantillon montrable **Complément du 25/09/2026** (nuancier relu en image) : le nuancier présente l'Imprimé 200 (p. 4) au même rang que les six autres vitrages, sans restriction d'emploi ; le rattachement aux seules portes classiques vient du catalogue portes, p. 150, et reste à confirmer. La photo « Listral » du catalogue général et la photo « Imprimé 200 » du nuancier montrent un grain voisin (**VER-54**) ; le Chinchilla n'a pas la même photo dans les deux documents (**CTR-30**) |
| CTR-15 | Collections de portes d'entrée | Catalogue portes, mars 2024 : **six** collections — Authentique, Contemporain, Graphite, Lumière, Classique, Éléments. Relu en image le 25/09/2026 pour les p. 1 à 40 : la collection Authentique y a les mêmes sept modèles (ADONIS, ANÉMONE, CHICORÉE, CHICORÉE 2, CHICORÉE 4, DOMINO, IRIS) et les mêmes trois gammes (PERFORM, HYBRIDE, TEXTURAL) qu'au catalogue général ; les modèles BEN et JACOB, déclinés 0, 1 et 2, sont rangés dans la collection Contemporain (p. 31 et 37), là où le catalogue général présente les monoblocs X-BEN-0 et X-JACOB-0 dans la Sélection Hexa. Relu en image le 25/09/2026 pour les p. 81 à 120 : la collection Contemporain s'étend jusqu'à la p. 97 ; la collection Graphite (p. 98 à 104) compte dix modèles en deux versions, monoblocs de la seule gamme TEXTURAL ; la collection Lumière s'ouvre p. 105. Relu en image le 25/09/2026 pour les p. 121 à 160 : la collection Lumière se termine p. 122 ; la collection Classique (p. 123 à 134) compte vingt-sept modèles, dont un GRANALA 122, homonyme du monobloc X-GRANALA de la Sélection Hexa du catalogue général (non rapprochés) ; la collection Éléments (p. 135 à 140) compte onze modèles déclinés « 1 » | Catalogue général, janvier 2026 : **deux** — Collection Authentique et Sélection Hexa, cette dernière absente du catalogue portes | **Aucune** — deux découpages incompatibles de la même offre | Modèle proposé au client puis introuvable à la commande |
| CTR-16 | Garantie du panneau de porte | Catalogue portes, mars 2024, PDF p. 158 (page imprimée 156, relue en image le 25/09/2026) : « Panneau de porte » 7 ans, « Panneau de porte modèle plaxé » 5 ans ; dépliant général, juin 2023, p. 7 (relu en image le 25/09/2026) : « Panneau de porte : 7 ans et 5 ans sur les modèles plaxés » | Catalogue général, janvier 2026, p. 35 : 10 ans, plaxé 7 ans | **10 ans**, source la plus récente : l'amélioration est postérieure à mars 2024 | Garantie sous-annoncée de 3 ans sur un poste coûteux |
| CTR-17 | Épaisseur de vitrage maximale du système 76 | Mise en œuvre Système 76 Advanced, profine, registre 2.1.1, p. 1 (PDF p. 15, version janvier 2016) : « mise en oeuvre de différentes épaisseurs de vitrage ou panneau de remplissage **de 16 à 48 mm** » ; même manuel, registre 2.1.1 Porte d'entrée, p. 1 (PDF p. 2, octobre 2021) : « différentes épaisseurs de vitrage (jusqu'à 48 mm) » | DTA n° 6/16-2334_V5, p. 9 : vitrage jusqu'à **50 mm** ; même DTA, p. 38, « Prise de volume » : 50 mm sur ouvrant et sur dormant, 44 mm avec rehausseur de parclose ; cahier technique PERFORM76, p. 5 : parcloses d'ouvrant jusqu'à **50 mm** | **50 mm**, valeur réglementaire du DTA, qui est la pièce opposable. Le manuel de fabrication donne par ailleurs 36 à 50 mm pour la variante AluClip Zero (registre 2.6.5) : la borne de 48 mm n'est donc pas une limite du système entier | Un vitrage de 50 mm refusé à tort en atelier, ou accepté sans vérifier la parclose |
| CTR-18 | Champs d'application de la ferrure Roto NX, côté paumelles P, oscillo-battant rectangulaire | Instructions de montage Roto NX KSR, novembre 2022, p. 21 à 27 : HFF mini 290 mm, CDR 1 N jusqu'à 1 400 mm de LFF et 2 600 mm de HFF, CDR 2 jusqu'à 2 400 mm de HFF | Catalogue Roto NX, juin 2023, p. 35 : HFF mini 280 mm, CDR 1 N jusqu'à 1 600 mm de LFF et 2 800 mm de HFF, CDR 2 jusqu'à 2 800 mm de HFF | **Les bornes les plus basses des deux documents**, en attendant l'arbitrage : les deux sont des documents ROTO du même produit, à sept mois d'écart, et aucune règle du wiki ne les départage | Vantail accepté en commande hors du champ d'application réel de la ferrure, sur quatre bornes dont trois en CDR |
| CTR-19 | Épaisseur de vitrage de trois parcloses PERFORM76 | Cahier technique PERFORM76, p. 5 (PDF 8) : 76508 à 48 mm, 2454 à 31 mm, 2433 à 33 mm | DTD n° DBV-25-6/16-2334_V5, p. 18 : 76508 à 46 mm, 2454 à 32 mm, 2433 à 34 mm. Poster Système 76 Advanced, profilés complémentaires, 2022, p. 1 : 76508 (46), 2454 (31), 2433 (34) — il suit le DTD pour 76508 et 2433, le cahier pour 2454. DTA n° 6/16-2334_V5, p. 19 (planche « Parcloses PVC ») : 76508 (46), 2454 (32), 2433 (34), comme le DTD, et **2638 (32)** là où le cahier (p. 5, PDF 8) et le poster donnent 31. Mise en œuvre Système 76 Advanced, registre 2.3.2 (PDF p. 89, 94 et 96, relus en image le 25/09/2026), épaisseur A avec joint de 4 mm : 76508 (46), 2454 (32), 2433 (34), 2638 (32), comme le DTD et le DTA | **Les valeurs du cahier technique** — c'est le document produit, celui qui fixe la fabrication PERFORM76 | Parclose commandée à l'épaisseur du DTD, en écart de 1 à 2 mm avec le vitrage réellement posé |
| CTR-20 | Largeur totale de cinq dormants larges du système 70 | Mise en œuvre Système 70 Plateforme, registre 2.1.2 : 6108 à 105 mm, 6109 à 125, 6110 à 145, 6111 à 165, 6158 à 210 | DTD n° DBV-24-6/16-2335_V5, p. 15 : 6108 à 95 mm, 6109 à 115, 6110 à 135, 6111 à 155, 6158 à 200 | **Les valeurs du classeur de fabrication**, qui sert déjà de référence aux cotes de débit ; l'écart constant de 10 mm sur les cinq références suggère une convention de mesure différente, non énoncée par l'un ou l'autre document | Élargisseur ou pièce d'appui commandé 10 mm trop court ou trop long sur les cinq dormants larges |
| CTR-21 | Renforts admis dans plusieurs profilés du système 76 | Poster Système 76 Advanced, profilés principaux, 2022, p. 1 (renforts écrits dans la chambre de chaque coupe) : **76206** V260, V333R/L, V335R/L ; **76102** V314.Z, V325, V353 ; **76101** V306.Z, V307.Z, V308, V309.Z, V310 ; **76303** V323.Z, V322 ; **76172** V314.Z, V325, V353 ; **76283** V314.Z, V326.Z, V337 R/L, V339 R/L, V353 | Mise en œuvre Système 76 Advanced, registres 2.1.2 et 2.1.3 : **76206** V323.Z, V322 (planche relue en image le 25/09/2026, PDF p. 38 : ces deux renforts sont dans le tableau d'accessoires du **76299**, titré « Compensateur », et le 76206 n'y est nommé que dans le cadre « utilisé avec » ; l'attribution au 76206 venait d'une lecture antérieure — le manuel ne donne au 76206 que V260, V333R/L et V335R/L, PDF p. 9, comme le poster) ; **76303** V323.Z, V322 **et V324** ; **76172** V314.Z (soudé), V325, V353 (soudé) ; **76283** V314.Z, V326.Z, V337 R/L, V339 R/L, V353 (planches relues en image le 25/09/2026, PDF p. 20 et 30 : l'ancienne lecture « V325 seul », « V326.Z seul » venait de la couche texte ; sur ces deux profilés le manuel concorde avec le poster) ; aucun **V329** dans le manuel, qui donne au **76101** de porte d'entrée V306.Z, V307.Z, V308, V309.Z, V310 (PDF p. 5). DTD n° DBV-25-6/16-2334_V5 : **76102** V314 ou **V326** ; **76101** V306, V307, V309 ou **V329**. DTA n° 6/16-2334_V5, p. 21 (« Pour profilé(s) » sous chaque renfort) : **76172** V314 ou V326 ; **76102** V314 ou V326 ; **76101** V306, V307, V309 ou V329 ; **V329** aussi pour 76171, 76173, 76180 ; **V306 et V307** aussi pour l'ouvrant **76271** ; **76303** V323 seul ; V308, V310, V319, V322, V324, V325, V353 non dessinés. Même planche au DTD, p. 20 ; la coupe « Assemblage traverse complémentaire » du DTD (p. 26) dessine le **V323** dans la traverse complémentaire **76299**, que la liste « Pour profilé(s) » du V323 ne nomme pas | **aucune** — les deux listes sont reportées sur [Renforts du système 76](/profiles/systeme-76-renforts.md) avec leur source. Le cas du 76206 n'est pas un écart de liste mais deux familles de renfort sans référence commune | Renfort commandé qui n'entre pas dans la chambre, ou calcul statique fait avec l'inertie d'un renfort non admis par l'autre document |
| CTR-22 | Renforts des élargisseurs 76700 à 76703 | Poster Système 76 Advanced, profilés complémentaires, 2022, p. 1 (renfort écrit dans la coupe), et cahier technique PERFORM76 : 76700 sans renfort, 76701 **V312.Z**, 76702 **V314.Z**, 76703 **V314.Z** dans deux chambres | Mise en œuvre Système 76 Advanced, registre 2.1.3 : 76700, 76701 et 76702 **V312.Z**, 76703 **V114** (lecture d'une ingestion antérieure). Relues en image le 25/09/2026 (PDF p. 40 et 41), les planches donnent : 76700 sans tableau d'accessoires, 76701 **V312.Z**, 76702 **V314.Z** « Renfort 2,0 mm, soudé », 76703 **V314.Z** « Renfort 2,0 mm, soudé » ; le V114 est celui des réhausses 76704, 76706 et 76777 (PDF p. 41 et 43). Le manuel concorde donc avec le poster et le cahier ; la fermeture de l'entrée reste à l'arbitrage Registre 2.5.1, p. 3 (PDF p. 253, version février 2017, relu en image le 25/09/2026) : 76701 **V312** « Renfort1,5 mm », 76702 et 76703 **V314** « Renfort2,0 mm », sans « .Z » ni « soudé », IG / IW 0,3 / 1,5 et 8,4 / 5,7 identiques | **Le poster et le cahier PERFORM76**, qui concordent et dont le cahier est le document produit | Élargisseur de 60 ou 120 mm renforcé avec un acier de 1,5 cm⁴ en vent au lieu de 5,7, ou renfort commandé qui n'entre pas dans la chambre |
| CTR-23 | Écarts de cote sur six profilés et renforts du système 76 | Poster Système 76 Advanced, profilés complémentaires, 2022, p. 1 : appui **6137** 156,5 mm hors tout ; liaison **A250** 75 mm ; **76604** cote de 5 mm ; poteau **8356** retour de 53 mm ; renfort **V263** 14,5 cm⁴ ; renfort **V317** dessiné 24 × 40 mm. DTA n° 6/16-2334_V5, p. 21 : **V291** coté **42 × 30 mm** (43 × 29 sur le poster des profilés principaux), **V317** 25 × 40 mm. Mise en œuvre Système 76 Advanced, registre 2.1.2 (PDF p. 33 et 34) : **V318.Z** coté 45 × **28,6** mm et **V320.Z** 37 × **28,6** mm, contre 45 × 29 et 37 × 29 sur le poster des profilés principaux (renforts relevés sur [Renforts du système 76](/profiles/systeme-76-renforts.md)). Mise en œuvre Système 76 Advanced, registre 2.1.3 (PDF p. 41 et 42, relus en image le 25/09/2026) : réhausse **76708** cotée **58,5** mm à gauche (58 sur le poster des profilés complémentaires) ; réhausse **76709** cotée **108,3** mm à gauche et **99,8** mm à droite, 88 et 76 mm en haut (une seule cote verticale de 100 mm sur le poster) ; réhausses 76704, 76706 et 76705 cotées 53, 53 et 158 mm à gauche, cote absente du poster (relevés sur [Profilés complémentaires du système 76](/profiles/systeme-76-profiles-complementaires.md#élargisseurs-et-réhausses)) ; même registre, p. 7 (PDF p. 45) : pièce d'appui **76768** cotée **42 mm** au nez (31 mm sur le poster), 196 et 136 mm comme au poster ; **6137** coté 156,5 mm comme au poster, sous le titre « Pièce d'appui de 157 mm » (entrée **INC-45**) ; p. 10 (PDF p. 48) : réhausse **76765** cotée **35** mm (35,2 sur le poster) ; p. 14 (PDF p. 52) : **A250** coté **75,2 × 38,6** mm (75 × 38,5 sur le poster) ; p. 15 (PDF p. 53) : poteau **8356** retour de **53** mm comme au poster, renfort **V263** **14,4** cm⁴ | Cahier technique PERFORM76 (6137 : 157 mm) ; pages PERFORM76 tirées du manuel de mise en œuvre et du DTD (A250 : 75,2 mm ; 76604 : 5,5 mm ; 8356 : 54 mm ; V263 : 14,4 cm⁴) ; poster des profilés principaux, 2022, p. 1 (V317 : 40 × 25 mm) | **aucune** — écarts de 0,1 à 1 mm, arrondi ou mesure prise à un autre point ; les deux valeurs sont portées sur [Profilés complémentaires du système 76](/profiles/systeme-76-profiles-complementaires.md) | Débit ou usinage faux de 0,5 à 1 mm si la cote est reprise de la mauvaise source |
| CTR-24 | Seconde cote horizontale des appuis PVC 6136 et 6137 | Poster Système 76 Advanced, profilés complémentaires, 2022, p. 1 : 6136 **70 mm**, 6137 **100 mm** (seconde cote horizontale, sous la longueur hors tout de 127 et 156,5 mm) | DTA n° 6/16-2334_V5, p. 17, planche « Pièces d'appui PVC » : 6136 **67 mm**, 6137 **97 mm**, seule cote portée ; cahier technique PERFORM76, p. 16 (PDF) : largeur 67 et 97 mm. Les 76758 (80 mm) et 76768 (136 mm) concordent dans les trois | **aucune** — l'écart de 3 mm, identique sur les deux appuis, peut venir d'un point de mesure différent ; les deux valeurs sont portées sur [Profilés complémentaires du système 76](/profiles/systeme-76-profiles-complementaires.md) | Saillie d'appui fausse de 3 mm si la cote du poster est prise pour celle du DTA et du cahier |
| CTR-25 | Matière des joints G049, G049.T, G050, G050.T, G051 et G051.T du système 76 | DTA n° 6/16-2334_V5, p. 19, planche « Garnitures de joint » : les six sont marqués **(TPE)** | Poster Système 76 Advanced, profilés principaux, 2022, p. 1 : G049.T, G050.T, G051.T **PVC**, G051 **EPDM**. DTD n° DBV-25-6/16-2334_V5, p. 9 : matières par coloris E400, E401, M400, G551. Mise en œuvre Système 76 Advanced, registre 2.1.1 (PDF p. 3 et 16) : G049.T, G050.T et G051.T rangés sous l'en-tête « Joint **TPE** » et légendés « (**PVC**) » sous leur dessin, G051 « (EPDM) » — le manuel porte les deux matières sur la même planche | **aucune** — les trois matières sont portées sur [Joints et garnitures des systèmes profine](/profiles/joints-et-garnitures-profine.md) avec leur source | Joint de remplacement commandé dans une matière qui n'est pas celle d'origine, en SAV ; compatibilité avec le film ou le mastic non vérifiée pour la bonne matière |
| CTR-26 | Cotes des battements 76401, 76471 et 76472 | Mise en œuvre Système 76 Advanced : planche « 76401 Battement 48 mm » de la porte d'entrée (registre 2.1.1, PDF p. 11), largeur **94,1** ; planche « 76471 Battement 62 mm » de la fenêtre (registre 2.1.2, PDF p. 32), largeur **94,1**, hauteur 62 décomposée **22 + 21 + 19**, sans cote basse ; planche « 76472 Battement 80 mm » (même page), 75 × 80 décomposé **22 + 41 + 17**. Relu à 400 et 500 dpi | Poster Système 76 Advanced, profilés principaux, 2022, p. 1, relevé porté sur [Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md) : 76401 **94,5** (48 = 22 + 16 + 10, comme au manuel) ; 76471 **94,5**, 62 = **29 + 16 + 17**, cote basse 17,5 ; 76472 80 = **29 + 34 + 17**, cote basse 17,5 | **aucune** — les deux jeux de cotes sont portés avec leur source ; les décompositions peuvent être prises à des points différents, ce qu'aucun document ne dit | 0,4 mm d'écart sur la largeur hors tout d'un battement, et une décomposition de hauteur reprise sur le mauvais document pour un usinage |
| CTR-27 | Finition du rouge 3004 de la LUMINE65 | Brochure LUMINE65, février 2025, p. 5 : « Rouge 3004 satiné » dans les 11 coloris 1 ou 2 faces | Catalogue général, janvier 2026, p. 18, nuancier LUMINE65 : « ROUGE 3004 granité ». Même catalogue : « Rouge 3004 satiné » pour l'HYBRIDE (p. 12), « ROUGE 3004 granité » pour la LUMINE55 (p. 18) | **satiné**, valeur du document produit (règle « un document produit bat un document général ») — à confirmer par l'atelier de laquage. Brochure relue en image à 200 dpi le 25/09/2026 : « Rouge 3004 satiné » confirmé, pastille satinée reprise sur [Coloris LUMINE](/coloris/coloris-lumine.md) | Commande de rouge 3004 dans une finition que la LUMINE65 ne propose pas, teinte refusée ou refaite à la livraison |
| CTR-28 | Finitions de cinq poignées et béquilles sur la LUMINE65 | Brochure LUMINE65, février 2025 : coulissant (p. 3) Sécustik® ATLANTA « blanc ou aspect inox », cuvette encastrée SEOUL « En option. Disponible en inox, blanc ou noir » ; porte-fenêtre (p. 4) poignée Sécustik® ATLANTA « blanc ou aspect inox », béquille ATLANTA plaque étroite et Sécustik® ATLANTA avec rosace « blanc ou aspect inox. Noir et 7016 en option ». La même brochure donne à la Sécustik® ATLANTA de fenêtre « blanc, aspect inox, caramel ou laiton » (p. 4) | Catalogue général, janvier 2026 : Sécustik® ATLANTA de coulissant « blanc, aspect inox, caramel ou laiton », SEOUL « noir, 7016, aspect inox ou blanc » (p. 17) ; ATLANTA plaque étroite et Sécustik® ATLANTA avec rosace « inox, blanc ou laiton », noir et 7016 en option (p. 15) | **aucune** — aucune règle ne départage une liste de finitions ; les deux listes sont portées ligne par ligne sur [Poignées et croisillons](/quincaillerie/poignees-et-croisillons.md), avec leur source | Poignée caramel ou laiton, ou SEOUL 7016, promise sur une LUMINE65 où elle n'existe peut-être pas ; ou finition inox refusée à tort |
| CTR-29 | Nombre de rails du LUMÉAL en 4 vantaux | Dépliant LUMÉAL, juin 2023, p. 2, « Les ouvertures » : « 4 vantaux - 2 ou 3 rails », schéma à 3 rails dessiné ; même texte et même schéma au dépliant d'avril 2026, p. 2 | Catalogue général, janvier 2026, p. 16 : « 4 vantaux - 2 rails » seulement ; la fiche du catalogue de conception TECHNAL LUMEAL GA réf. 5156.007 ([sa fiche](/sources/technal-lumeal-ga-conception.md)) liste « 4 vantaux 2 rails » et « 3 vantaux 3 rails » parmi ses typologies | **2 ou 3 rails**, document produit, sous réserve de la validation de faisabilité par un technicien PROFERM que le catalogue demande avant commande : le document du fabricant du système ne liste pas le 4 vantaux 3 rails | Un 4 vantaux 3 rails vendu sur la foi du dépliant et non fabricable, ou refusé à tort |
| CTR-30 | Photographie du vitrage Chinchilla | Nuancier des vitrages décoratifs, **sans date**, p. 2 : sous le titre « Chinchilla clair », un verre à fines stries rayonnantes, rendu relu le 25/09/2026 à 200 dpi | Catalogue général, janvier 2026, p. 26 : sous le titre « Chinchilla », un verre à grain serré, sans stries ; les cinq autres vitrages communs aux deux documents (Dépoli, Delta clair, Delta mat, Olivier, Clé de fleur) y sont la même prise de vue que dans le nuancier, pas celui-ci | Aucune — les deux photos sont montrées sur [Vitrages décoratifs](/vitrages/vitrages-decoratifs.md), chacune sous son nom ; demander au service commercial laquelle montre le Chinchilla vendu, et si « Chinchilla » et « Chinchilla clair » désignent le même verre | Un client qui choisit sur la photo du catalogue peut recevoir un verre d'un autre aspect |
| CTR-31 | Serrure de la porte d'entrée aluminium | Catalogue général, janvier 2026, p. 32 : « serrure à goujons pour la porte d'entrée aluminium », schéma légendé « À goujons. Sur ALU. » | Catalogue portes d'entrée, mars 2024 : « serrure à crochets pour la porte d'entrée PVC 118 et la porte d'entrée aluminium », schéma légendé « À pênes Sur ALU » (PDF p. 153) ; « Serrure 3 pênes pénétrants » (ouvrant SOLEAL, PDF p. 144) ; « Serrure 6 points automatique à crochet et gâche filante » de série sur SOLEAL (matrice, PDF p. 143) — voir **INC-08** | **aucune** — les documents ne permettent pas de trancher ; le même dessin de serrure est légendé « à goujons » en 2026 et « à pênes » en 2024 | Serrure de remplacement commandée sur la mauvaise désignation ; description erronée au client |
| CTR-32 | Référence de la réhausse de 35 mm cotée 35 / 21 + 46 / 26,5 et 43 | Poster Système 76 Advanced, profilés complémentaires, 2022, p. 1 : la coupe porte, écrite dans le profilé, la référence **76776** | Mise en œuvre Système 76 Advanced, registre 2.1.3, p. 8 (PDF p. 46, relu en image à 400 dpi le 25/09/2026) : même coupe, mêmes cotes, titrée « **76753** Réhausse de 35 mm » ; le cahier technique PERFORM76 range un appui **76753** de 35 mm, 21 / 46 sur 76, parmi les appuis des dormants 76171 et 76172 | **aucune** — les deux références sont portées sur [Profilés complémentaires du système 76](/profiles/systeme-76-profiles-complementaires.md) avec leur source ; le manuel et le cahier concordent, le poster est seul | Réhausse commandée sous une référence qui n'est pas celle du profilé voulu, ou refusée faute de la trouver au tarif |
| CTR-33 | Embouts de remplissage de la pièce d'appui 76768 | Poster Système 76 Advanced, profilés complémentaires, 2022, p. 1 : trois « Embout de remplissage » **M780, M781 et M782** dessinés dans les chambres du 76768 ; DTD n° DBV-25-6/16-2334_V5, p. 23 : M780 à M782 cotés | Mise en œuvre Système 76 Advanced, registre 2.1.3, p. 7 (PDF p. 45, relu en image le 25/09/2026) : une seule case « **M612** — Embout de remplissage — Pour 76768 », dessinée en trois morceaux | **aucune** — les deux nomenclatures sont portées sur [Nomenclature des profilés complémentaires du système 76](/profiles/systeme-76-accessoires-profiles-complementaires.md#pièces-dappui) ; la M612 peut être le lot des trois embouts, ce qu'aucun document n'écrit | Embouts commandés en double (M612 et M780 à M782), ou pièce d'appui livrée sans embouts |
| CTR-34 | Pièces d'assemblage des dormants 76171, 76172, 76173 et du meneau 76372 sur seuil A075 grugé | Poster Système 76 Advanced, profilés principaux, 2022, p. 1, tableau « Set d'ass. sur seuil A075 "grugeage" » : sets **M546** (76171), **M547** (76172), **M548** (76173), **M549** (76372) | Mise en œuvre Système 76 Advanced, registre 2.4.5, p. 7 (PDF p. 200, version octobre 2021, relu en image le 25/09/2026) : 76171 = M150 + M170 + S055 ; 76172 = M173 + M171 + S055 ; 76173 = M174 + 2 × M170 + 2 × S055 ; 76372 = 2 × S055 seules ; aucune M546 à M549 dans le registre 2.4.5 | **aucune** — les deux nomenclatures sont portées : poster sur [Assemblages du système 76](/profiles/systeme-76-assemblages.md), manuel sur [Mise en œuvre du seuil du système 76 Advanced](/procedures/mise-en-oeuvre-seuil-systeme-76.md) ; les M546 à M548 peuvent être les lots de ces pièces, ce qu'aucun document n'écrit | Sets commandés en double, ou meneau 76372 posé sur A075 sans pièce d'étanchéité |
| CTR-35 | Cotes de huit capots aluminium AluClip du système 76 | Poster Système 76 Advanced, profilés principaux, 2022, p. 1 : A385 retour **20,2** × 51 ; A035 20,3 × **108,2** ; A314 20,3 × **72** ; A346 retour **19,7** × 130 ; A045 **16,2 × 74** ; A070 16,3 × **116** ; A051 retour **19,9** × 67,9 ; A052 **19,9 × 85,7** (mm) | Mise en œuvre Système 76 Advanced, registre 2.6.2 « AluClip, Profilés principaux et accessoires », p. 1 à 6 imprimées (PDF p. 305 à 310, versions décembre 2016 et mars 2021, relues en image le 25/09/2026) : A385 retour **20,3** × 51 (sur les sept dormants) ; A035 20,3 × **107,9** ; A314 20,3 × **72,3** ; A346 retour **19,8** × 130 ; A045 **16,3 × 73,8** ; A070 16,3 × « **115.8** » ; A051 retour **19,8** × 67,9 ; A052 **19,8 × 85,9**. Les autres capots cotés sur les deux documents (A033, A034, A313, A042 avec 76281, A043, A044, A048, A069) ont les mêmes cotes | aucune arbitrée : les tableaux de [Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md) portent les deux jeux, chacun avec sa source | contrôle et commande des capots : une cote relevée sur un document ne se retrouve pas sur l'autre (écarts de 0,1 à 0,3 mm) |

# CTR-01 en détail

Les deux formulations ne portent peut-être pas sur le même objet :

| Source | Formulation exacte | Objet |
| --- | --- | --- |
| Cahier technique PERFORM76, p. 3 (PDF 6) | « Charge 100 Kg par ouvrant sur pivot bas » | l'ouvrant, sur le pivot bas |
| Catalogue général, p. 6 | « Pivot pouvant supporter le poids d'une fenêtre jusqu'à 130kg » | la fenêtre entière |

Un ouvrant n'est pas une fenêtre, et un pivot bas n'est pas l'ensemble du ferrage : les deux
chiffres peuvent coexister. Mais tant que personne ne l'a écrit, **l'écart de 30 % reste une
contradiction apparente**, et c'est la valeur basse qui engage.

Le catalogue attribue d'ailleurs les 130 kg aussi bien à la
[PERFORM](/gammes/perform.md) (p. 6) qu'à l'[HYBRIDE](/gammes/hybride.md) (p. 10), sans
distinguer les déclinaisons.

# CTR-03 à CTR-05 : le problème des garanties des gammes +

Les trois contradictions de garantie viennent toutes de la même source, la brochure
PERFORM+/HYBRIDE+ de **mai 2023**, face au catalogue général de **janvier 2026**.

| Composant | Catalogue général, janvier 2026 | Brochure gammes +, mai 2023 |
| --- | --- | --- |
| Structure | 15 ans | jusqu'à 20 ans |
| Laquage | 25 / 10 / 7 ans selon gamme | 7 ans |
| Ferrure Roto | 10 ans sur le fonctionnement | 10 ans sur le fonctionnement |
| Vitrage | 10 ans | 10 ans |
| Volet roulant | 5 ans | 7 ans |

Deux lectures, et elles n'ont pas les mêmes conséquences :

1. **Les gammes + ont des garanties propres, meilleures.** Dans ce cas les deux documents sont
   justes et il manque simplement une colonne « gammes + » dans le catalogue.
2. **Les garanties ont changé entre 2023 et 2026.** Dans ce cas la brochure est périmée et
   continue d'annoncer 20 ans à des clients qui n'en bénéficieront pas.

Les deux valeurs identiques (ferrure Roto, vitrage) ne départagent pas les hypothèses.

## Les neuf sources : la garantie structure suit le produit, pas la date

En classant les neuf documents du wiki par date d'édition :

| Source | Édition | Structure | Volet roulant |
| --- | --- | --- | --- |
| Brochure PERFORM+/HYBRIDE+ | mai 2023 | **20 ans** | 7 ans |
| Dépliant général | juin 2023 | 15 ans | 7 ans |
| Dépliant HYBRIDE | juin 2023 | 15 ans | 7 ans |
| Dépliant LUMÉAL | juin 2023 | 15 ans | 7 ans |
| Dépliant INNOSLIDE | janvier 2024 | 15 ans | 7 ans |
| Brochure LUMINE65 | février 2025 | **20 ans** | 5 ans |
| Brochure HYBRIDE | mars 2025 | 15 ans | non mentionné |
| Catalogue général | janvier 2026 | 15 ans | 5 ans |
| Dépliant LUMÉAL | avril 2026 | 15 ans | 7 ans |

### La chronologie est définitivement écartée

La brochure PERFORM+/HYBRIDE+ annonce **20 ans en mai 2023**. Les trois dépliants de **juin
2023** annoncent **15 ans**. **Un mois d'écart.** Aucune évolution de politique commerciale ne
se produit et ne s'inverse en un mois, sur des documents édités par la même entreprise.

L'ingestion du dépliant général confirme par ailleurs que les garanties ont bougé **dans les
deux sens** entre 2023 et 2026 : panneau de porte 7 → 10 ans, plaxage 5 → 10 ans, laquage 7 ans
forfaitaires → grille 25/10/7, et volet roulant 7 → 5 ans. Une érosion générale n'expliquerait
pas les trois améliorations.

### Ce que la garantie structure suit, en revanche

| Valeur | Documents qui l'annoncent | Produits concernés |
| --- | --- | --- |
| **20 ans** | brochure PERFORM+/HYBRIDE+, brochure LUMINE65 | [PERFORM+](/gammes/perform-plus.md), [HYBRIDE+](/gammes/hybride-plus.md), LUMINE65 |
| **15 ans** | les sept autres | PERFORM, HYBRIDE, LUMINE, TEXTURAL, INNOSLIDE, LUMÉAL, et les deux documents généraux |

**Les trois seuls produits à 20 ans sont les trois gammes de fenêtres haut de gamme les plus
récentes.** Le LUMÉAL, qui est lui aussi à ouvrant caché, reste à 15 ans dans ses deux éditions
— ce n'est donc pas l'ouvrant caché qui fait la différence, mais le positionnement du produit,
et peut-être la distinction fenêtre / coulissant.

**C'est une hypothèse cohérente avec les neuf documents, pas une certitude.** Elle reste à
confirmer auprès du service technique : la question à poser est simplement « quels produits
bénéficient de la garantie 20 ans, et depuis quand ».

### Le volet roulant, lui, reste inexplicable

7 ans dans six documents, 5 ans dans deux — la brochure LUMINE65 et le catalogue général. Ni la
date, ni la gamme ne les séparent : le LUMÉAL et le LUMINE65 appartiennent tous deux à la famille
LUMINE et se contredisent. Aucune valeur n'est retenue.

**Question à poser en priorité** : c'est la seule contradiction du registre qui expose à un
litige contractuel direct. Voir
[Garanties par composant](/garanties/garanties-par-composant.md) et
[Informations à vérifier](/anomalies/informations-a-verifier.md), entrée VER-02.

# CTR-06 et CTR-07 : l'HYBRIDE a changé entre mars 2025 et janvier 2026

Les deux écarts viennent du même couple de documents, et ils vont dans le même sens.

| Sujet | Brochure HYBRIDE, mars 2025 | Catalogue général, janvier 2026 |
| --- | --- | --- |
| Déclinaisons | **aucune**, une gamme unique | HYBRIDE70 et HYBRIDE76 |
| Épaisseur du profilé PVC | 72 mm | 70 à 76 mm |
| Uw | jusqu'à 1,2 W/m²K | jusqu'à 0,8 W/m²K |
| Chambres d'isolation | 5 | 5 |
| Charge du pivot | 130 kg | 130 kg |

Les deux valeurs identiques montrent que les blocs de texte ont été repris d'une édition à
l'autre : ce ne sont pas deux documents indépendants qui se contrediraient par accident, c'est
une fiche mise à jour.

L'hypothèse la plus économique est que **l'HYBRIDE était un produit unique à 72 mm jusqu'en 2025,
scindé ensuite en deux déclinaisons à 70 et 76 mm**, l'HYBRIDE76 apportant le Uw de 0,8 W/m²K.
Elle expliquerait les deux écarts d'un coup.

Si elle se vérifie, deux conséquences pratiques :

- le **72 mm ne correspond plus à aucun produit au catalogue**, mais correspond à toutes les
  HYBRIDE posées avant 2026 — ce qui compte en SAV et en remplacement à l'identique
- le **Uw de 1,2 W/m²K reste la valeur des menuiseries déjà livrées**, et c'est elle qu'il faut
  reprendre dans une étude thermique portant sur de l'existant

**À faire confirmer au bureau d'études.** Voir [HYBRIDE](/gammes/hybride.md) et
[Brochure HYBRIDE](/sources/brochure-hybride.md).

# La règle qui se dégage : le document produit prime sur le document général

Neuf documents plus tard, les contradictions ne se répartissent pas au hasard. **Chaque fois
qu'un document général et un document produit divergent, c'est le général qui est en défaut.**

| Entrée | Document général | Document produit | Qui a raison |
| --- | --- | --- | --- |
| CTR-13 | dépliant général, juin 2023 : Uw HYBRIDE 1,3 | dépliant HYBRIDE, **même mois** : 1,2 | le produit |
| CTR-08 | catalogue : tous coulissants E\*6A / V\*B2 | dépliant LUMÉAL : E\*7A / V\*B3, avec la dimension d'essai | le produit |
| CTR-11 | catalogue : tous coulissants E\*6A / V\*B2 | brochure LUMINE65 : E\*6A / V\*A3, sur 2 vantaux | le produit |
| CTR-09 | catalogue : « autre ferrure » 2 ans | dépliants LUMÉAL 2023 et 2026 : ferrure Technal 10 ans | le produit |
| — | catalogue : aucun classement AEV pour les fenêtres LUMINE | brochure LUMINE65 : A\*4 / E\*9A / V\*C3 | le produit, seul à le donner |
| — | catalogue : laquage Qualicoat 2 « 25 ans » | brochure LUMINE65 : 15 ans d'accroche **et** 25 ans de tenue | le produit, plus précis |

Le mécanisme est compréhensible : un document général résume quatre gammes en trente pages et
arrondit, un document produit reprend les procès-verbaux d'essai de son seul produit.

**Règle de travail, jusqu'à arbitrage contraire : pour une valeur technique — classement AEV,
Uw, acoustique, dimension limite, garantie d'un composant précis — citer le document produit.
Le catalogue général sert à savoir ce qui existe, pas à chiffrer.**

Cette règle ne vaut **pas** pour la garantie structure, où le catalogue et six autres documents
disent 15 ans contre deux documents produit à 20 ans : là, c'est le nombre qui penche de l'autre
côté, et seul le service technique peut trancher.

# Citations

[1] Catalogue menuiseries PROFERM, édition janvier 2026 — `raw/catalogue-general-2026-01.pdf`,
p. 6, 10, 11, 27 et 35
[2] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, pages imprimées 3 et 5 (PDF 6 et 8)
[3] Brochure Nouveautés PERFORM+ et HYBRIDE+, édition mai 2023 —
`raw/brochure-perform-plus-hybride-plus-2023-05.pdf`, p. 3
[4] Brochure HYBRIDE, édition mars 2025 — `raw/brochure-hybride-2025-03.pdf`, p. 2 et 3

# Voir aussi

- [Incohérences internes](/anomalies/incoherences-internes.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
- [Garanties par composant](/garanties/garanties-par-composant.md)
- [Poignée et pivot PERFORM76](/quincaillerie/perform76-poignee-et-pivot.md)
- [Parcloses PERFORM76](/profiles/perform76-parcloses.md)
