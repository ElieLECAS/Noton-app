---
type: Anomalie
title: Incohérences internes
description: Registre des passages où un même document se contredit lui-même ou contient une coquille manifeste, avec la correction probable.
tags: [anomalie, coquille, incoherence, a-corriger]
status: stable
sources:
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
    last_modified: 2025-06-19
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
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
  - resource: raw/catalogue-portes-entree-2024-03.pdf
    id: catalogue-portes-entree-2024-03
    title: Catalogue portes d'entrée PROFERM, édition mars 2024
    last_modified: 2024-03-31
  - resource: raw/nuancier-stores-2020.pdf
    id: nuancier-stores-2020
    title: Nuancier stores PROFERM, 2020
    last_modified: 2020-12-31
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
  - resource: raw/depliant-lumeal-2023-06.pdf
    id: depliant-lumeal-2023-06
    title: Dépliant LUMÉAL, édition juin 2023
    last_modified: 2023-06-30
  - resource: raw/depliant-lumeal-2026-04.pdf
    id: depliant-lumeal-2026-04
    title: Dépliant LUMÉAL, édition avril 2026
    last_modified: 2026-04-21
  - resource: raw/brochure-lumine65-2025-02.pdf
    id: brochure-lumine65-2025-02
    title: Brochure LUMINE65, édition février 2025
    last_modified: 2025-02-28
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
  - resource: raw/moustiquaires/Fiche produit Moustiquaire enroulable verticale SOPROFEN 2026.pdf
    id: soprofen-moustiquaire-enroulable-verticale-2026
    title: Fiche produit Moustiquaire ENROULABLE VERTICALE SOPROFEN
    last_modified: 2026-04-30
  - resource: raw/moustiquaires/Fiche produit volet traditionnel TRADI NON PREMONTE 2025 SOPROFEN.pdf
    id: soprofen-volet-tradi-non-premonte-2025
    title: Fiche produit volet traditionnel TRADI NON PRÉMONTÉ SOPROFEN
    last_modified: 2025-07-31
  - resource: raw/brochure-perform-plus-hybride-plus-2023-05.pdf
    id: brochure-perform-plus-hybride-plus-2023-05
    title: Brochure Nouveautés PERFORM+ et HYBRIDE+, édition mai 2023
    last_modified: 2023-05-31
generated:
  by: process:claude-code
  at: 2026-09-17T20:30:00Z
---

# Ce que contient ce registre

Les cas où **un seul document se contredit lui-même**, ou porte une coquille manifeste. Les
désaccords entre deux documents sont ailleurs :
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

Chaque entrée porte un identifiant stable en `INC-`. Aucune n'est résolue : elles attendent
l'arbitrage du service technique. **Ne jamais corriger le document source** — ce registre dit ce
qu'il faudrait corriger, pas ce qui l'a été.

Une entrée peut aussi être **retirée**, quand la vérification montre que le document ne dit pas
ce que le registre lui prêtait. Elle quitte alors le registre actif pour la section
« Entrées retirées » plus bas, en gardant son identifiant.

# Registre

| ID | Source | Page | Constat | Correction probable | Impact si non corrigé |
| --- | --- | --- | --- | --- | --- |
| INC-02 | Catalogue général | 16 vs 17 | Encadré p. 17 : « Coulissants : Uw jusqu'à 1,4 W/(m2.K) en CV avec Sw = 0,51 et TLw = 0,57 », sur vitrage 6/14/4. Page 16 : LUMÉAL55 à 1,2 (meilleur), LUMINE65 à 1,6 (moins bon). Les deux valeurs produit sont corroborées par leurs propres documents, ce ne sont donc pas des coquilles | Dire à quel coulissant s'applique le 1,4. Le SOLÉAL55 et le GALANDAGE55 n'ont de Uw nulle part dans le corpus : le 1,4 pourrait être leur valeur, laissée sans étiquette, plutôt qu'une valeur de gamme. Si c'est bien une valeur de gamme, donner la plage réelle 1,2 à 1,6 | Réponse fausse en appel d'offres, dans les deux sens : un 1,4 annoncé sur un LUMINE65 promet 0,2 de mieux que le produit, et sous-vend le LUMÉAL55 de 0,2 |
| INC-03 | Cahier technique PERFORM76 | 17 et 18 (PDF 20 et 21) | Blocs de combinaisons titrés « Dormant 76185 » sur les planches des dormants 76171 et 76172 | Libellés copiés : lire 76171 p. 17 et 76172 p. 18 | Nomenclature erronée, dormant commandé au mauvais gabarit |
| INC-04 | Cahier technique PERFORM76 | 17 et 18 (PDF 20 et 21) | « Appui 76751 et 76152 livrés non montés » | Le 76152 n'existe nulle part ailleurs ; lire 76752 | Référence introuvable à la commande |
| INC-05 | Cahier technique PERFORM76 | 3 (PDF 6) | Plages de hauteur d'ouvrant qui se chevauchent dans le tableau de position de poignée | Soit plusieurs positions admissibles par plage, soit bornes erronées : à trancher | Position de poignée indéterminée pour certains ouvrants |
| INC-06 | Cahier technique PERFORM76 | 14 et 16 (PDF 17 et 19) | Tableaux d'appuis des dormants 76177/76185 (p. 14) et 76180 (p. 16) commençant à 60 mm d'isolation, planches de tapées annonçant « Iso de 65 » sans tapée | Raccorder les deux valeurs | Écart mineur, choix d'appui ambigu en isolation faible |
| INC-07 | Nuancier stores 2020 | 9 | « Les Caramels : classe 1 sauf le Curacao 5035 et le Chocolat 5040 » — la classe de ces deux références n'est jamais donnée | Compléter la ligne avec leur classe NF EN 14501 | Deux tissus sur 29 sans classement de protection solaire, impossibles à chiffrer en étude thermique |
| INC-08 | Catalogue portes d'entrée | 142 et 151 | La serrure de la porte aluminium est appelée « 3 pênes pénétrants » p. 142, « à pênes » sur le schéma p. 151 et « à crochets » dans le texte de la p. 151 | Retenir « 3 pênes pénétrants », seule formulation technique détaillée ; corriger le texte de la p. 151 | Trois désignations pour une même serrure, dont une fausse, dans un seul document |
| INC-09 | Mise en œuvre Système 76 Advanced | registre 2.1.2, p. 2 vs p. 17 et 18 (PDF p. 18, 33 et 34) | Le sommaire des profilés annonce « 76303 Meneau de 110 mm » et « 76373 Meneau de 110 mm » ; les planches de détail donnent « 76303 Traverse d'ouvrant 119 mm » et « 76373 Meneau/Traverse 124 mm » | Retenir les planches de détail : 124 mm pour le 76373 recoupe le cahier technique PERFORM76. La valeur de 110 mm portée au sommaire pour le 76373 semble recopiée de la ligne du 76303 juste au-dessus. Relu en image le 25/09/2026 : le sommaire (PDF p. 18) donne bien « Meneau de 110 mm » au 76373 et au 76303 ; la planche du 76373 (PDF p. 33) est titrée « Meneau/Traverse 124 mm » et cotée 124 ; la planche du 76303 (PDF p. 34) est titrée « Traverse d'ouvrant **119 mm** » mais cotée **110** (21 + 68 + 21), comme le poster et la planche de porte d'entrée (PDF p. 13, « 76303 Meneau 110 mm »). Pour le 76303, c'est donc le titre de la planche qui est faux | Meneau commandé au mauvais gabarit, et 14 mm d'erreur sur une largeur hors tout d'ensemble |
| INC-10 | Mise en œuvre Système 76 Advanced | registre 2.3.3, p. 3 vs p. 5 | Deux distances maximales entre points de verrouillage dans le même registre : « les points de verrouillage (paumelle) ne doivent pas être distants de plus de 70 cm » p. 3, « les points de verrouillage (galets etc.) ne doivent pas être distants de plus de 80 cm » p. 5 | Les deux phrases visent probablement des organes différents — paumelles d'un côté, galets de crémone de l'autre. Le manuel ne le dit pas : à faire préciser, et retenir 70 cm en attendant | Entraxe de verrouillage surestimé de 10 cm, donc un point de fermeture en moins sur un grand ouvrant |
| INC-11 | Instructions de montage Roto NX KSR | toutes les pages | Chaque page porte **deux références de document et deux dates** : « Roto NX KSR - IMO_180_NX_FR_v2, Novembre 2022 » en pied de page, et « Roto NX IMO_455_FR_v2 · 07 / 2018 » juste en dessous. Les renvois internes suivent la seconde pagination et pointent vers les pages 215, 216 et 219, qui n'existent pas dans ce document de 124 pages | Le document est un extrait recomposé du manuel IMO_455 de juillet 2018 : renuméroter les renvois, ou indiquer qu'ils visent le manuel complet | Un lecteur qui suit un renvoi ne trouve rien, et personne ne sait laquelle des deux versions fait foi sur une valeur contestée |
| INC-12 | Mise en œuvre Système 70 Plateforme | registre 2.1.2, p. 1 vs p. 8 à 15 | Huit dormants portent deux largeurs différentes : le sommaire des profilés donne 91, 84, 87, 95, 107, 87, 117 et 122 mm pour les 6102, 6104, 6105, 6106, 6107, 6155, 6156 et 6159, leurs planches de détail donnent 57, 64, 67, 75, 87, 67, 97 et 102 mm. L'écart est de 20 mm sur sept d'entre eux et de 34 mm sur le 6102. Les huit autres dormants concordent | Établir ce que mesure chacune des deux pages. L'écart constant de 20 mm ressemble à deux conventions de mesure, pas à huit coquilles, mais aucune des deux pages ne le dit | Dormant commandé au mauvais gabarit sur la moitié de la gamme. La planche du 6159 est en outre titrée « 6156 », libellé recopié comme dans INC-03 |
| INC-13 | Catalogue Roto NX pour profils PVC | 36 | Dans un catalogue français, le tableau des champs d'application de la version 150 kg est **imprimé en allemand** — « Flügelfalzbreite », « Grundsicherheit », « unzulässiger Anwendungsbereich » — et il désigne les classes de sécurité par « RC » là où la page 35, en français, écrit « CDR » | Traduire la page 36. Retenir que RC et CDR désignent la même classification, celle de la DIN EN 1627-1630 | Un lecteur français ne lit pas les bornes de la ferrure 150 kg, qui est celle des vantaux lourds |
| INC-14 | DTD n° DBV-25-6/16-2334_V5 ; DTA n° 6/16-2334_V5 | DTD 47 ; DTA 43 | Le tableau d'assignation du drainage groupe les ouvrants « 76271, 76272, **78275**, 76279, 76281 » — la référence **78275** ne correspond à aucun profilé connu du système, alors que le **76275** est l'un des quatre ouvrants PERFORM76 et manque justement à cette liste. La même liste, avec le même 78275, figure sur la planche « Drainage et Décompression couleur sombre » du DTA (p. 43) | Coquille probable : lire 76275 | Un lecteur cherchant le drainage du 76275 ne le trouve pas, cherché sous 78275 il ne trouve rien non plus |
| INC-15 | Fiche produit Moustiquaire ENROULABLE VERTICALE SOPROFEN | 2 | La note de bas de page relative aux manœuvres manuelles porte « Pour MONO 54 CH » là où le tableau de dimensions et la planche technique nomment le produit « MOHO 54 CH » [6 p. 2] | Coquille manifeste : lire « MOHO 54 CH » | Risque de confusion de référence lors de la commande ou de l'intégration |
| INC-16 | Fiche produit volet traditionnel TRADI NON PRÉMONTÉ SOPROFEN | 1 | La rubrique « Facilité de pose » porte « Pose rapide grâce à ses consoles et tablier prémontés, solidaires de l'axe » et « Auto-portant sans déport », recopiés mot pour mot de la fiche TRADI PRÉMONTÉ, alors que la fiche concerne le volet non prémonté avec « Déport sur mesure » [7 p. 1] | Erreur de copier-coller manifeste : supprimer la mention de prémontage et d'autoportance sans déport pour ce modèle | Confusion sur le niveau de pré-assemblage et le mode de pose en atelier et chantier |
| INC-17 | Cahier technique PERFORM76 | 3 (PDF 6) | Pivot bas : les deux vis sont libellées « Réglage hauteur clé 6 pans de 4 mm (+ ou - 2 mm) », y compris la vis basse que désignent des flèches horizontales | La seconde est probablement le réglage latéral ; à confirmer au service technique | Réglage latéral du pivot non documenté, risque de dérégler la hauteur en voulant corriger le jeu latéral |
| INC-18 | Poster Système 76 Advanced, profilés principaux, 2022 | 1 | Le capot **A042** est dessiné deux fois avec deux hauteurs : « 43,5 » à côté de l'ouvrant 76281, « 43 » à côté de l'ouvrant 76274 ; la largeur du retour est de 16,5 mm sur les deux dessins. Même dédoublement dans le manuel de mise en œuvre, registre 2.1.2 : « E: 126,9 mm » sur la planche du 76281 (PDF p. 27), « E: 126,8 mm » sur celle du 76274 (PDF p. 31), S 68 mm sur les deux | Soit une seule hauteur est juste, soit les deux dessins désignent deux capots distincts sous la même référence : à trancher chez profine | Capot commandé ou débité sur une hauteur fausse de 0,5 mm, défaut d'emboîtement sur l'ouvrant |
| INC-19 | Poster Système 76 Advanced, profilés principaux, 2022 | 1 | Le renfort des dormants 76177, 76178 et 76185 est écrit « V291.Z » dans les trois coupes de dormant, mais « V291.Z 1 » sous le dessin coté du renfort (43 × 29 mm, IG 0,74, IW 2,26 cm⁴) | Lire V291.Z ; le « 1 » est probablement un indice de version ou une coquille | Référence introuvable à la commande si elle est recopiée avec le « 1 » |
| INC-20 | Poster Système 76 Advanced, profilés complémentaires, 2022 | 1 | Les inerties des renforts **V075** et **V264** sont écrites « I = 1,0 cm⁴ / I = 4,1 cm⁴ » et « I = 0,5 cm⁴ / I = 1,8 cm⁴ », sans l'indice G ou W que portent les dix autres renforts de la planche | Rétablir IG et IW ; l'ordre des autres renforts (IG puis IW) donne V075 IG 1,0 / IW 4,1, alors que le manuel de mise en œuvre donne au V075 IW 1,0 / IG 4,1 | Inertie de poids prise pour l'inertie de vent, dimension admissible fausse d'un facteur 4 |
| INC-21 | DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED | 5, 6, 7, 16 et 41 | Le texte de l'avis et du dossier technique écrit l'élargisseur d'ouvrant « **EO20545** » (lettre O) quatre fois (p. 5, 6 et 7) ; la planche « Elargisseur ouvrant » de l'annexe l'écrit « **E020545** (Alu - PA66) » (chiffre zéro) (p. 16), et la coupe de principe « Elargisseur d'ouvrant » de nouveau « **EO20545** » (lettre O) (p. 41). Le DTD n° DBV-25-6/16-2334_V5 porte les mêmes graphies aux mêmes endroits : « EO20545 » dans le texte (p. 3, 5), « E020545 (Alu - PA66) » sur la planche (p. 15), « EO20545 » sur la coupe de principe (p. 45) | Une seule des deux graphies est la référence profine ; la planche et le texte ne permettent pas de trancher | Référence introuvable à la commande ou dans un catalogue si la mauvaise graphie est recopiée |
| INC-22 | DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED | 6, 19 et 20 | La référence **76579** est une parclose de 48 mm sur la planche « Parcloses PVC » (p. 19, quatrième ligne, avec 76578 à 46 mm) et un profilé « Pose ITE (PVC) » d'une autre forme sur la planche « Profilés complémentaires » (p. 20) ; la remarque 1.3 (p. 6) parle du « profilé PVC réf. 76579 formant goutte d'eau en cas de mise en œuvre en applique extérieure » | Une des deux pièces porte une autre référence ; la parclose 76579 est aussi celle du cahier technique PERFORM76, ce qui désigne le profilé « Pose ITE » comme le plus probablement mal référencé, sans que le document permette de le trancher | Profilé goutte d'eau commandé sous 76579 et livré en parclose, ou l'inverse ; prescription 1.3 appliquée à la mauvaise pièce |
| INC-23 | DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED | 26 et 27 | Les pages 26 et 27, « Assemblage capots — Sur traverse dormant », sont identiques pixel pour pixel hors numéro de page | Une des deux pages est un doublon ; savoir auprès de profine si une autre planche (capots d'ouvrant ou de meneau, par exemple) devait y figurer. Le DTD n° DBV-25-6/16-2334_V5 porte le même doublon (p. 30 et 31, identiques hors numéro de page) ; il ne comble donc pas le manque, mais il dessine deux planches absentes du DTA, « Assemblage meneau / traverse intermédiaire dormant » (p. 25) et « … ouvrant » avec « Assemblage traverse complémentaire » (p. 26) | Le montage des capots d'une autre partie de la menuiserie n'est pas documenté par le DTA |
| INC-24 | DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED | 8 et 35 | Le § 2.2.3.3.4 (p. 8) assure l'équilibrage de pression par « la suppression de la lèvre du joint sur une longueur de **300 mm minimum** » ; la planche « Drainages et décompressions — Variante équilibrage de pression » (p. 35) montre la lèvre supprimée « sur **100 mm minimum** ». La planche avec capotage (p. 36) porte, elle, « sur 300 mm minimum ». Le DTD n° DBV-25-6/16-2334_V5 porte deux textes : le § 2.2.3.1.2 « Drainage et équilibrage de pression » du cadre dormant (p. 4), absent du DTA, donne « la mise en place d'un joint plat 9043 sur une longueur de **100 mm minimum** » et « la suppression de la lèvre du joint sur une longueur de **100 mm minimum** » ; le § 2.2.3.3.4 du capotage (p. 6) donne 300 mm, comme le DTA. Les planches du DTD sont celles du DTA (p. 39 : 100 mm, p. 40 avec capotage : 300 mm). Le DTD lie donc 100 mm au dormant nu et 300 mm au dormant capoté | Retenir 300 mm, valeur du texte et de la planche capotée, sous réserve de confirmation profine | Équilibrage de pression insuffisant si la lèvre n'est supprimée que sur 100 mm, et eau retenue en feuillure |
| INC-25 | DTD n° DBV-25-6/16-2334_V5, système 76 Advanced ; DTA n° 6/16-2334_V5 | DTD 4, 23 et 24 ; DTA 22 | L'embout de la pièce d'appui 6137 est écrit « **9F56.1** » dans le § 2.2.3.1.3 (DTD p. 4) et sur la planche « Embouts de pièces d'appui » (« EMBOUT 9F56.1 POUR 6137 (2 PIECES) », DTD p. 23), mais « Embouts de remplissage **9F56** », sans le suffixe .1, sur la coupe « Assemblage fourrure et pièce d'appui PVC » (DTD p. 24, DTA p. 22) | Lire 9F56.1 ; le suffixe .1 est porté par le texte et par la planche cotée, sous réserve que 9F56 ne désigne pas une version antérieure de l'embout | Embout commandé sous la référence sans suffixe, et livré dans une version qui n'est pas celle cotée |
| INC-26 | Mise en œuvre Système 76 Advanced | registre 2.1.1 Porte d'entrée, p. 4 et 5 imprimées (PDF p. 5 et 6) | Planche du dormant 76101 (PDF p. 5) : « **M529** — Clip de montage pour demi capot », avec le dessin d'un clip. Planche du dormant 76102 (PDF p. 6) : « **M529** — Pièce de remplissage pour V325 » et, dans une autre case, « **M569** — Clip de montage pour demi capot ». Relu à 500 dpi | Sur la planche du 76101, lire M569 : ailleurs dans le même registre la M529 est une pièce de remplissage de renfort, comme les M527 (pour V309) et M528 (pour V314), et le clip du demi-capot A385 est la M569 | Clip du demi-capot commandé sous la référence d'une pièce de remplissage, ou l'inverse |
| INC-27 | Mise en œuvre Système 76 Advanced | registre 2.1.1 Porte d'entrée, p. 7 imprimée (PDF p. 8) | Le renfort de l'ouvrant 76201 est écrit « **V308\*\*** », deux astérisques sans aucun renvoi sur la planche. Sur les planches des ouvrants 76206 et 76207 (PDF p. 9 et 10), « \*\* » renvoie à « Pour serrure Maco et Winkhaus avec hauteur de poignée de 1050 mm » | Soit un V308 usiné pour serrure Maco et Winkhaus, soit une marque restée d'une autre planche : à faire préciser par profine | Renfort non usiné monté là où la serrure exige un renfort usiné, ou l'inverse |
| INC-28 | Mise en œuvre Système 76 Advanced | registre 2.1.2, p. 5 imprimée (PDF p. 21), planche du dormant 76173 | Deux désignations permutées dans la dernière rangée d'accessoires : « **S050** — Insert de compensation pour aluClip dormant », avec le dessin d'une douille ; « **G023** — Gabarit de perçage pour A076/A077 », avec le dessin d'un insert. Sur les planches des 76171 et 76172 (PDF p. 19 et 20), G023 est l'« Insert de compensation pour aluClip dormant » et S048, S049 sont des « Douille de montage » | S050 : douille de montage ; G023 : insert de compensation pour aluClip dormant ; le gabarit de perçage pour A076/A077 est le T021 (PDF p. 37) | Insert commandé à la place d'une douille, ou l'inverse, pour le dormant 76173 |
| INC-29 | Mise en œuvre Système 76 Advanced | registre 2.1.2, p. 3 et 5 imprimées (PDF p. 19 et 21) | La pièce de remplissage **M170** est cotée « **8 x 32 x 70 mm** » sur la planche du dormant 76171 et « **28 x 32 x 70 mm** » sur celle du dormant 76173 ; même désignation, même dessin. Relu à 500 dpi | Un « 2 » manquant sur la planche du 76171 est probable, sans certitude : à faire confirmer par profine | Pièce de remplissage débitée ou commandée à 8 mm d'épaisseur au lieu de 28 |
| INC-30 | Mise en œuvre Système 76 Advanced | registre 2.1.2, p. 17 et 18 imprimées (PDF p. 33 et 34) | Le renfort **V324** est désigné « **Capot Alu** » sur la planche du meneau 76373, avec le dessin d'un renfort en L de 44 × 55 mm en tôle de 2,5 mm et « IG = 17,1 cm⁴, IW = 13,3 cm⁴ » ; sur la planche de la traverse 76303, il est désigné « Renfort 2,5 mm » mais sa case « Valeurs » porte « E: 290,4 mm, S: 150 mm », les valeurs du capot A070 de la case voisine | V324 : renfort 2,5 mm, IG 17,1 et IW 13,3 cm⁴ sur les deux profilés, à faire confirmer par profine pour le 76303 | Inertie du V324 dans la traverse 76303 introuvable, ou valeurs de capot prises pour des inerties dans un calcul statique |
| INC-31 | Catalogue général | 17 | Sur la page « Accessoirisez vos coulissants & galandages » de la gamme aluminium LUMINE, le bloc VITRAGES écrit « Nos menuiseries **PERFORM** peuvent être équipées de vitrages techniques. Plus de détails en page 27. », alors que le bloc VOLETS ROULANTS de la même page écrit « Nos menuiseries **LUMINE** » | Lire « Nos menuiseries LUMINE » : bloc recopié de la page PERFORM | Un lecteur peut croire que les vitrages techniques de la p. 27 ne concernent que la PERFORM et pas les coulissants aluminium |
| INC-32 | Brochure Nouveautés PERFORM+ et HYBRIDE+, mai 2023 | 3 | Le texte d'introduction des garanties s'arrête sur « PROFERM vous offre des garanties exceptionnelles et exclusives sur l'ensemble des », sans le mot qui suit ; la grille commence en dessous | Coquille : lire « sur l'ensemble des composants », comme la même phrase des brochures HYBRIDE (mars 2025, p. 3) et LUMINE65 | Aucun sur les durées ; la portée de l'engagement (« l'ensemble des » quoi) n'est pas écrite sur ce document |

# INC-02 en détail

Toutes les valeurs de Uw qui concernent un coulissant, relevées dans l'ensemble du corpus.
Uw en W/m²K, une ligne par énoncé de source.

| Énoncé | Uw (W/m²K) | Conditions données par la source | Source |
| --- | --- | --- | --- |
| LUMÉAL55 | 1,2 | Sw 0,46 et TLw 0,65 | Catalogue général, p. 16 |
| LUMÉAL55 | 1,2 | Sw 0,46 et TLw 0,65 | Dépliant LUMÉAL, édition juin 2023, p. 2 |
| LUMÉAL55 | 1,2 | Sw 0,46 et TLw 0,65 | Dépliant LUMÉAL, édition avril 2026, p. 2 |
| LUMINE65 | 1,6 | coulissant 2 vantaux 2 180 × 2 350 mm | Catalogue général, p. 16 |
| LUMINE65 | 1,6 | coulissant 2 vantaux 2 180 × 2 350 mm | Brochure LUMINE65, édition février 2025, p. 3 |
| SOLÉAL55 | - | aucune valeur dans aucune source du corpus | - |
| GALANDAGE55 | - | aucune valeur dans aucune source du corpus | - |
| « Coulissants », sans produit nommé | 1,4 | « en CV », Sw 0,51, TLw 0,57, vitrage 6/14/4 | Catalogue général, p. 17 |

Ce que ce relevé établit.

**Les valeurs produit ne sont pas des coquilles.** Le 1,2 du LUMÉAL55 et le 1,6 du
LUMINE65 figurent à l'identique, conditions d'essai comprises, dans les documents propres
à ces produits. Conformément à la règle du wiki — un document produit l'emporte sur un
document général sur une valeur technique — ce sont ces deux valeurs qu'il faut retenir
pour un chiffrage, et l'énoncé de la page 17 qui est en cause.

**La ligne « fenêtres » du même encadré, elle, se vérifie.** La page 17 annonce « Uw
jusqu'à 1,5 W/(m2.K) » pour les fenêtres, ce qui recoupe le 1,51 de la brochure LUMINE65
(fenêtre 1 vantail 1 010 × 1 365 mm, p. 2). Seule la ligne « coulissants » flotte.

**Le SOLÉAL55 et le GALANDAGE55 n'ont de Uw nulle part.** Ce sont les deux seuls des
quatre coulissants de la page 16 dans ce cas. Le 1,4 de la page 17 pourrait donc être
leur valeur, laissée sans étiquette, et non une valeur de gamme qui contredirait les deux
autres. Ce serait alors une donnée non attribuée plutôt qu'une incohérence.

**Ce qui n'a pas pu être tranché sur pièce.** Savoir si l'encadré « Le saviez-vous ? »
est placé dans la colonne SOLÉAL & GALANDAGE de la page 17 demande de regarder la
planche : la couche texte du PDF entrelace les colonnes et ne permet aucune conclusion
sur le placement. La question est à poser au service technique, en lui montrant la page.

Le sigle « CV » de cet encadré n'est défini nulle part — entrée **VER-35** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Voir [Coulissants aluminium](/gammes/coulissants-aluminium.md).

# Entrées retirées

Une entrée retirée **garde son identifiant à vie** : il n'est jamais réattribué, pour que le
suivi du service technique ne se décale pas. Une entrée ne figure ici que parce que la
vérification sur le document a montré que l'incohérence n'existait pas.

| ID | Retirée le | Ce que l'entrée affirmait | Ce que dit réellement le document |
| --- | --- | --- | --- |
| INC-01 | 2026-09-18 | Le catalogue général noterait le triple vitrage « 40 mm 4/14/14/4 » page 10, contre « 4/14/4/14/4 » page 27, la première notation ne totalisant ni 40 mm ni trois verres | Les pages 10 et 27 portent **toutes deux `4/14/4/14/4`**. La notation `4/14/14/4` n'apparaît nulle part dans le catalogue. La coquille avait été introduite à la transcription dans le wiki, le document source est cohérent |

INC-01 a été signalée par elie et vérifiée sur `raw/catalogue-general-2026-01.pdf` : extraction
du texte des pages 10 et 27, puis recherche de toutes les notations de vitrage du document. Les
pages qui répercutaient cette fausse contradiction ont été corrigées le même jour —
[Performances des vitrages](/vitrages/performances-vitrages.md), [HYBRIDE](/gammes/hybride.md)
et [Catalogue menuiseries PROFERM, édition janvier 2026](/sources/catalogue-general-2026.md).

# INC-05 en détail

Les plages de hauteur d'ouvrant du tableau de position de poignée [2 p. 3] se recouvrent à quatre endroits :

| Chevauchement | Plages concernées | Positions FFO données |
| --- | --- | --- |
| Plage identique | 300-600 mm, deux fois | 120 mm et 170 mm |
| Plage identique | 601-900 mm, deux fois | 220 mm et 263 mm |
| Recouvrement partiel | 601-900 et 801-1 000 mm | 263 mm et 413 mm |
| Recouvrement partiel | 1 201-1 800 et 1 601-1 800 mm | 563 mm et 763 mm |

Un ouvrant de 850 mm relève de deux lignes qui donnent **263 mm et 413 mm** : 150 mm d'écart, ce
n'est pas une tolérance. Le cas le plus problématique.

Deux lectures possibles, le document ne tranche pas : soit chaque ligne propose une position
admissible parmi d'autres, au choix du client ou de l'atelier, soit certaines bornes sont
fausses. **À faire valider avant d'automatiser ce tableau**, en particulier s'il doit alimenter
un configurateur. Voir
[Poignée et pivot PERFORM76](/quincaillerie/perform76-poignee-et-pivot.md).

# Citations

[1] Catalogue menuiseries PROFERM, édition janvier 2026 — `raw/catalogue-general-2026-01.pdf`,
p. 16 et 17
[2] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`, pages imprimées 3, 14, 16, 17 et 18 (PDF 6, 17, 19, 20 et 21)
[3] Dépliant LUMÉAL, édition juin 2023 — `raw/depliant-lumeal-2023-06.pdf`, p. 2
[4] Dépliant LUMÉAL, édition avril 2026 — `raw/depliant-lumeal-2026-04.pdf`, p. 2
[5] Brochure LUMINE65, édition février 2025 — `raw/brochure-lumine65-2025-02.pdf`, p. 2 et 3
[6] Fiche produit Moustiquaire ENROULABLE VERTICALE SOPROFEN, réf. DOC83151 Version 042026 — `raw/moustiquaires/Fiche produit Moustiquaire enroulable verticale SOPROFEN 2026.pdf`, p. 2
[7] Fiche produit volet traditionnel TRADI NON PRÉMONTÉ SOPROFEN, réf. DOC83116 Version 07/2025 — `raw/moustiquaires/Fiche produit volet traditionnel TRADI NON PREMONTE 2025 SOPROFEN.pdf`, p. 1

# Voir aussi

- [Contradictions entre sources](/anomalies/contradictions-entre-sources.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
- [Performances des vitrages](/vitrages/performances-vitrages.md)
- [Coulissants aluminium](/gammes/coulissants-aluminium.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Moustiquaires enroulables verticales](/equipements/moustiquaires-enroulables-verticales.md)
- [Volet roulant traditionnel TRADI NON PRÉMONTÉ](/equipements/volets-roulants-tradi-non-premonte.md)
