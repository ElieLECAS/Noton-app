---
type: Anomalie
title: Incohérences internes
description: Registre des passages où un même document se contredit lui-même ou contient une coquille manifeste, avec la correction probable.
tags: [anomalie, coquille, incoherence, a-corriger]
status: stable
sources:
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
| INC-09 | Mise en œuvre Système 76 Advanced | registre 2.1.2, p. 2 vs p. 17 et 18 | Le sommaire des profilés annonce « 76303 Meneau de 110 mm » et « 76373 Meneau de 110 mm » ; les planches de détail donnent « 76303 Traverse d'ouvrant 119 mm » et « 76373 Meneau/Traverse 124 mm » | Retenir les planches de détail : 124 mm pour le 76373 recoupe le cahier technique PERFORM76. La valeur de 110 mm portée au sommaire pour le 76373 semble recopiée de la ligne du 76303 juste au-dessus | Meneau commandé au mauvais gabarit, et 14 mm d'erreur sur une largeur hors tout d'ensemble |
| INC-10 | Mise en œuvre Système 76 Advanced | registre 2.3.3, p. 3 vs p. 5 | Deux distances maximales entre points de verrouillage dans le même registre : « les points de verrouillage (paumelle) ne doivent pas être distants de plus de 70 cm » p. 3, « les points de verrouillage (galets etc.) ne doivent pas être distants de plus de 80 cm » p. 5 | Les deux phrases visent probablement des organes différents — paumelles d'un côté, galets de crémone de l'autre. Le manuel ne le dit pas : à faire préciser, et retenir 70 cm en attendant | Entraxe de verrouillage surestimé de 10 cm, donc un point de fermeture en moins sur un grand ouvrant |
| INC-11 | Instructions de montage Roto NX KSR | toutes les pages | Chaque page porte **deux références de document et deux dates** : « Roto NX KSR - IMO_180_NX_FR_v2, Novembre 2022 » en pied de page, et « Roto NX IMO_455_FR_v2 · 07 / 2018 » juste en dessous. Les renvois internes suivent la seconde pagination et pointent vers les pages 215, 216 et 219, qui n'existent pas dans ce document de 124 pages | Le document est un extrait recomposé du manuel IMO_455 de juillet 2018 : renuméroter les renvois, ou indiquer qu'ils visent le manuel complet | Un lecteur qui suit un renvoi ne trouve rien, et personne ne sait laquelle des deux versions fait foi sur une valeur contestée |
| INC-12 | Mise en œuvre Système 70 Plateforme | registre 2.1.2, p. 1 vs p. 8 à 15 | Huit dormants portent deux largeurs différentes : le sommaire des profilés donne 91, 84, 87, 95, 107, 87, 117 et 122 mm pour les 6102, 6104, 6105, 6106, 6107, 6155, 6156 et 6159, leurs planches de détail donnent 57, 64, 67, 75, 87, 67, 97 et 102 mm. L'écart est de 20 mm sur sept d'entre eux et de 34 mm sur le 6102. Les huit autres dormants concordent | Établir ce que mesure chacune des deux pages. L'écart constant de 20 mm ressemble à deux conventions de mesure, pas à huit coquilles, mais aucune des deux pages ne le dit | Dormant commandé au mauvais gabarit sur la moitié de la gamme. La planche du 6159 est en outre titrée « 6156 », libellé recopié comme dans INC-03 |
| INC-13 | Catalogue Roto NX pour profils PVC | 36 | Dans un catalogue français, le tableau des champs d'application de la version 150 kg est **imprimé en allemand** — « Flügelfalzbreite », « Grundsicherheit », « unzulässiger Anwendungsbereich » — et il désigne les classes de sécurité par « RC » là où la page 35, en français, écrit « CDR » | Traduire la page 36. Retenir que RC et CDR désignent la même classification, celle de la DIN EN 1627-1630 | Un lecteur français ne lit pas les bornes de la ferrure 150 kg, qui est celle des vantaux lourds |
| INC-14 | DTD n° DBV-25-6/16-2334_V5 | 47 | Le tableau d'assignation du drainage groupe les ouvrants « 76271, 76272, **78275**, 76279, 76281 » — la référence **78275** ne correspond à aucun profilé connu du système, alors que le **76275** est l'un des quatre ouvrants PERFORM76 et manque justement à cette liste | Coquille probable : lire 76275 | Un lecteur cherchant le drainage du 76275 ne le trouve pas, cherché sous 78275 il ne trouve rien non plus |
| INC-15 | Fiche produit Moustiquaire ENROULABLE VERTICALE SOPROFEN | 2 | La note de bas de page relative aux manœuvres manuelles porte « Pour MONO 54 CH » là où le tableau de dimensions et la planche technique nomment le produit « MOHO 54 CH » [6 p. 2] | Coquille manifeste : lire « MOHO 54 CH » | Risque de confusion de référence lors de la commande ou de l'intégration |
| INC-16 | Fiche produit volet traditionnel TRADI NON PRÉMONTÉ SOPROFEN | 1 | La rubrique « Facilité de pose » porte « Pose rapide grâce à ses consoles et tablier prémontés, solidaires de l'axe » et « Auto-portant sans déport », recopiés mot pour mot de la fiche TRADI PRÉMONTÉ, alors que la fiche concerne le volet non prémonté avec « Déport sur mesure » [7 p. 1] | Erreur de copier-coller manifeste : supprimer la mention de prémontage et d'autoportance sans déport pour ce modèle | Confusion sur le niveau de pré-assemblage et le mode de pose en atelier et chantier |
| INC-17 | Cahier technique PERFORM76 | 3 (PDF 6) | Pivot bas : les deux vis sont libellées « Réglage hauteur clé 6 pans de 4 mm (+ ou - 2 mm) », y compris la vis basse que désignent des flèches horizontales | La seconde est probablement le réglage latéral ; à confirmer au service technique | Réglage latéral du pivot non documenté, risque de dérégler la hauteur en voulant corriger le jeu latéral |

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
