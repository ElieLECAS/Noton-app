---
type: Anomalie
title: Informations à vérifier
description: Registre des affirmations périmées, des données manquantes et des rattachements non sourcés relevés dans la documentation PROFERM.
tags: [anomalie, a-verifier, perime, donnee-manquante]
status: stable
sources:
  - resource: raw/catalogue-general-2026-01.pdf
    id: catalogue-general-2026
    title: Catalogue menuiseries PROFERM, édition janvier 2026
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
  - resource: raw/brochure-perform-plus-hybride-plus-2023-05.pdf
    id: brochure-perform-plus-hybride-plus-2023-05
    title: Brochure Nouveautés PERFORM+ et HYBRIDE+, édition mai 2023
  - resource: raw/brochure-hybride-2025-03.pdf
    id: brochure-hybride-2025-03
    title: Brochure HYBRIDE, édition mars 2025
  - resource: raw/depliant-innoslide-2024-01.pdf
    id: depliant-innoslide-2024-01
    title: Dépliant INNOSLIDE, édition janvier 2024
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
  - resource: raw/depliant-general-2023-06.pdf
    id: depliant-general-2023-06
    title: Dépliant général PROFERM, édition juin 2023
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
  - resource: raw/profine-mise-en-oeuvre-9708-montants-cintres-2017-07.pdf
    id: profine-9708-montants-cintres
    title: Mise en œuvre 9708, profine France, juillet 2017
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    id: dtd-6-16-2335-v5
    title: DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION
  - resource: raw/roto-nx-transformation-of-en-ob-2026-03.pdf
    id: proferm-transformation-of-ob-roto-nx
    title: Transformation OF en OB gamme ROTO NX, PROFERM, réf. PRO-PVC-OFOB-01
  - resource: raw/roto-nx-bras-report-de-charge.pdf
    id: roto-nx-report-de-charge
    title: Bras de report de charge ROTO NX
  - resource: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf
    id: proferm-eneo-cc-notice-2022
    title: Roto Safe E Eneo CC, notice simplifiée PROFERM, version 2, 2022
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage, réf. IMO_180_NX_FR_v2
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
generated:
  by: process:claude-code
  at: 2026-09-17T20:30:00Z
---

# Ce que contient ce registre

Trois natures de problème, qui ne sont ni des coquilles ni des contradictions :

- une **affirmation périmée** : le document dit vrai à sa date, plus aujourd'hui
- une **donnée manquante** : le wiki a besoin d'un chiffre qu'aucune source ne donne
- un **rattachement non sourcé** : une déduction plausible qu'aucun document n'écrit

Chaque entrée porte un identifiant stable en `VER-`. La distinction avec les autres registres
tient à l'action attendue : ici il faut **aller chercher une information**, pas corriger un
document.

# Affirmations périmées

| ID | Source | Page | Constat | À faire |
| --- | --- | --- | --- | --- |
| VER-01 | Catalogue général | 8, 12, 20 | « Profil 76 mm design disponible à compter du 2ème trimestre 2026 » | Échéance dépassée : confirmer la disponibilité effective et corriger la mention |
| VER-02 | Brochure PERFORM+/HYBRIDE+ | tout le document | Brochure de mai 2023 ; les gammes PERFORM+ et HYBRIDE+ **n'apparaissent nulle part** dans le catalogue général de janvier 2026 | Établir si les gammes + sont toujours commercialisées, renommées, ou abandonnées |
| VER-18 | Nuancier stores | tout le document | Nuancier de **2020**, source la plus ancienne du wiki ; les stores intégrés n'apparaissent dans **aucune** autre source, pas même le catalogue général qui traite pourtant les volets roulants | Établir si les stores intégrés sont toujours commercialisés, et si la condition de vitrage **4+16+4 clair** vaut encore alors que le vitrage de série est passé à 6/18/4 de 28 mm |
| VER-19 | Nuancier vitrages décoratifs | tout le document | **Aucune date d'édition, aucun code de version** — seul document du wiki dans ce cas | Dater le document, pour pouvoir trancher `CTR-14` face au catalogue |

**VER-02 est la question la plus structurante du registre.** Tant qu'elle n'est pas tranchée, le
statut des pages [PERFORM+](/gammes/perform-plus.md) et
[HYBRIDE+](/gammes/hybride-plus.md) reste incertain, et les contradictions de garantie CTR-03 à
CTR-05 restent insolubles. Voir
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

Un catalogue général qui ignore deux gammes lancées trois ans plus tôt n'a que deux explications
possibles : les gammes ont disparu, ou le catalogue est incomplet. Les deux méritent d'être
sues.

# Données manquantes

| ID | Sujet | Source muette | À demander à |
| --- | --- | --- | --- |
| VER-35 | Sens des sigles « DV » et « CV » qui qualifient les valeurs thermiques | Catalogue général, p. 17 : les deux sigles n'apparaissent qu'à cette page et ne sont définis nulle part dans le corpus. « DV » se lit « double vitrage » sans que la source l'écrive ; « CV » n'a pas de lecture évidente, alors que la même mention annonce un vitrage 6/14/4, qui est un double vitrage. Tant que le sens de « CV » n'est pas établi, on ne sait pas à quelle configuration se rapporte le Uw de 1,4 W/(m².K) des coulissants — voir `INC-02` | Service technique |
| VER-03 | Nombre de chambres et de joints de la PERFORM70 | Catalogue général, p. 6 : seule la PERFORM76 est décrite. Le système 70 Plateforme de profine compte **5 chambres** ([Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md)), mais rien ne dit que la PERFORM70 repose dessus — la réponse dépend de `VER-28` | Bureau d'études |
| VER-04 | Épaisseurs de profilé HYBRIDE70 et HYBRIDE76 | Catalogue général, p. 10 : une plage 70-76 mm pour la gamme entière. La brochure HYBRIDE de mars 2025 donne 72 mm pour une gamme alors unique — voir `CTR-06` | Bureau d'études |
| VER-05 | Épaisseur de profilé de la gamme TEXTURAL | Catalogue général, p. 20-23 : aucune cote | Bureau d'études |
| VER-06 | Composition du double vitrage « SGC ULTRA ONE » | Catalogue général, p. 27 : annoncé sans composition | Fournisseur de vitrage |
| VER-07 | Diamètre des paumelles Fapim Tube | Catalogue général, p. 32 : donné pour les Roto Solid B, pas pour les Fapim | Fournisseur |
| VER-08 | Tapées compatibles avec le dormant 76172 | Cahier technique PERFORM76 : aucune planche de tapées pour ce dormant | Bureau d'études |
| VER-09 | Abaque hauteur de coffre de volet roulant | Catalogue général, p. 28 : 200 ou 230 mm « selon la dimension de la menuiserie », sans abaque | Bureau d'études |
| VER-10 | Abaques de renfort KÖMMERLING | Catalogue général, p. 10 et 23 : « renfort selon abaques », abaques absents | KÖMMERLING ou bureau d'études |
| VER-21 | Compatibilité du dormant 76185 avec les battements 76471, 76472 et 76473 | Mise en œuvre Système 76 Advanced, registre 2.3.1, p. 5, 6 et 7 : les trois planches de cotes de débit avec battement listent 76171, 76172, 76173, 76177, 76178 et 76180, **jamais le 76185**, qui figure pourtant sur la planche générale des dormants | profine ou bureau d'études — savoir s'il s'agit d'une exclusion ou d'un oubli |
| VER-22 | Distinction entre les cas « a » et « b usiné » des cotes de débit de seuil aluminium | Mise en œuvre Système 76 Advanced, registre 2.3.1, p. 8 : l'en-tête à trois colonnes n'explicite pas ce qui est usiné dans chaque cas, alors que le débit du renfort vertical change de 9 mm entre les deux | profine — l'écart porte sur un renfort déjà coupé |
| VER-23 | Épaisseurs d'acier des trois renforts du meneau 76373, et inertie du V324 sur la traverse 76303 | Mise en œuvre Système 76 Advanced, registre 2.1.2, p. 17 : le cartouche du 76373 porte les inerties mais les étiquettes de référence du 76372 voisin. La légende du dessin donne V323.Z, V322 et V324 | profine — à relire sur le document avant tout calcul statique |
| VER-24 | Orientation de montage du renfort V317 | Mise en œuvre Système 76 Advanced : IW 4,8 / IG 2,3 sur le battement 76472 (registre 2.1.2, p. 16), IW 2,3 / IG 4,8 sur la réhausse 76705 (registre 2.1.3, p. 4). La permutation est cohérente avec un montage tourné de 90°, le manuel ne l'écrit pas | profine — confirmer que c'est bien l'orientation qui change et non une coquille |
| VER-25 | Longueur de l'acier dit « long » dans le tableau des poids d'ouvrant admissibles | Mise en œuvre Système 76 Advanced, registre 2.3.3, p. 5 : « court (55 mm) » et « long (5 mm) » — la seconde valeur ne peut pas être une longueur en regard de la première | profine — c'est la ligne qui fait passer une ferrure de 80 à 100 kg admissibles |
| VER-27 | Périmètre d'emploi du profilé acier 9708 | Mise en œuvre 9708, profine France, juillet 2017 : la note ne donne ni la longueur ni la section du profilé, ne nomme aucun système — ni 70, ni 76 Advanced — et ne fixe aucune flèche maximale rattrapable | profine — savoir sur quelles gammes et jusqu'à quelle déformation le procédé s'applique |
| VER-28 | Système profilé réel des gammes PERFORM70 et HYBRIDE70 | Aucun document PROFERM ne nomme le système 70 ni ne cite une référence en 6xxx. Les posters Gamme 70, le DTD 6/16-2335 et, depuis le 18/09/2026, le **manuel de fabrication complet du système 70 Plateforme** décrivent le système KÖMMERLING e.VOLUTION, sans lien écrit avec PROFERM | Bureau d'études — **c'est devenu la question la plus rentable du registre** : elle décide de l'exploitation de 371 pages de cotes, de renforts et d'abaques déjà versées dans `raw/` |
| VER-29 | Le battement 76453 du renfort inox VSF01 | DTD n° DBV-25-6/16-2334_V5, § 2.2.3.4 : « le renfort inox VSF01 peut être vissé à l'extérieur du battement 76453 ». Cette référence n'existe ni au cahier technique PERFORM76 ni au manuel de mise en œuvre profine, qui donnent 76471, 76472 et 76473 | profine — coquille probable pour 76473, mais une référence de battement ne se devine pas |
| VER-30 | Type d'ouverture de la ligne « 2 vantaux, 2,15 × 1,60 m » | DTD 6/16-2335 : « oscillo battante 2 vantaux ». DTA et DTD 6/16-2334 : « 2 vantaux OF », c'est-à-dire ouvrant à la française. Même cote, même position dans le tableau, libellé différent | CSTB ou profine — savoir si les deux systèmes diffèrent réellement sur cette configuration |
| VER-31 | Références, cotes et limites de la transformation OF en OB sur ROTO NX | Transformation OF en OB, PROFERM, réf. PRO-PVC-OFOB-01 rév. A : la têtière, le compas OB et la gâche OB sont dits « fournis », sans référence ni cote de perçage, et aucune limite dimensionnelle n'est rappelée pour l'ouvrant transformé | Service technique — un ouvrant dimensionné en OF peut sortir de son abaque une fois transformé en OB |
| VER-32 | Référence de commande et périmètre du report de charge NT Designo II | Bras de report de charge ROTO NX : la notice donne le montage et le réglage, jamais la référence de commande, ni si le procédé vaut pour les coulissants à pivot. **Le seuil, lui, est désormais connu** : les instructions de montage Roto NX KSR (p. 28 à 30) plafonnent le côté paumelles Designo à 100 kg sans report de charge, et à 150 kg avec | ROTO ou service technique — sans seuil, impossible de le prévoir au chiffrage |
| VER-33 | La troisième capacité du contrôle d'accès 4 en 1 | Notice simplifiée Eneo CC, p. 7 : le tableau des données techniques porte « 100 empreintes, 150 codes numériques, 200 » — la ligne ne dit pas ce que compte ce 200 | ROTO — probablement une capacité de badges ou de codes supplémentaires, mais rien ne le dit |
| VER-34 | Côté paumelles employé par PROFERM sur la Roto NX | Instructions de montage Roto NX KSR : le manuel couvre deux côtés paumelles, **P** et **Designo II**, dont les champs d'application diffèrent nettement — 1 600 mm de largeur maxi et 150 kg côté P, 1 400 mm et 100 kg côté Designo sans report de charge. Aucune source PROFERM ne dit lequel équipe les gammes | Service technique — le choix change les dimensions et le poids de vantail réalisables |

# Rattachements non sourcés

| ID | Déduction | Pourquoi elle est plausible | Pourquoi elle n'est pas sûre |
| --- | --- | --- | --- |
| VER-11 | Sécustik® est une marque du groupe ROTO | Les poignées Sécustik équipent les quatre gammes, et le RC2 repose sur « une poignée verrouillable Sécustik » à côté du Label ROTO Performance | **Aucune source ne l'écrit.** Le catalogue ne rattache jamais Sécustik à ROTO |
| VER-12 | ~~Les références 2xxx, 4xxx, 6xxx et 8xxx seraient des profils KÖMMERLING, les 76xxx propres à PROFERM~~ | — | **TRANCHÉE, et l'hypothèse était fausse dans les deux sens** : les posters profine recensent *toutes* les références du cahier PERFORM76, quelle que soit leur numérotation. Rien n'appartient en propre à PROFERM. Voir [Posters Système 76 Advanced](/sources/posters-systeme-76-advanced.md) |
| VER-13 | Les ouvrants 76272 et 76279 sont destinés aux portes-fenêtres ou aux grandes dimensions | Renfort tubulaire de section carrée, assemblages nettement plus larges | Le cahier ne donne aucun usage pour les deux paires d'ouvrants |
| VER-14 | La gamme TEXTURAL utilise des profils KÖMMERLING | Base PVC sertie d'aluminium, comme l'HYBRIDE qui est explicitement KÖMMERLING | Le catalogue ne nomme pas le fournisseur de profilé pour TEXTURAL |
| VER-15 | ALUPLAST fournirait le profilé du coulissant INNOSLIDE | ALUPLAST est crédité des photos du dépliant INNOSLIDE de janvier 2024, aux côtés de PROFERM et ROTO, et fabrique des profilés PVC | Un crédit photo n'est pas une preuve de fourniture. Le catalogue général rattache toute la gamme PVC au GREENLINE® de KÖMMERLING |
| VER-16 | LAKAL fournirait les volets roulants | LAKAL est crédité des photos du dépliant général de juin 2023 et fabrique des volets roulants. Aucun fournisseur de volet roulant n'est nommé dans le wiki, seul le motoriste [SOMFY](/fournisseurs/somfy.md) l'est | Crédit photo uniquement. Le catalogue ne nomme aucun fabricant de coffre ni de tablier |
| VER-17 | DEVGLASS fournirait le vitrage | DEVGLASS est crédité des photos du dépliant général de juin 2023 et transforme du verre. Le fournisseur de vitrage n'est nommé nulle part, alors que le catalogue détaille dix compositions | Crédit photo uniquement. Le « SGC ULTRA ONE » du catalogue suggère par ailleurs un autre nom |
| VER-20 | La justification de la substitution de quincaillerie ROTO à FERCO | Le DTA nomme FERCO comme quincaillerie du procédé et admet d'autres quincailleries « sur justifications » (p. 9). PROFERM emploie ROTO et revendique le Label ROTO Performance | La justification correspondante ne figure dans aucun document du wiki. **Le DTD n° DBV-25-6/16-2334_V5, révisé au GS6 du 13 mars 2025, nomme encore FERCO en deux endroits** avec la même réserve « sur justifications » : la pièce la plus récente de la chaîne réglementaire n'a pas enregistré le changement. Sans elle, l'emploi de ROTO n'est pas tracé vis-à-vis de l'Avis Technique |
| VER-26 | La gamme PERFORM76 est la variante **à joint central** du système 76 Advanced | Les sept dormants, les cinq ouvrants, les meneaux, les battements, les tapées et les appuis du cahier technique PERFORM76 figurent tous dans le manuel de mise en œuvre du 76 Advanced à joint central, aux mêmes références | **Aucun document PROFERM n'emploie les mots « joint central » ni « joint de frappe ».** Le système existe dans les deux variantes, avec des cotes de débit et des parcloses différentes : appliquer les cotes de débit du joint central à une fabrication à joint de frappe donnerait des pièces fausses |

**VER-12 est close, et sa réponse a une conséquence pratique immédiate** : toutes les références
du cahier PERFORM76 appartenant au système profine, **tout s'approvisionne chez le fournisseur**
et la nomenclature du cahier est directement utilisable pour commander. Voir
[KÖMMERLING](/fournisseurs/kommerling.md).

**VER-28 a changé de nature le 18/09/2026.** Tant qu'elle n'était documentée que par deux
planches A0 et un dossier réglementaire, elle relevait de la curiosité. Le manuel de mise en
œuvre du système 70 Plateforme étant désormais dans `raw/`, une réponse positive rendrait
exploitables les cotes de débit, les renforts, les abaques et la statique d'une gamme que le wiki
ne documente aujourd'hui que par son épaisseur et son Uw. Voir
[Profilés et renforts du système 70](/profiles/systeme-70-profiles-et-renforts.md).

**VER-20 est la nouvelle entrée sensible** : sans la justification de substitution de ROTO à
FERCO, l'emploi de la quincaillerie ROTO n'est pas tracé vis-à-vis de l'Avis Technique. C'est le
seul point du registre qui touche à la validité réglementaire d'une menuiserie posée. Voir
[DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md).

**VER-16 et VER-17 relèvent du même angle mort** : le wiki ne connaît aucun fournisseur de
vitrage ni de volet roulant, alors que ce sont deux postes entiers du catalogue. Les crédits
photos sont le seul indice disponible, et ils ne prouvent rien. Trois noms à faire confirmer d'un
coup au service achats — ALUPLAST, LAKAL, DEVGLASS.

**VER-15 touche au même sujet, côté coulissant.** Si le profilé de l'INNOSLIDE venait
d'ALUPLAST et non de KÖMMERLING, la page [INNOSLIDE](/gammes/innoslide.md) et le rattachement du
coulissant à la gamme [PERFORM](/gammes/perform.md) — présentée comme intégralement KÖMMERLING —
seraient à revoir. La question se règle en une phrase auprès du service achats.

# Citations

[1] Catalogue menuiseries PROFERM, édition janvier 2026 — `raw/catalogue-general-2026-01.pdf`,
p. 6, 8, 10, 12, 20 à 23, 27, 28 et 32
[2] Cahier technique PERFORM76, version 02/09/2026 CC03 —
`raw/cahier-technique-perform76-2026-09-02-cc03.pdf`
[3] Brochure Nouveautés PERFORM+ et HYBRIDE+, édition mai 2023 —
`raw/brochure-perform-plus-hybride-plus-2023-05.pdf`

# Voir aussi

- [Incohérences internes](/anomalies/incoherences-internes.md)
- [Contradictions entre sources](/anomalies/contradictions-entre-sources.md)
- [PERFORM+](/gammes/perform-plus.md)
- [HYBRIDE+](/gammes/hybride-plus.md)
- [KÖMMERLING](/fournisseurs/kommerling.md)
- [ROTO](/fournisseurs/roto.md)
