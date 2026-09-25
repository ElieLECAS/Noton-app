---
type: Anomalie
title: Informations à vérifier
description: Registre des affirmations périmées, des données manquantes et des rattachements non sourcés relevés dans la documentation PROFERM.
tags: [anomalie, a-verifier, perime, donnee-manquante]
status: stable
sources:
  - resource: raw/catalogue-portes-entree-2024-03.pdf
    id: catalogue-portes-entree-2024-03
    title: Catalogue portes d'entrée PROFERM, édition mars 2024
    last_modified: 2024-03-31
  - resource: raw/depliant-lumeal-2023-06.pdf
    id: depliant-lumeal-2023-06
    title: Dépliant LUMÉAL, édition juin 2023
    last_modified: 2023-06-30
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
  - resource: raw/depliant-innoslide-2024-01-a4-web.pdf
    id: depliant-innoslide-2024-01-a4-web
    title: Dépliant INNOSLIDE, édition janvier 2024, version A4 web
    last_modified: 2024-01-31
  - resource: raw/dta-trocal-76-advanced-6-16-2334-v5.pdf
    id: dta-6-16-2334-v5
    title: DTA n° 6/16-2334_V5, procédé TROCAL 76 ADVANCED
    last_modified: 2025-06-19
  - resource: raw/depliant-general-2023-06.pdf
    id: depliant-general-2023-06
    title: Dépliant général PROFERM, édition juin 2023
    last_modified: 2023-06-30
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
    last_modified: 2023-12-14
  - resource: raw/profine-mise-en-oeuvre-9708-montants-cintres-2017-07.pdf
    id: profine-9708-montants-cintres
    title: Mise en œuvre 9708, profine France, juillet 2017
    last_modified: 2017-07-06
  - resource: raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf
    id: dtd-6-16-2334-v5
    title: DTD n° DBV-25-6/16-2334_V5, système 76 Advanced
    last_modified: 2025-06-19
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    id: dtd-6-16-2335-v5
    title: DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION
    last_modified: 2025-04-15
  - resource: raw/roto-nx-transformation-of-en-ob-2026-03.pdf
    id: proferm-transformation-of-ob-roto-nx
    title: Transformation OF en OB gamme ROTO NX, PROFERM, réf. PRO-PVC-OFOB-01
    last_modified: 2026-03-19
  - resource: raw/roto-nx-bras-report-de-charge.pdf
    id: roto-nx-report-de-charge
    title: Bras de report de charge ROTO NX
  - resource: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf
    id: proferm-eneo-cc-notice-2022
    title: Roto Safe E Eneo CC, notice simplifiée PROFERM, version 2, 2022
    last_modified: 2022-12-31
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
  - resource: raw/nuancier-vitrages-decoratifs.pdf
    id: nuancier-vitrages-decoratifs
    title: Nuancier des vitrages décoratifs PROFERM
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
| VER-02 | Brochure PERFORM+/HYBRIDE+ | tout le document | Brochure de mai 2023 ; les gammes PERFORM+ et HYBRIDE+ **n'apparaissent nulle part** dans le catalogue général de janvier 2026 | Établir si les gammes + sont toujours commercialisées, renommées, ou abandonnées. Complément du 25/09/2026 : la brochure LUMINE65 (février 2025) et la brochure HYBRIDE (mars 2025) ne citent pas non plus les gammes + . Au catalogue général 2026, la poignée TOULON décalée n'est donnée que pour la LUMINE55 et l'ouvrant caché de fenêtre que pour la LUMINE65, deux versions aluminium |
| VER-18 | Nuancier stores | tout le document | Nuancier de **2020**, source la plus ancienne du wiki ; les stores intégrés n'apparaissent dans **aucune** autre source, pas même le catalogue général qui traite pourtant les volets roulants | Établir si les stores intégrés sont toujours commercialisés, et si la condition de vitrage **4+16+4 clair** vaut encore alors que le vitrage de série est passé à 6/18/4 de 28 mm **Complément du 25/09/2026** (onze pages relues en image) : aucune page n'imprime de date ni d'édition ; le millésime 2020 ne vient que du nom du fichier `nuancier-stores-2020.pdf`. La condition de vitrage est écrite « a) Vitrage 4+16+4 clair = aucune restriction d'usage. b) Autres vitrages : nous consulter » (p. 11) |
| VER-19 | Nuancier vitrages décoratifs | tout le document | **Aucune date d'édition, aucun code de version** — seul document du wiki dans ce cas | Dater le document, pour pouvoir trancher `CTR-14` face au catalogue **Relu le 25/09/2026** sur le rendu des cinq pages : aucune date, aucun code de version, aucune mention d'édition ; seules les coordonnées PROFERM de la p. 5 |
| VER-56 | Catalogue portes d'entrée | 160 | Les mentions légales de la quatrième de couverture portent « Édition juin 2023 » (relu en image le 25/09/2026) ; le wiki date ce catalogue de **mars 2024** (nom du fichier `catalogue-portes-entree-2024-03.pdf`, `last_modified: 2024-03-31`), sans qu'aucune page lue par les quatre tranches porte cette date | Dater l'édition : si c'est juin 2023, corriger `last_modified` sur toutes les pages qui citent le catalogue et relire **CTR-16**, qui compare ce catalogue au dépliant général de juin 2023 et au catalogue général 2026 |
| VER-57 | Mise en œuvre Système 76 Advanced | registre 2.4.2, p. 2 et 14 imprimées (PDF p. 128 et 140, version janvier 2016) | L'encadré « Alternative » dit : « Pour des profondeurs de perçages supérieures à 50 mm les trous oblongs peuvent être remplacés par 3 trous de Ø 6 ». Sur les dormants **76172** (p. 2) et **76102** (p. 14), le drainage par le bas est légendé « 5x25 mm ou 3 x Ø6 » alors que la cote portée le long du trou vertical est de **46 mm** ; sur les autres profilés, la mention « ou 3 x Ø6 » n'apparaît qu'à partir de 58 mm (76173 : 58 ; 76206 : 63 ; 76303 : 70 ; 76283 : 71 ; 76207 : 88), et elle est absente à 43 et 44 mm. Relu en image le 25/09/2026 | Demander à profine si la cote de 46 mm mesure la profondeur de perçage ou une autre longueur, et si les trois trous de Ø 6 sont admis sur les 76172 et 76102 |
| VER-58 | Mise en œuvre Système 76 Advanced | registre 2.4.3, p. 15, 18 et 19 imprimées (PDF p. 162, 165 et 166) | PF avec ouvrant 76281 en combinaison avec ouvrant 76272. Variante 1 (p. 15, mars 2021) : « Débit ouvrant 76272: DHT - 2X + 80 mm. » Variante 2 (p. 18, janvier 2023) : « Débiter le montant 76272, coupe 90° (DEO) : DHT - 2X - **80** fig.1 » ; même variante (p. 19) : « coupe 45° fig. 4: DEO = DHT - 2X + 80 + soudure (5 ou 6 mm) fig. 1 ». La fig. 1 dessine la longueur « DEO coupe à 90° » plus courte que « DEO coupes à 45° », sans coter l'écart. Relu en image le 25/09/2026 | Demander à profine si le débit à 90° est bien DHT − 2X − 80 (écart de 160 mm plus la soudure avec la coupe à 45°) ou DHT − 2X + 80, et quelle soudure (5 ou 6 mm) retenir |
| VER-59 | Mise en œuvre Système 76 Advanced | registre 2.6.2, p. 55 imprimée (PDF p. 359, version décembre 2016) | Rubrique « Joint » de la mise en œuvre AluClip : « Le joint de frappe dormant **G069** se positionne de façon continue dans la périphérie du cadre… », relu en image le 25/09/2026. Le G069 n'apparaît sur aucune autre planche du manuel, du poster ni du DTA ; le système d'étanchéité AluClip du même registre (p. 8, PDF p. 312) et le poster des profilés principaux donnent le **G161** comme joint de frappe dormant en variante AluClip | profine — dire si G069 est une ancienne référence du G161, une coquille, ou un autre joint |

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
| VER-22 | Distinction entre les cas « a » et « b usiné » des cotes de débit de seuil aluminium | Mise en œuvre Système 76 Advanced, registre 2.3.1, p. 8 : l'en-tête à trois colonnes n'explicite pas ce qui est usiné dans chaque cas, alors que le débit du renfort vertical change de 9 mm entre les deux. Relu en image le 25/09/2026 (PDF p. 88) : la coupe « Seuil alu A076 / A077 » porte deux repères ③a et ③b, ③b plus court que ③a, sans légende d'usinage | profine — l'écart porte sur un renfort déjà coupé |
| VER-23 | Épaisseurs d'acier des trois renforts du meneau 76373, et inertie du V324 sur la traverse 76303 | Mise en œuvre Système 76 Advanced, registre 2.1.2, p. 17 : le cartouche du 76373 porte les inerties mais les étiquettes de référence du 76372 voisin. La légende du dessin donne V323.Z, V322 et V324. Relu en image le 25/09/2026 (PDF p. 33 et 34) : le cartouche du 76373 porte bien ses propres références V323.Z (1,5 mm), V322 (2,5 mm) et V324 — l'association aux étiquettes du 76372 venait de la couche texte. Le V324 y est désigné « Capot Alu » mais dessiné en renfort de 2,5 mm avec IG 17,1 et IW 13,3 cm⁴ ; sur la planche du 76303, il est « Renfort 2,5 mm » avec les valeurs E et S du capot A070 à la place des inerties — voir `INC-30` | profine — à relire sur le document avant tout calcul statique |
| VER-24 | Orientation de montage du renfort V317 | Mise en œuvre Système 76 Advanced : IW 4,8 / IG 2,3 sur le battement 76472 (registre 2.1.2, p. 16), IW 2,3 / IG 4,8 sur la réhausse 76705 (registre 2.1.3, p. 4, PDF p. 42, relu en image le 25/09/2026 : le V317 y est dessiné couché, 40 × 25 × 2 mm, avec « IG = 4,8 cm⁴ », « IW = 2,3 cm⁴ » — le dessin de cette planche n'est pas tourné). La permutation est cohérente avec un montage tourné de 90°, le manuel ne l'écrit pas. Les deux posters 2022 dessinent le V317 couché (40 × 25 mm, IG 2,3 / IW 4,8, planche des profilés principaux) et debout (24 × 40 mm, IG 4,8 / IW 2,3, planche des profilés complémentaires), ce qui va dans le sens d'un montage tourné | profine — confirmer que c'est bien l'orientation qui change et non une coquille |
| VER-25 | Longueur de l'acier dit « long » dans le tableau des poids d'ouvrant admissibles | Mise en œuvre Système 76 Advanced, registre 2.3.3, p. 5 : « court (55 mm) » et « long (5 mm) » — la seconde valeur ne peut pas être une longueur en regard de la première (PDF p. 102, relu en image le 25/09/2026 : « long (5 mm) » confirmé ; les six rapports d'essais sont numérotés 12-002529-PR01 à PR06 PB-K20-09-de-01) | profine — c'est la ligne qui fait passer une ferrure de 80 à 100 kg admissibles |
| VER-27 | ~~Périmètre d'emploi du profilé acier 9708~~ (**Partiellement résolu**) | Fiche mise en œuvre 9708, profine France : la longueur (2 500 mm), la flèche initiale (40 mm au centre) et la section en U cotée 25 × 9 mm (rainure 6 mm) figurent explicitement sur le dessin technique `X 1:1` de la planche. Seul le rattachement formel à un système particulier (70 ou 76) reste non précisé (procédé universel pour portes PVC sans traverse intermédiaire) | profine |
| VER-28 | Système profilé réel des gammes PERFORM70 et HYBRIDE70 | Aucun document PROFERM ne nomme le système 70 ni ne cite une référence en 6xxx. Les posters Gamme 70, le DTD 6/16-2335 et, depuis le 18/09/2026, le **manuel de fabrication complet du système 70 Plateforme** décrivent le système KÖMMERLING e.VOLUTION, sans lien écrit avec PROFERM | Bureau d'études — **c'est devenu la question la plus rentable du registre** : elle décide de l'exploitation de 371 pages de cotes, de renforts et d'abaques déjà versées dans `raw/` |
| VER-29 | Le battement 76453 du renfort inox VSF01 | DTD n° DBV-25-6/16-2334_V5, § 2.2.3.4 : « le renfort inox VSF01 peut être vissé à l'extérieur du battement 76453 ». Cette référence n'existe ni au cahier technique PERFORM76 ni au manuel de mise en œuvre profine, qui donnent 76471, 76472 et 76473. Le texte est le même au DTA (p. 9) et au DTD (p. 6) ; dans le DTD, la coupe de principe du battement de 112 mm renforcé VSF01 (p. 43) ne légende pas le battement, la coupe voisine de même largeur le légende **76473**, et la planche des renforts donne le VSF01 « pour profilé 76473 » (p. 20) | profine — coquille probable pour 76473, mais une référence de battement ne se devine pas |
| VER-30 | Type d'ouverture de la ligne « 2 vantaux, 2,15 × 1,60 m » | DTD 6/16-2335 : « oscillo battante 2 vantaux ». DTA et DTD 6/16-2334 : « 2 vantaux OF », c'est-à-dire ouvrant à la française. Même cote, même position dans le tableau, libellé différent | CSTB ou profine — savoir si les deux systèmes diffèrent réellement sur cette configuration |
| VER-31 | Références, cotes et limites de la transformation OF en OB sur ROTO NX | Transformation OF en OB, PROFERM, réf. PRO-PVC-OFOB-01 rév. A : la têtière, le compas OB et la gâche OB sont dits « fournis », sans référence ni cote de perçage, et aucune limite dimensionnelle n'est rappelée pour l'ouvrant transformé | Service technique — un ouvrant dimensionné en OF peut sortir de son abaque une fois transformé en OB |
| VER-32 | Référence de commande et périmètre du report de charge NT Designo II | Bras de report de charge ROTO NX : la notice donne le montage et le réglage, jamais la référence de commande, ni si le procédé vaut pour les coulissants à pivot. **Le seuil, lui, est désormais connu** : les instructions de montage Roto NX KSR (p. 28 à 30) plafonnent le côté paumelles Designo à 100 kg sans report de charge, et à 150 kg avec | ROTO ou service technique — sans seuil, impossible de le prévoir au chiffrage |
| VER-33 | ~~La troisième capacité du contrôle d'accès 4 en 1~~ (**Résolu**) | Notice simplifiée Eneo CC, p. 7 : la ligne « 100 empreintes, 150 codes numériques, 200 Supports RFID eKeys » correspond à la capacité maximale de **200 supports RFID** (badges/cartes/porte-clés), tandis que les clés virtuelles eKeys sur smartphone via Bluetooth sont illimitées | **Résolu** — corroboré avec les spécifications SOREX SmartLock 4in1 (100 empreintes, 150 codes, 200 badges RFID, eKeys illimitées) |
| VER-34 | Côté paumelles employé par PROFERM sur la Roto NX | Instructions de montage Roto NX KSR : le manuel couvre deux côtés paumelles, **P** et **Designo II**, dont les champs d'application diffèrent nettement — 1 600 mm de largeur maxi et 150 kg côté P, 1 400 mm et 100 kg côté Designo sans report de charge. Aucune source PROFERM ne dit lequel équipe les gammes | Service technique — le choix change les dimensions et le poids de vantail réalisables |
| VER-36 | Famille d'emploi de sept parcloses PERFORM76 complémentaires | DTD n° DBV-25-6/16-2334_V5, p. 18 : sept parcloses (1511, 1512, 76513, 76531 à 76534) et un rehausseur (76570) suivent les séries d'ouvrant et de dormant sans étiquette « ouvrant » ni « dormant », avec un profil dessiné différent des deux séries. DTA n° 6/16-2334_V5, p. 19 : les mêmes sept parcloses et le 76570 « (Rehausseur de parclose) » forment la sixième ligne de la planche « Parcloses PVC », et la troisième ligne porte 6146 (40), 6148 (36), 6147 (32), 76512 (28), 76501 (24), 76523 (16), 76524 (30), toujours sans légende ouvrant ou dormant. Mise en œuvre Système 76 Advanced, registre 2.3.2 (PDF p. 89, relu en image le 25/09/2026) : les **76523** (16 mm) et **76524** (30 mm) sont dans le « Tableau de vitrage pour ouvrant » ; les 6146, 6147, 6148, 1511, 1512 et 76531 à 76534 ne figurent dans aucun tableau de vitrage du manuel | profine — savoir si elles montent sur l'ouvrant, le dormant, ou une troisième famille de profilé |
| VER-37 | Cotes de la traverse de meneau 6127 sur le DTD du système 70 | DTD n° DBV-24-6/16-2335_V5, p. 19 : la cote verticale du 6127 se lit 140 mm sur la planche rendue, contre une largeur non cotée par le classeur de fabrication pour ce même profilé | profine — confirmer la cote avant de l'inscrire dans le tableau des meneaux et traverses |
| VER-38 | Attribution des capots de réhabilitation aux dormants du système 70 | DTD n° DBV-24-6/16-2335_V5, p. 14 : cinq dormants rénovation (6102, 6105, 6107, 6155, 6156) partagent la note « capots pour dormants réhabilitation », sans dire lequel des capots 9C01.1 ou 9C02.1 chacun reçoit | profine — le capot dépend probablement de la largeur du dormant, comme pour 2501/2502, mais rien ne l'écrit |
| VER-39 | Affectation des seuils A075, A077 et A343 à un dormant du système 76 | DTA n° 6/16-2334_V5, p. 14 : le seuil A076 (76 mm) est déjà rattaché aux cinq dormants PERFORM76 par le cahier technique, mais A075 (76 mm, plus haut), A077 (123 mm) et A343 (135 mm) n'apparaissent que sur cette planche, sans dormant ni configuration nommés. Le poster des profilés principaux 2022 (p. 1) donne des sets d'assemblage sur A076 et A077 et sur A075 pour les dormants 76171, 76172, 76173, 76177, 76178, 76180, 76101 et 76102, sans nommer le A343 ; le DTA lui-même (p. 24) range le A343 avec les A076 et A077 pour l'assemblage du rejet d'eau (embouts M261, M163, M178), et le seuil A077 est dessiné sous le dormant 76180 dans la mise en place du capot sur seuil PMR (p. 33) — voir [Assemblages du système 76](/profiles/systeme-76-assemblages.md) | profine — savoir s'ils desservent un dormant large, une rénovation, ou une configuration hors PERFORM76 |
| VER-40 | Homonymie des références A469 à A473 entre le système 76 et le système 70 | DTA n° 6/16-2334_V5, p. 18 : A469 à A473 sont des **tapées aluminium** cotées 30 à 110 mm de haut, confirmé par la planche « Montage des tapées et appuis — Avec demi-capot en traverse basse » (p. 30), où les cinq sont dessinées en tapées contre le dormant. Le poster Gamme 70 KÖMMERLING nomme les mêmes cinq références « embouts d'extrémité de pièce d'appui », une fonction différente, sur une planche dont les cotes ne sont pas lisibles avec certitude | profine — savoir s'il s'agit d'une coïncidence de numérotation entre deux systèmes distincts ou d'une erreur de lecture du poster |
| VER-41 | Modèles de porte partageant une même référence (trois paires, et un troisième modèle sous la 3720) | Catalogue portes d'entrée, mars 2024, p. 54, 60, 62, 64 et 119-120 : les modèles ISAÏS et ISABELLE portent tous deux la référence 3720, MORGANE et NINON tous deux la référence 4060, THÉBÉ et AMALTHE tous deux la référence AMOEU07PROF, sans qu'aucune note ne les rapproche. Galerie des modèles du même catalogue (PDF p. 8, relue en image le 25/09/2026) : ISABELLE (page imprimée 53) et ISAIS (page imprimée 52, écrit sans tréma) y sont deux vignettes distinctes, de dessins différents ; MORGANE (60) et NINON (62) aussi. Pages des modèles relues en image le 25/09/2026 (tranche 41-80) : ISAÏS, écrit avec tréma sur sa page, « Réf. 3720 », insert inox 2 faces, rainurage et vitrage dépoli acide, présenté en Siena (p. 54) ; ISABELLE, « Réf. 3720 », insert inox 2 faces et rainurage, présentée en RAL 3700 (p. 55) : deux dessins différents, sans vitrage pour ISABELLE, sous la même référence ; le modèle à numéro 019-0, rainurage et vitrage dépoli acide, présenté en Chêne Doré, porte lui aussi « Réf. 3720 » (p. 60), soit trois dessins sous une référence. Pages relues en image le 25/09/2026 (tranche 81-120) : AMALTHE, « Réf. AMOEU07PROF », DIN Droite, œuvre sombre à traits dorés, présentée en Cuir noir intérieur (PDF p. 119) ; THÉBÉ, « Réf. AMOEU07PROF », DIN Droite, œuvre à bandes horizontales, présentée en RAL 7016 (PDF p. 120) : deux œuvres différentes sous la même référence ; la galerie les renvoie à deux pages distinctes (pages imprimées 117 et 118) | Service commercial — savoir s'il s'agit du même panneau vendu sous deux noms ou d'une erreur de référence |
| VER-42 | Pictogramme d'interdiction de la loupe « Pose en rénovation, version 1 » | Cahier technique PERFORM76, p. 8 (PDF 11) : une loupe montre une vis horizontale traversant le dormant rénovation et la compensation bois, marquée d'un pictogramme rouge d'interdiction, sans texte | Service technique PROFERM : quelle fixation ou quel détail est interdit |
| VER-43 | Profilé habillé par six capots aluminium du système 76 | Poster Système 76 Advanced, profilés principaux, 2022, p. 1 : chaque capot est dessiné à gauche du profilé qu'il habille, sauf **A073** (dessiné deux fois : entre le 76172 et le 76173, et à côté du 76201), **A506T** (sans cote, au-dessus du A073), **A043** (entre les ouvrants 76275 et 76271, à droite du A072 du 76271), **A046** et **A047** (à gauche des capots A069 et A070 des traverses 76301 et 76303). Aucun capot n'est dessiné à côté des ouvrants galbés 76275, 76276 et 76279. Complément du manuel de mise en œuvre, registre 2.1.1 Porte d'entrée (PDF p. 5 à 13) : les tableaux d'accessoires rattachent l'**A073** au dormant 76102, l'**A506T** (« profilé Alu protection recouvrement dormant ») aux dormants 76101 et 76102, l'**A046** (avec l'A069) au 76301 et l'**A047** (avec l'A070) au 76303, sans dire lequel des deux capots d'un meneau s'emploie dans quel cas Registre 2.1.2 de la fenêtre (PDF p. 20, 28, 29) : l'**A073** est « Capot Alu dormant ouverture extérieure ou élargisseur » dans la nomenclature du 76172 ; l'**A043** est donné avec l'A072 à l'ouvrant 76271 ; l'**A044** à l'ouvrant galbé 76279 comme au 76272 ; le 76283 et les galbés 76275 et 76276 n'ont aucun capot dans leur nomenclature Complément du 25/09/2026, registre 2.6.2 « AluClip, Profilés principaux et accessoires » (PDF p. 307) : l'**A043** est dessiné à gauche de l'ouvrant **76271**, l'A042 du 76281, l'A044 du 76272 ; les A073, A506T, A046, A047, A039, A318 et A055 ne sont pas redessinés dans ce registre, et le plan AluClip du 76172 avec l'ouvrant extérieur 76283 (PDF p. 318) associe l'**A073** au dormant 76172 et l'**A039** au 76283 | profine — table d'affectation capot → profilé |
| VER-44 | Set d'assemblage du dormant 76185 sur seuil aluminium | Poster Système 76 Advanced, profilés principaux, 2022, p. 1 : les cinq tableaux de sets et pièces sur seuils A076, A077 et A075 couvrent les dormants 76171, 76172, 76173, 76177, 76178, 76180, 76101, 76102 et les meneaux 76372, 76373, 76301, 76303 ; **le 76185 n'y figure pas**, alors que le seuil A076 est donné compatible avec les cinq dormants PERFORM76. Complément du manuel de mise en œuvre, registre 2.1.2 (PDF p. 25) : la planche du 76185 donne ses trois sets, **M609** (pour A076. A077, A343), **M610** (pour A076, A077, A343 grugé) et **M611** (pour A075 grugé) | profine ou bureau d'études — même question que VER-21 pour les battements |
| VER-45 | Famille d'emploi de sept parcloses du poster profine absentes du cahier PERFORM76 | Poster Système 76 Advanced, profilés complémentaires, 2022, p. 1 : la seconde ligne de parcloses prolonge les parcloses de dormant de la PERFORM76 par **2632 (26), 2630 (24), 2628 (22), 2626 (20), 2624 (18)** ; la première ligne porte **76512 et 76513 (28)** à côté de la 76526. Aucune ligne n'est légendée ouvrant ou dormant, et le cahier PERFORM76 écrit qu'aucune parclose de dormant ne descend sous 28 mm. DTA n° 6/16-2334_V5, p. 19 : la cinquième ligne porte 2638 (32), 2636 (30), 2634 (28), 2632 (26), 2630 (24), 2626 (20), 2624 (18), **sans la 2628**, et 76512 et 76513 (28) sont sur les lignes 3 et 6 Mise en œuvre Système 76 Advanced, registre 2.3.2 (PDF p. 89 et 96, relus en image le 25/09/2026) : les **76512 et 76513** (28 mm) sont dans le « Tableau de vitrage pour ouvrant », les **2624, 2626, 2628, 2630, 2632** (18 à 26 mm) dans le « Tableau de vitrage pour dormant et meneau », la 2628 comprise ; le manuel ne dit pas si ces parcloses sont proposées en PERFORM76 | profine — savoir si 2624 à 2632 sont des parcloses de dormant, et à quoi servent 76512 et 76513 |
| VER-46 | Sens des valeurs « E » et « S » des capots aluminium | Mise en œuvre Système 76 Advanced, registre 2.1.1 Porte d'entrée (PDF p. 5 à 13) : chaque capot, demi-capot et profilé de protection porte, dans sa case « Valeurs », deux cotes en mm, « E: » et « S: » (A030 : E 222, S 107 ; A506T : E 80,6, S 51,3 ; A041 : E 326,9, S 168). Aucune page du registre ne dit ce que mesurent E et S Complément du 25/09/2026 : le registre 2.6.2 AluClip (PDF p. 304 à 394), qui cote chaque capot par sa largeur de retour et sa hauteur, ne porte aucune valeur E ni S et ne les définit pas davantage | profine — savoir si E et S sont un développé, une surface vue ou autre, avant de s'en servir en chiffrage ou en débit |
| VER-47 | Famille des profilés 76301 et 76303 : meneau ou traverse d'ouvrant | Mise en œuvre Système 76 Advanced : « 76301 Meneau 84 mm », « 76303 Meneau 110 mm » sur les planches de la porte d'entrée (PDF p. 12 et 13), avec des sets d'assemblage sur seuil A076/A077 (M156, M157) et des patins d'étanchéité de seuil (J066, J067) ; « Meneau de 84 mm / Traverse » et « Meneau de 110 mm / Traverse », sous le titre « Meneau/Traverse et traverse d'ouvrant », au sommaire de la fenêtre (PDF p. 18). Le cahier technique PERFORM76 les appelle meneaux d'ouvrant, [Profilés principaux du système 76](/profiles/systeme-76-profiles-principaux.md) les range en traverses d'ouvrant. Sur les planches de la fenêtre (registre 2.1.2, PDF p. 34), les 76301 et 76303 sont titrés « Traverse d'ouvrant 84 mm » et « Traverse d'ouvrant 119 mm », sans set d'assemblage sur seuil ni patin dans leur nomenclature | profine — savoir si les 76301 et 76303 se montent en meneau de dormant (sur seuil) en porte d'entrée seulement, ou aussi en fenêtre |
| VER-48 | Emploi des profilés 76101 et 76102 hors cadre fixe | DTA n° 6/16-2334_V5, p. 6, remarque 1 du Groupe Spécialisé : « Les profilés réf. 76101 et 76102 doivent être exclusivement utilisés pour la réalisation de cadres fixes » — le DTA vise les fenêtres et portes-fenêtres. Mise en œuvre Système 76 Advanced, registre 2.1.1 Porte d'entrée (PDF p. 4 à 6) : « 76101 Dormant de 67 mm », « 76102 Dormant de 85 mm », dormants de la porte d'entrée, avec leurs pièces d'assemblage sur seuil. Les deux sont compatibles si la restriction ne vaut que pour les fenêtres sous DTA | profine ou bureau d'études — confirmer que l'emploi en dormant de porte d'entrée est hors du périmètre du DTA |
| VER-49 | Portée du laquage de l'HYBRIDE : quelles faces, quelle matière, quelles teintes | Catalogue général, p. 12, encadré « Intérieur et extérieur » : « Laquage toutes teintes possibles 2 faces identiques », puis « LAQUAGE POSSIBLE SI : - ouverture extérieure » et « Uniquement sur PVC. » Sur une HYBRIDE, l'extérieur est en aluminium et a son propre nuancier laqué : « 2 faces identiques » et « uniquement sur PVC » ne disent pas si le laquage couvre la seule face PVC intérieure, ou la face PVC et le capot aluminium dans la même teinte. P. 11, les avantages annoncent « toutes teintes RAL possibles » sans condition. Complément, brochure HYBRIDE de mars 2025, p. 3, même encadré « Intérieur et extérieur » : « Laquage toutes teintes possibles 2 faces identiques, uniquement sur menuiserie PVC en cas de cintrage et/ou ouverture extérieure. » La même phrase figure au dépliant HYBRIDE de juin 2023, p. 3. La brochure ajoute le cintrage comme cas, et écrit « menuiserie PVC » là où le catalogue écrit « PVC » : la phrase peut vouloir dire que, cintrée ou ouvrant à l'extérieur, la fenêtre est fabriquée entièrement en PVC et laquée 2 faces — aucun des deux documents ne l'écrit. Impact : un devis de menuiserie HYBRIDE laquée 2 faces peut être accepté pour une ouverture intérieure, ou refusé à tort ; une HYBRIDE cintrée laquée peut être chiffrée en aluminium extérieur. Voir [Coloris HYBRIDE](/coloris/coloris-hybride.md) **Complément du 25/09/2026** : le dépliant général de juin 2023, p. 4, relu en image, écrit sous « Intérieur et extérieur (2 faces identiques) » : « Laquage toutes teintes possibles si : ouverture extérieure, menuiserie cintrée », sans « uniquement sur menuiserie PVC ». | Service technique |
| VER-50 | Liste des « 9 coloris au prix du blanc » de la LUMINE65 | Brochure LUMINE65, février 2025, p. 5 : pastille « 9 coloris au prix du blanc » au-dessus de « 11 coloris 1 ou 2 faces », dont l'« Anodisé laqué argent contretypage (en option) ». Le nuancier ne marque pas les neuf teintes concernées : onze moins l'argent en option en laisse dix. La page [LUMINE](/gammes/lumine.md) numérotait dix teintes sous « 9 coloris au prix du blanc ». Catalogue général, p. 18 : aucune mention de prix. Brochure relue en image à 200 dpi le 25/09/2026 : aucune pastille du nuancier ne porte de marque qui distinguerait les neuf teintes. Voir [Coloris LUMINE](/coloris/coloris-lumine.md) | Service commercial |
| VER-51 | Configurations et formes réalisables du coulissant INNOSLIDE : plus de deux vantaux, formes cintrées | Dépliant INNOSLIDE, janvier 2024, version A4 web, p. 3 : un seul schéma, légendé « Une partie coulissante et une partie fixe » ; catalogue général, p. 8 : le même schéma. Aucune des deux sources n'écrit que d'autres configurations (deux vantaux coulissants, trois vantaux ou plus) ou une forme cintrée sont possibles ou impossibles. La page [INNOSLIDE](/gammes/innoslide.md) portait, sans source, « Configurations supérieures à 2 vantaux : le mécanisme Inowa est strictement limité à 1 coulissant + 1 fixe » et « Formes cintrées : la compression linéaire et le guidage sur rail interdisent tout cintrage » ; les deux phrases sont retirées le 25/09/2026, dépliant relu en image à 200 dpi | Service commercial, ROTO |
| VER-52 | Formes cintrées et hors d'équerre des coulissants aluminium | La page [Coulissants aluminium](/gammes/coulissants-aluminium.md) portait, sans source, « Cintrage et formes hors d'équerre : incompatibles avec les rails de roulement des chariots coulissants ». Le catalogue général, p. 16 à 18, et le dépliant LUMÉAL de juin 2023, relu en image à 200 dpi le 25/09/2026, n'en disent rien ; la phrase est retirée | Service technique, TECHNAL |
| VER-53 | Source de « Hydro CIRCAL » et de l'alliage « 6060 T6 » de l'aluminium bas carbone TECHNAL | La page [TECHNAL](/fournisseurs/technal.md) portait, sous le locator du dépliant LUMÉAL d'avril 2026, p. 2 : « 75 %, déchets post-consommation Hydro CIRCAL » et « Première qualité bâtiment (alliage 6060 T6) ». Les dépliants LUMÉAL de juin 2023 (p. 2) et d'avril 2026 (p. 2), relus en image le 25/09/2026, n'écrivent que « aluminium de première qualité fabriqué avec un minimum de 75 % d'aluminium recyclé en fin de vie (déchets post-consommation) » ; les deux mentions sont retirées du tableau. Elles peuvent venir d'un document TECHNAL non encore versé dans `raw/` | Service achats, TECHNAL |
| VER-55 | Référence de commande du modèle de porte LUC | Catalogue portes d'entrée, mars 2024, p. 28 : LUC est légendé « Insert inox 2 faces, Vitrage dépoli acide » sans ligne « Réf. », alors que NESTOR et ODILON, sur la même page, et tous les modèles à prénom des pages 23 à 29 portent une référence (7440, 7500…). Les modèles à numéro (706-0, 355-0, 550-2, 550-6) et à grille en applique n'en portent pas non plus, mais leur nom n'est pas un prénom. Voir [Collection Contemporain](/portes/collection-contemporain.md) | Service commercial |

# Rattachements non sourcés

| ID | Déduction | Pourquoi elle est plausible | Pourquoi elle n'est pas sûre |
| --- | --- | --- | --- |
| VER-11 | Sécustik® est une marque du groupe ROTO | Les poignées Sécustik équipent les quatre gammes, et le RC2 repose sur « une poignée verrouillable Sécustik » à côté du Label ROTO Performance | **Aucune source ne l'écrit.** Le catalogue ne rattache jamais Sécustik à ROTO |
| VER-12 | ~~Les références 2xxx, 4xxx, 6xxx et 8xxx seraient des profils KÖMMERLING, les 76xxx propres à PROFERM~~ | — | **TRANCHÉE, et l'hypothèse était fausse dans les deux sens** : les posters profine recensent *toutes* les références du cahier PERFORM76, quelle que soit leur numérotation. Rien n'appartient en propre à PROFERM. Voir [Posters Système 76 Advanced](/sources/posters-systeme-76-advanced.md) |
| VER-13 | Les ouvrants 76272 et 76279 sont destinés aux portes-fenêtres ou aux grandes dimensions | Renfort tubulaire de section carrée, assemblages nettement plus larges | Le cahier ne donne aucun usage pour les deux paires d'ouvrants |
| VER-14 | La gamme TEXTURAL utilise des profils KÖMMERLING | Base PVC sertie d'aluminium, comme l'HYBRIDE qui est explicitement KÖMMERLING | Le catalogue ne nomme pas le fournisseur de profilé dans la présentation de la TEXTURAL. **Complément du 25/09/2026** : la planche « Sécurité renforcée » de la TEXTURAL (catalogue général, p. 23) légende sa première photo « Profils KÖMMERLING®. Renfort selon abaques. Renfort total en option. », comme celles de l'HYBRIDE (p. 10) ; c'est un indice fort, pas une déclaration de fournisseur. Entrée laissée ouverte |
| VER-15 | ALUPLAST fournirait le profilé du coulissant INNOSLIDE | ALUPLAST est crédité des photos du dépliant INNOSLIDE de janvier 2024, aux côtés de PROFERM et ROTO, et fabrique des profilés PVC | Un crédit photo n'est pas une preuve de fourniture. Le catalogue général rattache toute la gamme PVC au GREENLINE® de KÖMMERLING. **Complément du 25/09/2026** : la marge de la p. 4 de la version A4 web, relue en image, écrit « Création : PROFERM - Crédits photos : PROFERM - ALUPLAST - ROTO » ; aucune page du dépliant ne nomme le fabricant du profilé ni un système (70 ou 76). Le rattachement de l'INNOSLIDE au système 76 (`systeme: 76` des pages INNOSLIDE) n'est écrit par aucune source non plus. Entrée laissée ouverte |
| VER-16 | LAKAL fournirait les volets roulants | LAKAL est crédité des photos du dépliant général de juin 2023 et fabrique des volets roulants. Aucun fournisseur de volet roulant n'est nommé dans le wiki, seul le motoriste [SOMFY](/fournisseurs/somfy.md) l'est | Crédit photo uniquement. Le catalogue ne nomme aucun fabricant de coffre ni de tablier |
| VER-17 | DEVGLASS fournirait le vitrage | DEVGLASS est crédité des photos du dépliant général de juin 2023 et transforme du verre. Le fournisseur de vitrage n'est nommé nulle part, alors que le catalogue détaille dix compositions | Crédit photo uniquement. Le « SGC ULTRA ONE » du catalogue suggère par ailleurs un autre nom |
| VER-20 | La justification de la substitution de quincaillerie ROTO à FERCO | Le DTA nomme FERCO comme quincaillerie du procédé et admet d'autres quincailleries « sur justifications » (p. 9) — relu le 25/09/2026 sur le rendu de la page : « Quincaillerie : FERCO », texte inchangé. PROFERM emploie ROTO et revendique le Label ROTO Performance | **Un premier élément trouvé le 19/09/2026** : les Directives générales profine, registre 1.3.4, listent ROTO comme quincaillerie approuvée pour la quasi-totalité des configurations de fenêtre et plusieurs portes, mais pas pour la porte d'entrée à un vantail seule, les seuils ni les ferme-portes — voir [ROTO](/fournisseurs/roto.md). Ce document ne cite aucun DTA et ne se présente pas explicitement comme la justification exigée. **Le DTD n° DBV-25-6/16-2334_V5, révisé au GS6 du 13 mars 2025, nomme encore FERCO en deux endroits** avec la même réserve « sur justifications » : la pièce la plus récente de la chaîne réglementaire n'a toujours pas enregistré le changement par son nom |
| VER-26 | La gamme PERFORM76 est la variante **à joint central** du système 76 Advanced | Les sept dormants, les cinq ouvrants, les meneaux, les battements, les tapées et les appuis du cahier technique PERFORM76 figurent tous dans le manuel de mise en œuvre du 76 Advanced à joint central, aux mêmes références | **Aucun document PROFERM n'emploie les mots « joint central » ni « joint de frappe ».** Le système existe dans les deux variantes, avec des cotes de débit et des parcloses différentes : appliquer les cotes de débit du joint central à une fabrication à joint de frappe donnerait des pièces fausses |
| VER-54 | Le vitrage « Listral » du catalogue général et le vitrage « Imprimé 200 » du nuancier des vitrages décoratifs seraient le même verre | Catalogue général, janvier 2026, p. 26 : la photo « Listral » montre un grain martelé serré ; nuancier des vitrages décoratifs, p. 4 : la photo « Imprimé 200 » montre un grain d'aspect voisin. Chacun des deux noms manque dans l'autre document (**CTR-14**) | Les deux photos sont deux prises de vue différentes, et aucun document ne met les deux noms en regard ; une ressemblance de photo n'est pas une identité de produit. À demander au service commercial ou au fournisseur de vitrage |

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
