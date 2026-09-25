---
type: Anomalie
title: Contradictions entre sources
description: Registre des points sur lesquels deux documents PROFERM affirment des choses différentes, avec la valeur à retenir en attendant l'arbitrage.
tags: [anomalie, contradiction, garantie, a-corriger]
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
| CTR-03 | Garantie structure | Catalogue général, p. 35 : 15 ans sur la structure de la fenêtre | Brochure PERFORM+/HYBRIDE+, p. 3 **et brochure LUMINE65, p. 5** : 20 ans sur la structure | **aucune** — la valeur ne suit ni la date ni le produit | Engagement contractuel erroné de 5 ans |
| CTR-04 | Garantie volet roulant | Catalogue général, p. 35 **et brochure LUMINE65** : 5 ans | **Six documents** : brochure PERFORM+/HYBRIDE+, les trois dépliants de juin 2023, dépliant INNOSLIDE, dépliant LUMÉAL 2026 : 7 ans | **aucune** — six sources contre deux, mais rien ne les sépare | Engagement contractuel erroné de 2 ans |
| CTR-05 | Garantie laquage | Catalogue général, p. 35 : 25 ans LUMINE65, 10 ans standards, 7 ans hors standards | Brochure PERFORM+/HYBRIDE+, p. 3 : 7 ans forfaitaires | **La grille du catalogue** hors gammes + | Garantie sous-annoncée ou sur-annoncée selon la gamme |
| CTR-06 | Épaisseur du profilé PVC de l'HYBRIDE | Brochure HYBRIDE, p. 2 : 72 mm, gamme unique | Catalogue général, p. 10 : 70 à 76 mm, deux déclinaisons | **70 à 76 mm**, source la plus récente | Réponse fausse sur l'épaisseur d'une HYBRIDE livrée avant 2026 |
| CTR-07 | Uw de l'HYBRIDE | Brochure HYBRIDE, p. 2 : jusqu'à 1,2 W/m²K | Catalogue général, p. 10 et 11 : jusqu'à 0,8 W/m²K | **0,8 W/m²K**, source la plus récente | Étude thermique fausse dans les deux sens, écart de 0,4 W/m²K |
| CTR-08 | Classement AEV du LUMÉAL | Dépliant LUMÉAL, p. 1 : A\*4 / E\*7A / V\*B3, sur 2 vantaux H 2,5 × L 3 m | Catalogue général, p. 17 : A\*4 / E\*6A / V\*B2 pour tous les coulissants | **A\*4 / E\*7A / V\*B3**, source la plus récente et seule à donner la dimension d'essai | LUMÉAL disqualifié à tort sur chantier exposé |
| CTR-09 | Garantie de la ferrure Technal | Dépliants LUMÉAL de **juin 2023 et avril 2026** : ferrure Technal 10 ans, autre ferrure 2 ans | Catalogue général, p. 35 : ferrure ROTO 10 ans, « autre ferrure » 2 ans | **10 ans pour Technal**, valeur constante sur trois ans | Garantie sous-annoncée de 8 ans sur toute la gamme aluminium, depuis au moins 2023 |
| CTR-10 | Coloris du LUMÉAL | Dépliant LUMÉAL, p. 2 : 3 standards + 7 à prix préférentiel, pas de bicoloration, pas de Chêne doré | Catalogue général, p. 18 : Blanc 9016 brillant, Chêne doré 2 faces, laquage toutes teintes RAL | **La liste du dépliant**, source la plus récente | Coloris vendu puis indisponible, ou inversement |
| CTR-11 | Classement AEV du coulissant LUMINE65 | Brochure LUMINE65, p. 3 : A\*4 / E\*6A / V\*A3, sur 2 vantaux | Catalogue général, p. 17 : A\*4 / E\*6A / V\*B2 pour tous les coulissants | **A\*4 / E\*6A / V\*A3**, seule source qui essaie ce produit précis | Classement produit faux en appel d'offres |
| CTR-12 | Nom de la poignée encastrée de coulissant | Brochure LUMINE65, p. 3 : **MLINI** | Catalogue général, p. 17 : **DEHLI** | **Aucune** — même fonction, même position dans la liste | Référence introuvable à la commande |
| CTR-13 | Uw de l'HYBRIDE en juin 2023 | Dépliant HYBRIDE, p. 2 : 1,2 W/m²K | Dépliant général **du même mois**, p. 4 : 1,3 W/m²K | **1,2 W/m²K**, valeur du document produit | Étude thermique fausse sur de l'existant posé avant 2026 |
| CTR-14 | Liste des vitrages décoratifs | Nuancier vitrages, **sans date** : 7 verres dont un **Imprimé 200** inconnu ailleurs | Catalogue général, p. 26 : 9 verres dont **Listral** et **Mimosa**, absents du nuancier | **Largement levée** : le catalogue portes, p. 150, révèle que l'imprimé 200 est le verre de face intérieure des panneaux classiques, pas un décoratif au choix | Reste le Listral et le Mimosa vendables sans échantillon montrable |
| CTR-15 | Collections de portes d'entrée | Catalogue portes, mars 2024 : **six** collections — Authentique, Contemporain, Graphite, Lumière, Classique, Éléments | Catalogue général, janvier 2026 : **deux** — Collection Authentique et Sélection Hexa, cette dernière absente du catalogue portes | **Aucune** — deux découpages incompatibles de la même offre | Modèle proposé au client puis introuvable à la commande |
| CTR-16 | Garantie du panneau de porte | Catalogue portes, mars 2024, p. 156 : 7 ans, plaxé 5 ans | Catalogue général, janvier 2026, p. 35 : 10 ans, plaxé 7 ans | **10 ans**, source la plus récente : l'amélioration est postérieure à mars 2024 | Garantie sous-annoncée de 3 ans sur un poste coûteux |
| CTR-17 | Épaisseur de vitrage maximale du système 76 | Mise en œuvre Système 76 Advanced, profine, registre 2.1.1, p. 1 : « mise en oeuvre de différentes épaisseurs de vitrage ou panneau de remplissage **de 16 à 48 mm** » | DTA n° 6/16-2334_V5, p. 9 : vitrage jusqu'à **50 mm** ; cahier technique PERFORM76, p. 5 : parcloses d'ouvrant jusqu'à **50 mm** | **50 mm**, valeur réglementaire du DTA, qui est la pièce opposable. Le manuel de fabrication donne par ailleurs 36 à 50 mm pour la variante AluClip Zero (registre 2.6.5) : la borne de 48 mm n'est donc pas une limite du système entier | Un vitrage de 50 mm refusé à tort en atelier, ou accepté sans vérifier la parclose |
| CTR-18 | Champs d'application de la ferrure Roto NX, côté paumelles P, oscillo-battant rectangulaire | Instructions de montage Roto NX KSR, novembre 2022, p. 21 à 27 : HFF mini 290 mm, CDR 1 N jusqu'à 1 400 mm de LFF et 2 600 mm de HFF, CDR 2 jusqu'à 2 400 mm de HFF | Catalogue Roto NX, juin 2023, p. 35 : HFF mini 280 mm, CDR 1 N jusqu'à 1 600 mm de LFF et 2 800 mm de HFF, CDR 2 jusqu'à 2 800 mm de HFF | **Les bornes les plus basses des deux documents**, en attendant l'arbitrage : les deux sont des documents ROTO du même produit, à sept mois d'écart, et aucune règle du wiki ne les départage | Vantail accepté en commande hors du champ d'application réel de la ferrure, sur quatre bornes dont trois en CDR |
| CTR-19 | Épaisseur de vitrage de trois parcloses PERFORM76 | Cahier technique PERFORM76, p. 5 (PDF 8) : 76508 à 48 mm, 2454 à 31 mm, 2433 à 33 mm | DTD n° DBV-25-6/16-2334_V5, p. 18 : 76508 à 46 mm, 2454 à 32 mm, 2433 à 34 mm | **Les valeurs du cahier technique** — c'est le document produit, celui qui fixe la fabrication PERFORM76 | Parclose commandée à l'épaisseur du DTD, en écart de 1 à 2 mm avec le vitrage réellement posé |
| CTR-20 | Largeur totale de cinq dormants larges du système 70 | Mise en œuvre Système 70 Plateforme, registre 2.1.2 : 6108 à 105 mm, 6109 à 125, 6110 à 145, 6111 à 165, 6158 à 210 | DTD n° DBV-24-6/16-2335_V5, p. 15 : 6108 à 95 mm, 6109 à 115, 6110 à 135, 6111 à 155, 6158 à 200 | **Les valeurs du classeur de fabrication**, qui sert déjà de référence aux cotes de débit ; l'écart constant de 10 mm sur les cinq références suggère une convention de mesure différente, non énoncée par l'un ou l'autre document | Élargisseur ou pièce d'appui commandé 10 mm trop court ou trop long sur les cinq dormants larges |

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
