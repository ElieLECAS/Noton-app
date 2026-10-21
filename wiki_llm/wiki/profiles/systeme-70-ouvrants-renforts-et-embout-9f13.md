---
type: Profilé
title: 'Renforts des ouvrants du système 70 Plateforme : Iz des plans 2008, IG et IW, embout 9F13'
description: Renforts admis par ouvrant du système 70 (V158, V258, V057, V059, V069, V003, V290, V154) avec leurs inerties IG et IW, renforts et Iz du classeur de 2008, et pièces dessinées avec les ouvrants dont l'embout 9F13.
tags: [systeme-70, gamme-70, e-volution, e-motion, e-xclusive, ouvrant, porte, capot, renfort, inertie]
systeme: 70
fournisseur: KÖMMERLING
usage: atelier
famille: ouvrants
status: draft
sources:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    id: profine-mise-en-oeuvre-systeme-70
    title: Mise en œuvre Système 70 Plateforme, profine, version septembre 2023
    last_modified: 2023-09-30
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    id: profine-plans-e-volution-2008
    title: Système e.VOLUTION, plan des profilés, système F 91, édition août 2008
    last_modified: 2008-08-31
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    id: dtd-6-16-2335-v5
    title: DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION
    last_modified: 2024-12-19
  - resource: raw/poster-kommerling-70-principaux-2025-03.pdf
    id: poster-kommerling-70-principaux
    title: Poster Kömmerling Gamme 70, profilés principaux, mars 2025
    last_modified: 2025-03-31
source_pages:
  - resource: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf
    pages: 5-6, 25-34
  - resource: raw/profine-plans-profiles-e-volution-2008-08.pdf
    pages: 9-17, 96-98, 102-108
  - resource: raw/dtd-6-16-2335-v5-e-volution.pdf
    pages: 16-18
  - resource: raw/poster-kommerling-70-principaux-2025-03.pdf
    pages: 1
generated:
  by: process:claude-code
  at: 2026-10-21T10:40:29Z
---

Cette page reprend, du manuel de mise en œuvre du système 70 Plateforme (profine, septembre 2023) et du classeur KÖMMERLING e.VOLUTION d'août 2008, les renforts des ouvrants et les pièces dessinées avec eux. Les cotes d'ensemble des ouvrants sont dans [Cotes des ouvrants, ouvrants de porte et capots du système 70 Plateforme, ouvrants du DTD](/profiles/systeme-70-ouvrants.md).

### Renforts des ouvrants des plans de 2008

Renforts écrits dans le tableau « Renfort | Iz » de chaque planche d'ouvrant. Quand trois
renforts sont listés, le premier est imprimé en gras. Une ligne par couple ouvrant et renfort ;
Iz en cm⁴ ; un tiret signifie qu'aucune inertie n'est imprimée [2 p. 9-17].

| Ouvrant | Renfort | Iz (cm⁴) | Imprimé en gras | Page PDF |
| --- | --- | --- | --- | --- |
| 6112 | V600 | 2,10 | oui | 9 |
| 6113 | V600 | 2,10 | oui | 9 |
| 6115 | V057 | 3,20 | oui | 10 |
| 6115 | V059 | 4,50 | non | 10 |
| 6115 | V069 | - | non | 10 |
| 6116 | V057 | 3,20 | oui | 10 |
| 6116 | V059 | 4,50 | non | 10 |
| 6116 | V069 | - | non | 10 |
| 0112 | V039 | 3,10 | oui | 11 |
| 0113 | V026 | 3,70 | oui | 11 |
| 2416 | V003 | 12,30 | oui | 12 |
| 0303 | V045, chambre haute | 2,70 | oui | 13 |
| 0303 | V031, chambre basse | 5,30 | oui | 13 |
| 2415 | V003 | 12,30 | oui | 14 |
| 2418 | V030 | 4,50 | oui | 15 |
| 6121 | V058 | 3,20 | oui | 16 |
| 6122 | V058 | 3,20 | oui | 16 |
| 6123 | V057 | 3,20 | oui | 17 |
| 6123 | V059 | 4,50 | non | 17 |
| 6123 | V069 | - | non | 17 |
| 6124 | V057 | 3,20 | oui | 17 |
| 6124 | V059 | 4,50 | non | 17 |
| 6124 | V069 | - | non | 17 |

(schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 9 à 17)

Ces renforts ne sont pas ceux du manuel de mise en œuvre de 2023 : le 6112 et le 6113 y reçoivent
le V158 ou le V258 (V600 en 2008), le 6121 et le 6122 aussi (V058 en 2008), et le V057 et le V059
y ont une IW de 3,8 et 4,4 cm⁴ sur les planches de profilé (Iz 3,20 et 4,50 en 2008 ; les plans
de combinaison AluClip du même manuel écrivent 3,2 pour le V057) ; les 2415 et 2416 y admettent aussi le
V290 et le V154 — entrée **CTR-69** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md). Les coupes combinées du même classeur
(registre 2, p. 52-79) donnent au 6112 le V058 (3,20) et au V057 l'Iz de 3,80 — entrée **INC-214**
du registre [Incohérences internes](/anomalies/incoherences-internes.md) ; elles sont sur
[Types d'ouverture et plans de combinaison du système 70](/profiles/systeme-70-plans-de-combinaison.md). Le V026 du 0113 porte
l'inertie de 3,7 cm⁴ que le DTD lui donne (voir *Renforts du DTD*) ; les V039 et V058 ne sont
dessinés sur aucun autre document du système 70.

Sur les coupes de porte d'entrée du registre 2, le V003 des ouvrants 2415 et 2416 porte Iz
12,80 cm⁴ au lieu des 12,30 de ces planches (**INC-216**) ; les ouvrants 6113 et 6116 y reçoivent le
V058 (3,20) et le V057 (3,80), comme les 6112 et 6115 (**INC-214**) [2 p. 96-98, 102-108].

### Pièces dessinées avec les ouvrants de 2008

Sur les planches p. 9, 10, 16 et 17, l'embout **9F13** (désignation F91-75- 9F13) est dessiné en
perspective à droite des ouvrants, devant l'extrémité d'un ouvrant coupé, sans légende de fonction ;
les planches des ouvrants élégance le dessinent devant un ouvrant élégance. Le manuel de mise en
œuvre le désigne « Embout ouvrant de gauche » (**INC-110**) [2 p. 9-10, 16-17].

![Embout 9F13 devant un ouvrant](/assets/profiles/systeme70/accessoires/embout-9f13-evo2008.png)

![Embout 9F13 devant un ouvrant élégance](/assets/profiles/systeme70/accessoires/embout-9f13-elegance-evo2008.png)

Sur les planches des ouvrants de porte 2416 (p. 12) et 2415 (p. 14) sont dessinés en plus : le
profilé acier **V004** (désignation F00-40- V004, le préfixe que la
[procédure d'assemblage du meneau](/procedures/assemblage-meneau-traverse-systeme-70.md) donne au
renfort acier V601), en coupe
hachurée, sans cote ni inertie ; une photographie légendée « F00-94- **9287** Insert soudable » et
« F00-87- **9679** Outil de serrage » ; et le profilé 9C42, repris dans
[Assemblages du système 70](/profiles/systeme-70-assemblages.md). La p. 12 dessine aussi le
profilé 9F46 [2 p. 12, 14].

![Profilé acier V004](/assets/profiles/systeme70/renforts/renfort-v004-evo2008.png)

![Insert soudable 9287 et outil de serrage 9679](/assets/profiles/systeme70/accessoires/insert-9287-outil-9679-evo2008.png)

La photographie montre deux ouvrants coupés d'onglet, face à face, avec l'insert entre leurs
chambres (schéma: raw/profine-plans-profiles-e-volution-2008-08.pdf, p. 12). Le poster et le
manuel légendent l'outil de serrage **T038** et l'insert 9287 « Insert de soudure » (voir
[Profilés complémentaires du système 70](/profiles/systeme-70-profiles-complementaires.md)).

# Renforts des ouvrants

Renforts admis par ouvrant, relevés sur la planche de chaque ouvrant du manuel, même convention
que pour les dormants. Le V069 et le V154 sont des renforts pré-usinés (livrés déjà
percés) de longueur fixe, sans inertie écrite [1 p. 25-34].

| Ouvrant | Renfort | Désignation sur la planche | Épaisseur d'acier (mm) | IG (cm⁴) | IW (cm⁴) | Longueur de livraison (m) |
| --- | --- | --- | --- | --- | --- | --- |
| 6112 | V158 | Renfort 2 mm | 2 | 0,5 | 3,2 | - |
| 6112 | V258 | Renfort 1,25 mm | 1,25 | 0,4 | 2,2 | - |
| 6113 | V158 | Renfort 2 mm | 2 | 0,5 | 3,2 | - |
| 6113 | V258 | Renfort 1,25 mm | 1,25 | 0,4 | 2,2 | - |
| 6115 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - |
| 6115 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - |
| 6115 | V069 | Renfort pré-usiné | - | - | - | 2 |
| 6116 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - |
| 6116 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - |
| 6116 | V069 | Renfort pré-usiné | - | - | - | 2 |
| 6121 | V158 | Renfort 2 mm | 2 | 0,5 | 3,2 | - |
| 6121 | V258 | Renfort 1,25 mm | 1,25 | 0,4 | 2,2 | - |
| 6122 | V158 | Renfort 2 mm | 2 | 0,5 | 3,2 | - |
| 6122 | V258 | Renfort 1,25 mm | 1,25 | 0,4 | 2,2 | - |
| 6123 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - |
| 6123 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - |
| 6123 | V069 | Renfort pré-usiné | - | - | - | 2 |
| 6124 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - |
| 6124 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - |
| 6124 | V069 | Renfort pré-usiné | - | - | - | 2 |
| 6117 | V158 | Renfort 2 mm | 2 | 0,5 | 3,2 | - |
| 6117 | V258 | Renfort 1,25 mm | 1,25 | 0,4 | 2,2 | - |
| 6118 | V158 | Renfort 2 mm | 2 | 0,5 | 3,2 | - |
| 6118 | V258 | Renfort 1,25 mm | 1,25 | 0,4 | 2,2 | - |
| 6119 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - |
| 6119 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - |
| 6119 | V069 | Renfort pré-usiné | - | - | - | 2 |
| 6120 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - |
| 6120 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - |
| 6120 | V069 | Renfort pré-usiné | - | - | - | 2 |
| 6150 | V158 | Renfort 2 mm | 2 | 0,5 | 3,2 | - |
| 6150 | V258 | Renfort 1,25 mm | 1,25 | 0,4 | 2,2 | - |
| 6151 | V158 | Renfort 2 mm | 2 | 0,5 | 3,2 | - |
| 6151 | V258 | Renfort 1,25 mm | 1,25 | 0,4 | 2,2 | - |
| 6152 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - |
| 6152 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - |
| 6152 | V069 | Renfort pré-usiné | - | - | - | 2 |
| 6153 | V057 | Renfort 2 mm | 2 | 4,8 | 3,8 | - |
| 6153 | V059 | Renfort 2 mm | 2 | 4,8 | 4,4 | - |
| 6153 | V069 | Renfort pré-usiné | - | - | - | 2 |
| 2415 | V003 | Renfort 2 mm | 2 | 19,7 | 12,3 | - |
| 2415 | V290 | Renfort 2 mm | 2 | 12,3 | 8,7 | - |
| 2415 | V154 | Renfort pré-usiné | - | - | - | 2,25 |
| 2416 | V003 | Renfort 2 mm | 2 | 19,7 | 12,3 | - |
| 2416 | V290 | Renfort 2 mm | 2 | 12,3 | 8,7 | - |
| 2416 | V154 | Renfort pré-usiné | - | - | - | 2,25 |
| 2418 | V030 | Renfort 2 mm | 2 | 4,0 | 4,5 | - |

(schéma: raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf, p. 25 à 34)

Le V030 est désigné « Renfort 2 mm » sur la planche du 2418 et « Renfort 1,5 mm » sur celle
du 2502 ; son dessin est coté 1,5 — entrée **INC-101** du registre
[Incohérences internes](/anomalies/incoherences-internes.md).

Le poster écrit le V069 dans la chambre des ouvrants 6115, 6119, 6120, 6123, 6124, 6152 et
6153, et le V154 dans celle du 2415 et du 2416 ; les planches du manuel les donnent aussi
(V069 sur les huit ouvrants de 77 et 97 mm, dont le 6116 ; V154 sur les 2415 et 2416) — voir
l'entrée **VER-61** du registre [Informations à vérifier](/anomalies/informations-a-verifier.md).

# Citations

[1] [Mise en œuvre Système 70 Plateforme, profine, version septembre 2023](raw/profine-mise-en-oeuvre-systeme-70-2023-09.pdf), p. 2-39 et 50

[2] [Système e.VOLUTION, plan des profilés, système F 91, édition août 2008](raw/profine-plans-profiles-e-volution-2008-08.pdf), p. 1-6, 8-23 et 47

[3] DTD n° DBV-24-6/16-2335_V5, système e.XCLUSIVE, e.MOTION, e.VOLUTION —
`raw/dtd-6-16-2335-v5-e-volution.pdf`, p. 14 à 18, 21, 23 et 25

[4] DTD n° DBV-25-6/16-2334_V5, système 76 Advanced —
`raw/dtd-6-16-2334-v5-systeme-76-advanced.pdf`, p. 4

[5] [Poster Kömmerling Gamme 70, profilés principaux, mars 2025](raw/poster-kommerling-70-principaux-2025-03.pdf), p. 1

# Voir aussi

Cette page est une partie d'un sujet qui en compte 3 ; les autres :

- [Cotes des ouvrants, ouvrants de porte et capots du système 70 Plateforme, ouvrants du DTD](/profiles/systeme-70-ouvrants.md)
- [Ouvrants des planches du manuel de mise en œuvre et des plans e.VOLUTION 2008, système 70](/profiles/systeme-70-ouvrants-manuel-et-plans-2008.md)
