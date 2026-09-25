---
type: Vitrage
title: Performances des vitrages
description: Compositions de vitrages thermiques, acoustiques, triples et de sécurité proposées par PROFERM, avec leurs coefficients et gammes compatibles.
tags: [vitrage, thermique, acoustique, securite, ug, stadip, sp10, triple-vitrage]
usage: chiffrage
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
source_pages:
  - resource: raw/catalogue-general-2026-01.pdf
    pages: 10, 27, 34-35
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    pages: 4
generated:
  by: process:gemini-coder
  at: 2026-09-19T17:15:00Z
---

# Définition et principes d'évaluation

La performance thermique d'un vitrage se mesure grâce à son coefficient **Ug** (coefficient de
transmission thermique du vitrage seul, en W/m²K) : plus ce coefficient est bas, plus le vitrage
est isolant [1 p. 27]. La performance d'un vitrage acoustique se mesure avec un coefficient
d'affaiblissement sonore en décibels (dB) ; elle est directement liée à l'épaisseur des vitrages
[1 p. 27].

Les compositions s'écrivent en millimètres, de l'extérieur vers l'intérieur : 6/18/4 est un verre
de 6 mm, une lame de gaz de 18 mm et un verre de 4 mm. Un chiffre double comme 44² ou 44.6 désigne
un verre feuilleté : deux verres de 4 mm assemblés par un ou plusieurs films PVB (PolyVinylButyral).
Le warm edge est un intercalaire à bord chaud (voir le [glossaire](/reference/glossaire.md)).

# Vitrages thermiques et triples

Compositions et Ug des vitrages thermiques ; une ligne par vitrage [1 p. 27].

| Type de vitrage | Composition (mm) | Remplissage | Intercalaire | Ug (W/m²K) | Gain | Schéma |
| --- | --- | --- | --- | --- | --- | ---: |
| Double vitrage standard, en standard | 6 / 18 / 4 | gaz argon | warm edge faible émissivité | 1,1 (gammes PVC) | - | ![Double vitrage standard](/assets/vitrages/performances/double-vitrage-standard.png) |
| Double vitrage « SGC ULTRA ONE », en option | - | - | - | 1,0 | - | - |
| Triple vitrage | 4 / 14 / 4 / 14 / 4 | air, argon ou krypton | - | 1,0 | 30 % par rapport au vitrage de base | ![Triple vitrage](/assets/vitrages/performances/triple-vitrage.png) |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

Le Ug de 1,1 W/m²K est donné pour le vitrage « de nos gammes PVC ». Le SGC ULTRA ONE est présenté
comme « le double vitrage le plus performant du marché » [1 p. 27]. Sur la PERFORM76, le double
vitrage de série est monté avec un intercalaire TGI noir [2 p. 4].

Le triple vitrage est composé de trois vitres séparées par deux lames remplies d'air, d'argon ou de
krypton ; il permet de répondre aux exigences des maisons passives. PROFERM recommande le triple
vitrage pour des fenêtres situées sur les façades nord et est [1 p. 27]. Sur la gamme HYBRIDE, le
triple vitrage en option (40 mm, 4/14/4/14/4) est donné pour 28 dB [1 p. 10].

# Vitrages acoustiques

Si les gammes PVC atteignent des niveaux d'affaiblissement acoustique de l'ordre de 31 dB avec leur
vitrage de base, les menuiseries PROFERM peuvent atteindre des performances plus élevées ; une
ligne par exposition au bruit [1 p. 27].

| Exposition | Affaiblissement acoustique (dB) | Composition (mm) | Intercalaire et gaz | Description | Schéma |
| --- | --- | --- | --- | --- | ---: |
| vitrage de base, gammes PVC | de l'ordre de 31 | 6 / 18 / 4 | warm edge, gaz argon | - | - |
| façades situées sur des routes à fort trafic | 33 | 10 / 14 / 4 | warm edge faible émissivité, gaz argon | cette solution permet de ne pas perdre tant en performance thermique qu'acoustique | ![Double vitrage 10/14/4](/assets/vitrages/performances/acoustique-10-14-4.png) |
| zones très bruyantes (aéroport, routes à très fort trafic ou autoroutes) | 40 | 44.6 / 14 / 10 | warm edge faible émissivité, gaz argon | allie aussi l'aspect sécurité grâce à sa face 44.6 silence qui se comporte comme un retardateur d'effraction | ![Double vitrage 44.6/14/10](/assets/vitrages/performances/acoustique-44-6-14-10.png) |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

Les deux schémas sont légendés « PLANITHERM ONE ». Celui du 44.6/14/10 porte « deux vitrages
assemblés avec un film PVB » et les épaisseurs 8,6 mm, 14 mm et 10 mm [1 p. 27].

# Vitrages de sécurité

Les vitrages sécurité sont composés de deux vitrages assemblés entre eux par un ou plusieurs films
PVB : un retardateur d'effraction efficace. PROFERM recommande deux solutions pour une sécurité
optimale du double vitrage ; une ligne par vitrage et par gamme [1 p. 27].

| Désignation | Composition (mm) | Intercalaire et gaz | Gammes | Épaisseurs portées sur le schéma (mm) | Schéma |
| --- | --- | --- | --- | --- | ---: |
| Vitrage STADIP | 44² / 16 / 4 | warm edge faible émissivité, gaz argon | [PERFORM](/gammes/perform.md), [TEXTURAL](/gammes/textural.md), [HYBRIDE](/gammes/hybride.md) | 8,2 / 16 / 4 | ![Vitrage STADIP](/assets/vitrages/performances/stadip.png) |
| Vitrage STADIP | 44² / 12 / 4 | warm edge faible émissivité, gaz argon | [LUMINE](/gammes/lumine.md) | - | - |
| Verre trempé « SP10 » | - | warm edge, gaz argon (schéma) | - | 10,3 / 18 / 4 | ![Verre trempé SP10](/assets/vitrages/performances/verre-trempe-sp10.png) |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

Le verre trempé « SP10 » appartient à la classe **P5A**, la classe la plus forte (classement de
résistance du verre feuilleté aux attaques manuelles, voir [Panneaux et monoblocs](/portes/panneaux-et-monoblocs.md)).
Il est composé de deux glaces de 4 mm et de six films PVB, qui lui permettent de résister aux jets
d'objets lourds et de freiner considérablement le risque. Il utilise du verre diamant extra-clair
pour une parfaite transparence. D'une épaisseur de 0,38 mm, les intercalaires PVB sont superposés
pour renforcer la résistance du verre [1 p. 27]. Aucune gamme n'est nommée pour le SP10.

# Vitrage de la fenêtre certifiée RC2

La menuiserie PERFORM76 certifiée classe anti-effraction **RC2** par CERIBOIS intègre un vitrage securit **44/6 collé** en feuillure combiné à un ferrage périmétrique et une poignée verrouillable Sécustik® [1 p. 34].

# Garanties

Les vitrages posés sur l'ensemble des menuiseries PROFERM bénéficient d'une garantie contractuelle de **10 ans** [1 p. 35].

# Citations

[1] [Catalogue menuiseries PROFERM, édition janvier 2026](raw/catalogue-general-2026-01.pdf), p. 10, 27, 34 et 35
[2] [Cahier technique PERFORM76, version 02/09/2026 CC03](raw/cahier-technique-perform76-2026-09-02-cc03.pdf), page du PDF 4

# Voir aussi

- [Vitrages décoratifs](/vitrages/vitrages-decoratifs.md)
- [PERFORM](/gammes/perform.md)
- [HYBRIDE](/gammes/hybride.md)
- [LUMINE](/gammes/lumine.md)
- [Labels et certifications](/certifications/labels-et-certifications.md)
- [Garanties par composant](/garanties/garanties-par-composant.md)
