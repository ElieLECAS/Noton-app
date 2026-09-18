---
type: Vitrage
title: Performances des vitrages
description: Compositions de vitrages thermiques, acoustiques, triples et de sécurité proposées par PROFERM, avec leurs coefficients.
tags: [vitrage, thermique, acoustique, securite, ug, stadip, triple-vitrage]
status: stable
sources:
  - resource: raw/catalogue-general-2026-01.pdf
    id: catalogue-general-2026
    title: Catalogue menuiseries PROFERM, édition janvier 2026
  - resource: raw/cahier-technique-perform76-2026-09-02-cc03.pdf
    id: cahier-technique-perform76-cc03
    title: Cahier technique PERFORM76, version 02/09/2026 CC03
generated:
  by: process:claude-code
  at: 2026-09-17T20:00:00Z
stale_after: 2027-01-31
---

# Comment lire les coefficients

La performance thermique d'un vitrage se mesure par son coefficient **Ug** : plus il est bas,
plus le vitrage est isolant (catalogue général, p. 27). La performance acoustique se mesure par
un coefficient d'affaiblissement sonore en décibels, directement lié à l'épaisseur des vitrages
(catalogue général, p. 27).

Ne pas confondre **Ug** (le vitrage seul) et **Uw** (la fenêtre complète, vitrage + menuiserie),
qui est la valeur donnée sur les pages de gammes.

# Cotes des vitrages thermiques

Compositions et coefficients des vitrages thermiques et triples, épaisseurs en mm, relevées sur
le catalogue général PROFERM (p. 27).

| Vitrage | Composition (mm) | Ug (W/m²K) | Gain vs base |
| --- | --- | --- | --- |
| Double vitrage standard | 6 / 18 / 4 | 1,1 | référence |
| Double vitrage « SGC ULTRA ONE », en option | - | 1,0 | - |
| Triple vitrage | 4 / 14 / 4 / 14 / 4 | 1,0 | + 30 % |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

Le double vitrage standard est monté avec intercalaire warm edge à faible émissivité et gaz
argon. Le Ug de 1,1 W/m²K est donné pour les gammes PVC (catalogue général, p. 27).

Le « SGC ULTRA ONE » est présenté par le catalogue comme « le double vitrage le plus performant
du marché » (catalogue général, p. 27). La composition exacte n'est pas donnée — à vérifier
auprès du fournisseur de vitrage.

Le triple vitrage est composé de trois vitres séparées par deux lames remplies d'air, d'argon ou
de krypton. PROFERM le recommande pour les fenêtres situées **sur les façades nord et est**, et
le présente comme la solution pour répondre aux exigences des maisons passives (catalogue
général, p. 27).

# Composition du triple vitrage

Le triple vitrage proposé en option sur la gamme [HYBRIDE](/gammes/hybride.md) est un
**4/14/4/14/4** : trois verres de 4 mm séparés par deux lames de 14 mm, soit **40 mm** au total,
pour un affaiblissement acoustique de **28 dB** (catalogue général, p. 10 et 27). Le catalogue
porte la même notation aux deux pages.

# L'intercalaire du vitrage de série

Le catalogue général parle d'un « intercalaire warm edge » sans nommer de référence. Le cahier
technique PERFORM76 est plus précis : le double vitrage de série de la PERFORM76 est un **28 mm
(6 mm - 18 mm gaz argon - 4 mm) avec intercalaire TGI de coloris noir**, pour un **Ug de
1,1 W/m²K** (cahier technique PERFORM76, p. 1).

Le Ug de 1,1 concorde avec celui annoncé au catalogue pour les gammes PVC. Les deux sources sont
cohérentes ; le cahier technique ajoute la référence d'intercalaire et sa couleur.

# L'épaisseur limite en PERFORM76

La plus épaisse parclose de la PERFORM76 accepte **50 mm** de vitrage (cahier technique
PERFORM76, p. 5). Or les vitrages de sécurité du catalogue sont plus épais :

| Vitrage | Épaisseur totale | Tenable en PERFORM76 ? |
| --- | --- | --- |
| Double vitrage de série 6/18/4 | 28 mm | oui, parclose 76526 |
| Triple vitrage 4/14/4/14/4 | 40 mm | oui, parclose 76505 |
| STADIP 44²/16/4 | ~ 64 mm | **non, au-delà de la gamme de parcloses** |
| STADIP 44²/12/4 | ~ 60 mm | **non, au-delà de la gamme de parcloses** |

**Ce point doit être vérifié avant de promettre un vitrage de sécurité sur une PERFORM76.** Le
catalogue annonce le STADIP 44²/16/4 comme disponible sur les gammes PERFORM, TEXTURAL et
HYBRIDE (catalogue général, p. 27), mais le cahier technique ne donne aucune parclose capable de
le tenir. Soit une parclose spécifique existe hors cahier, soit la feuillure diffère — à trancher
au bureau d'études. Voir [Parcloses PERFORM76](/profiles/perform76-parcloses.md).

Entrée **CTR-02** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md).

# Cotes des vitrages acoustiques

Compositions acoustiques recommandées selon l'exposition au bruit, épaisseurs en mm, relevées sur
le catalogue général PROFERM (p. 27).

| Exposition | Niveau sonore | Composition recommandée (mm) |
| --- | --- | --- |
| Standard, gammes PVC | ~ 31 dB atteints | 6 / 18 / 4 (vitrage de base) |
| Façade sur route à fort trafic | 33 dB | 10 / 14 / 4 |
| Zone très bruyante : aéroport, route ou autoroute | 40 dB | 44.6 / 14 / 10 |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

Les deux compositions renforcées sont montées avec intercalaire warm edge à faible émissivité et
gaz argon. Le catalogue précise que le 10/14/4 « permet de ne pas perdre tant en performance
thermique qu'acoustique » (catalogue général, p. 27).

Le 44.6/14/10 cumule deux fonctions : sa face 44.6 silence se comporte aussi comme un
**retardateur d'effraction** (catalogue général, p. 27).

# Le triple vitrage est moins performant en acoustique

Point contre-intuitif à retenir, et à annoncer au client avant qu'il ne le découvre : sur la
gamme HYBRIDE, le triple vitrage est donné à **28 dB** contre **31 dB** pour le double vitrage de
base 6/18/4 (catalogue général, p. 10).

C'est physiquement normal — un double vitrage asymétrique est souvent meilleur en acoustique
qu'un triple symétrique — mais cela signifie qu'un client qui prend le triple vitrage *pour le
bruit* fait un mauvais choix. Le triple se justifie sur le thermique et sur la maison passive,
pas sur l'acoustique.

# Cotes des vitrages de sécurité

Compositions des vitrages de sécurité, épaisseurs en mm, relevées sur le catalogue général
PROFERM (p. 27).

| Vitrage | Composition (mm) | Gammes concernées |
| --- | --- | --- |
| STADIP | 44² / 16 / 4 | [PERFORM](/gammes/perform.md), [TEXTURAL](/gammes/textural.md), [HYBRIDE](/gammes/hybride.md) |
| STADIP | 44² / 12 / 4 | [LUMINE](/gammes/lumine.md) |
| Verre trempé SP10 | 2 glaces de 4 mm + 6 films PVB de 0,38 mm | - |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

Les vitrages de sécurité sont composés de deux vitrages assemblés entre eux par un ou plusieurs
films PVB (PolyVinylButyral), qui agissent comme retardateur d'effraction (catalogue général,
p. 27).

Les deux STADIP sont montés avec intercalaire warm edge à faible émissivité et gaz argon. **La
composition diffère selon la gamme** : 16 mm de lame d'air pour PERFORM, TEXTURAL et HYBRIDE,
12 mm pour LUMINE.

Le verre trempé **SP10** appartient à la classe **P5A**, la classe la plus forte. Il est composé
de deux glaces de 4 mm et de six films PVB qui lui permettent de résister aux jets d'objets
lourds. Il utilise du verre diamant extra-clair pour une parfaite transparence, et ses
intercalaires PVB de 0,38 mm sont superposés pour renforcer la résistance (catalogue général,
p. 27).

# Le vitrage de la porte labellisée RC2

La fenêtre PERFORM76 obtient le label RC2 avec un vitrage **44/6 collé** et une quincaillerie
spécifique (catalogue général, p. 34) — une composition qui n'apparaît pas dans le cahier
technique des vitrages. Voir
[Labels et certifications](/certifications/labels-et-certifications.md).

# Garantie

Le vitrage est garanti **10 ans** (catalogue général, p. 35). Voir
[Garanties par composant](/garanties/garanties-par-composant.md).

# Citations

[1] Catalogue menuiseries PROFERM, édition janvier 2026 — `raw/catalogue-general-2026-01.pdf`,
p. 10, 27, 34 et 35

# Voir aussi

- [Vitrages décoratifs](/vitrages/vitrages-decoratifs.md)
- [HYBRIDE](/gammes/hybride.md)
- [Coulissants aluminium](/gammes/coulissants-aluminium.md)
- [Catalogue menuiseries PROFERM, édition janvier 2026](/sources/catalogue-general-2026.md)
