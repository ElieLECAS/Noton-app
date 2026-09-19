---
type: Vitrage
title: Performances des vitrages
description: Compositions de vitrages thermiques, acoustiques, triples et de sécurité proposées par PROFERM, avec leurs coefficients et gammes compatibles.
tags: [vitrage, thermique, acoustique, securite, ug, stadip, sp10, triple-vitrage]
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
    pages: 1, 5
generated:
  by: process:gemini-coder
  at: 2026-09-19T17:15:00Z
---

# Définition et principes d'évaluation

La performance thermique d'un vitrage se mesure par son coefficient surfacique **Ug** (exprimé en $\text{W/m²K}$) : plus le coefficient est bas, plus les déperditions de chaleur sont faibles [1 p. 27].

La performance acoustique est quantifiée par l'indice d'affaiblissement acoustique (exprimé en $\text{dB}$), directement conditionné par la dissymétrie et l'épaisseur des vitrages [1 p. 27].

# Vitrages thermiques et triples

Compositions, intercalaires et coefficients d'isolation des vitrages thermiques [1 p. 27].

| Type de vitrage | Composition verres / lame (mm) | Gaz | Intercalaire | Ug (W/m²K) | Gain thermique |
| --- | --- | --- | --- | --- | --- |
| Double vitrage standard | 6 / 18 / 4 | Argon | Warm Edge faible émissivité | 1,1 | Référence |
| SGC ULTRA ONE (option) | - | Argon | Warm Edge faible émissivité | 1,0 | Renforcé |
| Triple vitrage | 4 / 14 / 4 / 14 / 4 | Argon ou Krypton | Warm Edge faible émissivité | 1,0 | + 30 % vs standard |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

Sur les menuiseries PVC [PERFORM](/gammes/perform.md), le double vitrage de série est monté avec un intercalaire TGI noir de 18 mm et du gaz argon pour un Ug de 1,1 W/m²K [2 p. 1].

Le triple vitrage 40 mm (4/14/4/14/4) est préconisé pour les parois exposées au nord et à l'est, et permet de respecter les critères de la construction passive selon la FFCP [1 p. 27, 34]. Son affaiblissement acoustique est établi à 28 dB sur la gamme HYBRIDE [1 p. 10].

# Vitrages acoustiques

Compositions asymétriques recommandées selon le niveau d'exposition au bruit extérieur [1 p. 27].

| Exposition environnementale | Affaiblissement acoustique (dB) | Composition verres / lame (mm) | Gaz et intercalaire | Spécificité |
| --- | --- | --- | --- | --- |
| Standard (vitrage de base PVC) | ~ 31 | 6 / 18 / 4 | Argon, Warm Edge | Équilibre thermique et acoustique |
| Façade sur route à fort trafic | 33 | 10 / 14 / 4 | Argon, Warm Edge | Maintien de l'isolation thermique |
| Zone très bruyante (aéroport, autoroute) | 40 | 44.6 / 14 / 10 | Argon, Warm Edge | Face 44.6 Silence retardatrice d'effraction |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

Le vitrage 44.6/14/10 intègre une glace feuilletée acoustique 44.6 Silence (deux verres de 4 mm assemblés par six films PVB acoustiques), apportant simultanément une résistance retardatrice d'effraction [1 p. 27].

# Vitrages de sécurité

Les vitrages de sécurité intègrent des films de butyral de polyvinyle (PVB) intercalés entre les glaces pour maintenir le vitrage en place en cas d'impact et retarder l'effraction [1 p. 27].

| Désignation | Composition (mm) | Épaisseur totale (mm) | Gammes compatibles | Niveau de protection |
| --- | --- | --- | --- | --- |
| STADIP 44²/16/4 | 44.2 (8,76) / 16 / 4 | 28,76 | [PERFORM](/gammes/perform.md), [TEXTURAL](/gammes/textural.md), [HYBRIDE](/gammes/hybride.md) | Retardateur d'effraction standard |
| STADIP 44²/12/4 | 44.2 (8,76) / 12 / 4 | 24,76 | [LUMINE](/gammes/lumine.md) | Retardateur d'effraction module 55 mm |
| Verre trempé SP10 | SP10 (10,28) / lame / verre | variable | Toutes gammes | Haute sécurité classe P5A |

(schéma: raw/catalogue-general-2026-01.pdf, p. 27)

## Détail technique du STADIP 44²

Le vitrage STADIP $44^2$ (notation industrielle du 44.2) se compose de deux glaces de 4 mm reliées par deux intercalaires PVB de 0,38 mm, soit une épaisseur feuilletée de 8,76 mm [1 p. 27] :
* En version $44^2 / 16 / 4$ (épaisseur totale de 28,76 mm), le vitrage s'adapte directement aux parcloses standard de 28 mm des gammes PERFORM, HYBRIDE et TEXTURAL (parclose 76526 sur système 76) [2 p. 5].
* En version $44^2 / 12 / 4$ (épaisseur totale de 24,76 mm), le vitrage est adapté aux profondeurs de feuillure de 24 à 28 mm de la gamme aluminium LUMINE.

## Détail technique du verre SP10 (Classe P5A)

Le vitrage SP10 relève de la classe de résistance la plus élevée **P5A** selon la norme de résistance aux attaques manuelles [1 p. 27] :
* La face feuilletée SP10 comprend deux verres de 4 mm assemblés par **six films PVB superposés** de 0,38 mm chacun (épaisseur de la face SP10 : $4 + 4 + 6 \times 0,38 = 10,28 \text{ mm}$).
* Le verre employé est du **verre diamant extra-clair** pour préserver une transmission lumineuse et une transparence parfaites malgré la superposition des films.
* L'ensemble résiste aux jets répétés d'objets lourds.

## Vitrage de la fenêtre certifiée RC2

La menuiserie PERFORM76 certifiée classe anti-effraction **RC2** par CERIBOIS intègre un vitrage securit **44/6 collé** en feuillure combiné à un ferrage périmétrique et une poignée verrouillable Sécustik® [1 p. 34].

# Garanties

Les vitrages posés sur l'ensemble des menuiseries PROFERM bénéficient d'une garantie contractuelle de **10 ans** [1 p. 35].

# Citations

[1] [Catalogue menuiseries PROFERM, édition janvier 2026](raw/catalogue-general-2026-01.pdf), p. 10, 27, 34 et 35
[2] [Cahier technique PERFORM76, version 02/09/2026 CC03](raw/cahier-technique-perform76-2026-09-02-cc03.pdf), p. 1 et 5

# Voir aussi

- [Vitrages décoratifs](/vitrages/vitrages-decoratifs.md)
- [PERFORM](/gammes/perform.md)
- [HYBRIDE](/gammes/hybride.md)
- [LUMINE](/gammes/lumine.md)
- [Labels et certifications](/certifications/labels-et-certifications.md)
- [Garanties par composant](/garanties/garanties-par-composant.md)
