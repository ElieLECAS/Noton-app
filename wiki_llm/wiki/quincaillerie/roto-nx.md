---
type: Quincaillerie
title: Roto NX
description: Quincaillerie ROTO équipant les gammes PERFORM+ et HYBRIDE+, avec traitement Roto Sil Level 6, système TiltSafe et accès au RC2.
tags: [roto, roto-nx, tiltsafe, rc2, oscillo-battant, anticorrosion]
status: draft
sources:
  - resource: raw/brochure-perform-plus-hybride-plus-2023-05.pdf
    id: brochure-perform-plus-hybride-plus-2023-05
    title: Brochure Nouveautés PERFORM+ et HYBRIDE+, édition mai 2023
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
generated:
  by: process:claude-code
  at: 2026-09-17T20:30:00Z
stale_after: 2024-05-31
---

# Ce qu'est la Roto NX

Roto NX est la gamme de quincaillerie [ROTO](/fournisseurs/roto.md) dont PROFERM équipe les
fenêtres [PERFORM+](/gammes/perform-plus.md) et [HYBRIDE+](/gammes/hybride-plus.md) (brochure
PERFORM+/HYBRIDE+, p. 2).

C'est la seule gamme de quincaillerie nommée dans toute la documentation versée dans `raw/` : le
catalogue général parle de « quincaillerie ROTO » et de « ferrure ROTO » sans jamais nommer de
gamme.

# Composants

Relevé de la brochure PERFORM+/HYBRIDE+ (p. 2) :

| Composant | Rôle |
| --- | --- |
| Gâche de sécurité | étanchéité et résistance renforcées |
| Galet de sécurité | étanchéité et résistance renforcées |
| Pivot symétrique systématique | de série |
| Paumelles invisibles | **en option** |
| Crémone de semi-fixe | fermeture du semi-fixe |
| Système TiltSafe | sécurise la position oscillo-battante ouverte |
| Système anti-fausse manœuvre | empêche la manœuvre incorrecte |
| Réhausseur réglable monté sur la crémone | réglage, monté sur la crémone |
| Dispositif anti-claquement intégré | sur l'ouverture oscillo-battante |

Le pivot symétrique est **systématique**, les paumelles invisibles sont **en option** : la
distinction est explicite dans la brochure et change le chiffrage.

# Cotes

Largeurs d'ouverture oscillo-battante, en mm, relevées sur la brochure PERFORM+/HYBRIDE+ (p. 2).

| Caractéristique | Valeur (mm) |
| --- | --- |
| Largeur d'ouverture OB, position 1 | 80 |
| Largeur d'ouverture OB, position 2 | 140 |

Les deux largeurs sont équipées du **dispositif anti-claquement intégré**. La brochure ne dit pas
si le choix se fait à la commande ou si les deux positions sont disponibles sur une même
menuiserie — à vérifier.

# Traitement de surface Roto Sil Level 6

Le traitement de surface **Roto Sil Level 6** « surpasse les exigences de la classe anticorrosion
5, la classe maximale, pour un traitement d'une dureté exceptionnelle et d'une protection
durable contre la corrosion » (brochure PERFORM+/HYBRIDE+, p. 2).

| Caractéristique | Valeur |
| --- | --- |
| Classe anticorrosion de référence | 5, présentée comme la classe maximale |
| Positionnement annoncé | au-delà de la classe 5 |

L'argument est à manier avec précaution : la brochure affirme dépasser une classe qu'elle
présente elle-même comme maximale, sans nommer la norme qui définit ces classes. **Vérifier la
norme de référence avant de reprendre cet argument** face à un prescripteur — une classe
anticorrosion s'appuie normalement sur un essai normalisé et une durée d'exposition.

Ce traitement pourrait constituer un argument pour les chantiers en bord de mer, où le catalogue
général ne propose qu'une garantie corrosion en option, plus courte à moins de 10 km du littoral.
Voir [Garanties par composant](/garanties/garanties-par-composant.md).

# Accès au RC2

La Roto NX permet de « répondre à la classe de résistance 2 avec OB position ouverte (RC2) »
(brochure PERFORM+/HYBRIDE+, p. 2).

C'est une **seconde voie d'accès au RC2** chez PROFERM, distincte de celle documentée au
catalogue général :

| Voie | Produit | Ce qui porte le RC2 |
| --- | --- | --- |
| Par la quincaillerie | PERFORM+, HYBRIDE+ | Roto NX, avec oscillo-battant en position ouverte |
| Par le vitrage et le ferrage | PERFORM76 | vitrage 44/6 collé, ferrage périmétrique, poignée verrouillable Sécustik, labellisation CERIBOIS |

La mention « avec OB position ouverte » est inhabituelle et mérite d'être comprise : elle suggère
que la résistance RC2 est maintenue **même fenêtre entrouverte en oscillo-battant**, ce qui est
précisément la faiblesse classique de cette position. C'est vraisemblablement ce que sécurise le
système TiltSafe.

Le catalogue général rattache par ailleurs l'accès aux certifications RC1 et RC2 au **Label ROTO
Performance**, dont PROFERM est le premier bénéficiaire (catalogue général, p. 34). Voir
[Labels et certifications](/certifications/labels-et-certifications.md).

# Garantie

La ferrure ROTO est garantie **10 ans sur le fonctionnement**, valeur identique dans les deux
sources — c'est l'un des rares postes de garantie sur lequel la brochure et le catalogue général
s'accordent (brochure PERFORM+/HYBRIDE+, p. 3 ; catalogue général, p. 35).

# Ce que le manuel de montage ROTO ajoute à la brochure

Les [instructions de montage Roto NX KSR](/sources/roto-nx-ksr-montage.md), versées en novembre
2022, comblent trois des quatre manques que cette page signalait.

| Manque | Réponse du manuel |
| --- | --- |
| Abaques de charge | [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md) — LFF, HFF et poids de vantail par type d'ouverture et par classe de sécurité |
| Norme du traitement anticorrosion | **DIN EN 13126/8**, finition Roto Sil argent mat, **exempte de composés de chrome VI** |
| Périmètre du RC2 | classification **CDR selon DIN EN 1627-1630**, avec un champ d'application propre à chaque classe. Le **Tilt Safe** relève des classes **CDR 2 et CDR 2 N** |
| Références de ferrage | **toujours manquant** — les 21 planches d'aperçu du manuel sont des nomenclatures dessinées |

**Le manuel confirme la charge annoncée : jusqu'à 150 kg**, côté paumelles P reposant, pour
fenêtres et portes PVC. Il confirme aussi la **garantie 10 ans sur la fonctionnalité** des
ferrures et ajoute la certification **QM 328**.

Deux caractéristiques d'atelier que la brochure ne mentionnait pas : l'assemblage **« Clip&Fit »**
mécanique et sans perte de course, et l'assemblage sans vis de têtière et de boîtier de crémone
par le système **Easy Mix System**, pour des dimensions de fouillot **≥ 25 mm**.

Le manuel distingue enfin **trois galets de verrouillage** — E réglable en pression d'appui, P de
sécurité réglable en pression d'appui, V de sécurité réglable en pression d'appui **et en
hauteur** — là où la brochure ne parlait que d'un « galet de sécurité ».

# Ce qui reste à documenter

Cette page reste en `status: draft` : les produits que la Roto NX équipe sont de statut incertain,
entrée **VER-02** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Manquent encore : les **références de ferrage** configuration par configuration, et **le côté
paumelles que PROFERM emploie réellement** — P ou Designo II, dont les champs d'application
diffèrent nettement. Entrée **VER-34**.

# Citations

[1] Brochure Nouveautés PERFORM+ et HYBRIDE+, édition mai 2023 —
`raw/brochure-perform-plus-hybride-plus-2023-05.pdf`, p. 2 et 3
[2] [Catalogue menuiseries PROFERM, édition janvier 2026](/sources/catalogue-general-2026.md),
p. 34 et 35

# Trois interventions documentées sur ROTO NX

| Intervention | Page |
| --- | --- |
| Transformer un ouvrant à la française en oscillo-battant, sur PVC | [Transformation OF en OB](/procedures/transformation-of-en-ob-roto-nx.md) |
| Monter et régler le bras de report de charge NT Designo II | [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md) |
| Entretenir et contrôler une ferrure en service | [Maintenance d'une ferrure Roto NX](/procedures/maintenance-ferrure-roto-nx.md) |

# Voir aussi

- [ROTO](/fournisseurs/roto.md)
- [PERFORM+](/gammes/perform-plus.md)
- [HYBRIDE+](/gammes/hybride-plus.md)
- [Labels et certifications](/certifications/labels-et-certifications.md)
- [Brochure Nouveautés PERFORM+ et HYBRIDE+](/sources/brochure-perform-plus-hybride-plus.md)
- [Instructions de montage Roto NX KSR](/sources/roto-nx-ksr-montage.md)
- [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)
- [Maintenance d'une ferrure Roto NX](/procedures/maintenance-ferrure-roto-nx.md)
