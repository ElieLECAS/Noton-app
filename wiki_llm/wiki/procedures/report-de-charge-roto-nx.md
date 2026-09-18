---
type: Procédure
title: Report de charge ROTO NX
description: Montage et réglage du bras de report de charge NT Designo II sur quincaillerie ROTO NX, avec le critère visuel du cercle plein pour la tension du ressort.
tags: [roto-nx, nt-designo, report-de-charge, quincaillerie, pivot-angle, reglage, atelier]
status: stable
sources:
  - resource: raw/roto-nx-bras-report-de-charge.pdf
    id: roto-nx-report-de-charge
    title: Bras de report de charge ROTO NX, montage report de charge NT Designo II
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
generated:
  by: process:claude-code
  at: 2026-09-18T12:15:00Z
stale_after: 2027-12-31
---

# À quoi sert un report de charge

Le **report de charge NT Designo II** répartit le poids du vantail entre l'ouvrant et le dormant,
sur quincaillerie [ROTO NX](/quincaillerie/roto-nx.md). Il se compose de deux pièces — une pièce
d'ouvrant et une pièce de dormant — reliées par un ressort réglable et une tringle de soutien.

C'est la pièce qui permet de tenir un vantail lourd sans que le pivot d'angle encaisse seul la
charge.

# Montage, dans l'ordre

| Étape | Opération |
| --- | --- |
| 1 | Mettre et visser **le report de charge d'ouvrant en butée sur le pivot d'angle** |
| 2 | Retirer **la vis haute du palier d'angle** |
| 3 | Mettre et visser **la pièce dormant du report de charge sur le palier d'angle** |
| 4 | Mettre **le vantail sur le palier d'angle** |
| 5 | Mettre **la tringle de soutien du vantail dans le dormant** |

(schéma: raw/roto-nx-bras-report-de-charge.pdf)

**L'étape 2 est celle qu'on oublie** : la pièce de dormant se fixe à l'emplacement même de la vis
haute du palier d'angle. Sans l'avoir retirée, il n'y a pas de logement pour elle.

**L'ordre compte** : le vantail ne se repose sur son palier qu'après le vissage des deux pièces du
report de charge, pas avant.

# Réglage de la tension du ressort

| Paramètre | Valeur |
| --- | --- |
| Position du vantail pour le réglage | **ouverture à 90°** |
| Outil | **clé Allen de 4 mm** |
| Critère de réglage correct | les **deux arcs, le rouge et l'argent, forment un cercle plein** |

C'est un **contrôle visuel, pas un couple de serrage** : on tourne jusqu'à ce que les deux arcs se
rejoignent. Tant que le cercle est incomplet, le report de charge ne reprend pas la totalité de
l'effort prévu.

# Quand le report de charge devient nécessaire

Les [instructions de montage Roto NX KSR](/sources/roto-nx-ksr-montage.md) donnent trois champs
d'application du côté paumelles Designo, et c'est leur comparaison qui répond (p. 28 à 30).

| Configuration Designo II | LFF (mm) | HFF (mm) | Poids de vantail maxi (kg) |
| --- | --- | --- | --- |
| À la française et oscillo-battante, **sans** report de charge | 330 à 1400 | 280 à 2600 | 80 |
| À la française et oscillo-battante, **sans** report de charge | 600 à 1400 | 280 à 2600 | 100 |
| Oscillo-battante **avec** report de charge | 800 à 1400 | 1000 à 2600 | 150 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 28 à 30)

**Sans report de charge, le côté paumelles Designo plafonne à 100 kg.** Le report de charge est ce
qui ouvre la plage de 80 à 150 kg — mais il impose en contrepartie un vantail d'au moins **800 mm
de large et 1 000 mm de haut**. Il ne sert donc pas à rattraper un petit vantail lourd : celui-là
n'est pas réalisable.

**Au-delà de 130 kg, l'ouverture du compas doit être réduite à 80 mm** (manuel, p. 30). C'est la
contrepartie d'usage du vantail très lourd, et elle se dit au client avant la commande, pas après
la pose.

Voir [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md).

# Ce qui reste inconnu

- **la référence de commande** du NT Designo II et de ses composants
- **s'il s'applique aussi aux coulissants à pivot**, ou seulement aux oscillo-battants : les
  champs d'application du manuel ne parlent que de fenêtres à la française et oscillo-battantes

Entrée **VER-32** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md), dont la première question — le
seuil de mise en œuvre — trouve ici sa réponse pour le côté paumelles Designo.

Le wiki donne par ailleurs des poids d'ouvrant admissibles par ferrure côté profine, mais ils ne
mentionnent pas le report de charge — voir
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Provenance

Notice de montage illustrée, sans date ni numéro de version, extraite de
`raw/export_doc_192.zip`, source déclarée ROTO. Son titre interne est « Bras de report de charge
ROTO NX », celui de sa première planche « Montage report de charge NT Designo II ».

Le même montage figure au chapitre « Montage du report de charge NT Designo II » des
[instructions de montage Roto NX KSR](/sources/roto-nx-ksr-montage.md), p. 112 — la notice isolée
en est vraisemblablement un extrait diffusé à part.

# Citations

[1] Bras de report de charge ROTO NX, montage report de charge NT Designo II —
`raw/roto-nx-bras-report-de-charge.pdf`

[2] Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf.
IMO_180_NX_FR_v2, novembre 2022 —
`raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, p. 28 à 30 et 112

# Voir aussi

- [Roto NX](/quincaillerie/roto-nx.md)
- [ROTO](/fournisseurs/roto.md)
- [Transformation d'un ouvrant à la française en oscillo-battant](/procedures/transformation-of-en-ob-roto-nx.md)
- [Champs d'application Roto NX](/quincaillerie/roto-nx-champs-application.md)
- [Instructions de montage Roto NX KSR](/sources/roto-nx-ksr-montage.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
