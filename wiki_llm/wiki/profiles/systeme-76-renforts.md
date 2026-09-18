---
type: Profilé
title: Renforts du système 76
description: Les renforts acier du système 76 Advanced à joint central, leur épaisseur, leurs inerties IW et IG, et le profilé que chacun équipe.
tags: [systeme-76-advanced, renfort, acier, inertie, statique, atelier]
status: stable
sources:
  - resource: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf
    id: profine-mise-en-oeuvre-76-advanced
    title: Mise en œuvre Système 76 Advanced, profine
generated:
  by: process:claude-code
  at: 2026-09-18T09:00:00Z
stale_after: 2024-12-31
---

# Ce que désignent IW et IG

Chaque renfort acier du **système 76 Advanced à joint central** de
[profine](/fournisseurs/profine.md) porte deux valeurs d'inertie, et elles ne servent pas à la
même chose (registre 2.3.3, p. 2, version janvier 2016) :

| Valeur | Ce qu'elle borne | Ce qu'on en fait |
| --- | --- | --- |
| **IW** | la reprise des efforts dans la direction du vent | détermine la dimension réalisable sous une charge de vent donnée |
| **IG** | la reprise des efforts de poids | détermine l'**épaisseur de vitrage** admissible, donc le poids suspendu |

Un renfort à forte IW et faible IG tient le vent mais pas le triple vitrage. C'est la distinction
qui explique que deux renforts de même épaisseur d'acier ne donnent pas les mêmes limites.

Les inerties sont exprimées en **cm⁴**, l'épaisseur d'acier en **mm**. Les limites dimensionnelles
qui en découlent sont dans
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Renforts des dormants

Renforts admis par dormant, relevés sur les planches de profilés du registre 2.1.2 (p. 3 à 9,
version octobre 2021). Une ligne par couple dormant et renfort.

| Dormant | Renfort | Épaisseur d'acier (mm) | IW (cm⁴) | IG (cm⁴) |
| --- | --- | --- | --- | --- |
| 76171 | V306.Z | 1,5 | 2,3 | 1,3 |
| 76171 | V307.Z | 2,0 | 2,9 | 1,6 |
| 76171 | V308 | 2,5 | 3,4 | 1,9 |
| 76171 | V309.Z | 1,5 | 2,5 | 2,0 |
| 76171 | V310 | 2,0 | 3,2 | 2,5 |
| 76172 | V325 | 2,0 | 4,2 | 7,5 |
| 76173 | V306.Z | 1,5 | 2,3 | 1,3 |
| 76173 | V307.Z | 2,0 | 2,9 | 1,6 |
| 76173 | V308 | 2,5 | 3,4 | 1,9 |
| 76173 | V309.Z | 1,5 | 2,5 | 2,0 |
| 76173 | V310 | 2,0 | 3,2 | 2,5 |
| 76177 | V291.Z | 1,5 | 2,3 | 0,7 |
| 76178 | V291.Z | 1,5 | 2,3 | 0,7 |
| 76180 | V306.Z | 1,5 | 2,3 | 1,3 |
| 76180 | V307.Z | 2,0 | 2,9 | 1,6 |
| 76180 | V308 | 2,5 | 3,4 | 1,9 |
| 76180 | V309.Z | 1,5 | 2,5 | 2,0 |
| 76180 | V310 | 2,0 | 3,2 | 2,5 |
| 76185 | V291.Z | 1,5 | 2,3 | 0,7 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.2, p. 3 à 9)

**Les trois dormants rénovation 76177, 76178 et 76185 n'ont qu'un seul renfort possible**, le
V291.Z, et c'est celui de plus faible IG de tout le système : 0,7 cm⁴. Les trois dormants neufs
76171, 76173 et 76180 en acceptent cinq. Le 76172 est le seul à recevoir le V325, qui est aussi
le renfort de dormant le plus raide en poids du système.

À épaisseur d'acier égale, **le V309.Z apporte plus d'IG que le V306.Z** (2,0 contre 1,3 cm⁴)
pour une IW comparable : c'est le renfort à demander quand le vitrage est lourd et que le vent ne
l'est pas.

Voir [Dormants PERFORM76](/profiles/perform76-dormants.md).

# Renforts des ouvrants

Relevés sur les planches du registre 2.1.2 (p. 11 à 15, version octobre 2021) et sur les abaques
du registre 2.3.3 (p. 6 à 11).

| Ouvrant | Renfort | Épaisseur d'acier (mm) | IW (cm⁴) | IG (cm⁴) |
| --- | --- | --- | --- | --- |
| 76271 | V306.Z | 1,5 | 2,3 | 1,3 |
| 76271 | V307.Z | 2,0 | 2,9 | 1,6 |
| 76271 | V308 | 2,5 | 3,4 | 1,9 |
| 76271 | capot alu A072, sans renfort acier | — | 0,8 | 1,5 |
| 76272 | V326.Z | 2,0 | 5,0 | 5,4 |
| 76272 | V314.Z | 2,0 | 5,7 | 8,4 |
| 76274 | V266.Z | 2,0 | 2,5 | 0,5 |
| 76275 | V266.Z | 2,0 | 2,5 | 0,5 |
| 76276 | V266.Z | 2,0 | 2,5 | 0,5 |
| 76279 | V326.Z | 2,0 | 5,0 | 5,4 |
| 76279 | V314.Z | 2,0 | 5,7 | 8,4 |
| 76281 | V266.Z | 2,0 | 2,5 | 0,5 |
| 76283 | V326.Z | 2,0 | 5,0 | 5,4 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.2, p. 11 à 15)

**L'écart entre les ouvrants bas et les ouvrants hauts est un facteur 10 en IG** : 0,5 cm⁴ pour le
V266.Z des 76275 et 76281, 5,4 à 8,4 cm⁴ pour les renforts des 76272 et 76279. C'est ce qui
détermine l'épaisseur de vitrage admissible, pas la hauteur du profil elle-même.

Le **76271 équipé du capot aluminium A072 se passe de renfort acier** — c'est la variante AluClip
Pro. Ses inerties sont celles du capot, et elles sont les plus faibles du système en vent (0,8 cm⁴).
Une **équerre de feuillure J079 dans les quatre angles devient obligatoire à partir de 40 kg de
poids d'ouvrant** (registre 2.6.3, p. 1, version décembre 2016). Les battements, eux, restent
systématiquement renforcés.

Voir [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md).

# Renforts des meneaux, traverses et battements

Relevés sur les planches du registre 2.1.2 (p. 16 à 19, version octobre 2021).

| Profilé | Famille | Renfort | Épaisseur d'acier (mm) | IW (cm⁴) | IG (cm⁴) |
| --- | --- | --- | --- | --- | --- |
| 76300 | croisillon d'ouvrant | V312.Z | 1,5 | 1,5 | 0,3 |
| 76301 | traverse d'ouvrant | V320.Z | 1,5 | 3,5 | 2,4 |
| 76303 | traverse d'ouvrant | V323.Z | 1,5 | 9,2 | 13,0 |
| 76303 | traverse d'ouvrant | V322 | 2,5 | 16,8 | 22,0 |
| 76303 | traverse d'ouvrant | V324 | 2,5 | - | - |
| 76372 | meneau de dormant | V318.Z | 2,0 | 7,0 | 3,4 |
| 76372 | meneau de dormant | V319 | 2,5 | 9,6 | 4,2 |
| 76373 | meneau de dormant | V323.Z | - | 9,2 | 13,0 |
| 76373 | meneau de dormant | V322 | - | 16,8 | 22,0 |
| 76373 | meneau de dormant | V324 | - | 13,3 | 17,1 |
| 76471 | battement | V316 | 1,5 | 2,4 | 0,4 |
| 76472 | battement | V317 | 2,0 | 4,8 | 2,3 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.2, p. 16 à 19)

**Les épaisseurs d'acier des renforts du meneau 76373 ne sont pas reprises ici**, et l'inertie du
V324 monté sur le 76303 non plus : sur la planche p. 17, le texte du PDF associe au 76373 les
étiquettes du 76372 voisin, alors que la légende du dessin porte bien V323.Z, V322 et V324. Les
valeurs d'inertie données pour le 76373 sont celles portées dans son propre cartouche, corroborées
par la planche du 76303 qui reçoit les mêmes renforts. **À confirmer sur le document avant tout
calcul statique** — entrée **VER-23** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

Le **battement 76473 ne reçoit aucun renfort** dans le manuel, alors que les 76471 et 76472 en
reçoivent un chacun. C'est cohérent avec sa largeur : 46 mm contre 62 et 80 mm.

Voir [Meneaux PERFORM76](/profiles/perform76-meneaux.md) et
[Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md).

# Renforts des profilés complémentaires

Relevés sur les planches du registre 2.1.3 (p. 1 à 16, versions mars 2021 et septembre 2023).

| Profilé | Famille | Renfort | Épaisseur d'acier (mm) | IW (cm⁴) | IG (cm⁴) |
| --- | --- | --- | --- | --- | --- |
| 76700 | élargisseur 15 mm | V312.Z | 1,5 | 1,5 | 0,3 |
| 76701 | élargisseur 30 mm | V312.Z | 1,5 | 1,5 | 0,3 |
| 76702 | élargisseur 60 mm | V312.Z | 1,5 | 1,5 | 0,3 |
| 76703 | élargisseur 120 mm | V114 | 1,5 | 1,4 | 1,9 |
| 76704 | réhausse 45 mm | V114 | 1,5 | 1,4 | 1,9 |
| 76705 | réhausse 150 mm | V317 | 2,0 | 2,3 | 4,8 |
| 76706 | réhausse 45 mm | V114 | 1,5 | 1,4 | 1,9 |
| 76708 | réhausse 50 mm | V075 | 1,5 | 1,0 | 4,1 |
| 76709 | réhausse 100 mm | V299 | 1,5 | 0,6 | 1,8 |
| 76715 | réhausse 50 mm, complément du 76708 | V407 | 1,5 | 0,7 | 1,0 |
| 76722 | réhausse 35 mm | V167 | 1,5 | 1,07 | 1,61 |
| 76777 | réhausse 55 mm | V114 | 1,5 | 1,4 | 1,9 |
| 76605 | profilé de liaison en H | V330 | 2,5 | 5,5 | 0,8 |
| 76608 | profilé de liaison en H | V288 | 2,0 | 20,4 | 0,7 |
| 76206 | profilé complémentaire | V323.Z | 1,5 | 9,2 | 13,0 |
| 76206 | profilé complémentaire | V322 | 2,5 | 16,8 | 22,0 |
| 76800 | profilé de guidage de tablier | V262 | 2,0 | 7,6 | 7,6 |

(schéma: raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf, registre 2.1.3, p. 1 à 16)

Le **V288 du profilé de liaison 76608 est le renfort le plus raide au vent de tout le système** :
20,4 cm⁴, pour une IG quasi nulle de 0,7. C'est un profil de liaison, pas un porteur de vitrage.

**Le renfort V317 apparaît avec des inerties permutées selon la planche** : IW 4,8 / IG 2,3 sur le
battement 76472, IW 2,3 / IG 4,8 sur la réhausse 76705. C'est cohérent avec un montage tourné de
90°, l'axe fort passant du vent au poids — mais le manuel ne l'écrit pas. Entrée **VER-24** du
registre [Informations à vérifier](/anomalies/informations-a-verifier.md).

Voir [Élargisseurs et assemblage PERFORM76](/profiles/perform76-elargisseurs-et-assemblage.md).

# Vissage des renforts

Le manuel impose une **vis auto-perforeuse à tête fraisée** pour la fixation des renforts
(registre 2.4.1, p. 1 à 9, versions janvier 2016 à mars 2021).

| Renfort | Vis |
| --- | --- |
| V312.Z | 3,9 × 13 mm |
| tous les autres renforts du registre 2.4.1 | 3,9 × 16 mm |

**Les positions de vissage ne sont pas reprises dans ce wiki** : le registre 2.4.1 les donne sous
forme de dessins cotés à l'échelle 1:2, impossibles à transcrire sans les redessiner. Se reporter
au document pour percer.

Sur un profilé de couleur ou capoté aluminium, le vissage du renfort s'accompagne d'une
**ventilation obligatoire des préchambres extérieures** — voir
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Citations

[1] Mise en œuvre Système 76 Advanced, profine —
`raw/profine-mise-en-oeuvre-76-advanced-2023-12.pdf`, registres 2.1.2, 2.1.3, 2.3.3, 2.4.1 et 2.6.3

# Voir aussi

- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [Cotes de débit du système 76](/profiles/systeme-76-cotes-de-debit.md)
- [Dormants PERFORM76](/profiles/perform76-dormants.md)
- [Ouvrants et battements PERFORM76](/profiles/perform76-ouvrants-et-battements.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
