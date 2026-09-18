---
type: Quincaillerie
title: Champs d'application Roto NX
description: Les largeurs, hauteurs et poids de vantail admissibles de la ferrure Roto NX selon le type d'ouverture et la classe de sécurité, avec la règle qui convertit l'épaisseur de vitrage en poids.
tags: [roto, roto-nx, abaque, champ-application, poids-vantail, cdr, rc2, designo, chiffrage]
status: stable
sources:
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
generated:
  by: process:claude-code
  at: 2026-09-18T14:15:00Z
stale_after: 2025-11-30
---

# Les trois grandeurs qui bornent une ferrure Roto NX

Tout champ d'application de la ferrure [Roto NX](/quincaillerie/roto-nx.md) s'exprime avec les
mêmes trois grandeurs, et le manuel de montage les abrège partout (réf. IMO_180_NX_FR_v2,
p. 21 à 30).

| Sigle | Grandeur |
| --- | --- |
| LFF | largeur de fond de feuillure du vantail |
| HFF | hauteur de feuillure du vantail |
| PV | poids du vantail |

**Ce sont des cotes de feuillure de vantail, pas des cotes de baie ni de dormant.** Elles ne se
comparent donc ni aux dimensions maximales du
[DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md), qui sont des cotes de baie, ni aux
abaques profine, qui sont des cotes extérieures d'ouvrant. Les trois séries de limites
s'appliquent en même temps et chacune peut être la contraignante.

# Cotes des champs d'application, côté paumelles P

Limites de la ferrure Roto NX en montage classique, en mm et en kg, par type d'ouverture et par
classe de sécurité. Une ligne par couple type d'ouverture et classe.

| Type d'ouverture | Classe de sécurité | LFF mini (mm) | LFF maxi (mm) | HFF mini (mm) | HFF maxi (mm) | PV maxi (kg) |
| --- | --- | --- | --- | --- | --- | --- |
| Oscillo-battant, fenêtre rectangulaire, version 130 kg | Sécurité de base | 290 | 1600 | 290 | 2800 | 130 |
| Oscillo-battant, fenêtre rectangulaire, version 130 kg | CDR 1 N | 320 | 1400 | 290 | 2600 | 130 |
| Oscillo-battant, fenêtre rectangulaire, version 130 kg | CDR 2 et CDR 2 N | 320 | 1400 | 510 | 2400 | 130 |
| Oscillo-battant, fenêtre rectangulaire, version 150 kg | Sécurité de base | 290 | 1600 | 290 | 2800 | 150 |
| Oscillo-battant, fenêtre rectangulaire, version 150 kg | CDR 1 N | 320 | 1400 | 290 | 2600 | 150 |
| Oscillo-battant, fenêtre rectangulaire, version 150 kg | CDR 2 et CDR 2 N | 320 | 1400 | 510 | 2400 | 150 |
| Oscillo-battant, fenêtre cintrée | Sécurité de base | 400 | 1300 | 500 | 1900 | 80 |
| Soufflet, fenêtre rectangulaire | Sécurité de base | 310 | 2400 | 290 | 1200 | 80 |
| Fenêtre confort | Sécurité de base | 520 | 1400 | 530 | 1600 | 50 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 21 à 27)

**La sécurité coûte de la dimension, pas du poids.** Passer de la sécurité de base au CDR 2 ne
change jamais le poids admissible, mais il retire 200 mm de largeur, 400 mm de hauteur, et impose
un vantail d'au moins 510 mm de haut. Le champ de la sécurité de base n'est donc pas un
sur-ensemble utilisable pour chiffrer une fenêtre de sécurité.

**Le CDR 1 N et le CDR 2 n'ont pas la même contrainte basse** : 290 mm de HFF mini en CDR 1 N,
510 mm en CDR 2. Un vantail bas peut être classé CDR 1 N et refusé en CDR 2.

Les classes CDR sont celles de la **DIN EN 1627-1630** (manuel, p. 19), et la position de
basculement à retard d'effraction **Tilt Safe** relève des classes **CDR 2 et CDR 2 N** — ce qui
justifie l'argument commercial du « RC2 avec oscillo-battant en position ouverte ». Voir
[Roto NX](/quincaillerie/roto-nx.md).

# Cotes des champs d'application, côté paumelles Designo II

Limites du côté paumelles Designo, en mm et en kg, relevées p. 28 à 30. Ce sont des paumelles
invisibles, et leurs limites ne sont pas celles du côté paumelles P.

| Configuration Designo II | LFF mini (mm) | LFF maxi (mm) | HFF mini (mm) | HFF maxi (mm) | PV maxi (kg) |
| --- | --- | --- | --- | --- | --- |
| Fenêtre à la française et oscillo-battante, sans report de charge, 80 kg | 330 | 1400 | 280 | 2600 | 80 |
| Fenêtre à la française et oscillo-battante, sans report de charge, 100 kg | 600 | 1400 | 280 | 2600 | 100 |
| Oscillo-battante avec report de charge, 80 à 150 kg | 800 | 1400 | 1000 | 2600 | 150 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 28 à 30)

**Monter en poids sur Designo II coûte de la largeur mini, pas de la largeur maxi.** La largeur
maximale reste 1 400 mm dans les trois cas, mais le vantail doit faire au moins 600 mm de large
pour 100 kg, et au moins 800 mm de large et 1 000 mm de haut avec report de charge. Un petit
vantail lourd n'est pas réalisable.

**Au-delà de 130 kg de poids de vantail, l'ouverture du compas doit être réduite à 80 mm**
(manuel, p. 30). C'est une restriction d'usage, pas de fabrication : la fenêtre s'ouvrira moins.

Le montage du report de charge est décrit dans
[Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md).

# Le catalogue de juin 2023 donne d'autres bornes, et une classe de plus

Le [catalogue Roto NX pour profils PVC](/sources/roto-nx-catalogue-pvc.md) de juin 2023 reprend
les mêmes champs d'application, côté paumelles P, oscillo-battant rectangulaire, avec des bornes
qui ne sont pas celles du manuel de montage de novembre 2022.

| Version | Classe de sécurité | LFF mini (mm) | LFF maxi (mm) | HFF mini (mm) | HFF maxi (mm) | PV maxi (kg) |
| --- | --- | --- | --- | --- | --- | --- |
| 130 kg | Sécurité de base | 290 | 1600 | 280 | 2800 | 130 |
| 130 kg | CDR 1 N | 320 | 1600 | 280 | 2800 | 130 |
| 130 kg | CDR 2 et CDR 2 N | 320 | 1400 | 510 | 2800 | 130 |
| 130 kg | **CDR 3** | 490 | 1400 | 600 | 2800 | 130 |
| 150 kg | Sécurité de base | 290 | 1600 | 280 | 2800 | 150 |
| 150 kg | CDR 1 N | 320 | 1600 | 280 | 2800 | 150 |
| 150 kg | CDR 2 et CDR 2 N | 320 | 1400 | 510 | 2800 | 150 |
| 150 kg | **CDR 3** | 320 | 1400 | 510 | 2800 | 150 |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 35 et 36)

**La CDR 3 n'existe que dans le catalogue.** C'est la classe la plus élevée documentée pour la
Roto NX, et elle coûte de la largeur : 490 mm de LFF mini en version 130 kg, contre 320 mm en
CDR 2.

**Les deux documents ROTO ne donnent pas les mêmes bornes** sur quatre valeurs — hauteur de
feuillure minimale, largeur maximale en CDR 1 N, hauteur maximale en CDR 1 N et en CDR 2. Aucune
règle du wiki ne départage deux documents du même fabricant : entrée **CTR-18** du registre
[Contradictions entre sources](/anomalies/contradictions-entre-sources.md). **Pour un chiffrage,
retenir les bornes les plus basses des deux documents** tant que l'arbitrage n'est pas fait.

Le tableau de la version 150 kg est **imprimé en allemand** dans ce catalogue français, et il y
désigne les classes par « RC » là où les pages françaises écrivent « CDR » — entrée **INC-13** du
registre [Incohérences internes](/anomalies/incoherences-internes.md).

# Convertir une épaisseur de vitrage en poids de vantail

Les diagrammes portent le poids du vitrage en **kg/m²**, pas l'épaisseur. Le manuel donne la
conversion (p. 21) :

> **1 mm/m² d'épaisseur de vitre ≙ 2,5 kg**

| Poids de vitrage porté au diagramme (kg/m²) | Épaisseur de verre correspondante (mm) |
| --- | --- |
| 20 | 8 |
| 30 | 12 |
| 40 | 16 |
| 50 | 20 |
| 60 | 24 |
| 80 | 32 |

Les courbes des diagrammes ne sont pas transcrites : elles n'existent que sous forme de tracés,
avec des zones marquées **« champ d'application non autorisé »** et **« 2ᵉ compas nécessaire »**.
Pour un cas limite, lire le diagramme sur le document.

**Un point qui tombe dans la zone « 2ᵉ compas nécessaire » reste réalisable**, mais il change la
nomenclature du châssis. Ce n'est pas un refus, c'est une pièce en plus.

# Cotes de force de traction par poids de vantail

Forces de traction rapportées au palier de compas, relevées p. 20.

| Poids du vantail (kg) | Force de traction (N) |
| --- | --- |
| 60 | 1 650 |
| 70 | 1 900 |
| 80 | 2 200 |
| 90 | 2 450 |
| 100 | 2 700 |
| 110 | 3 000 |
| 120 | 3 250 |
| 130 | 3 500 |
| 140 | 3 900 |
| 150 | 4 200 |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 34)

Les instructions de montage ne donnent que les deux dernières lignes, 140 et 150 kg, aux mêmes
valeurs [1 p. 20]. Le catalogue descend jusqu'à 60 kg [2 p. 34].

Ces valeurs valent **également pour les paliers d'angle lorsque la fixation est réalisée selon le
palier de compas**. Le manuel renvoie à la **directive TBDK** pour les forces de traction en
fonction du poids de vantail — la même directive qui borne les poids d'ouvrant admissibles du
système profine, voir
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Cotes de la ferrure soufflet

La ferrure soufflet a des limites supplémentaires liées au compas d'arrêt, relevées p. 25.

| Condition | Règle |
| --- | --- |
| Au-delà de 501 mm | compas d'arrêt possible en haut **uniquement avec crémone verrou** |
| Au-delà de 621 mm | compas pêne demi-tour en haut possible avec crémone dans le chant et crémone OB |
| À partir de 260 mm | côtés paumelles K, E5, P, T, A |
| À partir de 360 mm | côtés paumelles K, E5, P, T, A, Designo, Alu |
| À partir de 520 mm | **tous** côtés paumelles |

Trois positions de compas d'arrêt sont possibles, et le poids admissible en dépend : **jusqu'à
80 kg** en position possible et en position alternative, **jusqu'à 60 kg** dans la seconde position
alternative. Le manuel ne précise pas laquelle est laquelle autrement que par un repère graphique.

**L'emploi d'un compas d'arrêt latéral avec le verrouilleur médian VM 200 n'est pas possible.**
C'est une exclusion nette, la seule de ce chapitre.

# Cotes d'ouverture de la ferrure soufflet par hauteur de vantail

Positions de palier et angles d'ouverture par tranche de hauteur de feuillure de vantail, relevés
p. 26. Une ligne par tranche de HFF.

| HFF (mm) | Type | Position palier de vantail (mm) | Position palier de dormant (mm) | Ouverture en position d'entrebâillement (mm) | Angle d'entrebâillement (°) | Angle de position de nettoyage (°) |
| --- | --- | --- | --- | --- | --- | --- |
| 290 à 400 | 1 | 250 | 45 | 180 à 245 | 33 | 90 |
| 401 à 560 | 1 | 280 | 75 | 205 à 275 | 27 | 67 |
| 561 à 700 | 2 | 525 | 170 | 225 à 277 | 22 | 88 |
| 701 à 850 | 2 | 575 | 220 | 244 à 292 | 19 | 72 |
| 851 à 1200 | 2 | 625 | 270 | 261 à 363 | 17 | 62 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 26)

**L'angle d'entrebâillement diminue quand le vantail grandit** — de 33° à 17° — alors que
l'ouverture en millimètres, elle, augmente. Les deux chiffres décrivent la même chose vue
autrement, et c'est l'angle qui compte pour l'encombrement intérieur.

L'**angle de position de nettoyage ne suit pas cette décroissance** : il passe de 90° à 67°, puis
remonte à 88° au changement de type de palier, avant de redescendre. C'est le passage du type 1 au
type 2 qui le provoque.

# Ce que ces champs d'application ne disent pas

- **quel côté paumelles PROFERM emploie** — P ou Designo II. Les limites diffèrent nettement, et
  aucune source PROFERM ne le précise. Entrée **VER-34** du registre
  [Informations à vérifier](/anomalies/informations-a-verifier.md)
- **les références de ferrage** correspondant à chaque configuration : les 21 planches d'aperçu du
  manuel sont des nomenclatures dessinées
- **le poids du vantail lui-même** : il se calcule à partir du vitrage et des profilés, le manuel
  ne donne que la conversion de l'épaisseur de verre

Le manuel rappelle enfin que **les indications des fabricants de profilés et des propriétaires de
systèmes ne doivent pas être dépassées** pour la détermination des formats et des poids de vantail
admissibles (p. 20). Autrement dit : quand le champ d'application Roto et l'abaque profine ne
donnent pas la même limite, **c'est la plus basse qui s'applique**.

# Citations

[1] Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf.
IMO_180_NX_FR_v2, novembre 2022 —
`raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, p. 19 à 30

[2] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 —
`raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 34 à 36

# Voir aussi

- [Roto NX](/quincaillerie/roto-nx.md)
- [Instructions de montage Roto NX KSR](/sources/roto-nx-ksr-montage.md)
- [Catalogue Roto NX pour profils PVC](/sources/roto-nx-catalogue-pvc.md)
- [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md)
- [ROTO](/fournisseurs/roto.md)
