---
type: Quincaillerie
title: 'Champs d''application Roto NX : cotes côté paumelles P et ferrure OB rectangulaire'
description: Bornes LFF, HFF et poids de vantail admissibles de la ferrure Roto NX côté paumelles P selon le type d'ouverture et la classe de sécurité, conversion verre-poids, diagrammes OB rectangulaires 130 et 150 kg.
tags: [roto, roto-nx, abaque, champ-application, poids-vantail, cdr, rc2, designo, soufflet, cintre, tbdk, chiffrage]
systeme: Roto NX
fournisseur: ROTO
usage: [chiffrage, atelier]
famille: champs-application
status: draft
sources:
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    id: roto-nx-ksr-montage-imo-180
    title: Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf. IMO_180_NX_FR_v2
    last_modified: 2022-11-30
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    id: roto-nx-catalogue-ctl-105
    title: Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023
    last_modified: 2023-06-30
source_pages:
  - resource: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf
    pages: 9, 21-34, 36-37
  - resource: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf
    pages: 14, 34-52, 278
generated:
  by: process:claude-code
  at: 2026-10-21T11:04:21Z
---

# Les trois grandeurs qui bornent une ferrure Roto NX

Un *champ d'application* est l'ensemble des dimensions et des poids de vantail pour lesquels la
ferrure [Roto NX](/quincaillerie/roto-nx.md) est admise. Il s'exprime avec trois grandeurs, que le
manuel de montage abrège partout [1 p. 9] :

| Sigle | Grandeur |
| --- | --- |
| LFF | largeur fond de feuillure |
| HFF | hauteur de feuillure d'ouvrant |
| PV | poids du vantail |

La *feuillure* est le décrochement du profilé d'ouvrant dans lequel se loge la ferrure (voir le
[glossaire](/reference/glossaire.md)) : LFF et HFF sont donc des cotes prises en fond de feuillure
du vantail, pas des cotes de baie. Les schémas d'utilisation respectifs doivent être respectés
impérativement. Pour la détermination des formats de vantail et des poids de vantail maximum
admissibles, les indications des fabricants de profilés et des propriétaires de systèmes ne
doivent pas par ailleurs être dépassées [1 p. 22] — pour le système profine, voir
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Cotes des champs d'application, côté paumelles P

Limites de la ferrure Roto NX côté paumelles P, en mm et en kg, par type d'ouverture et par
classe de sécurité, relevées dans les tableaux « Champ d'application » du manuel de montage. Une
ligne par couple type d'ouverture et classe ; CDR est la classe de résistance à l'effraction selon
la DIN EN 1627-1630 [1 p. 9, 21].

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

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 23-26, 29)

Les deux tableaux oscillo-battants (130 kg et 150 kg) valent pour les pictogrammes ouvrant à la
française, oscillo-battant et fenêtre à deux vantaux (ouvrant secondaire) imprimés en tête de
chaque diagramme [1 p. 23-24]. La LFF du soufflet porte un renvoi « [1] » dont la note n'est
imprimée nulle part sur la page (entrée **INC-193**) [1 p. 26].

Les classes CDR 1 N, CDR 2 et CDR 2 N sont les classes du système Roto NX : la ferrure est
adaptable « de la sécurité de base jusqu'aux fenêtres de sécurité testées de la classification CDR
selon la DIN EN 1627-1630 », et la position de basculement avec retard d'effraction Tilt Safe
relève de la classification CDR 2 / CDR 2 N [1 p. 21]. Voir [Roto NX](/quincaillerie/roto-nx.md).

# Convertir une épaisseur de vitrage en poids de vantail

Les données des diagrammes d'application indiquent le poids du vitrage en **kg/m²**, pas
l'épaisseur. La conversion est imprimée sous chaque diagramme [1 p. 23] :

> **1 mm/m² d'épaisseur de vitre ≙ 2,5 kg**

Autrement dit, chaque millimètre d'épaisseur de verre pèse 2,5 kg par mètre carré de vitrage.

# Diagrammes d'application, ferrure OB, fenêtre rectangulaire

Chaque diagramme porte la LFF en abscisse (axe horizontal, en mm) et la HFF en ordonnée (axe
vertical, en mm), quadrillés tous les 100 mm. Une courbe rouge par poids de vitrage (en kg/m²)
borne, à gauche et en dessous d'elle, les formats de vantail admis pour ce vitrage : c'est la
« limitation du format de vantail en cas de différentes épaisseurs de vitre ». La zone blanche
quadrillée est le **champ d'application non autorisé** ; la zone gris foncé signale qu'un **2ᵉ
compas est nécessaire** ; le fond gris clair, non légendé, couvre le reste du champ. Une droite
noire borne le champ par le bas : sous elle, un vantail trop bas pour sa largeur sort du champ
[1 p. 23-24].

## Version 130 kg

![Diagramme d'application Roto NX, OB rectangulaire, 130 kg](/assets/quincaillerie/roto-nx-ksr/champs-application/diagramme-ob-rectangulaire-130-kg.png)

Trois courbes : 60, 50 et 40 kg/m². La courbe des 60 kg/m² descend du bord haut (HFF 2 800) à
partir d'une LFF d'environ 800 mm jusqu'à environ 1 630 mm de HFF pour une LFF d'environ 1 350 mm,
puis devient verticale jusqu'à la droite basse ; celle des 50 kg/m² part d'environ 940 mm de LFF,
descend jusqu'à environ 1 780 mm de HFF à environ 1 480 mm de LFF, puis devient verticale ; celle
des 40 kg/m² part d'environ 1 160 mm de LFF et atteint le bord droit (LFF 1 600) vers 2 030 mm de
HFF. La zone « 2ᵉ compas nécessaire » s'étend de 1 400 à 1 600 mm de LFF, entre la droite basse et
la courbe des 40 kg/m² [1 p. 23].

Relevé tous les 100 mm de LFF, précision de lecture ±25 mm. « 2800 » est le bord haut du
diagramme ; « hors champ » signifie que la LFF dépasse la verticale de la courbe.

| LFF (mm) | HFF maxi 60 kg/m² (mm) | HFF maxi 50 kg/m² (mm) | HFF maxi 40 kg/m² (mm) | HFF mini, droite basse (mm) | Zone |
| --- | --- | --- | --- | --- | --- |
| 300 | 2800 | 2800 | 2800 | bord bas | - |
| 400 | 2800 | 2800 | 2800 | bord bas | - |
| 500 | 2800 | 2800 | 2800 | 330 | - |
| 600 | 2800 | 2800 | 2800 | 400 | - |
| 700 | 2800 | 2800 | 2800 | 465 | - |
| 800 | 2800 | 2800 | 2800 | 530 | - |
| 900 | 2450 | 2800 | 2800 | 600 | - |
| 1000 | 2210 | 2630 | 2800 | 665 | - |
| 1100 | 2010 | 2400 | 2800 | 730 | - |
| 1200 | 1850 | 2190 | 2700 | 800 | - |
| 1300 | 1700 | 2030 | 2500 | 865 | - |
| 1400 | hors champ | 1890 | 2320 | 930 | 2ᵉ compas nécessaire |
| 1500 | hors champ | hors champ | 2180 | 995 | 2ᵉ compas nécessaire |
| 1600 | hors champ | hors champ | 2030 | 1065 | 2ᵉ compas nécessaire |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 23)

Le catalogue Roto NX pour profils PVC de juin 2023 imprime le même diagramme sous le titre
« Ferrure OB pivotante / de battant pivotant fenêtre rectangulaire, 130 kg », sans les
pictogrammes d'ouverture ni la phrase « Limitation du format de vantail » : mêmes trois courbes
(60, 50 et 40 kg/m²), mêmes deux zones légendées, même droite basse. Relu courbe par courbe à 600
et 900 dpi, il donne les valeurs du tableau ci-dessus aux écarts de lecture près (±25 mm) :
verticale des 60 kg/m² à 1 350 mm de LFF, haut à 1 635 mm de HFF ; verticale des 50 kg/m² à
1 480 mm, haut à 1 780 mm ; courbe des 40 kg/m² au bord droit vers 2 035 mm [2 p. 35]. Ses bornes
chiffrées sont dans la section *Le catalogue de juin 2023 donne d'autres bornes* ci-dessous.

![Diagramme d'application Roto NX, OB rectangulaire, 130 kg, catalogue 2023](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-ob-rectangulaire-130-kg.png)

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 35)

## Version 150 kg

![Diagramme d'application Roto NX, OB rectangulaire, 150 kg](/assets/quincaillerie/roto-nx-ksr/champs-application/diagramme-ob-rectangulaire-150-kg.png)

Quatre courbes : 80, 60, 50 et 40 kg/m². La courbe des 80 kg/m² part du bord haut à environ
700 mm de LFF et devient verticale à environ 1 170 mm de LFF (HFF environ 1 650) ; celle des
60 kg/m² part d'environ 910 mm et devient verticale à environ 1 350 mm (HFF environ 1 890) ; celle
des 50 kg/m² part d'environ 1 080 mm et devient verticale à environ 1 470 mm (HFF environ 2 060) ;
celle des 40 kg/m² part d'environ 1 180 mm et atteint le bord droit (LFF 1 600) vers 2 060 mm de
HFF. La zone « 2ᵉ compas nécessaire » s'étend de 1 400 à 1 600 mm de LFF [1 p. 24].

Relevé tous les 100 mm de LFF, précision de lecture ±25 mm, mêmes conventions que pour la
version 130 kg.

| LFF (mm) | HFF maxi 80 kg/m² (mm) | HFF maxi 60 kg/m² (mm) | HFF maxi 50 kg/m² (mm) | HFF maxi 40 kg/m² (mm) | HFF mini, droite basse (mm) | Zone |
| --- | --- | --- | --- | --- | --- | --- |
| 300 | 2800 | 2800 | 2800 | 2800 | bord bas | - |
| 400 | 2800 | 2800 | 2800 | 2800 | bord bas | - |
| 500 | 2800 | 2800 | 2800 | 2800 | 330 | - |
| 600 | 2800 | 2800 | 2800 | 2800 | 400 | - |
| 700 | 2800 | 2800 | 2800 | 2800 | 465 | - |
| 800 | 2450 | 2800 | 2800 | 2800 | 535 | - |
| 900 | 2160 | 2800 | 2800 | 2800 | 600 | - |
| 1000 | 1915 | 2520 | 2800 | 2800 | 670 | - |
| 1100 | 1760 | 2310 | 2750 | 2800 | 735 | - |
| 1200 | hors champ | 2100 | 2540 | 2750 | 800 | - |
| 1300 | hors champ | 1940 | 2350 | 2540 | 870 | - |
| 1400 | hors champ | hors champ | 2180 | 2360 | 935 | 2ᵉ compas nécessaire |
| 1500 | hors champ | hors champ | hors champ | 2210 | 1000 | 2ᵉ compas nécessaire |
| 1600 | hors champ | hors champ | hors champ | 2060 | 1070 | 2ᵉ compas nécessaire |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 24)

Le catalogue Roto NX pour profils PVC de juin 2023 imprime, pour la version 150 kg, un diagramme
à **trois courbes seulement : 60, 50 et 40 kg/m²**. La courbe des 80 kg/m² n'y figure pas ; les
trois courbes imprimées suivent celles des 60, 50 et 40 kg/m² du tableau ci-dessus aux écarts de
lecture près (relu à 600 dpi, ±25 mm : départ des 60 kg/m² vers 910 mm de LFF, verticale à
1 350 mm, haut à 1 890 mm de HFF ; verticale des 50 kg/m² à 1 480 mm, haut à 2 055 mm ; courbe
des 40 kg/m² au bord droit vers 2 070 mm). Sa légende est imprimée en allemand :
« unzulässiger Anwendungsbereich » (champ d'application non autorisé) et « Zweitschere
erforderlich » (2ᵉ compas nécessaire) [2 p. 36]. Pour un vitrage de 80 kg/m², le catalogue ne
donne donc aucune courbe : entrée **CTR-106** ; la langue, entrée **INC-13**.

![Diagramme d'application Roto NX, OB rectangulaire, 150 kg, catalogue 2023](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-ob-rectangulaire-150-kg.png)

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 36)

La droite basse est tracée aux mêmes points sur les deux diagrammes, aux écarts de lecture près.
Les verticales des courbes descendent jusqu'à cette droite ; sous elle, le champ est non autorisé
[1 p. 23-24].

# Citations

[1] Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf.
IMO_180_NX_FR_v2, novembre 2022 —
`raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, p. 9, 21 à 37 (numérotation du PDF)

[2] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 —
`raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 14, 34 à 52 et 278

# Voir aussi

- [Roto NX](/quincaillerie/roto-nx.md)
- [Instructions de montage Roto NX KSR](/sources/roto-nx-ksr-montage.md)
- [Roto NX KSR — conventions et consignes de sécurité](/procedures/roto-nx-ksr-consignes-generales.md)
- [Catalogue Roto NX pour profils PVC](/sources/roto-nx-catalogue-pvc.md)
- [Configurations Roto NX KSR — ouvrant à soufflet](/quincaillerie/roto-nx-ksr-soufflet.md)
- [Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md)
- [ROTO](/fournisseurs/roto.md)

Autres parties de ce sujet :

- [Champs d'application Roto NX : ferrure OB de fenêtre inclinée et de fenêtre cintrée](/quincaillerie/roto-nx-champs-application-ob-inclinee-cintree.md)
- [Champs d'application Roto NX : ferrure soufflet, compas d'arrêt et cotes d'ouverture](/quincaillerie/roto-nx-champs-application-soufflet.md)
- [Champs d'application Roto NX : paumelle à soufflet sur recouvrement et fenêtre confort](/quincaillerie/roto-nx-champs-application-paumelle-soufflet-confort.md)
- [Champs d'application Roto NX côté paumelles Designo II : 80, 100 et 150 kg](/quincaillerie/roto-nx-champs-application-designo.md)
- [Roto NX : bornes catalogue 2023 (CDR 3), force de traction, dimensionnement, tolérance](/quincaillerie/roto-nx-champs-application-catalogue-2023-forces-dimensionnement.md)
