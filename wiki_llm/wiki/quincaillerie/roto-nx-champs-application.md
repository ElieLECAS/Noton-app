---
type: Quincaillerie
title: Champs d'application Roto NX
description: Les largeurs, hauteurs et poids de vantail admissibles de la ferrure Roto NX selon le type d'ouverture et la classe de sécurité — tableaux, diagrammes d'application relevés courbe par courbe, positions des compas d'arrêt de la ferrure soufflet, forces de traction TBDK — avec la règle qui convertit l'épaisseur de vitrage en poids.
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
    pages: 34-36
generated:
  by: process:claude-code
  at: 2026-09-18T14:15:00Z
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

La droite basse est tracée aux mêmes points sur les deux diagrammes, aux écarts de lecture près.
Les verticales des courbes descendent jusqu'à cette droite ; sous elle, le champ est non autorisé
[1 p. 23-24].

# Diagramme d'application, ferrure OB, fenêtre cintrée

La fenêtre cintrée a un sommet en arc de cercle, de rayon R. **Le rayon (R) de la fenêtre cintrée
doit correspondre à la moitié de la LFF** [1 p. 25].

![Diagramme d'application Roto NX, OB cintré](/assets/quincaillerie/roto-nx-ksr/champs-application/diagramme-ob-cintre.png)

Le diagramme porte la LFF en abscisse (400 à 1 300 mm) et la HFF en ordonnée (500 à 1 900 mm),
quadrillés tous les 100 mm ; l'arc tracé au-dessus rappelle le cintre et son rayon R. Trois courbes
rouges : 40, 30 et 20 kg/m². Quatre zones sont légendées : champ d'application non autorisé (blanc
quadrillé), 2ᵉ compas nécessaire (gris clair), 2ᵉ compas possible mais non nécessaire (gris foncé),
2ᵉ compas non possible (hachures). Le gris foncé occupe la partie gauche du champ, à gauche de la
courbe des 40 kg/m² et au-dessus d'environ 950 mm de HFF ; le gris clair s'étend de la courbe des
40 kg/m² au bord droit ; les hachures couvrent la bande sous environ 950 mm de HFF ; sous la courbe
des 20 kg/m², en bas à droite, le champ est non autorisé [1 p. 25].

Les courbes sont faites de segments droits ; le tableau donne leurs sommets, relevés à ±25 mm. Les
trois courbes se rejoignent au coin bas du champ (LFF 500, HFF 500).

| Courbe | Sommet haut (LFF × HFF, mm) | Sommet intermédiaire (LFF × HFF, mm) | Sommet de la bande hachurée (LFF × HFF, mm) | Point bas (LFF × HFF, mm) |
| --- | --- | --- | --- | --- |
| 40 kg/m² | 850 × 1900 | 950 × 1600 | 900 × 950 | 500 × 500 |
| 30 kg/m² | 1100 × 1900 | 1100 × 1600 | 1000 × 950 | 500 × 500 |
| 20 kg/m² | 1300 × 1900 | 1300 × 1600 | 1200 × 950 | 500 × 500 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 25)

# Diagramme d'application, ferrure soufflet, fenêtre rectangulaire

La ferrure soufflet équipe un vantail qui bascule seulement, sans ouverture à la française. Ses
limites propres sont dans le tableau des champs P ci-dessus (LFF 310-2400, HFF 290-1200, 80 kg)
[1 p. 26].

![Diagramme d'application Roto NX, soufflet rectangulaire](/assets/quincaillerie/roto-nx-ksr/champs-application/diagramme-soufflet-rectangulaire.png)

Le diagramme porte la LFF en abscisse (graduée 500, 1 000, 1 500, 2 000 ; bornée à gauche vers
310 mm et à droite à 2 400 mm) et la HFF en ordonnée (graduée 500, 1 000 ; bornée en bas vers
290 mm et en haut à 1 200 mm). Deux cotes sont portées : **621** mm sur l'axe des LFF et
**560** mm sur l'axe des HFF. Sous l'axe, deux plages : **[A]** jusqu'à 1 200 mm de LFF, **[B]** de
1 200 à 2 400 mm [1 p. 26].

<table>
<thead>
<tr><th>Zone du diagramme</th><th>LFF (mm)</th><th>HFF (mm)</th><th>Teinte : compas d'arrêt</th><th>Hachures : compas d'entrebâillement et de nettoyage</th></tr>
</thead>
<tbody>
<tr><td>gauche, haut</td><td>jusqu'à 621</td><td>560 à 1200</td><td>2 compas d'arrêt latéralement</td><td>supplémentaire</td></tr>
<tr><td>gauche, bas</td><td>jusqu'à 621</td><td>jusqu'à 560</td><td>2 compas d'arrêt latéralement</td><td>-</td></tr>
<tr><td>milieu, haut</td><td>621 à 1200</td><td>560 à 1200</td><td>1 compas d'arrêt en haut ou 2 compas d'arrêt latéralement</td><td>supplémentaire</td></tr>
<tr><td>milieu, bas</td><td>621 à 1200</td><td>jusqu'à 560</td><td>1 compas d'arrêt en haut ou 2 compas d'arrêt latéralement</td><td>additionnel en cas de compas d'arrêt(s) en haut</td></tr>
<tr><td>droite, haut</td><td>1200 à 2400</td><td>560 jusqu'à la limite oblique</td><td>2 compas d'arrêt en haut ou 2 compas d'arrêt latéralement</td><td>supplémentaire</td></tr>
<tr><td>droite, bas</td><td>1200 à 2400</td><td>jusqu'à 560</td><td>2 compas d'arrêt en haut ou 2 compas d'arrêt latéralement</td><td>additionnel en cas de compas d'arrêt(s) en haut</td></tr>
<tr><td>au-dessus de la limite oblique</td><td>1200 à 2400</td><td>de la limite oblique à 1200</td><td>champ d'application non autorisé</td><td>-</td></tr>
</tbody>
</table>

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 26)

La limite oblique est un segment droit qui part du coin haut (LFF 1 200, HFF 1 200) et descend
jusqu'à environ 800 mm de HFF au bord droit (LFF 2 400), relevé à ±25 mm [1 p. 26].

La légende des plages imprime deux fois la même lettre : « [A] = 2 paumelles au minimum » et
« [A] = 3 paumelles au minimum », alors que le dessin porte les deux plages [A] et [B] ; la
correspondance de la seconde ligne n'est pas écrite (entrée **INC-192**) [1 p. 26].

**Compas d'entrebâillement et de nettoyage** : recommandé ; nécessaire en cas d'imposte (selon
RAL RG 607 / 12). Compas d'entrebâillement et de nettoyage jusqu'à 60 kg maxi [1 p. 26].

## Positions des compas d'arrêt de la ferrure soufflet

La planche suivante place, pour chaque format de vantail soufflet, les compas d'arrêt possibles.
Un *compas d'arrêt* retient le vantail basculé à son ouverture maximale. Les colonnes sont les LFF
(200 à 2 400 mm, tous les 100 mm), les lignes les HFF (200 à 1 200 mm, tous les 100 mm) ; chaque
case dessine le vantail vu de face avec ses compas : des points noirs sur les côtés (compas
latéraux), un triangle sur le haut (un compas en haut) ou deux cercles sur le haut (deux compas en
haut). Les teintes et hachures sont celles du diagramme précédent [1 p. 27]. La planche dessine
des vantaux à 200 et 300 mm de LFF et à 200 mm de HFF, en deçà des minima du tableau (310 mm de
LFF, 290 mm de HFF) : entrée **INC-194**.

![Positions des compas d'arrêt, ferrure soufflet Roto NX](/assets/quincaillerie/roto-nx-ksr/champs-application/soufflet-positions-compas-arret.png)

<table>
<thead>
<tr><th>HFF (mm)</th><th>LFF 200 à 500 (mm)</th><th>LFF 600 (mm)</th><th>LFF 700 à 1100 (mm)</th><th>LFF 1200 à 2400 (mm)</th></tr>
</thead>
<tbody>
<tr><td>1200</td><td>compas latéraux</td><td>latéraux + triangle en haut</td><td>latéraux + triangle en haut</td><td>latéraux + deux cercles en haut, LFF 1200 seulement</td></tr>
<tr><td>1100</td><td>compas latéraux</td><td>latéraux + triangle en haut</td><td>latéraux + triangle en haut</td><td>latéraux + deux cercles en haut, LFF 1200 à 1500</td></tr>
<tr><td>1000</td><td>compas latéraux</td><td>latéraux + triangle en haut</td><td>latéraux + triangle en haut</td><td>latéraux + deux cercles en haut, LFF 1200 à 1800</td></tr>
<tr><td>900</td><td>compas latéraux</td><td>latéraux + triangle en haut</td><td>latéraux + triangle en haut</td><td>latéraux + deux cercles en haut, LFF 1200 à 2100</td></tr>
<tr><td>600 à 800</td><td>compas latéraux</td><td>latéraux + triangle en haut</td><td>latéraux + triangle en haut</td><td>latéraux + deux cercles en haut, LFF 1200 à 2400</td></tr>
<tr><td>300 à 500</td><td>compas latéraux</td><td>latéraux + triangle en haut</td><td>latéraux + triangle en haut</td><td>latéraux + deux cercles en haut, LFF 1200 à 2400</td></tr>
<tr><td>200</td><td>aucun vantail dessiné</td><td>aucun vantail dessiné</td><td>triangle en haut seulement</td><td>deux cercles en haut seulement, LFF 1200 à 2400</td></tr>
</tbody>
</table>

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 27)

Deux traits verticaux, repérés **[A]** (entre 500 et 600 mm de LFF) et **[B]** (entre 600 et
700 mm), et trois traits horizontaux, repérés **[C]** (entre 200 et 300 mm de HFF), **[D]** (entre
300 et 400 mm) et **[E]** (entre 500 et 600 mm), renvoient aux notes suivantes [1 p. 27] :

| Repère | Note |
| --- | --- |
| [A] | au-delà de 501 mm, compas d'arrêt possible en haut uniquement avec crémone verrou |
| [B] | au-delà de 621 mm, compas pêne demi-tour en haut possible avec crémone dans le chant et crémone OB |
| [C] | à partir de **260 mm** K, E5, P, T, A |
| [D] | à partir de **360 mm** K, E5, P, T, A, Designo, Alu |
| [E] | à partir de **520 mm**, tous côtés paumelles |

Les lettres K, E5, P, T, A, Designo et Alu désignent des côtés paumelles (le système de paumelles
du vantail) ; le manuel ne détaille que P et Designo II.

La planche liste trois positions de compas d'arrêt [1 p. 27] :

- Position possible compas d'arrêt jusqu'à 80 kg ;
- Position alternative compas d'arrêt jusqu'à 80 kg ;
- Position alternative compas d'arrêt jusqu'à 60 kg.

![Légende des positions de compas d'arrêt](/assets/quincaillerie/roto-nx-ksr/champs-application/soufflet-positions-compas-arret-legende.png)

Les trois pastilles de cette légende sont imprimées comme trois taches sombres presque identiques,
qui ne reprennent ni le point noir, ni le triangle, ni les deux cercles de la planche : quelle
position vaut 60 kg et laquelle 80 kg n'est pas lisible (entrée **INC-191**).

**L'utilisation de compas d'arrêt latéral en liaison avec le verrouilleur médian VM 200 n'est pas
possible** [1 p. 27].

# Cotes d'ouverture de la ferrure soufflet par hauteur de vantail

Le dessin montre en coupe verticale un vantail soufflet basculé (gris, l'ouvrant) sur son dormant
(rose saumon), avec son compas d'arrêt, la poignée et, en pointillés, le vantail en position de
nettoyage ouvert à plat vers l'intérieur. Les cotes du dessin ne portent pas leurs lettres ; la
légende les nomme [A] à [F] [1 p. 28].

![Cotes d'ouverture de la ferrure soufflet](/assets/quincaillerie/roto-nx-ksr/champs-application/soufflet-cotes-ouverture.png)

Positions de palier et angles d'ouverture par tranche de hauteur de feuillure de vantail, une
ligne par tranche de HFF [F]. Le type (1 ou 2) est celui du tableau, sans autre précision sur la
page [1 p. 28].

| HFF (mm) | Type | Position palier de vantail (mm) | Position palier de dormant (mm) | Ouverture en position d'entrebâillement (mm) | Angle d'entrebâillement (°) | Angle de position de nettoyage (°) |
| --- | --- | --- | --- | --- | --- | --- |
| 290 à 400 | 1 | 250 | 45 | 180 à 245 | 33 | 90 |
| 401 à 560 | 1 | 280 | 75 | 205 à 275 | 27 | 67 |
| 561 à 700 | 2 | 525 | 170 | 225 à 277 | 22 | 88 |
| 701 à 850 | 2 | 575 | 220 | 244 à 292 | 19 | 72 |
| 851 à 1200 | 2 | 625 | 270 | 261 à 363 | 17 | 62 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 28)

Correspondance avec la légende du dessin : [A] position palier de vantail, [B] position palier de
dormant, [C] ouverture position d'entrebâillement, [D] angle d'ouverture position
d'entrebâillement, [E] angle d'ouverture position de nettoyage, [F] hauteur de feuillure de vantail
(HFF) [1 p. 28].

# Diagramme d'application, fenêtre confort

La fenêtre confort (pictogramme oscillo-battant) a son propre champ d'application, en sécurité de
base : LFF 520 à 1 400 mm, HFF 530 à 1 600 mm, 50 kg maximum (ligne « Fenêtre confort » du tableau
des champs P) [1 p. 29].

![Diagramme d'application Roto NX, fenêtre confort](/assets/quincaillerie/roto-nx-ksr/champs-application/diagramme-fenetre-confort.png)

Le diagramme porte la LFF en abscisse (520 à 1 400 mm, seule la graduation 1 000 est chiffrée) et
la HFF en ordonnée (530 à 1 600 mm), quadrillés tous les 100 mm. Quatre courbes : 50, 40, 30 et
20 kg/m². La seule zone légendée est le champ d'application non autorisé (blanc quadrillé), sous
une droite basse qui part d'environ 800 mm de LFF au bord bas et atteint environ 930 mm de HFF au
bord droit. Les courbes des 50, 40 et 30 kg/m² descendent du bord haut puis deviennent verticales
(vers 905, 1 005 et 1 150 mm de LFF) jusqu'à la droite basse ; celle des 20 kg/m² est presque
verticale, d'environ 1 370 mm de LFF en haut à environ 1 345 mm à sa rencontre avec la droite basse
[1 p. 29].

Relevé tous les 100 mm de LFF, précision de lecture ±25 mm ; « 1600 » est le bord haut du
diagramme, « 530 » son bord bas.

| LFF (mm) | HFF maxi 50 kg/m² (mm) | HFF maxi 40 kg/m² (mm) | HFF maxi 30 kg/m² (mm) | HFF maxi 20 kg/m² (mm) | HFF mini, droite basse (mm) |
| --- | --- | --- | --- | --- | --- |
| 600 | 1600 | 1600 | 1600 | 1600 | 530 |
| 700 | 1370 | 1600 | 1600 | 1600 | 530 |
| 800 | 1200 | 1480 | 1600 | 1600 | 530 |
| 900 | 1070 | 1300 | 1600 | 1600 | 600 |
| 1000 | hors champ | 1180 | 1545 | 1600 | 665 |
| 1100 | hors champ | hors champ | 1420 | 1600 | 730 |
| 1200 | hors champ | hors champ | hors champ | 1600 | 795 |
| 1300 | hors champ | hors champ | hors champ | 1600 | 860 |
| 1400 | hors champ | hors champ | hors champ | hors champ | 930 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 29)

# Cotes des champs d'application, côté paumelles Designo II

Le côté paumelles Designo (Designo II) a ses propres champs d'application, un par diagramme, en
mm et en kg [1 p. 30-32] :

| Configuration Designo II | LFF mini (mm) | LFF maxi (mm) | HFF mini (mm) | HFF maxi (mm) | PV maxi (kg) |
| --- | --- | --- | --- | --- | --- |
| Fenêtre à la française et oscillo-battante, sans report de charge, 80 kg | 330 | 1400 | 280 | 2600 | 80 |
| Fenêtre à la française et oscillo-battante, sans report de charge, 100 kg | 600 | 1400 | 280 | 2600 | 100 |
| Oscillo-battante avec report de charge, 80 à 150 kg | 800 | 1400 | 1000 | 2600 | 150 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 30-32)

Les deux premiers diagrammes portent les pictogrammes ouvrant à la française, oscillo-battant et
deux vantaux (« Fenêtre OF & OB ») ; celui du report de charge ne porte que l'oscillo-battant et
les deux vantaux (« Oscillo-battant avec report de charge de 80 – 150 kg »), et son tableau écrit
le poids « max. 80 – 150 kg » [1 p. 30-32].

**Attention : si le poids de vantail est supérieur à 130 kg, réduire l'ouverture du compas à
80 mm** [1 p. 32].

Le montage du report de charge est décrit dans
[Report de charge ROTO NX](/procedures/report-de-charge-roto-nx.md).

Les trois diagrammes Designo II se lisent comme ceux du côté P : LFF en abscisse, HFF en ordonnée,
quadrillage de 100 mm, une courbe rouge par poids de vitrage, zone blanche quadrillée non
autorisée, zone gris foncé « 2ᵉ compas nécessaire » de 1 200 à 1 400 mm de LFF.

## Designo II, OF et OB sans report de charge, 80 kg

![Diagramme d'application Designo II, OF et OB, 80 kg](/assets/quincaillerie/roto-nx-ksr/champs-application/diagramme-designo-of-ob-80-kg.png)

Quatre courbes : 50, 40, 30 et 20 kg/m². Celle des 50 kg/m² devient verticale vers 1 315 mm de LFF
(HFF environ 1 215) ; les trois autres atteignent le bord droit. Au-dessus de la courbe des
20 kg/m², en haut à droite, le champ est non autorisé. Relevé tous les 100 mm de LFF, précision de
lecture ±25 mm, « 2600 » étant le bord haut [1 p. 30].

| LFF (mm) | HFF maxi 50 kg/m² (mm) | HFF maxi 40 kg/m² (mm) | HFF maxi 30 kg/m² (mm) | HFF maxi 20 kg/m² (mm) | HFF mini, droite basse (mm) | Zone |
| --- | --- | --- | --- | --- | --- | --- |
| 400 | 2600 | 2600 | 2600 | 2600 | bord bas | - |
| 500 | 2600 | 2600 | 2600 | 2600 | 325 | - |
| 600 | 2600 | 2600 | 2600 | 2600 | 395 | - |
| 700 | 2290 | 2600 | 2600 | 2600 | 465 | - |
| 800 | 2000 | 2510 | 2600 | 2600 | 530 | - |
| 900 | 1775 | 2215 | 2600 | 2600 | 595 | - |
| 1000 | 1595 | 1990 | 2600 | 2600 | 665 | - |
| 1100 | 1450 | 1815 | 2420 | 2600 | 735 | - |
| 1200 | 1330 | 1670 | 2215 | 2600 | 800 | 2ᵉ compas nécessaire |
| 1300 | 1215 | 1535 | 2050 | 2600 | 870 | 2ᵉ compas nécessaire |
| 1400 | hors champ | 1430 | 1905 | 2400 | 930 | 2ᵉ compas nécessaire |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 30)

La courbe des 20 kg/m² part du bord haut vers 1 295 mm de LFF [1 p. 30].

## Designo II, OF et OB sans report de charge, 100 kg

![Diagramme d'application Designo II, OF et OB, 100 kg](/assets/quincaillerie/roto-nx-ksr/champs-application/diagramme-designo-of-ob-100-kg.png)

Le champ commence à 600 mm de LFF (bord gauche vertical, de HFF environ 400 jusqu'au bord haut) ;
la droite basse va de (LFF 600, HFF environ 400) à (LFF 1 400, HFF environ 925), en segment droit.
En haut à droite, un segment noir borne le champ de (LFF environ 1 290, HFF 2 600) à (LFF 1 400,
HFF environ 2 410). Trois courbes, chacune devenant verticale jusqu'à la droite basse [1 p. 31] :

| Courbe | Départ au bord haut, LFF (mm) | Points relevés sur la partie courbe (LFF × HFF, mm) | Verticale à LFF (mm) | Haut de la verticale, HFF (mm) | Pied de la verticale sur la droite basse, HFF (mm) |
| --- | --- | --- | --- | --- | --- |
| 50 kg/m² | 770 | 800 × 2500 ; 900 × 2225 ; 1000 × 2000 | 1055 | 1900 | 700 |
| 40 kg/m² | 960 | 1000 × 2500 ; 1100 × 2310 | 1180 | 2120 | 790 |
| 30 kg/m² | 1290 | - | 1365 | 2450 | 910 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 31)

Valeurs relevées à ±25 mm.

## Designo II, oscillo-battant avec report de charge, 80 à 150 kg

![Diagramme d'application Designo II avec report de charge](/assets/quincaillerie/roto-nx-ksr/champs-application/diagramme-designo-ob-report-de-charge.png)

Le champ est un polygone : bord gauche à 800 mm de LFF (de HFF 1 000 au bord haut 2 600), bord bas à
1 000 mm de HFF (de LFF 800 à environ 950), puis une droite qui monte jusqu'à environ 1 470 mm de
HFF au bord droit (LFF 1 400) ; en haut à droite, un segment borne le champ de (LFF environ 1 300,
HFF 2 600) à (LFF 1 400, HFF environ 2 410). Quatre courbes, chacune devenant verticale jusqu'à la
limite basse [1 p. 32] :

| Courbe | Départ de la partie courbe (LFF × HFF, mm) | Verticale à LFF (mm) | Haut de la verticale, HFF (mm) | Pied de la verticale, HFF (mm) |
| --- | --- | --- | --- | --- |
| 80 kg/m² | 800 × 2340 (bord gauche), passe par 900 × 2110 | 970 | 1935 | 1015 |
| 60 kg/m² | 960 × 2600 (bord haut), passe par 1000 × 2500 | 1120 | 2240 | 1180 |
| 50 kg/m² | 1160 × 2600 (bord haut) | 1230 | 2450 | 1290 |
| 40 kg/m² | - | 1370 | 2450 | 1450 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 32)

Valeurs relevées à ±25 mm. La zone « 2ᵉ compas nécessaire » couvre 1 200 à 1 400 mm de LFF.

# Diagramme d'application, ouvrant à soufflet, côté paumelles Designo

Champ d'application de l'ouvrant à soufflet côté paumelles Designo : LFF 450 à 1 400 mm, HFF 370 à
1 200 mm, poids de vantail 80 kg maximum ; son diagramme et ses zones de compas sont sur
[Configurations Roto NX KSR — ouvrant à soufflet](/quincaillerie/roto-nx-ksr-soufflet.md)
[1 p. 33].

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

La classe CDR 3 ne figure que dans le catalogue ; le manuel de montage s'arrête à CDR 2 / CDR 2 N.

**Les deux documents ROTO ne donnent pas les mêmes bornes** sur quatre valeurs — hauteur de
feuillure minimale, largeur maximale en CDR 1 N, hauteur maximale en CDR 1 N et en CDR 2. Entrée
**CTR-18** du registre [Contradictions entre sources](/anomalies/contradictions-entre-sources.md) ;
en attendant l'arbitrage, la valeur retenue pour un chiffrage est la borne la plus basse des deux
documents.

Le tableau de la version 150 kg est **imprimé en allemand** dans ce catalogue français, et il y
désigne les classes par « RC » là où les pages françaises écrivent « CDR » — entrée **INC-13** du
registre [Incohérences internes](/anomalies/incoherences-internes.md).

# Cotes de force de traction par poids de vantail

Pour la fixation des pièces de ferrure porteuses affectant la sécurité, telles que palier de compas
et palier d'angle, les forces de traction doivent être orientées perpendiculairement au plan de
l'ouvrant selon le tableau ci-dessous (valeurs des forces de traction en fonction des poids de
vantail par le TBDK) [1 p. 21-22]. La *force de traction* est l'effort, en newtons (N), que la
fixation doit supporter sans s'arracher.

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

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 21-22)

Les valeurs indiquées sont données par référence au palier de compas. Elles sont également valables
pour les paliers d'angle lorsque la fixation est réalisée selon le palier de compas. **Respecter la
directive TBDK pour les forces de traction en fonction du poids de vantail** ; autres informations
sur www.beschlagindustrie.de [1 p. 22]. Le catalogue de juin 2023 donne les mêmes dix valeurs
[2 p. 34]. La TBDK borne aussi les poids d'ouvrant admissibles du système profine, voir
[Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md).

# Cotes de dimensionnement des profilés

Le dessin désigne les éléments d'une fenêtre, en coupe horizontale au droit de la ferrure : le
dormant [5] en rose saumon, l'ouvrant [6] en gris clair, la parclose [7], la rainure de vantail [8]
(la rainure du profilé d'ouvrant où se loge la ferrure), et quatre cotes : la cote de l'axe [1], le
jeu en feuillure [2] (l'espace entre ouvrant et dormant), la largeur de recouvrement [3] (la largeur
dont l'ouvrant recouvre le dormant) et la hauteur de recouvrement [4] [1 p. 34].

![Désignations de l'élément de fenêtre et rainure de vantail](/assets/quincaillerie/roto-nx-ksr/champs-application/designations-element-de-fenetre.png)

Systèmes d'axe de ferrage admis par la Roto NX, en mm. La désignation se lit « jeu de joint /
largeur de recouvrement - axe de ferrage » [1 p. 34].

| Système | Axe de ferrage [1] (mm) | Jeu de joint [2] (mm) | Largeur de recouvrement [3] (mm) |
| --- | --- | --- | --- |
| 12/18-9 | 9 | 12 | 18 |
| 12/18-13 | 13 | 12 | 18 |
| 12/20-9 | 9 | 12 | 20 |
| 12/20-13 | 13 | 12 | 20 |
| 12/21-13 | 13 | 12 | 21 |
| 12/22-13 | 13 | 12 | 22 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 34)

Le jeu de joint est écrit « 12 mm - 0,5 mm / + 1,5 mm » dans une cellule commune aux six systèmes
[1 p. 34].

Sous le dessin d'ensemble, trois détails agrandis donnent les cotes recommandées pour le
dimensionnement des profilés [1 p. 34] :

| Détail | Cotes portées (mm) |
| --- | --- |
| Rainure, premier détail | 16,2 (+0,2 / −0,1) ; 12,2 (+0,2 / −0,1) ; 9,3 ±0,2 ; 2,2 ±0,2 |
| Rainure, second détail | 16,2 (+0,2 / −0,1) ; 12,2 (+0,2 / −0,1) ; 9,3 ±0,2 ; 2,2 ±0,2 ; 2,8 |
| Feuillure du dormant, détail de la ferrure | angle ≤ 5° ; 3 (+0,5) |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 34)

Sur les deux détails de rainure, 16,2 et 12,2 sont cotées horizontalement, 9,3, 2,2 et 2,8
verticalement. Le manuel ne nomme pas les deux variantes de rainure.

# Cotes de tolérance de châssis fixe, côté paumelles P

L'*encombrement de la paumelle* est la place que prennent les paumelles (charnières) entre
l'ouvrant et le dormant. Tolérance de châssis fixe pour 20 mm de largeur de recouvrement, en mm ;
[A] est la tolérance de châssis fixe, [B] la hauteur de recouvrement, [C] et [D] les jeux en haut et
en bas du vantail [1 p. 36].

![Encombrement de la paumelle, côté paumelles P](/assets/quincaillerie/roto-nx-ksr/champs-application/encombrement-paumelle-cote-p.png)

| Poids du vantail (kg) | Angle d'ouverture | Tolérance de châssis fixe [A] (mm) | Hauteur de recouvrement mini [B] (mm) | En haut [C] (mm) | En bas [D] (mm) |
| --- | --- | --- | --- | --- | --- |
| 130 | env. 180° | 21,0 | 16 | 1,0 | 8 |
| 150 | env. 180° | 26,5 | 16 | 1,0 | 8 |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 36)

**Tolérance, caches compris.** Angle d'ouverture jusqu'à 20 mm de hauteur de recouvrement
[1 p. 36]. L'angle « env. 180° » porte un renvoi « [2] » dont la note n'est imprimée nulle part sur
la page (entrée **INC-193**).

Les deux vues de face (poids de vantail 130 kg et 150 kg) montrent le vantail gris sur le dormant
rose saumon, paumelles à droite : la cote horizontale portée au droit de la paumelle haute vaut
21 mm en 130 kg et 26,5 mm en 150 kg, celle portée au droit de la paumelle basse 19 mm dans les
deux cas, et une cote verticale de 8 mm est portée sous le vantail [1 p. 36].

# Fixation d'une fenêtre de sécurité

La *fenêtre de sécurité* est une fenêtre classée retard d'effraction (classes CDR). Le manuel
suggère sa fixation dans la maçonnerie par des blocs d'écartement [1 p. 37].

![Suggestion de fixation pour fenêtre de sécurité](/assets/quincaillerie/roto-nx-ksr/champs-application/fixation-fenetre-de-securite.png)

Le dessin montre la fenêtre vue de face dans l'ouvrage de maçonnerie [1], le dormant [3] en rose
saumon et les blocs d'écartement [2] en noir, entre maçonnerie et dormant.

| Élément | Valeur |
| --- | --- |
| Dispositif | bloc d'écartement [2], entre l'ouvrage de maçonnerie [1] et le dormant [3] |
| Emplacement | installer un bloc d'écartement dans la zone des vissages de gâche de sécurité |
| Distance aux angles (montants et traverses) | ~150 mm |
| Entraxe entre blocs sur les montants | ~400 mm |

(schéma: raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf, p. 37)

La consigne est imprimée « dans la zone des vissages de gâche de sécurités de sécurité ». Le texte
qui suit est reproduit tel quel [1 p. 37] :

> « Les fenêtre à retard d'effraction au sens de la DIN EN 1627–1630 ne doivent être désignées
> comme telles qu'uniquement lorsque le montage a été effectué en tous points selon la norme
> prescrite. »

# Ce que la source ne donne pas

- **quel côté paumelles PROFERM emploie** — P ou Designo II. Entrée **VER-34** du registre
  [Informations à vérifier](/anomalies/informations-a-verifier.md)
- **le poids des profilés** : le manuel ne donne que la conversion de l'épaisseur de verre en
  poids de vitrage

# Citations

[1] Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf.
IMO_180_NX_FR_v2, novembre 2022 —
`raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, p. 9, 21 à 37 (numérotation du PDF)

[2] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 —
`raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 34 à 36

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
