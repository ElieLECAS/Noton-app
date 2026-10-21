---
type: Quincaillerie
title: 'Champs d''application Roto NX : ferrure soufflet, compas d''arrêt et cotes d''ouverture'
description: Diagramme d'application de la ferrure soufflet Roto NX (LFF 310 à 2400, HFF 290 à 1200, 80 kg), nombre et position des compas d'arrêt par format de vantail, cotes d'ouverture et angles par hauteur de feuillure.
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

Ferrure soufflet Roto NX côté paumelles P, du manuel de montage KSR et du catalogue Roto NX pour profils PVC. Les bornes communes sont sur [Champs d'application Roto NX : cotes côté paumelles P et ferrure OB rectangulaire](/quincaillerie/roto-nx-champs-application.md).

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

Le catalogue Roto NX pour profils PVC de juin 2023 imprime le même diagramme sous le titre
« Ferrure soufflet fenêtre rectangulaire » (relu à 400 dpi : cotes 621 et 560, plages [A] et [B],
limite oblique jusqu'à environ 800 mm de HFF au bord droit, mêmes zones), avec la même légende,
où le compas est appelé **compas soufflet** : « 2 compas soufflet latéralement », « 1 compas
soufflet en haut ou 2 compas soufflet latéralement », « 2 compas soufflet en haut ou 2 compas
soufflet latéralement ». Son tableau donne les mêmes bornes (LFF 310 – 2 400 mm, HFF
290 – 1 200 mm, poids d'ouvrant max. 80 kg, sécurité de base), le même encadré sur le compas
d'entrebâillement et de nettoyage, et imprime la note du renvoi [1] de la LFF : « LFF 310 –
449 mm uniquement avec crémone verrou » (**INC-193**) [2 p. 42]. La légende des plages y porte
aussi deux fois « [A] » (**INC-192**).

![Diagramme d'application Roto NX, soufflet rectangulaire, catalogue 2023](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-soufflet-rectangulaire.png)

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 42)

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

Le catalogue Roto NX pour profils PVC de juin 2023 imprime la même planche (relue à 500 dpi :
mêmes vantaux dessinés, mêmes symboles, mêmes repères [A] à [E] et mêmes teintes), avec une
légende dont les trois symboles sont ceux de la planche (**INC-191**) [2 p. 43] :

| Symbole de la planche | Position |
| --- | --- |
| ● point noir (sur les côtés du vantail) | Position possible compas soufflet jusqu'à 80 kg |
| ○ cercle (en haut du vantail) | Position alternative compas soufflet jusqu'à 80 kg |
| △ triangle (en haut du vantail) | Position alternative compas soufflet jusqu'à 60 kg |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 43)

Les notes [A] à [E] y sont les mêmes, le compas étant appelé compas soufflet (« [A] au-delà de
501 mm, compas soufflet possible en haut uniquement avec crémone verrou »), sauf la note [B] :
« au-delà de 621 mm, compas soufflet en haut possible avec crémone dans le chant et crémone
OB », là où le manuel écrit « compas pêne demi-tour » — entrée **CTR-107**. L'encadré INFO est le
même : « Utilisation de compas soufflet latéral en liaison avec verrouilleur VM 200 non
possible » [2 p. 43].

![Positions des compas soufflet, catalogue 2023](/assets/quincaillerie/roto-nx-catalogue/diagrammes/soufflet-positions-compas.png)

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

Le catalogue Roto NX pour profils PVC de juin 2023 imprime le même dessin et le même tableau (cinq
tranches de HFF, mêmes types et mêmes valeurs) ; sur son dessin, les cotes portent leurs lettres
[A] à [F], et le palier de dormant est coté : **42** mm entre ses deux vis extrêmes, **28,5** mm
entre l'axe de son trou central et celui de sa vis basse, la cote [B] partant de cet axe central.
Sa légende écrit « [A] Position palier d'ouvrant » et « [F] Hauteur fond de feuillure d'ouvrant
(HFF) » [2 p. 44].

![Cotes d'ouverture de la ferrure soufflet, catalogue 2023](/assets/quincaillerie/roto-nx-catalogue/diagrammes/soufflet-cotes-ouverture.png)

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 44)

# Citations

[1] Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf.
IMO_180_NX_FR_v2, novembre 2022 —
`raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, p. 9, 21 à 37 (numérotation du PDF)

[2] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 —
`raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 14, 34 à 52 et 278

# Voir aussi

Cette page est une partie d'un sujet qui en compte 6 ; les autres :

- [Champs d'application Roto NX : cotes côté paumelles P et ferrure OB rectangulaire](/quincaillerie/roto-nx-champs-application.md)
- [Champs d'application Roto NX : ferrure OB de fenêtre inclinée et de fenêtre cintrée](/quincaillerie/roto-nx-champs-application-ob-inclinee-cintree.md)
- [Champs d'application Roto NX : paumelle à soufflet sur recouvrement et fenêtre confort](/quincaillerie/roto-nx-champs-application-paumelle-soufflet-confort.md)
- [Champs d'application Roto NX côté paumelles Designo II : 80, 100 et 150 kg](/quincaillerie/roto-nx-champs-application-designo.md)
- [Roto NX : bornes catalogue 2023 (CDR 3), force de traction, dimensionnement, tolérance](/quincaillerie/roto-nx-champs-application-catalogue-2023-forces-dimensionnement.md)
