---
type: Quincaillerie
title: 'Champs d''application Roto NX : paumelle à soufflet sur recouvrement et fenêtre confort'
description: Largeurs, hauteurs et poids d'ouvrant admis pour la paumelle à soufflet sur recouvrement d'ouvrant pivotant ou basculant, et courbes par poids de vitrage de la fenêtre confort (LFF 520 à 1400, 50 kg).
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

Diagrammes d'application Roto NX côté paumelles P, du catalogue Roto NX pour profils PVC et du manuel de montage KSR. Les bornes communes sont sur [Champs d'application Roto NX : cotes côté paumelles P et ferrure OB rectangulaire](/quincaillerie/roto-nx-champs-application.md).

# Diagramme d'application, paumelle à soufflet pour recouvrement d'ouvrant, ouvrants pivotants

La *paumelle à soufflet pour recouvrement d'ouvrant* est la paumelle de la ferrure soufflet côté
paumelles P, posée sur le recouvrement de l'ouvrant (la partie de l'ouvrant qui recouvre le
dormant, voir le [glossaire](/reference/glossaire.md)). Section 2.2.5 du catalogue, intitulée
« Bandeau de recouvrement d'ouvrant pivotant fenêtre rectangulaire » : **Paumelle à soufflet pour
recouvrement d'ouvrant oscillo-battant P dans ouvrants pivotants uniquement en association avec
pivot d'angle / palier d'angle** [2 p. 45].

| Grandeur | Sécurité de base |
| --- | --- |
| Largeur de feuillure d'ouvrant (LFO) | 290 – 1 200 mm |
| Hauteur de feuillure d'ouvrant (HFO) | 280 – 2 400 mm |
| Poids d'ouvrant (PV) | max. 80 kg |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 45)

![Diagramme d'application, paumelle à soufflet pour recouvrement d'ouvrant, ouvrants pivotants](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-soufflet-recouvrement-ouvrant-pivotant.png)

Le diagramme porte la largeur en abscisse (290 à 1 200 mm, graduée 500 et 1 000, quadrillée tous
les 100 mm) et la hauteur en ordonnée (280 à 2 400 mm, graduée 500, 1 000, 1 500 et 2 000,
quadrillée tous les 100 mm). Trois courbes rouges, 50, 40 et 30 kg/m², bornent à gauche et en
dessous d'elles les formats admis pour ce poids de vitrage ; une droite noire borne le champ par
le bas. La seule zone légendée est le **champ d'application non autorisé** (blanc quadrillé), au
bas à droite sous la droite noire et en haut à droite au-delà de la courbe des 30 kg/m². La
conversion est écrite « 1 mm/m² d'épaisseur de vitre = 2,5 kg » [2 p. 45].

Relevé à 500 dpi tous les 100 mm de largeur, précision de lecture ±25 mm ; « 2400 » est le bord
haut du diagramme ; la droite basse est un segment droit, lu à chaque graduation. Les courbes des 50 et 40 kg/m² sont brisées aux points
(700 × 2 000), (800 × 1 800) et (900 × 1 600) pour la première, (800 × 2 300), (900 × 2 000) et
(1 000 × 1 800) pour la seconde.

| Largeur LFO (mm) | Hauteur maxi 50 kg/m² (mm) | Hauteur maxi 40 kg/m² (mm) | Hauteur maxi 30 kg/m² (mm) | Hauteur mini, droite basse (mm) |
| --- | --- | --- | --- | --- |
| 300 | 2400 | 2400 | 2400 | bord bas |
| 400 | 2400 | 2400 | 2400 | bord bas |
| 500 | 2400 | 2400 | 2400 | bord bas |
| 600 | 2400 | 2400 | 2400 | 280 |
| 700 | 2000 | 2400 | 2400 | 360 |
| 800 | 1800 | 2300 | 2400 | 445 |
| 900 | 1600 | 2000 | 2400 | 525 |
| 1000 | 1440 | 1800 | 2400 | 610 |
| 1100 | 1310 | 1650 | 2200 | 690 |
| 1200 | 1200 | 1500 | 2000 | 770 |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 45)

La courbe des 50 kg/m² part du bord haut à 600 mm de largeur, celle des 40 kg/m² vers 770 mm,
celle des 30 kg/m² à 1 000 mm ; la droite basse part du bord bas (280 mm) à 600 mm de largeur
[2 p. 45].

# Diagramme d'application, paumelle à soufflet pour recouvrement d'ouvrant basculant

Section 2.2.6 du catalogue : paumelle à soufflet pour recouvrement d'ouvrant basculant (un ouvrant
qui bascule seulement, sans ouverture à la française), fenêtre rectangulaire, côté paumelles P
[2 p. 46].

| Grandeur | Sécurité de base |
| --- | --- |
| Largeur fond de feuillure d'ouvrant (LFF) | 180 – 1 800 mm [2] |
| Hauteur fond de feuillure d'ouvrant (HFF) | 290 – 1 200 mm |
| Poids d'ouvrant (PO) | 60 ou 80 kg max. |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 46)

Le renvoi [2] de la LFF a sa note en pied de page : **LFF 180 – 449 mm uniquement avec crémone
verrou**. Sous le diagramme, deux plages de largeur : [A] jusqu'à 1 200 mm, portant « ≤ 60 kg »,
et [B] de 1 200 à 1 800 mm, portant « ≤ 80 kg » ; la légende écrit ici **[A] = 2 paumelles au
minimum, [B] = 3 paumelles au minimum** [2 p. 46].

![Diagramme d'application, paumelle à soufflet pour recouvrement d'ouvrant basculant](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-soufflet-recouvrement-ouvrant-basculant.png)

Le diagramme porte la LFF en abscisse (180 à 1 800 mm, graduée 500, 1 000 et 1 500, cote **621**
portée) et la HFF en ordonnée (290 à 1 200 mm, graduée 500 et 1 000, cote **560** portée),
quadrillées tous les 100 mm. Il se lit comme celui de la ferrure soufflet : la teinte dit quels
compas soufflet poser, les hachures quel compas d'entrebâillement et de nettoyage ajouter ; la
limite oblique part du coin haut à 1 200 mm de LFF (1 200 mm de HFF) et descend jusqu'à 1 000 mm
de HFF au bord droit (1 800 mm de LFF), relevée à 450 dpi à ±25 mm [2 p. 46].

<table>
<thead>
<tr><th>Zone du diagramme</th><th>LFF (mm)</th><th>HFF (mm)</th><th>Teinte : compas soufflet</th><th>Hachures : compas d'entrebâillement et de nettoyage</th></tr>
</thead>
<tbody>
<tr><td>gauche, haut</td><td>jusqu'à 621</td><td>560 à 1200</td><td>2 compas soufflet sur le côté</td><td>supplémentaire</td></tr>
<tr><td>gauche, bas</td><td>jusqu'à 621</td><td>jusqu'à 560</td><td>2 compas soufflet sur le côté</td><td>-</td></tr>
<tr><td>milieu, haut</td><td>621 à 1200</td><td>560 à 1200</td><td>1 compas soufflet en haut ou 2 compas soufflet latéralement</td><td>supplémentaire</td></tr>
<tr><td>milieu, bas</td><td>621 à 1200</td><td>jusqu'à 560</td><td>1 compas soufflet en haut ou 2 compas soufflet latéralement</td><td>additionnel en cas de compas soufflet en haut</td></tr>
<tr><td>droite, haut</td><td>1200 à 1800</td><td>560 jusqu'à la limite oblique</td><td>2 compas soufflet en haut ou 2 compas soufflet latéralement</td><td>supplémentaire</td></tr>
<tr><td>droite, bas</td><td>1200 à 1800</td><td>jusqu'à 560</td><td>2 compas soufflet en haut ou 2 compas soufflet latéralement</td><td>additionnel en cas de compas soufflet en haut</td></tr>
<tr><td>au-dessus de la limite oblique</td><td>1200 à 1800</td><td>de la limite oblique à 1200</td><td>champ d'application non autorisé</td><td>-</td></tr>
</tbody>
</table>

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 46)

**Compas d'entrebâillement et de nettoyage** : recommandé ; nécessaire en cas d'imposte (selon
RAL RG 607 / 12). Compas d'entrebâillement et de nettoyage jusqu'à 60 kg maxi [2 p. 46].

Le dessin des cotes d'ouverture et son tableau (positions des paliers, ouverture et angles par
tranche de HFF, types 1 et 2) sont imprimés à l'identique de ceux de la ferrure soufflet, section
*Cotes d'ouverture de la ferrure soufflet par hauteur de vantail* ci-dessus : mêmes cinq tranches,
mêmes valeurs, mêmes cotes 42 et 28,5 au palier de dormant [2 p. 47].

![Cotes d'ouverture, paumelle à soufflet pour recouvrement d'ouvrant basculant](/assets/quincaillerie/roto-nx-catalogue/diagrammes/soufflet-recouvrement-ouvrant-basculant-cotes-ouverture.png)

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 47)

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

Le catalogue Roto NX pour profils PVC de juin 2023 (section 2.2.7 « Fenêtre confort ») imprime le
même diagramme et les mêmes bornes (LFF 520 – 1 400 mm, HFF 530 – 1 600 mm, poids d'ouvrant
max. 50 kg, sécurité de base). Relu à 600 dpi, il donne les mêmes verticales aux écarts de lecture
près (905, 1 005 et 1 150 mm de LFF pour les 50, 40 et 30 kg/m², hauts de verticale vers 1 060,
1 175 et 1 345 mm de HFF ; courbe des 20 kg/m² d'environ 1 390 mm en haut à environ 1 365 mm en
bas) et la même droite basse, de 800 mm de LFF au bord bas jusqu'à 930 mm de HFF au bord droit
[2 p. 48].

![Diagramme d'application Roto NX, fenêtre confort, catalogue 2023](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-fenetre-confort.png)

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 48)

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
- [Champs d'application Roto NX : ferrure soufflet, compas d'arrêt et cotes d'ouverture](/quincaillerie/roto-nx-champs-application-soufflet.md)
- [Champs d'application Roto NX côté paumelles Designo II : 80, 100 et 150 kg](/quincaillerie/roto-nx-champs-application-designo.md)
- [Roto NX : bornes catalogue 2023 (CDR 3), force de traction, dimensionnement, tolérance](/quincaillerie/roto-nx-champs-application-catalogue-2023-forces-dimensionnement.md)
