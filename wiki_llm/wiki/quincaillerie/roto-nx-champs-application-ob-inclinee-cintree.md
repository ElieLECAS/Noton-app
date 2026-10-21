---
type: Quincaillerie
title: 'Champs d''application Roto NX : ferrure OB de fenêtre inclinée et de fenêtre cintrée'
description: Sommets du champ admis de la ferrure oscillo-battante Roto NX pour fenêtre inclinée à 25°, 40°, 50° et −15°, et courbes par poids de vitrage (20, 30, 40 kg/m²) de la fenêtre cintrée, avec la règle du rayon.
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

Diagrammes d'application de la ferrure OB Roto NX, côté paumelles P, pour les fenêtres inclinées et cintrées, du manuel de montage KSR et du catalogue Roto NX pour profils PVC. Les bornes communes et le diagramme rectangulaire sont sur [Champs d'application Roto NX : cotes côté paumelles P et ferrure OB rectangulaire](/quincaillerie/roto-nx-champs-application.md).

# Diagrammes d'application, ferrure OB, fenêtre inclinée

La ferrure OB (oscillo-battante) de fenêtre inclinée, côté paumelles P, a un diagramme par angle
d'inclinaison : 25°, 40°, 50° et −15°. Pour les quatre angles, le tableau « Champ
d'application » renvoie au diagramme pour la LFF et la HFF (« voir diagramme ») et borne le
poids d'ouvrant à **80 kg maximum**, en sécurité de base ; chaque page rappelle la conversion
« 1 mm/m² épaisseur de vitre ≙ 2,5 kg » et de respecter la directive TBDK pour les forces de
traction [2 p. 37-40].

Chaque diagramme porte la LFF en abscisse (de 0 à 1 300 mm, graduée 500 et 1 000, quadrillée tous
les 100 mm) et la HFF en ordonnée (de 250 ou 350 mm à 2 400 mm, graduée tous les 500 mm). Le
champ admis est le polygone gris, borné par des segments rouges ; le blanc quadrillé est le
**champ d'application non autorisé**. Deux repères encadrent le diagramme : **[A] = côté
crémone** en haut à gauche, **[B] = coté axe** en haut à droite. Un point rouge marque le
sommet bas du polygone, sur son bord gauche, et deux flèches rouges en partent, l'une vers le
haut le long du bord gauche, l'autre le long du bord bas. Aucune courbe par poids de vitrage n'est
tracée. Le catalogue ne dit ni ce que mesure l'angle d'inclinaison, ni ce que signifient sur le
diagramme les repères [A] et [B], le point et les flèches : entrée **VER-121** [2 p. 37-40].

Les sommets de chaque polygone sont relevés ci-dessous ; « chiffré » signifie que la valeur est
imprimée sur le diagramme, « lu » qu'elle est lue sur le quadrillage (précision ±25 mm). Les côtés
du polygone sont des segments droits.

## Angle d'inclinaison 25°

![Diagramme d'application Roto NX, OB fenêtre inclinée 25°](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-ob-inclinee-25.png)

| Sommet du champ admis | LFF (mm) | HFF (mm) | Origine des valeurs |
| --- | --- | --- | --- |
| bas gauche, sur le point rouge | 265 | 470 | LFF chiffrée, HFF lue sur le bord bas |
| haut gauche | 265 | 2400 | LFF chiffrée, HFF lue (bord haut) |
| haut droit | 1000 | 2400 | lus |
| droit, fin du côté oblique | 1300 | 1600 | LFF lue (bord droit), HFF chiffrée |
| bas droit | 1300 | 960 | LFF lue (bord droit), HFF chiffrée |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 37)

Le bord bas du champ est le segment rouge qui part de (LFF 0, HFF 350, chiffrée) et monte jusqu'au
bas droit (1 300 × 960). Un second segment rouge, hors du champ, part du bord gauche du
diagramme à 2 280 mm de HFF (chiffrée) et rejoint le sommet haut gauche (265 × 2 400) [2 p. 37].

## Angle d'inclinaison 40°

![Diagramme d'application Roto NX, OB fenêtre inclinée 40°](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-ob-inclinee-40.png)

| Sommet du champ admis | LFF (mm) | HFF (mm) | Origine des valeurs |
| --- | --- | --- | --- |
| bas gauche, sur le point rouge | 225 | 535 | LFF chiffrée, HFF lue sur le bord bas |
| haut gauche | 225 | 2400 | LFF chiffrée, HFF lue (bord haut) |
| haut droit | 1000 | 2400 | lus |
| droit, fin du côté oblique | 1300 | 1600 | LFF lue (bord droit), HFF chiffrée |
| bas droit | 1300 | 1440 | LFF lue (bord droit), HFF chiffrée |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 38)

Le bord bas part de (LFF 0, HFF 350, chiffrée) et monte jusqu'au bas droit (1 300 × 1 440). Le
segment rouge hors du champ part du bord gauche à 2 215 mm de HFF (chiffrée) et rejoint le sommet
haut gauche (225 × 2 400) [2 p. 38].

## Angle d'inclinaison 50°

![Diagramme d'application Roto NX, OB fenêtre inclinée 50°](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-ob-inclinee-50.png)

| Sommet du champ admis | LFF (mm) | HFF (mm) | Origine des valeurs |
| --- | --- | --- | --- |
| bas gauche, sur le point rouge | 200 | 585 | LFF chiffrée, HFF lue sur le bord bas |
| haut gauche | 200 | 2400 | LFF chiffrée, HFF lue (bord haut) |
| haut droit | 1000 | 2400 | lus |
| pointe droite | 1230 | 1815 | chiffrés |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 39)

Le champ n'atteint pas le bord droit du diagramme (1 300 mm) : le côté oblique descend du haut
droit jusqu'à la pointe (1 230 × 1 815), d'où le bord bas redescend jusqu'à (LFF 0, HFF 350,
chiffrée). Le segment rouge hors du champ part du bord gauche à 2 180 mm de HFF (chiffrée) et
rejoint le sommet haut gauche (200 × 2 400) [2 p. 39].

## Angle d'inclinaison −15°

![Diagramme d'application Roto NX, OB fenêtre inclinée −15°](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-ob-inclinee-moins-15.png)

| Sommet du champ admis | LFF (mm) | HFF (mm) | Origine des valeurs |
| --- | --- | --- | --- |
| bas gauche, sur le point rouge | 280 | 280 | LFF chiffrée, HFF lue sur le bord bas |
| haut gauche | 280 | 2325 | LFF chiffrée, HFF lue, égale au 2 325 imprimé dans le coin haut gauche |
| haut, cassure du bord supérieur | 1120 | 2100 | chiffrés |
| droit, fin du côté oblique | 1300 | 1600 | LFF lue (bord droit), HFF chiffrée |
| bas droit | 1300 | 250 | LFF lue (bord droit), HFF chiffrée (bord bas) |

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 40)

Sur ce diagramme, le bord supérieur du champ descend : il part du coin haut gauche du diagramme
(LFF 0, HFF 2 400 lue), passe par le sommet haut gauche (280 × 2 325) et la cassure
(1 120 × 2 100), puis le côté oblique descend jusqu'à 1 600 mm de HFF au bord droit. Le bord bas
part de (LFF 0, HFF 350, chiffrée), descend jusqu'au point rouge (280 × 280) et atteint le bord
bas du diagramme, 250 mm de HFF, vers 400 mm de LFF (lu) ; de là, le champ est borné par le bord
bas jusqu'à 1 300 mm. [2 p. 40]

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

Le catalogue Roto NX pour profils PVC de juin 2023 imprime ce diagramme et son champ
d'application **en allemand**, sous le titre « Drehkipp-Beschlag Rundbogenfenster » (ferrure
oscillo-battante de fenêtre cintrée) : FFB (largeur de fond de feuillure) 400 – 1 300 mm, FFH
(hauteur de fond de feuillure) 500 – 1 900 mm, FG (poids de vantail) max. 80 kg, en
« Grundsicherheit » (sécurité de base) ; la note « Der Radius (R) des Rundbogenfensters muss der
Hälfte der FFB entsprechen » est la règle du rayon ci-dessus. Relu à 450 dpi, le diagramme donne
les mêmes sommets que le tableau (850, 950 et 900 mm pour les 40 kg/m² ; 1 100, 1 100 et 1 000 mm
pour les 30 kg/m² ; 1 300, 1 300 et 1 200 mm pour les 20 kg/m², aux hauteurs 1 900, 1 600 et
950 mm) et les mêmes quatre zones, légendées « unzulässiger Anwendungsbereich »,
« Zweitschere erforderlich » (gris clair), « Zweitschere möglich aber nicht erforderlich » (gris
foncé) et « Zweitschere nicht möglich » (hachures) [2 p. 41]. La langue : entrée **INC-13**.

![Diagramme d'application Roto NX, OB cintré, catalogue 2023](/assets/quincaillerie/roto-nx-catalogue/diagrammes/diagramme-ob-cintre.png)

(schéma: raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf, p. 41)

# Citations

[1] Roto NX KSR, instructions de montage fenêtres et portes-fenêtres en PVC, réf.
IMO_180_NX_FR_v2, novembre 2022 —
`raw/roto-nx-ksr-montage-pvc-imo-180-2022-11.pdf`, p. 9, 21 à 37 (numérotation du PDF)

[2] Roto NX, catalogue pour profils PVC, réf. CTL_105_FR_v5, juin 2023 —
`raw/roto-nx-catalogue-pvc-ctl-105-2023-06.pdf`, p. 14, 34 à 52 et 278

# Voir aussi

Cette page est une partie d'un sujet qui en compte 6 ; les autres :

- [Champs d'application Roto NX : cotes côté paumelles P et ferrure OB rectangulaire](/quincaillerie/roto-nx-champs-application.md)
- [Champs d'application Roto NX : ferrure soufflet, compas d'arrêt et cotes d'ouverture](/quincaillerie/roto-nx-champs-application-soufflet.md)
- [Champs d'application Roto NX : paumelle à soufflet sur recouvrement et fenêtre confort](/quincaillerie/roto-nx-champs-application-paumelle-soufflet-confort.md)
- [Champs d'application Roto NX côté paumelles Designo II : 80, 100 et 150 kg](/quincaillerie/roto-nx-champs-application-designo.md)
- [Roto NX : bornes catalogue 2023 (CDR 3), force de traction, dimensionnement, tolérance](/quincaillerie/roto-nx-champs-application-catalogue-2023-forces-dimensionnement.md)
