---
type: Équipement
title: Volet roulant coffre demi-linteau BLOC LX
description: Volet roulant SOPROFEN pour coffres demi-linteaux en construction neuve — cotes de fabrication, compatibilités coffres préfabriqués (Terreal, Genova, Prefatec, Imerys, Stradal), lames PVC/alu, renforts d'inertie et motorisations.
tags: [equipement, volet-roulant, bloc-lx, demi-linteau, soprofen, neuf, isolation, renfort, somfy]
status: stable
sources:
  - resource: raw/moustiquaires/export_doc_133.zip
    id: soprofen-guide-technique-bloc-lx
    title: Guide technique volet roulant demi-linteau Bloc LX SOPROFEN
    last_modified: 2024-01-01
  - resource: raw/moustiquaires/export_doc_136(1).zip
    id: soprofen-guide-pose-bloc-lx-demi-linteau
    title: Guide de pose Bloc LX coffre demi-linteau SOPROFEN, réf. NO-BB-T-BLX-00-FR-202111
    last_modified: 2021-11-01
source_pages:
  - resource: raw/moustiquaires/export_doc_133.zip
    pages: 1-16
  - resource: raw/moustiquaires/export_doc_136(1).zip
    pages: 1-3
generated:
  by: process:gemini-coder
  at: 2026-09-21T14:05:00Z
---

# Présentation générale du système BLOC LX

Le volet roulant **BLOC LX** de [SOPROFEN](/fournisseurs/soprofen.md) est un système conçu pour l'intégration invisible sous coffre demi-linteau en construction neuve [1 p. 110]. Il est assemblé en atelier directement sur la traverse haute de la menuiserie extérieure (PVC ou aluminium), son axe étant disposé à l'horizontale et ses coulisses fixées verticalement entre les tapées d'isolation [1 p. 110, 2 p. 1].

### Périmètre des fournitures

* **Prestations SOPROFEN incluses** : tablier complet, coulisses pré-percées d'usine, consoles support d'axe, axe d'enroulement, manœuvre (manuelle ou motorisée), et demi-coffre (tiroir) avec ses embouts latéraux d'étanchéité [1 p. 110].
* **Prestations hors fourniture SOPROFEN** : coffre demi-linteau maçonné ou terre cuite, menuiserie extérieure, tapées d'isolation et trappe intérieure de visite [1 p. 110].

---

# Cotes de fabrication et règles d'encombrement

Les dimensions d'exécution d'un volet BLOC LX sont déterminées par deux cotes fondamentales exprimées en millimètres [1 p. 111] :

* **Largeur de fabrication $L$** : correspond à la **largeur de dos à dos des coulisses** [1 p. 111].
  * Largeur maximale admissible pour un tablier d'un seul tenant : **3 500 mm** en lames aluminium LA50 [1 p. 111].
  * Encombrement hors-tout du tiroir demi-coffre : $\text{Largeur demi-coffre} = L + 270\text{ mm}$ (débordement symétrique de $135\text{ mm}$ de chaque côté de la menuiserie, avec l'axe de coulisse positionné à $70\text{ mm}$ du bord extérieur de la console) [1 p. 111].
* **Hauteur de fabrication $H$** : correspond à la hauteur mesurée du **bas des coulisses au-dessus de la feuillure de la menuiserie**, alignée rigoureusement avec le dessus de la trappe du coffre [1 p. 111].
* **Contrainte d'alignement** : l'alignement entre le dessus de la trappe de visite et la feuillure du dormant est impératif afin de garantir que le tablier s'enroule librement sans frotter contre la partie supérieure du demi-linteau ni contre la sous-face [1 p. 111].
* **Réservation gros œuvre** : hauteur standard de réservation demi-linteau $H_{\text{demi-linteau}} = 292,5\text{ mm}$, offrant un passage vertical libre de 216 mm minimum sans profilé de renfort [1 p. 111, 124, 125].

---

# Compatibilité et cotes d'intégration selon les coffres demi-linteaux

L'encombrement du volet BLOC LX s'adapte aux coffres préfabriqués des principaux industriels du gros œuvre (brique terre cuite ou béton). Les cotes fonctionnelles d'intégration varient selon le fabricant et le type de manœuvre retenu (cotes en millimètres pour une isolation de 120 mm) [1 p. 124, 125] :

| Fabricant de coffre demi-linteau | Type de manœuvre | Déport support $Sd$ (mm) | Axe tige $DC$ (mm) | Débord intérieur $Ti$ (mm) | Débord extérieur $Te$ (mm) | Diamètre d'enroulement $\varnothing C$ (mm) | Débord console $A$ (mm) | Axe coulisse console $B$ (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TERREAL | Treuil | 148,5 | 15 | 50 | 150 | 190 | 135 | 70 |
| TERREAL | Moteur | - | - | 55 | 80 | 180 | 135 | 70 |
| GENOVA | Treuil | 148,5 | 15 | 50 | 150 | 190 | 135 | 70 |
| GENOVA | Moteur | - | - | 50 | 150 | 190 | 135 | 70 |
| PREFATEC | Treuil | 148,5 | 15 | 70 | 210 | 180 | 135 | 70 |
| PREFATEC | Moteur | - | - | 70 | 210 | 180 | 135 | 70 |
| IMERYS | Treuil | 148,5 | 15 | 55 | 80 | 180 | 135 | 70 |
| IMERYS | Moteur | - | - | 50 | 55 | 180 | 135 | 70 |
| STRADAL | Moteur uniquement | - | - | 100 | 100 | 165 | 135 | 70 |

(schéma: raw/moustiquaires/export_doc_133.zip, p. 124 et 125)

> [!IMPORTANT]
> **Incompatibilité formelle du coffre STRADAL avec la manœuvre par treuil** : le demi-linteau STRADAL ne disposant d'aucune réservation latérale pour le passage du déport de cardan, la manœuvre manuelle par treuil y est strictement impossible. Les volets sous coffre STRADAL doivent obligatoirement être motorisés [1 p. 114, 120, 124, 125].

---

# Tabliers, lames et limites dimensionnelles

Le BLOC LX accepte des tabliers en lames de PVC extrudé ou en aluminium profilé double paroi avec mousse polyuréthane standard ou haute densité (HD) [1 p. 112].

### Caractéristiques mécaniques des lames

| Référence de lame | Matériau | Pas de lame (mm) | Épaisseur nominale (mm) | Largeur max 1 tablier (mm) | Poids tablier (kg/m²) | Hauteur max $H$ coffre Stradal (mm) | Hauteur max $H$ autres coffres (mm) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L 37 | PVC | 37 | 8 | 1 500 | 3,2 | 2 750 | 3 000 |
| L 50 | PVC | 50 | 12 | 2 000 | 3,5 | 1 850 | 2 150 |
| LATH 37 | Alu thermique | 37 | 8 | 2 500 | 2,7 | 2 750 | 3 000 |
| LA 37 | Alu standard | 37 | 8 | 2 500 | 2,7 | 2 750 | 3 000 |
| LAHD 37 | Alu haute densité | 37 | 8 | 2 500 | 2,7 | 2 750 | 3 000 |
| LATH 50 | Alu thermique | 50 | 13 | 3 500 | 2,8 | 1 950 | 2 250 |
| LA 50 | Alu standard | 50 | 13 | 3 500 | 2,8 | 1 950 | 2 250 |
| LAHD 50 | Alu haute densité | 50 | 13 | 3 500 | 3,3 | 1 950 | 2 250 |

(schéma: raw/moustiquaires/export_doc_133.zip, p. 112)

*Majoration de poids* : ajouter **1,1 kg par mètre linéaire** pour la lame finale en aluminium [1 p. 112].
*Note de hauteur* : les « autres coffres » regroupent TERREAL, GENOVA, PREFATEC, IMERYS et EVENO (Ligne A du barème Soprofen) [1 p. 112].

---

# Coulisses et lames finales

### Typologie des coulisses

Les coulisses sont livrées en standard avec un pré-perçage étagé d'usine $\varnothing 5\text{ mm}$ et $\varnothing 10\text{ mm}$ à entraxe régulier de 500 mm pour fixation directe entre tapées [1 p. 110] :
* **CTA 09** : section $41,5 \times 29\text{ mm}$, réservée aux tabliers en lames de 37 mm [1 p. 110].
* **CTA 13 E** : section $41,5 \times 29\text{ mm}$, pour tabliers en lames de 50 mm jusqu'à 3 000 mm de largeur [1 p. 110].
* **CTA 13** : ancienne coulisse grande section $54,7 \times 29\text{ mm}$, maintenue pour harmonisation SAV et **obligatoire pour toutes les largeurs supérieures à 3 000 mm** [1 p. 110].

### Lames finales

* **LF40 (Standard)** : profilé aluminium extrudé de 34 mm de hauteur pour lames de 37 et 50 mm, équipée de butoirs d'arrêt cachés escamotables **BLF3-V1** et lestée selon les dimensions du tablier [1 p. 113].
* **Variantes optionnelles** :
  * Butoirs apparents **KZ 12/60** (coloris blanc, gris ou brun) : leur présence impose d'avoir posé la trappe intérieure de visite avant toute manœuvre du volet [1 p. 113].
  * Lame finale **LF40 sans joint**.
  * Lame finale compacte **LF30** (limitée aux teintes Blanc, Beige 1015, Gris 7016 et Brun) [1 p. 110, 113].

---

# Motorisations et manœuvres

Le volet BLOC LX intègre des motorisations [SOMFY](/fournisseurs/somfy.md) ou SOPROFEN M-Soft², ainsi que des manœuvres manuelles par treuil [1 p. 113-117].

### Synthèse des motorisations et largeurs minimales ($L_{\min}$)

| Protocole | Modèle de motorisation | Code commande | Manœuvre de secours | Largeur mini $L_{\min}$ (mm) | Largeur max moteur court (mm) |
| --- | --- | --- | --- | --- | --- |
| Radio RTS | Oximo court RTS | R1 | Non | 500 | 799 |
| Radio RTS | Oximo RTS standard | R1 | Non | 750 | - |
| Radio RTS | Oximo court RTS + secours | R1S | Oui | 600 | 944 |
| Radio RTS | Oximo RTS + secours | R1S | Oui | 850 | - |
| Radio IO | Oximo court IO | IO1 | Non | 500 | 659 |
| Radio IO | S&SO RS100 IO standard | IO1 | Non | 610 | - |
| Radio IO | Oximo court IO + secours | IO1S | Oui | 600 | 804 |
| Radio IO | S&SO RS100 IO + secours | IO1S | Oui | 710 | - |
| Radio M-SOFT 2 | M-Soft² MVM court | R11 | Non | 530 | 659 |
| Radio M-SOFT 2 | M-Soft² MVM standard | R11 | Non | 610 | - |
| Filaire WT | Ilmo court | M | Non | 500 | 699 |
| Filaire WT | Ilmo standard | M | Non | 650 | - |
| Filaire WT | Ilmo court + secours | MS | Oui | 600 | 844 |
| Filaire WT | Ilmo standard + secours | MS | Oui | 750 | - |
| Filaire WT | Ilmo 12 tours (couple réduit) | M1 | Non | 650 | - |
| Filaire WT | Ilmo 12 tours + secours | M1S | Oui | 750 | - |
| Filaire M-SOFT 2 | M-Soft² MVEC 06/23 | M11 | Non | 530 | - |
| Filaire M-SOFT 2 | M-Soft² MVE standard | M11 | Non | 610 | - |
| Hybride filaire/radio | S&SO RS100 IO Hybrid | IOH | Non | 610 | - |
| Hybride filaire/radio | S&SO RS100 IO Hybrid + secours | IOHS | Oui | 710 | - |

(schéma: raw/moustiquaires/export_doc_133.zip, p. 113)

> [!WARNING]
> **Incompatibilité des protocoles radio** : les technologies radio Somfy RTS, Somfy IO et Soprofen M-Soft 2 sont strictement incompatibles entre elles et ne peuvent être appairées sur les mêmes points de commande [1 p. 115, 118].

### Manœuvres manuelles par treuil

* **Treuil TX13/5-7 (Code T)** : treuil sans fin de course équipé de série d'un verrouillage automatique TLX, d'un cardan double CS20/12-7 (tige hexagonale 7 mm de 500 mm) et d'une tige oscillante $\varnothing 12\text{ mm}$ en acier revêtu polyester [1 p. 114].
* **Treuil TX13/5-7FC (Code T10)** : treuil à fin de course avec attaches souples de tablier [1 p. 114].
* **Déport latéral par barrettes BTD** : permet d'ajuster l'écartement de la tige oscillante pour ne pas heurter l'ouvrant de la fenêtre [1 p. 120] :
  * **Position A** (7 barrettes BTD) : distance axe tige au dos coulisse $DC = 15\text{ mm}$ (réservation coffre $R = 50\text{ mm}$).
  * **Position B** (9 barrettes BTD) : $DC = 25\text{ mm}$ ($R = 60\text{ mm}$).
  * **Position C** (11 barrettes BTD) : $DC = 35\text{ mm}$ ($R = 70\text{ mm}$).

---

# Projection à l'italienne (Manuelle)

Le volet BLOC LX propose en option un mécanisme de projection vers l'extérieur pour tamiser le rayonnement direct [1 p. 121] :
* **Bras de projection** : compas de longueur 300 mm (hauteur minimale de fenêtre $H_{\min} = 920\text{ mm}$) ou 470 mm ($H_{\min} = 1 260\text{ mm}$), avec traverse basse de solidarisation EP25 [1 p. 121].
* **Obligations techniques** : tablier d'un seul tenant en lames LA37 ou LA50 (largeur max 1 500 mm), butoirs apparents KZ 12/60, tapée d'au moins 60 mm et espace libre minimum de 30 mm entre la coulisse et la menuiserie pour le débattement des bras [1 p. 121].
* **Interdiction absolue** : la projection est **strictement interdite en manœuvre motorisée** ; elle ne peut être mise en œuvre qu'avec une manœuvre manuelle par treuil [1 p. 121].

---

# Profilés acier de renfort d'inertie

Afin de contrer la poussée du vent sur les ensembles composés et les grandes largeurs, un renfort acier tubulaire plié ($49,7 \times 32\text{ mm}$, épaisseur 2,5 mm) de longueur $L$ peut être vissé sur la traverse haute de menuiserie [1 p. 122, 2 p. 2] :

| Code renfort | Destination doublage | Dimensions section (mm) | Épaisseur acier (mm) | Inertie axiale $I_{xx'}$ (cm⁴) | Inertie principale $I_{yy'}$ (cm⁴) |
| --- | --- | --- | --- | --- | --- |
| Renfort code 1 | ISO 100 et ISO 120 | $49,7 \times 32$ | 2,5 | 2,60 | 5,56 |
| Renfort code 2 | ISO 140 et ISO 160 | $49,7 \times 32$ | 2,5 | 3,15 | 18,70 |

(schéma: raw/moustiquaires/export_doc_133.zip, p. 122)

---

# Options et accessoires d'atelier

1. **Trappe de visite VX03/18** : sous-face aluminium laquée avec recouvrement réglable de 35 à 55 mm, clipsée sur supports DL-SFI (intérieur), DL-SFE (extérieur) et équerres d'extrémité DL-ESF [1 p. 122].
2. **Pattes déformables VX26** : brides de fixation d'angle assurant la triangulation rigide entre les joues du coffre et le dormant de menuiserie (vis 15/3-9/9-5) [1 p. 121, 2 p. 2].
3. **Mortaises d'entrée d'air VMC** : usinage d'atelier en partie haute sans fourniture de grille [1 p. 123] :
   * **Type A** : entaille unique de $250 \times 15\text{ mm}$.
   * **Type B** : double fente de $2 \times (172 \times 12\text{ mm})$ espacée par un pontet matière de 5 mm.
4. **Isolation thermique** : tiroir équipé en standard du bouclier isolant DL52 (possibilité d'option sans isolant) [1 p. 123].

---

# Citations

[1] [Guide technique volet roulant demi-linteau Bloc LX SOPROFEN](raw/moustiquaires/export_doc_133.zip), p. 110 à 125
[2] [Guide de pose Bloc LX coffre demi-linteau SOPROFEN, réf. NO-BB-T-BLX-00-FR-202111](raw/moustiquaires/export_doc_136(1).zip), p. 1 à 3

---

# Voir aussi

- [SOPROFEN](/fournisseurs/soprofen.md)
- [SOMFY](/fournisseurs/somfy.md)
- [Volets roulants](/equipements/volets-roulants.md)
- [Assemblage et pose du coffre demi-linteau Bloc LX SOPROFEN](/procedures/pose-bloc-lx-demi-linteau-soprofen.md)
- [Guide technique volet roulant demi-linteau Bloc LX SOPROFEN](/sources/soprofen-guide-technique-bloc-lx.md)
- [Guide de pose Bloc LX coffre demi-linteau SOPROFEN (2021)](/sources/soprofen-guide-pose-bloc-lx-demi-linteau.md)
