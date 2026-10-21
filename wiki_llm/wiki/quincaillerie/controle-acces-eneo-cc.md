---
type: Quincaillerie
title: 'Contrôle d''accès 4 en 1 Roto Safe E Eneo CC : caractéristiques, fraisage, câble, pose'
description: Ce qu'est le contrôle d'accès 4 en 1 de la serrure motorisée Roto Safe E Eneo CC (code PIN, empreinte, Bluetooth, RFID), compatibilités, fraisage du vantail et du dormant, retournement du pêne, passage de câble, dimensions d'usinage et pose du module.
tags: [roto, safe-e, eneo-cc, controle-acces, porte-entree, serrure-motorisee, rfid, biometrie, cablage, telecommande, fraisage]
systeme: Roto Safe E
fournisseur: ROTO
usage: [pose, sav]
famille: controle-acces
status: draft
sources:
  - resource: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf
    id: proferm-eneo-cc-notice-2022
    title: Roto Safe E Eneo CC, notice simplifiée PROFERM, version 2, 2022
    last_modified: 2022-12-31
source_pages:
  - resource: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf
    pages: 1-10
generated:
  by: process:claude-code
  at: 2026-10-21T10:43:53Z
---

# Ce qu'est le contrôle d'accès 4 en 1 Roto Safe E Eneo CC

La **Roto Safe E Eneo CC** est une serrure motorisée de porte d'entrée : un moteur électrique,
logé dans le vantail (la partie mobile de la porte, l'[ouvrant](/reference/glossaire.md)), manœuvre
la serrure et ses points de verrouillage. Le **système de contrôle d'accès 4 en 1** (écrit aussi
« 4in1 » ou « 4en1 ») est le boîtier à clavier posé à l'extérieur qui commande son ouverture.
La porte peut être ouverte en saisissant un **code PIN** ou en utilisant une **empreinte
digitale**, un **smartphone compatible Bluetooth** ou un **support compatible RFID** (badge ou
porte-clés lu sans contact) [1 p. 4].

![Module de contrôle d'accès 4 en 1, clavier et lecteur d'empreinte](/assets/quincaillerie/eneo-cc/module-4-en-1-couverture.png)

La photo montre la face avant du module : le lecteur d'empreinte rond en haut, le clavier de
chiffres 1 à 9 et 0, la touche **×** (effacer) et la touche **✓** (validation) [1 p. 1].
Voir [Serrure motorisée](/quincaillerie/serrure-motorisee.md) pour l'offre PROFERM qui propose
cette serrure.

# Caractéristiques

Données techniques du module de contrôle d'accès 4 en 1, une ligne par caractéristique, dans
les termes du tableau de la notice [1 p. 7].

| Caractéristique | Valeur |
| --- | --- |
| Dimensions extérieures (L × H × P) | 55 × 99,8 × 19,8 mm |
| Tension de fonctionnement | 12 V - 24 V DC, 200 mA |
| Données sur les relais | 1 A 250 V cap. de commutation |
| Plage de température | Operation : −20 to +60 °C |
| Indice IP | IP66 (une fois collé, imperm. à l'eau) |
| Capacité de stockage | « 100 empreintes 150 codes numériques 200 Supports RFID eKeys illimités » (voir ci-dessous) |
| Cryptage | AES 128 bit |
| Standards | Conformité CE |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 7)

Lecture des sigles : L × H × P = largeur × hauteur × profondeur ; DC = courant continu ; « cap.
de commutation » = pouvoir de coupure du contact du relais ; l'indice IP (indice de protection)
IP66 désigne un boîtier étanche aux poussières et aux jets d'eau puissants, valable ici « une
fois collé » ; AES 128 bit est l'algorithme de chiffrement des échanges ; une **eKey** est une
clé virtuelle attribuée à un smartphone par l'application.

La cellule « Capacité de stockage » est imprimée sur quatre lignes, sans séparateur entre les
nombres et les supports ; la lecture « 100 empreintes, 150 codes numériques, 200 supports RFID,
eKeys illimités » est la plus directe, mais la découpe n'est pas écrite (**VER-33**).

Le transformateur qui alimente l'installation, dessiné sur le plan de câblage, porte
« IN : 100-240 V AC, OUT : 24 V DC / 2.5 A » : il reçoit le secteur (100 à 240 V en courant
alternatif) et rend du 24 V en courant continu, sous 2,5 A au plus [1 p. 6].

# Compatibilités

| Élément | Référence ou valeur | Rôle |
| --- | --- | --- |
| Passage de câble, partie dormant | 817028 | pièce fixée dans le dormant, repère [1] Dormant du dessin de montage |
| Passage de câble, partie ouvrant | 820255 | pièce fixée dans l'ouvrant, repère [2] Ouvrant du dessin de montage |
| Câble entre la serrure et le passage de câble | « Cable – type EZ », 3 m | relie les bornes de la serrure aux bornes 1 à 6 côté ouvrant |
| Télécommandes | jusqu'à 30 par récepteur radio | récepteur de l'Eneo C, CC ou CF |
| Application | SOREX SmartLock, Android et iOS | réglage des paramètres du 4 en 1 |

Le document complet dont cette notice est un extrait est « Roto Safe E instructions
d'installation : Eneo C | CC | CF : IMO_438 » ; chaque page renvoie aux « instructions de montage
complètes IMO_438_DE » et porte « Sous réserve de modifications » [1 p. 2-10]. Les deux pièces de
passage de câble sont décrites en détail sur
[Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md).

# Fraisage de la serrure et des gâches

Le **fraisage** est l'usinage, à la fraiseuse, des logements qui reçoivent la serrure dans le
vantail et les [gâches](/reference/glossaire.md) dans le dormant (le cadre fixe). Les cotes sont
en millimètres ; les positions verticales sont prises depuis le **centre du boîtier serrure**, le
repère que la planche désigne par une flèche [1 p. 2].

## Fraisage du vantail

![Fraisage du vantail pour la serrure Roto Safe E Eneo CC](/assets/quincaillerie/eneo-cc/fraisage-vantail-serrure.png)

La planche dessine le vantail sous trois vues, de gauche à droite : le **chant** du vantail (la
tranche verticale côté serrure, où affleure la [têtière](/reference/glossaire.md) — la longue
plaque métallique de la serrure, percée pour les pênes), avec ses fraisages en rouge ; une coupe
du vantail en gris montrant la profondeur de chaque logement ; la serrure elle-même, vue de côté,
avec ses boîtiers. Un médaillon en pointillés agrandit la section de la rainure de têtière. Une
remarque encadrée précise : **« La largeur du fraisage dépend de la largeur de la têtière
utilisée ! »** [1 p. 2].

**Rainure de têtière (médaillon).** Sur toute la hauteur du chant, une feuillure de la largeur
de la têtière (« Largeur têtière », non chiffrée), profonde de 3 mm, reçoit la têtière ; au fond,
une rainure centrale de 12 mm de large descend de 6 mm plus bas.

**Fraisages du chant.** Quatre lumières de 16 mm de large, en rouge, traversent le fond de la
rainure. La hauteur totale du chant dessiné est cotée 2 200 mm.

| Fraisage du chant | Position | Longueur (mm) | Largeur (mm) |
| --- | --- | --- | --- |
| Lumière haute | axe à 752 au-dessus du centre du boîtier serrure | 176 | 16 |
| Lumière de la serrure | traversée par la ligne du centre du boîtier serrure | 226 | 16 |
| Lumière intermédiaire basse | la cote 438 sous le centre du boîtier serrure tombe à 151,5 du haut de la lumière | 220 | 16 |
| Lumière basse | axe à 738 au-dessous du centre du boîtier serrure | 176 | 16 |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 2)

**Logements dans l'épaisseur du vantail (coupe grise et serrure).**

| Logement | Élément logé et ses cotes (mm) | Profondeur du logement (mm) |
| --- | --- | --- |
| Boîtier haut | boîtier de 43 × 150 | 45 |
| Boîtier de serrure | boîtier de 200 de haut ; axe du fouillot à la distance D de la têtière | D + 20 |
| Unité motrice (sous le boîtier de serrure) | boîtier de 56 × 195 | 60 |
| Boîtier bas | boîtier de 43 × 150 | 45 |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 2)

Autour du boîtier de serrure, la coupe porte les perçages de la béquille et du cylindre :

- un perçage **Ø 20** pour le [fouillot](/reference/glossaire.md) (la pièce qui reçoit le carré de
  la béquille), dont l'axe est à **1 020 mm** du bas du vantail et à **22 mm** au-dessus du centre
  du boîtier serrure ;
- sous lui, le perçage oblong du cylindre (le barillet de la clé), à l'entraxe **E92** — 92 mm
  entre l'axe du fouillot et celui du cylindre —, coté **20** et **17** mm, à extrémités en rayon
  « R » dont la valeur n'est pas écrite ;
- en tête du vantail, une cote de **300 mm** au-dessus du boîtier haut.

La lettre **D** n'est pas chiffrée sur la planche : c'est la distance, cotée sur la serrure, entre
la têtière et l'axe du fouillot.

## Fraisage du dormant

![Fraisage du dormant pour les gâches de la serrure Roto Safe E Eneo CC](/assets/quincaillerie/eneo-cc/fraisage-dormant-gaches.png)

Le dormant est dessiné en saumon, vu de face, avec son **axe de fraisage** en trait mixte ; les
fraisages sont en rouge. À droite de chaque fraisage, la gâche correspondante est dessinée de
profil (avec sa cote de profil), puis de face, puis montée sur la longue platine perforée
dessinée à l'extrême droite sur toute la hauteur. Deux remarques encadrées : **« Le fraisage
dépend des hauteurs de gâche. »** et **« L'axe de fraisage dépend du profil utilisé. »**
[1 p. 2].

| Fraisage du dormant | Position de l'axe | Hauteur (mm) | Cotes en largeur (mm) | Cote de profil de la gâche (mm) |
| --- | --- | --- | --- | --- |
| Gâche haute | 752 au-dessus du centre du boîtier serrure | 135 | 19 | 24,5 |
| « Fraisage pour dispositif d'ouverture » | 48 au-dessus du centre du boîtier serrure | 82 | 14,5 et 9,5 | 30 |
| Fraisage central inférieur | 44 au-dessous du centre du boîtier serrure | 75 | 7 et 8 | 19,1 |
| Perçage rond | 438 au-dessous du centre du boîtier serrure | Ø 20 | - | 18,5 |
| Gâche basse | 738 au-dessous du centre du boîtier serrure | 135 | 19 | 24,5 |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 2)

Les cotes 14,5 / 9,5 et 7 / 8 sont portées de part et d'autre des deux fraisages centraux, sans
dire de quel bord elles partent. La planche ne nomme ni le « dispositif d'ouverture » ni la pièce
logée dans le perçage Ø 20 (**VER-111**).

# Retournement du pêne

Le pêne est la pièce qui sort de la têtière pour s'engager dans la gâche. La notice donne le
geste du retournement en cinq étapes, sans dire dans quel cas il se fait [1 p. 2].

![Retournement du pêne de la serrure Eneo CC, étapes 1 à 5](/assets/quincaillerie/eneo-cc/retournement-du-pene.png)

Les deux vues montrent le boîtier de serrure ; les repères carrés 1 à 5 renvoient aux étapes
ci-dessous : à gauche, la tige qu'on enfonce jusqu'au « Click » (1) et le pêne qui sort (2) ; à
droite, le pêne retourné (3), réinséré (4), et la goupille repoussée (5).

1. Insérer la tige Ø max. 2,5 mm dans le trou prévu à cet effet jusqu'à entendre un clic.
   **Ne pas faire sortir complètement la goupille de verrouillage du pêne !**
2. Sortir le pêne.
3. Retourner le pêne.
4. Insérer le pêne de façon droite et appuyer.
5. Pousser la goupille de verrouillage du pêne.

# Passage de câble

Le **passage de câble** fait passer les fils électriques du dormant, où arrive l'alimentation,
à l'ouvrant, où se trouve la serrure. Il se compose de deux pièces [1 p. 3].

![Pièce de passage de câble 817028](/assets/quincaillerie/eneo-cc/passage-de-cable-817028.png)

La **817028** est une longue platine portant un boîtier, avec la sortie de câble en haut et un
câble qui ressort en bas.

![Pièce de passage de câble 820255](/assets/quincaillerie/eneo-cc/passage-de-cable-820255.png)

La **820255** est une platine d'où part un câble protégé par une gaine spiralée, terminé par un
connecteur, et d'où sort à angle droit un câble à brancher.

![Montage du passage de câble : dormant [1] et ouvrant [2]](/assets/quincaillerie/eneo-cc/passage-de-cable-montage.png)

Le dessin de montage assemble les deux pièces : **[1] Dormant**, en saumon, la 817028 ;
**[2] Ouvrant**, en gris, la 820255, dont la gaine spiralée vient s'engager en haut de la pièce
de dormant [1 p. 3].

# Pose du module de contrôle d'accès 4 en 1

## Dimensions d'usinage

Le module s'encastre dans un fraisage rectangulaire de **40 mm de large et 86 mm de haut**, aux
coins arrondis de rayon **5 mm** (R 5) [1 p. 4].

![Dimensions d'usinage du module 4 en 1 : 40 × 86 mm, R 5](/assets/quincaillerie/eneo-cc/dimensions-usinage-module.png)

## Étapes

**Le transformateur ne doit pas encore être connecté à l'alimentation électrique** pendant ces
opérations [1 p. 4].

1. Effectuer le fraisage.
2. Nettoyez la surface.
3. Faites passer le câble dans le trou prévu à cet effet. Raccordez le câble de l'unité
   extérieure au câble de l'unité intérieure à l'intérieur.
   *Info* : en cas d'installation dans le mur, le relais doit être monté dans la zone sécurisée.
   Cela permet d'éviter qu'il ne soit manipulé de l'extérieur.
4. Retirez le film de protection de la bande adhésive double face. Insérez le système de
   contrôle d'accès 4en1 dans le vantail de la porte ou dans le mur et collez-le.
   *Info, étanchéité* : appuyez sur le système de contrôle d'accès 4en1 de tous les côtés. Un
   scellement supplémentaire est recommandé pour les surfaces texturées.
5. Connectez le transformateur à l'alimentation électrique.

[1 p. 4-5]

# Citations

[1] Roto Safe E Eneo CC, notice simplifiée « Montage & Programmation des contrôles d'accès »,
PROFERM / Roto, version 2, 2022 — `raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf`, p. 1 à 10

# Voir aussi

- [Serrure motorisée](/quincaillerie/serrure-motorisee.md)
- [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md)
- [ROTO](/fournisseurs/roto.md)
- [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md)
- [Glossaire](/reference/glossaire.md)
- [Roto Safe E Eneo CC, notice simplifiée PROFERM](/sources/proferm-roto-eneo-cc-notice-simplifiee.md)

Autres parties de ce sujet :

- [Roto Safe E Eneo CC : plan de câblage, autotest, réinitialisation, télécommande, dépannage](/quincaillerie/controle-acces-eneo-cc-cablage-autotest-reinitialisation-telecommande-depannage.md)
- [Roto Safe E Eneo CC : entretien, sécurité, garantie et responsabilité](/quincaillerie/controle-acces-eneo-cc-entretien-securite-garantie.md)
