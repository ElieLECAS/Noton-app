---
type: Quincaillerie
title: Contrôle d'accès 4 en 1 Roto Safe E Eneo CC
description: Le contrôle d'accès 4 en 1 et la serrure motorisée Roto Safe E Eneo CC des portes PROFERM — fraisages de la serrure et des gâches, passage de câble, pose du module, plan de câblage, autotest, réinitialisation, télécommande, dépannage et entretien.
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
  at: 2026-09-29T18:00:00Z
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

# Plan de câblage

Le plan relie la serrure (en haut), le passage de câble (au milieu), puis, côté dormant, le
transformateur et un bouton poussoir ; le contrôle d'accès 4 en 1 et un contacteur jour / nuit
sont raccordés en haut, côté ouvrant [1 p. 6].

![Plan de câblage Roto Safe E Eneo CC avec contrôle d'accès 4 en 1](/assets/quincaillerie/eneo-cc/plan-de-cablage.png)

Comment lire le plan, de haut en bas :

- **En haut**, un boîtier (sans nom sur le plan) porte deux connecteurs : **« Noir »** avec les
  bornes K1b, K1a (le contact dessiné en symbole d'interrupteur) et IN2 ; **« Vert »** avec les
  bornes GND, +24 V et IN1. Les fils qui en sortent sont, dans l'ordre : Rose, Gris, Jaune, Vert,
  Brun, Blanc.
- **À gauche**, le **contrôle d'accès 4en1** (« Extérieur : "Déverrouillage de la porte" système de
  contrôle d'accès ») est relié à la **Boite noire**, elle-même reliée au connecteur marqué
  **« Blanc »** (« JPrise connection JST ») ; ses trois fils Brun, Vert et Blanc se branchent, par
  les points de jonction, sur les fils brun, vert et blanc de la serrure.
- **À droite**, le **Contacteur Jour / Nuit** (« Intérieur, Peut être utilisé en option ») :
  l'interrupteur dessiné dans le cercle relie le fil jaune (IN2) au fil brun (+24 V) ; le fil vert
  du connecteur JST est prolongé jusqu'au contour du contacteur.
- **Au milieu**, le **« Cable – type EZ, 3 m »** descend jusqu'au bornier **côté ouvrant** du
  **passage de câble**, puis ressort **côté dormant** avec les mêmes six couleurs.
- **En bas, côté dormant** : le brun va à la borne **24V (+V)** et le vert à la borne **GND (−V)**
  du **Transformateur Eneo** (bornes secteur N et L ; IN : 100-240 V AC, OUT : 24 V DC / 2.5 A) ;
  le blanc et le brun vont au **Bouton poussoir** (« Intérieur : "Déverrouillage de la porte" en
  option pour l'Eneo CC »). Le jaune, le gris et le rose ne sont raccordés à rien côté dormant.

Bornier du passage de câble, côté ouvrant, relevé sur le plan :

| Borne | Couleur du fil |
| --- | --- |
| 1 | Blanc |
| 2 | Brun |
| 3 | Vert |
| 4 | Jaune |
| 5 | Gris |
| 6 | Rose |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 6)

**Affectation des connecteurs et des câbles**, dans les termes de la notice :

| Couleur du fil | Affectation |
| --- | --- |
| Blanc | IN1 / input 1 (OUVERT) |
| Brun | +24 V |
| Vert | GND |
| Jaune | IN2 / input 2 (commutateur jour/nuit) |
| Gris | K1a pot.-contact libre |
| Rose | K1b pot.-contact libre |

IN1 et IN2 sont les deux entrées de commande (IN1 commande l'ouverture, IN2 le passage jour /
nuit) ; GND est la masse, le 0 V ; « pot.-contact libre » désigne un
[contact libre de potentiel](/reference/glossaire.md), c'est-à-dire un contact sec qui ne fournit
aucune tension.

**Le fil jaune doit rester non attribué, sinon il pontera l'interrupteur de l'ouvrant.** Les
bornes 5 et 6 sont reliées entre elles en interne par un relais et une résistance de 47 ohms. La
charge maximale des contacts est de 24 V / 40 mA [1 p. 6].

La notice de la jonction de câble Roto Safe E dessine les mêmes fils marron, vert et blanc sur ce
connecteur JST, mais donne pour les câbles du 4 en 1 « entre connecteur JST et Black Box » les
couleurs rouge, noir et jaune ([Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md),
**CTR-66**).

# Test à l'aide de la fonction autotest

L'autotest est un mécanisme de test automatique pour tester le câblage et les connexions avec
le moteur de la serrure. La mise en service à l'aide de l'application n'est pas nécessaire. Le
nombre de tests n'est pas limité. **Il n'est possible que dans l'état de livraison** [1 p. 5].

1. Entrer le code 123456 sur le clavier.
2. Confirmez avec la touche validation.
   → La porte s'ouvre.

# Réinitialisation (paramètres d'usine)

La **boîte noire** est l'unité intérieure du contrôle d'accès ; elle se trouve à l'intérieur, où
elle est protégée [1 p. 7]. Deux moyens de revenir aux paramètres d'usine :

| Moyen | Opération |
| --- | --- |
| Boîte noire | appuyer sur le bouton Reset de l'unité intérieure (env. 3 secondes) jusqu'à ce que deux signaux soient émis en succession rapide |
| Application SOREX SmartLock | via le premier utilisateur enregistré dans l'application : Paramètres → « Supprimer » ; noter la portée de l'appareil |

# Application SOREX SmartLock

Les paramètres du 4in1 peuvent être contrôlés à l'aide d'une application, disponible pour les
systèmes d'exploitation Android et iOS [1 p. 7].

![Application SOREX SmartLock : Google Play, App Store et code QR](/assets/quincaillerie/eneo-cc/application-sorex-smartlock.png)

Sous les logos Google Play et App Store, un premier code QR ; le second accompagne la mention
« Pour des instructions détaillées, voir SOREX-Unilock-WiFi.pdf ». Ce document n'est pas dans
le corpus.

![Code QR des instructions SOREX-Unilock-WiFi](/assets/quincaillerie/eneo-cc/qr-instructions-sorex-unilock-wifi.png)

# Télécommande

Jusqu'à **30 télécommandes** peuvent être associées au récepteur radio de l'Eneo C | CC | CF
[1 p. 8].

![Deux télécommandes porte-clés de l'Eneo](/assets/quincaillerie/eneo-cc/telecommandes.png)

Le récepteur radio comprend un code spécifique : ce n'est que lorsque le code transmis par la
télécommande correspond à celui du récepteur radio que ce dernier accepte les signaux transmis
par l'émetteur. Chaque bouton de la télécommande peut être utilisé pour différentes Eneos :
2 Eneos peuvent être contrôlées séparément avec une seule télécommande. Il est possible de
programmer les deux boutons pour une seule Eneo [1 p. 8].

Association d'une télécommande :

1. Ouvrir la porte.
2. Verrouiller la serrure avec la clé lorsque la porte est ouverte.
3. Insérer une tige Ø 3 mm maxi. dans le trou situé sous la zone du capteur (zone en PVC noir)
   pour activer l'association.
4. La serrure effectue un bip sonore continu de 18 secondes. Cela indique que la serrure Eneo CC
   est prête pour l'association.
5. Appuyer sur le bouton de la télécommande.
6. L'association de la télécommande est confirmée par un bip de 2 secondes.

# Dépannage

## Clavier du contrôle d'accès 4 en 1

Assistance en cas de panne du clavier, une ligne par cause [1 p. 7].

| Erreur | Cause | Correction de l'erreur |
| --- | --- | --- |
| Le code n'a pas été accepté | Le code est bloqué | Avant d'entrer le code, appuyez sur la touche « X » du clavier de code pour effacer les chiffres qui ont été précédemment entrés |
| Le code n'a pas été accepté | Les touches du 4in1 ont déjà été pressées | Avant d'entrer le code, appuyez sur la touche « X » du clavier de code pour effacer les chiffres qui ont été précédemment entrés |
| Après la saisie de plusieurs codes incorrects, le clavier cesse de répondre | Si un code incorrect a été saisi cinq fois, le clavier est bloqué pendant 5 minutes | Attendez que le temps de blocage soit écoulé |

## Serrure Eneo CC

Tableau des erreurs de la serrure, une ligne par origine ; les signaux sonores (« bipe 3 x »,
« bipe 2 x ») sont des signaux d'erreur de la serrure [1 p. 9].

| Erreur | Origine | Solution |
| --- | --- | --- |
| Le système ne fonctionne pas. Eneo CC ne réagit pas ; pas de signal sonore | Il n'y a pas d'alimentation 220 volts fournie à l'entrée primaire du transformateur | L'installation électrique doit être réalisée uniquement par un professionnel qualifié, comme précisé dans les instructions de montage IMO_438 |
| Le système ne fonctionne pas | Il n'y a pas d'alimentation 24 volts fournie à l'entrée secondaire du transformateur | Vérifier les contacts du boîtier d'alimentation |
| Le système ne fonctionne pas | 24 volts ne sont pas fournis à la serrure Eneo CC | Vérifier les câbles de connexion entre le boîtier d'alimentation et la serrure Eneo CC et les remplacer si nécessaire |
| Le système ne fonctionne pas | 24 volts sont fournis à la serrure Eneo CC, cependant les +/− ont été intervertis | Inverser les polarités au niveau de l'entrée secondaire du transformateur |
| Le système ne fonctionne pas | L'unité motrice est dans sa position finale et ne reçoit pas de signal de mouvement | Vérifier les câbles qui transmettent le signal ou changer la distance de l'Eneo CC (distance : 1–2 m) |
| Le système ne fonctionne pas | Ne fonctionne toujours pas ? | Éteindre le transformateur, attendre 10 secondes et le rallumer ; tester avec l'Unité de Contrôle Eneo ; contacter un spécialiste |
| Eneo CC ne se verrouille pas automatiquement | La porte n'est pas complètement fermée | Fermer la porte complètement |
| Eneo CC ne se verrouille pas automatiquement | Eneo est en mode de fonctionnement de jour | Changer pour le mode nuit (24 volts ne peuvent pas être fournis à l'entrée 2 pour le mode de fonctionnement nuit) |
| Eneo CC ne se verrouille pas automatiquement | L'aimant en feuillure est mal aligné | Vérifier la position de l'aimant et l'ajuster |
| Eneo CC ne se verrouille pas complètement (signal d'erreur) | Porte et gâches ne sont pas alignés correctement (signal d'erreur : l'Eneo bipe 3 x) | Ajuster la porte et les gâches (se référer aux instructions de mise en service) |
| Eneo CC ne se verrouille pas complètement (signal d'erreur) | Corps étranger dans la gâche (signal d'erreur : l'Eneo bipe 3 x) | Retirer le corps étranger |
| Eneo CC ne se verrouille pas complètement (signal d'erreur) | Le pêne ne s'engage pas correctement et la porte s'ouvre un peu (signal d'erreur : contact reed non fermé : l'Eneo bipe 2 x) | Ouvrir la porte électriquement et repousser la porte |
| Eneo CC ne se verrouille pas complètement (signal d'erreur) | Eneo CC a été activée au cylindre | Si la porte a été déverrouillée manuellement, elle doit être verrouillée à nouveau manuellement (se référer aux instructions de mise en service) |
| La porte ne se déverrouille pas | Aucun signal de la télécommande ou aucun signal entrant aux récepteurs de l'Eneo CC | Programmer la télécommande comme décrit dans la notice. Vérifier respectivement les paramètres et le contrôle d'accès. Se référer aux notices de contrôles d'accès |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 9)

Le **contact reed** est un interrupteur magnétique, qui se ferme en présence d'un aimant ; l'« entrée primaire » du
transformateur est le côté secteur, l'« entrée secondaire » le côté 24 V.

# Entretien

## Avertissement

**Risques de blessures par des opérations de maintenance non conformes ! La maintenance réalisée
de manière incorrecte peut entraîner de graves blessures ou d'importants dommages matériels.**
[1 p. 10]

- Avant de commencer les travaux, prévoir un espace de manœuvre suffisant pour le montage.
- Veiller à ce que le lieu de montage soit bien rangé et propre.
- S'assurer que la porte ne peut pas s'ouvrir ou se fermer inopinément pendant les travaux de
  maintenance.
- Faire effectuer les travaux de réglage sur les ferrures par une entreprise spécialisée.

## Qui fait quoi

Les trois tableaux de la notice répartissent chaque opération entre l'**entreprise
spécialisée** et l'**utilisateur final**, avec trois symboles [1 p. 10] :

| Symbole de la notice | Sens |
| --- | --- |
| ■ | Exécution **uniquement** par une entreprise spécialisée |
| – | Exécution **pas** par l'utilisateur final. L'utilisateur final n'est pas autorisé à effectuer des travaux de montage |
| □ | Exécution aussi bien par une entreprise spécialisée que par l'utilisateur final |

**Au moins une fois par an :**

| Opération | Entreprise spécialisée | Utilisateur final |
| --- | --- | --- |
| Le cas échéant, resserrer les vis de fixation | ■ | – |
| Remplacer les vis endommagées | ■ | – |
| Le cas échéant, remplacer les pièces | ■ | – |
| Appliquer une huile spéciale sans résine ni acide sur toutes les pièces mobiles | □ | □ |
| Appliquer une huile spéciale sans résine ni acide sur les gâches en acier | □ | □ |

**Inspection — au moins une fois par an, tous les 6 mois dans les bâtiments scolaires et
hôteliers :**

| Opération | Entreprise spécialisée | Utilisateur final |
| --- | --- | --- |
| Vérifier que les ferrures qui assurent la sécurité sont bien fixées | □ | □ |
| Vérifier l'usure des ferrures qui assurent la sécurité | □ | □ |
| Vérifier le bon fonctionnement des parties mobiles | □ | □ |
| Vérifier le bon fonctionnement des points de fermeture | □ | □ |
| La mobilité des ferrures peut se contrôler à la poignée de la porte : ce contrôle s'effectue à l'aide d'une clé dynamométrique | ■ | – |
| Il est possible d'améliorer la mobilité des ferrures en ajoutant des graisses / huiles ou en réajustant les ferrures | ■ | – |

**Entretien (nettoyage) :**

| Opération | Entreprise spécialisée | Utilisateur final |
| --- | --- | --- |
| Ôter les dépôts et saletés des ferrures | □ | □ |
| Ne jamais utiliser de détergents agressifs ou acides ni d'abrasifs | □ | □ |
| Utiliser uniquement un détergent doux au pH neutre et dilué | □ | □ |
| Nettoyer uniquement à l'aide d'un chiffon doux | □ | □ |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 10)

Une **clé dynamométrique** est une clé qui mesure la force (le couple) appliquée.

## Protection de l'environnement

**Remarque !** Respecter les consignes suivantes pour la protection de l'environnement pendant
les travaux de maintenance [1 p. 10] :

- Éliminer l'excès de graisse sur les points de graissage et mettre au rebut selon les
  dispositions locales applicables.
- Récupérer les huiles vidangées dans des récipients appropriés et les éliminer dans le respect
  de l'environnement.

## Portée des recommandations

Aucune revendication juridique ne saurait découler de ces recommandations, leur application
dépend de chaque cas concret. Le fabricant de portes est tenu de mentionner l'existence de ces
instructions de maintenance aux maîtres d'œuvre et aux utilisateurs finaux. La société Roto Frank
AG recommande au fabricant de portes de conclure un contrat de maintenance avec ses clients
[1 p. 10].

# Sécurité, garantie et responsabilité

**Note.** Une installation, un entretien ou une utilisation incorrects peuvent entraîner des
situations dangereuses. Cet extrait ne remplace pas une documentation complète. Le non-respect de
cette documentation dégage le fabricant du matériel de sa responsabilité. Notez les instructions
complètes d'installation, d'entretien et d'utilisation [1 p. 4].

**Garantie.** La garantie ne couvre que les composants originaux de Roto. Roto se réserve le droit
d'apporter des modifications techniques dans le cadre de l'amélioration des caractéristiques de
performance et du développement ultérieur [1 p. 4].

**Utilisation prévue.** L'utilisation stipulée comprend également le respect de toutes les
informations contenues dans la documentation spécifique au produit, comme par exemple : ce bref
mode d'emploi ; les instructions d'installation, d'entretien et d'utilisation ; les catalogues de
produits ; les informations et spécifications des fabricants de profilés (par exemple, profilés
en métal léger, etc.) ; les lois et directives nationales applicables [1 p. 4].

**Protection des droits d'auteur.** Le contenu du document est protégé par des droits d'auteur.
Il peut être utilisé pour travailler avec le matériel. Toute autre utilisation est interdite sans
l'autorisation écrite du fabricant [1 p. 4].

**Limitation de la responsabilité.** Toutes les informations et instructions du document ont été
compilées en tenant compte des normes et réglementations applicables, des derniers développements
technologiques et de nombreuses années de connaissances et d'expérience. Le fabricant du matériel
n'assume aucune responsabilité pour les dommages causés par :

- le non-respect du présent document et de tous les documents spécifiques au produit et autres
  directives applicables (voir les chapitres intitulés « Sécurité » et « Utilisation stipulée ») ;
- une utilisation inappropriée / mauvaise utilisation (voir les chapitres intitulés « Sécurité »
  et « Utilisation conforme ») ;
- une publication insuffisante, le non-respect des instructions d'installation et le non-respect
  des diagrammes d'utilisation (s'ils existent) ;
- un encrassement important.

Les prétentions de tiers à l'encontre du fabricant de ferrures pour des dommages résultant d'une
mauvaise utilisation ou du non-respect de l'obligation d'instruction de la part des commerçants
de ferrures, des fabricants de fenêtres, de portes et de portes de balcon et des commerçants
d'éléments de construction ou du maître d'ouvrage sont transmises en conséquence. Les
obligations convenues dans le contrat de livraison, les conditions générales, les conditions de
livraison du fabricant de matériel informatique et les dispositions légales applicables au moment
de la conclusion du contrat sont applicables [1 p. 4].

Les chapitres « Utilisation stipulée » et « Utilisation conforme » auxquels renvoie cette liste
s'intitulent « Utilisation prévue » dans la notice (**INC-189**).

# Ce que la source ne donne pas

- la valeur de **D** (distance têtière – axe du fouillot) et du rayon **R** du perçage de cylindre ;
- le nom du « dispositif d'ouverture » et de la pièce logée dans le perçage Ø 20 du dormant
  (**VER-111**) ;
- les références de commande de la serrure, du module 4 en 1, de la boîte noire, du
  transformateur, du bouton poussoir, du contacteur jour / nuit et des télécommandes ;
- le nom du boîtier dessiné en haut du plan de câblage.

# Citations

[1] Roto Safe E Eneo CC, notice simplifiée « Montage & Programmation des contrôles d'accès »,
PROFERM / Roto, version 2, 2022 — `raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf`, p. 1 à 10

# Voir aussi

- [Serrure motorisée](/quincaillerie/serrure-motorisee.md)
- [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md)
- [ROTO](/fournisseurs/roto.md)
- [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md)
- [Glossaire](/reference/glossaire.md)
