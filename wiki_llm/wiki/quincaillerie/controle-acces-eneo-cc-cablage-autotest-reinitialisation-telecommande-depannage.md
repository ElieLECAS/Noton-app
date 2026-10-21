---
type: Quincaillerie
title: 'Roto Safe E Eneo CC : plan de câblage, autotest, réinitialisation, télécommande, dépannage'
description: Plan de câblage du contrôle d'accès 4 en 1 avec la serrure Roto Safe E Eneo CC, test par la fonction autotest, réinitialisation aux paramètres d'usine, application SOREX SmartLock, télécommande et tableaux de dépannage du clavier et de la serrure.
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

Cette page reprend, de la notice simplifiée PROFERM Roto Safe E Eneo CC (version 2, 2022), le câblage, la mise en service, la télécommande et le dépannage. Contexte commun : [Contrôle d'accès 4 en 1 Roto Safe E Eneo CC : caractéristiques, fraisage, câble, pose](/quincaillerie/controle-acces-eneo-cc.md).

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

# Citations

[1] Roto Safe E Eneo CC, notice simplifiée « Montage & Programmation des contrôles d'accès »,
PROFERM / Roto, version 2, 2022 — `raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf`, p. 1 à 10

# Voir aussi

Cette page est une partie d'un sujet qui en compte 3 ; les autres :

- [Contrôle d'accès 4 en 1 Roto Safe E Eneo CC : caractéristiques, fraisage, câble, pose](/quincaillerie/controle-acces-eneo-cc.md)
- [Roto Safe E Eneo CC : entretien, sécurité, garantie et responsabilité](/quincaillerie/controle-acces-eneo-cc-entretien-securite-garantie.md)
