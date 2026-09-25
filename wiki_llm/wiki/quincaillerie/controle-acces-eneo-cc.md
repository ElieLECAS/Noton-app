---
type: Quincaillerie
title: Contrôle d'accès 4 en 1 Roto Safe E Eneo CC
description: Le contrôle d'accès 4 en 1 des portes PROFERM — code PIN, empreinte, Bluetooth et RFID — avec ses caractéristiques électriques, son câblage et sa procédure de réinitialisation.
tags: [roto, safe-e, eneo-cc, controle-acces, porte-entree, serrure-motorisee, rfid, biometrie, cablage]
systeme: Roto Safe E
fournisseur: ROTO
usage: [chiffrage, pose]
status: stable
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
  at: 2026-09-19T22:30:00Z
---

# Ce qu'est le contrôle d'accès 4 en 1

Le **4 en 1** est le module de contrôle d'accès du système **Roto Safe E Eneo CC** que PROFERM
propose sur ses portes d'entrée. Il **ouvre la porte de quatre façons** :

| Moyen d'ouverture | Support |
| --- | --- |
| Code PIN | clavier du module |
| Empreinte digitale | lecteur biométrique du module |
| Smartphone | Bluetooth |
| Badge ou porte-clés | support compatible RFID |

Il se pilote par l'application **SOREX SmartLock**, disponible sur Android et iOS. Voir
[Serrure motorisée](/quincaillerie/serrure-motorisee.md) pour la serrure qu'il commande.

# Cotes et caractéristiques du module

Caractéristiques du module de contrôle d'accès 4 en 1, relevées sur la notice simplifiée PROFERM
(p. 7).

| Caractéristique | Valeur |
| --- | --- |
| Dimensions extérieures, L × H × P (mm) | 55 × 99,8 × 19,8 |
| Tension de fonctionnement | 12 à 24 V DC, 200 mA |
| Pouvoir de coupure du relais | 1 A sous 250 V |
| Plage de température en fonctionnement (°C) | −20 à +60 |
| Capacité, empreintes digitales | 100 |
| Capacité, codes numériques | 150 |
| Capacité, supports RFID (badges, porte-clés) | 200 |
| Supports Bluetooth eKeys (via application smartphone) | illimités |
| Cryptage | AES 128 bits |
| Conformité | CE |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 7)

La valeur « 200 » correspond à la capacité maximale de supports RFID physiques mémorisables (badges ou cartes d'accès), distincte des clés virtuelles eKeys illimitées gérées dans l'application SOREX SmartLock.

# Alimentation

| Élément | Valeur |
| --- | --- |
| Entrée du transformateur Eneo | 100 à 240 V AC |
| Sortie du transformateur Eneo | 24 V DC, 2,5 A |

**Le transformateur ne doit pas être raccordé au réseau avant la fin du montage** : la notice le
signale dès la première étape, et le raccordement est la dernière opération de la pose.

# Câblage : à quoi correspond chaque couleur

Affectation des six fils du module de contrôle d'accès, relevée sur la notice (plan de câblage).

| Couleur du fil | Affectation |
| --- | --- |
| Blanc | IN1, entrée 1 — OUVERT |
| Brun | +24 V |
| Vert | GND |
| Jaune | IN2, entrée 2 — commutateur jour/nuit |
| Gris | K1a, contact libre de potentiel |
| Rose | K1b, contact libre de potentiel |

**Le fil jaune doit rester non affecté**, sans quoi il ponte l'interrupteur de l'ouvrant. Les
**bornes 5 et 6 sont reliées entre elles en interne** par un relais et une résistance de 47 ohms (charge maximale des contacts 24 V / 40 mA).

C'est la seule prescription du document dont le non-respect ne se voit pas à la pose : un fil
jaune raccordé laisse la porte fonctionner et neutralise silencieusement la détection d'ouvrant.

Le passage de câble entre dormant et ouvrant se fait par deux pièces dédiées :

| Pièce | Référence | Côté |
| --- | --- | --- |
| Passage de câble, partie dormant | 817028 | dormant |
| Passage de câble, partie ouvrant | 820255 | ouvrant |

Voir [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md) pour les
variantes complètes et leurs caractéristiques.

# Pose du module, dans l'ordre

| Étape | Opération |
| --- | --- |
| 1 | Effectuer le fraisage — **le transformateur n'est pas encore raccordé** |
| 2 | Nettoyer la surface de pose |
| 3 | Passer le câble dans le trou prévu, raccorder le câble de l'unité extérieure à celui de l'unité intérieure, **à l'intérieur** |
| 4 | Retirer le film de protection de la bande adhésive double face, insérer le module dans le vantail ou dans le mur et le coller |
| 5 | Appuyer sur le module **de tous les côtés** — un scellement supplémentaire est recommandé sur les surfaces texturées |
| 6 | Raccorder le transformateur à l'alimentation électrique |

**Fraisage du module lui-même** : un rectangle de **40 × 86 mm**, coins arrondis à un rayon de
**5 mm** [1 p. 4].

# Fraisage de la serrure et des gâches (cote par cote)

Cotes relevées sur la planche d'atelier de la notice (p. 2) pour un vantail standard de hauteur 2 200 mm.

### Fraisage du vantail (serrure Roto Safe E Eneo CC)

Le repère vertical de référence est le **centre du boîtier serrure**, positionné à **1 020 mm** du bas de l'ouvrant :

| Zone de fraisage | Position par rapport au centre boîtier (1 020 mm) | Dimensions du fraisage (H × L × P) | Détails et têtière |
| --- | --- | --- | --- |
| Boîtier central de serrure | Centré à 0 mm (cote 1 020 mm du sol) | H 200 mm × P $(D + 20)\text{ mm}$ (où $D$ = axe de fouillot) | Carré fouillot à 1 020 mm, entraxe béquille/cylindre E92 mm (cylindre à 928 mm du sol) |
| Logement moteur arrière | Zone inférieure arrière du boîtier central | H 195 mm × L 56 mm × P 60 mm | Logement de l'unité motrice électrique Eneo |
| Boîtier supérieur (point haut) | Centré à **+752 mm** (cote 1 772 mm du sol) | H 150 mm × L 43 mm × P 45 mm | Fraisage têtière : 16 × 176 mm |
| Coffre intermédiaire haut | Centré à **+438 mm** (cote 1 458 mm du sol) | Fraisage selon profil | Fraisage têtière : 16 × 226 mm |
| Coffre intermédiaire bas | Centré à **-438 mm** (ou −151,5 mm) | H 151,5 à 220 mm | Fraisage têtière : 16 × 220 mm |
| Boîtier inférieur (point bas) | Centré à **-738 mm** (cote 282 mm du sol) | H 150 mm × L 43 mm × P 45 mm | Fraisage têtière : 16 × 176 mm |
| Goulotte / passage supérieur | À +300 mm au-dessus du centre boîtier | Profondeur selon profil | Raccordement vers la zone de passage de câble |
| Rainure de têtière continue | Sur toute la hauteur de la têtière | Largeur égale à la têtière, profondeur 3 mm | Rainure centrale de passage : L 12 mm × P 6 mm |

### Fraisage du dormant (gâches et capteurs)

Aligné sur l'axe de fraisage dormant face au centre du boîtier de serrure :

| Gâche | Position par rapport au centre boîtier | Cotes de fraisage (H × L × P) | Spécificités |
| --- | --- | --- | --- |
| Gâche haute | Centré à **+752 mm** | H 135 mm × L 24,5 mm × P 19 mm | Gâche de sécurité pour crochet/goujon |
| Gâche intermédiaire haute | Centré à **+438 mm** | H 135 mm × L 24,5 mm × P 19 mm | Gâche pour point de verrouillage secondaire |
| Gâche centrale (pêne 1/2 tour et pêne dormant) | Centré à 0 mm (repère 1 020 mm) | Découpe pêne : H 14,5 mm, 9,5 mm, 44 mm, 48 mm, 75 mm, 82 mm ; Largeur 30 mm ; P 19,1 mm | Logement central pour pêne et gâche électrique éventuelle |
| Logement capteur magnétique / reed | À -20 mm sous le centre, décalé à 18,5 mm de l'axe | Perçage circulaire **Ø 20 mm**, profondeur 18,5 mm | Détecteur de fermeture d'ouvrant pour verrouillage automatique |
| Gâche basse | Centré à **-738 mm** | H 135 mm × L 24,5 mm × P 19 mm | Gâche de sécurité pour crochet/goujon |

# Retournement du pêne

Procédure en cinq gestes, pour changer le sens d'ouverture de la porte (notice, montage de la
serrure).

| Étape | Geste |
| --- | --- |
| 1 | Insérer une tige de **Ø 2,5 mm maximum** dans le trou prévu, jusqu'au **clic** |
| 2 | Sortir le pêne |
| 3 | Retourner le pêne |
| 4 | Réinsérer le pêne droit et appuyer |
| 5 | Repousser la goupille de verrouillage du pêne |

**Ne jamais faire sortir complètement la goupille de verrouillage du pêne** — la notice le signale
en avertissement dès la première étape.

# Autotest et réinitialisation

Le module dispose d'un **mécanisme d'autotest** qui vérifie le câblage et les connexions avec le
moteur de la serrure. **La mise en service par l'application n'est pas nécessaire** pour le
lancer, et **le nombre de tests n'est pas limité**.

La réinitialisation aux paramètres d'usine se fait de deux façons :

| Moyen | Opération |
| --- | --- |
| Bouton Reset, sur la boîte noire à l'intérieur | appuyer **environ 3 secondes**, jusqu'à **deux signaux émis en succession rapide** |
| Application SOREX SmartLock | via le **premier utilisateur enregistré** : Paramètres puis « Supprimer » |

La **boîte noire** est l'unité intérieure, placée à l'abri dans le vantail ou dans le mur : c'est
elle qui protège l'électronique et porte le bouton Reset.

**Le code d'usine 123456, suivi de la touche de validation, ouvre la porte et teste
l'installation** — mais seulement dans l'état de livraison, avant toute programmation par
l'utilisateur [1 p. 5].

# Télécommande

Jusqu'à **30 télécommandes** peuvent être associées au récepteur radio de l'Eneo C, CC ou CF
[1 p. 8]. Le récepteur porte un code propre : il n'accepte que les signaux d'une télécommande
programmée avec ce code. Un même bouton de télécommande peut piloter des Eneo différentes — deux
Eneo se pilotent séparément avec une seule télécommande — et les deux boutons d'une télécommande
peuvent aussi être programmés pour une seule Eneo.

| Étape | Opération |
| --- | --- |
| 1 | Ouvrir la porte |
| 2 | Verrouiller la serrure avec la clé, porte ouverte |
| 3 | Insérer une tige de Ø 3 mm maximum dans le trou situé sous la zone du capteur (zone en PVC noir), pour activer l'association |
| 4 | Attendre le bip sonore continu de 18 secondes, qui signale que la serrure est prête pour l'association |
| 5 | Appuyer sur le bouton de la télécommande |
| 6 | L'association est confirmée par un bip de 2 secondes |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 8)

# Dépannage

Table d'assistance en cas de panne, relevée sur la notice (p. 9).

| Symptôme | Cause possible | Solution |
| --- | --- | --- |
| Le système ne fonctionne pas, aucun signal sonore | absence de 220 V à l'entrée primaire du transformateur | installation électrique réservée à un professionnel qualifié, selon l'IMO_438 |
| Le système ne fonctionne pas | absence de 24 V à la sortie du transformateur | vérifier les contacts du boîtier d'alimentation |
| Le système ne fonctionne pas | 24 V absents à la serrure | vérifier et remplacer si besoin les câbles de connexion |
| Le système ne fonctionne pas | polarité +/− inversée à la sortie du transformateur | inverser les polarités à l'entrée secondaire |
| Le système ne fonctionne pas | l'unité motrice en position finale ne reçoit pas de signal de mouvement | vérifier les câbles ou changer la distance de l'Eneo CC (1 à 2 m) |
| Eneo CC ne se verrouille pas automatiquement | porte non complètement fermée | fermer complètement la porte |
| Eneo CC ne se verrouille pas automatiquement | mode de fonctionnement de jour actif | passer en mode nuit |
| Eneo CC ne se verrouille pas automatiquement | aimant en feuillure mal aligné | vérifier et ajuster la position de l'aimant |
| Ne se verrouille pas complètement, signal d'erreur (bip ×3) | porte et gâches mal alignées, ou corps étranger dans la gâche | ajuster porte et gâches, ou retirer le corps étranger |
| Ne se verrouille pas complètement, signal d'erreur (bip ×2) | pêne mal engagé, contact reed non fermé | ouvrir la porte électriquement et repousser la porte |
| Porte activée manuellement au cylindre | déverrouillage manuel | reverrouiller manuellement après un déverrouillage manuel |
| La porte ne se déverrouille pas | aucun signal reçu de la télécommande ou du contrôle d'accès | programmer la télécommande, vérifier les paramètres du contrôle d'accès |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 9)

**Si le système ne fonctionne toujours pas** : éteindre le transformateur, attendre 10 secondes,
le rallumer, tester avec l'Unité de Contrôle Eneo, puis contacter un spécialiste [1 p. 9].

# Entretien

Trois tables de maintenance relevées sur la notice (p. 10), chacune répartie entre **entreprise
spécialisée** et **utilisateur final**.

**Au moins une fois par an :**

| Opération | Entreprise spécialisée | Utilisateur final |
| --- | --- | --- |
| Resserrer les vis de fixation desserrées | oui | oui |
| Remplacer les vis endommagées | oui | non |
| Remplacer les pièces si besoin | oui | non |
| Appliquer une huile sans résine ni acide sur les pièces mobiles | oui | oui |
| Appliquer une huile sans résine ni acide sur les gâches en acier | oui | oui |

**Inspection, au moins une fois par an, tous les 6 mois en établissement scolaire ou hôtelier :**

| Opération | Entreprise spécialisée | Utilisateur final |
| --- | --- | --- |
| Vérifier la fixation des ferrures de sécurité | oui | oui |
| Vérifier l'usure des ferrures de sécurité | oui | oui |
| Vérifier le fonctionnement des parties mobiles | oui | oui |
| Vérifier le fonctionnement des points de fermeture | oui | oui |
| Améliorer la mobilité par graissage/huilage ou réajustement des ferrures | oui | non |

**Nettoyage :**

| Opération | Entreprise spécialisée | Utilisateur final |
| --- | --- | --- |
| Ôter dépôts et saletés des ferrures | oui | oui |
| Nettoyer au détergent doux, pH neutre et dilué, chiffon doux | oui | oui |
| Utiliser un détergent agressif, acide ou abrasif | **interdit dans tous les cas** | **interdit dans tous les cas** |

**L'utilisateur final n'est jamais autorisé aux travaux de montage** — les cases qu'il ne coche
pas dans ces trois tables sont toutes réservées à l'entreprise spécialisée, sans exception
[1 p. 10]. Roto Frank recommande au fabricant de portes de conclure un contrat de maintenance
avec ses clients.

# Élimination des déchets

Les déchets électroniques du module suivent les directives européennes RoHS (2002/95/CE) et
WEEE (2002/96/CE) et, en Allemagne, la loi ElektroG : pas de mise au rebut avec les ordures
ménagères, remise à un site d'élimination approprié [1 p. 7].

# Conditions de garantie et responsabilités

La garantie constructeur couvre exclusivement les composants d'origine Roto [1 p. 4]. L'installation électrique doit être réalisée par un professionnel qualifié conformément aux instructions Roto IMO_438 [1 p. 4, 9].

# Citations

[1] Roto Safe E Eneo CC, notice simplifiée PROFERM, version 2, 2022 —
`raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf`, p. 1 à 10

# Voir aussi

- [Serrure motorisée](/quincaillerie/serrure-motorisee.md)
- [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md)
- [ROTO](/fournisseurs/roto.md)
- [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
