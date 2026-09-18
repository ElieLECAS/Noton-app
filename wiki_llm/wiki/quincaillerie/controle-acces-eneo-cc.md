---
type: Quincaillerie
title: Contrôle d'accès 4 en 1 Roto Safe E Eneo CC
description: Le contrôle d'accès 4 en 1 des portes PROFERM — code PIN, empreinte, Bluetooth et RFID — avec ses caractéristiques électriques, son câblage et sa procédure de réinitialisation.
tags: [roto, safe-e, eneo-cc, controle-acces, porte-entree, serrure-motorisee, rfid, biometrie, cablage]
status: stable
sources:
  - resource: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf
    id: proferm-eneo-cc-notice-2022
    title: Roto Safe E Eneo CC, notice simplifiée PROFERM, version 2, 2022
    last_modified: 2022-12-31
generated:
  by: process:claude-code
  at: 2026-09-18T12:30:00Z
stale_after: 2026-12-31
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
| Supports RFID eKeys | illimités |
| Cryptage | AES 128 bits |
| Conformité | CE |

(schéma: raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf, p. 7)

Le tableau de la notice porte une troisième valeur de capacité, **200**, sans que la ligne
indique de quoi il s'agit. Elle n'est pas reprise ici. Entrée **VER-33** du registre
[Informations à vérifier](/anomalies/informations-a-verifier.md).

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
**bornes 5 et 6 sont reliées entre elles en interne** par un relais et une résistance.

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

**Les cotes d'usinage ne sont pas reprises dans ce wiki.** La notice les porte sur des dessins
cotés dont le texte extrait est illisible : la largeur du fraisage dépend de la largeur de la
têtière, l'axe de fraisage dépend du profil, et le fraisage dépend des hauteurs de gâche. Ces trois
dépendances sont énoncées par la notice, les valeurs sont à lire sur les planches.

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

# Ce que ce document n'est pas

> « Cet extrait ne remplace pas une documentation complète. Le non-respect de cette documentation
> dégage le fabricant du matériel de sa responsabilité » (notice simplifiée, avertissement).

La notice PROFERM est un **extrait** des instructions d'installation Roto complètes, référencées
**IMO_438** pour les Eneo C, CC et CF. **La garantie ne couvre que les composants d'origine
Roto.** Voir [ROTO](/fournisseurs/roto.md).

Le document prévoit un entretien **au moins une fois par an**, avec une répartition explicite des
tâches entre **entreprise spécialisée** et **utilisateur final** — resserrage des vis de fixation,
remplacement des vis endommagées. Les réglages de ferrure sont réservés à l'entreprise
spécialisée.

# Provenance

Notice simplifiée **PROFERM**, « Eneo CC, montage et programmation des contrôles d'accès »,
**version 2, 2022**, extraite de `raw/export_doc_189.zip`, source déclarée ROTO. C'est un document
co-marqué : mise en page PROFERM, contenu Roto.

# Citations

[1] Roto Safe E Eneo CC, notice simplifiée PROFERM, version 2, 2022 —
`raw/proferm-roto-eneo-cc-notice-simplifiee-2022.pdf`, p. 1 à 10

# Voir aussi

- [Serrure motorisée](/quincaillerie/serrure-motorisee.md)
- [Jonction de câble Roto Safe E](/quincaillerie/roto-safe-e-jonction-de-cable.md)
- [ROTO](/fournisseurs/roto.md)
- [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
