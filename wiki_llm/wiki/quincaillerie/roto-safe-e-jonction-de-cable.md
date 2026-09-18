---
type: Quincaillerie
title: Jonction de câble Roto Safe E
description: Les variantes de jonction de câble qui alimentent une serrure motorisée Roto Safe E entre dormant et ouvrant, avec ou sans bloc d'alimentation intégré.
tags: [roto, safe-e, eneo, jonction-cable, alimentation, porte-entree, ip67, cablage]
status: stable
sources:
  - resource: raw/roto-safe-e-jonction-de-cable-2024-11.pdf
    id: roto-safe-e-jonction-de-cable
    title: Roto Safe E, jonction de câble, réf. SUG_28_FR_v3, novembre 2024
    last_modified: 2024-11-30
generated:
  by: process:claude-code
  at: 2026-09-18T12:45:00Z
stale_after: 2026-12-31
---

# Le problème que résout la jonction de câble

Une serrure motorisée est dans l'**ouvrant**, le courant arrive par le **dormant**. La **jonction
de câble Roto Safe E** est le passage électrique entre les deux, dimensionné pour supporter
l'ouverture et la fermeture répétées de la porte.

Elle s'emploie sur les **profils bois, PVC et aluminium**, et couvre deux familles de serrures
motorisées :

| Famille | Modèles |
| --- | --- |
| E700 | Eneo A, Eneo AF |
| E610 et E611 | Eneo CC, Eneo CF |

Chacune se câble de **deux façons** : avec un bloc d'alimentation secteur externe, ou avec le
**bloc d'alimentation intégré** à la jonction elle-même. Voir
[Serrure motorisée](/quincaillerie/serrure-motorisee.md) et
[Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md).

# Les variantes, référence par référence

Variantes de jonction de câble relevées sur la notice Roto Safe E (p. 10 et 13 à 16). Toutes
portent une **connexion enfichable démontable à 6 broches**.

| Référence | Bloc d'alimentation | Particularité |
| --- | --- | --- |
| 820187 | sans | câble moulé, support disponible |
| 820194 | sans | côté douille avec spirale montée sur coffret de réception, montage en U |
| 820255 | sans | tôle de protection incluse, 16 mm |
| 2045681 | **intégré** | têtière **ronde** |
| 2045682 | **intégré** | têtière **carrée** |

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 10 et 13 à 16)

**La jonction entre dormant et ouvrant est IP67 sur toutes les variantes.** L'indice ne vaut que
pour la connexion enfichable elle-même, connectée — pas pour l'ensemble du montage.

Les deux variantes à bloc d'alimentation intégré apportent quatre choses que les autres n'ont
pas :

- un **câble de 3 m directement raccordable au réseau 230 V**, en Plug & Play
- un **clameau pour les systèmes de contrôle d'accès externes**
- un **contrôle de la tension par une DEL visible**
- **aucune fente d'aération à prévoir dans le bois**

Le **820255** est aussi la pièce de passage de câble côté ouvrant du
[contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md), dont la pièce de
dormant est la **817028**.

# Cotes et caractéristiques du bloc d'alimentation intégré

Caractéristiques des jonctions **2045681** (têtière ronde) et **2045682** (têtière carrée),
relevées sur la notice (p. 11).

| Caractéristique | Valeur |
| --- | --- |
| Câble d'alimentation | H03VV-F, 2 × 1,5 mm² |
| Entrée | 230 V AC, 50 à 60 Hz |
| Sortie | 24 V DC, 2,1 A, 50,4 W |
| Degré de protection selon DIN 40050 | IP67, connecté, sur la seule connexion enfichable entre ouvrant et dormant |
| Plage de température, à l'arrêt (°C) | −10 à +70 |
| Plage de température, en mouvement (°C) | −10 à +50 |
| Normes | conformité CE |

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 11)

**Les deux plages de température ne sont pas les mêmes**, et c'est la seconde qui commande : une
porte qui s'ouvre au-delà de +50 °C sort du domaine d'emploi, même si le matériel supporte +70 °C
à l'arrêt. Sur une porte exposée plein sud avec un panneau sombre, l'écart de 20 °C n'est pas
théorique.

# Cotes du câble de liaison

| Élément | Câble |
| --- | --- |
| Douille | LIF9Y11Y, 6 × 0,25 mm² |
| Connecteur | LIF9Y11Y, 6 × 0,25 mm² |

Le même câble des deux côtés, **six conducteurs de 0,25 mm²** — ce qui correspond aux six broches
de la connexion enfichable et aux six fils du module de contrôle d'accès.

# Ce qui n'a pas été transcrit

Les **dimensions de perçage et de fraisage** (§ 4.3, p. 19 et 20) et les **plans de câblage**
détaillés par modèle (§ 5, p. 25 à 32) sont des planches cotées. Elles n'ont pas été reprises :
il faut les lire sur le document au moment de l'usinage.

La notice est un **document complémentaire**, qui ne remplace ni les notices de montage des
ferrures, ni les documents techniques des fabricants de profilés. Pour les Eneo CC et CF, la
notice de montage de référence est l'**IMO_438**, celle-là même que cite la
[notice simplifiée PROFERM](/quincaillerie/controle-acces-eneo-cc.md).

# Provenance

Notice d'instructions de montage et d'utilisation éditée par **Roto Frank Fenster- und
Türtechnologie GmbH**, Leinfelden-Echterdingen, réf. **SUG_28_FR_v3**, **novembre 2024**, extraite
de `raw/export_doc_125.zip`. C'est un document constructeur, non co-marqué PROFERM.

# Citations

[1] Roto Safe E, jonction de câble, réf. SUG_28_FR_v3, novembre 2024 —
`raw/roto-safe-e-jonction-de-cable-2024-11.pdf`, p. 3, 10, 11, 12 et 13 à 16

# Voir aussi

- [Serrure motorisée](/quincaillerie/serrure-motorisee.md)
- [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md)
- [ROTO](/fournisseurs/roto.md)
- [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md)
