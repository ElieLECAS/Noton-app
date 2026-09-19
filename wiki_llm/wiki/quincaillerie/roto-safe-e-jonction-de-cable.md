---
type: Quincaillerie
title: Jonction de câble Roto Safe E
description: Les variantes de jonction de câble qui alimentent une serrure motorisée Roto Safe E entre dormant et ouvrant, avec ou sans bloc d'alimentation intégré.
tags: [roto, safe-e, eneo, jonction-cable, alimentation, porte-entree, ip67, cablage]
famille: roto-safe-e
status: stable
sources:
  - resource: raw/roto-safe-e-jonction-de-cable-2024-11.pdf
    id: roto-safe-e-jonction-de-cable
    title: Roto Safe E, jonction de câble, réf. SUG_28_FR_v3, novembre 2024
    last_modified: 2024-11-30
source_pages:
  - resource: raw/roto-safe-e-jonction-de-cable-2024-11.pdf
    pages: 3, 5, 10-40
generated:
  by: process:claude-code
  at: 2026-09-19T22:00:00Z
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

Le site de montage est le dormant ou la feuillure de porte. Le trou de perçage pour le passage de
la ligne fait **Ø 16 mm**, ébarbé et débarrassé de ses copeaux avant le passage du câble, et toute
torsion des fils doit être évitée sur la longueur de la ligne, en particulier dans le ressort
métallique [1 p. 17]. Au-delà de l'état général de la technique, le montage suit les
réglementations **VDS 2311** et **VDE 0100** [1 p. 17].

# Les variantes, référence par référence

Variantes de jonction de câble relevées sur la notice Roto Safe E (p. 10 et 13 à 16). Toutes
portent une **connexion enfichable démontable à 6 broches**.

| Référence | Côté | Bloc d'alimentation | Particularité | Accessoire requis |
| --- | --- | --- | --- | --- |
| 820187 | dormant | sans | câble moulé, support disponible | - |
| 820194 | ouvrant | sans | côté douille avec spirale montée sur coffret de réception, montage en U, angle de rotation 180° | câble type E (633291) ou type EZ (633292) |
| 820255 | ouvrant | sans | montage longitudinal, spirale, tôle de protection incluse (16 mm), câble pré-assemblé pour jonction, verrouillage Eneo et contrôle d'accès (0,3 m et 3 m) | - |
| 2045681 | dormant | **intégré** | têtière **ronde** | - |
| 2045682 | dormant | **intégré** | têtière **carrée** | - |

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 10 et 13 à 16)

Les pièces se combinent en **quatre variantes de montage**, selon le jeu disponible en feuillure
et selon la présence du bloc d'alimentation intégré [1 p. 13-16] :

| Variante | Jeu en feuillure (mm) | Pièce dormant | Pièce ouvrant | Pièces complémentaires |
| --- | --- | --- | --- | --- |
| 1 | 12 / 16 | 820187 | 820255 | tôle de protection |
| 2 | 4 / 12 | 820187 | 820194 | tôle de protection, équerre de fixation |
| 3 | 12 / 16 | 2045681 ou 2045682 | 820255 | - |
| 4 | 4 / 12 | 2045681 ou 2045682 | 820194 | équerre de fixation |

**Le jeu en feuillure décide de la pièce d'ouvrant, pas le bloc d'alimentation** : 820255 sert au
jeu large (12/16 mm), 820194 au jeu réduit (4/12 mm), quelle que soit la pièce dormant retenue.

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

# Cotes des pièces

Dimensions des pièces elles-mêmes, en mm, relevées sur les planches de la notice (p. 17-18).

| Pièce | Largeur (mm) | Longueur utile (mm) | Longueur totale (mm) | Perçages (Ø mm) |
| --- | --- | --- | --- | --- |
| Dormant sans bloc, 820187 | 10,4 | 43-57,5 | - | 4,9 et 3,2 |
| Dormant avec bloc, 2045681/2045682 | 24 | 34-302 | 421 (entraxe) / 445 | - |
| Ouvrant sans coffret, montage longitudinal, 820255 | 10-16 | 62-126 (110 entraxe) | - | 4,9 et 3,2 |

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 17 et 18)

**La jonction avec bloc d'alimentation a les mêmes dimensions en têtière ronde (2045681) et en
têtière carrée (2045682)** [1 p. 18].

# Cotes de perçage et de fraisage

Perçages et fraisages du dormant et de l'ouvrant, en mm, relevés sur les planches de la notice
(§ 4.3, p. 19 à 21) selon les quatre combinaisons de pièces.

| Combinaison | Pièce dormant : perçages (Ø mm, entraxes) | Pièce ouvrant : fraisage, profondeur (mm) | Pièce ouvrant : dimensions du logement (mm) |
| --- | --- | --- | --- |
| Dormant sans bloc + ouvrant sans coffret | Ø 7, entraxes 12,5 et 28 | 30 | logement 35 × 10, trou Ø 5,5 à 5,5 mm du bord |
| Dormant sans bloc + ouvrant avec coffret | Ø 7, entraxes 12,5 et 28 | 60 (fraisage) et 1 (rainure) | hauteur totale 210, utile 194, largeur 20 |
| Dormant avec bloc + ouvrant sans coffret | hauteur totale 445, utile 421-400, largeur 24 | 30 | logement 35 × 10, trou Ø 5,5 à 5,5 mm du bord |
| Dormant avec bloc + ouvrant avec coffret | hauteur totale 445, utile 421-400, largeur 24 | 60 (fraisage) et 1 (rainure) | hauteur totale 210, utile 194, largeur 20 |

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 19 à 21)

**Les trous de perçage ne doivent contenir aucune bavure ni copeaux.** Les vis se serrent
uniquement à la main ; un pré-perçage des trous de vis est recommandé [1 p. 19-21].

# Montage

Étapes de montage de la jonction de câble, dans l'ordre (§ 4.4, p. 21-24).

| Étape | Opération |
| --- | --- |
| 1 | Effectuer les fraisages et les perçages (voir *Cotes de perçage et de fraisage*) |
| 2 | Passer le câble à travers l'ouvrant ou le dormant, en posant une boucle pour assurer une réserve de câble suffisante — sans écraser ni endommager le câble lors du montage du dormant |
| 3 | Côté ouvrant : fixer le boîtier de réception à l'aide des vis |
| 4 | Côté dormant : mettre en place les pièces |
| 5 | Serrer le côté connecteur à l'aide d'une vis |
| 6 | Accrocher l'ouvrant et établir la connexion enfichable, en gardant le connecteur et la douille propres — des copeaux dans le connecteur ou la douille provoquent un court-circuit |
| 7 | Fixer la connexion enfichable à l'aide de sa vis, en préservant sa protection contre la torsion |
| 8 | Sur la variante à bloc d'alimentation secteur intégré, vérifier le fonctionnement (voir *Dépannage*) |

**Le raccordement du bloc d'alimentation intégré (2045681, 2045682) au réseau 230 V est réservé à
un électricien spécialisé** : le courant électrique peut entraîner des blessures mortelles, et les
prescriptions nationales en vigueur doivent être respectées (en Allemagne, notamment VDE 0100)
[1 p. 24].

# Plans de câblage

Deux familles de serrure motorisée se câblent différemment sur la connexion enfichable à 6
broches, relevé sur les plans de câblage (§ 5, p. 25-33). Les exemples d'application sont fournis
sans engagement : le respect des normes et dispositions applicables reste sous la responsabilité
de l'installateur [1 p. 25].

| Borne | Fil | Fonction | E700 (Eneo A, Eneo AF) | E610 / E611 (Eneo CC, Eneo CF) |
| --- | --- | --- | --- | --- |
| 1 | Blanc | IN1, entrée 1 (OUVERT) | utilisé | utilisé |
| 2 | Marron | +24 V | utilisé | utilisé |
| 3 | Vert | GND | utilisé | utilisé |
| 4 | Jaune | IN2, entrée 2 (interrupteur jour/nuit) | inoccupé | utilisé, en option |
| 5 | Gris | K1a, contact libre de potentiel | inoccupé | utilisé, en option |
| 6 | Rose | K1b, contact libre de potentiel | inoccupé | utilisé, en option |

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 26 à 33)

**Sur la famille E610/E611, le fil jaune doit rester dégagé** : raccordé, il ponte l'interrupteur
côté ouvrant. **Les bornes 5 et 6 (fils gris et rose) sont reliées entre elles en interne** par un
relais et une résistance de 47 ohms, pour une charge maximale de 24 V / 40 mA [1 p. 31 et 33].

**Deux autres câbles du même montage reprennent les mêmes noms de couleur pour des signaux
différents — ne pas les confondre avec la connexion enfichable ci-dessus** :

| Câble | Fil | Fonction |
| --- | --- | --- |
| Lecteur d'empreinte (connecteur JST → boîte noire) | Marron | +24 V |
| Lecteur d'empreinte (connecteur JST → boîte noire) | Jaune | GND |
| Lecteur d'empreinte (connecteur JST → boîte noire) | Vert | Commande (OUVERT) |
| Système de contrôle d'accès 4 en 1 (connecteur JST → boîte noire) | Rouge | +24 V |
| Système de contrôle d'accès 4 en 1 (connecteur JST → boîte noire) | Noir | GND |
| Système de contrôle d'accès 4 en 1 (connecteur JST → boîte noire) | Jaune | Commande (OUVERT) |

Sur la variante à bloc d'alimentation intégré, le raccordement secteur se fait par un câble de
3 m côté dormant (230 V AC, fils noir et rouge en +24 V/GND côté bloc), et **le côté 24 V ne doit
pas être câblé** au réseau [1 p. 28 et 32]. D'autres plans de câblage existent sous la référence
**IMO_310**, hors de ce corpus [1 p. 25].

# Dépannage

Diagnostic relevé sur la notice (§ 6.1, p. 34).

| Erreur | Cause | Dépannage | Qui |
| --- | --- | --- | --- |
| Absence de courant électrique | Connexion enfichable desserrée | Fixer le connecteur | entreprise spécialisée ou utilisateur final |
| Absence de courant électrique | Rupture du câble | Remplacer le câble | entreprise spécialisée |
| Absence de courant électrique | Connexion électrique manquante | Vérifier les connexions enfichables, l'alimentation électrique (la LED doit être allumée) et le bloc d'alimentation | entreprise spécialisée |

Contrôle de fonctionnement de la variante à bloc d'alimentation secteur intégré : brancher le
câble d'alimentation sur le 230 V — **uniquement par un électricien spécialisé** — puis vérifier
que la LED s'allume en vert, signe de tension présente [1 p. 34].

# Démontage et mise au rebut

Le démontage se fait, sauf indication contraire, **dans l'ordre inverse du montage** [1 p. 35] :

| Étape | Opération |
| --- | --- |
| 1 | Couper l'alimentation électrique, débrancher le connecteur |
| 2 | Desserrer la vis et la retirer |
| 3 | Desserrer la connexion enfichable à l'aide d'une clé six-pans ou d'un tournevis adapté — jamais en tirant sur le ressort, qui serait endommagé |
| 4 | Décrocher l'ouvrant |
| 5 | Protéger la connexion enfichable contre la poussière et l'humidité |

**L'ouvrant peut tomber lors d'un démontage non conforme** : le protéger contre les chutes, à
deux personnes, et ne faire réaliser le démontage que par une entreprise spécialisée. Le port de
charges lourdes est plafonné à **25 kg pour un homme et 10 kg pour une femme** [1 p. 35].

Les ferrures sont mises au rebut comme **matières premières**, dans un centre de valorisation
écologique. Les déchets électroniques suivent les directives européennes RoHS (2002/95/CE) et
WEEE (2002/96/CE) et, en Allemagne, la loi ElektroG [1 p. 36].

# Conformité électrique

Les deux variantes à bloc d'alimentation intégré (2045681 têtière ronde, 2045682 têtière carrée)
sont déclarées conformes aux directives européennes **2014/35/UE** (basse tension), **2011/65/UE**
et **2015/863** (RoHS), selon les normes **EN 60335-1:2012**, **EN 60335-2-103:2006** et
**EN IEC 62368-1:2014** [1 p. 37-40]. Déclaration signée à Kalsdorf b. Graz (Autriche), le
6 novembre 2024, par le gérant de Roto Frank Austria GmbH.

# Ce que la source ne donne pas

**Le côté verrouillage** — les serrures motorisées E610/E611 Eneo CC/CF et E700 Eneo A elles-mêmes,
au-delà de leur jonction de câble — est couvert par des références absentes de `raw/` : les
notices de montage **IMO_438** et **IMO_506**, le catalogue **CTL_86**, et les plans de câblage
**IMO_310** [1 p. 5]. Voir [Serrure motorisée](/quincaillerie/serrure-motorisee.md) pour ce que le
wiki documente par ailleurs de ces serrures.

Les consignes de sécurité génériques (symboles, groupes cibles, limitation de responsabilité,
utilisation conforme, p. 5 à 9) ne portent aucune donnée propre à une référence ou à PROFERM et ne
sont pas reprises au-delà de cette mention.

# Provenance

Notice d'instructions de montage et d'utilisation éditée par **Roto Frank Fenster- und
Türtechnologie GmbH**, Leinfelden-Echterdingen, réf. **SUG_28_FR_v3**, **novembre 2024**, extraite
de `raw/export_doc_125.zip`. C'est un document constructeur, non co-marqué PROFERM.

# Citations

[1] Roto Safe E, jonction de câble, réf. SUG_28_FR_v3, novembre 2024 —
`raw/roto-safe-e-jonction-de-cable-2024-11.pdf`, p. 3, 5 et 10 à 40

# Voir aussi

- [Serrure motorisée](/quincaillerie/serrure-motorisee.md)
- [Contrôle d'accès 4 en 1 Eneo CC](/quincaillerie/controle-acces-eneo-cc.md)
- [ROTO](/fournisseurs/roto.md)
- [Sécurité des portes d'entrée](/quincaillerie/securite-portes-entree.md)
