---
type: Quincaillerie
title: 'Jonction de câble Roto Safe E : plans de câblage des serrures Eneo E700, E610 et E611'
description: Plans de câblage de la jonction de câble Roto Safe E avec ou sans bloc d'alimentation intégré pour les serrures Eneo A / AF (E700) et Eneo CC / CF (E610, E611), câbles du lecteur d'empreinte et du contrôle d'accès 4 en 1.
tags: [roto, safe-e, eneo, jonction-cable, alimentation, porte-entree, ip67, cablage, connecteur-jst]
systeme: Roto Safe E
fournisseur: ROTO
usage: [pose, sav]
famille: jonction-de-cable
status: draft
sources:
  - resource: raw/roto-safe-e-jonction-de-cable-2024-11.pdf
    id: roto-safe-e-jonction-de-cable
    title: Roto Safe E, jonction de câble, réf. SUG_28_FR_v3, novembre 2024
    last_modified: 2024-11-30
source_pages:
  - resource: raw/roto-safe-e-jonction-de-cable-2024-11.pdf
    pages: 1-2, 5-19, 25-34, 36-41
generated:
  by: process:claude-code
  at: 2026-10-21T10:43:51Z
---

Cette page reprend, de la notice Roto Safe E « jonction de câble » (SUG_28, novembre 2024), les plans de câblage et la portée des plans. Contexte commun : [Jonction de câble Roto Safe E : propriétés, quatre variantes de montage et cotes](/quincaillerie/roto-safe-e-jonction-de-cable.md).

# Plans de câblage

## Ce que valent les plans

Les illustrations suivantes sont des exemples d'application sans engagement. Elles sont
uniquement destinées à aider les clients, mais ne constituent pas de solutions spécifiques aux
clients. Le client est responsable du bon fonctionnement du produit. Le client est tenu de
garantir une installation, une utilisation, un entretien et un fonctionnement fiables ; en outre,
le client est tenu de ne faire installer les appareils que par un électricien qualifié, faute de
quoi il peut y avoir un risque d'incendie ou d'électrocution. Le respect des normes et
dispositions applicables doit être garanti à tout moment, l'installateur de l'installation en
assume l'entière responsabilité ; tant que ce respect n'est pas garanti, l'installation ne doit
pas être utilisée. Les exemples d'application sont fournis sans aucune garantie : Roto Frank
Fenster- und Türtechnologie GmbH décline toute responsabilité pour les dommages causés par leur
utilisation, à moins que les dispositions légales ne prévoient une responsabilité obligatoire en
cas de faute intentionnelle. D'autres plans de câblage existent sous la référence **IMO_310**
[1 p. 25].

**Les travaux d'installation et d'entretien ne peuvent être effectués que par des entreprises
spécialisées en électricité.** Toute intervention sur une tension du secteur de 230 V (ou 115 V)
présente un danger de mort ; tous les travaux ne doivent être effectués que lorsque l'appareil est
hors tension [1 p. 25].

## Branchement de composants externes sur la jonction avec bloc d'alimentation intégré

Dans le cas d'une jonction de câble avec bloc d'alimentation intégré, les composants externes
sont directement branchés à la jonction de câble côté dormant ; pour ce faire, démonter cette
dernière. Pour le branchement de composants externes, raccorder un câble de 6 × 0,25 mm² à la
jonction de câble, puis le faire passer à l'extérieur [1 p. 25].

Affectation des bornes de la jonction de câble avec bloc d'alimentation intégré ; IN1 et IN2
sont les entrées de commande, GND la masse (0 V), K1a et K1b les deux bornes d'un
[contact libre de potentiel](/reference/glossaire.md) (contact sec, qui ne fournit aucune
tension) [1 p. 25] :

| Borne | Affectation |
| --- | --- |
| 1 | IN1 / entrée 1 (OUVERT) |
| 2 | +24 V |
| 3 | GND |
| 4 | IN2 / entrée 2 |
| 5 | K1a pot. - contact libre |
| 6 | K1b pot. - contact libre |

Les bornes 5 et 6 sont reliées entre elles en interne par un relais et une résistance de
47 ohms. La charge maximale des contacts est de 24 V / 40 mA [1 p. 25].

## Câbles du lecteur d'empreinte et du système de contrôle d'accès 4 en 1

Deux listes donnent les couleurs des câbles « entre connecteur JST et Black Box ». Le
**connecteur JST** est la petite prise du câble qui vient du lecteur extérieur ; la **Black Box**
(« Boîte noire » sur les plans) est l'unité intérieure du contrôle d'accès (voir le
[glossaire](/reference/glossaire.md)).

| Équipement | Couleur du fil | Affectation | Pages où la liste est imprimée |
| --- | --- | --- | --- |
| Lecteur d'empreinte | Marron | +24 V | 25, 29, 31, 33 |
| Lecteur d'empreinte | Jaune | GND | 25, 29, 31, 33 |
| Lecteur d'empreinte | Vert | Commande (OUVERT) | 25, 29, 31, 33 |
| Système de contrôle d'accès (ZKS) 4 en 1 | Rouge | +24 V | 26 |
| Système de contrôle d'accès (ZKS) 4 en 1 | Noir | GND | 26 |
| Système de contrôle d'accès (ZKS) 4 en 1 | Jaune | Commande (OUVERT) | 26 |

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 25 et 26)

**Ces couleurs ne sont pas celles de la connexion enfichable à 6 broches** : le jaune y est
l'entrée IN2 et le vert la masse (voir les affectations plus bas). Sur les quatre plans de
câblage qui suivent, le connecteur JST est la boîte « Blanc » reliée à la « Boîte noire », et ses
trois fils sont dessinés **Marron, Vert, Blanc**, piqués sur les fils marron (+24 V), vert (GND)
et blanc (IN1) de la serrure ; la notice simplifiée de l'Eneo CC dessine les mêmes couleurs sur ce
connecteur. Couleurs du 4 en 1 à confirmer : **CTR-66**, non arbitrée [1 p. 26, 28, 30, 32].

Les circuits de base suivants servent d'exemples [1 p. 26].

## E700 – Eneo A et E700 – Eneo AF, jonction de câble sans bloc d'alimentation secteur

Le plan relie, de haut en bas, la serrure motorisée, le câble de l'ouvrant, la connexion
enfichable, puis côté dormant un bouton et le bloc d'alimentation Eneo externe [1 p. 26].

![Plan de câblage E700, jonction de câble sans bloc d'alimentation secteur](/assets/quincaillerie/roto-safe-e-jonction-de-cable/cablage-e700-sans-bloc.png)

Comment lire le plan :

- **En haut**, la serrure (dessinée coupée) porte deux connecteurs : **« Noir »**, dont les trois
  fils sont inoccupés, et **« Vert »**, bornes GND (fil vert), +24 V (fil marron) et IN1 (fil
  blanc).
- **À gauche**, le boîtier extérieur « Lecteur d'empreinte / Système de contrôle d'accès 4 en 1 »
  (« Extérieur : Système de contrôle d'accès « Déverrouiller les portes » ») est relié à la
  **Boîte noire**, elle-même reliée à la **connexion enfichable JST** « Blanc », dont les fils
  Marron, Vert et Blanc se piquent sur les fils marron, vert et blanc de la serrure.
- **Au milieu**, le **câble – type EZ, 3 m** descend jusqu'à la **jonction de câble
  enfichable** : bornes 1 Blanc, 2 Marron, 3 Vert, 4 à 6 inoccupées, côté ouvrant au-dessus du
  trait, côté dormant en dessous.
- **En bas, côté dormant** : le blanc et le marron vont au **bouton** (« À l'intérieur : bouton
  « déverrouiller les portes » ») ; le marron va aussi à la borne **24V (+V)** et le vert à la
  borne **GND (−V)** du **bloc d'alimentation Eneo** (bornes secteur N et L ; IN : 100-240 V AC ;
  OUT : 24 V DC / 2.5 A).

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 26)

Affectation des connecteurs et des câbles de l'E700 sans bloc d'alimentation secteur [1 p. 27] :

| Couleur du fil | Borne de la connexion enfichable | Affectation |
| --- | --- | --- |
| blanc | 1 | IN1 / entrée 1 (OUVERT) |
| marron | 2 | +24 V |
| vert | 3 | GND |
| jaune | - | inoccupé |
| gris | - | inoccupé |
| rose | - | inoccupé |

Les numéros de borne sont ceux du plan de la p. 26, qui ne les porte que pour les trois fils
occupés.

## E700 – Eneo A et E700 – Eneo AF, jonction de câble avec bloc d'alimentation intégré

Même partie haute que le plan précédent ; côté dormant, la pièce à bloc d'alimentation intégré
remplace le bloc externe [1 p. 28].

![Plan de câblage E700, jonction de câble avec bloc d'alimentation intégré](/assets/quincaillerie/roto-safe-e-jonction-de-cable/cablage-e700-avec-bloc.png)

Comment lire le plan :

- **En haut**, identique au plan sans bloc : serrure (connecteurs « Noir » inoccupé et « Vert »
  GND / +24 V / IN1), connexion enfichable JST « Blanc » (Marron, Vert, Blanc), Boîte noire,
  boîtier extérieur « Système de contrôle d'accès / lecteur d'empreinte / système de contrôle
  d'accès 4 en 1 », câble – type EZ, 3 m.
- **Jonction de câble enfichable** : côté ouvrant, bornes 1 Blanc, 2 Marron, 3 Vert, 4 à 6
  inoccupées ; côté dormant, les mêmes bornes 1 à 6, plus deux bornes **+24V** et **GND**.
- **Sous les bornes 1 à 6**, le « **câble pour jonction de câble** » descend vers le **bouton**
  (« Intérieur : Bouton « déverrouiller les portes » en option »), raccordé au marron et au blanc.
- **Des bornes +24V et GND** partent un fil **Rouge** (+24 V) et un fil **Noir** (GND) vers le
  **bloc d'alimentation** (IN : 100-230 V AC ; OUT : 24 V DC), alimenté par un **câble 230 V AC de
  3 m**. Un encadré INFO, à côté de ces deux bornes, porte : « **Le côté 24 V ne doit pas être
  câblé.** » — ce qu'il interdit exactement n'est pas dit : **VER-112**.

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 28)

**Si le bloc d'alimentation est raccordé au 230 V avant que la jonction de câble ne soit
raccordée, cela peut provoquer un court-circuit.** Côté ouvrant, raccorder le câble type EZ à la
jonction de câble (pour le câblage, voir le plan de câblage) ; côté dormant, raccorder le câble de
la jonction de câble, par exemple au bouton, à l'interphone, etc. [1 p. 29].

Affectation des connecteurs et des câbles de l'E700 avec bloc d'alimentation intégré :
identique à la version sans bloc — blanc IN1 / entrée 1 (OUVERT), marron +24 V, vert GND, jaune,
gris et rose inoccupés [1 p. 29].

## E610 – Eneo CC et E611 – Eneo CF, jonction de câble sans bloc d'alimentation secteur

Sur les E610 et E611, les six fils sont occupés : le jaune porte l'entrée IN2 de
l'**interrupteur jour / nuit** et le gris et le rose le contact libre K1a / K1b [1 p. 30-31].

![Plan de câblage E610 / E611, jonction de câble sans bloc d'alimentation secteur](/assets/quincaillerie/roto-safe-e-jonction-de-cable/cablage-e610-e611-sans-bloc.png)

Comment lire le plan :

- **En haut**, la serrure porte le connecteur **« Noir »** — bornes K1b (fil rose), K1a (fil
  gris), reliées par le symbole d'un contact, et IN2 (fil jaune) — et le connecteur **« Vert »**
  — GND (fil vert), +24 V (fil marron), IN1 (fil blanc).
- **À gauche**, le boîtier extérieur « Système de contrôle d'accès / Lecteur d'empreinte /
  Système de contrôle d'accès 4 en 1 », la Boîte noire et la connexion enfichable JST « Blanc »
  (Marron, Vert, Blanc piqués sur les fils de même couleur de la serrure).
- **À droite**, l'**interrupteur jour / nuit** (« À l'intérieur : Peut être utilisé en
  option. ») : le contact dessiné dans le cercle relie le fil jaune (IN2) au fil marron (+24 V) ;
  le fil vert est tracé jusqu'au bord de l'interrupteur.
- **Au milieu**, le câble – type EZ, 3 m, et la jonction de câble enfichable : bornes 1 Blanc,
  2 Marron, 3 Vert, 4 Jaune, 5 Gris, 6 Rose, les mêmes six couleurs ressortant côté dormant.
- **En bas, côté dormant** : le blanc et le marron vont au **bouton** (« À l'intérieur : Bouton
  « déverrouiller les portes » en option ») ; le marron va à la borne 24V (+V) et le vert à la
  borne GND (−V) du **bloc d'alimentation Eneo** (N, L ; IN : 100-240V AC ; OUT : 24 V DC/ 2.5 A).
  Le jaune, le gris et le rose ne sont raccordés à rien côté dormant.

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 30)

Affectation des connecteurs et des câbles des E610 / E611 sans bloc d'alimentation secteur
[1 p. 31] :

| Couleur du fil | Borne de la connexion enfichable | Affectation |
| --- | --- | --- |
| blanc | 1 | IN1 / entrée 1 (OUVERT) |
| marron | 2 | +24 V |
| vert | 3 | GND |
| jaune | 4 | IN2 / entrée 2 (interrupteur jour / nuit) |
| gris | 5 | K1a pot. - contact libre |
| rose | 6 | K1b pot. - contact libre |

**Le fil jaune doit rester dégagé, faute de quoi l'interrupteur côté ouvrant est ponté.** Les
bornes 5 et 6 (fils gris et rose) sont reliées entre elles en interne par un relais et une
résistance de 47 ohms. La charge maximale des contacts est de 24 V / 40 mA [1 p. 31].

## E610 – Eneo CC et E611 – Eneo CF, jonction de câble avec bloc d'alimentation intégré

![Plan de câblage E610 / E611, jonction de câble avec bloc d'alimentation intégré](/assets/quincaillerie/roto-safe-e-jonction-de-cable/cablage-e610-e611-avec-bloc.png)

Comment lire le plan : la partie haute est celle du plan sans bloc (serrure « Noir » K1b / K1a /
IN2 et « Vert » GND / +24 V / IN1, JST « Blanc », Boîte noire, boîtier extérieur ;
interrupteur jour / nuit « Intérieur : Peut être utilisé en option. », dont les fils sont
étiquetés Jaune, Vert et Marron). La jonction de câble enfichable porte côté ouvrant les bornes
1 Blanc, 2 Marron, 3 Vert, 4 Jaune, 5 Gris, 6 Rose, et côté dormant les mêmes bornes 1 à 6 plus
+24V et GND. Le câble pour jonction de câble (Rose, Gris, Jaune, Vert, Marron, Blanc) va au
bouton (« Intérieur : Bouton « déverrouiller les portes » en option ») par le marron et le blanc.
Les bornes +24V et GND sont reliées par un fil **Rouge** et un fil **Noir** au **bloc
d'alimentation** (+24 V, GND ; IN : 100-230 V AC ; OUT : 24 V DC ; câble 230 V AC de 3 m), avec
le même encadré « **Le côté 24 V ne doit pas être câblé.** » (**VER-112**) [1 p. 32].

(schéma: raw/roto-safe-e-jonction-de-cable-2024-11.pdf, p. 32)

**Si le bloc d'alimentation est raccordé au 230 V avant que la jonction de câble ne soit
raccordée, cela peut provoquer un court-circuit.** Côté ouvrant, raccorder le câble type EZ à la
jonction de câble (pour le câblage, voir le plan de câblage) ; côté dormant, raccorder le câble de
la jonction de câble, par exemple au bouton, à l'interphone, etc. [1 p. 32].

L'affectation des connecteurs et des câbles des E610 / E611 avec bloc d'alimentation intégré est
identique à celle de la version sans bloc (tableau ci-dessus), avec la même consigne sur le fil
jaune, les bornes 5 et 6 et la charge maximale de 24 V / 40 mA [1 p. 33].

# Citations

[1] Roto Safe E, jonction de câble, notice d'instructions de montage et d'utilisation, réf.
SUG_28_FR_v3, novembre 2024 — `raw/roto-safe-e-jonction-de-cable-2024-11.pdf`, p. 1, 2, 5 à 19, 25 à 34
et 36 à 41

# Voir aussi

Cette page est une partie d'un sujet qui en compte 3 ; les autres :

- [Jonction de câble Roto Safe E : propriétés, quatre variantes de montage et cotes](/quincaillerie/roto-safe-e-jonction-de-cable.md)
- [Jonction de câble Roto Safe E : dépannage, mise au rebut, conformité électrique, notice](/quincaillerie/roto-safe-e-jonction-de-cable-depannage-conformite-consignes-notice.md)
