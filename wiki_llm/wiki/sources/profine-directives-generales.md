---
type: Document source
title: Directives générales profine, version janvier 2023
description: Manuel de mise en œuvre de 113 pages de profine, qui fixe la terminologie, les tailles d'ouvrants et la méthode de calcul des dimensions d'élément.
tags: [profine, directives, terminologie, mise-en-oeuvre, dimensions, manuel]
status: draft
sources:
  - resource: raw/profine-directives-generales-2023-01.pdf
    id: profine-directives-generales-2023
    title: Directives générales profine, version janvier 2023
generated:
  by: process:claude-code
  at: 2026-09-18T00:00:00Z
stale_after: 2024-12-31
---

# Ce que contient le document

**113 pages**, version **janvier 2023**. C'est le manuel de mise en œuvre de
[profine](/fournisseurs/profine.md), organisé en **registres** numérotés — l'équivalent
industriel d'un référentiel de fabrication.

**C'est le tome 1 d'un classeur en deux volumes.** Le tome 2,
[Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md), reprend la
même table des matières et porte les registres `2.x` propres au système 76 : cotes de débit,
abaques dimensionnels, renforts, usinages. Les deux se citent l'un l'autre — le tome 2 renvoie au
registre 1.2.3 pour les directives de renforcement et au registre 1.3.3 pour les exigences
statiques, qui sont ici.

Le document précise sa propre portée : il « comprend toutes les instructions permettant une
utilisation conforme » des systèmes profine, et les produits sont « soumis à des essais de type
initiaux » auxquels ce manuel est rattaché. **Ce n'est donc pas de la documentation commerciale
mais une pièce du système qualité** : s'en écarter peut invalider les essais de type.

**Chaque page porte la mention « Sous réserve de modifications techniques ! »**, et la version est
datée. Un exemplaire périmé n'a aucune valeur.

# Ce qui est exploité pour l'instant

Seul le registre **1.1.2 Terminologie et légendes** a été dépouillé. Il donne trois choses.

## Cotes des tailles d'ouvrants minimales

| Type de butée | Largeur mini (mm) | Hauteur mini (mm) |
| --- | --- | --- |
| Oscillo-battante | 340 | 660 |
| Soufflet, impostes | 660 | 340 |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.1.2, p. 1)

Les deux valeurs sont **inversées entre les deux types** : un oscillo-battant doit être plus haut
que large, un soufflet plus large que haut. Le [DTA](/certifications/dta-6-16-2334.md) donne les
maxima, ce manuel donne les minima — les deux sont nécessaires pour borner une commande.

## La règle de calcul d'une dimension d'élément

**Les cotes maximales des registres sont des cotes extérieures d'ouvrant, pas des dimensions de
fenêtre.** Pour obtenir la cote finale, il faut ajouter les profilés adjacents sur tous les côtés
(directives générales, registre 1.1.2, p. 1).

Exemple donné par le manuel :

| Poste | Valeur (mm) |
| --- | --- |
| Cote extérieure d'ouvrant | 2 100 |
| Vue intérieure de dormant, 2 × 42 | 84 |
| Coffre de volet roulant | 205 |
| Profilés d'élargissement | 120 |
| **Hauteur totale d'élément** | **2 509** |

(schéma: raw/profine-directives-generales-2023-01.pdf, registre 1.1.2, p. 1)

**Près de 20 % d'écart entre la cote d'ouvrant et la cote d'élément** dans cet exemple. C'est
l'erreur de métré la plus coûteuse possible, et le manuel ouvre dessus.

## La terminologie de référence

Le registre fixe le vocabulaire de toutes les cotes d'une menuiserie, avec un repère par notion :

| Repère | Notion | Repère | Notion |
| --- | --- | --- | --- |
| A | Dormant | J | Hauteur joint comprimé |
| B | Ouvrant | K | Recouvrement ouvrant |
| C | Parclose | L | Hauteur de calage vitrage |
| D | Rainure crémone | M | Hauteur pré-cale |
| E | Dos de dormant | N | Jeu de fonctionnement |
| F | Chambre des renforts | O | Dimension extérieure dormant |
| G | Rainure parclose | P | Clair de jour vitrage |
| H | Feuillure vitrage | Q | Clair de jour ouvrant |
| I | Feuillure quincaillerie | R | Clair de jour dormant |

Il définit également les sigles employés dans tout le système : **DHT** dimension hors tout,
**CCD** cote clair de dormant, **CCO** cote clair d'ouvrant, **CCV** cote clair de vitrage.

C'est la seule définition formelle de ce vocabulaire dans tout le corpus, et elle vaut pour lire
n'importe quelle planche profine.

# Ce qui reste à dépouiller

**Les 107 autres pages n'ont pas été exploitées.** Le document est organisé en registres couvrant
la composition des menuiseries — fenêtre, porte-fenêtre, coulissant, chicane — et vraisemblablement
les abaques de renfort, les tolérances et les règles d'usinage que le wiki réclame depuis
l'ingestion du catalogue général.

C'est la source la plus prometteuse encore inexploitée du corpus. À traiter registre par
registre plutôt qu'en bloc, en commençant par ceux qui répondent à une question ouverte du
registre [Informations à vérifier](/anomalies/informations-a-verifier.md).

# Provenance du fichier

PDF extrait de `raw/export_doc_146.zip`. Titre interne « Directives_Générales_08_2023 »,
identifiant 146.

# Citations

[1] Directives générales profine, version janvier 2023 —
`raw/profine-directives-generales-2023-01.pdf`, registre 1.1.2

# Voir aussi

- [profine](/fournisseurs/profine.md)
- [DTA n° 6/16-2334_V5](/certifications/dta-6-16-2334.md)
- [Posters Système 76 Advanced](/sources/posters-systeme-76-advanced.md)
- [Mise en œuvre Système 76 Advanced](/sources/profine-mise-en-oeuvre-76-advanced.md)
- [Abaques dimensionnels du système 76](/profiles/systeme-76-abaques-dimensionnels.md)
- [Informations à vérifier](/anomalies/informations-a-verifier.md)
