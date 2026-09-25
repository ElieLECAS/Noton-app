---
type: Fournisseur
title: SOMFY
description: Fournisseur des motorisations et de la domotique des volets roulants proposés par PROFERM.
tags: [fournisseur, somfy, motorisation, volet-roulant, domotique, tahoma]
fournisseur: SOMFY
status: draft
sources:
  - resource: raw/catalogue-general-2026-01.pdf
    id: catalogue-general-2026
    title: Catalogue menuiseries PROFERM, édition janvier 2026
    last_modified: 2026-01-31
generated:
  by: process:claude-code
  at: 2026-09-17T19:00:00Z
---

# Ce que PROFERM lui achète

SOMFY fournit les **motorisations et la domotique** des volets roulants PROFERM. C'est le seul
motoriste du corpus. Le volet roulant lui-même, coffre et tablier, n'a pas de fabricant nommé.

| Produit SOMFY | Rôle |
| --- | --- |
| Moteur IO RS100 | motorisation haut de gamme |
| Moteur filaire | motorisation économique |
| Radio IO solaire | alimentation sans câblage |
| RADIO IO HOMECONTROL | protocole des points de commande |
| Boîtier TaHoma | pilotage domotique |
| Commandes RTS | télécommandes compatibles |

(schéma: raw/catalogue-general-2026-01.pdf, p. 28)

# Caractéristiques du moteur IO RS100

| Caractéristique | Détail |
| --- | --- |
| Fins de course | démarrage et arrêt en douceur |
| Vitesse | moteur bi-vitesse |
| Mode discret | fonctionnement silencieux |
| Détection d'obstacles | arrêt immédiat |
| Détection du gel | évite la détérioration du matériel |

(schéma: raw/catalogue-general-2026-01.pdf, p. 28)

# Restrictions du moteur filaire

**Le système filaire ne commande qu'un seul volet** [1 p. 28]. Un chiffrage multi-ouvertures en
filaire demande donc autant de commandes que de volets.

# Radio IO solaire

Le capteur solaire supprime le câblage électrique. Il se pose **sur le coffre en rénovation** et
**au-dessus de la menuiserie en neuf**, et fonctionne avec toute la gamme des commandes RTS et
TaHoma [1 p. 28].

# Compatibilité

**Tous les volets roulants proposés par PROFERM sont compatibles RADIO IO HOMECONTROL** [1 p. 28].
Le pilotage se fait par télécommande, par boîtier TaHoma ou par smartphone, à distance comme sur
place.

# Garantie propre

La motorisation est garantie **7 ans** [1 p. 35], deux ans de plus que le volet roulant lui-même.
Voir [Garanties par composant](/garanties/garanties-par-composant.md).

# Ce que le corpus ne donne pas

Aucune coordonnée, aucune référence commerciale, aucune puissance ni couple moteur, aucun abaque
de dimensionnement.

La serrure motorisée de porte d'entrée n'est **pas** un produit SOMFY : elle repose sur un moteur
Roto Safe E Eneo CC et l'application SOREX. Voir
[Serrure motorisée](/quincaillerie/serrure-motorisee.md).

# Citations

[1] Catalogue menuiseries PROFERM, édition janvier 2026 — `raw/catalogue-general-2026-01.pdf`,
p. 28 et 35

# Voir aussi

- [Volets roulants](/equipements/volets-roulants.md)
- [Moustiquaires enroulables verticales](/equipements/moustiquaires-enroulables-verticales.md)
- [SOPROFEN](/fournisseurs/soprofen.md)
- [Serrure motorisée](/quincaillerie/serrure-motorisee.md)
- [Garanties par composant](/garanties/garanties-par-composant.md)
